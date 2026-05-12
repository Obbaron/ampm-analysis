"""
DBSCAN clustering for AMPM data with anisotropic Z-scaling.

Key design choices:
- Cluster a downsampled representative set, then propagate labels to all rows
  via nearest-neighbor lookup in the same scaled coordinate space.  Scales to
  100M+ rows without choking on memory.
- Anisotropic distance metric via Z scaling: internally we replace Z with
  Z * (eps_xy / eps_z), then run isotropic DBSCAN with eps = eps_xy.  This is
  equivalent to using an axis-aligned ellipsoidal neighborhood with the
  intuitive parameters: eps_xy is your XY radius, eps_z is your Z radius.
- Stable labels: clusters are renumbered by centroid (X, then Y, then Z) so
  that running on the same data gives the same labels.  Noise stays at -1.
- Output is a polars DataFrame with the original columns plus a 'cluster'
  integer column, ready for plotting (color="cluster") or remapping to
  part IDs.

Typical usage
-------------
    from ampm.clustering import k_distance_curve, cluster_dbscan
    from ampm.plotting import scatter2d, scatter3d

    # 1. Tune eps_xy by inspecting the k-distance plot
    kdc = k_distance_curve(df, k=10, mode="3d", eps_xy=1.0, eps_z=0.1)
    fig = scatter2d(kdc, x="Rank", y="k-distance (mm)", equal_aspect=False,
                    title="k-distance (find the elbow)")
    fig.show()

    # 2. Cluster
    clustered = cluster_dbscan(
        df, eps_xy=2.0, eps_z=0.1, min_samples=10, mode="3d",
    )

    # 3. Plot
    fig = scatter3d(clustered, x="Demand X", y="Demand Y", z="Z",
                    color="cluster", colorscale="Turbo")
    fig.show()
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import polars as pl
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree, NearestNeighbors

Mode = Literal["2d", "3d"]
DEFAULT_COLUMNS: tuple[str, str, str] = ("Demand X", "Demand Y", "Z")


def _validate(
    df: pl.DataFrame,
    mode: Mode,
    columns: Sequence[str],
    eps_xy: float,
    eps_z: float | None,
) -> tuple[str, str, str | None]:
    if mode not in ("2d", "3d"):
        raise ValueError(f"mode must be '2d' or '3d', got {mode!r}")
    if mode == "3d":
        if len(columns) < 3:
            raise ValueError(f"mode='3d' needs 3 columns, got {tuple(columns)}")
        if eps_z is None:
            raise ValueError("mode='3d' requires eps_z")
        if eps_z <= 0:
            raise ValueError(f"eps_z must be positive, got {eps_z}")
        x_col, y_col, z_col = columns[0], columns[1], columns[2]
    else:
        if len(columns) < 2:
            raise ValueError(f"mode='2d' needs 2 columns, got {tuple(columns)}")
        x_col, y_col = columns[0], columns[1]
        z_col = None
    if eps_xy <= 0:
        raise ValueError(f"eps_xy must be positive, got {eps_xy}")

    needed = [c for c in (x_col, y_col, z_col) if c is not None]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Column(s) not in DataFrame: {missing}")

    return x_col, y_col, z_col


def _to_scaled_array(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str | None,
    eps_xy: float,
    eps_z: float | None,
) -> np.ndarray:
    """
    Pull spatial columns into a NumPy array with Z scaled so that one unit of
    distance in scaled space corresponds to eps_xy in the original X/Y or
    eps_z in the original Z.
    """
    x = df[x_col].to_numpy().astype(np.float64, copy=False)
    y = df[y_col].to_numpy().astype(np.float64, copy=False)
    if z_col is None:
        return np.column_stack([x, y])
    z = df[z_col].to_numpy().astype(np.float64, copy=False)
    z_scaled = z * (eps_xy / eps_z)  # type: ignore[operator]
    return np.column_stack([x, y, z_scaled])


def _stabilize_labels(
    labels: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    """
    Renumber cluster labels so they're sorted by centroid (x, y[, z]).
    Noise (-1) stays as -1.
    """
    unique = np.unique(labels[labels >= 0])
    if unique.size == 0:
        return labels
    centroids = np.array([coords[labels == c].mean(axis=0) for c in unique])
    # lexsort sorts by the LAST key first, so reverse the column order:
    # we want primary by axis 0 (X), then 1 (Y), then 2 (Z).
    order = np.lexsort(centroids.T[::-1])
    remap = {old: new for new, old in enumerate(unique[order])}
    out = labels.copy()
    for old, new in remap.items():
        out[labels == old] = new
    return out


def k_distance_curve(
    df: pl.DataFrame,
    k: int = 10,
    *,
    sample_size: int = 50_000,
    mode: Mode = "3d",
    eps_xy: float = 1.0,
    eps_z: float | None = 1.0,
    columns: Sequence[str] = DEFAULT_COLUMNS,
    seed: int | None = None,
) -> pl.DataFrame:
    """
    Compute a sorted k-distance curve.  Each value is the distance from a
    sampled point to its k-th nearest neighbor, in the same scaled coordinate
    space DBSCAN will use.  Look for an "elbow" — that y-value is a sensible
    candidate for eps_xy.

    Parameters
    ----------
    df
        Input DataFrame.
    k
        Which neighbor to measure (typically equal to your planned
        ``min_samples`` value).
    sample_size
        Number of points to sample for the curve. Lower = faster, less smooth.
        50,000 is plenty for visualizing the elbow.
    mode
        ``'2d'`` or ``'3d'``.
    eps_xy, eps_z
        Used only to set the Z-axis scaling. The magnitudes don't matter for
        finding the elbow, only their ratio. If you change the ratio later
        you will need to recompute the curve.
    columns
        Column names ``(x, y, z)``.  In 2D mode only the first two are used.
    seed
        Random seed for sampling.

    Returns
    -------
    polars.DataFrame with columns:
        ``Rank``           — integer 0..N-1, the sort position
        ``k-distance (mm)``— the k-th nearest-neighbor distance in scaled space
    Plot ``k-distance (mm)`` vs ``Rank`` with ``scatter2d(equal_aspect=False)``.
    """
    x_col, y_col, z_col = _validate(df, mode, columns, eps_xy, eps_z)

    n = df.height
    if n == 0:
        return pl.DataFrame({"Rank": [], "k-distance (mm)": []})

    sample_size = min(sample_size, n)
    sampled = (
        df.sample(n=sample_size, seed=seed, shuffle=False) if sample_size < n else df
    )
    coords = _to_scaled_array(sampled, x_col, y_col, z_col, eps_xy, eps_z)

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree")
    nn.fit(coords)
    distances, _ = nn.kneighbors(coords)
    kth = distances[:, k]
    kth.sort()

    return pl.DataFrame(
        {
            "Rank": np.arange(kth.size, dtype=np.int64),
            "k-distance (mm)": kth,
        }
    )


# DBSCAN over downsampled dataset
def cluster_dbscan(
    df: pl.DataFrame,
    *,
    eps_xy: float,
    eps_z: float | None = None,
    min_samples: int = 10,
    mode: Mode = "3d",
    representative_size: int = 200_000,
    columns: Sequence[str] = DEFAULT_COLUMNS,
    seed: int | None = None,
    stable_labels: bool = True,
) -> pl.DataFrame:
    """
    DBSCAN on a downsampled representative, then propagate labels back to the
    full DataFrame via nearest-neighbor lookup in the scaled coordinate space.

    Parameters
    ----------
    df
        Input DataFrame.
    eps_xy
        Neighborhood radius in the X/Y plane (mm).  Tune via k_distance_curve.
    eps_z
        Neighborhood radius along Z (mm). Required for 3D.  Typically a small
        multiple of layer thickness, e.g. 0.06–0.1 mm.
    min_samples
        Minimum points in a neighborhood to form a core point. Default 10.
    mode
        ``'2d'`` (cluster on X, Y) or ``'3d'`` (cluster on X, Y, Z). Default 3d.
    representative_size
        How many points to actually run DBSCAN on. Lower = faster, but small
        clusters may be missed if they fall through the sampling.
    columns
        Spatial column names ``(x, y, z)``.
    seed
        Random seed for the representative sample.
    stable_labels
        If True (default), relabel clusters by centroid order so labels are
        deterministic across runs on the same data.

    Returns
    -------
    Original DataFrame with an added integer ``cluster`` column.  -1 = noise.
    """
    x_col, y_col, z_col = _validate(df, mode, columns, eps_xy, eps_z)

    if df.height == 0:
        return df.with_columns(pl.lit(0, dtype=pl.Int64).alias("cluster")).head(0)

    if df.height > representative_size:
        rep = df.sample(n=representative_size, seed=seed, shuffle=False)
    else:
        rep = df

    rep_coords = _to_scaled_array(rep, x_col, y_col, z_col, eps_xy, eps_z)

    db = DBSCAN(eps=eps_xy, min_samples=min_samples, algorithm="ball_tree", n_jobs=-1)
    rep_labels = db.fit_predict(rep_coords)

    if stable_labels:
        rep_labels = _stabilize_labels(rep_labels, rep_coords)

    if df.height == rep.height:
        full_labels = rep_labels
    else:
        full_labels = _propagate_labels(
            df,
            rep,
            rep_labels,
            x_col,
            y_col,
            z_col,
            eps_xy,
            eps_z,
        )

    return df.with_columns(pl.Series("cluster", full_labels, dtype=pl.Int64))


def _propagate_labels(
    df: pl.DataFrame,
    rep: pl.DataFrame,
    rep_labels: np.ndarray,
    x_col: str,
    y_col: str,
    z_col: str | None,
    eps_xy: float,
    eps_z: float | None,
) -> np.ndarray:
    """
    Assign each row of df the label of its nearest neighbor in rep.
    Uses a BallTree on rep in the scaled coordinate space.
    """
    rep_coords = _to_scaled_array(rep, x_col, y_col, z_col, eps_xy, eps_z)
    full_coords = _to_scaled_array(df, x_col, y_col, z_col, eps_xy, eps_z)

    tree = BallTree(rep_coords)
    _, idx = tree.query(full_coords, k=1)
    return rep_labels[idx[:, 0]]


def cluster_summary(
    df: pl.DataFrame,
    cluster_col: str = "cluster",
    columns: Sequence[str] = DEFAULT_COLUMNS,
) -> pl.DataFrame:
    """
    Per-cluster row count, bounding box, and centroid.  Useful for matching
    clusters back to physical part IDs.

    Returns a DataFrame sorted by cluster id, with columns:
    cluster, n_rows, x_min, x_max, x_mean, y_min, y_max, y_mean,
    (z_min, z_max, z_mean if those columns exist).
    """
    if cluster_col not in df.columns:
        raise KeyError(f"Column {cluster_col!r} not in DataFrame")
    have_z = len(columns) >= 3 and columns[2] in df.columns
    x_col, y_col = columns[0], columns[1]
    for c in (x_col, y_col):
        if c not in df.columns:
            raise KeyError(f"Column {c!r} not in DataFrame")

    aggs = [
        pl.len().alias("n_rows"),
        pl.col(x_col).min().alias("x_min"),
        pl.col(x_col).max().alias("x_max"),
        pl.col(x_col).mean().alias("x_mean"),
        pl.col(y_col).min().alias("y_min"),
        pl.col(y_col).max().alias("y_max"),
        pl.col(y_col).mean().alias("y_mean"),
    ]
    if have_z:
        z_col = columns[2]
        aggs.extend(
            [
                pl.col(z_col).min().alias("z_min"),
                pl.col(z_col).max().alias("z_max"),
                pl.col(z_col).mean().alias("z_mean"),
            ]
        )

    return df.group_by(cluster_col).agg(aggs).sort(cluster_col)


# Chunked DBSCAN over the FULL dataset (no downsampling)
class _UnionFind:
    """
    Disjoint-set / union-find over arbitrary hashable keys.

    Used to merge cluster labels across chunk boundaries: each global label
    starts as a (chunk_id, local_label) pair, and overlap points create
    union edges between two such pairs.
    """

    def __init__(self) -> None:
        self._parent: dict = {}
        self._rank: dict = {}

    def add(self, x) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0

    def find(self, x):
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Union by rank.
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def components(self) -> dict:
        """Return {key -> root_key} for every key ever added."""
        return {k: self.find(k) for k in self._parent}


def cluster_dbscan_chunked(
    df: pl.DataFrame,
    *,
    eps_xy: float,
    eps_z: float | None = None,
    min_samples: int = 10,
    mode: Mode = "3d",
    layers_per_chunk: int = 50,
    overlap_layers: int | None = None,
    layer_thickness: float = 0.03,
    columns: Sequence[str] = DEFAULT_COLUMNS,
    layer_col: str = "layer",
    stable_labels: bool = True,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    DBSCAN over the FULL data using layer-wise chunking with overlap merging.

    Unlike ``cluster_dbscan`` (which clusters a downsampled representative
    and propagates labels), this version runs DBSCAN on every point in each
    chunk. That avoids sampling-density artifacts but is slower and uses
    more memory per chunk.

    Memory scales with chunk size, not total data size, so this is the
    function to use when downsample-and-propagate gives unwanted
    fragmentation due to representative sparsity.

    Parameters
    ----------
    df
        Input DataFrame. Must contain ``layer_col`` and the spatial columns.
    eps_xy
        Neighborhood radius in the X/Y plane (mm).
    eps_z
        Neighborhood radius along Z (mm). Required for 3D.
    min_samples
        Minimum points in a neighborhood to form a core point.
    mode
        ``'2d'`` or ``'3d'``.
    layers_per_chunk
        Number of layers per chunk (including overlap). Default 50, suitable
        for a 16 GB machine with ~250k rows/layer (~12.5M rows per chunk =
        ~2 GB during DBSCAN with float32). Bump higher if you have RAM.
    overlap_layers
        Number of overlap layers shared with the next chunk. If None (default),
        computed as ``max(2, ceil(eps_z / layer_thickness) * 2)``, ensuring
        every boundary point sees its full eps_z neighborhood in both chunks.
        Must be < layers_per_chunk; warned and clamped if too small.
    layer_thickness
        Used only for the default overlap calculation.
    columns
        Spatial column names ``(x, y, z)``.
    layer_col
        Name of the layer column.
    stable_labels
        Relabel by centroid order so labels are deterministic. Default True.
    verbose
        Print per-chunk progress. Default True.

    Returns
    -------
    DataFrame with an added ``cluster`` Int32 column. -1 = noise.
    """
    x_col, y_col, z_col = _validate(df, mode, columns, eps_xy, eps_z)
    if layer_col not in df.columns:
        raise KeyError(f"layer_col {layer_col!r} not in DataFrame")
    if df.is_empty():
        return df.with_columns(pl.lit(-1, dtype=pl.Int32).alias("cluster"))

    if mode == "3d":
        min_overlap = max(2, int(np.ceil(eps_z / layer_thickness) * 2))
    else:
        min_overlap = 1

    if overlap_layers is None:
        overlap_layers = min_overlap
    elif overlap_layers < min_overlap:
        if verbose:
            print(
                f"WARNING: overlap_layers={overlap_layers} is below the "
                f"minimum needed for eps_z={eps_z} (>= {min_overlap}). "
                f"Clamping to {min_overlap}."
            )
        overlap_layers = min_overlap

    if overlap_layers >= layers_per_chunk:
        raise ValueError(
            f"overlap_layers ({overlap_layers}) must be < layers_per_chunk "
            f"({layers_per_chunk}). Increase layers_per_chunk."
        )

    layers_in_data = df.select(layer_col).unique().sort(layer_col)[layer_col].to_numpy()
    layer_min = int(layers_in_data.min())
    layer_max = int(layers_in_data.max())
    stride = layers_per_chunk - overlap_layers

    chunks: list[tuple[int, int]] = []  # (lo, hi) inclusive
    lo = layer_min
    while lo <= layer_max:
        hi = min(lo + layers_per_chunk - 1, layer_max)
        chunks.append((lo, hi))
        if hi == layer_max:
            break
        lo += stride

    if verbose:
        print(
            f"Chunked DBSCAN: {len(chunks)} chunks of up to "
            f"{layers_per_chunk} layers (overlap {overlap_layers}), "
            f"layers {layer_min}-{layer_max}"
        )

    # Process chunk
    n = df.height
    row_chunk = np.full(n, -1, dtype=np.int32)
    row_local = np.full(n, -1, dtype=np.int32)

    layer_arr = df[layer_col].to_numpy()
    uf = _UnionFind()

    for chunk_id, (lo, hi) in enumerate(chunks):
        in_chunk = (layer_arr >= lo) & (layer_arr <= hi)
        chunk_indices = np.flatnonzero(in_chunk)
        if chunk_indices.size == 0:
            continue
        chunk_df = df[chunk_indices]
        coords = _to_scaled_array(chunk_df, x_col, y_col, z_col, eps_xy, eps_z)

        if verbose:
            print(
                f"  chunk {chunk_id + 1}/{len(chunks)} "
                f"layers {lo}-{hi}: {chunk_df.height:,} rows"
            )

        labels = DBSCAN(
            eps=eps_xy,
            min_samples=min_samples,
            algorithm="ball_tree",
            n_jobs=-1,
        ).fit_predict(coords)

        for L in np.unique(labels):
            if L >= 0:
                uf.add((chunk_id, int(L)))

        # For rows that already have a label from a previous chunk (i.e.,
        # they fell in the previous chunk's overlap zone), union the two
        # labels — provided neither is noise. Trust non-noise: if one is -1
        # and the other is a real label, take the real label.
        prev_chunk = row_chunk[chunk_indices]
        prev_local = row_local[chunk_indices]
        already_labeled = prev_chunk >= 0
        if already_labeled.any():
            prev_keys_chunk = prev_chunk[already_labeled]
            prev_keys_local = prev_local[already_labeled]
            new_labels = labels[already_labeled]

            both = (new_labels >= 0) & (prev_keys_local >= 0)
            if both.any():
                pc = prev_keys_chunk[both].astype(np.int64)
                pl_ = prev_keys_local[both].astype(np.int64)
                nl = new_labels[both].astype(np.int64)

                packed = (pc << np.int64(40)) | (pl_ << np.int64(20)) | nl
                unique_pairs = np.unique(packed)
                for p in unique_pairs:
                    a_chunk = int(p >> np.int64(40)) & 0xFFFFF
                    a_local = int(p >> np.int64(20)) & 0xFFFFF
                    b_local = int(p) & 0xFFFFF
                    uf.union((a_chunk, a_local), (chunk_id, b_local))

        new_is_real = labels >= 0
        prev_is_real = prev_local >= 0
        # Overwrite when either:
        #   (a) row is unlabeled (prev = -1), OR
        #   (b) prev was noise and new is real, OR
        #   (c) prev was real and new is real (both will be unioned, doesn't
        #       matter which wins — they map to the same global root).
        # Don't overwrite when prev was real and new is noise.
        overwrite = ~(prev_is_real & ~new_is_real)
        target = chunk_indices[overwrite]
        row_chunk[target] = chunk_id
        row_local[target] = labels[overwrite]

    # Resolve global labels
    roots: dict = {}
    next_id = 0
    components = uf.components()
    for key, root in components.items():
        if root not in roots:
            roots[root] = next_id
            next_id += 1

    final_labels = np.full(n, -1, dtype=np.int32)
    real = row_local >= 0
    if real.any():
        chunks_arr = row_chunk[real]
        locals_arr = row_local[real]
        packed = (chunks_arr.astype(np.uint64) << np.uint64(32)) | locals_arr.astype(
            np.uint64
        )
        unique_packed, inverse = np.unique(packed, return_inverse=True)
        unique_global = np.empty(unique_packed.size, dtype=np.int32)
        for i, p in enumerate(unique_packed):
            c = int(p >> np.uint64(32))
            l = int(p & np.uint64(0xFFFFFFFF))
            root = uf.find((c, l))
            unique_global[i] = roots[root]
        final_labels[real] = unique_global[inverse]

    if stable_labels and (final_labels >= 0).any():
        spatial = _to_scaled_array(df, x_col, y_col, z_col, eps_xy, eps_z)
        final_labels = _stabilize_labels(final_labels, spatial)

    return df.with_columns(pl.Series("cluster", final_labels, dtype=pl.Int32))
