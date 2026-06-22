"""
STL masking of AMPM data

Uses full plate export from QuantAM STL to keep only data points
whose (x, y) falls inside that layer's polygon.

Usage
-----
    from ampm import DataStore
    from ampm.masking import build_mask, apply_mask

    store = DataStore(SOURCE_DIR)
    df = store.query(layers=(101, 200))

    mask = build_mask(
        "parts.stl",
        layers=range(101, 201),
        layer_thickness=0.03,
        cache_path="parts.mask.pkl",  # optional but recommended
    )
    df_part_only = apply_mask(df, mask)

The mask is a dict ``{layer_number: shapely.MultiPolygon}``.  Layers with no
intersection are absent from the dict; those rows are dropped by ``apply_mask``.
"""

from __future__ import annotations

import hashlib
import pickle
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
import shapely
import trimesh
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

Mask = dict[int, BaseGeometry]

# Size threshold used by stl_hash to switch from full hashing to sampling.
LARGE_STL_BYTES = 256 * 2**20  # 256 MB


def build_mask(
    stl_path: str | Path,
    layers: Iterable[int],
    layer_thickness: float = 0.03,
    buffer_mm: float = 0.0,
    cache_path: str | Path | None = None,
    force: bool = False,
) -> Mask:
    """
    Slice ``stl_path`` at ``layer_n * layer_thickness`` for each layer and
    return a dict mapping layer number to the 2D mask geometry at that height.

    Parameters
    ----------
    stl_path
        Path to the STL file (binary or ASCII).
    layers
        Layer numbers to slice. Layers with no intersection are silently
        dropped — this is normal for layers above/below the part.
    layer_thickness
        Physical layer thickness in mm. Z for layer N is ``N * layer_thickness``.
    buffer_mm
        Optional polygon buffer applied after slicing.
          > 0 : grow the polygon outward (lenient — include points near the edge)
          < 0 : shrink the polygon inward (strict — exclude near-edge points)
          = 0 : no buffer (default)
    cache_path
        If given, pickle the mask to this path on first build and reload it on
        subsequent calls. The cache is keyed to a hash of (stl content,
        layer_thickness, buffer_mm, layer set) so it is invalidated automatically
        when any of those change.
    force
        If True, ignore any existing cache and rebuild.

    Returns
    -------
    dict[int, shapely geometry]
    """
    stl_path = Path(stl_path)

    if not stl_path.is_file():
        raise FileNotFoundError(f"STL not found:\n{stl_path}")

    layer_list = sorted(set(int(L) for L in layers))
    if not layer_list:
        raise ValueError("No layers requested.")

    cache_key = _cache_key(stl_path, layer_list, layer_thickness, buffer_mm)

    if cache_path is not None:
        cache_path = Path(cache_path)

        if not force and cache_path.is_file():
            cached = _load_cache(cache_path)

            if cached is not None and cached.get("key") == cache_key:

                return cached["mask"]

    raw_mask: Mask | None = None
    try:
        raw_mask = _slice_streaming(stl_path, layer_list, layer_thickness)

    except ValueError:
        raw_mask = None  # ASCII or malformed binary -> use trimesh

    except Exception as exc:  # pragma: no cover - defensive fallback
        warnings.warn(
            f"Streaming slicer errored ({exc!r}); falling back to trimesh.",
            RuntimeWarning,
            stacklevel=2,
        )
        raw_mask = None

    if raw_mask is None:
        raw_mask = _slice_trimesh(stl_path, layer_list, layer_thickness)

    mask: Mask = {}
    for layer_n, geom in raw_mask.items():
        if buffer_mm != 0.0:
            geom = geom.buffer(buffer_mm)

            if geom.is_empty:
                continue

        if not isinstance(geom, (Polygon, MultiPolygon)):
            continue

        mask[layer_n] = geom

    if cache_path is not None:
        _save_cache(cache_path, {"key": cache_key, "mask": mask})

    return mask


def _slice_trimesh(
    stl_path: Path, layer_list: list[int], layer_thickness: float
) -> Mask:
    """In-memory slicing via trimesh."""
    mesh = trimesh.load_mesh(stl_path)

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(
            f"Expected a single mesh from {stl_path}, got {type(mesh).__name__}. "
        )

    if mesh.is_empty or mesh.bounds is None or len(mesh.faces) == 0:
        raise ValueError(f"trimesh loaded {stl_path} as an EMPTY mesh.")

    heights = np.array([L * layer_thickness for L in layer_list], dtype=float)

    # section_multiplane is much faster than calling section() in a loop.
    # plane_origin is the *base* point; heights are offsets along plane_normal.
    sections = mesh.section_multiplane(
        plane_origin=[0.0, 0.0, 0.0],
        plane_normal=[0.0, 0.0, 1.0],
        heights=heights,
    )

    mask: Mask = {}
    for layer_n, section in zip(layer_list, sections):
        if section is None:
            continue  # plane misses the mesh
        try:
            polys = list(section.polygons_full)
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ModuleNotFoundError(
                f"Slicing needs a trimesh support package that is not "
                f"installed: {e.name!r}. The slicing path "
                f"(section.polygons_full) relies on shapely, networkx and "
                f"rtree, which are pinned project dependencies and should "
                f"already be present.\n"
                f"To repair the environment offline, reinstall from the "
                f"bundled wheel set (no network needed):\n"
                f"    pip install --no-index --find-links wheels\\windows {e.name}\n"
                f"or re-run the project offline install to restore all "
                f"dependencies."
            ) from e
        if not polys:  # pragma: no cover
            continue
        mask[layer_n] = polys[0] if len(polys) == 1 else MultiPolygon(polys)
    return mask


def _slice_streaming(
    stl_path: Path, layer_list: list[int], layer_thickness: float
) -> Mask:
    """Constant-memory slicing for huge STL files (e.g. lattice builds)."""
    try:
        from .stl_stream import slice_stl_streaming
    except ImportError:  # running as a plain module outside the package
        from stl_stream import slice_stl_streaming

    return slice_stl_streaming(stl_path, layer_list, layer_thickness)


def apply_mask(
    df: pl.DataFrame,
    mask: Mask,
    x_col: str = "Demand X",
    y_col: str = "Demand Y",
    layer_col: str = "layer",
    chunk_rows: int = 8_000_000,
) -> pl.DataFrame:
    """
    Return only the rows of ``df`` that fall inside the mask polygon for
    their respective layer.

    Rows whose layer is not in the mask (e.g. layers above or below the part,
    or layers we never sliced) are dropped.

    Memory: the containment test is streamed in chunks of rows, so peak
    overhead is one boolean array of length N plus one chunk of coordinates
    (~200 MB at the default), independent of build density.

    Parameters
    ----------
    df
        DataFrame with columns ``x_col``, ``y_col``, ``layer_col``.
    mask
        Output of ``build_mask``.
    x_col, y_col, layer_col
        Column names. Defaults match the DataStore output.
    chunk_rows
        Rows tested per chunk. Smaller bounds memory tighter; larger is
        marginally faster.
    """
    keep = apply_mask_keep(
        df,
        mask,
        x_col=x_col,
        y_col=y_col,
        layer_col=layer_col,
        chunk_rows=chunk_rows,
    )
    if keep is None:  # empty input
        return df
    return df.filter(pl.Series(keep))


def apply_mask_keep(
    df: pl.DataFrame,
    mask: Mask,
    x_col: str = "Demand X",
    y_col: str = "Demand Y",
    layer_col: str = "layer",
    chunk_rows: int = 8_000_000,
) -> np.ndarray | None:
    """
    Computes boolean keep-array for ``df`` against ``mask`` without
    filtering the DataFrame.

    Parameters
    ----------
    df
        DataFrame with columns ``x_col``, ``y_col``, ``layer_col``.
    mask
        Output of ``build_mask``.
    x_col, y_col, layer_col
        Column names. Defaults match the DataStore output.
    chunk_rows
        Rows tested per chunk.

    Returns
    -------
    ndarray[bool, shape (df.height,), or None]
        True where the row falls inside its layer's geometry; None if
        ``df`` is empty.
    """
    for c in (x_col, y_col, layer_col):
        if c not in df.columns:
            raise KeyError(f"Column {c!r} not in DataFrame")

    if df.is_empty():
        return None

    for geom in mask.values():
        shapely.prepare(geom)

    keep = np.zeros(df.height, dtype=bool)

    for start in range(0, df.height, chunk_rows):
        sub = df.slice(start, chunk_rows).select(x_col, y_col, layer_col)
        xs = sub[x_col].to_numpy()
        ys = sub[y_col].to_numpy()
        layers = sub[layer_col].to_numpy()

        for layer_n in np.unique(layers):
            geom = mask.get(int(layer_n))
            if geom is None:
                continue
            idx = np.flatnonzero(layers == layer_n)
            keep[start + idx] = shapely.contains_xy(geom, xs[idx], ys[idx])

    return keep


def stl_hash(stl_path: str | Path) -> str:
    """SHA-256 fingerprint of the STL file contents."""
    stl = Path(stl_path)
    size = stl.stat().st_size
    hash = hashlib.sha256()
    with stl.open("rb") as file:
        if size <= LARGE_STL_BYTES:
            for chunk in iter(lambda: file.read(1 << 20), b""):
                hash.update(chunk)
        else:
            sample = 8 * 2**20
            hash.update(f"sampled:{size}:".encode())
            for offset in sorted({0, max(0, size // 2), max(0, size - sample)}):
                file.seek(offset)
                hash.update(file.read(sample))
    return hash.hexdigest()


def _cache_key(
    stl_path: Path,
    layers: list[int],
    layer_thickness: float,
    buffer_mm: float,
) -> str:
    """Hash STL contents + slicing params so the cache invalidates correctly."""
    hash = hashlib.sha256(stl_hash(stl_path).encode())
    hash.update(repr((layers, layer_thickness, buffer_mm)).encode())
    return hash.hexdigest()


def _save_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_cache(path: Path) -> dict | None:
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return None
