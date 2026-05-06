"""
tune_eps.py — interactive DBSCAN tuning workflow.

Walks through three stages of tuning the in-plane neighbourhood radius
EPS_XY for clustering AMPM data into individual parts:

  Stage 1: Compute and plot the k-distance curve. The "elbow" of this
           curve is a good initial guess for EPS_XY.
  Stage 2: Run DBSCAN with the chosen EPS_XY and inspect the result.
           Check cluster count, noise fraction, and per-cluster sizes.
  Stage 3: Validate cluster-to-part mapping. The QuantAM parts CSV gives
           the known number of parts and their XY positions. A correctly
           tuned EPS_XY produces (a) the right number of clusters and
           (b) sub-millimeter centroid distances to known part positions.

How to use
----------
1. Set EPS_XY below to your initial guess (try 0.5 if you have no idea).
2. Run `python tune_eps.py`.
3. Look at the k-distance plot. If the elbow is at a different y-value
   than your guess, update EPS_XY and rerun.
4. Check Stage 2's cluster count. If it doesn't match the expected
   number of parts, adjust EPS_XY:
       - Too many clusters → EPS_XY too small, increase it
       - Too few clusters  → EPS_XY too large, decrease it
5. Iterate until Stage 3 reports max distance < ~1 mm and no warnings.

Note on EPS_Z
-------------
EPS_Z controls clustering in the through-thickness direction. It is
typically picked as a small multiple of layer thickness — usually
2 * LAYER_THICKNESS = 0.06 mm. This is robust to single-layer data gaps
without bridging across larger discontinuities. You generally do NOT
need to tune EPS_Z via the k-distance curve; the rule of thumb works.

If your build has frequent missing-data layers, increase EPS_Z to
4-5 * LAYER_THICKNESS. If parts are clustering into vertical slabs
rather than full-height columns, EPS_Z is too small.
"""

# Make config.py at the project root importable.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl

from ampm import DataStore
from ampm.clustering import (
    cluster_dbscan,
    cluster_summary,
    k_distance_curve,
)
from ampm.mask_cache import mask_or_load
from ampm.masking import apply_mask, build_mask
from ampm.parts import QuantAMParts, compute_part_id_map
from ampm.plotting import scatter2d

from config import (
    LAYER_THICKNESS,
    MASK_CACHE,
    MASK_KEEP_CACHE,
    PARTS_CSV,
    SOURCE,
    STL,
)


# ----- Tuning parameters (edit these) -----
EPS_XY = 0.3                # The parameter you're tuning (mm)
EPS_Z = 2 * LAYER_THICKNESS # Through-thickness eps; rule of thumb above
MIN_SAMPLES = 10            # DBSCAN density threshold; same as your `cov.py`
K = 10                      # k-th nearest neighbour for the k-distance curve
MODE = "3d"                 # "2d" or "3d" — JR299 needs 3d
SAMPLE_SIZE = 50_000        # k-distance curve points; 50k is plenty


def main() -> None:
    # ----- Load and mask (uses caches if present) -----
    store = DataStore(SOURCE, layer_thickness=LAYER_THICKNESS)
    print(store)

    df = store.query()
    print(f"Full slice: {df.height:,} rows")

    mask_params = {
        "layers": (min(store.layers), max(store.layers)),
        "stl": str(STL),
        "buffer_mm": 0.0,
        "layer_thickness": LAYER_THICKNESS,
    }

    def do_masking(d: pl.DataFrame) -> pl.DataFrame:
        mask = build_mask(
            STL,
            layers=store.layers,
            layer_thickness=LAYER_THICKNESS,
            buffer_mm=0.0,
            cache_path=MASK_CACHE,
        )
        return apply_mask(d, mask)

    df_masked = mask_or_load(
        df,
        cache_path=MASK_KEEP_CACHE,
        mask_fn=do_masking,
        params=mask_params,
        strict=True,
    )
    print(f"After mask: {df_masked.height:,} rows")
    del df

    # ----- Stage 1: k-distance curve -----
    print(f"\n{'=' * 60}")
    print(f"Stage 1: k-distance curve (k={K}, sample={SAMPLE_SIZE:,})")
    print(f"{'=' * 60}")
    print(
        "Plotting the k-th nearest-neighbour distance for each sampled "
        "point. The 'elbow' is a good candidate for EPS_XY.\n"
    )

    curve = k_distance_curve(
        df_masked,
        k=K,
        sample_size=SAMPLE_SIZE,
        mode=MODE,
        eps_xy=EPS_XY,
        eps_z=EPS_Z,
        seed=0,
    )

    fig_kdist = scatter2d(
        curve,
        x="Rank",
        y="k-distance (mm)",
        equal_aspect=False,
        size=4,
        title=(
            f"k-distance curve (k={K}, mode={MODE}). "
            f"Look for the elbow — that's your EPS_XY."
        ),
        xaxis_title="Rank (sorted)",
        yaxis_title=f"Distance to {K}-th neighbour (mm)",
    )
    fig_kdist.show()

    # Print a couple of quantiles so the user has numerical anchors
    # without having to read pixel positions off the plot.
    q50 = curve["k-distance (mm)"].quantile(0.50)
    q90 = curve["k-distance (mm)"].quantile(0.90)
    q95 = curve["k-distance (mm)"].quantile(0.95)
    q99 = curve["k-distance (mm)"].quantile(0.99)
    print(f"k-distance quantiles (sample of {curve.height:,} points):")
    print(f"  50th percentile: {q50:.3f} mm")
    print(f"  90th percentile: {q90:.3f} mm")
    print(f"  95th percentile: {q95:.3f} mm")
    print(f"  99th percentile: {q99:.3f} mm")
    print(
        f"\nThe elbow usually sits between the 90th and 99th percentile. "
        f"Your current EPS_XY = {EPS_XY} mm."
    )

    # ----- Stage 2: trial DBSCAN with current EPS_XY -----
    print(f"\n{'=' * 60}")
    print(f"Stage 2: DBSCAN with EPS_XY = {EPS_XY}, EPS_Z = {EPS_Z}")
    print(f"{'=' * 60}")

    clustered = cluster_dbscan(
        df_masked,
        eps_xy=EPS_XY,
        eps_z=EPS_Z,
        min_samples=MIN_SAMPLES,
        mode=MODE,
    )

    n_clusters = sum(1 for c in clustered["cluster"].unique() if c >= 0)
    n_noise = (clustered["cluster"] == -1).sum()
    noise_pct = n_noise / clustered.height
    print(f"Clusters found: {n_clusters}")
    print(f"Noise points:   {n_noise:,} ({noise_pct:.2%})")

    summary = cluster_summary(clustered)
    if n_clusters > 0:
        print("\nPer-cluster row counts and centroids:")
        cluster_rows = summary.filter(pl.col("cluster") >= 0)
        rows_min = cluster_rows["n_rows"].min()
        rows_max = cluster_rows["n_rows"].max()
        rows_ratio = rows_max / rows_min if rows_min > 0 else float("inf")
        print(
            cluster_rows.select(
                ["cluster", "n_rows", "x_mean", "y_mean", "z_min", "z_max"]
            )
        )
        print(
            f"\nCluster size ratio (max/min): {rows_ratio:.1f}× — "
            "low values (< 3×) usually mean clean clustering."
        )

    # ----- Stage 3: validate against parts CSV -----
    print(f"\n{'=' * 60}")
    print("Stage 3: validate cluster-to-part mapping")
    print(f"{'=' * 60}")

    quantam = QuantAMParts.from_path(PARTS_CSV)
    parts_table = quantam.parent_parts()
    n_expected = parts_table.height
    print(
        f"\nExpected {n_expected} parts (from {Path(PARTS_CSV).name}); "
        f"DBSCAN found {n_clusters} clusters."
    )

    if n_clusters != n_expected:
        if n_clusters > n_expected:
            print(
                f"  → {n_clusters - n_expected} too many clusters. "
                "Try increasing EPS_XY."
            )
        else:
            print(
                f"  → {n_expected - n_clusters} too few clusters. "
                "Try decreasing EPS_XY."
            )

    # compute_part_id_map prints its own diagnostics about collisions,
    # far matches, and unmatched parts.
    mapping = compute_part_id_map(clustered, parts_table)

    if n_clusters == n_expected and len(mapping) == n_expected:
        print(
            f"\n✓ Tuning looks good: {n_clusters} clusters, "
            f"{len(mapping)} parts mapped, "
            f"noise {noise_pct:.2%}. EPS_XY = {EPS_XY} appears correct."
        )
    else:
        print(
            "\n✗ Tuning is not yet correct. Adjust EPS_XY based on the "
            "guidance above and rerun."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
