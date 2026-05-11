"""
tune_eps.py - interactive DBSCAN tuning workflow.

Walks through three stages of tuning the in-plane neighborhood radius
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

EPS_XY = 0.65
EPS_Z = 2 * LAYER_THICKNESS
K = 10
MODE = "3d"
SAMPLE_SIZE = 1000000


def main() -> None:
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

    def masking_wrapper(d: pl.DataFrame) -> pl.DataFrame:
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
        mask_fn=masking_wrapper,
        params=mask_params,
        strict=True,
    )
    print(f"After mask: {df_masked.height:,} rows")
    del df

    print(f"Stage 1: k-distance curve (k={K}, sample={SAMPLE_SIZE:,})")
    print(
        "Plotting the k-th nearest-neighbor distance for each sampled "
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

    # fig_kdist = scatter2d(
    #    curve,
    #    x="Rank",
    #    y="k-distance (mm)",
    #    equal_aspect=False,
    #    size=4,
    #    title=(
    #        f"k-distance curve (k={K}, mode={MODE}). "
    #        f"Look for the elbow — that's your EPS_XY."
    #    ),
    #    xaxis_title="Rank (sorted)",
    #    yaxis_title=f"Distance to {K}-th neighbor (mm)",
    # )
    # fig_kdist.show()

    q50 = curve["k-distance (mm)"].quantile(0.50)
    q90 = curve["k-distance (mm)"].quantile(0.90)
    q95 = curve["k-distance (mm)"].quantile(0.95)
    q99 = curve["k-distance (mm)"].quantile(0.99)
    print(f"k-distance quantiles (sample of {curve.height:,} points):")
    print(f"  50th percentile: {q50:.3f} mm")
    print(f"  90th percentile: {q90:.3f} mm")
    print(f"  95th percentile: {q95:.3f} mm")
    print(f"  99th percentile: {q99:.3f} mm")
    print("\nThe elbow usually sits between the 90th and 99th percentile. ")


if __name__ == "__main__":
    main()
