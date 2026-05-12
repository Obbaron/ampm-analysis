"""
03_compute_cov.py

Computes the Coefficient of Variation (CoV) of each monitored signal,
grouped by part, in three different modes. Then joins the result with
the QuantAM laser parameters so each part's CoV sits alongside its
Hatches Power and Hatch Speed.

Three CoV modes
---------------
- ``overall``: total variability across all rows in each part
  (intra-layer + drift + outliers all mixed together).
- ``per_layer_mean``: average within-layer CoV. This filters out
  layer-to-layer drift, leaving only within-layer noise.
- ``across_layers``: CoV of per-layer means. The complement of
  per_layer_mean - only drift, no within-layer noise.

A part with high overall CoV but low per_layer_mean was clean within
each layer but drifted over the build. A part with low overall but
high per_layer_mean has noisy layers that happen to average to similar
means. The three modes together diagnose where instability resides.

Branching
---------
Set USE_DIRECT_ASSIGNMENT to choose direct assignment or DBSCAN.

Optional XY-bias correction
---------------------------
For MAIN machine MeltVIEW data, the melt-pool signal has a smooth
spatial bias that inflates per-part CoV. See ``cov.py`` in the
examples folder for how to apply ``MeltPoolCorrection.apply()`` before
computing CoV. Omitted here for clarity.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import polars as pl

from ampm import DataStore
from ampm.cluster_cache import cluster_or_load
from ampm.clustering import cluster_dbscan_chunked
from ampm.mask_cache import mask_or_load
from ampm.masking import apply_mask, build_mask
from ampm.parts import (
    QuantAMParts,
    apply_part_id_map,
    assign_nearest_part,
    compute_part_id_map,
    join_parts_with_stats,
)
from ampm.stats import compute_cov
from config import (
    CLUSTER_CACHE,
    LAYER_THICKNESS,
    MASK_CACHE,
    MASK_KEEP_CACHE,
    PARTS_CSV,
    SOURCE,
    STL,
)

USE_DIRECT_ASSIGNMENT = True  # False = use DBSCAN clustering
MAX_DISTANCE_MM = None  # only used when USE_DIRECT_ASSIGNMENT = True

# DBSCAN parameters only used when USE_DIRECT_ASSIGNMENT = False
EPS_XY = 0.3
EPS_Z = 0.06
MIN_SAMPLES = 10
LAYERS_PER_CHUNK = 11
OVERLAP_LAYERS = None

SIGNALS = [
    "MeltVIEW melt pool (mean)",
    "Laser output power (mean)",
]


def main() -> None:
    store = DataStore(SOURCE, layer_thickness=LAYER_THICKNESS)

    df = store.query()
    print(f"Loaded {df.height:,} rows across {len(store.layers)} layers.")

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
    print(f"After mask: {df_masked.height:,} rows.")

    del df

    quantam = QuantAMParts.from_path(PARTS_CSV)
    parts_table = quantam.parent_parts()
    print(f"Loaded {parts_table.height} parts from {Path(PARTS_CSV).name}.")

    if USE_DIRECT_ASSIGNMENT:
        print("\nAssigning each row to its nearest part (direct method)...")
        assigned = assign_nearest_part(
            df_masked,
            parts_table,
            max_distance_mm=MAX_DISTANCE_MM,
            noise_label="noise",
        )
    else:
        cluster_params = {
            "layers": (min(store.layers), max(store.layers)),
            "stl": str(STL),
            "buffer_mm": 0.0,
            "eps_xy": EPS_XY,
            "eps_z": EPS_Z,
            "min_samples": MIN_SAMPLES,
            "mode": "3d",
            "layers_per_chunk": LAYERS_PER_CHUNK,
            "overlap_layers": OVERLAP_LAYERS,
            "layer_thickness": LAYER_THICKNESS,
        }

        def clustering_wrapper(d: pl.DataFrame) -> pl.DataFrame:
            return cluster_dbscan_chunked(
                d,
                eps_xy=EPS_XY,
                eps_z=EPS_Z,
                min_samples=MIN_SAMPLES,
                mode="3d",
                layers_per_chunk=LAYERS_PER_CHUNK,
                overlap_layers=OVERLAP_LAYERS,
                layer_thickness=LAYER_THICKNESS,
                verbose=True,
            )

        print("\nClustering with chunked DBSCAN...")
        clustered = cluster_or_load(
            df_masked,
            cache_path=CLUSTER_CACHE,
            cluster_fn=clustering_wrapper,
            params=cluster_params,
            strict=True,
        )

        print("\nMatching clusters to parts by nearest centroid...")
        mapping = compute_part_id_map(clustered, parts_table)
        assigned = apply_part_id_map(clustered, mapping, noise_label="noise")

    print("\nCoV mode: 'overall'")
    print("Total variability across all rows per part.\n")
    cov_overall = compute_cov(
        assigned,
        SIGNALS,
        group_by="part_id",
        mode="overall",
        noise_label="noise",
    )
    print(cov_overall)

    print("\nCoV mode: 'per_layer_mean'")
    print("Mean of per-layer CoVs (filters out drift).\n")
    cov_per_layer = compute_cov(
        assigned,
        SIGNALS,
        group_by="part_id",
        mode="per_layer_mean",
        noise_label="noise",
    )
    print(cov_per_layer)

    print("\nCoV mode: 'across_layers'")
    print("CoV of per-layer means (only drift).\n")
    cov_across = compute_cov(
        assigned,
        SIGNALS,
        group_by="part_id",
        mode="across_layers",
        noise_label="noise",
    )
    print(cov_across)

    print("\nCoV (overall) joined with laser parameters")
    parts_with_speed = quantam.volume_parameters_with_speed()
    joined = join_parts_with_stats(cov_overall, parts_with_speed)
    print(
        joined.select(
            [
                "part_id",
                "Hatches Power",
                "Hatch Speed",
                *[f"cov_{s}" for s in SIGNALS],
            ]
        )
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
