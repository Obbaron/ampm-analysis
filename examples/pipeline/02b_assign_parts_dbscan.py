"""
02b_assign_parts_dbscan.py

Clusters the masked data with DBSCAN, then matches each cluster to a
QuantAM Part ID by the nearest centroid. The result is a DataFrame
with a ``part_id`` string column.

Pipeline branching
------------------
- Use DBSCAN when parts are small and closely-spaced.
- Use direct assignment when parts are large and well-separated.

Choice of DBSCAN variant
------------------------
This script uses ``cluster_dbscan_chunked``, which runs DBSCAN on the
full data within overlapping layer chunks and merges labels across
boundaries. It is robust at low ``eps_xy`` values but memory-heavy:
peak RAM is roughly ``LAYERS_PER_CHUNK × rows_per_layer × ~3``.
For 250k rows/layer that's ~3-6 GB peak.

The alternative ``cluster_dbscan`` (downsample-and-propagate) is faster
and lighter on memory but fragile at low ``eps_xy`` on builds with a
large spatial footprint. The representative sample becomes spatially
sparse and DBSCAN fragments or labels everything as noise.

Tuning ``EPS_XY``
-----------------
Run ``examples/tune_eps.py`` first to find the right value for your
build. The defaults below were validated on the JR299 sterling build.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import polars as pl

from ampm import DataStore
from ampm.cluster_cache import cluster_or_load
from ampm.clustering import cluster_dbscan_chunked, cluster_summary
from ampm.mask_cache import mask_or_load
from ampm.masking import apply_mask, build_mask
from ampm.parts import QuantAMParts, apply_part_id_map, compute_part_id_map
from ampm.config import create_or_load_config


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage: python 02b_assign_parts_dbscan.py <build_directory>")
    config = create_or_load_config(sys.argv[1])

    SOURCE = config["SOURCE"]
    STL = config["STL"]
    PARTS_CSV = config["PARTS_CSV"]
    LAYER_THICKNESS = config["LAYER_THICKNESS"]
    MASK_CACHE = config["MASK_CACHE"]
    MASK_KEEP_CACHE = config["MASK_KEEP_CACHE"]
    CLUSTER_CACHE = config["CLUSTER_CACHE"]
    EPS_XY = config["EPS_XY"]
    EPS_Z = config["EPS_Z"]
    MIN_SAMPLES = config["MIN_SAMPLES"]
    LAYERS_PER_CHUNK = config["LAYERS_PER_CHUNK"]
    OVERLAP_LAYERS = config["OVERLAP_LAYERS"]

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

    def clustering_wrapper(d: pl.DataFrame) -> pl.DataFrame:  # same as masking_wrapper
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

    n_clusters = sum(1 for c in clustered["cluster"].unique() if c >= 0)
    n_noise = (clustered["cluster"] == -1).sum()
    print(
        f"\nFound {n_clusters} clusters, {n_noise:,} noise pts "
        f"({n_noise / clustered.height:.1%})."
    )

    print("\nCluster summary:")
    print(cluster_summary(clustered))

    quantam = QuantAMParts.from_path(PARTS_CSV)
    parts_table = quantam.parent_parts()
    print(f"\nLoaded {parts_table.height} parts from {Path(PARTS_CSV).name}.")

    print("\nMatching clusters to parts by nearest centroid...")
    mapping = compute_part_id_map(clustered, parts_table)
    print("\nCluster -> Part mapping:")
    for cid in sorted(mapping):
        print(f"  cluster {cid:2d} -> {mapping[cid]}")

    assigned = apply_part_id_map(clustered, mapping, noise_label="noise")

    print(
        f"\nResult: {assigned.height:,} rows, "
        f"{assigned['part_id'].n_unique()} unique part_id values."
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
