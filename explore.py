"""
explore.py — full pipeline: load, mask, cluster, optionally correct
melt-pool signal, assign part IDs, analyze process stability via CoV,
and visualize, including a parametric process map and a per-layer signal
viewer.
"""

from pathlib import Path

import polars as pl

from ampm import DataStore
from ampm.sampling import prepare_for_plot
from ampm.plotting import scatter3d, scatter2d, scatter2d_layered, bar, contour
from ampm.masking import build_mask, apply_mask
from ampm.mask_cache import mask_or_load
from ampm.clustering import cluster_dbscan_chunked, cluster_summary
from ampm.cluster_cache import cluster_or_load
from ampm.stats import compute_cov
from ampm.correction import MeltPoolCorrection
from ampm.parts import (
    QuantAMParts,
    apply_part_id_map,
    compute_part_id_map,
    join_parts_with_stats,
)

from config import (
    SOURCE,
    STL,
    PARTS_CSV,
    MASK_CACHE,
    MASK_KEEP_CACHE,
    CLUSTER_CACHE,
    LAYER_THICKNESS,
)

EPS_XY = 0.3
EPS_Z = 0.06
MIN_SAMPLES = 10
LAYERS_PER_CHUNK = 11
OVERLAP_LAYERS = 2

CORRECT_MELTPOOL = False

SIGNALS = [
    "MeltVIEW melt pool (mean)",
    "MeltVIEW plasma (mean)",
    "Laser back reflection (mean)",
    "Laser output power (mean)",
]
COV_PLOT_SIGNAL = (
    "MeltVIEW melt pool (mean) corrected"
    if CORRECT_MELTPOOL
    else "MeltVIEW melt pool (mean)"
)


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

    def do_masking(d: pl.DataFrame) -> pl.DataFrame:
        mask = build_mask(
            STL,
            layers=store.layers,
            layer_thickness=LAYER_THICKNESS,
            buffer_mm=0.0,
            cache_path=MASK_CACHE,
        )
        print(f"Mask covers {len(mask)} of {len(store.layers)} layers")
        return apply_mask(d, mask)

    df_masked = mask_or_load(
        df,
        cache_path=MASK_KEEP_CACHE,
        mask_fn=do_masking,  # I don't like lambda functions
        params=mask_params,
        strict=True,
    )
    survival = df_masked.height / df.height
    print(f"After mask: {df_masked.height:,} rows ({survival:.1%} kept)")
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

    def do_clustering(d: pl.DataFrame) -> pl.DataFrame:
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

    clustered = cluster_or_load(
        df_masked,
        cache_path=CLUSTER_CACHE,
        cluster_fn=do_clustering,
        params=cluster_params,
        strict=True,
    )

    n_clusters = sum(1 for c in clustered["cluster"].unique() if c >= 0)
    n_noise = (clustered["cluster"] == -1).sum()
    print(
        f"\n{n_clusters} clusters, {n_noise:,} noise pts ({n_noise/clustered.height:.1%})"
    )
    # print(cluster_summary(clustered))

    quantam = QuantAMParts.from_path(PARTS_CSV)
    parts_table = quantam.parent_parts()
    print(f"\nLoaded {parts_table.height} parts from {Path(PARTS_CSV).name}")

    mapping = compute_part_id_map(clustered, parts_table)
    print("\nCluster -> Part mapping:")
    for cid in sorted(mapping):
        print(f"  cluster {cid:2d} -> {mapping[cid]}")

    clustered = apply_part_id_map(clustered, mapping, noise_label="noise")

    if CORRECT_MELTPOOL:
        print("\nApplying MAIN-machine MeltVIEW XY-bias correction...")
        correction = MeltPoolCorrection()
        clustered = correction.apply(clustered)
        print(f"  added column: {COV_PLOT_SIGNAL!r}")
        signals_for_cov = [
            COV_PLOT_SIGNAL if s == "MeltVIEW melt pool (mean)" else s for s in SIGNALS
        ]
    else:
        signals_for_cov = SIGNALS

    print("\nComputing overall Coefficient of Variation...")
    cov_overall = compute_cov(
        clustered,
        signals_for_cov,
        group_by="part_id",
        mode="overall",
        noise_label="noise",
    )
    print(cov_overall)

    print("\nLinking parts to CoV...")
    parts_with_speed = quantam.volume_parameters_with_speed()
    joined = join_parts_with_stats(cov_overall, parts_with_speed)
    print(
        joined.select(
            [
                "part_id",
                "Hatches Power",
                "Hatch Speed",
                f"cov_{COV_PLOT_SIGNAL}",
            ]
        )
    )

    clustered = clustered.join(
        cov_overall.select(["part_id", f"cov_{COV_PLOT_SIGNAL}"]),
        on="part_id",
        how="left",
    )

    sample = prepare_for_plot(clustered, target_points=80_000, method="random", seed=0)

    print("\nCreating 3D scatter plot...")
    fig_3d = scatter3d(
        sample,
        x="Demand X",
        y="Demand Y",
        z="Z",
        color=f"cov_{COV_PLOT_SIGNAL}",
        size=2,
        colorscale="Turbo",
        title=f"3D view coloured by overall CoV — {COV_PLOT_SIGNAL}",
        xaxis_title="X (mm)",
        yaxis_title="Y (mm)",
        zaxis_title="Z (mm)",
        colorbar_title="CoV",
        hover_columns=["part_id"],
    )
    fig_3d.show()

    print("\nCreating parameter contour plot...")
    fig_process_map = contour(
        joined,
        x="Hatch Speed",
        y="Hatches Power",
        z=f"cov_{COV_PLOT_SIGNAL}",
        title=f"Process map: {COV_PLOT_SIGNAL} CoV vs laser parameters",
        xaxis_title="Hatches Speed (mm/s)",
        yaxis_title="Hatches Power (W)",
        colorbar_title="CoV",
        colorscale="Turbo",
        hover_columns=["part_id"],
    )
    fig_process_map.show()


if __name__ == "__main__":
    main()
