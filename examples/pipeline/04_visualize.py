"""
04_visualize.py

Generates the three standard visualizations for AMPM analysis:

1. 3D scatter coloured by per-part overall CoV, showing which parts
   were noisy across the build.
2. KDE comparison of the 3 most stable vs 3 least stable parts on the
   melt-pool signal to show distribution shape (not just summary CoV).
3. Parametric contour plot of CoV vs (Hatch Speed, Hatches Power).
    Skipped on builds where every part has the same laser parameters.

All plots open in a browser via ``fig.show()``.
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
from ampm.plotting import contour, kde, scatter3d
from ampm.sampling import prepare_for_plot
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

USE_DIRECT_ASSIGNMENT = True
MAX_DISTANCE_MM = None  # direct-assignment

# DBSCAN parameters
EPS_XY = 0.3
EPS_Z = 0.06
MIN_SAMPLES = 10
LAYERS_PER_CHUNK = 11
OVERLAP_LAYERS = None

SIGNAL = "MeltVIEW melt pool (mean)"
SIGNALS_FOR_COV = [SIGNAL, "Laser output power (mean)"]

TARGET_POINTS_3D = 80_000  # 3D plots downsample to this many points


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

    print("\nComputing overall CoV...")
    cov_overall = compute_cov(
        assigned,
        SIGNALS_FOR_COV,
        group_by="part_id",
        mode="overall",
        noise_label="noise",
    )

    parts_with_speed = quantam.volume_parameters_with_speed()
    joined = join_parts_with_stats(cov_overall, parts_with_speed)

    print(f"\nSampling {TARGET_POINTS_3D:,} points for the 3D plots...")
    sample = prepare_for_plot(
        assigned, target_points=TARGET_POINTS_3D, method="random", seed=0
    )
    sample = sample.join(
        joined.select(["part_id", "Hatches Power", "Hatch Speed", f"cov_{SIGNAL}"]),
        on="part_id",
        how="left",
    )

    print("\nPlot 1/3: 3D scatter coloured by per-part CoV...")
    fig_cov_3d = scatter3d(
        sample,
        x="Demand X",
        y="Demand Y",
        z="Z",
        color=f"cov_{SIGNAL}",
        size=2,
        colorscale="Turbo",
        title=f"Parts coloured by overall CoV of '{SIGNAL}'",
        xaxis_title="X (mm)",
        yaxis_title="Y (mm)",
        zaxis_title="Z (mm)",
        colorbar_title="CoV",
        hover_columns=["part_id", "Hatches Power", "Hatch Speed"],
    )
    fig_cov_3d.show()

    print("\nPlot 2/3: KDE comparison of best vs worst parts...")
    ranked = cov_overall.sort(f"cov_{SIGNAL}")
    n_select = min(3, ranked.height // 2)
    best = ranked.head(n_select)["part_id"].to_list()
    worst = ranked.tail(n_select)["part_id"].to_list()
    fig_dist = kde(
        assigned,
        column=SIGNAL,
        group_by="part_id",
        groups=best + worst,
        title=f"{SIGNAL} distribution: most stable vs least stable parts",
        xaxis_title=SIGNAL,
        colorscale="Turbo",
    )
    fig_dist.show()

    has_power_variation = joined["Hatches Power"].n_unique() > 1
    has_speed_variation = joined["Hatch Speed"].n_unique() > 1
    if has_power_variation or has_speed_variation:
        print("\nPlot 3/3: parametric process map...")
        fig_process_map = contour(
            joined,
            x="Hatch Speed",
            y="Hatches Power",
            z=f"cov_{SIGNAL}",
            title=f"Process map: CoV of '{SIGNAL}' vs laser parameters",
            xaxis_title="Hatch Speed (mm/s)",
            yaxis_title="Hatches Power (W)",
            colorbar_title="CoV",
            colorscale="Turbo",
            hover_columns=["part_id"],
        )
        fig_process_map.show()
    else:
        print(
            "\nPlot 3/3: skipped. All parts have identical laser "
            "parameters, process map would collapse to a single point."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
