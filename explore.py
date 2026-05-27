"""
explore.py: load, mask, assign each row to its nearest part,
optionally correct melt-pool signal, analyze process stability via
CoV, and visualize chosen data columns.
"""

import sys
from pathlib import Path

import polars as pl
from PyQt6.QtWidgets import QApplication, QFileDialog

from ampm import DataStore
from ampm.config import create_or_load_config
from ampm.correction import MeltPoolCorrection
from ampm.mask_cache import mask_or_load
from ampm.masking import apply_mask, build_mask
from ampm.parts import (
    QuantAMParts,
    assign_nearest_part,
    join_parts_with_stats,
)
from ampm.plotting import bar, contour, kde, scatter2d, scatter2d_layered, scatter3d
from ampm.sampling import prepare_for_plot
from ampm.stats import compute_cov

CORRECT_MELTPOOL = False


def main() -> None:
    if len(sys.argv) >= 2:
        build_dir = sys.argv[1]
    else:
        QApplication(sys.argv)
        build_dir = QFileDialog.getExistingDirectory(None, "Select Build Directory")
        if not build_dir:
            sys.exit("No directory selected.")
    config = create_or_load_config(build_dir)

    SOURCE = config["SOURCE"]
    STL = config["STL"]
    PARTS_CSV = config["PARTS_CSV"]
    LAYER_THICKNESS = config["LAYER_THICKNESS"]
    MASK_CACHE = config["MASK_CACHE"]
    MASK_KEEP_CACHE = config["MASK_KEEP_CACHE"]
    MAX_DISTANCE_MM = config["MAX_DISTANCE_MM"]
    SIGNALS = config["SIGNALS"]
    COV_PLOT_SIGNAL = (
        "MeltVIEW melt pool (mean) corrected" if CORRECT_MELTPOOL else SIGNALS[0]
    )

    store = DataStore(SOURCE, layer_thickness=LAYER_THICKNESS)

    df = store.query()
    print(f"Full slice: {df.height:,} rows")

    mask_params = {
        "layers": (min(store.layers), max(store.layers)),
        "stl": str(STL),
        "buffer_mm": 0.0,
        "layer_thickness": LAYER_THICKNESS,
    }

    def masking_wrapper(d: pl.DataFrame) -> pl.DataFrame:  # wrapper function
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
        mask_fn=masking_wrapper,
        params=mask_params,
        strict=True,
    )
    survival = df_masked.height / df.height
    print(f"After mask: {df_masked.height:,} rows ({survival:.1%} kept)")
    del df

    quantam = QuantAMParts.from_path(PARTS_CSV)
    parts_table = quantam.parent_parts()
    print(f"\nLoaded {parts_table.height} parts from {Path(PARTS_CSV).name}")

    print("\nAssigning each row to its nearest part...")
    clustered = assign_nearest_part(
        df_masked,
        parts_table,
        max_distance_mm=MAX_DISTANCE_MM,
        noise_label="noise",
    )
    del df_masked

    if CORRECT_MELTPOOL:
        print("\nApplying MAIN machine meltpool XY-bias correction...")
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

    print("\nDone.")


if __name__ == "__main__":
    main()
