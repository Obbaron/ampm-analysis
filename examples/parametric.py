"""
parametric.py

Loads cached AMPM data, applies the cached mask, assigns each row to
its nearest part by 2D Euclidean distance, then produces three plots:

1. 3D scatter coloured by per-part overall CoV of the chosen signal.
   Hover shows part_id, Power, Speed for context.
2. KDE comparison of the most-stable vs least-stable parts on the
   chosen signal.
3. Parametric contour plot of CoV vs (Speed, Power).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl

from ampm import DataStore
from ampm.mask_cache import mask_or_load
from ampm.masking import apply_mask, build_mask
from ampm.parts import (
    QuantAMParts,
    assign_nearest_part,
    join_parts_with_stats,
)
from ampm.plotting import contour, kde, scatter3d
from ampm.sampling import prepare_for_plot
from ampm.stats import compute_cov
from config import (
    LAYER_THICKNESS,
    MASK_CACHE,
    MASK_KEEP_CACHE,
    PARTS_CSV,
    SOURCE,
    STL,
)

MAX_DISTANCE_MM = None
SIGNAL = "MeltVIEW melt pool (mean)"
TARGET_POINTS_3D = 80_000
N_BEST_WORST = 3


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

    print("\nAssigning each row to its nearest part...")
    assigned = assign_nearest_part(
        df_masked,
        parts_table,
        max_distance_mm=MAX_DISTANCE_MM,
        noise_label="noise",
    )
    del df_masked

    print("\nComputing overall CoV...")
    cov_overall = compute_cov(
        assigned,
        [SIGNAL],
        group_by="part_id",
        mode="overall",
        noise_label="noise",
    )
    print(cov_overall)

    print("\nJoining CoV with laser parameters...")
    parts_with_speed = quantam.volume_parameters_with_speed()
    joined = join_parts_with_stats(cov_overall, parts_with_speed)
    print(joined.select(["part_id", "Hatches Power", "Hatch Speed", f"cov_{SIGNAL}"]))

    print(f"\nSampling {TARGET_POINTS_3D:,} points for the 3D plot...")
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

    print(
        f"\nPlot 2/3: KDE comparison of best {N_BEST_WORST} vs worst "
        f"{N_BEST_WORST} parts..."
    )
    ranked = cov_overall.sort(f"cov_{SIGNAL}")
    n_select = min(N_BEST_WORST, ranked.height // 2)
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

    print("\nDone.")


if __name__ == "__main__":
    main()
