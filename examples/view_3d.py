"""
view_3d.py

Loads cached AMPM data, applies the cached mask, assigns each row to
its nearest part, then opens a single 3D scatter plot whose colour
encodes a user-chosen signal. A dropdown menu at the top of the figure
lets you swap between several signals without re-rendering.

Hover on any point shows the active signal value, the part_id, and the
laser parameters (Hatches Power, Hatch Speed).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import plotly.graph_objects as go
import polars as pl

from ampm import DataStore
from ampm.mask_cache import mask_or_load
from ampm.masking import apply_mask, build_mask
from ampm.parts import QuantAMParts, assign_nearest_part
from ampm.sampling import prepare_for_plot
from config import (
    LAYER_THICKNESS,
    MASK_CACHE,
    MASK_KEEP_CACHE,
    PARTS_CSV,
    SOURCE,
    STL,
)

MAX_DISTANCE_MM = None
TARGET_POINTS = 50_000
POINT_SIZE = 2

SIGNALS = [
    "MeltVIEW melt pool (mean)",
    "MeltVIEW plasma (mean)",
    "Laser back reflection (mean)",
    "Laser output power (mean)",
]

COLORSCALE = "Turbo"


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
    print(f"\nLoaded {parts_table.height} parts from {Path(PARTS_CSV).name}")

    print("Assigning each row to its nearest part...")
    assigned = assign_nearest_part(
        df_masked,
        parts_table,
        max_distance_mm=MAX_DISTANCE_MM,
        noise_label="noise",
    )
    del df_masked

    print(f"\nSampling {TARGET_POINTS:,} points for the 3D plot...")
    sample = prepare_for_plot(
        assigned, target_points=TARGET_POINTS, method="random", seed=0
    )

    parts_with_speed = quantam.volume_parameters_with_speed()
    sample = sample.join(
        parts_with_speed.select(
            [pl.col("Part ID").alias("part_id"), "Hatches Power", "Hatch Speed"]
        ),
        on="part_id",
        how="left",
    )

    x = sample["Demand X"].to_numpy()
    y = sample["Demand Y"].to_numpy()
    z = sample["Z"].to_numpy()

    color_arrays = {sig: sample[sig].to_numpy() for sig in SIGNALS}
    color_ranges = {
        sig: (float(np.nanmin(color_arrays[sig])), float(np.nanmax(color_arrays[sig])))
        for sig in SIGNALS
    }

    part_id_arr = sample["part_id"].to_numpy()
    power_arr = sample["Hatches Power"].to_numpy()
    speed_arr = sample["Hatch Speed"].to_numpy()

    def build_customdata(sig: str) -> np.ndarray:
        return np.column_stack([color_arrays[sig], part_id_arr, power_arr, speed_arr])

    def build_hovertemplate(sig: str) -> str:
        return (
            f"X: %{{x:.2f}}<br>"
            f"Y: %{{y:.2f}}<br>"
            f"Z: %{{z:.2f}}<br>"
            f"{sig}: %{{customdata[0]}}<br>"
            f"part_id: %{{customdata[1]}}<br>"
            f"Hatches Power: %{{customdata[2]}}<br>"
            f"Hatch Speed: %{{customdata[3]}}"
            "<extra></extra>"
        )

    initial_sig = SIGNALS[0]
    cmin0, cmax0 = color_ranges[initial_sig]

    trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(
            size=POINT_SIZE,
            color=color_arrays[initial_sig],
            colorscale=COLORSCALE,
            cmin=cmin0,
            cmax=cmax0,
            showscale=True,
            colorbar=dict(title=initial_sig),
        ),
        customdata=build_customdata(initial_sig),
        hovertemplate=build_hovertemplate(initial_sig),
    )

    buttons = []
    for sig in SIGNALS:
        cmin, cmax = color_ranges[sig]
        buttons.append(
            dict(
                label=sig,
                method="restyle",
                args=[
                    {
                        "marker.color": [color_arrays[sig]],
                        "marker.cmin": [cmin],
                        "marker.cmax": [cmax],
                        "marker.colorbar.title.text": [sig],
                        "customdata": [build_customdata(sig)],
                        "hovertemplate": [build_hovertemplate(sig)],
                    }
                ],
            )
        )

    fig = go.Figure(data=[trace])
    fig.update_layout(
        title="3D Signal Viewer",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
        ),
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                x=1.02,
                y=1.0,
                xanchor="left",
                yanchor="top",
                showactive=True,
                pad=dict(t=4, r=4),
            )
        ],
    )

    print("\nPlotting 3D view...")
    fig.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
