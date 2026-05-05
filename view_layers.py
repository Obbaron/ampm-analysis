"""
view_layers.py

Loads cached AMPM data via DataStore (handles bracketed paths and the
parquet layout), applies the cached mask, and hands the result to the
scatter2d_layered function, which random-downsamples each layer to the
target point count internally.

Run AFTER explore.py has populated the mask cache.
"""

from ampm import DataStore
from ampm.mask_cache import load_mask_keep
from ampm.plotting import scatter2d_layered

from config import SOURCE, MASK_KEEP_CACHE, LAYER_THICKNESS

LAYERED_SIGNALS = [
    "MeltVIEW melt pool (mean)",
    "MeltVIEW plasma (mean)",
]
POINTS_PER_LAYER = 5_000


def main() -> None:
    store = DataStore(SOURCE, layer_thickness=LAYER_THICKNESS)
    print(store)

    needed_cols = ["Demand X", "Demand Y", "Start time", *LAYERED_SIGNALS]
    print(f"Loading columns {needed_cols}...")
    df = store.query(columns=needed_cols)
    print(f"  loaded {df.height:,} rows")

    print("Applying cached mask...")
    df_masked = load_mask_keep(df, MASK_KEEP_CACHE, strict=False, verbose=True)
    print(f"  {df_masked.height:,} rows after mask")
    del df

    print(
        f"Building interactive viewer ({len(LAYERED_SIGNALS)} signals, "
        f"{POINTS_PER_LAYER}/layer)..."
    )
    fig = scatter2d_layered(
        df_masked,
        x="Demand X",
        y="Demand Y",
        color_columns=LAYERED_SIGNALS,
        layer_col="layer",
        points_per_layer=POINTS_PER_LAYER,
        size=4.0,
        title="Per-layer signal viewer",
        xaxis_title="X (mm)",
        yaxis_title="Y (mm)",
        colorscale="Turbo",
    )
    fig.show()


if __name__ == "__main__":
    main()
