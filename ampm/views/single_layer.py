"""
single_layer.py
"""

NAME = "Single Layer"
DESCRIPTION = "Static 2D scatter of one layer, colored by a selected signal."

AXES = {
    "x": {"label": "X axis", "default": "Demand X"},
    "y": {"label": "Y axis", "default": "Demand Y"},
    "color": {"label": "Signal (color)", "default": "MeltVIEW melt pool (mean)"},
}

SETTINGS = {
    "LAYER": {
        "type": "int",
        "default": 1,
        "min": 0,
        "max": 100_000,
        "label": "Layer",
    },
    "MAX_POINTS": {
        "type": "int",
        "default": 100_000,
        "min": 100,
        "max": 1_000_000,
        "label": "Max points",
    },
}


def run(df, config, axes, settings):
    import polars as pl

    from ampm.plotting import scatter2d

    x = axes.get("x")
    y = axes.get("y")
    color = axes.get("color")

    if not x or not y:
        raise ValueError("Both X and Y axes must be set.")

    layer = settings.get("LAYER", 1)
    max_points = settings.get("MAX_POINTS", 100_000)

    sub = df.filter(pl.col("layer") == layer)
    if sub.is_empty():
        layers = df["layer"].unique().sort()
        raise ValueError(
            f"No rows for layer {layer}. "
            f"Layers in data range from {layers.min()} to {layers.max()}."
        )

    n = sub.height
    if n > max_points:
        sub = sub.sample(n=max_points, seed=0)
        print(f"Layer {layer}: sampled {max_points:,}/{n:,} points.")
    else:
        print(f"Layer {layer}: {n:,} points.")

    scatter2d(
        sub,
        x=x,
        y=y,
        color=color,
        size=6.0,
        title=f"Layer {layer}",
        xaxis_title=x,
        yaxis_title=y,
        colorscale="Turbo",
        equal_aspect=True,
    ).show()

    print("Done.")
