"""
layer_viewer.py
"""

NAME = "Layer Viewer"
DESCRIPTION = "Per-layer slider viewer with signal dropdown."

AXES = {
    "x": {"label": "X axis", "default": "Demand X"},
    "y": {"label": "Y axis", "default": "Demand Y"},
    "color": {"label": "Colour", "default": "MeltVIEW melt pool (mean)"},
}

SETTINGS = {
    "POINTS_PER_LAYER": {
        "type": "int",
        "default": 5_000,
        "min": 100,
        "max": 50_000,
        "label": "Points per layer",
    },
}


def run(df, config, axes, settings):
    from ampm.plotting import scatter2d_layered

    points_per_layer = settings.get("POINTS_PER_LAYER", 5_000)

    print(f"Building layer viewer ({points_per_layer} pts/layer)...")
    scatter2d_layered(
        df,
        x=axes["x"],
        y=axes["y"],
        color_columns=[axes["color"]],
        layer_col="layer",
        points_per_layer=points_per_layer,
        size=4.0,
        title="Per-layer signal viewer",
        xaxis_title=axes["x"],
        yaxis_title=axes["y"],
        colorscale="Turbo",
    ).show()

    print("Done.")
