"""
contour.py
"""

NAME = "Contour"
DESCRIPTION = "Filled contour over a (x, y) grid, e.g. CoV vs speed and power."

AXES = {
    "x": {"label": "X axis", "default": "Hatch Speed"},
    "y": {"label": "Y axis", "default": "Hatches Power"},
    "z": {"label": "Z (colour)", "default": None},
}

SETTINGS = {
    "SHOW_POINTS": {
        "type": "bool",
        "default": True,
        "label": "Show data points",
    },
}


def run(df, config, axes, settings):
    from ampm.plotting import contour

    show_points = settings.get("SHOW_POINTS", True)

    print(f"Plotting contour: {axes['z']} vs ({axes['x']}, {axes['y']})...")
    contour(
        df,
        x=axes["x"],
        y=axes["y"],
        z=axes["z"],
        title=f"{axes['z']} vs ({axes['x']}, {axes['y']})",
        xaxis_title=axes["x"],
        yaxis_title=axes["y"],
        colorbar_title=axes["z"],
        colorscale="Turbo",
        show_points=show_points,
        hover_columns=["part_id"] if "part_id" in df.columns else [],
    ).show()

    print("Done.")
