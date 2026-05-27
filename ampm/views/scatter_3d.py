"""
scatter_3d.py
"""

NAME = "3D Scatter"
DESCRIPTION = "Interactive 3D point cloud colored by any column."

AXES = {
    "x": {"label": "X axis", "default": "Demand X"},
    "y": {"label": "Y axis", "default": "Demand Y"},
    "z": {"label": "Z axis", "default": "Z"},
    "color": {"label": "Colour", "default": None},
}

SETTINGS = {
    "SAMPLE_SIZE": {
        "type": "int",
        "default": 80_000,
        "min": 1000,
        "max": 1_000_000,
        "label": "Sample size",
    },
    "POINT_SIZE": {
        "type": "int",
        "default": 2,
        "min": 1,
        "max": 10,
        "label": "Point size",
    },
}


def run(df, config, axes, settings):
    from ampm.plotting import scatter3d
    from ampm.sampling import prepare_for_plot

    sample_size = settings.get("SAMPLE_SIZE", 80_000)
    point_size = settings.get("POINT_SIZE", 2)

    print(f"Sampling {sample_size:,} points...")
    sample = prepare_for_plot(df, target_points=sample_size, method="random", seed=0)

    print("Rendering 3D scatter...")
    scatter3d(
        sample,
        x=axes["x"],
        y=axes["y"],
        z=axes["z"],
        color=axes.get("color"),
        size=point_size,
        colorscale="Turbo",
        title=f"3D Scatter — colour: {axes.get('color', 'none')}",
        xaxis_title=axes["x"],
        yaxis_title=axes["y"],
        zaxis_title=axes["z"],
        colorbar_title=axes.get("color", ""),
        hover_columns=["part_id"],
    ).show()

    print("Done.")
