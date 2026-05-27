"""
bar.py
"""

NAME = "Bar"
DESCRIPTION = "Bar chart over a categorical axis, e.g. CoV per part."

AXES = {
    "x": {"label": "Category axis", "default": "part_id"},
    "y": {"label": "Value axis", "default": None},
    "color": {"label": "Colour (optional)", "default": None},
}

SETTINGS = {
    "SORT_BY": {
        "type": "choice",
        "options": ["none", "x", "y"],
        "default": "y",
        "label": "Sort by",
    },
    "SORT_DESCENDING": {
        "type": "bool",
        "default": False,
        "label": "Sort descending",
    },
    "ORIENTATION": {
        "type": "choice",
        "options": ["v", "h"],
        "default": "v",
        "label": "Orientation",
    },
}


def run(df, config, axes, settings):
    from ampm.plotting import bar

    sort_by = settings.get("SORT_BY", "y")
    if sort_by == "none":
        sort_by = None

    print(f"Plotting bar chart: {axes['y']} by {axes['x']}...")
    bar(
        df,
        x=axes["x"],
        y=axes["y"],
        color=axes.get("color"),
        sort_by=sort_by,
        sort_descending=settings.get("SORT_DESCENDING", False),
        orientation=settings.get("ORIENTATION", "v"),
        title=f"{axes['y']} by {axes['x']}",
        xaxis_title=axes["x"],
        yaxis_title=axes["y"],
    ).show()

    print("Done.")
