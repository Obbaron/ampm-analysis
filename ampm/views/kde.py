"""
kde.py
"""

NAME = "KDE"
DESCRIPTION = "Overlaid distribution curves comparing parts on a chosen signal."

AXES = {
    "column": {"label": "Signal", "default": "MeltVIEW melt pool (mean)"},
    "group_by": {"label": "Group by", "default": "part_id"},
}

SETTINGS = {
    "N_GROUPS": {
        "type": "int",
        "default": 6,
        "min": 2,
        "max": 20,
        "label": "Number of groups (best + worst)",
    },
}


def run(df, config, axes, settings):
    from ampm.plotting import kde

    column = axes["column"]
    group_by = axes["group_by"]
    n_groups = settings.get("N_GROUPS", 6)
    n_each = n_groups // 2

    # Pick best and worst by mean of the signal
    group_means = (
        df.filter(df[group_by] != "noise")
        .group_by(group_by)
        .agg(__import__("polars").col(column).mean().alias("_mean"))
        .sort("_mean")
    )

    n_select = min(n_each, group_means.height // 2)
    best = group_means.head(n_select)[group_by].to_list()
    worst = group_means.tail(n_select)[group_by].to_list()
    groups = best + worst

    print(f"Plotting KDE for {len(groups)} groups on '{column}'...")
    kde(
        df,
        column=column,
        group_by=group_by,
        groups=groups,
        title=f"{column} distribution: best vs worst by mean",
        xaxis_title=column,
        colorscale="Turbo",
    ).show()

    print("Done.")
