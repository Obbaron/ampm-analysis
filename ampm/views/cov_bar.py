"""
cov_bar.py
"""

NAME = "CoV Bar Chart"
DESCRIPTION = "Compute coefficient of variation and display per part as a bar chart."

AXES = {
    "signal": {"label": "Signal", "default": "MeltVIEW melt pool (mean)"},
}

SETTINGS = {
    "COV_MODE": {
        "type": "choice",
        "options": ["overall", "per_layer_mean", "across_layers"],
        "default": "overall",
        "label": "CoV mode",
    },
    "SORT_DESCENDING": {
        "type": "bool",
        "default": False,
        "label": "Sort descending",
    },
}


def run(df, config, axes, settings):
    from ampm.plotting import bar
    from ampm.stats import compute_cov

    signal = axes["signal"]
    mode = settings.get("COV_MODE", "overall")

    print(f"Computing CoV (mode: {mode}) for '{signal}'...")
    cov = compute_cov(
        df,
        [signal],
        group_by="part_id",
        mode=mode,
        noise_label="noise",
    )
    print(cov)

    cov_col = f"cov_{signal}"

    print("\nPlotting CoV bar chart...")
    bar(
        cov,
        x="part_id",
        y=cov_col,
        sort_by="y",
        sort_descending=settings.get("SORT_DESCENDING", False),
        title=f"CoV ({mode}) of '{signal}' per part",
        xaxis_title="Part",
        yaxis_title=f"CoV ({mode})",
    ).show()

    print("Done.")
