"""
k_distance.py
"""

NAME = "K-Distance Curve"
DESCRIPTION = "Compute the k-distance curve to find the right EPS_XY for DBSCAN."

AXES = {}

SETTINGS = {
    "EPS_XY": {
        "type": "float",
        "default": 0.3,
        "label": "EPS_XY (mm)",
    },
    "EPS_Z": {
        "type": "float",
        "default": 0.06,
        "label": "EPS_Z (mm)",
    },
    "K": {
        "type": "int",
        "default": 10,
        "min": 1,
        "max": 100,
        "label": "K (neighbors)",
    },
    "SAMPLE_SIZE": {
        "type": "int",
        "default": 1_000_000,
        "min": 1000,
        "max": 10_000_000,
        "label": "Sample size",
    },
}


def run(df, config, axes, settings):
    from ampm.clustering import k_distance_curve
    from ampm.plotting import scatter2d

    eps_xy = settings.get("EPS_XY", 0.3)
    eps_z = settings.get("EPS_Z", 0.06)
    k = settings.get("K", 10)
    sample_size = settings.get("SAMPLE_SIZE", 1_000_000)

    print(f"Computing k-distance curve (k={k}, sample={sample_size:,})...")
    curve = k_distance_curve(
        df,
        k=k,
        sample_size=sample_size,
        mode="3d",
        eps_xy=eps_xy,
        eps_z=eps_z,
        seed=0,
    )

    q50 = curve["k-distance (mm)"].quantile(0.50)
    q90 = curve["k-distance (mm)"].quantile(0.90)
    q95 = curve["k-distance (mm)"].quantile(0.95)
    q99 = curve["k-distance (mm)"].quantile(0.99)
    print(f"\nk-distance quantiles (sample of {curve.height:,} points):")
    print(f"  50th percentile: {q50:.3f} mm")
    print(f"  90th percentile: {q90:.3f} mm")
    print(f"  95th percentile: {q95:.3f} mm")
    print(f"  99th percentile: {q99:.3f} mm")
    print("\nThe elbow usually sits between the 90th and 99th percentile.")

    print("\nPlotting k-distance curve...")
    scatter2d(
        curve,
        x="Rank",
        y="k-distance (mm)",
        equal_aspect=False,
        size=4,
        title=f"k-distance curve (k={k}). Look for the elbow.",
        xaxis_title="Rank (sorted)",
        yaxis_title=f"Distance to {k}-th neighbor (mm)",
    ).show()

    print("Done.")
