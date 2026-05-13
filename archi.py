import re

import plotly.graph_objects as go
import polars as pl

FILEPATH = (
    r"C:\Users\ohp460\Documents\Code\ampm-analysis\etc\Archimedes Density TiAg.xlsx"
)
SHEET_NAME = "Ti-Sterling"

df = pl.read_excel(FILEPATH, sheet_name=SHEET_NAME, engine="calamine")


def find_col(df, name):
    return next(
        (c for c in df.columns if re.match(rf"{name}\s*(\(.*\))?$", c, re.IGNORECASE)),
        None,
    )


density_cols = [c for c in df.columns if re.match(r"Density\s*\(?\d\)?$", c)]
speed_col = find_col(df, "Speed")
power_col = find_col(df, "Power")

df = (
    df.with_columns(pl.concat_list(density_cols).alias("_densities"))
    .with_columns(
        pl.col("_densities")
        .list.eval(pl.element().drop_nulls().mean())
        .list.first()
        .alias("Density AVG (calc)"),
        pl.col("_densities")
        .list.eval(pl.element().drop_nulls().std())
        .list.first()
        .alias("Density σ (calc)"),
    )
    .drop("_densities")
)

speed = df[speed_col].to_list()
power = df[power_col].to_list()
density_avg = df["Density AVG (calc)"].to_list()

fig = go.Figure()

fig.add_trace(
    go.Contour(
        x=speed,
        y=power,
        z=density_avg,
        colorscale="Turbo",
        contours=dict(coloring="heatmap", showlabels=True),
        colorbar=dict(title="Density (g/cm³)"),
        hovertemplate="Speed: %{x} mm/s<br>Power: %{y} W<br>Density: %{z:.4f} g/cm³<extra></extra>",
    )
)

fig.add_trace(
    go.Scatter(
        x=speed,
        y=power,
        mode="markers",
        marker=dict(size=7, color="white", line=dict(width=1.5, color="black")),
        hovertemplate="Speed: %{x} mm/s<br>Power: %{y} W<extra></extra>",
        showlegend=False,
    )
)

fig.update_layout(
    title=f"{SHEET_NAME} — Archimedes density",
    xaxis_title=speed_col,
    yaxis_title=power_col,
    width=800,
    height=500,
)

fig.write_html(r"C:\Users\ohp460\Documents\Code\ampm-analysis\JR314_layers.html")
fig.show()
