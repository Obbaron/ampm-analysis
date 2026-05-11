import re

import polars as pl

from ampm.plotting import scatter2d

FILEPATH = (
    r"C:\Users\ohp460\Documents\Code\ampm-analysis\etc\Archimedes Density TiAg.xlsx"
)
SHEET_NAME = "Ti-6Ag"

df = pl.read_excel(FILEPATH, sheet_name=SHEET_NAME, engine="calamine")


# Match columns by base name, ignoring any units in parentheses
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

display_cols = (
    [df.columns[0], speed_col, power_col]
    + density_cols
    + ["Density AVG (calc)", "Density σ (calc)"]
)
with pl.Config(tbl_cols=20, tbl_width_chars=150, tbl_rows=30, float_precision=6):
    print(f"Sheet: {SHEET_NAME}  |  Density columns: {density_cols}")
    print(df.select(display_cols))

fig = scatter2d(df=df, x="Speed (mm/s)", y="Density AVG (calc)")
fig.show()
