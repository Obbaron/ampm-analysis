import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def plot_density_scatter(
    file_path,
    sheet_name=None,
    x_col="Energy Density (J/m)",
    y_col="Density AVG (g/cm^3)",
    yerr_col="σ",
    label_col="Cube ID",
    annotate=True,
):
    df = pl.read_excel(file_path, sheet_name=sheet_name)

    df = df.rename({col: col.strip() for col in df.columns})

    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    yerr = df[yerr_col].to_numpy()
    labels = df[label_col].to_numpy()

    plt.figure()
    plt.errorbar(x, y, yerr=yerr, fmt="o", color="black", ecolor="black")

    if annotate:
        for i, label in enumerate(labels):
            plt.annotate(
                str(label),
                (x[i], y[i]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    plt.xlabel(x_col)
    plt.ylabel(y_col)

    title = "Archimedes Density"
    if sheet_name:
        title += f" - {sheet_name}"
    plt.title(title)

    plt.show()


def plot_density_contour(
    file_path,
    sheet_name,
    x_col="Speed (mm/s)",
    y_col="Power (W)",
    z_col="Density AVG (g/cm^3)",
    grid_res=100,
    levels=20,
    method="cubic",
    cmap="viridis",
    show_points=True,
):
    df = pl.read_excel(file_path, sheet_name=sheet_name)

    df = df.rename({col: col.strip() for col in df.columns})

    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    z = df[z_col].to_numpy()

    xi = np.linspace(x.min(), x.max(), grid_res)
    yi = np.linspace(y.min(), y.max(), grid_res)
    xi, yi = np.meshgrid(xi, yi)

    zi = griddata((x, y), z, (xi, yi), method=method)

    plt.figure()
    contour = plt.contourf(xi, yi, zi, levels=levels, cmap=cmap)
    plt.colorbar(contour, label=z_col)

    if show_points:
        plt.scatter(x, y, color="black", s=10)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"Density Contour - {sheet_name}")

    plt.show()


plot_density_contour(
    file_path=r"C:\Users\ohp460\Documents\Code\ampm-data\Archimedes Density TiAg.xlsx",
    sheet_name="Ti-Sterling",
)

# plot_density_scatter(
#    file_path=r"C:\Users\ohp460\Documents\Code\ampm-data\Archimedes Density TiAg.xlsx",
#    sheet_name="Ti-6Ag",
# )
