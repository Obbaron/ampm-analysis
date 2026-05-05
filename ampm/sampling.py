"""
Downsamplers for AMPM data prior to plotting.

Plotly's Scatter3d gets sluggish above ~500k points and unresponsive above ~1M.
These functions reduce a DataFrame to a target row count using one of three
strategies:

- random:  uniform random sample. Fast, simple, but uneven coverage.
- stride:  take every Nth row. Deterministic, preserves scan-time ordering.
- grid:    spatial 3D voxel binning. One representative row per occupied voxel,
           with a configurable aggregation (max / mean / median / first) on
           one or more "color" columns. Best for spatial overviews because it
           preserves coverage even where data is dense.
"""

from __future__ import annotations

from typing import Iterable, Literal

import polars as pl

AggMethod = Literal["max", "mean", "median", "first"]


def downsample_random(
    df: pl.DataFrame,
    n: int,
    seed: int | None = None,
) -> pl.DataFrame:
    """
    Uniform random sample of rows. If df already has <= n rows, returns it
    unchanged.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if df.height <= n:
        return df
    return df.sample(n=n, seed=seed, shuffle=False)


def downsample_stride(df: pl.DataFrame, n: int) -> pl.DataFrame:
    """
    Take every k-th row, where k is chosen so the result has ~n rows.
    Preserves the original ordering (useful for trajectory plots).
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if df.height <= n:
        return df
    step = max(1, df.height // n)
    return df.gather_every(step)


def downsample_grid(
    df: pl.DataFrame,
    n: int,
    x_col: str = "Demand X",
    y_col: str = "Demand Y",
    z_col: str = "Z",
    agg_columns: Iterable[str] | None = None,
    method: AggMethod = "max",
    group_by: str | None = None,
) -> pl.DataFrame:
    """
    Spatial voxel downsampling. Bins points into a 3D grid sized so that the
    expected number of occupied voxels is ~n, then aggregates each voxel into
    a single representative row.

    Parameters
    ----------
    df
        Input DataFrame with at least the three spatial columns.
    n
        Target number of output rows. When ``group_by`` is set, this is the
        target row count **per group** (e.g. per layer); otherwise it's the
        target for the whole DataFrame. Actual count may differ — voxels are
        only emitted where data exists, so a sparse cloud yields fewer points.
    x_col, y_col, z_col
        Column names for the three spatial axes. When ``group_by`` is set
        and equals ``z_col`` (or you don't have a third spatial axis), pass
        the same column for ``z_col`` and the binning still works — ``z_col``
        contributes one bin per group, which is the desired behaviour.
    agg_columns
        Columns to aggregate per voxel. If None, all numeric columns other
        than the spatial axes are aggregated. Non-numeric columns are dropped.
    method
        How to aggregate per voxel: "max", "mean", "median", or "first".
    group_by
        Optional grouping column. When set, downsampling is performed
        independently for each unique value of this column — useful for
        per-layer downsampling on multi-layer datasets. The grouping
        column is preserved in the output.

    Returns
    -------
    polars.DataFrame
        One row per occupied voxel (per group, if ``group_by`` is set).
        Spatial columns hold the voxel-centroid position (mean of source
        points in that voxel); aggregated columns hold the chosen statistic.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    for c in (x_col, y_col, z_col):
        if c not in df.columns:
            raise KeyError(f"Spatial column {c!r} not in DataFrame")
    if group_by is not None and group_by not in df.columns:
        raise KeyError(f"group_by column {group_by!r} not in DataFrame")

    if group_by is not None:
        partitions = df.partition_by(group_by, as_dict=True)
        out_frames: list[pl.DataFrame] = []
        for key, sub in partitions.items():
            group_value = key[0] if isinstance(key, tuple) else key
            if sub.height <= n:
                out_frames.append(sub)
                continue
            downsampled = downsample_grid(
                sub.drop(group_by) if group_by in sub.columns else sub,
                n,
                x_col=x_col,
                y_col=y_col,
                z_col=z_col,
                agg_columns=agg_columns,
                method=method,
                group_by=None,
            )
            downsampled = downsampled.with_columns(pl.lit(group_value).alias(group_by))
            out_frames.append(downsampled)
        if not out_frames:
            return df.head(0)
        return pl.concat(out_frames, how="vertical_relaxed")

    if df.height <= n:
        return df

    bounds = df.select(
        pl.col(x_col).min().alias("x_lo"),
        pl.col(x_col).max().alias("x_hi"),
        pl.col(y_col).min().alias("y_lo"),
        pl.col(y_col).max().alias("y_hi"),
        pl.col(z_col).min().alias("z_lo"),
        pl.col(z_col).max().alias("z_hi"),
    ).row(0, named=True)

    x_span = max(bounds["x_hi"] - bounds["x_lo"], 1e-12)
    y_span = max(bounds["y_hi"] - bounds["y_lo"], 1e-12)
    z_span = max(bounds["z_hi"] - bounds["z_lo"], 1e-12)

    DEGEN_THRESHOLD = 1e-9
    x_degen = (bounds["x_hi"] - bounds["x_lo"]) < DEGEN_THRESHOLD
    y_degen = (bounds["y_hi"] - bounds["y_lo"]) < DEGEN_THRESHOLD
    z_degen = (bounds["z_hi"] - bounds["z_lo"]) < DEGEN_THRESHOLD

    spans_for_volume = []
    if not x_degen:
        spans_for_volume.append(x_span)
    if not y_degen:
        spans_for_volume.append(y_span)
    if not z_degen:
        spans_for_volume.append(z_span)
    n_active = max(1, len(spans_for_volume))
    volume = 1.0
    for s in spans_for_volume:
        volume *= s
    k = (n / volume) ** (1.0 / n_active)

    x_bins = 1 if x_degen else max(1, int(round(x_span * k)))
    y_bins = 1 if y_degen else max(1, int(round(y_span * k)))
    z_bins = 1 if z_degen else max(1, int(round(z_span * k)))

    x_size = x_span / x_bins
    y_size = y_span / y_bins
    z_size = z_span / z_bins

    spatial = {x_col, y_col, z_col}
    if agg_columns is None:
        agg_cols = [
            c
            for c, dtype in zip(df.columns, df.dtypes)
            if c not in spatial and dtype.is_numeric()
        ]
    else:
        agg_cols = [c for c in agg_columns if c not in spatial]
        missing = [c for c in agg_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Aggregation column(s) not in DataFrame: {missing}")

    def agg_expr(col: str) -> pl.Expr:
        c = pl.col(col)
        if method == "max":
            return c.max().alias(col)
        if method == "mean":
            return c.mean().alias(col)
        if method == "median":
            return c.median().alias(col)
        if method == "first":
            return c.first().alias(col)
        raise ValueError(f"Unknown method {method!r}")

    binned = df.with_columns(
        ((pl.col(x_col) - bounds["x_lo"]) / x_size)
        .floor()
        .cast(pl.Int32)
        .alias("__xi"),
        ((pl.col(y_col) - bounds["y_lo"]) / y_size)
        .floor()
        .cast(pl.Int32)
        .alias("__yi"),
        ((pl.col(z_col) - bounds["z_lo"]) / z_size)
        .floor()
        .cast(pl.Int32)
        .alias("__zi"),
    )

    aggregated = (
        binned.group_by(["__xi", "__yi", "__zi"])
        .agg(
            pl.col(x_col).mean().alias(x_col),
            pl.col(y_col).mean().alias(y_col),
            pl.col(z_col).mean().alias(z_col),
            *[agg_expr(c) for c in agg_cols],
        )
        .drop("__xi", "__yi", "__zi")
    )

    return aggregated


def prepare_for_plot(
    df: pl.DataFrame,
    target_points: int = 100_000,
    method: Literal["random", "stride", "grid"] = "random",
    *,
    seed: int | None = None,
    # grid-only kwargs:
    x_col: str = "Demand X",
    y_col: str = "Demand Y",
    z_col: str = "Z",
    agg_columns: Iterable[str] | None = None,
    grid_method: AggMethod = "max",
) -> pl.DataFrame:
    """
    Reduce a DataFrame to ~target_points rows using the chosen strategy.

    Parameters
    ----------
    df
        Input DataFrame.
    target_points
        Approximate row count of the output.
    method
        - "random" (default): uniform random sample.
        - "stride": every Nth row.
        - "grid": 3D voxel binning over (x_col, y_col, z_col).
    seed
        Random seed for reproducibility (random method only).
    x_col, y_col, z_col, agg_columns, grid_method
        Forwarded to ``downsample_grid`` when method="grid".
    """
    if method == "random":
        return downsample_random(df, target_points, seed=seed)
    if method == "stride":
        return downsample_stride(df, target_points)
    if method == "grid":
        return downsample_grid(
            df,
            target_points,
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            agg_columns=agg_columns,
            method=grid_method,
        )
    raise ValueError(f"Unknown method {method!r}")
