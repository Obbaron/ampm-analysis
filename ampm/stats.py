"""
Per-group statistics for AMPM data — currently just coefficient of variation.

The coefficient of variation (CoV) is std / |mean|. It's dimensionless, so
unlike raw std it lets you compare the *stability* of signals across parts
that have very different absolute levels (different laser parameters,
different geometries).

Three CoV modes capture different physical questions:

  overall          : single CoV computed across all rows in each group.
                     Captures total variability — intra-layer noise plus
                     layer-to-layer drift plus outliers.

  per_layer_mean   : compute CoV within each (group, layer), then average
                     across layers. Captures intra-layer stability,
                     filtering out layer-to-layer drift.

  across_layers    : compute the mean within each (group, layer), then take
                     CoV across the per-layer means. Captures layer-to-layer
                     drift specifically — useful for finding parts where the
                     process slowly walks off-target.
"""

from __future__ import annotations

from typing import Literal, Sequence

import polars as pl

CovMode = Literal["overall", "per_layer_mean", "across_layers"]


def compute_cov(
    df: pl.DataFrame,
    columns: Sequence[str],
    *,
    group_by: str = "part_id",
    layer_col: str = "layer",
    mode: CovMode = "overall",
    drop_noise: bool = True,
    noise_label: str | None = None,
    eps: float = 1e-12,
) -> pl.DataFrame:
    """
    Compute coefficient of variation per group, for each requested column.

    Parameters
    ----------
    df
        Input DataFrame with at least ``group_by``, all columns in
        ``columns``, and (for non-overall modes) ``layer_col``.
    columns
        Numeric column names to compute CoV for. Each produces a
        ``cov_<column>`` column in the output.
    group_by
        Column to group by. Default ``"part_id"``.
    layer_col
        Layer column name; only used for ``per_layer_mean`` and
        ``across_layers``. Default ``"layer"``.
    mode
        ``"overall"`` (default), ``"per_layer_mean"``, or ``"across_layers"``.
        See module docstring for what each captures.
    drop_noise
        If True, exclude rows whose ``group_by`` value equals ``noise_label``
        (default ``None``). The noise group's CoV isn't physically meaningful.
    noise_label
        The value of ``group_by`` that means "no group". Default ``None``,
        which drops null entries when ``drop_noise`` is True. Set to
        e.g. ``"noise"`` if you used ``apply_part_id_map(noise_label="noise")``.
    eps
        Floor on |mean| in the denominator. Groups whose mean magnitude is
        below ``eps`` get a null CoV instead of a division blow-up.

    Returns
    -------
    DataFrame with columns ``[group_by, n_rows, cov_<col1>, cov_<col2>, ...]``
    sorted by ``group_by``. Per-mode notes:

    - ``overall``: ``n_rows`` is total rows in the group.
    - ``per_layer_mean``: ``n_rows`` is the number of layers averaged.
    - ``across_layers``: ``n_rows`` is the number of layers contributing means.
    """
    if mode not in ("overall", "per_layer_mean", "across_layers"):
        raise ValueError(f"unknown mode {mode!r}")

    if group_by not in df.columns:
        raise KeyError(f"group_by column {group_by!r} not in DataFrame")

    cols = list(columns)
    if not cols:
        raise ValueError("columns must not be empty")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Column(s) not in DataFrame: {missing}")

    if mode in ("per_layer_mean", "across_layers") and layer_col not in df.columns:
        raise KeyError(f"mode={mode!r} requires layer_col {layer_col!r} in DataFrame")

    work = df
    if drop_noise:
        if noise_label is None:
            work = work.filter(pl.col(group_by).is_not_null())
        else:
            work = work.filter(pl.col(group_by) != noise_label)

    if work.is_empty():
        return pl.DataFrame(
            {group_by: [], "n_rows": [], **{f"cov_{c}": [] for c in cols}}
        )

    if mode == "overall":
        return _compute_overall(work, cols, group_by, eps)
    if mode == "per_layer_mean":
        return _compute_per_layer_mean(work, cols, group_by, layer_col, eps)
    return _compute_across_layers(work, cols, group_by, layer_col, eps)


def _cov_expr(col: str, eps: float) -> pl.Expr:
    """std / |mean|, with null when |mean| < eps to avoid blow-ups."""
    mean = pl.col(col).mean()
    return (
        pl.when(mean.abs() < eps)
        .then(None)
        .otherwise(pl.col(col).std() / mean.abs())
        .alias(f"cov_{col}")
    )


def _compute_overall(
    df: pl.DataFrame,
    cols: list[str],
    group_by: str,
    eps: float,
) -> pl.DataFrame:
    aggs = [pl.len().alias("n_rows")] + [_cov_expr(c, eps) for c in cols]
    return df.group_by(group_by).agg(aggs).sort(group_by)


def _compute_per_layer_mean(
    df: pl.DataFrame,
    cols: list[str],
    group_by: str,
    layer_col: str,
    eps: float,
) -> pl.DataFrame:
    # Step 1: CoV within each (group, layer).
    per_layer = df.group_by([group_by, layer_col]).agg(
        [_cov_expr(c, eps) for c in cols]
    )
    # Step 2: average those layer-CoVs per group, ignoring nulls (zero-mean layers).
    final = (
        per_layer.group_by(group_by)
        .agg(
            [pl.len().alias("n_rows")]
            + [pl.col(f"cov_{c}").mean().alias(f"cov_{c}") for c in cols]
        )
        .sort(group_by)
    )
    return final


def _compute_across_layers(
    df: pl.DataFrame,
    cols: list[str],
    group_by: str,
    layer_col: str,
    eps: float,
) -> pl.DataFrame:
    # Step 1: per-layer mean of each column.
    per_layer_mean = df.group_by([group_by, layer_col]).agg(
        [pl.col(c).mean().alias(c) for c in cols]
    )
    # Step 2: CoV across those per-layer means within each group.
    final = (
        per_layer_mean.group_by(group_by)
        .agg([pl.len().alias("n_rows")] + [_cov_expr(c, eps) for c in cols])
        .sort(group_by)
    )
    return final
