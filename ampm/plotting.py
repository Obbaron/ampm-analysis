"""
Dumb plotters for AMPM data.

These functions take a polars DataFrame plus column names and produce a Plotly
Figure. They do NOT downsample, filter, or transform — pass in pre-downsampled
data (see ampm.sampling) before calling them.

PERFORMANCE NOTE
----------------
Plotly's Scatter3d uses Three.js, not WebGL.  Practical guidance:
  -  ~50k pts:   smooth on most laptops
  - ~100k pts:   comfortable for exploration (recommended ceiling)
  - ~200k pts:   noticeable lag when rotating
  - ~500k pts:   browser may freeze
Stay <= 100k unless you really need more.

scatter2d uses Scattergl (WebGL), which handles ~1M points without trouble.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
import plotly.graph_objects as go
import polars as pl


def _check_columns(df: pl.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Column(s) not in DataFrame: {missing}.\n"
            f"Available columns: {df.columns}"
        )


def _resolve_size(df: pl.DataFrame, size: float | str) -> float | list[float]:
    """If size is a column name, return its values; otherwise return the constant."""
    if isinstance(size, str):
        _check_columns(df, [size])
        return df[size].to_list()
    return float(size)


def _resolve_color(
    df: pl.DataFrame,
    color_col: str,
) -> tuple[list, dict | None]:
    """
    Return ``(values, colorbar_overrides)`` to plug into a Plotly marker dict.

    For numeric columns, ``values`` is the raw column. ``colorbar_overrides``
    is None.

    For string / categorical columns, ``values`` is an integer code per row
    (0..K-1, dense rank). ``colorbar_overrides`` is a dict that lays the
    colorbar out as a discrete legend showing the original string labels at
    the integer tick positions — the user sees the labels they wrote, not
    integer codes.
    """
    series = df[color_col]
    if series.dtype.is_numeric():
        return series.to_list(), None

    uniques = series.unique().sort()
    label_to_code = {label: i for i, label in enumerate(uniques.to_list())}
    codes = series.replace_strict(label_to_code, return_dtype=pl.Int32)

    n = len(uniques)
    overrides = {
        "tickmode": "array",
        "tickvals": list(range(n)),
        "ticktext": [str(v) for v in uniques.to_list()],
    }
    return codes.to_list(), overrides


def _build_hover(
    df: pl.DataFrame,
    base_cols: Sequence[str],
    extra: Sequence[str] | None,
) -> tuple[list[list], str]:
    """
    Build customdata + hovertemplate for the given base columns plus extras.

    Returns
    -------
    customdata : list of column values, transposed for Plotly
    template   : hovertemplate string
    """
    cols = list(base_cols)
    if extra:
        _check_columns(df, extra)
        for c in extra:
            if c not in cols:
                cols.append(c)

    customdata = df.select(cols).rows()

    parts = [f"{c}: %{{customdata[{i}]}}" for i, c in enumerate(cols)]
    template = "<br>".join(parts) + "<extra></extra>"
    return customdata, template


def scatter3d(
    df: pl.DataFrame,
    x: str,
    y: str,
    z: str,
    color: str | None = None,
    size: float | str = 2.0,
    *,
    title: str | None = None,
    colorscale: str = "Viridis",
    color_range: tuple[float, float] | None = None,
    opacity: float = 0.8,
    hover_columns: Sequence[str] | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    zaxis_title: str | None = None,
    colorbar_title: str | None = None,
) -> go.Figure:
    """
    Build a 3D scatter plot. Pass in already-downsampled data.

    Parameters
    ----------
    df
        Pre-downsampled polars DataFrame.
    x, y, z
        Column names for the three spatial axes (any columns are allowed; you
        don't have to use Demand X/Y/Z if you want non-spatial views).
    color
        Column name to map to point color, or None for a single uniform color.
    size
        Either a fixed marker size in pixels, or a column name to map to size.
        Be cautious with column-mapped size — large dynamic ranges produce
        unreadable plots; consider scaling first.
    title
        Optional figure title.
    colorscale
        Plotly colorscale name (e.g. "Viridis", "Plasma", "Turbo", "RdBu").
    color_range
        (lo, hi) to clip the colorbar. Useful for ignoring outliers.
    opacity
        0–1.
    hover_columns
        Extra columns to show in the hover tooltip (besides x, y, z, color).
    xaxis_title, yaxis_title, zaxis_title, colorbar_title
        Override axis / colorbar labels. Default is the column name.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    base = [x, y, z] + ([color] if color else [])
    _check_columns(df, base)

    if df.height > 200_000:
        print(
            f"WARNING: scatter3d called with {df.height:,} points. Plotly's "
            "Scatter3d gets sluggish above ~200k. Consider downsampling further."
        )

    customdata, hover_template = _build_hover(df, base, hover_columns)

    marker = dict(
        size=_resolve_size(df, size),
        opacity=opacity,
    )
    if color:
        color_values, colorbar_overrides = _resolve_color(df, color)
        marker["color"] = color_values
        marker["colorscale"] = colorscale
        cb = {"title": colorbar_title if colorbar_title is not None else color}
        if colorbar_overrides:
            cb.update(colorbar_overrides)
        marker["colorbar"] = cb
        marker["showscale"] = True
        if color_range is not None:
            marker["cmin"], marker["cmax"] = color_range

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=df[x].to_list(),
                y=df[y].to_list(),
                z=df[z].to_list(),
                mode="markers",
                marker=marker,
                customdata=customdata,
                hovertemplate=hover_template,
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=xaxis_title if xaxis_title is not None else x,
            yaxis_title=yaxis_title if yaxis_title is not None else y,
            zaxis_title=zaxis_title if zaxis_title is not None else z,
            aspectmode="data",  # respects real geometry
        ),
        margin=dict(l=0, r=0, t=40 if title else 0, b=0),
    )
    return fig


def scatter2d(
    df: pl.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    size: float | str = 8.0,
    *,
    title: str | None = None,
    colorscale: str = "Viridis",
    color_range: tuple[float, float] | None = None,
    opacity: float = 0.8,
    hover_columns: Sequence[str] | None = None,
    equal_aspect: bool = True,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    colorbar_title: str | None = None,
) -> go.Figure:
    """
    Build a 2D scatter plot using Scattergl (WebGL).  Comfortable up to ~1M points.

    Parameters mirror scatter3d, minus the z axis.

    equal_aspect
        If True (default), forces equal scaling on X and Y. This is what you
        want for spatial top-down views where geometry shouldn't be distorted.
        Set False for non-spatial scatters (e.g. signal vs. time).
    xaxis_title, yaxis_title, colorbar_title
        Override axis / colorbar labels. Default is the column name.
    """
    base = [x, y] + ([color] if color else [])
    _check_columns(df, base)

    if df.height > 1_000_000:
        print(
            f"WARNING: scatter2d called with {df.height:,} points. Scattergl "
            "handles up to ~1M comfortably; beyond that consider downsampling."
        )

    customdata, hover_template = _build_hover(df, base, hover_columns)

    marker = dict(
        size=_resolve_size(df, size),
        opacity=opacity,
    )
    if color:
        color_values, colorbar_overrides = _resolve_color(df, color)
        marker["color"] = color_values
        marker["colorscale"] = colorscale
        cb = {"title": colorbar_title if colorbar_title is not None else color}
        if colorbar_overrides:
            cb.update(colorbar_overrides)
        marker["colorbar"] = cb
        marker["showscale"] = True
        if color_range is not None:
            marker["cmin"], marker["cmax"] = color_range

    fig = go.Figure(
        data=[
            go.Scattergl(
                x=df[x].to_list(),
                y=df[y].to_list(),
                mode="markers",
                marker=marker,
                customdata=customdata,
                hovertemplate=hover_template,
            )
        ]
    )
    layout_kwargs: dict = dict(
        title=title,
        xaxis_title=xaxis_title if xaxis_title is not None else x,
        yaxis_title=yaxis_title if yaxis_title is not None else y,
        margin=dict(l=40, r=40, t=40 if title else 20, b=40),
    )
    fig.update_layout(**layout_kwargs)
    if equal_aspect:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def bar(
    df: pl.DataFrame,
    x: str,
    y: str,
    *,
    color: str | None = None,
    title: str | None = None,
    colorscale: str = "Viridis",
    color_range: tuple[float, float] | None = None,
    opacity: float = 0.85,
    hover_columns: Sequence[str] | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    colorbar_title: str | None = None,
    sort_by: Literal["x", "y", "none"] = "none",
    sort_descending: bool = False,
    orientation: Literal["v", "h"] = "v",
) -> go.Figure:
    """
    Build a bar chart. Pass in one row per bar.

    Parameters
    ----------
    df
        DataFrame with one row per bar.
    x
        Category column (usually string-typed). For horizontal orientation
        this becomes the y axis of the plot — but the API stays consistent:
        ``x`` is always the category column, ``y`` is always the value column.
    y
        Numeric value column.
    color
        Optional column to colour bars by. Numeric columns produce a
        continuous colorscale; string columns get discrete categorical
        coloring with a label-aware colorbar.
    title, xaxis_title, yaxis_title, colorbar_title
        Optional overrides. Defaults are the column names.
    colorscale, color_range, opacity
        Forwarded to Plotly. Same semantics as scatter functions.
    hover_columns
        Extra columns to show in the hover tooltip alongside x, y, and color.
    sort_by
        ``"x"`` to sort by category, ``"y"`` to sort by value, ``"none"`` to
        keep DataFrame order. Default ``"none"``.
    sort_descending
        Reverse the sort. Default False (ascending).
    orientation
        ``"v"`` for vertical bars (default), ``"h"`` for horizontal. Use
        horizontal when category labels are long.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    base = [x, y] + ([color] if color else [])
    _check_columns(df, base)

    if sort_by == "x":
        df = df.sort(x, descending=sort_descending)
    elif sort_by == "y":
        df = df.sort(y, descending=sort_descending)
    elif sort_by != "none":
        raise ValueError(f"sort_by must be 'x', 'y', or 'none', got {sort_by!r}")

    customdata, hover_template = _build_hover(df, base, hover_columns)

    marker: dict = {"opacity": opacity}
    if color:
        color_values, colorbar_overrides = _resolve_color(df, color)
        marker["color"] = color_values
        marker["colorscale"] = colorscale
        cb = {"title": colorbar_title if colorbar_title is not None else color}
        if colorbar_overrides:
            cb.update(colorbar_overrides)
        marker["colorbar"] = cb
        marker["showscale"] = True
        if color_range is not None:
            marker["cmin"], marker["cmax"] = color_range

    if orientation == "v":
        plot_x, plot_y = df[x].to_list(), df[y].to_list()
    elif orientation == "h":
        plot_x, plot_y = df[y].to_list(), df[x].to_list()
    else:
        raise ValueError(f"orientation must be 'v' or 'h', got {orientation!r}")

    fig = go.Figure(
        data=[
            go.Bar(
                x=plot_x,
                y=plot_y,
                orientation=orientation,
                marker=marker,
                customdata=customdata,
                hovertemplate=hover_template,
            )
        ]
    )

    if orientation == "v":
        x_title = xaxis_title if xaxis_title is not None else x
        y_title = yaxis_title if yaxis_title is not None else y
    else:
        x_title = xaxis_title if xaxis_title is not None else y
        y_title = yaxis_title if yaxis_title is not None else x

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        margin=dict(l=60, r=40, t=40 if title else 20, b=60),
    )
    return fig


def contour(
    df: pl.DataFrame,
    x: str,
    y: str,
    z: str,
    *,
    show_points: bool = True,
    point_size: float = 6.0,
    point_color: str = "white",
    point_outline: str = "black",
    title: str | None = None,
    colorscale: str = "Viridis",
    color_range: tuple[float, float] | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    colorbar_title: str | None = None,
    hover_columns: Sequence[str] | None = None,
) -> go.Figure:
    """
    Plot a filled contour map of z over the (x, y) plane.

    Input is long-form: one row per measured (x, y) coordinate with its z
    value. Each unique x value becomes a column in the contour grid; each
    unique y value becomes a row. Cells with no data are shown as gaps.

    Parameters
    ----------
    df
        DataFrame with at least x, y, z columns. Each row is one sample.
    x, y, z
        Column names. x and y form the grid axes; z is the contour value.
    show_points
        Overlay markers at the actual measured (x, y) positions.
        Default True — keeps the visualization honest about which points
        were measured vs interpolated.
    point_size, point_color, point_outline
        Styling for the overlay markers when ``show_points=True``.
    title, colorscale, color_range, xaxis_title, yaxis_title, colorbar_title
        Standard Plotly parameters. Same semantics as scatter2d.
    hover_columns
        Extra columns to show in hover tooltips on the overlaid points.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    _check_columns(df, [x, y, z])
    grid = df.pivot(
        on=x,
        index=y,
        values=z,
        aggregate_function="first",
    ).sort(y)

    y_values = grid[y].to_list()
    x_col_names = [c for c in grid.columns if c != y]
    x_values = [_try_parse_numeric(c) for c in x_col_names]

    sort_order = sorted(range(len(x_values)), key=lambda i: x_values[i])
    x_values = [x_values[i] for i in sort_order]
    z_matrix = []
    for row in grid.iter_rows(named=True):
        z_matrix.append([row[x_col_names[i]] for i in sort_order])

    contour_kwargs: dict = dict(
        x=x_values,
        y=y_values,
        z=z_matrix,
        colorscale=colorscale,
        line_smoothing=0,
        contours_coloring="fill",
        connectgaps=False,
        colorbar=dict(title=colorbar_title if colorbar_title is not None else z),
    )
    if color_range is not None:
        contour_kwargs["zmin"], contour_kwargs["zmax"] = color_range

    fig = go.Figure(data=[go.Contour(**contour_kwargs)])

    if show_points:
        base_cols = [x, y, z]
        customdata, hover_template = _build_hover(df, base_cols, hover_columns)
        fig.add_trace(
            go.Scatter(
                x=df[x].to_list(),
                y=df[y].to_list(),
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=point_color,
                    line=dict(color=point_outline, width=1),
                ),
                customdata=customdata,
                hovertemplate=hover_template,
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title if xaxis_title is not None else x,
        yaxis_title=yaxis_title if yaxis_title is not None else y,
        margin=dict(l=60, r=40, t=40 if title else 20, b=60),
    )
    return fig


def _try_parse_numeric(s: str) -> float | str:
    """Try parsing a string as numeric. Polars pivot turns numeric column
    headers into strings; we convert back so the contour grid axis is
    properly numeric."""
    try:
        return float(s)
    except (ValueError, TypeError):
        return s


def scatter2d_layered(
    df: pl.DataFrame,
    x: str,
    y: str,
    color_columns: str | Sequence[str],
    *,
    layer_col: str = "layer",
    points_per_layer: int = 10_000,
    seed: int = 0,
    size: float = 8.0,
    colorscale: str = "Viridis",
    opacity: float = 0.85,
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    colorbar_title: str | None = None,
    equal_aspect: bool = True,
    hover_columns: Sequence[str] | None = None,
) -> go.Figure:
    """
    Top-down 2D scatter with a slider over layers and a dropdown over
    signal columns.

    Each layer is a separate Plotly trace; the slider toggles which one is
    visible. The dropdown rewrites the active signal's color array via a
    Plotly ``restyle`` update.

    Performance notes
    -----------------
    The rendered HTML embeds all per-layer point arrays. With M layers, S
    signals, and N points per layer, the file holds roughly M*S*N float
    entries. For 334 layers x 4 signals x 10,000 points that's ~13M floats
    plus coordinates — large but manageable. Use ``points_per_layer`` to
    trade resolution against file size; ``hover_columns`` further inflates
    the file (customdata duplicated per signal swap), so leave it None
    unless needed.

    Parameters
    ----------
    df
        DataFrame with at least x, y, layer_col, and all color_columns.
    x, y
        Column names for the spatial axes (typically Demand X / Demand Y).
    color_columns
        Either a single column name or a list of column names. If multiple,
        a dropdown menu is rendered. The first entry is the initially-shown
        signal.
    layer_col
        Layer column name. Default ``"layer"``.
    points_per_layer
        Random downsample target per layer. Default 10_000. Most layers
        look like a continuous fill at 5-10k points; bumping higher mostly
        inflates the HTML file size without visible benefit.
    seed
        RNG seed for the per-layer downsample. Default 0.
    size, opacity, colorscale
        Marker styling. ``colorscale`` applies to whichever signal is active.
    title, xaxis_title, yaxis_title, colorbar_title
        Standard Plotly label overrides.
    equal_aspect
        Lock 1:1 aspect for spatial XY data (default True).
    hover_columns
        Extra columns to surface in the hover tooltip alongside x, y, and
        the active signal. **Each extra column adds a copy per signal in
        the HTML**, so leave None for the lowest file size.

    Returns
    -------
    plotly.graph_objects.Figure with a slider and (if multiple signals) a
    signal dropdown.
    """
    if isinstance(color_columns, str):
        color_columns = [color_columns]
    color_columns = list(color_columns)
    if not color_columns:
        raise ValueError("color_columns must not be empty")

    base_cols = [x, y, layer_col, *color_columns]
    _check_columns(df, base_cols)
    if hover_columns:
        _check_columns(df, hover_columns)

    color_ranges: dict[str, tuple[float, float]] = {}
    for c in color_columns:
        col = df[c].drop_nulls()
        if col.is_empty():
            color_ranges[c] = (0.0, 1.0)
        else:
            color_ranges[c] = (float(col.min()), float(col.max()))

    cols_needed = list(
        dict.fromkeys([x, y, layer_col, *color_columns, *(hover_columns or [])])
    )
    rng = np.random.default_rng(seed)

    partitions = df.select(cols_needed).partition_by(layer_col, as_dict=True)

    layer_data: list[dict] = []
    for key, sub in partitions.items():
        layer_value = key[0] if isinstance(key, tuple) else key
        n = sub.height
        if n == 0:
            continue
        if n > points_per_layer:
            idx = rng.choice(n, size=points_per_layer, replace=False)
            sub = sub[idx]

        rec: dict = {
            "layer": int(layer_value),
            "x": sub[x].to_numpy(),
            "y": sub[y].to_numpy(),
            "colors_by_signal": {c: sub[c].to_numpy() for c in color_columns},
            "n_points": sub.height,
        }
        if hover_columns:
            rec["hover_arrays"] = {hc: sub[hc].to_numpy() for hc in hover_columns}
        layer_data.append(rec)

    if not layer_data:
        raise ValueError("No data found for any layer")

    layer_data.sort(key=lambda d: d["layer"])

    initial_signal = color_columns[0]
    traces = []
    for ld in layer_data:
        marker = dict(
            size=size,
            opacity=opacity,
            color=ld["colors_by_signal"][initial_signal],
            colorscale=colorscale,
            cmin=color_ranges[initial_signal][0],
            cmax=color_ranges[initial_signal][1],
            showscale=True,
            colorbar=dict(
                title=colorbar_title if colorbar_title is not None else initial_signal,
            ),
        )

        customdata = _stack_customdata(
            ld["colors_by_signal"][initial_signal],
            ld.get("hover_arrays"),
            hover_columns,
        )
        hover_template = _build_layered_hover_template(
            x,
            y,
            initial_signal,
            hover_columns,
        )

        traces.append(
            go.Scattergl(
                x=ld["x"],
                y=ld["y"],
                mode="markers",
                marker=marker,
                customdata=customdata,
                hovertemplate=hover_template,
                name=f"layer {ld['layer']}",
                visible=False,
            )
        )
    traces[0].visible = True

    slider_steps = []
    for i, ld in enumerate(layer_data):
        visibility = [False] * len(traces)
        visibility[i] = True
        slider_steps.append(
            dict(
                method="update",
                args=[{"visible": visibility}],
                label=str(ld["layer"]),
            )
        )

    sliders = [
        dict(
            active=0,
            currentvalue=dict(prefix=f"{layer_col}: ", font=dict(size=14)),
            pad=dict(t=50),
            steps=slider_steps,
        )
    ]

    updatemenus = []
    if len(color_columns) > 1:
        signal_buttons = []
        for sig in color_columns:
            cmin, cmax = color_ranges[sig]
            color_arrays = [ld["colors_by_signal"][sig] for ld in layer_data]

            new_customdata = [
                _stack_customdata(
                    ld["colors_by_signal"][sig],
                    ld.get("hover_arrays"),
                    hover_columns,
                )
                for ld in layer_data
            ]
            new_hovertemplate = _build_layered_hover_template(
                x,
                y,
                sig,
                hover_columns,
            )

            signal_buttons.append(
                dict(
                    label=sig,
                    method="restyle",
                    args=[
                        {
                            "marker.color": color_arrays,
                            "marker.cmin": [cmin] * len(traces),
                            "marker.cmax": [cmax] * len(traces),
                            "marker.colorbar.title.text": [
                                colorbar_title if colorbar_title is not None else sig
                            ]
                            * len(traces),
                            "customdata": new_customdata,
                            "hovertemplate": [new_hovertemplate] * len(traces),
                        }
                    ],
                )
            )
        updatemenus.append(
            dict(
                buttons=signal_buttons,
                direction="down",
                x=1.02,
                y=1.0,
                xanchor="left",
                yanchor="top",
                showactive=True,
                pad=dict(t=4, r=4),
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title if xaxis_title is not None else x,
        yaxis_title=yaxis_title if yaxis_title is not None else y,
        sliders=sliders,
        updatemenus=updatemenus,
        margin=dict(l=60, r=60, t=60 if title else 30, b=80),
        showlegend=False,
    )
    if equal_aspect:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _stack_customdata(
    signal_values: np.ndarray,
    hover_arrays: dict[str, np.ndarray] | None,
    hover_columns: Sequence[str] | None,
) -> np.ndarray:
    """
    Build customdata as a 2D NumPy array.

    Layout: column 0 is the active signal value (so the hover template can
    reference customdata[0] regardless of which signal is active);
    columns 1.. are the hover_columns in the order given.
    """
    if hover_columns and hover_arrays:
        cols = [signal_values] + [hover_arrays[hc] for hc in hover_columns]
        return np.column_stack(cols)
    return signal_values.reshape(-1, 1)


def _build_layered_hover_template(
    x_name: str,
    y_name: str,
    signal_name: str,
    hover_columns: Sequence[str] | None,
) -> str:
    """Build the hover template string for the layered scatter."""
    parts = [
        f"{x_name}: %{{x}}",
        f"{y_name}: %{{y}}",
        f"{signal_name}: %{{customdata[0]}}",
    ]
    if hover_columns:
        for j, hc in enumerate(hover_columns):
            parts.append(f"{hc}: %{{customdata[{j+1}]}}")
    return "<br>".join(parts) + "<extra></extra>"
