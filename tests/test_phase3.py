"""
Tests for ampm.plotting.

We don't render figures here; we inspect their structure to confirm the
right traces, data, and styling are produced.
"""

from __future__ import annotations

import sys
from pathlib import Path
import io
import contextlib

import numpy as np
import polars as pl
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))

from ampm.plotting import scatter2d, scatter3d, scatter2d_layered, kde


def make_df(n: int = 1000, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "Demand X": rng.uniform(-30, 30, n),
            "Demand Y": rng.uniform(-30, 30, n),
            "Z": rng.uniform(0, 6, n),
            "MeltVIEW melt pool (mean)": rng.normal(500, 100, n),
            "layer": rng.integers(101, 201, n).astype(np.int32),
            "Start time": rng.integers(0, 30_000_000, n),
        }
    )


def test_scatter3d_basic() -> None:
    df = make_df()
    fig = scatter3d(df, x="Demand X", y="Demand Y", z="Z")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    trace = fig.data[0]
    assert trace.type == "scatter3d"
    assert len(trace.x) == 1000
    assert len(trace.y) == 1000
    assert len(trace.z) == 1000
    assert trace.marker.color is None
    assert fig.layout.scene.aspectmode == "data"
    print("  scatter3d basic OK")


def test_scatter3d_with_color() -> None:
    df = make_df()
    fig = scatter3d(
        df,
        x="Demand X",
        y="Demand Y",
        z="Z",
        color="MeltVIEW melt pool (mean)",
    )
    trace = fig.data[0]
    assert len(trace.marker.color) == 1000
    assert trace.marker.colorscale is not None
    assert trace.marker.showscale is True
    assert trace.marker.colorbar.title.text == "MeltVIEW melt pool (mean)"
    print("  scatter3d with color OK")


def test_scatter3d_color_range_clip() -> None:
    df = make_df()
    fig = scatter3d(
        df,
        x="Demand X",
        y="Demand Y",
        z="Z",
        color="MeltVIEW melt pool (mean)",
        color_range=(400, 600),
    )
    trace = fig.data[0]
    assert trace.marker.cmin == 400
    assert trace.marker.cmax == 600
    print("  scatter3d color_range OK")


def test_scatter3d_size_column() -> None:
    df = make_df()
    fig = scatter3d(
        df,
        x="Demand X",
        y="Demand Y",
        z="Z",
        size="MeltVIEW melt pool (mean)",
    )
    trace = fig.data[0]
    assert hasattr(trace.marker.size, "__len__") or isinstance(
        trace.marker.size, (list, tuple)
    )
    assert len(trace.marker.size) == 1000
    print("  scatter3d size from column OK")


def test_scatter3d_hover_columns() -> None:
    df = make_df()
    fig = scatter3d(
        df,
        x="Demand X",
        y="Demand Y",
        z="Z",
        color="MeltVIEW melt pool (mean)",
        hover_columns=["layer", "Start time"],
    )
    trace = fig.data[0]
    assert len(trace.customdata) == 1000
    assert len(trace.customdata[0]) == 6
    template = trace.hovertemplate
    for col in [
        "Demand X",
        "Demand Y",
        "Z",
        "MeltVIEW melt pool (mean)",
        "layer",
        "Start time",
    ]:
        assert col in template, f"{col} missing from hovertemplate"
    print("  scatter3d hover_columns OK")


def test_scatter3d_label_overrides() -> None:
    df = make_df()
    fig = scatter3d(
        df,
        x="Demand X",
        y="Demand Y",
        z="Z",
        color="MeltVIEW melt pool (mean)",
        xaxis_title="X (mm)",
        yaxis_title="Y (mm)",
        zaxis_title="Z (mm)",
        colorbar_title="Melt pool (a.u.)",
    )
    assert fig.layout.scene.xaxis.title.text == "X (mm)"
    assert fig.layout.scene.yaxis.title.text == "Y (mm)"
    assert fig.layout.scene.zaxis.title.text == "Z (mm)"
    assert fig.data[0].marker.colorbar.title.text == "Melt pool (a.u.)"
    print("  scatter3d label overrides OK")


def test_scatter2d_label_overrides() -> None:
    df = make_df()
    fig = scatter2d(
        df,
        x="Demand X",
        y="Demand Y",
        color="MeltVIEW melt pool (mean)",
        xaxis_title="X (mm)",
        yaxis_title="Y (mm)",
        colorbar_title="Melt pool (a.u.)",
    )
    assert fig.layout.xaxis.title.text == "X (mm)"
    assert fig.layout.yaxis.title.text == "Y (mm)"
    assert fig.data[0].marker.colorbar.title.text == "Melt pool (a.u.)"
    print("  scatter2d label overrides OK")


def test_scatter3d_categorical_color() -> None:
    """String/categorical color column should be encoded to ints with discrete colorbar ticks."""
    df = make_df()
    labels = ["A", "B", "C", "D"]
    df = df.with_columns(
        pl.Series("part_id", [labels[i % 4] for i in range(df.height)])
    )
    fig = scatter3d(
        df,
        x="Demand X",
        y="Demand Y",
        z="Z",
        color="part_id",
    )
    trace = fig.data[0]
    assert all(
        isinstance(c, int) for c in trace.marker.color
    ), f"Expected int codes, got {set(type(c).__name__ for c in trace.marker.color)}"
    assert list(trace.marker.colorbar.tickvals) == [0, 1, 2, 3]
    assert list(trace.marker.colorbar.ticktext) == ["A", "B", "C", "D"]
    print("  scatter3d categorical color OK")


def test_scatter2d_categorical_color() -> None:
    df = make_df()
    df = df.with_columns(pl.Series("part_id", ["X", "Y"] * (df.height // 2)))
    fig = scatter2d(df, x="Demand X", y="Demand Y", color="part_id")
    trace = fig.data[0]
    assert all(isinstance(c, int) for c in trace.marker.color)
    assert list(trace.marker.colorbar.ticktext) == ["X", "Y"]
    print("  scatter2d categorical color OK")


def test_numeric_color_unchanged() -> None:
    """Numeric color should still pass through as raw values, not get encoded."""
    df = make_df()
    fig = scatter3d(
        df,
        x="Demand X",
        y="Demand Y",
        z="Z",
        color="MeltVIEW melt pool (mean)",
    )
    trace = fig.data[0]
    assert any(isinstance(c, float) for c in trace.marker.color)
    assert trace.marker.colorbar.tickvals is None
    print("  numeric color unchanged OK")


def test_scatter3d_unknown_column_raises() -> None:
    df = make_df()
    try:
        scatter3d(df, x="bogus", y="Demand Y", z="Z")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
    print("  scatter3d unknown column raises OK")


def test_scatter2d_basic() -> None:
    df = make_df()
    fig = scatter2d(df, x="Demand X", y="Demand Y")
    trace = fig.data[0]
    assert trace.type == "scattergl"  # WebGL backend
    assert len(trace.x) == 1000
    assert fig.layout.yaxis.scaleanchor == "x"
    print("  scatter2d basic OK")


def test_scatter2d_with_color() -> None:
    df = make_df()
    fig = scatter2d(
        df,
        x="Demand X",
        y="Demand Y",
        color="MeltVIEW melt pool (mean)",
    )
    trace = fig.data[0]
    assert len(trace.marker.color) == 1000
    assert trace.marker.colorscale is not None
    print("  scatter2d with color OK")


def test_scatter2d_no_equal_aspect() -> None:
    df = make_df()
    fig = scatter2d(
        df, x="Start time", y="MeltVIEW melt pool (mean)", equal_aspect=False
    )
    assert fig.layout.yaxis.scaleanchor is None
    print("  scatter2d equal_aspect=False OK")


def test_scatter2d_unknown_column_raises() -> None:
    df = make_df()
    try:
        scatter2d(df, x="Demand X", y="bogus")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
    print("  scatter2d unknown column raises OK")


def _make_layered_df(
    n_per_layer: int = 200, n_layers: int = 5, seed: int = 0
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for layer in range(1, n_layers + 1):
        for _ in range(n_per_layer):
            rows.append(
                {
                    "Demand X": rng.uniform(-30, 30),
                    "Demand Y": rng.uniform(-30, 30),
                    "layer": layer,
                    "meltpool": rng.uniform(150, 250) + layer * 5,
                    "plasma": rng.uniform(50, 100) + layer * 2,
                }
            )
    return pl.DataFrame(rows)


def test_scatter2d_layered_single_signal() -> None:
    df = _make_layered_df()
    fig = scatter2d_layered(df, x="Demand X", y="Demand Y", color_columns="meltpool")
    assert len(fig.data) == 5
    assert len(fig.layout.sliders[0].steps) == 5
    assert not fig.layout.updatemenus
    assert fig.data[0].visible is True
    assert all(t.visible is False for t in fig.data[1:])
    print("  scatter2d_layered single signal OK")


def test_scatter2d_layered_multi_signal_dropdown() -> None:
    df = _make_layered_df()
    fig = scatter2d_layered(
        df, x="Demand X", y="Demand Y", color_columns=["meltpool", "plasma"]
    )
    assert len(fig.data) == 5
    assert len(fig.layout.updatemenus) == 1
    buttons = fig.layout.updatemenus[0].buttons
    assert len(buttons) == 2
    assert buttons[0].label == "meltpool"
    assert buttons[1].label == "plasma"
    print("  scatter2d_layered multi-signal dropdown OK")


def test_scatter2d_layered_slider_labels_are_layer_numbers() -> None:
    df = _make_layered_df(n_layers=5)
    fig = scatter2d_layered(df, x="Demand X", y="Demand Y", color_columns="meltpool")
    labels = [s.label for s in fig.layout.sliders[0].steps]
    assert labels == ["1", "2", "3", "4", "5"]
    print("  slider labels are layer numbers OK")


def test_scatter2d_layered_skips_empty_layers() -> None:
    """Layers with no data should not appear in the slider."""
    df = _make_layered_df()  # has layers 1-5
    df_subset = df.filter(pl.col("layer").is_in([2, 4]))
    fig = scatter2d_layered(
        df_subset, x="Demand X", y="Demand Y", color_columns="meltpool"
    )
    assert len(fig.data) == 2
    labels = [s.label for s in fig.layout.sliders[0].steps]
    assert labels == ["2", "4"]
    print("  empty layers excluded from slider OK")


def test_scatter2d_layered_downsample_per_layer() -> None:
    """Layers with more rows than points_per_layer should be downsampled."""
    df = _make_layered_df(n_per_layer=500, n_layers=3)
    fig = scatter2d_layered(
        df, x="Demand X", y="Demand Y", color_columns="meltpool", points_per_layer=100
    )
    for trace in fig.data:
        assert len(trace.x) <= 100
    print("  downsample per layer OK")


def test_scatter2d_layered_color_range_per_signal() -> None:
    """Dropdown options carry their own cmin/cmax values."""
    df = _make_layered_df()
    fig = scatter2d_layered(
        df, x="Demand X", y="Demand Y", color_columns=["meltpool", "plasma"]
    )
    buttons = fig.layout.updatemenus[0].buttons
    for btn in buttons:
        restyle = btn.args[0]
        assert "marker.cmin" in restyle
        assert "marker.cmax" in restyle
        n_traces = len(fig.data)
        assert len(restyle["marker.cmin"]) == n_traces
    print("  per-signal color range OK")


def test_scatter2d_layered_unknown_column_raises() -> None:
    df = _make_layered_df()
    try:
        scatter2d_layered(df, x="Demand X", y="Demand Y", color_columns="bogus")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
    try:
        scatter2d_layered(df, x="bogus", y="Demand Y", color_columns="meltpool")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
    print("  scatter2d_layered unknown column raises OK")


def test_scatter2d_layered_empty_color_columns_raises() -> None:
    df = _make_layered_df()
    try:
        scatter2d_layered(df, x="Demand X", y="Demand Y", color_columns=[])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    print("  scatter2d_layered empty color_columns raises OK")


def _make_kde_df(seed: int = 0) -> pl.DataFrame:
    """3 parts with very different distributions so curves are easy to verify."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(500):
        rows.append({"part_id": "Part(1)", "signal": rng.normal(200, 10)})
    for _ in range(500):
        rows.append({"part_id": "Part(2)", "signal": rng.normal(250, 30)})
    for _ in range(500):
        rows.append({"part_id": "Part(3)", "signal": rng.normal(220, 60)})
    for _ in range(50):
        rows.append({"part_id": "noise", "signal": rng.uniform(0, 500)})
    return pl.DataFrame(rows)


def test_kde_basic() -> None:
    df = _make_kde_df()
    fig = kde(df, column="signal", group_by="part_id")
    # 3 traces (noise dropped by default).
    assert len(fig.data) == 3
    names = {t.name for t in fig.data}
    assert names == {"Part(1)", "Part(2)", "Part(3)"}
    for t in fig.data:
        assert len(t.x) == 200
        assert len(t.y) == 200
        area = np.trapezoid(t.y, t.x)
        assert 0.8 < area < 1.2, f"{t.name}: area = {area}"
    print("  kde basic OK")


def test_kde_filter_groups() -> None:
    df = _make_kde_df()
    fig = kde(df, column="signal", group_by="part_id", groups=["Part(1)", "Part(3)"])
    assert len(fig.data) == 2
    names = {t.name for t in fig.data}
    assert names == {"Part(1)", "Part(3)"}
    print("  kde groups filter OK")


def test_kde_drop_noise_default() -> None:
    df = _make_kde_df()
    fig = kde(df, column="signal", group_by="part_id")
    names = {t.name for t in fig.data}
    assert "noise" not in names
    print("  kde drops noise by default OK")


def test_kde_keep_noise() -> None:
    df = _make_kde_df()
    fig = kde(df, column="signal", group_by="part_id", drop_noise=False)
    names = {t.name for t in fig.data}
    assert "noise" in names
    print("  kde keeps noise when drop_noise=False OK")


def test_kde_range_clip() -> None:
    df = _make_kde_df()
    fig = kde(df, column="signal", group_by="part_id", range_clip=(150, 300))
    for t in fig.data:
        assert all(150 <= xv <= 300 for xv in t.x)
    print("  kde range_clip OK")


def test_kde_no_fill_lines_only() -> None:
    df = _make_kde_df()
    fig = kde(df, column="signal", group_by="part_id", fill=False)
    for t in fig.data:
        assert t.fill in (None, "none")
    print("  kde fill=False OK")


def test_kde_warns_on_many_groups() -> None:
    """With > 12 groups, a readability warning should be printed."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(15):
        for _ in range(50):
            rows.append(
                {"part_id": f"Part({i})", "signal": rng.normal(200 + i * 5, 10)}
            )
    df = pl.DataFrame(rows)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fig = kde(df, column="signal", group_by="part_id")
    assert "Warning" in buf.getvalue()
    assert "may be hard to read" in buf.getvalue()
    assert len(fig.data) == 15
    print("  kde warns when >12 groups OK")


def test_kde_unknown_column_raises() -> None:
    df = _make_kde_df()
    try:
        kde(df, column="bogus", group_by="part_id")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
    print("  kde unknown column raises OK")


def test_kde_empty_input_raises() -> None:
    """If filtering removes all rows, raise."""
    df = _make_kde_df()
    try:
        kde(df, column="signal", group_by="part_id", groups=["NoSuchPart"])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    print("  kde empty after filtering raises OK")


def test_kde_label_overrides() -> None:
    df = _make_kde_df()
    fig = kde(
        df,
        column="signal",
        group_by="part_id",
        title="My Distribution",
        xaxis_title="Signal value",
        yaxis_title="Probability density",
    )
    assert fig.layout.title.text == "My Distribution"
    assert fig.layout.xaxis.title.text == "Signal value"
    assert fig.layout.yaxis.title.text == "Probability density"
    print("  kde label overrides OK")


def test_kde_max_points_per_group_caps_sampling() -> None:
    """When a group has > max_points_per_group, it gets randomly sampled."""
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(100_000):
        rows.append({"part_id": "Big", "signal": rng.normal(200, 20)})
    for _ in range(100):
        rows.append({"part_id": "Small", "signal": rng.normal(250, 30)})
    df = pl.DataFrame(rows)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fig = kde(
            df,
            column="signal",
            group_by="part_id",
            max_points_per_group=5_000,
            verbose=True,
        )
    output = buf.getvalue()
    assert "Big: sampled 5,000/100,000" in output
    assert "Small: sampled" not in output
    assert len(fig.data) == 2
    print("  kde max_points_per_group caps + reports OK")


def test_kde_max_points_per_group_disabled() -> None:
    """max_points_per_group=None disables sampling."""
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(20_000):
        rows.append({"part_id": "P", "signal": rng.normal(200, 20)})
    df = pl.DataFrame(rows)

    import io, contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        kde(
            df,
            column="signal",
            group_by="part_id",
            max_points_per_group=None,
            verbose=True,
        )
    assert "sampled" not in buf.getvalue()
    print("  kde max_points_per_group=None disables sampling OK")


def test_kde_verbose_false_silent() -> None:
    """verbose=False suppresses sampling messages."""
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(50_000):
        rows.append({"part_id": "P", "signal": rng.normal(200, 20)})
    df = pl.DataFrame(rows)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        kde(
            df,
            column="signal",
            group_by="part_id",
            max_points_per_group=1_000,
            verbose=False,
        )
    assert "sampled" not in buf.getvalue()
    print("  kde verbose=False suppresses sampling messages OK")


def test_kde_sampling_preserves_curve_shape() -> None:
    """Sampled KDE should be visually similar to full-data KDE."""
    rng = np.random.default_rng(0)
    n = 50_000
    vals = rng.normal(200, 25, n)
    df = pl.DataFrame({"part_id": ["A"] * n, "signal": vals})

    fig_full = kde(
        df,
        column="signal",
        group_by="part_id",
        max_points_per_group=None,
        verbose=False,
    )
    fig_sampled = kde(
        df,
        column="signal",
        group_by="part_id",
        max_points_per_group=5_000,
        verbose=False,
    )

    y_full = np.array(fig_full.data[0].y)
    y_sampled = np.array(fig_sampled.data[0].y)
    rel_l2 = np.sqrt(np.mean((y_full - y_sampled) ** 2)) / np.sqrt(np.mean(y_full**2))
    assert rel_l2 < 0.1, f"Sampled KDE differs too much: rel L2 = {rel_l2}"
    print(f"  kde sampled curve matches full curve (rel L2 = {rel_l2:.4f}) OK")


def main() -> None:
    print("Phase 3 plotting tests:")
    test_scatter3d_basic()
    test_scatter3d_with_color()
    test_scatter3d_color_range_clip()
    test_scatter3d_size_column()
    test_scatter3d_hover_columns()
    test_scatter3d_label_overrides()
    test_scatter3d_categorical_color()
    test_scatter3d_unknown_column_raises()
    test_scatter2d_basic()
    test_scatter2d_with_color()
    test_scatter2d_no_equal_aspect()
    test_scatter2d_label_overrides()
    test_scatter2d_categorical_color()
    test_scatter2d_unknown_column_raises()
    test_numeric_color_unchanged()
    test_scatter2d_layered_single_signal()
    test_scatter2d_layered_multi_signal_dropdown()
    test_scatter2d_layered_slider_labels_are_layer_numbers()
    test_scatter2d_layered_skips_empty_layers()
    test_scatter2d_layered_downsample_per_layer()
    test_scatter2d_layered_color_range_per_signal()
    test_scatter2d_layered_unknown_column_raises()
    test_scatter2d_layered_empty_color_columns_raises()
    test_kde_basic()
    test_kde_filter_groups()
    test_kde_drop_noise_default()
    test_kde_keep_noise()
    test_kde_range_clip()
    test_kde_no_fill_lines_only()
    test_kde_warns_on_many_groups()
    test_kde_unknown_column_raises()
    test_kde_empty_input_raises()
    test_kde_label_overrides()
    test_kde_max_points_per_group_caps_sampling()
    test_kde_max_points_per_group_disabled()
    test_kde_verbose_false_silent()
    test_kde_sampling_preserves_curve_shape()
    print("\nAll Phase 3 tests passed")


if __name__ == "__main__":
    main()
