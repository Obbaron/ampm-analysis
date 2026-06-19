"""
Tests for ``plotting.py`` — Plotly figure builders.

These assert on the structure of the returned ``go.Figure`` (trace types, data
arrays, marker/colorbar config, layout titles, sliders/menus) rather than
rendering anything. No HTML or display is produced.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from plotly.graph_objects import Figure

import ampm.plotting as plotting
from ampm.plotting import (
    _build_hover,
    _resolve_color,
    _resolve_size,
    _stack_customdata,
    _try_parse_numeric,
    _with_opacity,
    bar,
    contour,
    kde,
    scatter2d,
    scatter2d_layered,
    scatter3d,
)


def xyz_df(n=5, color=None):
    data = {
        "X": pl.Series([float(i) for i in range(n)]),
        "Y": pl.Series([float(i) for i in range(n)]),
        "Z": pl.Series([float(i) for i in range(n)]),
    }
    if color is not None:
        data["c"] = pl.Series(color)
    return pl.DataFrame(data)


class TestHelpers:
    def test_resolve_size_constant(self):
        assert _resolve_size(xyz_df(), 3.0) == 3.0

    def test_resolve_size_column(self):
        df = xyz_df(3)
        assert _resolve_size(df, "X") == [0.0, 1.0, 2.0]

    def test_resolve_size_missing_column_raises(self):
        with pytest.raises(KeyError):
            _resolve_size(xyz_df(), "nope")

    def test_resolve_color_numeric_passthrough(self):
        values, overrides = _resolve_color(xyz_df(3, color=[1.0, 2.0, 3.0]), "c")
        assert values == [1.0, 2.0, 3.0]
        assert overrides is None

    def test_resolve_color_categorical_codes_and_labels(self):
        values, overrides = _resolve_color(xyz_df(3, color=["b", "a", "b"]), "c")
        # dense rank over sorted uniques {a:0, b:1}
        assert values == [1, 0, 1]
        assert overrides is not None
        assert overrides["ticktext"] == ["a", "b"]
        assert overrides["tickvals"] == [0, 1]

    def test_build_hover_template_and_customdata(self):
        df = xyz_df(2)
        customdata, template = _build_hover(df, ["X", "Y"], None)
        assert customdata == [(0.0, 0.0), (1.0, 1.0)]
        assert "customdata[0]" in template and "<extra></extra>" in template

    def test_build_hover_warns_and_omits_missing_extra(self, capsys):
        df = xyz_df(2)
        customdata, template = _build_hover(df, ["X"], ["does_not_exist"])
        assert "not in the data" in capsys.readouterr().out
        assert "does_not_exist" not in template

    def test_try_parse_numeric(self):
        assert _try_parse_numeric("3.5") == 3.5
        assert _try_parse_numeric("abc") == "abc"

    def test_with_opacity(self):
        assert _with_opacity("rgb(10,20,30)", 0.5) == "rgba(10,20,30,0.5)"
        # Non rgb() input is passed through unchanged.
        assert _with_opacity("blue", 0.5) == "blue"

    def test_stack_customdata_signal_only(self):
        out = _stack_customdata(np.array([1.0, 2.0]), None, None)
        assert out.shape == (2, 1)


class TestScatter3d:
    def test_basic_trace_and_data(self):
        fig = scatter3d(xyz_df(4), "X", "Y", "Z")
        assert len(fig.data) == 1
        t = fig.data[0]
        assert t.type == "scatter3d"
        assert list(t.x) == [0.0, 1.0, 2.0, 3.0]
        assert t.marker.color is None  # no color column

    def test_missing_column_raises(self):
        with pytest.raises(KeyError):
            scatter3d(xyz_df(), "X", "Y", "missing")

    def test_numeric_color_sets_colorbar(self):
        fig = scatter3d(xyz_df(3, color=[1.0, 2.0, 3.0]), "X", "Y", "Z", color="c")
        m = fig.data[0].marker
        assert m.showscale is True
        assert m.colorbar.title.text == "c"

    def test_categorical_color_uses_labels(self):
        fig = scatter3d(xyz_df(3, color=["x", "y", "x"]), "X", "Y", "Z", color="c")
        assert list(fig.data[0].marker.colorbar.ticktext) == ["x", "y"]

    def test_null_color_rows_filtered(self, capsys):
        df = xyz_df(3, color=[1.0, None, 3.0])
        fig = scatter3d(df, "X", "Y", "Z", color="c")
        assert len(fig.data[0].x) == 2
        assert "null" in capsys.readouterr().out

    def test_all_null_color_raises(self):
        df = xyz_df(2, color=[None, None])
        with pytest.raises(ValueError, match="null"):
            scatter3d(df, "X", "Y", "Z", color="c")

    def test_axis_titles_default_and_override(self):
        fig = scatter3d(xyz_df(), "X", "Y", "Z")
        assert fig.layout.scene.xaxis.title.text == "X"
        fig2 = scatter3d(xyz_df(), "X", "Y", "Z", xaxis_title="Across")
        assert fig2.layout.scene.xaxis.title.text == "Across"

    def test_color_range_sets_cmin_cmax(self):
        fig = scatter3d(
            xyz_df(3, color=[1.0, 2.0, 3.0]),
            "X",
            "Y",
            "Z",
            color="c",
            color_range=(0.0, 5.0),
        )
        assert fig.data[0].marker.cmin == 0.0
        assert fig.data[0].marker.cmax == 5.0


class TestScatter2d:
    def test_uses_scattergl(self):
        fig = scatter2d(xyz_df(3), "X", "Y")
        assert fig.data[0].type == "scattergl"

    def test_equal_aspect_sets_scaleanchor(self):
        fig = scatter2d(xyz_df(3), "X", "Y", equal_aspect=True)
        assert fig.layout.yaxis.scaleanchor == "x"

    def test_equal_aspect_off(self):
        fig = scatter2d(xyz_df(3), "X", "Y", equal_aspect=False)
        assert fig.layout.yaxis.scaleanchor is None


class TestBar:
    def _df(self):
        return pl.DataFrame({"k": ["a", "b", "c"], "v": [3.0, 1.0, 2.0]})

    def test_basic_bar(self):
        fig = bar(self._df(), "k", "v")
        assert fig.data[0].type == "bar"
        assert list(fig.data[0].x) == ["a", "b", "c"]

    def test_sort_by_y(self):
        fig = bar(self._df(), "k", "v", sort_by="y")
        assert list(fig.data[0].x) == ["b", "c", "a"]  # ascending by value

    def test_sort_by_y_descending(self):
        fig = bar(self._df(), "k", "v", sort_by="y", sort_descending=True)
        assert list(fig.data[0].x) == ["a", "c", "b"]

    def test_invalid_sort_by_raises(self):
        with pytest.raises(ValueError, match="sort_by"):
            bar(self._df(), "k", "v", sort_by="z")

    def test_horizontal_swaps_axes(self):
        fig = bar(self._df(), "k", "v", orientation="h")
        # For horizontal, category goes on the y axis.
        assert list(fig.data[0].y) == ["a", "b", "c"]
        assert list(fig.data[0].x) == [3.0, 1.0, 2.0]

    def test_invalid_orientation_raises(self):
        with pytest.raises(ValueError, match="orientation"):
            bar(self._df(), "k", "v", orientation="diagonal")


class TestContour:
    def _grid_df(self):
        # 2x2 grid of (x, y) with z values.
        return pl.DataFrame(
            {
                "x": [0.0, 1.0, 0.0, 1.0],
                "y": [0.0, 0.0, 1.0, 1.0],
                "z": [1.0, 2.0, 3.0, 4.0],
            }
        )

    def test_contour_with_points(self):
        fig = contour(self._grid_df(), "x", "y", "z", show_points=True)
        types = [t.type for t in fig.data]
        assert "contour" in types
        assert "scatter" in types  # overlay points
        assert len(fig.data) == 2

    def test_contour_without_points(self):
        fig = contour(self._grid_df(), "x", "y", "z", show_points=False)
        assert len(fig.data) == 1
        assert fig.data[0].type == "contour"

    def test_missing_column_raises(self):
        with pytest.raises(KeyError):
            contour(self._grid_df(), "x", "y", "missing")


class TestScatter2dLayered:
    def _df(self, n_per_layer=4):
        rows = []
        for layer in (1, 2):
            for i in range(n_per_layer):
                rows.append((float(i), float(i), layer, float(i), float(2 * i)))
        return pl.DataFrame(
            {
                "X": [r[0] for r in rows],
                "Y": [r[1] for r in rows],
                "layer": [r[2] for r in rows],
                "s1": [r[3] for r in rows],
                "s2": [r[4] for r in rows],
            }
        )

    def test_empty_color_columns_raises(self):
        with pytest.raises(ValueError, match="color_columns"):
            scatter2d_layered(self._df(), "X", "Y", [])

    def test_one_trace_per_layer_first_visible(self):
        fig = scatter2d_layered(self._df(), "X", "Y", "s1")
        assert len(fig.data) == 2  # two layers
        assert fig.data[0].visible is True
        assert fig.data[1].visible is False
        assert len(fig.layout.sliders[0].steps) == 2

    def test_single_signal_has_no_dropdown(self):
        fig = scatter2d_layered(self._df(), "X", "Y", "s1")
        assert len(fig.layout.updatemenus) == 0

    def test_multiple_signals_add_dropdown(self):
        fig = scatter2d_layered(self._df(), "X", "Y", ["s1", "s2"])
        assert len(fig.layout.updatemenus) == 1
        labels = [b.label for b in fig.layout.updatemenus[0].buttons]
        assert labels == ["s1", "s2"]

    def test_downsamples_per_layer(self):
        # 50 points per layer, cap at 10 -> each trace has 10 points.
        fig = scatter2d_layered(
            self._df(n_per_layer=50), "X", "Y", "s1", points_per_layer=10
        )
        assert all(len(t.x) == 10 for t in fig.data)

    def test_no_data_raises(self):
        empty = pl.DataFrame(
            {"X": [], "Y": [], "layer": pl.Series([], dtype=pl.Int64), "s1": []}
        )
        with pytest.raises(ValueError, match="No data"):
            scatter2d_layered(empty, "X", "Y", "s1")


class TestKde:
    def _df(self, seed=0):
        rng = np.random.default_rng(seed)
        a = rng.normal(0.0, 1.0, 200)
        b = rng.normal(5.0, 1.0, 200)
        return pl.DataFrame(
            {
                "part_id": ["A"] * 200 + ["B"] * 200,
                "value": np.concatenate([a, b]),
            }
        )

    def test_one_trace_per_group(self):
        fig = kde(self._df(), "value", verbose=False)
        assert len(fig.data) == 2
        assert sorted(t.name for t in fig.data) == ["A", "B"]

    def test_group_filter(self):
        fig = kde(self._df(), "value", groups=["A"], verbose=False)
        assert len(fig.data) == 1
        assert fig.data[0].name == "A"

    def test_missing_column_raises(self):
        with pytest.raises(KeyError):
            kde(self._df(), "missing", verbose=False)

    def test_empty_after_filter_raises(self):
        with pytest.raises(ValueError, match="No rows remaining"):
            kde(self._df(), "value", groups=["does_not_exist"], verbose=False)

    def test_fill_sets_fillcolor(self):
        fig = kde(self._df(), "value", groups=["A"], fill=True, verbose=False)
        assert fig.data[0].fill == "tozeroy"

    def test_drops_noise_group(self):
        df = pl.DataFrame(
            {"part_id": ["A"] * 50 + ["noise"] * 50, "value": list(range(100))}
        )
        fig = kde(df, "value", noise_label="noise", verbose=False)
        names = [t.name for t in fig.data]
        assert "noise" not in names


class TestColorscaleHelper:
    def test_sample_colorscale_returns_rgb(self):
        out = plotting._sample_colorscale("Viridis", 0.5)
        assert out.startswith("rgb")
