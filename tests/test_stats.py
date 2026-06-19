"""
Tests for ``stats.py`` — per-group coefficient of variation in three modes.

CoV is std / |mean| with polars' default sample std (ddof=1), so the
values [1, 2, 3] give std 1.0, mean 2.0, CoV 0.5
"""

from __future__ import annotations

import polars as pl
import pytest

from ampm.stats import compute_cov


def cov_df(rows, *, signal="s", group="part_id", layer="layer"):
    """rows: list of (group_value, layer_value, signal_value)."""
    return pl.DataFrame(
        {
            group: [r[0] for r in rows],
            layer: pl.Series([r[1] for r in rows], dtype=pl.Int32),
            signal: pl.Series([float(r[2]) for r in rows], dtype=pl.Float64),
        }
    )


class TestValidation:
    def test_unknown_mode_raises(self):
        df = cov_df([("A", 1, 1.0)])
        with pytest.raises(ValueError, match="unknown mode"):
            compute_cov(df, ["s"], mode="bogus")  # type: ignore

    def test_missing_group_by_raises(self):
        df = cov_df([("A", 1, 1.0)])
        with pytest.raises(KeyError, match="group_by"):
            compute_cov(df, ["s"], group_by="nope")

    def test_empty_columns_raises(self):
        df = cov_df([("A", 1, 1.0)])
        with pytest.raises(ValueError, match="must not be empty"):
            compute_cov(df, [])

    def test_missing_signal_column_raises(self):
        df = cov_df([("A", 1, 1.0)])
        with pytest.raises(KeyError, match="not in DataFrame"):
            compute_cov(df, ["missing"])

    def test_layer_mode_without_layer_col_raises(self):
        df = pl.DataFrame({"part_id": ["A"], "s": [1.0]})  # no layer column
        with pytest.raises(KeyError, match="requires layer_col"):
            compute_cov(df, ["s"], mode="per_layer_mean")


class TestOverall:
    def test_basic_cov(self):
        df = cov_df(
            [
                ("A", 1, 1.0),
                ("A", 1, 2.0),
                ("A", 1, 3.0),
                ("B", 1, 2.0),
                ("B", 1, 2.0),
                ("B", 1, 2.0),
            ]
        )
        out = compute_cov(df, ["s"]).sort("part_id")
        assert out["part_id"].to_list() == ["A", "B"]
        assert out["n_rows"].to_list() == [3, 3]
        assert out["cov_s"].to_list() == pytest.approx([0.5, 0.0])

    def test_output_columns_and_sort(self):
        df = cov_df([("B", 1, 1.0), ("A", 1, 1.0)])
        out = compute_cov(df, ["s"])
        assert out.columns == ["part_id", "n_rows", "cov_s"]
        assert out["part_id"].to_list() == ["A", "B"]  # sorted

    def test_multiple_columns(self):
        df = pl.DataFrame(
            {"part_id": ["A", "A", "A"], "s1": [1.0, 2.0, 3.0], "s2": [2.0, 2.0, 2.0]}
        )
        out = compute_cov(df, ["s1", "s2"])
        assert "cov_s1" in out.columns and "cov_s2" in out.columns
        assert out["cov_s1"].to_list() == pytest.approx([0.5])
        assert out["cov_s2"].to_list() == pytest.approx([0.0])

    def test_mean_below_eps_is_null(self):
        df = cov_df([("A", 1, 1.0), ("A", 1, -1.0)])  # mean 0
        out = compute_cov(df, ["s"], eps=1e-9)
        assert out["cov_s"].to_list() == [None]


class TestNoiseHandling:
    def test_drops_null_group_by_default(self):
        df = pl.DataFrame({"part_id": ["A", None], "s": [1.0, 99.0]})
        out = compute_cov(df, ["s"])
        assert out["part_id"].to_list() == ["A"]

    def test_drops_named_noise_label(self):
        df = pl.DataFrame({"part_id": ["A", "noise"], "s": [1.0, 99.0]})
        out = compute_cov(df, ["s"], noise_label="noise")
        assert out["part_id"].to_list() == ["A"]

    def test_all_noise_returns_empty_with_columns(self):
        df = pl.DataFrame({"part_id": [None, None], "s": [1.0, 2.0]})
        out = compute_cov(df, ["s"])
        assert out.height == 0
        assert out.columns == ["part_id", "n_rows", "cov_s"]


class TestPerLayerMean:
    def test_averages_within_layer_cov_across_layers(self):
        # Layer 1: [1,2,3] -> cov 0.5 ; Layer 2: [2,2,2] -> cov 0.0
        # per_layer_mean = mean(0.5, 0.0) = 0.25 ; n_rows = 2 layers.
        df = cov_df(
            [
                ("A", 1, 1.0),
                ("A", 1, 2.0),
                ("A", 1, 3.0),
                ("A", 2, 2.0),
                ("A", 2, 2.0),
                ("A", 2, 2.0),
            ]
        )
        out = compute_cov(df, ["s"], mode="per_layer_mean")
        assert out["cov_s"].to_list() == pytest.approx([0.25])
        assert out["n_rows"].to_list() == [2]


class TestAcrossLayers:
    def test_cov_of_per_layer_means(self):
        # Layer means: layer1 mean=2, layer2 mean=5 -> means [2,5].
        # std(ddof1)=sqrt(4.5)=2.1213..., mean=3.5 -> cov=0.60609...
        df = cov_df(
            [
                ("A", 1, 1.0),
                ("A", 1, 2.0),
                ("A", 1, 3.0),
                ("A", 2, 4.0),
                ("A", 2, 5.0),
                ("A", 2, 6.0),
            ]
        )
        out = compute_cov(df, ["s"], mode="across_layers")
        assert out["cov_s"].to_list() == pytest.approx([2.1213203 / 3.5], rel=1e-4)
        assert out["n_rows"].to_list() == [2]
