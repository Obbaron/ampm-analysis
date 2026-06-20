"""
Tests for ``sampling.py`` — random / stride / grid downsamplers and the
``prepare_for_plot`` dispatcher.
"""

from __future__ import annotations

import polars as pl
import pytest

from ampm.sampling import (
    downsample_grid,
    downsample_random,
    downsample_stride,
    prepare_for_plot,
)


def spatial_df(xs, ys=None, zs=None, **extra):
    n = len(xs)
    data = {
        "Demand X": pl.Series([float(v) for v in xs], dtype=pl.Float64),
        "Demand Y": pl.Series(
            [float(v) for v in (ys if ys is not None else [0.0] * n)], dtype=pl.Float64
        ),
        "Z": pl.Series(
            [float(v) for v in (zs if zs is not None else [0.0] * n)], dtype=pl.Float64
        ),
    }
    for k, v in extra.items():
        data[k] = pl.Series(k, list(v))
    return pl.DataFrame(data)


class TestRandom:
    def test_non_positive_n_raises(self):
        with pytest.raises(ValueError, match="positive"):
            downsample_random(spatial_df(range(10)), 0)

    def test_returns_unchanged_when_small_enough(self):
        df = spatial_df(range(5))
        assert downsample_random(df, 10).height == 5

    def test_reduces_to_n_rows(self):
        df = spatial_df(range(100))
        out = downsample_random(df, 10, seed=1)
        assert out.height == 10

    def test_seed_is_reproducible(self):
        df = spatial_df(range(100))
        a = downsample_random(df, 10, seed=7)
        b = downsample_random(df, 10, seed=7)
        assert a.equals(b)


class TestStride:
    def test_non_positive_n_raises(self):
        with pytest.raises(ValueError, match="positive"):
            downsample_stride(spatial_df(range(10)), -1)

    def test_returns_unchanged_when_small_enough(self):
        df = spatial_df(range(5))
        assert downsample_stride(df, 10).height == 5

    def test_takes_every_kth_row_preserving_order(self):
        df = spatial_df(range(100))
        out = downsample_stride(df, 10)  # step = 100 // 10 = 10
        assert out.height == 10
        assert out["Demand X"].to_list() == [float(i) for i in range(0, 100, 10)]


class TestGrid:
    def test_non_positive_n_raises(self):
        with pytest.raises(ValueError, match="positive"):
            downsample_grid(spatial_df(range(10)), 0)

    def test_missing_spatial_column_raises(self):
        df = pl.DataFrame({"Demand X": [1.0], "Demand Y": [2.0]})  # no Z
        with pytest.raises(KeyError, match="Spatial column"):
            downsample_grid(df, 5)

    def test_missing_group_by_column_raises(self):
        df = spatial_df(range(10))
        with pytest.raises(KeyError, match="group_by"):
            downsample_grid(df, 5, group_by="layer")

    def test_returns_unchanged_when_small_enough(self):
        df = spatial_df(range(3), v=[1, 2, 3])
        assert downsample_grid(df, 10).height == 3

    def test_reduces_and_preserves_columns(self):
        df = spatial_df(range(10), v=[float(i) for i in range(10)])
        out = downsample_grid(df, 5, method="max")
        assert out.height < df.height
        assert set(out.columns) == {"Demand X", "Demand Y", "Z", "v"}
        # Helper bin columns must not leak.
        assert not any(c.startswith("__") for c in out.columns)
        # Spatial centroids stay within original bounds; global max preserved.
        x = out["Demand X"].to_numpy()
        assert x.min() >= 0.0 and x.max() <= 9.0
        assert out["v"].max() == 9.0

    def test_explicit_agg_columns_missing_raises(self):
        df = spatial_df(range(10), v=[float(i) for i in range(10)])
        with pytest.raises(KeyError, match="Aggregation column"):
            downsample_grid(df, 5, agg_columns=["nonexistent"])

    def test_aggregation_methods_on_single_voxel(self):
        # All points share one spatial location -> exactly one voxel.
        df = spatial_df([0.0] * 4, ys=[0.0] * 4, zs=[0.0] * 4, v=[1.0, 2.0, 3.0, 4.0])
        assert downsample_grid(df, 1, method="max")["v"].to_list() == [4.0]
        assert downsample_grid(df, 1, method="mean")["v"].to_list() == [2.5]
        assert downsample_grid(df, 1, method="median")["v"].to_list() == [2.5]
        first = downsample_grid(df, 1, method="first")["v"].to_list()
        assert first[0] in {1.0, 2.0, 3.0, 4.0}

    def test_unknown_method_raises(self):
        df = spatial_df([0.0] * 4, v=[1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="Unknown method"):
            downsample_grid(df, 1, method="bogus")  # type: ignore

    def test_group_by_partitions_independently(self):
        # Two layers, each with enough points to be downsampled.
        df = spatial_df(
            xs=[float(i) for i in range(10)] + [float(i) for i in range(10)],
            ys=[0.0] * 20,
            zs=[0.0] * 20,
            v=[float(i) for i in range(20)],
            layer=[1] * 10 + [2] * 10,
        )
        out = downsample_grid(df, 5, group_by="layer", method="max")
        assert set(out["layer"].unique().to_list()) == {1, 2}
        assert "layer" in out.columns
        assert out.height <= df.height

    def test_group_by_keeps_small_groups_whole(self):
        df = spatial_df(
            xs=[0.0, 1.0, 5.0],
            ys=[0.0, 0.0, 0.0],
            zs=[0.0, 0.0, 0.0],
            v=[1.0, 2.0, 3.0],
            layer=[1, 1, 2],
        )
        # n=5 exceeds every group's size, so all rows survive.
        out = downsample_grid(df, 5, group_by="layer")
        assert out.height == 3

    def test_group_by_equals_z_col(self):
        # group_by == z_col must not drop its own spatial axis. Each group has a
        # single Z value, so the Z axis collapses to one bin per group.
        df = spatial_df(
            xs=[float(i) for i in range(20)] + [0.0, 1.0],
            ys=[0.0] * 22,
            zs=[1.0] * 20 + [2.0, 2.0],  # Z=1 has 20 rows (downsampled), Z=2 has 2
            v=[float(i) for i in range(22)],
        )
        out = downsample_grid(df, 5, group_by="Z", z_col="Z", method="max")
        assert set(out["Z"].unique().to_list()) == {1.0, 2.0}
        # The large Z=1 group is downsampled (fewer than its 20 rows) while its
        # max is preserved; the small Z=2 group is kept whole.
        z1 = out.filter(pl.col("Z") == 1.0)
        assert z1.height < 20
        assert z1["v"].max() == 19.0
        assert out.filter(pl.col("Z") == 2.0).height == 2


class TestPrepareForPlot:
    def test_random_route(self):
        df = spatial_df(range(100))
        out = prepare_for_plot(df, target_points=10, method="random", seed=1)
        assert out.height == 10

    def test_stride_route(self):
        df = spatial_df(range(100))
        out = prepare_for_plot(df, target_points=10, method="stride")
        assert out.height == 10

    def test_grid_route(self):
        df = spatial_df(range(20), v=[float(i) for i in range(20)])
        out = prepare_for_plot(df, target_points=5, method="grid", grid_method="max")
        assert out.height <= df.height
        assert "v" in out.columns

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            prepare_for_plot(spatial_df(range(10)), method="nope")  # type: ignore


class TestGridCoverage:
    def test_full_3d_spread_bins_all_axes(self):
        # Non-degenerate X, Y and Z so the multi-axis voxel maths runs (all
        # three spans contribute to the volume / bin-count calculation).
        import numpy as np

        rng = np.random.default_rng(0)
        n = 2000
        df = spatial_df(
            xs=rng.uniform(0, 10, n).tolist(),
            ys=rng.uniform(0, 10, n).tolist(),
            zs=rng.uniform(0, 5, n).tolist(),
            v=rng.uniform(0, 1, n).tolist(),
        )
        out = downsample_grid(df, 100, method="mean")
        assert out.height < df.height
        x = out["Demand X"].to_numpy()
        y = out["Demand Y"].to_numpy()
        z = out["Z"].to_numpy()
        assert x.min() >= 0.0 and x.max() <= 10.0
        assert y.min() >= 0.0 and y.max() <= 10.0
        assert z.min() >= 0.0 and z.max() <= 5.0

    def test_group_by_all_empty_returns_empty(self):
        # Empty frame + group column -> no partitions -> head(0) branch.
        empty = pl.DataFrame(
            {
                "Demand X": [],
                "Demand Y": [],
                "Z": [],
                "layer": pl.Series([], dtype=pl.Int64),
                "v": [],
            }
        )
        out = downsample_grid(empty, 5, group_by="layer")
        assert out.height == 0
