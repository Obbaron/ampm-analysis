"""
Tests for ``clustering.py`` — anisotropic DBSCAN with label propagation,
chunked clustering with cross-boundary merging, and supporting utilities.

DBSCAN is deterministic on fixed inputs, so these use well-separated synthetic
blobs / lines and assert exact cluster counts and stable label ordering.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ampm.clustering import (
    _stabilize_labels,
    _to_scaled_array,
    _UnionFind,
    _validate,
    cluster_dbscan,
    cluster_dbscan_chunked,
    cluster_summary,
    k_distance_curve,
)

THICKNESS = 0.03


def two_blobs_2d(n_per=80, sep=100.0, spread=0.3, seed=0):
    """Two tight, far-apart 2D blobs. Blob A near origin, blob B near (sep,sep)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, spread, size=(n_per, 2))
    b = rng.normal(sep, spread, size=(n_per, 2))
    pts = np.vstack([a, b])
    return pl.DataFrame({"Demand X": pts[:, 0], "Demand Y": pts[:, 1]})


def vertical_line_3d(layers, per_layer=5, thickness=THICKNESS):
    """A single dense column at (0,0) repeated across layers; Z = layer*thickness."""
    rows = []
    for L in layers:
        for _ in range(per_layer):
            rows.append((0.0, 0.0, L, L * thickness))
    return pl.DataFrame(
        {
            "Demand X": pl.Series([r[0] for r in rows], dtype=pl.Float64),
            "Demand Y": pl.Series([r[1] for r in rows], dtype=pl.Float64),
            "layer": pl.Series([r[2] for r in rows], dtype=pl.Int16),
            "Z": pl.Series([r[3] for r in rows], dtype=pl.Float64),
        }
    )


class TestValidate:
    def test_bad_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            _validate(
                pl.DataFrame({"Demand X": [], "Demand Y": []}),
                "4d",  # type: ignore
                ("Demand X", "Demand Y"),
                1.0,
                None,
            )

    def test_3d_requires_eps_z(self):
        df = pl.DataFrame({"Demand X": [0.0], "Demand Y": [0.0], "Z": [0.0]})
        with pytest.raises(ValueError, match="requires eps_z"):
            _validate(df, "3d", ("Demand X", "Demand Y", "Z"), 1.0, None)

    def test_3d_eps_z_must_be_positive(self):
        df = pl.DataFrame({"Demand X": [0.0], "Demand Y": [0.0], "Z": [0.0]})
        with pytest.raises(ValueError, match="eps_z must be positive"):
            _validate(df, "3d", ("Demand X", "Demand Y", "Z"), 1.0, 0.0)

    def test_3d_needs_three_columns(self):
        df = pl.DataFrame({"Demand X": [0.0], "Demand Y": [0.0]})
        with pytest.raises(ValueError, match="needs 3 columns"):
            _validate(df, "3d", ("Demand X", "Demand Y"), 1.0, 0.1)

    def test_eps_xy_must_be_positive(self):
        df = pl.DataFrame({"Demand X": [0.0], "Demand Y": [0.0]})
        with pytest.raises(ValueError, match="eps_xy must be positive"):
            _validate(df, "2d", ("Demand X", "Demand Y"), 0.0, None)

    def test_missing_columns_raise_keyerror(self):
        df = pl.DataFrame({"Demand X": [0.0]})
        with pytest.raises(KeyError, match="not in DataFrame"):
            _validate(df, "2d", ("Demand X", "Demand Y"), 1.0, None)

    def test_returns_columns_for_each_mode(self):
        df = pl.DataFrame({"Demand X": [0.0], "Demand Y": [0.0], "Z": [0.0]})
        assert _validate(df, "3d", ("Demand X", "Demand Y", "Z"), 1.0, 0.1) == (
            "Demand X",
            "Demand Y",
            "Z",
        )
        assert _validate(df, "2d", ("Demand X", "Demand Y"), 1.0, None) == (
            "Demand X",
            "Demand Y",
            None,
        )


class TestScaledArray:
    def test_2d_shape(self):
        df = pl.DataFrame({"Demand X": [1.0, 2.0], "Demand Y": [3.0, 4.0]})
        arr = _to_scaled_array(df, "Demand X", "Demand Y", None, 1.0, None)
        assert arr.shape == (2, 2)

    def test_3d_scales_z_by_ratio(self):
        df = pl.DataFrame({"Demand X": [0.0], "Demand Y": [0.0], "Z": [2.0]})
        arr = _to_scaled_array(df, "Demand X", "Demand Y", "Z", eps_xy=1.0, eps_z=0.5)
        # z_scaled = z * (eps_xy / eps_z) = 2 * (1/0.5) = 4
        assert arr[0, 2] == pytest.approx(4.0)


class TestStabilizeLabels:
    def test_relabels_by_centroid_x(self):
        # Cluster at high X is initially label 0; should become the higher id.
        coords = np.array([[100.0, 0.0], [101.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        labels = np.array([0, 0, 1, 1])
        out = _stabilize_labels(labels, coords)
        # The low-X cluster (originally label 1) must now be 0.
        assert out[2] == 0 and out[3] == 0
        assert out[0] == 1 and out[1] == 1

    def test_noise_preserved(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        labels = np.array([-1, -1])
        out = _stabilize_labels(labels, coords)
        assert out.tolist() == [-1, -1]


class TestKDistanceCurve:
    def test_empty_df_returns_empty_curve(self):
        df = pl.DataFrame({"Demand X": [], "Demand Y": [], "Z": []})
        out = k_distance_curve(df, k=3, mode="3d", eps_xy=1.0, eps_z=0.1)
        assert out.columns == ["Rank", "k-distance (mm)"]
        assert out.height == 0

    def test_sorted_ascending_with_expected_rows(self):
        df = two_blobs_2d(n_per=15, seed=1)
        out = k_distance_curve(df, k=3, mode="2d", eps_xy=1.0, eps_z=1.0, seed=1)
        vals = out["k-distance (mm)"].to_list()
        assert out.height == df.height
        assert vals == sorted(vals)
        assert out["Rank"].to_list() == list(range(df.height))


class TestClusterDbscan:
    def test_empty_df_returns_empty_with_cluster_column(self):
        df = pl.DataFrame({"Demand X": [], "Demand Y": []})
        out = cluster_dbscan(df, eps_xy=1.0, mode="2d")
        assert "cluster" in out.columns
        assert out.height == 0

    def test_two_blobs_two_clusters(self):
        df = two_blobs_2d(n_per=80, seed=2)
        out = cluster_dbscan(df, eps_xy=5.0, min_samples=5, mode="2d", seed=2)
        labels = set(out["cluster"].to_list())
        assert labels == {0, 1}  # no noise expected for tight blobs

    def test_stable_labels_order_by_x(self):
        df = two_blobs_2d(n_per=60, sep=100.0, seed=3)
        out = cluster_dbscan(df, eps_xy=5.0, min_samples=5, mode="2d", seed=3)
        # First 60 rows are the origin blob -> should be label 0 under stable ordering.
        assert out["cluster"].to_list()[0] == 0
        assert out["cluster"].to_list()[-1] == 1

    def test_propagation_path_labels_all_rows(self):
        df = two_blobs_2d(n_per=150, seed=4)  # 300 rows
        out = cluster_dbscan(
            df, eps_xy=5.0, min_samples=5, mode="2d", representative_size=80, seed=4
        )
        assert out.height == df.height
        assert set(out["cluster"].to_list()) == {0, 1}

    def test_cluster_dtype_is_int64(self):
        df = two_blobs_2d(n_per=20, seed=5)
        out = cluster_dbscan(df, eps_xy=5.0, min_samples=5, mode="2d", seed=5)
        assert out.schema["cluster"] == pl.Int64


class TestClusterSummary:
    def test_missing_cluster_column_raises(self):
        df = pl.DataFrame({"Demand X": [1.0], "Demand Y": [2.0]})
        with pytest.raises(KeyError, match="cluster"):
            cluster_summary(df)

    def test_counts_and_bbox(self):
        df = pl.DataFrame(
            {
                "Demand X": [0.0, 2.0, 10.0, 12.0],
                "Demand Y": [0.0, 2.0, 10.0, 12.0],
                "cluster": [0, 0, 1, 1],
            }
        )
        s = cluster_summary(df, columns=("Demand X", "Demand Y"))
        assert s["cluster"].to_list() == [0, 1]
        assert s["n_rows"].to_list() == [2, 2]
        row0 = s.row(0, named=True)
        assert row0["x_min"] == pytest.approx(0.0)
        assert row0["x_max"] == pytest.approx(2.0)
        assert row0["x_mean"] == pytest.approx(1.0)

    def test_z_columns_present_only_when_z_exists(self):
        with_z = pl.DataFrame(
            {"Demand X": [0.0], "Demand Y": [0.0], "Z": [1.0], "cluster": [0]}
        )
        s = cluster_summary(with_z)
        assert "z_min" in s.columns and "z_mean" in s.columns

        without_z = pl.DataFrame({"Demand X": [0.0], "Demand Y": [0.0], "cluster": [0]})
        s2 = cluster_summary(without_z, columns=("Demand X", "Demand Y"))
        assert "z_min" not in s2.columns


class TestUnionFind:
    def test_add_and_find_self(self):
        uf = _UnionFind()
        uf.add("a")
        assert uf.find("a") == "a"

    def test_union_merges_roots(self):
        uf = _UnionFind()
        for k in ("a", "b", "c"):
            uf.add(k)
        uf.union("a", "b")
        assert uf.find("a") == uf.find("b")
        assert uf.find("c") != uf.find("a")

    def test_components_map_to_roots(self):
        uf = _UnionFind()
        for k in ("a", "b", "c", "d"):
            uf.add(k)
        uf.union("a", "b")
        uf.union("c", "d")
        comps = uf.components()
        assert comps["a"] == comps["b"]
        assert comps["c"] == comps["d"]
        assert comps["a"] != comps["c"]


class TestClusterDbscanChunked:
    def test_empty_df_returns_minus_one_column(self):
        df = pl.DataFrame(
            {
                "Demand X": [],
                "Demand Y": [],
                "Z": [],
                "layer": pl.Series([], dtype=pl.Int16),
            }
        )
        out = cluster_dbscan_chunked(
            df, eps_xy=1.0, eps_z=0.1, layer_col="layer", verbose=False
        )
        assert out["cluster"].to_list() == []
        assert out.schema["cluster"] == pl.Int32

    def test_missing_layer_column_raises(self):
        df = pl.DataFrame({"Demand X": [0.0], "Demand Y": [0.0], "Z": [0.0]})
        with pytest.raises(KeyError, match="layer_col"):
            cluster_dbscan_chunked(
                df, eps_xy=1.0, eps_z=0.1, layer_col="layer", verbose=False
            )

    def test_overlap_not_less_than_chunk_raises(self):
        df = vertical_line_3d(range(1, 11))
        with pytest.raises(ValueError, match="must be <"):
            cluster_dbscan_chunked(
                df,
                eps_xy=1.0,
                eps_z=0.06,
                layers_per_chunk=4,
                overlap_layers=4,
                layer_thickness=THICKNESS,
                verbose=False,
            )

    def test_single_cluster_merged_across_chunks(self):
        df = vertical_line_3d(range(1, 21), per_layer=6)
        out = cluster_dbscan_chunked(
            df,
            eps_xy=1.0,
            eps_z=0.06,
            min_samples=5,
            layers_per_chunk=8,
            layer_thickness=THICKNESS,
            verbose=False,
        )
        labels = set(out["cluster"].to_list())
        assert labels == {0}  # one cluster, no noise

    def test_result_dtype_is_int32(self):
        df = vertical_line_3d(range(1, 11), per_layer=6)
        out = cluster_dbscan_chunked(
            df,
            eps_xy=1.0,
            eps_z=0.06,
            min_samples=5,
            layers_per_chunk=8,
            layer_thickness=THICKNESS,
            verbose=False,
        )
        assert out.schema["cluster"] == pl.Int32

    def test_below_minimum_overlap_is_clamped_not_fatal(self):
        df = vertical_line_3d(range(1, 21), per_layer=6)
        out = cluster_dbscan_chunked(
            df,
            eps_xy=1.0,
            eps_z=0.06,
            min_samples=5,
            layers_per_chunk=10,
            overlap_layers=1,
            layer_thickness=THICKNESS,
            verbose=False,
        )
        assert out.height == df.height
        assert set(out["cluster"].to_list()) == {0}
