"""
Tests for ``cluster_cache.py`` — persistence of DBSCAN cluster labels keyed by
``(layer, Start time)``.
"""

from __future__ import annotations

import polars as pl
import pytest

from ampm.cluster_cache import (
    CACHE_FORMAT_VERSION,
    _format_param_diff,
    cluster_or_load,
    load_cluster_labels,
    save_cluster_labels,
)

PARAMS = {"eps_xy": 0.3, "eps_z": 0.06, "min_samples": 10, "mode": "3d"}


class TestSave:
    def test_writes_file_with_only_key_and_label_columns(self, keyed_df, tmp_path):
        clustered = keyed_df(
            [(1, 100), (1, 101), (2, 100)],
            cluster=[0, 0, 1],
            extra={"Demand X": [1.0, 2.0, 3.0]},  # should NOT be persisted
        )
        path = tmp_path / "c.pq"
        save_cluster_labels(clustered, path, params=PARAMS, verbose=False)
        assert path.is_file()
        stored = pl.read_parquet(path)
        assert stored.columns == ["layer", "Start time", "cluster"]
        assert stored.height == 3

    def test_missing_label_column_raises_keyerror(self, keyed_df, tmp_path):
        df = keyed_df([(1, 100)])  # no 'cluster'
        with pytest.raises(KeyError, match="cluster"):
            save_cluster_labels(df, tmp_path / "c.pq", verbose=False)

    def test_missing_key_column_raises_keyerror(self, tmp_path):
        df = pl.DataFrame({"layer": [1], "cluster": [0]})  # no 'Start time'
        with pytest.raises(KeyError, match="Start time"):
            save_cluster_labels(df, tmp_path / "c.pq", verbose=False)

    def test_non_unique_keys_raises_valueerror(self, keyed_df, tmp_path):
        df = keyed_df([(1, 100), (1, 100)], cluster=[0, 1])  # duplicate key
        with pytest.raises(ValueError, match="not unique"):
            save_cluster_labels(df, tmp_path / "c.pq", verbose=False)

    def test_non_json_serializable_params_raises_typeerror(self, keyed_df, tmp_path):
        df = keyed_df([(1, 100)], cluster=[0])
        bad_params = {("tuple", "key"): "value"}  # non-string key -> not JSON
        with pytest.raises(TypeError, match="JSON-serializable"):
            save_cluster_labels(df, tmp_path / "c.pq", params=bad_params, verbose=False)

    def test_path_objects_in_params_are_stringified(self, keyed_df, tmp_path):
        df = keyed_df([(1, 100)], cluster=[0])
        # default=str should let Path values through without error.
        save_cluster_labels(
            df, tmp_path / "c.pq", params={"stl": tmp_path / "x.stl"}, verbose=False
        )
        assert (tmp_path / "c.pq").is_file()


class TestLoad:
    def test_roundtrip_labels_match(self, keyed_df, tmp_path):
        clustered = keyed_df([(1, 100), (1, 101), (2, 100)], cluster=[5, 5, 7])
        path = tmp_path / "c.pq"
        save_cluster_labels(clustered, path, params=PARAMS, verbose=False)

        df = keyed_df([(1, 100), (1, 101), (2, 100)])
        out = load_cluster_labels(df, path, expect_params=PARAMS, verbose=False)
        assert out["cluster"].to_list() == [5, 5, 7]
        assert out.schema["cluster"] == pl.Int32

    def test_unmatched_rows_get_minus_one(self, keyed_df, tmp_path):
        clustered = keyed_df([(1, 100)], cluster=[3])
        path = tmp_path / "c.pq"
        save_cluster_labels(clustered, path, verbose=False)

        df = keyed_df([(1, 100), (1, 999), (2, 100)])  # last two not in cache
        out = load_cluster_labels(df, path, verbose=False)
        assert out["cluster"].to_list() == [3, -1, -1]

    def test_key_dtype_mismatch_is_cast(self, keyed_df, tmp_path):
        # Cache written with Int16 layer; df has Int64 layer -> must still join.
        clustered = keyed_df([(1, 100)], cluster=[9], layer_dtype=pl.Int16)
        path = tmp_path / "c.pq"
        save_cluster_labels(clustered, path, verbose=False)

        df = keyed_df([(1, 100)], layer_dtype=pl.Int64)
        out = load_cluster_labels(df, path, verbose=False)
        assert out["cluster"].to_list() == [9]

    def test_missing_file_raises_filenotfound_strict_and_non_strict(
        self, keyed_df, tmp_path
    ):
        df = keyed_df([(1, 100)])
        missing = tmp_path / "nope.pq"
        with pytest.raises(FileNotFoundError):
            load_cluster_labels(df, missing, strict=True, verbose=False)
        with pytest.raises(FileNotFoundError):
            load_cluster_labels(df, missing, strict=False, verbose=False)

    def test_missing_key_column_raises_keyerror(self, keyed_df, tmp_path):
        clustered = keyed_df([(1, 100)], cluster=[0])
        path = tmp_path / "c.pq"
        save_cluster_labels(clustered, path, verbose=False)
        df = pl.DataFrame({"layer": [1]})  # no 'Start time'
        with pytest.raises(KeyError, match="Start time"):
            load_cluster_labels(df, path, verbose=False)

    def test_no_version_metadata(self, keyed_df, tmp_path):
        # A plain Polars parquet has none of the ampm metadata keys.
        path = tmp_path / "plain.pq"
        keyed_df([(1, 100)], cluster=[0]).write_parquet(path)
        df = keyed_df([(1, 100)])
        with pytest.raises(ValueError, match="no version metadata"):
            load_cluster_labels(df, path, strict=True, verbose=False)
        with pytest.raises(FileNotFoundError):
            load_cluster_labels(df, path, strict=False, verbose=False)

    def test_version_mismatch(self, keyed_df, tmp_path, monkeypatch):
        clustered = keyed_df([(1, 100)], cluster=[0])
        path = tmp_path / "c.pq"
        save_cluster_labels(clustered, path, verbose=False)
        # Bump the expected version after writing the cache.
        monkeypatch.setattr(
            "ampm.cluster_cache.CACHE_FORMAT_VERSION", CACHE_FORMAT_VERSION + 1
        )
        df = keyed_df([(1, 100)])
        with pytest.raises(ValueError, match="version"):
            load_cluster_labels(df, path, strict=True, verbose=False)
        with pytest.raises(FileNotFoundError):
            load_cluster_labels(df, path, strict=False, verbose=False)

    def test_params_mismatch(self, keyed_df, tmp_path):
        clustered = keyed_df([(1, 100)], cluster=[0])
        path = tmp_path / "c.pq"
        save_cluster_labels(clustered, path, params=PARAMS, verbose=False)
        df = keyed_df([(1, 100)])
        other = {**PARAMS, "eps_xy": 0.99}
        with pytest.raises(ValueError, match="params"):
            load_cluster_labels(
                df, path, expect_params=other, strict=True, verbose=False
            )
        with pytest.raises(FileNotFoundError):
            load_cluster_labels(
                df, path, expect_params=other, strict=False, verbose=False
            )


class TestClusterOrLoad:
    def test_computes_and_saves_on_cache_miss(self, keyed_df, tmp_path):
        df = keyed_df([(1, 100), (1, 101)])
        path = tmp_path / "c.pq"
        calls = []

        def cluster_fn(d):
            calls.append(1)
            return d.with_columns(pl.Series("cluster", [0, 1], dtype=pl.Int32))

        out = cluster_or_load(df, path, cluster_fn, params=PARAMS, verbose=False)
        assert calls == [1]  # computed
        assert out["cluster"].to_list() == [0, 1]
        assert path.is_file()  # and saved

    def test_uses_cache_on_hit_without_recomputing(self, keyed_df, tmp_path):
        df = keyed_df([(1, 100), (1, 101)])
        path = tmp_path / "c.pq"
        save_cluster_labels(
            keyed_df([(1, 100), (1, 101)], cluster=[2, 3]),
            path,
            params=PARAMS,
            verbose=False,
        )

        def cluster_fn(d):
            raise AssertionError("cluster_fn must not run on a cache hit")

        out = cluster_or_load(df, path, cluster_fn, params=PARAMS, verbose=False)
        assert out["cluster"].to_list() == [2, 3]

    def test_param_mismatch_recomputes_when_not_strict(self, keyed_df, tmp_path):
        df = keyed_df([(1, 100)])
        path = tmp_path / "c.pq"
        save_cluster_labels(
            keyed_df([(1, 100)], cluster=[2]), path, params=PARAMS, verbose=False
        )
        calls = []

        def cluster_fn(d):
            calls.append(1)
            return d.with_columns(pl.Series("cluster", [42], dtype=pl.Int32))

        out = cluster_or_load(
            df,
            path,
            cluster_fn,
            params={**PARAMS, "eps_xy": 9.9},
            strict=False,
            verbose=False,
        )
        assert calls == [1]
        assert out["cluster"].to_list() == [42]

    def test_param_mismatch_raises_when_strict(self, keyed_df, tmp_path):
        df = keyed_df([(1, 100)])
        path = tmp_path / "c.pq"
        save_cluster_labels(
            keyed_df([(1, 100)], cluster=[2]), path, params=PARAMS, verbose=False
        )

        def cluster_fn(d):
            raise AssertionError("should not recompute under strict mismatch")

        with pytest.raises(ValueError):
            cluster_or_load(
                df,
                path,
                cluster_fn,
                params={**PARAMS, "eps_xy": 9.9},
                strict=True,
                verbose=False,
            )


class TestParamDiff:
    def test_none_cache_reports_missing_metadata(self):
        text = _format_param_diff(None, {"a": 1})
        assert "no params" in text

    def test_reports_field_level_difference(self):
        text = _format_param_diff({"a": 1, "b": 2}, {"a": 1, "b": 3})
        assert "b" in text and "2" in text and "3" in text
        assert "a:" not in text  # unchanged key 'a' is not listed

    def test_no_difference_message(self):
        text = _format_param_diff({"a": 1}, {"a": 1})
        assert "no field-level differences" in text
