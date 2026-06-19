"""
Tests for ``mask_cache.py`` — streaming persistence of mask-keep keys
``(layer, Start time)``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ampm.mask_cache import (
    _META_VERSION_KEY,
    CACHE_FORMAT_VERSION,
    _format_param_diff,
    load_mask_keep,
    mask_or_load,
    save_mask_keep,
    save_mask_keep_from_keep,
)

PARAMS = {"stl": "parts.stl", "buffer_mm": 0.0, "layer_thickness": 0.03}


class TestSave:
    def test_writes_keys_with_metadata(self, keyed_df, tmp_path):
        masked = keyed_df([(1, 10), (1, 11), (2, 10)])
        path = tmp_path / "m.pq"
        save_mask_keep(masked, path, params=PARAMS, verbose=False)
        assert path.is_file()
        stored = pl.read_parquet(path)
        assert sorted(stored.columns) == ["Start time", "layer"]
        assert stored.height == 3

    def test_empty_dataframe_raises(self, keyed_df, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            save_mask_keep(keyed_df([]), tmp_path / "m.pq", verbose=False)

    def test_missing_key_column_raises(self, tmp_path):
        df = pl.DataFrame({"layer": [1, 2]})  # no 'Start time'
        with pytest.raises(KeyError, match="Start time"):
            save_mask_keep(df, tmp_path / "m.pq", verbose=False)

    def test_non_json_params_raises_typeerror(self, keyed_df, tmp_path):
        masked = keyed_df([(1, 10)])
        with pytest.raises(TypeError, match="JSON-serializable"):
            save_mask_keep(masked, tmp_path / "m.pq", params={("k",): 1}, verbose=False)

    def test_duplicate_keys_within_layer_raises(self, keyed_df, tmp_path):
        masked = keyed_df([(1, 10), (1, 10)])  # same key twice, contiguous
        with pytest.raises(ValueError, match="not unique"):
            save_mask_keep(masked, tmp_path / "m.pq", verbose=False)

    def test_from_keep_writes_only_kept_rows(self, keyed_df, tmp_path):
        full = keyed_df([(1, 10), (1, 11), (2, 10)])
        keep = np.array([True, False, True])
        path = tmp_path / "m.pq"
        save_mask_keep_from_keep(full, keep, path, params=PARAMS, verbose=False)
        stored = pl.read_parquet(path).sort(["layer", "Start time"])
        assert list(zip(stored["layer"].to_list(), stored["Start time"].to_list())) == [
            (1, 10),
            (2, 10),
        ]

    def test_from_keep_shape_mismatch_raises(self, keyed_df, tmp_path):
        full = keyed_df([(1, 10), (1, 11)])
        with pytest.raises(ValueError, match="shape"):
            save_mask_keep_from_keep(
                full, np.array([True]), tmp_path / "m.pq", verbose=False
            )

    def test_from_keep_all_false_raises(self, keyed_df, tmp_path):
        full = keyed_df([(1, 10), (1, 11)])
        with pytest.raises(ValueError, match="empty"):
            save_mask_keep_from_keep(
                full, np.array([False, False]), tmp_path / "m.pq", verbose=False
            )

    def test_global_uniqueness_fallback_passes_for_unique_keys(
        self, keyed_df, tmp_path
    ):
        # Non-contiguous layers (1,2,1) force the global uniqueness check;
        # keys are still globally unique, so the save must succeed.
        masked = keyed_df([(1, 10), (2, 10), (1, 11)])
        path = tmp_path / "m.pq"
        save_mask_keep(masked, path, verbose=False)
        assert pl.read_parquet(path).height == 3

    def test_global_uniqueness_fallback_detects_duplicate(self, keyed_df, tmp_path):
        # Non-contiguous AND a true duplicate across the split runs.
        masked = keyed_df([(1, 10), (2, 10), (1, 10)])
        with pytest.raises(ValueError, match="not unique"):
            save_mask_keep(masked, tmp_path / "m.pq", verbose=False)


class TestLoad:
    def _save(self, keyed_df, path, keys, **kw):
        save_mask_keep(keyed_df(keys), path, verbose=False, **kw)

    def test_roundtrip_filters_to_cached_keys(self, keyed_df, tmp_path):
        path = tmp_path / "m.pq"
        self._save(keyed_df, path, [(1, 10), (2, 10)], params=PARAMS)

        full = keyed_df([(1, 10), (1, 11), (2, 10), (3, 10)])
        out = load_mask_keep(full, path, expect_params=PARAMS, verbose=False)
        got = sorted(zip(out["layer"].to_list(), out["Start time"].to_list()))
        assert got == [(1, 10), (2, 10)]

    def test_key_dtype_mismatch_is_cast(self, keyed_df, tmp_path):
        path = tmp_path / "m.pq"
        save_mask_keep(keyed_df([(1, 10)], layer_dtype=pl.Int16), path, verbose=False)
        full = keyed_df([(1, 10), (1, 11)], layer_dtype=pl.Int64)
        out = load_mask_keep(full, path, verbose=False)
        assert out.height == 1
        assert out["layer"].to_list() == [1]

    def test_descending_cache_uses_fallback_path(self, keyed_df, tmp_path):
        # Cache written with layers in descending order trips the non-ascending
        # branch of _keep_from_cached_keys; result must still be correct.
        path = tmp_path / "m.pq"
        save_mask_keep(keyed_df([(3, 10), (2, 10), (1, 10)]), path, verbose=False)
        full = keyed_df([(1, 10), (2, 10), (3, 10), (4, 10)])
        out = load_mask_keep(full, path, verbose=False)
        got = sorted(zip(out["layer"].to_list(), out["Start time"].to_list()))
        assert got == [(1, 10), (2, 10), (3, 10)]

    def test_missing_file_raises(self, keyed_df, tmp_path):
        full = keyed_df([(1, 10)])
        with pytest.raises(FileNotFoundError):
            load_mask_keep(full, tmp_path / "nope.pq", strict=True, verbose=False)
        with pytest.raises(FileNotFoundError):
            load_mask_keep(full, tmp_path / "nope.pq", strict=False, verbose=False)

    def test_missing_key_column_raises(self, keyed_df, tmp_path):
        path = tmp_path / "m.pq"
        self._save(keyed_df, path, [(1, 10)])
        df = pl.DataFrame({"layer": [1]})
        with pytest.raises(KeyError, match="Start time"):
            load_mask_keep(df, path, verbose=False)

    def test_no_version_metadata(self, keyed_df, tmp_path):
        path = tmp_path / "plain.pq"
        keyed_df([(1, 10)]).write_parquet(path)
        full = keyed_df([(1, 10)])
        with pytest.raises(ValueError, match="no version metadata"):
            load_mask_keep(full, path, strict=True, verbose=False)
        with pytest.raises(FileNotFoundError):
            load_mask_keep(full, path, strict=False, verbose=False)

    def test_version_mismatch(self, keyed_df, tmp_path, monkeypatch):
        path = tmp_path / "m.pq"
        self._save(keyed_df, path, [(1, 10)])
        monkeypatch.setattr("ampm.mask_cache.CACHE_FORMAT_VERSION", CACHE_FORMAT_VERSION + 1)
        full = keyed_df([(1, 10)])
        with pytest.raises(ValueError, match="version"):
            load_mask_keep(full, path, strict=True, verbose=False)
        with pytest.raises(FileNotFoundError):
            load_mask_keep(full, path, strict=False, verbose=False)

    def test_params_mismatch(self, keyed_df, tmp_path):
        path = tmp_path / "m.pq"
        self._save(keyed_df, path, [(1, 10)], params=PARAMS)
        full = keyed_df([(1, 10)])
        other = {**PARAMS, "buffer_mm": 0.5}
        with pytest.raises(ValueError, match="params"):
            load_mask_keep(full, path, expect_params=other, strict=True, verbose=False)
        with pytest.raises(FileNotFoundError):
            load_mask_keep(full, path, expect_params=other, strict=False, verbose=False)

    def test_zero_row_cache(self, tmp_path):
        # Craft a 0-row but validly-versioned cache file.
        path = tmp_path / "empty.pq"
        schema = pa.schema(
            [("layer", pa.int16()), ("Start time", pa.int32())],
            metadata={_META_VERSION_KEY: str(CACHE_FORMAT_VERSION).encode()},
        )
        pq.write_table(schema.empty_table(), path)
        full = pl.DataFrame(
            {
                "layer": pl.Series([1], dtype=pl.Int16),
                "Start time": pl.Series([10], dtype=pl.Int32),
            }
        )
        with pytest.raises(ValueError, match="0 keys"):
            load_mask_keep(full, path, strict=True, verbose=False)

    def test_no_matching_rows_raises(self, keyed_df, tmp_path):
        path = tmp_path / "m.pq"
        self._save(keyed_df, path, [(5, 10)])
        full = keyed_df([(1, 10), (2, 10)])
        with pytest.raises(ValueError, match="0 of"):
            load_mask_keep(full, path, strict=True, verbose=False)
        with pytest.raises(FileNotFoundError):
            load_mask_keep(full, path, strict=False, verbose=False)


class TestMaskOrLoad:
    def test_requires_a_callable(self, keyed_df, tmp_path):
        with pytest.raises(TypeError, match="mask_fn or keep_fn"):
            mask_or_load(keyed_df([(1, 10)]), tmp_path / "m.pq", params=PARAMS)

    def test_keep_fn_computes_saves_and_filters(self, keyed_df, tmp_path):
        full = keyed_df([(1, 10), (1, 11), (2, 10)])
        path = tmp_path / "m.pq"
        out = mask_or_load(
            full,
            path,
            keep_fn=lambda d: np.array([True, False, True]),
            params=PARAMS,
            verbose=False,
        )
        assert sorted(zip(out["layer"].to_list(), out["Start time"].to_list())) == [
            (1, 10),
            (2, 10),
        ]
        assert path.is_file()

    def test_cache_hit_does_not_recompute(self, keyed_df, tmp_path):
        full = keyed_df([(1, 10), (1, 11)])
        path = tmp_path / "m.pq"
        save_mask_keep(keyed_df([(1, 10)]), path, params=PARAMS, verbose=False)

        def keep_fn(_):
            raise AssertionError("must not recompute on cache hit")

        out = mask_or_load(full, path, keep_fn=keep_fn, params=PARAMS, verbose=False)
        assert out["Start time"].to_list() == [10]

    def test_keep_fn_none_returns_input(self, keyed_df, tmp_path):
        full = keyed_df([(1, 10)])
        out = mask_or_load(
            full,
            tmp_path / "m.pq",
            keep_fn=lambda d: None,
            params=PARAMS,
            verbose=False,
        )
        assert out.equals(full)

    def test_keep_fn_all_false_raises_runtimeerror(self, keyed_df, tmp_path):
        full = keyed_df([(1, 10), (1, 11)])
        with pytest.raises(RuntimeError, match="kept 0"):
            mask_or_load(
                full,
                tmp_path / "m.pq",
                keep_fn=lambda d: np.zeros(d.height, dtype=bool),
                params=PARAMS,
                verbose=False,
            )

    def test_legacy_mask_fn_path(self, keyed_df, tmp_path):
        full = keyed_df([(1, 10), (1, 11), (2, 10)])
        path = tmp_path / "m.pq"

        def mask_fn(d):
            return d.filter(pl.col("layer") == 1)

        out = mask_or_load(full, path, mask_fn, params=PARAMS, verbose=False)
        assert out["layer"].unique().to_list() == [1]
        assert path.is_file()

    def test_mask_fn_empty_result_raises(self, keyed_df, tmp_path):
        full = keyed_df([(1, 10)])
        with pytest.raises(RuntimeError, match="kept 0"):
            mask_or_load(
                full,
                tmp_path / "m.pq",
                lambda d: d.filter(pl.col("layer") == 999),
                params=PARAMS,
                verbose=False,
            )


class TestParamDiff:
    def test_none_cache(self):
        assert "no params" in _format_param_diff(None, {"a": 1})

    def test_difference_listed(self):
        text = _format_param_diff({"buffer_mm": 0.0}, {"buffer_mm": 0.5})
        assert "buffer_mm" in text

    def test_no_difference(self):
        assert "no field-level differences" in _format_param_diff({"a": 1}, {"a": 1})
