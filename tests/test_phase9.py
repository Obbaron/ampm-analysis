"""
Tests for ampm.mask_cache.

Verify that mask survivors round-trip through the cache, that param
mismatches invalidate properly, and that the bracketed-path bug from
Phase 7 doesn't reappear here.
"""
from __future__ import annotations

import sys

import shutil
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from ampm.mask_cache import (
    load_mask_keep,
    mask_or_load,
    save_mask_keep,
)


def make_df(n: int = 1000, seed: int = 0) -> pl.DataFrame:
    """Build a DataFrame with unique (layer, Start time) keys."""
    rng = np.random.default_rng(seed)
    rows_per_layer = min(100, n)
    n_layers = max(1, (n + rows_per_layer - 1) // rows_per_layer)
    layer = np.repeat(np.arange(1, n_layers + 1, dtype=np.int16), rows_per_layer)[:n]
    start_time = np.tile(
        np.arange(rows_per_layer, dtype=np.int32) * 70, n_layers
    )[:n]
    return pl.DataFrame({
        "layer": layer,
        "Start time": start_time,
        "Demand X": rng.uniform(-30, 30, n).astype(np.float32),
        "Demand Y": rng.uniform(-30, 30, n).astype(np.float32),
        "Z": rng.uniform(0, 6, n).astype(np.float32),
    })


def make_masked_df(df: pl.DataFrame, keep_fraction: float = 0.5) -> pl.DataFrame:
    """Drop a deterministic subset of rows so we have a known 'mask survivor' set."""
    n_keep = int(df.height * keep_fraction)
    return df.head(n_keep)


def test_round_trip_basic() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="mask_cache_test_"))
    try:
        df = make_df(500)
        masked = make_masked_df(df, 0.5)
        cache = tmp / "keep.pq"
        save_mask_keep(masked, cache, params={"a": 1}, verbose=False)
        assert cache.is_file()

        loaded = load_mask_keep(df, cache, expect_params={"a": 1}, verbose=False)
        assert loaded.height == masked.height
        # Same rows by key.
        a = loaded.select(["layer", "Start time"]).sort(["layer", "Start time"])
        b = masked.select(["layer", "Start time"]).sort(["layer", "Start time"])
        assert a.equals(b)
        print("  round-trip basic OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_params_mismatch_strict_raises() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="mask_cache_test_"))
    try:
        df = make_df(200)
        masked = make_masked_df(df)
        cache = tmp / "keep.pq"
        save_mask_keep(masked, cache, params={"buffer_mm": 0.0}, verbose=False)
        try:
            load_mask_keep(df, cache, expect_params={"buffer_mm": 0.1}, verbose=False)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError on params mismatch")
        print("  params mismatch (strict) raises OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_params_mismatch_lax_falls_through() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="mask_cache_test_"))
    try:
        df = make_df(200)
        masked = make_masked_df(df)
        cache = tmp / "keep.pq"
        save_mask_keep(masked, cache, params={"buffer_mm": 0.0}, verbose=False)
        try:
            load_mask_keep(
                df, cache,
                expect_params={"buffer_mm": 0.1},
                strict=False, verbose=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("expected FileNotFoundError under strict=False")
        print("  params mismatch (lax) falls through OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_missing_file_raises() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="mask_cache_test_"))
    try:
        df = make_df(100)
        try:
            load_mask_keep(df, tmp / "nope.pq", verbose=False)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("expected FileNotFoundError")
        print("  missing file raises OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_uniqueness_check_at_save() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="mask_cache_test_"))
    try:
        bad = pl.DataFrame({
            "layer": [1, 1],
            "Start time": [100, 100],  # duplicate keys
        })
        try:
            save_mask_keep(bad, tmp / "keep.pq", verbose=False)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError on duplicate keys")
        print("  uniqueness check at save OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_mask_or_load_first_call_computes() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="mask_cache_test_"))
    try:
        df = make_df(300)
        cache = tmp / "keep.pq"
        call_count = {"n": 0}
        def mask_fn(d):
            call_count["n"] += 1
            return d.head(d.height // 2)

        out = mask_or_load(df, cache, mask_fn, params={"k": 1}, verbose=False)
        assert call_count["n"] == 1
        assert out.height == 150
        assert cache.is_file()
        print("  mask_or_load first call computes & saves OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_mask_or_load_second_call_loads() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="mask_cache_test_"))
    try:
        df = make_df(300)
        cache = tmp / "keep.pq"
        call_count = {"n": 0}
        def mask_fn(d):
            call_count["n"] += 1
            return d.head(d.height // 2)

        mask_or_load(df, cache, mask_fn, params={"k": 1}, verbose=False)
        out = mask_or_load(df, cache, mask_fn, params={"k": 1}, verbose=False)
        # Second call should NOT have invoked mask_fn.
        assert call_count["n"] == 1, f"mask_fn called {call_count['n']} times"
        assert out.height == 150
        print("  mask_or_load second call loads OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_mask_or_load_param_change_recomputes() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="mask_cache_test_"))
    try:
        df = make_df(300)
        cache = tmp / "keep.pq"
        call_count = {"n": 0}
        def mask_fn(d):
            call_count["n"] += 1
            return d.head(d.height // 2)

        mask_or_load(df, cache, mask_fn, params={"buffer": 0.0}, verbose=False)
        mask_or_load(
            df, cache, mask_fn,
            params={"buffer": 0.1}, strict=False, verbose=False,
        )
        assert call_count["n"] == 2
        # Third call with the new params should hit the cache.
        mask_or_load(df, cache, mask_fn, params={"buffer": 0.1}, verbose=False)
        assert call_count["n"] == 2
        print("  mask_or_load param-change triggers recompute OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_dtype_tolerant_join() -> None:
    """Different dtypes on the input than what's cached should still join."""
    tmp = Path(tempfile.mkdtemp(prefix="mask_cache_test_"))
    try:
        df = make_df(200)
        masked = make_masked_df(df)
        cache = tmp / "keep.pq"
        save_mask_keep(masked, cache, verbose=False)
        wider = df.with_columns(
            pl.col("layer").cast(pl.Int32),
            pl.col("Start time").cast(pl.Int64),
        )
        out = load_mask_keep(wider, cache, verbose=False)
        assert out.height == masked.height
        print("  dtype-tolerant join OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bracketed_path_round_trip() -> None:
    """Regression for Windows path containing '[' and ']' (glob metacharacters)."""
    parent = Path(tempfile.mkdtemp(prefix="mask_brackets_"))
    bracketed = parent / "[3] Export Packets"
    bracketed.mkdir()
    try:
        df = make_df(200)
        masked = make_masked_df(df)
        cache = bracketed / "keep.pq"
        save_mask_keep(masked, cache, params={"a": 1}, verbose=False)
        out = load_mask_keep(df, cache, expect_params={"a": 1}, verbose=False)
        assert out.height == masked.height
        print("  bracketed path round-trip OK")
    finally:
        shutil.rmtree(parent, ignore_errors=True)


def main() -> None:
    print("Phase 9 mask-cache tests:")
    test_round_trip_basic()
    test_params_mismatch_strict_raises()
    test_params_mismatch_lax_falls_through()
    test_missing_file_raises()
    test_uniqueness_check_at_save()
    test_mask_or_load_first_call_computes()
    test_mask_or_load_second_call_loads()
    test_mask_or_load_param_change_recomputes()
    test_dtype_tolerant_join()
    test_bracketed_path_round_trip()
    print("\nAll Phase 9 tests passed")


if __name__ == "__main__":
    main()
