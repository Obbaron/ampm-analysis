"""
Tests for ampm.cluster_cache.

Verify save/load round-tripping, params-based invalidation, partial-row
matching, and dtype tolerance for the (layer, Start time) key.
"""
from __future__ import annotations

import sys

import shutil
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from ampm.cluster_cache import (
    cluster_or_load,
    load_cluster_labels,
    save_cluster_labels,
)

def make_df(n: int = 1000, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    # Make (layer, Start time) globally unique. Distribute n rows across
    # layers of 100 rows each (or one layer if n < 100).
    rows_per_layer = min(100, n)
    n_layers = max(1, (n + rows_per_layer - 1) // rows_per_layer)
    layer = np.repeat(np.arange(1, n_layers + 1, dtype=np.int16), rows_per_layer)[:n]
    start_time = np.tile(np.arange(rows_per_layer, dtype=np.int32) * 70, n_layers)[:n]
    return pl.DataFrame({
        "layer": layer,
        "Start time": start_time,
        "Demand X": rng.uniform(-30, 30, n).astype(np.float32),
        "Demand Y": rng.uniform(-30, 30, n).astype(np.float32),
        "Z": rng.uniform(0, 6, n).astype(np.float32),
    })

def make_clustered(n: int = 1000, seed: int = 0) -> pl.DataFrame:
    df = make_df(n, seed)
    # Assign labels deterministically: 0 if X<0, 1 otherwise.
    cluster = np.where(df["Demand X"].to_numpy() < 0, 0, 1).astype(np.int32)
    return df.with_columns(pl.Series("cluster", cluster))

def test_round_trip_basic() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="cache_test_"))
    try:
        clustered = make_clustered(500)
        cache = tmp / "labels.pq"
        save_cluster_labels(clustered, cache, params={"a": 1}, verbose=False)
        assert cache.is_file()

        # Reload by joining onto the same df shorn of its labels.
        df = clustered.drop("cluster")
        loaded = load_cluster_labels(df, cache, expect_params={"a": 1}, verbose=False)
        assert loaded.shape == (500, 6)  # original 5 cols + cluster
        assert loaded["cluster"].to_list() == clustered["cluster"].to_list()
        print("  round-trip basic OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_params_mismatch_strict_raises() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="cache_test_"))
    try:
        clustered = make_clustered(200)
        cache = tmp / "labels.pq"
        save_cluster_labels(clustered, cache, params={"eps_xy": 0.3}, verbose=False)

        df = clustered.drop("cluster")
        try:
            load_cluster_labels(df, cache, expect_params={"eps_xy": 0.6}, verbose=False)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError on params mismatch")
        print("  params mismatch (strict) raises OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_params_mismatch_lax_falls_through() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="cache_test_"))
    try:
        clustered = make_clustered(200)
        cache = tmp / "labels.pq"
        save_cluster_labels(clustered, cache, params={"eps_xy": 0.3}, verbose=False)

        df = clustered.drop("cluster")
        try:
            load_cluster_labels(
                df, cache,
                expect_params={"eps_xy": 0.6},
                strict=False,
                verbose=False,
            )
        except FileNotFoundError:
            pass  # expected — caller should fall back to recompute
        else:
            raise AssertionError("expected FileNotFoundError under strict=False mismatch")
        print("  params mismatch (lax) falls through OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_missing_file() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="cache_test_"))
    try:
        df = make_df(100)
        try:
            load_cluster_labels(df, tmp / "nope.pq", verbose=False)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("expected FileNotFoundError")
        print("  missing file raises OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_partial_match_unmatched_rows_get_neg1() -> None:
    """Cache has 100 rows; new df has 200 rows (100 in cache + 100 new).
    The 100 new rows should get cluster=-1."""
    tmp = Path(tempfile.mkdtemp(prefix="cache_test_"))
    try:
        clustered = make_clustered(100)
        cache = tmp / "labels.pq"
        save_cluster_labels(clustered, cache, verbose=False)

        # Build a bigger df: same first 100 rows + 100 new rows with different keys.
        extra = pl.DataFrame({
            "layer": np.full(100, 999, dtype=np.int16),
            "Start time": np.arange(100, dtype=np.int32),
            "Demand X": np.zeros(100, dtype=np.float32),
            "Demand Y": np.zeros(100, dtype=np.float32),
            "Z": np.zeros(100, dtype=np.float32),
        })
        df = pl.concat([clustered.drop("cluster"), extra])
        out = load_cluster_labels(df, cache, verbose=False)
        labels = out["cluster"].to_numpy()
        assert (labels[:100] == clustered["cluster"].to_numpy()).all()
        assert (labels[100:] == -1).all()
        print("  partial match: unmatched rows get -1 OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_uniqueness_check_at_save() -> None:
    """save should refuse if (layer, Start time) is not unique."""
    tmp = Path(tempfile.mkdtemp(prefix="cache_test_"))
    try:
        # Two rows with identical keys.
        bad = pl.DataFrame({
            "layer": [1, 1],
            "Start time": [100, 100],
            "cluster": [0, 1],
        })
        try:
            save_cluster_labels(bad, tmp / "labels.pq", verbose=False)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError on duplicate keys")
        print("  uniqueness check at save OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_cluster_or_load_first_call_computes() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="cache_test_"))
    try:
        df = make_df(300)
        cache = tmp / "labels.pq"
        call_count = {"n": 0}
        def fn(d):
            call_count["n"] += 1
            return d.with_columns(
                pl.Series("cluster", np.zeros(d.height, dtype=np.int32))
            )

        out = cluster_or_load(df, cache, fn, params={"k": 1}, verbose=False)
        assert call_count["n"] == 1
        assert "cluster" in out.columns
        assert cache.is_file()
        print("  cluster_or_load first call computes & saves OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_cluster_or_load_second_call_loads() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="cache_test_"))
    try:
        df = make_df(300)
        cache = tmp / "labels.pq"
        call_count = {"n": 0}
        def fn(d):
            call_count["n"] += 1
            return d.with_columns(
                pl.Series("cluster", np.full(d.height, 7, dtype=np.int32))
            )

        cluster_or_load(df, cache, fn, params={"k": 1}, verbose=False)
        out = cluster_or_load(df, cache, fn, params={"k": 1}, verbose=False)
        # Second call should NOT have invoked fn.
        assert call_count["n"] == 1, f"fn called {call_count['n']} times, expected 1"
        # And should still have the labels we stored.
        assert (out["cluster"].to_numpy() == 7).all()
        print("  cluster_or_load second call loads (no recompute) OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_cluster_or_load_param_change_recomputes() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="cache_test_"))
    try:
        df = make_df(300)
        cache = tmp / "labels.pq"
        call_count = {"n": 0}
        def fn(d):
            call_count["n"] += 1
            return d.with_columns(
                pl.Series("cluster", np.zeros(d.height, dtype=np.int32))
            )

        cluster_or_load(df, cache, fn, params={"eps": 0.3}, verbose=False)
        # Change params with strict=False — should recompute.
        cluster_or_load(df, cache, fn, params={"eps": 0.6},
                        strict=False, verbose=False)
        assert call_count["n"] == 2
        # And the cache now reflects the new params.
        cluster_or_load(df, cache, fn, params={"eps": 0.6}, verbose=False)
        assert call_count["n"] == 2  # third call: hit
        print("  cluster_or_load param-change triggers recompute OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_dtype_tolerant_join() -> None:
    """If df's key columns have a different dtype than the cache, join still works."""
    tmp = Path(tempfile.mkdtemp(prefix="cache_test_"))
    try:
        clustered = make_clustered(100)
        cache = tmp / "labels.pq"
        save_cluster_labels(clustered, cache, verbose=False)

        # Cast df's key columns to a different (but compatible) dtype.
        df = clustered.drop("cluster").with_columns(
            pl.col("layer").cast(pl.Int32),
            pl.col("Start time").cast(pl.Int64),
        )
        out = load_cluster_labels(df, cache, verbose=False)
        # All rows should still be matched.
        assert (out["cluster"] != -1).sum() == 100
        print("  dtype-tolerant join OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_non_serializable_params_raises() -> None:
    """Passing a Path object should still work because we use default=str."""
    tmp = Path(tempfile.mkdtemp(prefix="cache_test_"))
    try:
        clustered = make_clustered(50)
        cache = tmp / "labels.pq"
        # Path is not JSON-serializable by default, but our default=str handles it.
        save_cluster_labels(
            clustered, cache,
            params={"stl": Path("/some/path.stl")},
            verbose=False,
        )
        assert cache.is_file()
        print("  Path objects in params handled via default=str OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_bracketed_path_round_trip() -> None:
    """Regression for Windows path containing '[' and ']' (glob metacharacters)."""
    parent = Path(tempfile.mkdtemp(prefix="cache_brackets_"))
    bracketed = parent / "[3] Export Packets"
    bracketed.mkdir()
    try:
        clustered = make_clustered(200)
        cache = bracketed / "labels.pq"
        save_cluster_labels(clustered, cache, params={"a": 1}, verbose=False)
        df = clustered.drop("cluster")
        loaded = load_cluster_labels(df, cache, expect_params={"a": 1}, verbose=False)
        assert loaded["cluster"].to_list() == clustered["cluster"].to_list()
        print("  bracketed path round-trip OK")
    finally:
        shutil.rmtree(parent, ignore_errors=True)

def main() -> None:
    print("Phase 7 cluster-cache tests:")
    test_round_trip_basic()
    test_params_mismatch_strict_raises()
    test_params_mismatch_lax_falls_through()
    test_missing_file()
    test_partial_match_unmatched_rows_get_neg1()
    test_uniqueness_check_at_save()
    test_cluster_or_load_first_call_computes()
    test_cluster_or_load_second_call_loads()
    test_cluster_or_load_param_change_recomputes()
    test_dtype_tolerant_join()
    test_non_serializable_params_raises()
    test_bracketed_path_round_trip()
    print("\nAll Phase 7 tests passed")

if __name__ == "__main__":
    main()
