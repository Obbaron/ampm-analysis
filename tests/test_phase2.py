"""
Tests for ampm.sampling.

We synthesize a known DataFrame (random points + a known hot-spot signal) and
verify each downsampler produces the expected shape and statistics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from ampm.sampling import (
    downsample_grid,
    downsample_random,
    downsample_stride,
    prepare_for_plot,
)

def make_synthetic(n: int = 1_000_000, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    y = rng.uniform(0, 10, n)
    z = rng.uniform(0, 10, n)
    signal = rng.normal(10.0, 1.0, n)

    # Plant ~50 hot-spot points near (5, 5, 5).
    hot_idx = rng.choice(n, size=50, replace=False)
    x[hot_idx] = 5 + rng.normal(0, 0.05, 50)
    y[hot_idx] = 5 + rng.normal(0, 0.05, 50)
    z[hot_idx] = 5 + rng.normal(0, 0.05, 50)
    signal[hot_idx] = 1000.0

    return pl.DataFrame({
        "Demand X": x,
        "Demand Y": y,
        "Z": z,
        "signal": signal,
    })

def test_random() -> None:
    df = make_synthetic(10_000)
    out = downsample_random(df, 1000, seed=42)
    assert out.height == 1000, out.height
    assert set(out.columns) == set(df.columns)
    out2 = downsample_random(df, 1000, seed=42)
    assert out.equals(out2)
    small = downsample_random(df.head(50), 1000)
    assert small.height == 50
    try:
        downsample_random(df, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    print("  random OK")

def test_stride() -> None:
    df = make_synthetic(10_000)
    out = downsample_stride(df, 1000)
    assert out.height == 1000, out.height
    assert out["Demand X"][0] == df["Demand X"][0]
    assert out["Demand X"][1] == df["Demand X"][10]
    print("  stride OK")

def test_grid_max_preserves_peak() -> None:
    df = make_synthetic(100_000, seed=1)
    out = downsample_grid(df, 5_000, method="max")
    assert out["signal"].max() >= 999.0, out["signal"].max()
    assert 1_000 <= out.height <= 10_000, out.height
    assert out["Demand X"].min() >= df["Demand X"].min() - 1e-9
    assert out["Demand X"].max() <= df["Demand X"].max() + 1e-9
    print(f"  grid+max: {out.height} voxels, peak preserved")

def test_grid_mean_smooths_peak() -> None:
    df = make_synthetic(100_000, seed=1)
    out_max = downsample_grid(df, 5_000, method="max")
    out_mean = downsample_grid(df, 5_000, method="mean")
    assert out_mean["signal"].max() < out_max["signal"].max()
    print(f"  grid+mean: peak smoothed from {out_max['signal'].max():.0f} to {out_mean['signal'].max():.0f}")

def test_grid_median() -> None:
    df = make_synthetic(100_000, seed=1)
    out = downsample_grid(df, 5_000, method="median")
    assert out["signal"].median() < 50, out["signal"].median()
    print("  grid+median OK")

def test_grid_explicit_agg_columns() -> None:
    df = make_synthetic(50_000)
    out = downsample_grid(df, 1000, agg_columns=["signal"], method="max")
    assert set(out.columns) == {"Demand X", "Demand Y", "Z", "signal"}
    print("  grid + explicit agg_columns OK")

def test_grid_unknown_column_raises() -> None:
    df = make_synthetic(1_000)
    try:
        downsample_grid(df, 100, agg_columns=["bogus"])
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
    print("  grid + unknown col raises OK")

def test_prepare_for_plot_dispatch() -> None:
    df = make_synthetic(20_000)
    r1 = prepare_for_plot(df, 1000, method="random", seed=1)
    r2 = prepare_for_plot(df, 1000, method="stride")
    r3 = prepare_for_plot(df, 1000, method="grid", grid_method="max")
    assert r1.height == 1000
    assert r2.height == 1000
    assert r3.height > 0
    try:
        prepare_for_plot(df, 1000, method="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    print("  prepare_for_plot dispatch OK")

def test_grid_group_by_basic() -> None:
    """downsample_grid should respect group_by, downsampling each group
    independently and preserving the group_by column."""
    rng = np.random.default_rng(0)
    rows = []
    for layer in (1, 2, 3):
        for _ in range(2000):
            rows.append({
                "Demand X": rng.uniform(-10, 10),
                "Demand Y": rng.uniform(-10, 10),
                "Z": float(layer),
                "signal": rng.uniform(0, 100),
                "layer": layer,
            })
    df = pl.DataFrame(rows)
    result = downsample_grid(df, 100, group_by="layer")
    assert "layer" in result.columns
    # Each layer should be downsampled near the target (allow 30% slack
    # for grid rounding — actual count is bins-with-data, not exactly n).
    for layer in (1, 2, 3):
        n = result.filter(pl.col("layer") == layer).height
        assert 0 < n <= 200, f"layer {layer} had {n} rows"
    # All three layers should be present.
    assert set(result["layer"].unique().to_list()) == {1, 2, 3}
    print("  grid group_by basic OK")

def test_grid_group_by_preserves_max() -> None:
    """With method='max', the per-layer maximum should survive."""
    rng = np.random.default_rng(0)
    rows = []
    for layer in (1, 2):
        for _ in range(1000):
            rows.append({
                "Demand X": rng.uniform(-10, 10),
                "Demand Y": rng.uniform(-10, 10),
                "Z": float(layer),
                "signal": rng.uniform(0, 100),
                "layer": layer,
            })
    # Plant a known peak in each layer.
    rows.append({"Demand X": 0, "Demand Y": 0, "Z": 1.0, "signal": 9999.0, "layer": 1})
    rows.append({"Demand X": 0, "Demand Y": 0, "Z": 2.0, "signal": 8888.0, "layer": 2})
    df = pl.DataFrame(rows)
    result = downsample_grid(df, 50, group_by="layer", method="max")
    layer1_max = result.filter(pl.col("layer") == 1)["signal"].max()
    layer2_max = result.filter(pl.col("layer") == 2)["signal"].max()
    assert layer1_max == 9999.0, f"Expected 9999, got {layer1_max}"
    assert layer2_max == 8888.0, f"Expected 8888, got {layer2_max}"
    print("  grid group_by preserves max per group OK")

def test_grid_group_by_unknown_column_raises() -> None:
    df = make_synthetic(100)
    try:
        downsample_grid(df, 50, group_by="bogus")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
    print("  grid group_by unknown raises OK")

def main() -> None:
    print("Phase 2 sampling tests:")
    test_random()
    test_stride()
    test_grid_max_preserves_peak()
    test_grid_mean_smooths_peak()
    test_grid_median()
    test_grid_explicit_agg_columns()
    test_grid_unknown_column_raises()
    test_prepare_for_plot_dispatch()
    test_grid_group_by_basic()
    test_grid_group_by_preserves_max()
    test_grid_group_by_unknown_column_raises()
    print("\nAll Phase 2 tests passed")

if __name__ == "__main__":
    main()
