"""
Tests for ampm.stats (compute_cov) and ampm.plotting.bar.

Build small DataFrames with known mean/std and verify CoV across the three
modes. Then verify bar() produces the right Figure structure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))

from ampm.stats import compute_cov
from ampm.plotting import bar

def _make_two_part_data(seed: int = 0) -> pl.DataFrame:
    """
    Two parts. Part A has stable signal (~500 +/- small jitter).
    Part B has noisier signal (~500 +/- bigger jitter).
    Both have 5 layers with 100 points each.
    Plus 50 noise rows with part_id = "noise".
    """
    rng = np.random.default_rng(seed)
    rows = []
    for part_id, sigma in [("A", 1.0), ("B", 50.0)]:
        for layer in range(1, 6):
            for _ in range(100):
                rows.append({
                    "part_id": part_id,
                    "layer": layer,
                    "signal": rng.normal(500, sigma),
                })
    for _ in range(50):
        rows.append({
            "part_id": "noise",
            "layer": 0,
            "signal": rng.uniform(-1000, 1000),
        })
    return pl.DataFrame(rows)

def test_overall_cov_basic() -> None:
    df = _make_two_part_data()
    cov = compute_cov(df, ["signal"], group_by="part_id", noise_label="noise")
    # Expect 2 rows (noise dropped), one per part.
    assert cov.height == 2
    assert "cov_signal" in cov.columns
    rows = {r["part_id"]: r for r in cov.iter_rows(named=True)}
    # B should have a much larger CoV than A (sigma 50 vs 1).
    assert rows["B"]["cov_signal"] > rows["A"]["cov_signal"] * 10
    print("  overall CoV: noisier part has higher CoV OK")

def test_overall_cov_mathematical_correctness() -> None:
    """Hand-calculate CoV and verify."""
    df = pl.DataFrame({
        "part_id": ["A"] * 5,
        "signal": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    cov = compute_cov(df, ["signal"], group_by="part_id", drop_noise=False)
    # mean = 3, std (sample) = sqrt(((1-3)^2 + ... + (5-3)^2) / 4) = sqrt(10/4) ≈ 1.5811
    # CoV = 1.5811 / 3 ≈ 0.5270
    val = cov["cov_signal"][0]
    assert abs(val - 0.5270) < 0.001, val
    print("  overall CoV: matches hand calc OK")

def test_drop_noise_default() -> None:
    df = _make_two_part_data()
    # noise_label="noise" matches what apply_part_id_map writes.
    cov = compute_cov(df, ["signal"], group_by="part_id", noise_label="noise")
    parts = set(cov["part_id"].to_list())
    assert "noise" not in parts
    assert parts == {"A", "B"}
    # And with drop_noise=False, the noise group is included.
    cov2 = compute_cov(df, ["signal"], group_by="part_id",
                       noise_label="noise", drop_noise=False)
    assert "noise" in set(cov2["part_id"].to_list())
    print("  drop_noise default + override OK")

def test_per_layer_mean_mode() -> None:
    """
    Construct data where every layer has tiny intra-layer variation but the
    layer means drift dramatically. per_layer_mean should be small;
    across_layers should be large.
    """
    rng = np.random.default_rng(0)
    rows = []
    for layer in range(1, 11):
        layer_mean = 100 + layer * 50  # drifts from 150 to 600
        for _ in range(100):
            rows.append({
                "part_id": "X",
                "layer": layer,
                "signal": rng.normal(layer_mean, 1.0),  # tiny intra-layer std
            })
    df = pl.DataFrame(rows)

    overall = compute_cov(df, ["signal"], group_by="part_id",
                          mode="overall", drop_noise=False)
    per_layer = compute_cov(df, ["signal"], group_by="part_id",
                            mode="per_layer_mean", drop_noise=False)
    across = compute_cov(df, ["signal"], group_by="part_id",
                         mode="across_layers", drop_noise=False)

    o = overall["cov_signal"][0]
    p = per_layer["cov_signal"][0]
    a = across["cov_signal"][0]

    # per_layer_mean should be tiny (intra-layer std ~1, mean ~150-600).
    assert p < 0.02, f"per_layer_mean CoV {p} should be small"
    # across_layers should reflect the drift (std of layer means / overall mean).
    # Layer means 150..600, std ≈ 150, mean of means ≈ 375 → CoV ≈ 0.4.
    assert 0.3 < a < 0.5, f"across_layers CoV {a} unexpected"
    # overall CoV should be similar to across-layers because the drift
    # dominates the intra-layer noise.
    assert abs(o - a) / a < 0.2, f"overall ({o}) should ~match across_layers ({a})"
    print(f"  three modes: overall={o:.3f}, per_layer={p:.3f}, across={a:.3f} OK")

def test_zero_mean_returns_null() -> None:
    """A column with mean=0 shouldn't blow up; it should yield null CoV."""
    df = pl.DataFrame({
        "part_id": ["A"] * 4,
        "signal": [-2.0, -1.0, 1.0, 2.0],  # mean = 0
    })
    cov = compute_cov(df, ["signal"], group_by="part_id", drop_noise=False)
    assert cov["cov_signal"][0] is None
    print("  zero-mean → null CoV OK")

def test_multiple_columns_in_one_call() -> None:
    df = pl.DataFrame({
        "part_id": ["A", "A", "B", "B"],
        "signal_1": [10.0, 12.0, 100.0, 110.0],
        "signal_2": [1.0, 1.5, 50.0, 80.0],
    })
    cov = compute_cov(
        df, ["signal_1", "signal_2"], group_by="part_id", drop_noise=False,
    )
    assert "cov_signal_1" in cov.columns
    assert "cov_signal_2" in cov.columns
    assert cov.height == 2
    print("  multiple columns OK")

def test_unknown_column_raises() -> None:
    df = pl.DataFrame({"part_id": ["A"], "signal": [1.0]})
    try:
        compute_cov(df, ["bogus"], group_by="part_id", drop_noise=False)
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
    try:
        compute_cov(df, ["signal"], group_by="bogus", drop_noise=False)
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
    print("  unknown column raises OK")

def test_unknown_mode_raises() -> None:
    df = pl.DataFrame({"part_id": ["A"], "signal": [1.0]})
    try:
        compute_cov(df, ["signal"], group_by="part_id",
                    mode="bogus", drop_noise=False)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    print("  unknown mode raises OK")

def test_per_layer_mode_requires_layer_col() -> None:
    df = pl.DataFrame({
        "part_id": ["A", "A"],
        "signal": [1.0, 2.0],
    })
    try:
        compute_cov(df, ["signal"], group_by="part_id",
                    mode="per_layer_mean", drop_noise=False)
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
    print("  per_layer_mean requires layer_col OK")

def test_noise_label_string() -> None:
    """noise_label='noise' should drop only string-labelled noise rows."""
    df = pl.DataFrame({
        "part_id": ["A", "A", "noise", "noise"],
        "signal": [1.0, 2.0, 100.0, 200.0],
    })
    cov = compute_cov(df, ["signal"], group_by="part_id",
                      noise_label="noise")
    assert cov.height == 1
    assert cov["part_id"][0] == "A"
    print("  noise_label string drops correctly OK")

def test_empty_after_drop_returns_empty() -> None:
    df = pl.DataFrame({"part_id": [None, None], "signal": [1.0, 2.0]})
    cov = compute_cov(df, ["signal"], group_by="part_id")
    assert cov.is_empty()
    print("  all-noise input returns empty OK")

def _bar_df() -> pl.DataFrame:
    return pl.DataFrame({
        "part_id": ["A", "B", "C", "D"],
        "cov_signal": [0.1, 0.05, 0.3, 0.15],
        "n_rows": [1000, 800, 1200, 950],
    })

def test_bar_basic() -> None:
    fig = bar(_bar_df(), x="part_id", y="cov_signal")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    trace = fig.data[0]
    assert trace.type == "bar"
    assert list(trace.x) == ["A", "B", "C", "D"]
    assert list(trace.y) == [0.1, 0.05, 0.3, 0.15]
    print("  bar basic OK")

def test_bar_sort_by_y() -> None:
    fig = bar(_bar_df(), x="part_id", y="cov_signal", sort_by="y")
    trace = fig.data[0]
    # Ascending by default → smallest CoV first.
    assert list(trace.x) == ["B", "A", "D", "C"]
    print("  bar sort_by=y ascending OK")

def test_bar_sort_descending() -> None:
    fig = bar(_bar_df(), x="part_id", y="cov_signal",
              sort_by="y", sort_descending=True)
    trace = fig.data[0]
    assert list(trace.x) == ["C", "D", "A", "B"]
    print("  bar sort descending OK")

def test_bar_horizontal() -> None:
    fig = bar(_bar_df(), x="part_id", y="cov_signal", orientation="h")
    trace = fig.data[0]
    assert trace.orientation == "h"
    # Horizontal: y of plot is the categorical axis.
    assert list(trace.y) == ["A", "B", "C", "D"]
    assert list(trace.x) == [0.1, 0.05, 0.3, 0.15]
    print("  bar horizontal OK")

def test_bar_with_numeric_color() -> None:
    fig = bar(_bar_df(), x="part_id", y="cov_signal", color="n_rows")
    trace = fig.data[0]
    assert trace.marker.colorscale is not None
    assert list(trace.marker.color) == [1000, 800, 1200, 950]
    print("  bar numeric color OK")

def test_bar_with_categorical_color() -> None:
    df = _bar_df().with_columns(
        pl.Series("group", ["g1", "g1", "g2", "g2"])
    )
    fig = bar(df, x="part_id", y="cov_signal", color="group")
    trace = fig.data[0]
    # Should be encoded to ints, with discrete colorbar ticks.
    assert all(isinstance(c, int) for c in trace.marker.color)
    assert list(trace.marker.colorbar.ticktext) == ["g1", "g2"]
    print("  bar categorical color OK")

def test_bar_label_overrides() -> None:
    fig = bar(
        _bar_df(), x="part_id", y="cov_signal",
        title="My title",
        xaxis_title="Part",
        yaxis_title="Coefficient of Variation",
    )
    assert fig.layout.title.text == "My title"
    assert fig.layout.xaxis.title.text == "Part"
    assert fig.layout.yaxis.title.text == "Coefficient of Variation"
    print("  bar label overrides OK")

def test_bar_unknown_column_raises() -> None:
    try:
        bar(_bar_df(), x="bogus", y="cov_signal")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
    print("  bar unknown column raises OK")

def test_bar_invalid_sort_by() -> None:
    try:
        bar(_bar_df(), x="part_id", y="cov_signal", sort_by="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    print("  bar invalid sort_by raises OK")

def _grid_df() -> pl.DataFrame:
    """3x3 grid: speeds 500/1000/1500, powers 100/150/200, cov known."""
    rows = []
    for speed in (500, 1000, 1500):
        for power in (100, 150, 200):
            # Fake CoV: lower at "ideal" combos, higher at extremes.
            cov = 0.1 + (abs(speed - 1000) / 1000) + (abs(power - 150) / 200)
            rows.append({"speed": float(speed), "power": float(power), "cov": cov})
    return pl.DataFrame(rows)

def test_contour_basic() -> None:
    from ampm.plotting import contour
    df = _grid_df()
    fig = contour(df, x="speed", y="power", z="cov")
    # 1 contour trace + 1 scatter trace (show_points default True).
    assert len(fig.data) == 2
    assert fig.data[0].type == "contour"
    assert fig.data[1].type == "scatter"
    # Contour z matrix should be 3x3.
    assert len(fig.data[0].z) == 3
    assert len(fig.data[0].z[0]) == 3
    # Axes should be sorted numerically.
    assert list(fig.data[0].x) == [500.0, 1000.0, 1500.0]
    assert list(fig.data[0].y) == [100.0, 150.0, 200.0]
    print("  contour basic (3x3 grid) OK")

def test_contour_show_points_false() -> None:
    from ampm.plotting import contour
    df = _grid_df()
    fig = contour(df, x="speed", y="power", z="cov", show_points=False)
    assert len(fig.data) == 1
    assert fig.data[0].type == "contour"
    print("  contour show_points=False OK")

def test_contour_irregular_data_has_gaps() -> None:
    """When the grid is incomplete, missing cells become null in the z matrix."""
    from ampm.plotting import contour
    df = pl.DataFrame({
        "speed": [500.0, 1000.0, 1500.0, 500.0],
        "power": [100.0, 100.0, 100.0, 200.0],  # no (1000,200) or (1500,200)
        "cov": [0.1, 0.2, 0.3, 0.4],
    })
    fig = contour(df, x="speed", y="power", z="cov")
    z = fig.data[0].z
    # power=200 row should have only one value; rest are null.
    p200_row_idx = list(fig.data[0].y).index(200.0)
    p200_row = z[p200_row_idx]
    n_nulls = sum(1 for v in p200_row if v is None)
    assert n_nulls == 2
    print("  contour with gaps OK")

def test_contour_label_overrides() -> None:
    from ampm.plotting import contour
    df = _grid_df()
    fig = contour(
        df, x="speed", y="power", z="cov",
        title="Process map",
        xaxis_title="Hatch Speed (mm/s)",
        yaxis_title="Hatch Power (W)",
        colorbar_title="CoV",
    )
    assert fig.layout.title.text == "Process map"
    assert fig.layout.xaxis.title.text == "Hatch Speed (mm/s)"
    assert fig.layout.yaxis.title.text == "Hatch Power (W)"
    print("  contour label overrides OK")

def test_contour_unknown_column_raises() -> None:
    from ampm.plotting import contour
    df = _grid_df()
    try:
        contour(df, x="bogus", y="power", z="cov")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
    print("  contour unknown column raises OK")

def main() -> None:
    print("Phase 10 stats + bar tests:")
    test_overall_cov_basic()
    test_overall_cov_mathematical_correctness()
    test_drop_noise_default()
    test_per_layer_mean_mode()
    test_zero_mean_returns_null()
    test_multiple_columns_in_one_call()
    test_unknown_column_raises()
    test_unknown_mode_raises()
    test_per_layer_mode_requires_layer_col()
    test_noise_label_string()
    test_empty_after_drop_returns_empty()
    test_bar_basic()
    test_bar_sort_by_y()
    test_bar_sort_descending()
    test_bar_horizontal()
    test_bar_with_numeric_color()
    test_bar_with_categorical_color()
    test_bar_label_overrides()
    test_bar_unknown_column_raises()
    test_bar_invalid_sort_by()
    test_contour_basic()
    test_contour_show_points_false()
    test_contour_irregular_data_has_gaps()
    test_contour_label_overrides()
    test_contour_unknown_column_raises()
    print("\nAll Phase 10 tests passed")

if __name__ == "__main__":
    main()
