"""
Tests for ampm.correction.

Verify the polynomial math is correct, the DataFrame application works,
default vs custom polynomials behave properly, and edge cases (zero
denominator, missing columns, shape mismatches) raise sensibly.
"""
from __future__ import annotations

import sys
from pathlib import Path


import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))

from ampm.correction import MeltPoolCorrection


def test_predict_at_origin_matches_expected() -> None:
    """At (X=0, Y=0, L=0), the polynomial should equal the constant term."""
    corr = MeltPoolCorrection()
    pred = corr.predict(np.array([0.0]), np.array([0.0]), np.array([0.0]))
    # The constant term is at index 6 of the default coefficients.
    expected = corr.coefficients[6]
    assert abs(pred[0] - expected) < 1e-9
    print(f"  predict at origin (L=0) = {pred[0]:.6f} matches constant OK")


def test_predict_at_origin_with_laser_view() -> None:
    """At (X=0, Y=0, L=L), should be constant + c0*L + c9*L^2."""
    corr = MeltPoolCorrection()
    L = 100.0
    pred = corr.predict(np.array([0.0]), np.array([0.0]), np.array([L]))
    expected = (
        corr.coefficients[6]      # constant
        + corr.coefficients[0] * L  # c * L
        + corr.coefficients[9] * L * L  # c * L^2
    )
    assert abs(pred[0] - expected) < 1e-9
    print(f"  predict at origin with L={L} matches hand calc OK")


def test_predict_general_position() -> None:
    """Spot-check the full polynomial expansion at a generic point."""
    corr = MeltPoolCorrection()
    X, Y, L = 10.0, -20.0, 50.0
    pred = corr.predict(np.array([X]), np.array([Y]), np.array([L]))

    expected = (
        corr.coefficients[0] * L          # L
        + corr.coefficients[1] * Y          # Y
        + corr.coefficients[2] * Y * L      # Y*L
        + corr.coefficients[3] * X          # X
        + corr.coefficients[4] * X * L      # X*L
        + corr.coefficients[5] * X * Y      # X*Y
        + corr.coefficients[6]              # constant
        + corr.coefficients[7] * X * X      # X^2
        + corr.coefficients[8] * Y * Y      # Y^2
        + corr.coefficients[9] * L * L      # L^2
    )
    assert abs(pred[0] - expected) < 1e-9
    print(f"  predict at (X={X}, Y={Y}, L={L}) matches hand calc OK")


def test_apply_basic() -> None:
    """Apply correction to a small DataFrame and check the result."""
    corr = MeltPoolCorrection()
    df = pl.DataFrame({
        "Demand X": [0.0, 10.0, -10.0, 20.0],
        "Demand Y": [0.0, 5.0, -5.0, -15.0],
        "LaserVIEW (mean)": [100.0, 100.0, 100.0, 100.0],
        "MeltVIEW melt pool (mean)": [200.0, 200.0, 200.0, 200.0],
    })
    out = corr.apply(df)

    # Default output column name.
    assert "MeltVIEW melt pool (mean) corrected" in out.columns
    # Original column preserved.
    assert "MeltVIEW melt pool (mean)" in out.columns

    # At (0,0), corrected should equal measured (ratio=1).
    row0 = out.filter(pl.col("Demand X") == 0.0).row(0, named=True)
    assert abs(row0["MeltVIEW melt pool (mean) corrected"] - 200.0) < 1e-3

    # At non-origin points, the correction should differ from raw.
    other = out.filter(pl.col("Demand X") != 0.0)
    diffs = (
        other["MeltVIEW melt pool (mean) corrected"].to_numpy()
        - other["MeltVIEW melt pool (mean)"].to_numpy()
    )
    assert (np.abs(diffs) > 0).any(), "Expected at least some non-zero correction"
    print("  apply() basic case OK (origin unchanged, off-origin corrected)")


def test_apply_custom_output_column() -> None:
    corr = MeltPoolCorrection()
    df = pl.DataFrame({
        "Demand X": [10.0],
        "Demand Y": [5.0],
        "LaserVIEW (mean)": [100.0],
        "MeltVIEW melt pool (mean)": [200.0],
    })
    out = corr.apply(df, output_col="my_corrected")
    assert "my_corrected" in out.columns
    assert "MeltVIEW melt pool (mean) corrected" not in out.columns
    print("  apply() custom output column OK")


def test_apply_custom_input_columns() -> None:
    corr = MeltPoolCorrection()
    df = pl.DataFrame({
        "x": [10.0],
        "y": [5.0],
        "lv": [100.0],
        "mp": [200.0],
    })
    out = corr.apply(
        df,
        x_col="x", y_col="y", laser_view_col="lv", meltpool_col="mp",
    )
    assert "mp corrected" in out.columns
    print("  apply() custom input column names OK")


def test_apply_missing_column_raises() -> None:
    corr = MeltPoolCorrection()
    df = pl.DataFrame({
        "Demand X": [10.0],
        "Demand Y": [5.0],
        # Missing LaserVIEW (mean)
        "MeltVIEW melt pool (mean)": [200.0],
    })
    try:
        corr.apply(df)
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
    print("  apply() raises on missing column OK")


def test_predict_shape_mismatch_raises() -> None:
    corr = MeltPoolCorrection()
    try:
        corr.predict(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    print("  predict() raises on shape mismatch OK")


def test_custom_coefficients() -> None:
    """Pass custom power matrix and coefficients."""
    # Simple polynomial: p(X, Y, L) = 1 + X + Y + L
    pm = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    coefs = np.array([1.0, 1.0, 1.0, 1.0])
    corr = MeltPoolCorrection(power_matrix=pm, coefficients=coefs)
    pred = corr.predict(np.array([2.0]), np.array([3.0]), np.array([5.0]))
    # 1 + 2 + 3 + 5 = 11
    assert pred[0] == 11.0
    print("  custom power matrix + coefficients OK")


def test_invalid_construction_raises() -> None:
    """Mismatched power_matrix and coefficients shapes."""
    try:
        MeltPoolCorrection(
            power_matrix=np.array([[0, 0, 0], [1, 0, 0]]),
            coefficients=np.array([1.0]),  # length doesn't match
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    # power_matrix wrong width
    try:
        MeltPoolCorrection(
            power_matrix=np.array([[0, 0]]),  # only 2 columns
            coefficients=np.array([1.0]),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    print("  invalid construction raises OK")


def test_zero_denominator_yields_nan() -> None:
    """If the polynomial happens to be 0 at a position, the corrected value
    should be NaN, not Inf."""
    pm = np.array([
        [0, 0, 0],
        [1, 0, 0],
    ])
    coefs = np.array([5.0, -1.0])  # p = 5 - X.  p(5, *, *) = 0
    corr = MeltPoolCorrection(power_matrix=pm, coefficients=coefs)
    df = pl.DataFrame({
        "Demand X": [5.0, 0.0],
        "Demand Y": [0.0, 0.0],
        "LaserVIEW (mean)": [0.0, 0.0],
        "MeltVIEW melt pool (mean)": [100.0, 100.0],
    })
    out = corr.apply(df)
    vals = out["MeltVIEW melt pool (mean) corrected"].to_numpy()
    # X=5 makes denominator zero: NaN / null result.
    assert np.isnan(vals[0]) or vals[0] is None
    # X=0 is fine: ratio = 5/5 = 1, corrected = 100.
    assert abs(vals[1] - 100.0) < 1e-3
    print("  zero denominator yields NaN OK")


def test_origin_invariance() -> None:
    """A row at (0, 0) should have corrected ≈ measured exactly."""
    corr = MeltPoolCorrection()
    df = pl.DataFrame({
        "Demand X": [0.0, 0.0, 0.0],
        "Demand Y": [0.0, 0.0, 0.0],
        "LaserVIEW (mean)": [50.0, 100.0, 150.0],
        "MeltVIEW melt pool (mean)": [180.0, 200.0, 220.0],
    })
    out = corr.apply(df)
    raw = out["MeltVIEW melt pool (mean)"].to_numpy()
    corrected = out["MeltVIEW melt pool (mean) corrected"].to_numpy()
    # Should be identical at the origin (within float32 precision).
    assert np.allclose(raw, corrected, atol=1e-3)
    print("  origin invariance OK")


def test_realistic_data_shape_preserved() -> None:
    """Apply to a moderate-size random DataFrame and verify shape/columns."""
    rng = np.random.default_rng(0)
    n = 10000
    df = pl.DataFrame({
        "Demand X": rng.uniform(-30, 30, n).astype(np.float32),
        "Demand Y": rng.uniform(-30, 30, n).astype(np.float32),
        "LaserVIEW (mean)": rng.uniform(50, 150, n).astype(np.float32),
        "MeltVIEW melt pool (mean)": rng.uniform(150, 250, n).astype(np.float32),
    })
    corr = MeltPoolCorrection()
    out = corr.apply(df)
    assert out.height == n
    assert out["MeltVIEW melt pool (mean) corrected"].dtype == pl.Float32
    # Most corrected values should be close to original (within reasonable factor).
    ratio = (
        out["MeltVIEW melt pool (mean) corrected"].to_numpy()
        / out["MeltVIEW melt pool (mean)"].to_numpy()
    )
    # Drop any NaNs from edge cases, then check most rows are within ±20%.
    finite_ratio = ratio[np.isfinite(ratio)]
    assert (np.abs(finite_ratio - 1.0) < 0.2).mean() > 0.95
    print("  realistic-data shape preserved, corrections sensible OK")


def main() -> None:
    print("Phase 11 correction tests:")
    test_predict_at_origin_matches_expected()
    test_predict_at_origin_with_laser_view()
    test_predict_general_position()
    test_apply_basic()
    test_apply_custom_output_column()
    test_apply_custom_input_columns()
    test_apply_missing_column_raises()
    test_predict_shape_mismatch_raises()
    test_custom_coefficients()
    test_invalid_construction_raises()
    test_zero_denominator_yields_nan()
    test_origin_invariance()
    test_realistic_data_shape_preserved()
    print("\nAll Phase 11 tests passed")


if __name__ == "__main__":
    main()
