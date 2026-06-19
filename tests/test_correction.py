"""
Tests for ``correction.py`` — polynomial XY-bias correction of the MeltVIEW
melt-pool signal.

Most tests use a small hand-built polynomial so the expected output can be
computed exactly, rather than relying on the default MAIN-machine calibration
coefficients.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ampm.correction import (
    _DEFAULT_COEFFICIENTS,
    _DEFAULT_POWER_MATRIX,
    MeltPoolCorrection,
)


def signal_df(xs, ys, lv, meas):
    return pl.DataFrame(
        {
            "Demand X": pl.Series(xs, dtype=pl.Float32),
            "Demand Y": pl.Series(ys, dtype=pl.Float32),
            "LaserVIEW (mean)": pl.Series(lv, dtype=pl.Float32),
            "MeltVIEW melt pool (mean)": pl.Series(meas, dtype=pl.Float32),
        }
    )


class TestConstruction:
    def test_defaults_have_consistent_shapes(self):
        c = MeltPoolCorrection()
        assert c.power_matrix.shape == (len(_DEFAULT_COEFFICIENTS), 3)
        assert c.coefficients.shape == (_DEFAULT_POWER_MATRIX.shape[0],)

    def test_custom_matching_arrays_ok(self):
        c = MeltPoolCorrection(np.array([[1, 0, 0], [0, 1, 0]]), np.array([1.0, 2.0]))
        assert c.coefficients.shape == (2,)

    def test_power_matrix_wrong_width_raises(self):
        with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
            MeltPoolCorrection(np.array([[1, 0], [0, 1]]), np.array([1.0, 2.0]))

    def test_coefficients_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            MeltPoolCorrection(np.array([[1, 0, 0], [0, 1, 0]]), np.array([1.0]))

    def test_repr_reports_terms_and_default_flag(self):
        assert "using_default=True" in repr(MeltPoolCorrection())
        custom = MeltPoolCorrection(np.array([[0, 0, 0]]), np.array([1.0]))
        r = repr(custom)
        assert "n_terms=1" in r and "using_default=False" in r


class TestPredict:
    def test_constant_term(self):
        c = MeltPoolCorrection(np.array([[0, 0, 0]]), np.array([5.0]))
        out = c.predict(
            np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])
        )
        assert out.tolist() == [5.0, 5.0]

    def test_single_x_term(self):
        c = MeltPoolCorrection(np.array([[1, 0, 0]]), np.array([2.0]))
        out = c.predict(np.array([3.0, 10.0]), np.zeros(2), np.zeros(2))
        assert out.tolist() == [6.0, 20.0]

    def test_combined_terms(self):
        # p(x,y,L) = 10 + x + y*L
        c = MeltPoolCorrection(
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]]), np.array([10.0, 1.0, 1.0])
        )
        out = c.predict(np.array([2.0]), np.array([3.0]), np.array([4.0]))
        # 10 + 2 + (3*4) = 24
        assert out[0] == pytest.approx(24.0)

    def test_shape_mismatch_raises(self):
        c = MeltPoolCorrection(np.array([[1, 0, 0]]), np.array([1.0]))
        with pytest.raises(ValueError, match="same shape"):
            c.predict(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0]))


class TestApply:
    def test_missing_column_raises(self):
        c = MeltPoolCorrection()
        df = pl.DataFrame({"Demand X": [1.0], "Demand Y": [2.0]})  # missing signals
        with pytest.raises(KeyError):
            c.apply(df)

    def test_default_output_column_name_and_dtype(self):
        c = MeltPoolCorrection(np.array([[0, 0, 0]]), np.array([1.0]))
        out = c.apply(signal_df([1.0], [1.0], [1.0], [2.0]))
        assert "MeltVIEW melt pool (mean) corrected" in out.columns
        assert out["MeltVIEW melt pool (mean) corrected"].dtype == pl.Float32

    def test_custom_output_column(self):
        c = MeltPoolCorrection(np.array([[0, 0, 0]]), np.array([1.0]))
        out = c.apply(signal_df([1.0], [1.0], [1.0], [2.0]), output_col="fixed")
        assert "fixed" in out.columns

    def test_constant_polynomial_is_identity(self):
        # p constant -> p(0,0,L)/p(X,Y,L) == 1 -> corrected == measured.
        c = MeltPoolCorrection(np.array([[0, 0, 0]]), np.array([7.0]))
        df = signal_df([1.0, 9.0], [2.0, 8.0], [1.0, 1.0], [3.0, 4.0])
        out = c.apply(df)
        assert out["MeltVIEW melt pool (mean) corrected"].to_list() == pytest.approx(
            [3.0, 4.0]
        )

    def test_known_ratio(self):
        # p(x,y,L) = 10 + x. corrected = measured * 10 / (10 + x).
        c = MeltPoolCorrection(np.array([[0, 0, 0], [1, 0, 0]]), np.array([10.0, 1.0]))
        df = signal_df([0.0, 10.0], [0.0, 0.0], [1.0, 1.0], [100.0, 100.0])
        out = c.apply(df)["MeltVIEW melt pool (mean) corrected"].to_list()
        assert out == pytest.approx([100.0, 50.0])

    def test_zero_denominator_becomes_nan(self):
        # p(x,y,L) = x. At x=0 the denominator is 0 -> corrected is NaN.
        c = MeltPoolCorrection(np.array([[1, 0, 0]]), np.array([1.0]))
        df = signal_df([0.0, 5.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0])
        col = c.apply(df)["MeltVIEW melt pool (mean) corrected"]
        assert col.is_nan().to_list() == [True, False]
        assert col.to_list()[1] == pytest.approx(0.0)

    def test_original_column_preserved(self):
        c = MeltPoolCorrection(np.array([[0, 0, 0]]), np.array([1.0]))
        df = signal_df([1.0], [1.0], [1.0], [3.0])
        out = c.apply(df)
        assert out["MeltVIEW melt pool (mean)"].to_list() == pytest.approx([3.0])


class TestDefaultCalibration:
    def test_apply_runs_and_is_finite_near_origin(self):
        # The default polynomial is dominated by its large constant term, so
        # corrected values near the plate centre should be finite and close to
        # the measured value.
        c = MeltPoolCorrection()
        df = signal_df(
            [0.0, 1.0, -1.0],
            [0.0, 1.0, -1.0],
            [100.0, 100.0, 100.0],
            [50.0, 50.0, 50.0],
        )
        out = c.apply(df)["MeltVIEW melt pool (mean) corrected"]
        assert out.is_finite().all()
        assert out.to_list() == pytest.approx([50.0, 50.0, 50.0], rel=0.1)
