"""
Spatial XY-bias correction for the MeltVIEW melt-pool sensor signal on MAIN machine.

Background
----------
The MeltVIEW sensor on the Renishaw 500S MAIN machine has a positional bias
in its melt-pool intensity reading: the same physical melt pool produces a
slightly different value depending on where on the build plate it occurs.
The bias is a smooth function of (X, Y) and depends weakly on the
LaserVIEW reading too.

This module applies a pre-fitted polynomial regression model that captures
that bias, then divides it out to produce a normalized "as if measured at
the build-plate origin" signal.

Calibration scope
-----------------
The polynomial coefficients in :class:`MeltPoolCorrection` were fitted
specifically for:

  * the **MAIN** machine (NOT the RBV)
  * the **MeltVIEW melt pool (mean)** column
  * the standard build-plate orientation

If you're working with data from the RBV machine, a different sensor, or a
re-calibrated polynomial, instantiate ``MeltPoolCorrection`` with your own
``power_matrix`` and ``coefficients`` arguments rather than relying on the
defaults.

Math
----
The polynomial prediction is::

    p(X, Y, L) = sum_i  coefficients[i] * X^a_i * Y^b_i * L^c_i

where ``[a_i, b_i, c_i] = power_matrix[i]`` and ``L`` is LaserVIEW (mean).

The corrected signal at each row is::

    corrected = measured * p(0, 0, L) / p(X, Y, L)

i.e. divide out the spatial component while holding LaserVIEW fixed.
"""

from __future__ import annotations

import numpy as np
import polars as pl

_DEFAULT_POWER_MATRIX: np.ndarray = np.array(
    [
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [0, 0, 0],
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
    ],
    dtype=np.int8,
)
_DEFAULT_COEFFICIENTS: np.ndarray = np.array(
    [
        -3.21229823e-01,
        -1.36372304e-01,
        1.93998884e-03,
        1.81922155e-02,
        -7.99325110e-04,
        -5.58476892e-04,
        1.20919463e02,
        -1.25344185e-03,
        -5.77211144e-04,
        3.72306302e-03,
    ],
    dtype=np.float64,
)


class MeltPoolCorrection:
    """
    XY-bias correction for the MeltVIEW melt-pool signal on the MAIN machine.

    The default ``power_matrix`` and ``coefficients`` are calibrated for the
    MAIN machine's MeltVIEW melt pool (mean) signal **only**. For other
    sensors, machines, or signals, supply your own ``power_matrix`` and
    ``coefficients`` (they must have matching first-axis lengths).

    Parameters
    ----------
    power_matrix
        Shape (N, 3) integer array where each row is the polynomial exponent
        for (X, Y, LaserVIEW). Defaults to the MAIN machine calibration.
    coefficients
        Shape (N,) array of polynomial coefficients aligned with
        ``power_matrix``. Defaults to the MAIN machine calibration.

    Examples
    --------
    >>> correction = MeltPoolCorrection()
    >>> df_corrected = correction.apply(clustered)
    >>> # Adds 'MeltVIEW melt pool (mean) corrected' column.
    """

    def __init__(
        self,
        power_matrix: np.ndarray | None = None,
        coefficients: np.ndarray | None = None,
    ) -> None:
        self.power_matrix = (
            np.asarray(power_matrix, dtype=np.int64)
            if power_matrix is not None
            else _DEFAULT_POWER_MATRIX.astype(np.int64)
        )
        self.coefficients = (
            np.asarray(coefficients, dtype=np.float64)
            if coefficients is not None
            else _DEFAULT_COEFFICIENTS
        )

        if self.power_matrix.ndim != 2 or self.power_matrix.shape[1] != 3:
            raise ValueError(
                f"power_matrix must have shape (N, 3), got {self.power_matrix.shape}"
            )
        if self.coefficients.shape != (self.power_matrix.shape[0],):
            raise ValueError(
                f"coefficients shape {self.coefficients.shape} does not match "
                f"power_matrix's first axis ({self.power_matrix.shape[0]})"
            )

    def __repr__(self) -> str:
        return (
            f"MeltPoolCorrection(n_terms={len(self.coefficients)}, "
            f"using_default={np.array_equal(self.coefficients, _DEFAULT_COEFFICIENTS)})"
        )

    def predict(
        self,
        x: np.ndarray,
        y: np.ndarray,
        laser_view: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate the polynomial at the given coordinates.

        Inputs are 1-D arrays of equal length. Returns a 1-D array of
        predicted melt-pool values (same length).
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        laser_view = np.asarray(laser_view, dtype=np.float64)
        if not (x.shape == y.shape == laser_view.shape):
            raise ValueError(
                f"x, y, laser_view must have the same shape; "
                f"got {x.shape}, {y.shape}, {laser_view.shape}"
            )

        out = np.zeros_like(x)
        for (a, b, c), coef in zip(self.power_matrix, self.coefficients):
            term = np.full_like(x, coef)
            if a:
                term *= x**a
            if b:
                term *= y**b
            if c:
                term *= laser_view**c
            out += term
        return out

    def apply(
        self,
        df: pl.DataFrame,
        *,
        x_col: str = "Demand X",
        y_col: str = "Demand Y",
        laser_view_col: str = "LaserVIEW (mean)",
        meltpool_col: str = "MeltVIEW melt pool (mean)",
        output_col: str | None = None,
    ) -> pl.DataFrame:
        """
        Apply the correction to a DataFrame and return a new DataFrame with
        a corrected column added.

        For each row the corrected value is::

            measured * p(0, 0, L) / p(X, Y, L)

        Parameters
        ----------
        df
            Input DataFrame. Must contain ``x_col``, ``y_col``,
            ``laser_view_col``, and ``meltpool_col``.
        x_col, y_col
            Demand position columns. Defaults match DataStore output.
        laser_view_col
            LaserVIEW (mean) column. Default ``"LaserVIEW (mean)"``.
        meltpool_col
            MeltVIEW melt pool (mean) column. Default
            ``"MeltVIEW melt pool (mean)"``.
        output_col
            Name of the new corrected column. Default is
            ``"<meltpool_col> corrected"`` so the original column is
            preserved.

        Returns
        -------
        DataFrame with the corrected column appended.

        Notes
        -----
        Rows where the predicted denominator is non-finite, zero, or
        negative produce a null in the corrected column rather than NaN/Inf.
        This protects downstream stats from contamination but warns nothing,
        on the assumption that you trust the calibration on its intended
        spatial domain.
        """
        for c in (x_col, y_col, laser_view_col, meltpool_col):
            if c not in df.columns:
                raise KeyError(f"Column {c!r} not in DataFrame")

        if output_col is None:
            output_col = f"{meltpool_col} corrected"

        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()
        lv = df[laser_view_col].to_numpy()
        m = df[meltpool_col].to_numpy()

        x = x.astype(np.float64, copy=False)
        y = y.astype(np.float64, copy=False)
        lv = lv.astype(np.float64, copy=False)

        zeros = np.zeros_like(x)
        denom = self.predict(x, y, lv)
        numer = self.predict(zeros, zeros, lv)
        ratio = np.divide(
            numer,
            denom,
            out=np.full_like(denom, np.nan),
            where=(denom != 0) & np.isfinite(denom),
        )
        corrected = m.astype(np.float64) * ratio

        bad = ~np.isfinite(corrected)
        if bad.any():
            corrected = corrected.copy()
            corrected[bad] = np.nan
        return df.with_columns(pl.Series(output_col, corrected.astype(np.float32)))
