# Spatial XY-bias correction

`ampm.correction.MeltPoolCorrection` divides out a spatial bias in the MeltVIEW melt-pool sensor signal so that values at different XY positions on the build plate become comparable.

## Why this exists

The same physical melt pool produces a slightly different reading depending on where on the build plate it occurs. The bias is smooth — a polynomial function of (X, Y) with a weak dependence on LaserVIEW intensity — and was characterized off-line by fitting a polynomial regression model to known-flat reference data.

Without the correction, the XY-bias contaminates anything that compares parts at different positions:

- Per-part CoV is inflated for parts near the edges of the build plate
- Process maps show spurious gradients along whichever direction in (speed, power) space happens to correlate with XY position
- Cross-build comparisons mix sensor bias with real process variation

With the correction, what remains is the actual process variability.

## Calibration scope — important

The default polynomial coefficients in `MeltPoolCorrection` are calibrated for:

- The **MAIN machine** (NOT the RBV / build-volume validation machine)
- The **`MeltVIEW melt pool (mean)`** column specifically
- Standard build-plate orientation

Applying the default correction to data from a different sensor or machine is **wrong** — the polynomial doesn't generalize. For RBV data, a different sensor, or a re-calibrated polynomial, instantiate `MeltPoolCorrection` with your own coefficients:

```python
correction = MeltPoolCorrection(
    power_matrix=my_power_matrix,        # (N, 3) array of [X, Y, L] exponents
    coefficients=my_coefficients,        # (N,) array
)
```

If you don't have a calibration for your sensor, **don't apply the correction**. Leave `CORRECT_MELTPOOL = False` in `cov.py` and accept that XY-bias contributes to your CoV numbers.

## The math

The polynomial prediction at a point is:

```
p(X, Y, L) = sum_i  coefficients[i] * X^a_i * Y^b_i * L^c_i
```

where `[a_i, b_i, c_i] = power_matrix[i]` and `L` is `LaserVIEW (mean)`. The default polynomial has 10 terms:

| Term | a | b | c | Coefficient |
|------|---|---|---|-------------|
| L | 0 | 0 | 1 | -3.21e-1 |
| Y | 0 | 1 | 0 | -1.36e-1 |
| Y·L | 0 | 1 | 1 | 1.94e-3 |
| X | 1 | 0 | 0 | 1.82e-2 |
| X·L | 1 | 0 | 1 | -7.99e-4 |
| X·Y | 1 | 1 | 0 | -5.58e-4 |
| (constant) | 0 | 0 | 0 | 120.92 |
| X² | 2 | 0 | 0 | -1.25e-3 |
| Y² | 0 | 2 | 0 | -5.77e-4 |
| L² | 0 | 0 | 2 | 3.72e-3 |

The correction at each row is:

```
corrected = measured * p(0, 0, L) / p(X, Y, L)
```

That is, scale the measured value by the ratio of "what the polynomial says we'd see at the build-plate origin" to "what the polynomial says we'd see at this XY position." LaserVIEW is held constant on both sides — this is a spatial correction only, not a laser-power correction.

## Usage

```python
from ampm.correction import MeltPoolCorrection

correction = MeltPoolCorrection()
clustered = correction.apply(clustered)
# Adds 'MeltVIEW melt pool (mean) corrected' column.
```

The original `MeltVIEW melt pool (mean)` column is preserved. The new column gets `" corrected"` appended to its name. To override the new column name:

```python
clustered = correction.apply(clustered, output_col="my_corrected_signal")
```

To use non-default input column names (e.g., for a different machine where columns are named differently):

```python
clustered = correction.apply(
    clustered,
    x_col="X",
    y_col="Y",
    laser_view_col="laser_view",
    meltpool_col="meltpool",
)
```

## What the correction looks like in practice

For data near the build plate origin (X, Y ≈ 0, 0), the correction factor is ~1.0 — the corrected and original values are nearly identical.

For data at the edges (X or Y ≈ ±30 mm), the correction is typically a few percent. Multiplying or dividing by the same factor doesn't change much for any single point, but it removes a systematic bias that compounds when you compute statistics across thousands or millions of points.

If you toggle `CORRECT_MELTPOOL` on and off and compare the resulting CoV maps, you should see:

- **Without correction**: CoV varies somewhat with XY position (i.e., the contour map shows a slight gradient correlated with whichever axis sweep aligns with build-plate position)
- **With correction**: CoV varies primarily with laser parameters; the spatial gradient is reduced

If toggling makes no visible difference, the bias is small relative to the parameter-driven variance — your conclusion was already robust without it.

## When the correction breaks down

Edge cases that produce nulls in the corrected column:

- **Polynomial denominator is zero or negative** at some position. Yields a null (rather than NaN/Inf) so downstream stats aren't contaminated. Rare for the default coefficients within the build plate, but can happen at extreme positions or with custom polynomials.
- **Non-finite LaserVIEW values** in the input. The polynomial result is non-finite, so the corrected value is null.

The function silently produces nulls in these cases — it doesn't warn. If you suspect this is happening, count nulls in the output column:

```python
n_null = clustered["MeltVIEW melt pool (mean) corrected"].null_count()
print(f"{n_null:,} rows have null corrected value")
```

## Fitting your own polynomial

To recalibrate for a different sensor or machine:

1. Collect AMPM data while printing a known-flat reference geometry (e.g., a single thick layer covering the entire build plate at constant laser parameters)
2. Mask to the printed region
3. Fit a polynomial regression with sklearn's `PolynomialFeatures` + `LinearRegression`, predicting `MeltVIEW melt pool (mean)` from `[Demand X, Demand Y, LaserVIEW (mean)]`
4. Extract the `power_matrix` and `coefficients` from the fitted model
5. Pass them to `MeltPoolCorrection(power_matrix=..., coefficients=...)`

A polynomial of degree 2 with 10 features (the structure of the default) is a good starting point. Higher-order polynomials risk overfitting the calibration data — the bias should be smooth.
