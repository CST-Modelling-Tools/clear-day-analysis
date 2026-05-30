# Method: Clear-Day Envelope And Day Classification

## Inputs

The workflow accepts supported Typical Meteorological Year (TMY) CSV files with direct normal irradiance (DNI) measurements.

Currently supported sources are:

- NSRDB TMY CSV
- Solargis TMY60 P50 CSV
- PVGIS 5.x TMY CSV

TMY ingestion is centralized in `src/clear_day_analysis/tmy_reader.py`. Source-specific readers normalize provider formats into a common DataFrame and metadata schema. New workflows should use:

```python
read_tmy_csv(path, source="auto")
```

The normalized DataFrame is expected to include at least:

- `datetime`: repository-standard UTC analysis timestamp
- `DNI`: direct normal irradiance
- preferably `GHI`: global horizontal irradiance

## Datetime Convention

The analysis timestamp is always `df["datetime"]`.

It is timezone-aware UTC and is used for solar position, clear-day fitting, daily integrals, classification, plotting, and exports unless a workflow explicitly documents a different grouping column.

TMY files can contain source-year discontinuities because each month may come from a different historical year. Normalizing timestamps avoids scrambled annual ordering and unstable daily grouping.

PVGIS-specific behavior:

- `pvgis_datetime_utc` preserves the original PVGIS UTC timestamp and source year for auditing.
- `datetime` is normalized to a monotonic, non-leap synthetic TMY calendar while preserving month, day, hour, minute, and second.

Source-specific timestamps are not used for analysis.

## Workflow

The standard workflow is:

1. Read TMY data.
2. Compute solar position.
3. Fit the clear-day envelope.
4. Build the clear-day DNI model.
5. Compute daily DNI integral ratios.
6. Classify days.
7. Generate plots or exports.

## Solar Geometry

For each timestamp `t`, the solar elevation angle `alpha(t)` is computed using a C++ implementation of a PSA-based sun position algorithm wrapped with pybind11.

Definitions:

- `sin_alpha(t) = sin(alpha(t))`
- `x(t) = 1 / sin_alpha(t)`

The variable `x(t)` is used in the linearized ASHRAE clear-day model.

Only points where the sun is sufficiently above the horizon are used to avoid low-angle effects. The default minimum solar elevation is 5 degrees.

## ASHRAE Clear-Day DNI Model

The ASHRAE formulation for direct normal irradiance under clear-sky conditions is:

```text
DNI_clear(t) = E0 * exp(-beta / sin_alpha(t))
```

Taking the natural logarithm gives:

```text
ln(DNI(t)) = a + b * x(t)
```

where:

- `a = ln(E0)`
- `b = -beta`
- `x(t) = 1 / sin_alpha(t)`

This allows estimation of `E0` and `beta` via ordinary least squares in log-space.

## Iterative Fitting With Student-t Corridor

The clear-day parameters are estimated using all valid DNI points of the year, not day-by-day.

The iterative procedure is:

1. Filter to `DNI > 0` and solar elevation greater than or equal to the minimum elevation.
2. Fit `ln(DNI) = a + b * x` in log-space.
3. Compute a Student-t prediction corridor.
4. Reject outliers.
5. Repeat until convergence.

The default outlier mode removes points below the lower corridor because clouds reduce DNI. A two-sided mode is also available.

Convergence is determined by negligible change in fitted parameters and/or very few new outliers removed in an iteration.

## Envelope Enforcement

The iterative OLS plus corridor procedure yields a robust clear-sky fit, but not necessarily a strict envelope.

To enforce envelope behavior, a final upward parallel shift is applied in log-space:

1. Compute residuals from the fitted line.
2. Compute a high residual quantile, such as 98%.
3. Apply a non-negative upward shift to `a`.

This increases `E0` by a multiplicative factor while keeping `beta` unchanged.

## Daily DNI Energy Integrals

For each day `d`, energy-like integrals are computed:

```text
H_dni(d) = sum(DNI(t) * dt)
H_clear(d) = sum(DNI_clear(t) * dt)
```

where `dt` is the timestep in hours, inferred from the data.

Only points consistent with the clear-day model filtering are used.

The daily ratio is:

```text
R(d) = H_dni(d) / H_clear(d)
```

## Day Classification

Each day is classified from the daily ratio `R(d)` using fixed physical thresholds:

- `extremely_clear`: `R >= 0.90`
- `clear`: `0.70 <= R < 0.90`
- `cloudy`: `0.40 <= R < 0.70`
- `extremely_cloudy`: `R < 0.40`

These thresholds are intentionally location-independent because they compare measured DNI to a clear-day reference under identical solar geometry.

## TMY Ordering Note

Daily outputs are ordered by natural TMY month/day ordering where needed. This avoids provider source-year artifacts and keeps results interpretable as one synthetic annual TMY sequence.
