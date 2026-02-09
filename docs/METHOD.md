# Method: Clear-day envelope and day classification

## Inputs

- NSRDB Typical Meteorological Year (TMY) CSV with DNI measurements.
- Solar position (sun elevation) computed for each timestamp.

Only points where the sun is sufficiently above the horizon are used to avoid low-angle effects. By default:
- Minimum solar elevation: 5°.

## Solar geometry

For each timestamp t, the solar elevation angle α(t) is computed using a C++ implementation of a PSA-based sun position algorithm (wrapped with pybind11).

We define:

- sinα(t) = sin(α(t))
- x(t) = 1 / sinα(t)

The variable x(t) is used in the linearized ASHRAE clear-day model.

## ASHRAE clear-day DNI model

The ASHRAE formulation for direct normal irradiance (DNI) under clear-sky conditions is assumed to be:

DNI_clear(t) = E0 · exp( −β / sinα(t) )

Taking the natural logarithm gives a linear relationship:

ln(DNI(t)) = a + b · x(t)

where:

- a = ln(E0)
- b = −β
- x(t) = 1 / sinα(t)

This allows estimation of E0 and β via ordinary least squares (OLS) in log-space.

## Iterative fitting with Student-t corridor

The clear-day parameters are estimated using all valid DNI points of the year (not day-by-day), through an iterative procedure:

1. Initial filtering:
   - DNI > 0
   - solar elevation ≥ α_min (default 5°)

2. OLS fit in log-space:
   ln(DNI) = a + b · x

3. Computation of a Student-t prediction interval ("error corridor"):

   ŷ(x) ± t_{α/2,ν} · s · √(1 + 1/n + (x − x̄)² / Sxx)

   where:
   - s is the residual standard deviation
   - ν = n − 2 degrees of freedom

4. Outlier rejection:
   - Default mode ("lower"): remove points below the lower corridor
     (physically, clouds reduce DNI)
   - Optional mode ("two_sided"): remove points outside the corridor

5. Repeat steps 2–4 until convergence.

Convergence is determined by:
- negligible change in fitted parameters, and/or
- very few new outliers removed in an iteration.

## Envelope enforcement

The iterative OLS + corridor procedure yields a robust clear-sky fit, but not necessarily a strict envelope.

To enforce envelope behavior, a final upward parallel shift is applied in log-space:

- Residuals: r(t) = ln(DNI(t)) − (a + b · x(t))
- Compute a high quantile (e.g. 98%) of r(t)
- Apply a shift:

  a_env = a + max(0, Δa)

This increases E0 by a factor exp(Δa) while keeping β unchanged.

## Daily DNI energy integrals

For each day d, energy-like integrals are computed:

H_dni(d) = Σ DNI(t) · Δt
H_clear(d) = Σ DNI_clear(t) · Δt

where Δt is the timestep (in hours), inferred from the data.

Only points consistent with the clear-day model filtering (elevation ≥ α_min) are used.

The daily ratio is defined as:

R(d) = H_dni(d) / H_clear(d)

## Day classification

Each day is classified based on the ratio R(d) using fixed physical thresholds:

- extremely_clear: R ≥ 0.90
- clear: 0.70 ≤ R < 0.90
- cloudy: 0.40 ≤ R < 0.70
- extremely_cloudy: R < 0.40

These thresholds are intentionally location-independent, as they compare measured DNI to a clear-day reference under identical solar geometry.

## TMY ordering note

In NSRDB TMY files, different months may originate from different historical years. For presentation purposes, daily results are ordered by (month, day) rather than by full calendar date.