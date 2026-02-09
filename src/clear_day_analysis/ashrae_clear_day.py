from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


OutlierMode = Literal["lower", "two_sided"]


@dataclass(frozen=True)
class IterationInfo:
    iteration: int
    n_in: int
    n_outliers: int
    n_kept: int
    a: float                # intercept in log-space = ln(E0)  (OLS line, not envelope-shifted)
    b: float                # slope in log-space = -beta
    rmse: float             # RMSE in log-space
    tcrit: float            # t critical used
    confidence: float


@dataclass(frozen=True)
class AshraeFitResult:
    # Final parameters returned (may be envelope-shifted if enforce_envelope=True)
    E0: float
    beta: float
    a: float
    b: float

    confidence: float
    alpha_min_deg: float
    outlier_mode: OutlierMode
    max_iter: int
    converged: bool
    n_final: int
    history: list[IterationInfo]

    # Envelope shift diagnostics
    enforce_envelope: bool
    envelope_quantile: float
    delta_a: float  # amount added to 'a' in log-space to enforce envelope (0 if disabled)


def _t_critical(confidence: float, df: int) -> float:
    """
    Two-sided t critical value for prediction interval:
      t_{alpha/2, df} where alpha = 1 - confidence.

    Tries SciPy. If SciPy is not available, falls back to a normal approximation.
    """
    alpha = 1.0 - confidence
    if df <= 0:
        return float("nan")

    try:
        from scipy.stats import t  # type: ignore
        return float(t.ppf(1.0 - alpha / 2.0, df))
    except Exception:
        from statistics import NormalDist
        return float(NormalDist().inv_cdf(1.0 - alpha / 2.0))


def _ols_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    OLS fit y = a + b x in closed form.
    Returns (a, b).
    """
    xbar = float(np.mean(x))
    ybar = float(np.mean(y))
    dx = x - xbar
    Sxx = float(np.sum(dx * dx))
    if Sxx <= 0.0:
        raise ValueError("Sxx is zero; x has no variance. Check elevation filtering / data.")
    b = float(np.sum(dx * (y - ybar)) / Sxx)
    a = ybar - b * xbar
    return a, b


def _prediction_band(
    x: np.ndarray,
    y: np.ndarray,
    a: float,
    b: float,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute Student-t prediction interval band for each x_i in log-space.

    Returns:
      yhat, halfwidth, rmse, tcrit
    where:
      yhat = a + b x
      band = yhat +/- halfwidth
    """
    n = len(x)
    if n < 3:
        raise ValueError("Need at least 3 points to compute t-based corridor (n>=3).")

    yhat = a + b * x
    resid = y - yhat
    SSE = float(np.sum(resid * resid))
    df = n - 2
    s = np.sqrt(SSE / df)

    xbar = float(np.mean(x))
    Sxx = float(np.sum((x - xbar) ** 2))

    tcrit = _t_critical(confidence, df)

    # Prediction interval standard error at each x:
    # s * sqrt( 1 + 1/n + (x - xbar)^2 / Sxx )
    se_pred = s * np.sqrt(1.0 + (1.0 / n) + ((x - xbar) ** 2) / Sxx)

    halfwidth = tcrit * se_pred
    rmse = float(np.sqrt(SSE / n))

    return yhat, halfwidth, rmse, tcrit


def fit_ashrae_clear_day(
    df: pd.DataFrame,
    *,
    dni_col: str = "DNI",
    elevation_col: str = "sun_elevation_deg",
    alpha_min_deg: float = 5.0,
    confidence: float = 0.95,
    outlier_mode: OutlierMode = "lower",
    max_iter: int = 25,
    min_points: int = 200,
    eps_a: float = 1e-4,
    eps_b: float = 1e-4,
    min_outliers_to_continue: int = 0,
    enforce_envelope: bool = False,
    envelope_quantile: float = 0.98,
) -> AshraeFitResult:
    """
    Fit ASHRAE clear-day model using iterative OLS + Student-t corridor outlier rejection.

    Model (after log transform):
      DNI = E0 * exp( -beta / sin(alpha) )
      y = ln(DNI) = a + b x
      with x = 1 / sin(alpha), a = ln(E0), b = -beta

    Data selection:
      DNI > 0
      elevation_deg >= alpha_min_deg

    Outlier corridor:
      Student-t prediction interval in y-space.
      - outlier_mode="lower": remove points below (y < yhat - halfwidth)
      - outlier_mode="two_sided": remove points outside [lower, upper]

    Convergence:
      Stop if either:
        (1) n_outliers <= min_outliers_to_continue, OR
        (2) parameter stabilization: |Δa| < eps_a AND |Δb| < eps_b

    Optional envelope enforcement (recommended for integral-ratio classification):
      If enforce_envelope=True, after the iterative fit converges, we apply a FINAL
      upward parallel shift in log-space:
          delta_a = quantile( y - (a + b x), envelope_quantile )
          a_env = a + delta_a
      This keeps slope b (=> beta) unchanged and increases E0 so the resulting curve
      behaves more like an envelope (reduces daily ratios > 1).

    Notes:
      - For correct t-Student critical values, install SciPy:
          pip install scipy
        Without SciPy, we fall back to a normal approximation.
    """
    if dni_col not in df.columns:
        raise KeyError(f"Missing DNI column: {dni_col}")
    if elevation_col not in df.columns:
        raise KeyError(f"Missing elevation column: {elevation_col}")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be between 0 and 1 (e.g., 0.95).")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1.")
    if eps_a < 0 or eps_b < 0:
        raise ValueError("eps_a and eps_b must be >= 0.")
    if min_outliers_to_continue < 0:
        raise ValueError("min_outliers_to_continue must be >= 0.")
    if not (0.0 < envelope_quantile < 1.0):
        raise ValueError("envelope_quantile must be between 0 and 1 (e.g., 0.98).")

    dni = pd.to_numeric(df[dni_col], errors="coerce").to_numpy(dtype=float)
    elev_deg = pd.to_numeric(df[elevation_col], errors="coerce").to_numpy(dtype=float)

    # Initial filter
    mask = np.isfinite(dni) & np.isfinite(elev_deg) & (dni > 0.0) & (elev_deg >= alpha_min_deg)
    idx = np.where(mask)[0]

    if len(idx) < min_points:
        raise ValueError(
            f"Too few points after filtering: {len(idx)}. "
            f"Try lowering alpha_min_deg or min_points."
        )

    history: list[IterationInfo] = []
    converged = False

    prev_a: Optional[float] = None
    prev_b: Optional[float] = None

    # Keep the last valid (x,y,a,b) for envelope enforcement
    last_x: Optional[np.ndarray] = None
    last_y: Optional[np.ndarray] = None
    last_a: Optional[float] = None
    last_b: Optional[float] = None

    for it in range(1, max_iter + 1):
        n_in = len(idx)

        elev_rad = np.deg2rad(elev_deg[idx])
        sin_alpha = np.sin(elev_rad)

        good = np.isfinite(sin_alpha) & (sin_alpha > 0.0)
        idx = idx[good]
        n_in = len(idx)

        if n_in < 3:
            break

        x = 1.0 / sin_alpha[good]  # 1/sin(alpha)
        y = np.log(dni[idx])       # ln(DNI)

        a, b = _ols_line(x, y)

        yhat, halfwidth, rmse, tcrit = _prediction_band(x, y, a, b, confidence)
        lower = yhat - halfwidth
        upper = yhat + halfwidth

        if outlier_mode == "lower":
            keep = y >= lower
        elif outlier_mode == "two_sided":
            keep = (y >= lower) & (y <= upper)
        else:
            raise ValueError(f"Invalid outlier_mode: {outlier_mode}")

        n_kept = int(np.sum(keep))
        n_outliers = n_in - n_kept

        history.append(
            IterationInfo(
                iteration=it,
                n_in=n_in,
                n_outliers=n_outliers,
                n_kept=n_kept,
                a=float(a),
                b=float(b),
                rmse=float(rmse),
                tcrit=float(tcrit),
                confidence=float(confidence),
            )
        )

        # Save last valid state for potential envelope shift
        last_x = x
        last_y = y
        last_a = float(a)
        last_b = float(b)

        # Convergence (2): parameter stabilization
        if prev_a is not None and prev_b is not None:
            if abs(a - prev_a) < eps_a and abs(b - prev_b) < eps_b:
                converged = True
                break

        prev_a = float(a)
        prev_b = float(b)

        # Convergence (1): (near-)no outliers removed
        if n_outliers <= min_outliers_to_continue:
            converged = True
            break

        # Update idx to kept points
        idx = idx[keep]

        if len(idx) < 3:
            break

    if not history:
        raise RuntimeError("Fitting failed: no iterations completed.")

    # Final line from last iteration recorded
    last_hist = history[-1]
    a_final = float(last_hist.a)
    b_final = float(last_hist.b)

    # Optional envelope enforcement: shift intercept upward in log-space
    delta_a = 0.0
    if enforce_envelope:
        if last_x is None or last_y is None or last_a is None or last_b is None:
            raise RuntimeError("Internal error: missing final x/y for envelope enforcement.")

        yhat_last = last_a + last_b * last_x
        resid_last = last_y - yhat_last

        # We want the envelope above most points: shift so that a high quantile residual becomes 0
        # If this quantile is negative (rare), do not shift downward.
        q = float(np.quantile(resid_last, envelope_quantile))
        delta_a = max(0.0, q)

        a_final = a_final + delta_a
        # b_final unchanged

    E0 = float(np.exp(a_final))
    beta = float(-b_final)

    return AshraeFitResult(
        E0=E0,
        beta=beta,
        a=a_final,
        b=b_final,
        confidence=float(confidence),
        alpha_min_deg=float(alpha_min_deg),
        outlier_mode=outlier_mode,
        max_iter=int(max_iter),
        converged=converged,
        n_final=int(last_hist.n_kept),
        history=history,
        enforce_envelope=bool(enforce_envelope),
        envelope_quantile=float(envelope_quantile),
        delta_a=float(delta_a),
    )