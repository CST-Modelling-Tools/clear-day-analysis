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
    a: float                # intercept in log-space = ln(E0)
    b: float                # slope in log-space = -beta
    rmse: float             # RMSE in log-space
    tcrit: float            # t critical used
    confidence: float


@dataclass(frozen=True)
class AshraeFitResult:
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


def _t_critical(confidence: float, df: int) -> float:
    """
    Two-sided t critical value for prediction interval:
      t_{alpha/2, df} where alpha = 1 - confidence.

    Tries SciPy. If SciPy is not available, falls back to a normal approximation.
    """
    alpha = 1.0 - confidence
    if df <= 0:
        # avoid crashing; corridor becomes meaningless with df<=0
        return float("nan")

    try:
        from scipy.stats import t  # type: ignore
        return float(t.ppf(1.0 - alpha / 2.0, df))
    except Exception:
        # Fallback: normal approximation (good for large df)
        # For small df, this underestimates tcrit slightly.
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
) -> AshraeFitResult:
    """
    Fit ASHRAE clear-day model using iterative OLS + Student-t corridor outlier rejection.

    Model (as in Section 4, after log transform):
      DNI = E0 * exp( -beta / sin(alpha) )
      y = ln(DNI) = a + b x
      with x = 1 / sin(alpha), a = ln(E0), b = -beta

    Data selection:
      DNI > 0
      elevation_deg >= alpha_min_deg

    Outlier corridor:
      Student-t prediction interval in y-space.
      - outlier_mode="lower": remove points below (y < yhat - halfwidth)
      - outlier_mode="two_sided": remove points outside (|y - yhat| > halfwidth)

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

    dni = pd.to_numeric(df[dni_col], errors="coerce").to_numpy(dtype=float)
    elev_deg = pd.to_numeric(df[elevation_col], errors="coerce").to_numpy(dtype=float)

    # Filter initial set
    mask = np.isfinite(dni) & np.isfinite(elev_deg) & (dni > 0.0) & (elev_deg >= alpha_min_deg)
    idx = np.where(mask)[0]

    if len(idx) < min_points:
        raise ValueError(
            f"Too few points after filtering: {len(idx)}. "
            f"Try lowering alpha_min_deg or min_points."
        )

    history: list[IterationInfo] = []
    converged = False

    # Iterate
    for it in range(1, max_iter + 1):
        n_in = len(idx)

        elev_rad = np.deg2rad(elev_deg[idx])
        sin_alpha = np.sin(elev_rad)

        # Guard against tiny/invalid sin(alpha)
        good = np.isfinite(sin_alpha) & (sin_alpha > 0.0)
        idx = idx[good]
        n_in = len(idx)

        if n_in < 3:
            break

        x = 1.0 / sin_alpha[good]          # 1/sin(alpha)
        y = np.log(dni[idx])               # ln(DNI)

        # OLS
        a, b = _ols_line(x, y)

        # Corridor (prediction interval)
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

        # Convergence: no new outliers removed
        if n_outliers == 0:
            converged = True
            break

        # Update idx to kept points
        idx = idx[keep]

        # Safety: if the filter collapses too much, stop.
        if len(idx) < 3:
            break

    # Final parameters from last iteration (or compute from last history entry)
    if not history:
        raise RuntimeError("Fitting failed: no iterations completed.")

    last = history[-1]
    a_final = last.a
    b_final = last.b

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
        n_final=int(last.n_kept),
        history=history,
    )