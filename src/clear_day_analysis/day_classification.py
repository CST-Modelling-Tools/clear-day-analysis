from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


DayClass = Literal["extremely_clear", "clear", "cloudy", "extremely_cloudy"]


@dataclass(frozen=True)
class DailyIntegralResult:
    date: object  # datetime.date
    H_dni: float
    H_dni_clear: float
    ratio: float
    n_points: int
    dt_hours: float


def add_clear_dni_model(
    df: pd.DataFrame,
    *,
    E0: float,
    beta: float,
    dni_col: str = "DNI",
    elevation_col: str = "sun_elevation_deg",
    alpha_min_deg: float = 5.0,
    clear_col: str = "dni_clear_model",
) -> pd.DataFrame:
    """
    Add a column with the clear-day DNI model:
      DNI_clear = E0 * exp( -beta / sin(alpha) )

    Only computed where:
      - elevation >= alpha_min_deg
      - DNI is finite
    Else set to NaN.

    Returns a COPY of df with the new column.
    """
    if elevation_col not in df.columns:
        raise KeyError(f"Missing elevation column: {elevation_col}")
    if dni_col not in df.columns:
        raise KeyError(f"Missing DNI column: {dni_col}")

    out = df.copy()

    elev_deg = pd.to_numeric(out[elevation_col], errors="coerce").to_numpy(dtype=float)
    dni = pd.to_numeric(out[dni_col], errors="coerce").to_numpy(dtype=float)

    elev_rad = np.deg2rad(elev_deg)
    sin_alpha = np.sin(elev_rad)

    valid = np.isfinite(dni) & np.isfinite(sin_alpha) & (elev_deg >= alpha_min_deg) & (sin_alpha > 0.0)

    dni_clear = np.full(len(out), np.nan, dtype=float)
    # compute only for valid points
    dni_clear[valid] = float(E0) * np.exp(-float(beta) / sin_alpha[valid])

    out[clear_col] = dni_clear
    return out


def _infer_dt_hours(dt: pd.Series) -> float:
    """
    Infer the time step (hours) from a datetime series.
    Uses median diff of sorted times. Works for hourly, half-hourly, etc.
    """
    if not pd.api.types.is_datetime64_any_dtype(dt):
        raise TypeError("datetime column must be datetime64 dtype")

    dt_sorted = dt.sort_values()
    diffs = dt_sorted.diff().dropna()

    if len(diffs) == 0:
        raise ValueError("Cannot infer timestep: need at least 2 timestamps.")

    # Convert to hours
    median_seconds = float(diffs.dt.total_seconds().median())
    if not np.isfinite(median_seconds) or median_seconds <= 0:
        raise ValueError("Cannot infer timestep: median diff is invalid.")

    return median_seconds / 3600.0


def daily_dni_integral_ratio(
    df: pd.DataFrame,
    *,
    datetime_col: str = "datetime",
    dni_col: str = "DNI",
    clear_col: str = "dni_clear_model",
    alpha_min_deg: Optional[float] = None,
    elevation_col: str = "sun_elevation_deg",
) -> pd.DataFrame:
    """
    Compute daily integrals and ratio:

      H_dni       = sum( DNI(t) * dt )
      H_dni_clear = sum( DNI_clear(t) * dt )
      ratio       = H_dni / H_dni_clear

    If dt is constant, ratio equals sum(DNI)/sum(DNI_clear), but we compute dt explicitly.

    Points used:
      - DNI finite and >= 0
      - DNI_clear finite and > 0
      - If alpha_min_deg is provided: elevation >= alpha_min_deg

    Returns a DataFrame with columns:
      date, H_dni, H_dni_clear, ratio, n_points, dt_hours
    """
    for c in (datetime_col, dni_col, clear_col):
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")

    dt_hours = _infer_dt_hours(df[datetime_col])

    dni = pd.to_numeric(df[dni_col], errors="coerce").to_numpy(dtype=float)
    dni = np.where(np.isfinite(dni) & (dni >= 0.0), dni, np.nan)

    dni_clear = pd.to_numeric(df[clear_col], errors="coerce").to_numpy(dtype=float)

    use = np.isfinite(dni) & np.isfinite(dni_clear) & (dni_clear > 0.0)

    if alpha_min_deg is not None:
        if elevation_col not in df.columns:
            raise KeyError(f"Missing elevation column: {elevation_col}")
        elev_deg = pd.to_numeric(df[elevation_col], errors="coerce").to_numpy(dtype=float)
        use = use & np.isfinite(elev_deg) & (elev_deg >= float(alpha_min_deg))

    tmp = df.loc[use, [datetime_col]].copy()
    tmp["date"] = tmp[datetime_col].dt.date
    tmp["dni_w"] = dni[use] * dt_hours
    tmp["dni_clear_w"] = dni_clear[use] * dt_hours

    daily = (
        tmp.groupby("date", as_index=False)
        .agg(H_dni=("dni_w", "sum"), H_dni_clear=("dni_clear_w", "sum"), n_points=("dni_w", "size"))
    )

    # ratio; if H_dni_clear is 0 (shouldn't happen with filtering), set ratio = 0
    daily["ratio"] = np.where(daily["H_dni_clear"] > 0.0, daily["H_dni"] / daily["H_dni_clear"], 0.0)
    daily["dt_hours"] = dt_hours

    return daily


def classify_days_by_ratio(
    daily_df: pd.DataFrame,
    *,
    ratio_col: str = "ratio",
    class_col: str = "class",
    thr_extremely_clear: float = 0.90,
    thr_clear: float = 0.70,
    thr_cloudy: float = 0.40,
) -> pd.DataFrame:
    """
    Classify days into 4 categories using the daily integral ratio R:

      extremely_clear: R >= thr_extremely_clear
      clear:           thr_clear <= R < thr_extremely_clear
      cloudy:          thr_cloudy <= R < thr_clear
      extremely_cloudy: R < thr_cloudy

    Returns a COPY of daily_df with an added `class` column.
    """
    if ratio_col not in daily_df.columns:
        raise KeyError(f"Missing ratio column: {ratio_col}")

    out = daily_df.copy()
    r = pd.to_numeric(out[ratio_col], errors="coerce").to_numpy(dtype=float)
    r = np.where(np.isfinite(r), r, 0.0)  # treat missing as 0 => extremely_cloudy

    labels = np.full(len(out), "extremely_cloudy", dtype=object)
    labels[(r >= thr_cloudy) & (r < thr_clear)] = "cloudy"
    labels[(r >= thr_clear) & (r < thr_extremely_clear)] = "clear"
    labels[r >= thr_extremely_clear] = "extremely_clear"

    out[class_col] = labels
    return out