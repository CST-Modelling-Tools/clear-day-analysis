from __future__ import annotations

import numpy as np
import pandas as pd

from ._sunpos import sunpos as _sunpos

RAD2DEG = 180.0 / np.pi


def _extract_utc_components(dt: pd.Series) -> tuple[np.ndarray, ...]:
    """
    Extract UTC components from a datetime Series.

    If timezone-aware: convert to UTC.
    If naive: assume already UTC.
    """
    if not pd.api.types.is_datetime64_any_dtype(dt):
        raise TypeError("datetime column must be datetime64 dtype")

    if getattr(dt.dt, "tz", None) is not None:
        dt_utc = dt.dt.tz_convert("UTC")
    else:
        dt_utc = dt

    years = dt_utc.dt.year.to_numpy()
    months = dt_utc.dt.month.to_numpy()
    days = dt_utc.dt.day.to_numpy()
    hours = dt_utc.dt.hour.to_numpy()
    minutes = dt_utc.dt.minute.to_numpy()
    seconds = (
        dt_utc.dt.second.to_numpy(dtype=float)
        + dt_utc.dt.microsecond.to_numpy(dtype=float) * 1e-6
    )
    return years, months, days, hours, minutes, seconds


def compute_sun_position_columns(
    df: pd.DataFrame,
    datetime_col: str,
    lat_deg: float,
    lon_deg: float,
    *,
    daylight_elevation_deg: float = 0.0,
    prefix: str = "sun_",
) -> pd.DataFrame:
    """
    Compute sun position for each row and append columns.

    Notes
    -----
    - Time is interpreted as UTC/UT (no timezone conversion inside sunpos()).
    - Longitude convention: East positive, West negative.
    - Azimuth convention: 0°=North, 90°=East, 180°=South, 270°=West.

    Adds (with prefix):
      zenith_rad, azimuth_rad, declination_rad, hour_angle_rad,
      elevation_rad, elevation_deg, azimuth_deg,
      is_daylight
    """
    if datetime_col not in df.columns:
        raise KeyError(f"Missing column: {datetime_col}")

    years, months, days, hours, minutes, seconds = _extract_utc_components(df[datetime_col])

    n = len(df)
    zen = np.empty(n, dtype=float)
    azi = np.empty(n, dtype=float)
    dec = np.empty(n, dtype=float)
    ha = np.empty(n, dtype=float)

    # For hourly TMY (8760 rows), this loop is fine.
    # If later you do minute-level (525,600 rows) we can vectorize in C++.
    for i in range(n):
        out = _sunpos(
            int(years[i]),
            int(months[i]),
            int(days[i]),
            int(hours[i]),
            int(minutes[i]),
            float(seconds[i]),
            float(lat_deg),
            float(lon_deg),
        )
        zen[i] = out["zenith_rad"]
        azi[i] = out["azimuth_rad"]
        dec[i] = out["declination_rad"]
        ha[i] = out["hour_angle_rad"]

    elev = 0.5 * np.pi - zen
    elev_deg = elev * RAD2DEG
    azi_deg = azi * RAD2DEG

    out_df = df.copy()
    out_df[f"{prefix}zenith_rad"] = zen
    out_df[f"{prefix}azimuth_rad"] = azi
    out_df[f"{prefix}declination_rad"] = dec
    out_df[f"{prefix}hour_angle_rad"] = ha
    out_df[f"{prefix}elevation_rad"] = elev
    out_df[f"{prefix}elevation_deg"] = elev_deg
    out_df[f"{prefix}azimuth_deg"] = azi_deg
    out_df[f"{prefix}is_daylight"] = elev_deg > daylight_elevation_deg

    return out_df