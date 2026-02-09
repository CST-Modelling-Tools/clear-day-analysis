import numpy as np
import pandas as pd

from clear_day_analysis.day_classification import (
    add_clear_dni_model,
    daily_dni_integral_ratio,
    classify_days_by_ratio,
)


def test_daily_integral_ratio_and_classification():
    dt1 = pd.date_range("2020-01-01 08:30:00+00:00", periods=10, freq="h")
    dt2 = pd.date_range("2020-01-02 08:30:00+00:00", periods=10, freq="h")
    datetime = dt1.append(dt2)

    elev = np.array([30.0] * 20)

    E0 = 1000.0
    beta = 0.10
    sin_alpha = np.sin(np.deg2rad(30.0))
    dni_clear_val = E0 * np.exp(-beta / sin_alpha)

    dni_day1 = np.full(10, 0.90 * dni_clear_val)  # extremely_clear boundary
    dni_day2 = np.full(10, 0.20 * dni_clear_val)  # extremely_cloudy
    dni = np.concatenate([dni_day1, dni_day2])

    df = pd.DataFrame({"datetime": datetime, "sun_elevation_deg": elev, "DNI": dni})

    df2 = add_clear_dni_model(
        df,
        E0=E0,
        beta=beta,
        dni_col="DNI",
        elevation_col="sun_elevation_deg",
        alpha_min_deg=5.0,
        clear_col="dni_clear_model",
    )

    daily = daily_dni_integral_ratio(
        df2,
        datetime_col="datetime",
        dni_col="DNI",
        clear_col="dni_clear_model",
        alpha_min_deg=5.0,
        elevation_col="sun_elevation_deg",
        sort_by_month_day=True,
    )

    r1 = float(daily.loc[daily["date"] == dt1[0].date(), "ratio"].iloc[0])
    r2 = float(daily.loc[daily["date"] == dt2[0].date(), "ratio"].iloc[0])

    assert abs(r1 - 0.90) < 1e-6
    assert abs(r2 - 0.20) < 1e-6

    daily_cls = classify_days_by_ratio(daily)

    c1 = daily_cls.loc[daily_cls["date"] == dt1[0].date(), "class"].iloc[0]
    c2 = daily_cls.loc[daily_cls["date"] == dt2[0].date(), "class"].iloc[0]

    assert c1 == "extremely_clear"
    assert c2 == "extremely_cloudy"