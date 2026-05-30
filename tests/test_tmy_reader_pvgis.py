from pathlib import Path

import pandas as pd

from clear_day_analysis import compute_sun_position_columns
from clear_day_analysis.day_classification import classify_days_by_ratio, daily_dni_integral_ratio
from clear_day_analysis.tmy_reader import read_pvgis_tmy_csv, read_tmy_csv


DATA_DIR = Path(__file__).parent / "data"
SYNTHETIC_YEAR = 2001


def test_read_pvgis_tmy_csv_basic():
    df, md = read_pvgis_tmy_csv(DATA_DIR / "pvgis_tmy_minimal.csv")

    assert len(df) == 3
    assert "datetime" in df.columns
    assert "pvgis_datetime_utc" in df.columns
    assert "DNI" in df.columns
    assert "GHI" in df.columns
    assert "DHI" in df.columns

    assert df["datetime"].iloc[0].tzinfo is not None
    assert df["datetime"].iloc[0].utcoffset().total_seconds() == 0
    assert str(df["pvgis_datetime_utc"].iloc[0]) == "2020-01-01 00:00:00+00:00"
    assert str(df["datetime"].iloc[0]) == "2001-01-01 00:00:00+00:00"

    assert df["DNI"].iloc[2] == 810
    assert df["GHI"].iloc[2] == 650
    assert df["DHI"].iloc[2] == 120

    assert md.source.startswith("PVGIS")
    assert "PVGIS-SARAH3" in md.source
    assert md.location_id == "pvgis_tmy_minimal"
    assert abs(md.latitude - 33.01) < 1e-9
    assert abs(md.longitude - (-113.38)) < 1e-9
    assert abs(md.elevation_m - 350.0) < 1e-9
    assert md.time_zone == 0.0
    assert md.local_time_zone == 0.0


def test_read_tmy_csv_auto_detects_pvgis():
    df, md = read_tmy_csv(DATA_DIR / "pvgis_tmy_minimal.csv")

    assert len(df) == 3
    assert df["DNI"].iloc[2] == 810
    assert df["GHI"].iloc[2] == 650
    assert md.source.startswith("PVGIS")


def test_read_pvgis_preserves_source_years_and_normalizes_calendar(tmp_path: Path):
    csv_text = """Latitude (decimal degrees): 39.063
Longitude (decimal degrees): -1.832
Elevation (m): 680.0
month,year
1,2020
2,2019
3,2020
time(UTC),T2m,RH,G(h),Gb(n),Gd(h),IR(h),WS10m,WD10m,SP
20200131:2300,5.0,80,0,0,0,250,1,180,94000
20190201:0000,6.0,75,0,0,0,252,1,180,94010
20200301:0000,7.0,70,10,20,5,260,1,180,94020
"""
    p = tmp_path / "pvgis_cross_year.csv"
    p.write_text(csv_text, encoding="utf-8")

    df, _ = read_pvgis_tmy_csv(p)

    assert df["pvgis_datetime_utc"].dt.year.tolist() == [2020, 2019, 2020]
    assert df["datetime"].dt.year.tolist() == [SYNTHETIC_YEAR] * 3
    assert df["datetime"].dt.month.tolist() == [1, 2, 3]
    assert df["datetime"].dt.day.tolist() == [31, 1, 1]
    assert df["datetime"].is_monotonic_increasing


def test_pvgis_8760_normalized_calendar_supports_downstream_grouping(tmp_path: Path):
    source_year_by_month = {
        1: 2020,
        2: 2019,
        3: 2020,
        4: 2020,
        5: 2018,
        6: 2015,
        7: 2010,
        8: 2010,
        9: 2012,
        10: 2012,
        11: 2020,
        12: 2006,
    }

    rows = [
        "Latitude (decimal degrees): 39.063",
        "Longitude (decimal degrees): -1.832",
        "Elevation (m): 680.0",
        "month,year",
    ]
    rows.extend(f"{month},{year}" for month, year in source_year_by_month.items())
    rows.append("time(UTC),T2m,RH,G(h),Gb(n),Gd(h),IR(h),WS10m,WD10m,SP")

    synthetic_hours = pd.date_range(
        f"{SYNTHETIC_YEAR}-01-01 00:00",
        periods=8760,
        freq="h",
        tz="UTC",
    )
    for ts in synthetic_hours:
        source_ts = ts.replace(year=source_year_by_month[int(ts.month)])
        rows.append(f"{source_ts.strftime('%Y%m%d:%H%M')},20,50,20,10,5,300,2,180,95000")

    p = tmp_path / "pvgis_8760.csv"
    p.write_text("\n".join(rows), encoding="utf-8")

    df, md = read_pvgis_tmy_csv(p)

    assert len(df) == 8760
    assert md.source.startswith("PVGIS")
    assert df["pvgis_datetime_utc"].dt.year.nunique() > 1
    assert df["datetime"].dt.year.unique().tolist() == [SYNTHETIC_YEAR]
    assert df["datetime"].is_monotonic_increasing
    assert df["datetime"].dt.date.nunique() == 365

    sun_df = compute_sun_position_columns(
        df.head(3),
        datetime_col="datetime",
        lat_deg=md.latitude,
        lon_deg=md.longitude,
    )
    assert "sun_elevation_deg" in sun_df.columns
    assert sun_df["datetime"].dt.year.eq(SYNTHETIC_YEAR).all()

    df = df.copy()
    df["dni_clear_model"] = 20.0
    daily = daily_dni_integral_ratio(
        df,
        datetime_col="datetime",
        dni_col="DNI",
        clear_col="dni_clear_model",
    )
    assert len(daily) == 365
    assert pd.to_datetime(daily["date"]).dt.year.unique().tolist() == [SYNTHETIC_YEAR]

    daily_cls = classify_days_by_ratio(daily)
    assert len(daily_cls) == 365
    assert set(daily_cls["class"]) == {"cloudy"}
