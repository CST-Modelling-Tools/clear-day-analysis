from pathlib import Path

import pandas as pd

from clear_day_analysis.day_classification import daily_dni_integral_ratio
from clear_day_analysis.tmy_reader import read_solargis_tmy60_p50_csv, read_tmy_csv


SYNTHETIC_YEAR = 2001


def test_read_solargis_tmy60_p50_csv_basic(tmp_path: Path):
    # Minimal Solargis-like file:
    # metadata rows + data header row + data rows
    csv_text = """Source,Solargis_TMY60_P50
Site name,Hyder AZ
Latitude,33.01
Longitude,-113.38
Elevation,350
Local Time Zone,UTC-7
Year,Month,Day,Hour,Minute,DNI,GHI
2020,1,1,12,0,800,500
2020,1,1,13,0,850,540
"""
    p = tmp_path / "solargis.csv"
    p.write_text(csv_text, encoding="utf-8")

    df, md = read_solargis_tmy60_p50_csv(p)

    assert len(df) == 2
    assert "datetime" in df.columns
    assert "tmy_datetime_local" in df.columns
    assert "solargis_datetime_utc" in df.columns
    assert "DNI" in df.columns
    assert "GHI" in df.columns

    # UTC-7 local time -> UTC add 7h
    assert str(df["solargis_datetime_utc"].iloc[0]) == "2020-01-01 19:00:00+00:00"
    assert str(df["datetime"].iloc[0]) == "2001-01-01 19:00:00+00:00"
    assert str(df["datetime"].iloc[1]) == "2001-01-01 20:00:00+00:00"
    assert str(df["tmy_datetime_local"].iloc[0]) == "2001-01-01 12:00:00"
    assert str(df["tmy_datetime_local"].iloc[1]) == "2001-01-01 13:00:00"
    assert df["datetime"].dt.year.tolist() == [SYNTHETIC_YEAR] * 2
    assert df["tmy_datetime_local"].dt.tz is None
    assert df["datetime"].is_monotonic_increasing

    assert md.source == "Solargis_TMY60_P50"
    assert md.location_id == "Hyder AZ"
    assert abs(md.latitude - 33.01) < 1e-9
    assert abs(md.longitude - (-113.38)) < 1e-9
    assert abs(md.local_time_zone - (-7.0)) < 1e-9


def test_read_tmy_csv_auto_detects_solargis(tmp_path: Path):
    csv_text = """Source,Solargis_TMY60_P50
Site name,Hyder AZ
Latitude,33.01
Longitude,-113.38
Elevation,350
Local Time Zone,UTC-7
Year,Month,Day,Hour,Minute,DNI,GHI
2020,1,1,12,0,800,500
2020,1,1,13,0,850,540
"""
    p = tmp_path / "solargis.csv"
    p.write_text(csv_text, encoding="utf-8")

    df, md = read_tmy_csv(p, source="auto")

    assert len(df) == 2
    assert df["DNI"].iloc[0] == 800
    assert str(df["solargis_datetime_utc"].iloc[0]) == "2020-01-01 19:00:00+00:00"
    assert str(df["datetime"].iloc[0]) == "2001-01-01 19:00:00+00:00"
    assert str(df["tmy_datetime_local"].iloc[0]) == "2001-01-01 12:00:00"
    assert df["datetime"].is_monotonic_increasing
    assert md.source == "Solargis_TMY60_P50"


def test_read_solargis_preserves_source_years_and_normalizes_calendar(tmp_path: Path):
    csv_text = """Source,Solargis_TMY60_P50
Site name,Hyder AZ
Latitude,33.01
Longitude,-113.38
Elevation,350
Local Time Zone,UTC+0
Year,Month,Day,Hour,Minute,DNI,GHI
2020,1,31,23,0,800,500
2019,2,1,0,0,850,540
2020,3,1,0,0,700,450
"""
    p = tmp_path / "solargis_cross_year.csv"
    p.write_text(csv_text, encoding="utf-8")

    df, _ = read_solargis_tmy60_p50_csv(p)

    assert df["solargis_datetime_utc"].dt.year.tolist() == [2020, 2019, 2020]
    assert df["datetime"].dt.year.tolist() == [SYNTHETIC_YEAR] * 3
    assert df["datetime"].dt.month.tolist() == [1, 2, 3]
    assert df["datetime"].dt.day.tolist() == [31, 1, 1]
    assert df["tmy_datetime_local"].dt.year.tolist() == [SYNTHETIC_YEAR] * 3
    assert df["datetime"].is_monotonic_increasing


def test_read_solargis_tmy60_p50_report_style(tmp_path: Path):
    csv_text = """#TYPICAL METEOROLOGICAL YEAR (P50) - HOURLY VALUES
#File type: Solargis_TMY60_P50
#Site name: Asab - Abu Dhabi - United Arab Emirates (AE)
#Latitude: 23.363652
#Longitude: 54.357946
#Elevation: 128.0 m a.s.l.
#TMY created from selected years: 1:2006 2:2003
#Columns:
#Day - Day of year
#Time - Time reference UTC+4, time step 60 min, time format HH:MM
#Data:
Day;Time;GHI;DNI;DIF
1;0:30;0;0;0
1;1:30;0;0;0
32;0:30;0;0;0
"""
    p = tmp_path / "solargis_report.csv"
    p.write_text(csv_text, encoding="utf-8")

    df, md = read_solargis_tmy60_p50_csv(p)

    assert len(df) == 3
    assert "datetime" in df.columns
    assert "tmy_datetime_local" in df.columns
    assert "solargis_datetime_utc" in df.columns
    assert "DNI" in df.columns
    assert "GHI" in df.columns

    # UTC+4 local time -> UTC subtract 4h
    assert str(df["solargis_datetime_utc"].iloc[0]) == "2005-12-31 20:30:00+00:00"
    assert str(df["solargis_datetime_utc"].iloc[2]) == "2003-01-31 20:30:00+00:00"
    assert str(df["datetime"].iloc[0]) == "2000-12-31 20:30:00+00:00"
    assert str(df["datetime"].iloc[1]) == "2000-12-31 21:30:00+00:00"
    assert str(df["datetime"].iloc[2]) == "2001-01-31 20:30:00+00:00"
    assert str(df["tmy_datetime_local"].iloc[0]) == "2001-01-01 00:30:00"
    assert str(df["tmy_datetime_local"].iloc[1]) == "2001-01-01 01:30:00"
    assert str(df["tmy_datetime_local"].iloc[2]) == "2001-02-01 00:30:00"
    assert df["tmy_datetime_local"].dt.tz is None
    assert df["datetime"].is_monotonic_increasing

    assert md.source == "Solargis_TMY60_P50"
    assert "Asab" in md.location_id
    assert abs(md.latitude - 23.363652) < 1e-9
    assert abs(md.longitude - 54.357946) < 1e-9
    assert abs(md.local_time_zone - 4.0) < 1e-9


def test_solargis_report_style_tmy_datetime_local_has_365_complete_days(tmp_path: Path):
    rows = [
        "#TYPICAL METEOROLOGICAL YEAR (P50) - HOURLY VALUES",
        "#File type: Solargis_TMY60_P50",
        "#Site name: Abu Dhabi Test",
        "#Latitude: 23.363652",
        "#Longitude: 54.357946",
        "#Elevation: 128.0 m a.s.l.",
        "#Columns:",
        "#Day - Day of year",
        "#Time - Time reference UTC+4, time step 60 min, time format HH:MM",
        "#Data:",
        "Day;Time;GHI;DNI;DIF",
    ]
    for day in range(1, 366):
        for hour in range(24):
            rows.append(f"{day};{hour}:30;20;10;5")

    p = tmp_path / "solargis_report_8760.csv"
    p.write_text("\n".join(rows), encoding="utf-8")

    df, _ = read_solargis_tmy60_p50_csv(p)

    local = df["tmy_datetime_local"]
    assert len(df) == 8760
    assert str(df["datetime"].iloc[0]) == "2000-12-31 20:30:00+00:00"
    assert str(df["datetime"].iloc[-1]) == "2001-12-31 19:30:00+00:00"
    assert str(local.iloc[0]) == "2001-01-01 00:30:00"
    assert str(local.iloc[-1]) == "2001-12-31 23:30:00"
    assert local.dt.tz is None

    date_counts = local.dt.date.value_counts()
    assert len(date_counts) == 365
    assert date_counts.unique().tolist() == [24]

    df = df.copy()
    df["dni_clear_model"] = 20.0
    daily = daily_dni_integral_ratio(
        df,
        datetime_col="tmy_datetime_local",
        dni_col="DNI",
        clear_col="dni_clear_model",
    )

    assert len(daily) == 365
    assert pd.to_datetime(daily["date"]).dt.year.unique().tolist() == [SYNTHETIC_YEAR]
    assert abs(float(daily["ratio"].mean()) - 0.5) < 1e-12
