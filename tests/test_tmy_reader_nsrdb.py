from pathlib import Path

import pandas as pd

from clear_day_analysis.day_classification import daily_dni_integral_ratio
from clear_day_analysis.tmy_reader import read_nsrdb_tmy_csv, read_tmy_csv


SYNTHETIC_YEAR = 2001


def _write_minimal_nsrdb(path: Path) -> None:
    csv_text = """Source,Location ID,Latitude,Longitude,Elevation,Time Zone,Local Time Zone
NSRDB,Test Site,33.01,-113.38,350,0,-7
Year,Month,Day,Hour,Minute,DNI,GHI
2020,1,31,23,30,800,500
2019,2,1,0,30,850,540
2020,3,1,0,30,700,450
"""
    path.write_text(csv_text, encoding="utf-8")


def test_read_nsrdb_tmy_csv_basic(tmp_path: Path):
    p = tmp_path / "nsrdb.csv"
    _write_minimal_nsrdb(p)

    df, md = read_nsrdb_tmy_csv(p)

    assert len(df) == 3
    assert "datetime" in df.columns
    assert "tmy_datetime_local" in df.columns
    assert "nsrdb_datetime_utc" in df.columns
    assert "DNI" in df.columns
    assert "GHI" in df.columns
    assert str(df["nsrdb_datetime_utc"].iloc[0]) == "2020-01-31 23:30:00+00:00"
    assert str(df["nsrdb_datetime_utc"].iloc[1]) == "2019-02-01 00:30:00+00:00"
    assert str(df["datetime"].iloc[0]) == "2001-01-31 23:30:00+00:00"
    assert str(df["datetime"].iloc[1]) == "2001-02-01 00:30:00+00:00"
    assert str(df["tmy_datetime_local"].iloc[0]) == "2001-01-31 16:30:00"
    assert str(df["tmy_datetime_local"].iloc[1]) == "2001-01-31 17:30:00"
    assert df["datetime"].dt.year.tolist() == [SYNTHETIC_YEAR] * 3
    assert df["tmy_datetime_local"].dt.year.tolist() == [SYNTHETIC_YEAR] * 3
    assert df["tmy_datetime_local"].dt.tz is None
    assert df["datetime"].is_monotonic_increasing
    assert df["datetime"].iloc[0].utcoffset().total_seconds() == 0

    assert df["DNI"].iloc[1] == 850
    assert df["GHI"].iloc[1] == 540
    assert md.source == "NSRDB"
    assert md.location_id == "Test Site"
    assert abs(md.latitude - 33.01) < 1e-9
    assert abs(md.longitude - (-113.38)) < 1e-9
    assert abs(md.elevation_m - 350.0) < 1e-9
    assert md.time_zone == 0.0
    assert md.local_time_zone == -7.0


def test_read_tmy_csv_auto_detects_nsrdb(tmp_path: Path):
    p = tmp_path / "nsrdb.csv"
    _write_minimal_nsrdb(p)

    df, md = read_tmy_csv(p, source="auto")

    assert len(df) == 3
    assert df["DNI"].iloc[0] == 800
    assert str(df["nsrdb_datetime_utc"].iloc[0]) == "2020-01-31 23:30:00+00:00"
    assert str(df["datetime"].iloc[0]) == "2001-01-31 23:30:00+00:00"
    assert str(df["tmy_datetime_local"].iloc[0]) == "2001-01-31 16:30:00"
    assert df["datetime"].is_monotonic_increasing
    assert md.source == "NSRDB"


def test_nsrdb_tmy_datetime_local_wraps_utc_boundary_for_hourly_year(tmp_path: Path):
    rows = [
        "Source,Location ID,Latitude,Longitude,Elevation,Time Zone,Local Time Zone",
        "NSRDB,Boundary Site,33.01,-113.38,350,0,-7",
        "Year,Month,Day,Hour,Minute,DNI,GHI",
    ]
    synthetic_hours = pd.date_range(
        f"{SYNTHETIC_YEAR}-01-01 00:30:00+00:00",
        periods=8760,
        freq="h",
    )
    for ts in synthetic_hours:
        source_ts = ts.replace(year=2020)
        rows.append(
            f"{source_ts.year},{source_ts.month},{source_ts.day},"
            f"{source_ts.hour},{source_ts.minute},10,20"
        )

    p = tmp_path / "nsrdb_8760.csv"
    p.write_text("\n".join(rows), encoding="utf-8")

    df, _ = read_nsrdb_tmy_csv(p)

    local = df["tmy_datetime_local"]
    assert len(df) == 8760
    assert local.dt.tz is None
    assert local.dt.year.unique().tolist() == [SYNTHETIC_YEAR]
    assert str(local.iloc[0]) == "2001-12-31 17:30:00"
    assert str(local.iloc[-1]) == "2001-12-31 16:30:00"

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
    dec31 = pd.Timestamp(f"{SYNTHETIC_YEAR}-12-31").date()
    assert int(daily.loc[daily["date"] == dec31, "n_points"].iloc[0]) == 24
    assert abs(float(daily["ratio"].mean()) - 0.5) < 1e-12
