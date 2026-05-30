from pathlib import Path

from clear_day_analysis.tmy_reader import read_nsrdb_tmy_csv, read_tmy_csv


def _write_minimal_nsrdb(path: Path) -> None:
    csv_text = """Source,Location ID,Latitude,Longitude,Elevation,Time Zone,Local Time Zone
NSRDB,Test Site,33.01,-113.38,350,0,-7
Year,Month,Day,Hour,Minute,DNI,GHI
2020,1,1,12,30,800,500
2020,1,1,13,30,850,540
"""
    path.write_text(csv_text, encoding="utf-8")


def test_read_nsrdb_tmy_csv_basic(tmp_path: Path):
    p = tmp_path / "nsrdb.csv"
    _write_minimal_nsrdb(p)

    df, md = read_nsrdb_tmy_csv(p)

    assert len(df) == 2
    assert "datetime" in df.columns
    assert "DNI" in df.columns
    assert "GHI" in df.columns
    assert str(df["datetime"].iloc[0]) == "2020-01-01 12:30:00+00:00"
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

    assert len(df) == 2
    assert df["DNI"].iloc[0] == 800
    assert str(df["datetime"].iloc[0]) == "2020-01-01 12:30:00+00:00"
    assert md.source == "NSRDB"
