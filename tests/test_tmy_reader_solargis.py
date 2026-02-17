from pathlib import Path

from clear_day_analysis.tmy_reader import read_solargis_tmy60_p50_csv


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
    assert "DNI" in df.columns
    assert "GHI" in df.columns

    # UTC-7 local time -> UTC add 7h
    assert str(df["datetime"].iloc[0]) == "2020-01-01 19:00:00+00:00"
    assert str(df["datetime"].iloc[1]) == "2020-01-01 20:00:00+00:00"

    assert md.source == "Solargis_TMY60_P50"
    assert md.location_id == "Hyder AZ"
    assert abs(md.latitude - 33.01) < 1e-9
    assert abs(md.longitude - (-113.38)) < 1e-9
    assert abs(md.local_time_zone - (-7.0)) < 1e-9


def test_read_solargis_tmy60_p50_report_style(tmp_path: Path):
    csv_text = """#TYPICAL METEOROLOGICAL YEAR (P50) - HOURLY VALUES
#File type: Solargis_TMY60_P50
#Site name: Asab - Abu Dhabi - United Arab Emirates (AE)
#Latitude: 23.363652
#Longitude: 54.357946
#Elevation: 128.0 m a.s.l.
#Columns:
#Day - Day of year
#Time - Time reference UTC+4, time step 60 min, time format HH:MM
#Data:
Day;Time;GHI;DNI;DIF
1;0:30;0;0;0
1;1:30;0;0;0
"""
    p = tmp_path / "solargis_report.csv"
    p.write_text(csv_text, encoding="utf-8")

    df, md = read_solargis_tmy60_p50_csv(p)

    assert len(df) == 2
    assert "datetime" in df.columns
    assert "DNI" in df.columns
    assert "GHI" in df.columns

    # UTC+4 local time -> UTC subtract 4h
    assert str(df["datetime"].iloc[0]) == "2000-12-31 20:30:00+00:00"
    assert str(df["datetime"].iloc[1]) == "2000-12-31 21:30:00+00:00"

    assert md.source == "Solargis_TMY60_P50"
    assert "Asab" in md.location_id
    assert abs(md.latitude - 23.363652) < 1e-9
    assert abs(md.longitude - 54.357946) < 1e-9
    assert abs(md.local_time_zone - 4.0) < 1e-9
