from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_tmy_sun_position_dni import export_sun_position_dni


def _write_export_nsrdb_fixture(path: Path) -> None:
    rows = [
        "Source,Location ID,Latitude,Longitude,Elevation,Time Zone,Local Time Zone",
        "NSRDB,Export Test Site,33.01,-113.38,350,0,0",
        "Year,Month,Day,Hour,Minute,DNI,GHI",
    ]
    hours = pd.date_range("2001-01-01 00:00:00+00:00", periods=8760, freq="h")
    for ts in hours:
        rows.append(f"{ts.year},{ts.month},{ts.day},{ts.hour},{ts.minute},800,900")

    path.write_text("\n".join(rows), encoding="utf-8")


def test_export_sun_position_dni_includes_standard_datetime_columns(tmp_path: Path):
    tmy_csv = tmp_path / "nsrdb_export_fixture.csv"
    _write_export_nsrdb_fixture(tmy_csv)

    out = export_sun_position_dni(
        tmy_csv,
        tmp_path / "sun_position_dni.csv",
    )

    exported = pd.read_csv(out)

    expected = {
        "datetime",
        "tmy_datetime_local",
        "Year",
        "Month",
        "Day",
        "Hour",
        "Minute",
        "Second",
        "sun_azimuth_deg",
        "sun_elevation_deg",
        "DNI",
        "dni_clear_model",
        "Sun Azimuth (deg)",
        "Sun Elevation (deg)",
        "DNI (W/m2)",
        "Clear DNI (W/m2)",
    }
    assert expected.issubset(exported.columns)
    assert len(exported) == 8760
    assert exported["datetime"].str.endswith("+00:00").all()
    assert exported["tmy_datetime_local"].iloc[0] == "2001-01-01 00:00:00"

    daytime = exported["sun_elevation_deg"] >= 5.0
    assert daytime.sum() > 200
    assert np.isfinite(exported.loc[daytime, "dni_clear_model"]).all()
    assert (exported.loc[daytime, "dni_clear_model"] > 0.0).all()
