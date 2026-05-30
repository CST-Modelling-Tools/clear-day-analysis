from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_tmy_sun_position_dni import export_sun_position_dni


DATA_DIR = Path(__file__).parent / "data"


def test_export_sun_position_dni_includes_standard_datetime_columns(tmp_path: Path):
    out = export_sun_position_dni(
        DATA_DIR / "pvgis_tmy_minimal.csv",
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
        "Sun Azimuth (deg)",
        "Sun Elevation (deg)",
        "DNI (W/m2)",
    }
    assert expected.issubset(exported.columns)
    assert len(exported) == 3
    assert exported["datetime"].str.endswith("+00:00").all()
    assert exported["tmy_datetime_local"].iloc[0] == "2001-01-01 00:00:00"
