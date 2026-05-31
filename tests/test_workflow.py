from pathlib import Path

import numpy as np
import pandas as pd

from clear_day_analysis.workflow import ClearDNIModelSpec, run_clear_day_workflow


def _write_nsrdb_workflow_fixture(path: Path) -> None:
    rows = [
        "Source,Location ID,Latitude,Longitude,Elevation,Time Zone,Local Time Zone,",
        "NSRDB,Workflow Test Site,33.01,-113.38,350,0,0,",
        "Year,Month,Day,Hour,Minute,DNI,GHI,DHI",
    ]
    hours = pd.date_range("2001-01-01 00:00:00+00:00", periods=8760, freq="h")
    for ts in hours:
        rows.append(f"{ts.year},{ts.month},{ts.day},{ts.hour},{ts.minute},800,900,100")

    path.write_text("\n".join(rows), encoding="utf-8")


def test_run_clear_day_workflow_adds_requested_clear_models(tmp_path: Path):
    tmy_csv = tmp_path / "nsrdb_workflow_fixture.csv"
    _write_nsrdb_workflow_fixture(tmy_csv)

    result = run_clear_day_workflow(
        tmy_csv,
        daylight_elevation_deg=0.0,
        record_snapshots=True,
        clear_models=(
            ClearDNIModelSpec(
                clear_col="dni_clear_model_fit",
                alpha_min_deg=5.0,
            ),
            ClearDNIModelSpec(
                clear_col="dni_clear_model_export",
                alpha_min_deg=0.0,
                require_finite_dni=False,
                fill_value=0.0,
            ),
        ),
    )

    df = result.df

    assert result.metadata.source == "NSRDB"
    assert result.fit.E0 > 0.0
    assert result.fit.snapshots is not None
    assert len(result.fit.snapshots) >= 1
    assert "sun_azimuth_deg" in df.columns
    assert "sun_elevation_deg" in df.columns
    assert "sun_is_daylight" in df.columns
    assert "dni_clear_model_fit" in df.columns
    assert "dni_clear_model_export" in df.columns

    below_fit_domain = df["sun_elevation_deg"] < 5.0
    daytime = df["sun_elevation_deg"] > 0.0
    nighttime = df["sun_elevation_deg"] <= 0.0

    assert df.loc[below_fit_domain, "dni_clear_model_fit"].isna().all()
    assert np.isfinite(df.loc[daytime, "dni_clear_model_export"]).all()
    assert (df.loc[daytime, "dni_clear_model_export"] > 0.0).all()
    assert (df.loc[nighttime, "dni_clear_model_export"] == 0.0).all()
