from pathlib import Path
import sys
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_sun_position_reference_days import (
    plot_sun_position_reference_days,
    select_reference_day_points,
)


def _write_sun_position_export_fixture(path: Path) -> None:
    rows = ["tmy_datetime_local,sun_azimuth_deg,sun_elevation_deg,sun_is_daylight,DNI,dni_clear_model"]
    for month, day in [(3, 20), (6, 21), (12, 21)]:
        for hour, azimuth, elevation in [
            (5, 82.0, -3.0),
            (8, 105.0, 12.0),
            (11, 150.0, 42.0),
            (14, 215.0, 38.0),
            (17, 260.0, 8.0),
            (20, 286.0, -4.0),
        ]:
            dni = max(0.0, elevation * 18.0)
            dni_clear = max(0.0, elevation * 20.0)
            rows.append(
                f"2001-{month:02d}-{day:02d} {hour:02d}:00:00,"
                f"{azimuth},{elevation},{str(elevation > 0.0)},{dni},{dni_clear}"
            )

    path.write_text("\n".join(rows), encoding="utf-8")


def test_select_reference_day_points_selects_three_daylight_dates(tmp_path: Path):
    csv_path = tmp_path / "sun_position_export.csv"
    _write_sun_position_export_fixture(csv_path)
    df = pd.read_csv(csv_path)

    selections = select_reference_day_points(df, irradiance_col="DNI")

    assert [selection.key for selection in selections] == [
        "winter_solstice",
        "spring_equinox",
        "summer_solstice",
    ]
    assert [str(selection.selected_date) for selection in selections] == [
        "2001-12-21",
        "2001-03-20",
        "2001-06-21",
    ]
    assert all((selection.points["sun_elevation_deg"] > 0.0).all() for selection in selections)
    assert [len(selection.points) for selection in selections] == [4, 4, 4]


def test_plot_sun_position_reference_days_creates_png_with_selected_irradiance(tmp_path: Path):
    csv_path = tmp_path / "sun_position_export.csv"
    out_path = tmp_path / "reference_days.png"
    _write_sun_position_export_fixture(csv_path)

    generated = plot_sun_position_reference_days(
        csv_path,
        out_path,
        location_name="Test Site",
        irradiance_col="dni_clear_model",
        connect_lines=True,
        label_values=True,
        vmin=0.0,
        vmax=1100.0,
        dpi=80,
    )

    assert generated == out_path
    assert generated.exists()
    assert generated.stat().st_size > 0


def test_plot_sun_position_reference_days_rejects_invalid_irradiance_column(tmp_path: Path):
    csv_path = tmp_path / "sun_position_export.csv"
    _write_sun_position_export_fixture(csv_path)

    with pytest.raises(ValueError, match="missing required column.*not_a_column"):
        plot_sun_position_reference_days(
            csv_path,
            tmp_path / "reference_days.png",
            irradiance_col="not_a_column",
            dpi=80,
        )


def test_plot_sun_position_reference_days_rejects_invalid_color_limits(tmp_path: Path):
    csv_path = tmp_path / "sun_position_export.csv"
    _write_sun_position_export_fixture(csv_path)

    with pytest.raises(ValueError, match="vmax must be greater than vmin"):
        plot_sun_position_reference_days(
            csv_path,
            tmp_path / "reference_days.png",
            vmin=1000.0,
            vmax=100.0,
            dpi=80,
        )


def test_plot_sun_position_reference_days_default_does_not_annotate_hours(tmp_path: Path):
    csv_path = tmp_path / "sun_position_export.csv"
    _write_sun_position_export_fixture(csv_path)

    with patch("matplotlib.axes.Axes.annotate", side_effect=AssertionError("hour labels should not be drawn")):
        generated = plot_sun_position_reference_days(
            csv_path,
            tmp_path / "reference_days.png",
            dpi=80,
        )

    assert generated.exists()
