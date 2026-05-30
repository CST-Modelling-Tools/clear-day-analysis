#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from clear_day_analysis import compute_sun_position_columns
from clear_day_analysis.tmy_reader import read_tmy_csv


def export_sun_position_dni(input_csv: Path, output_csv: Path | None = None) -> Path:
    input_csv = Path(input_csv)
    if output_csv is not None:
        output_csv = Path(output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"TMY CSV not found: {input_csv}")

    df, md = read_tmy_csv(input_csv, source="auto")

    missing = [col for col in ("datetime", "DNI") if col not in df.columns]
    if missing:
        raise ValueError(f"Parsed TMY CSV is missing required column(s): {', '.join(missing)}")

    df = compute_sun_position_columns(
        df,
        datetime_col="datetime",
        lat_deg=md.latitude,
        lon_deg=md.longitude,
        daylight_elevation_deg=0.0,
    )

    datetime_utc = pd.to_datetime(df["datetime"], utc=True)
    out = pd.DataFrame(
        {
            "Year": datetime_utc.dt.year,
            "Month": datetime_utc.dt.month,
            "Day": datetime_utc.dt.day,
            "Hour": datetime_utc.dt.hour,
            "Minute": datetime_utc.dt.minute,
            "Second": datetime_utc.dt.second,
            "Sun Azimuth (deg)": df["sun_azimuth_deg"],
            "Sun Elevation (deg)": df["sun_elevation_deg"],
            "DNI (W/m2)": df["DNI"],
        }
    )

    if output_csv is None:
        output_csv = input_csv.parent / f"{input_csv.stem}_sun_position_dni_utc.csv"

    out.to_csv(output_csv, index=False)
    return output_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export normalized UTC time, sun position, and DNI columns from a TMY CSV.",
    )
    parser.add_argument("input_csv", type=Path, help="Path to NSRDB, Solargis, or PVGIS TMY CSV")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to next to the input file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        output_csv = export_sun_position_dni(args.input_csv, args.output)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Wrote: {output_csv}")


if __name__ == "__main__":
    main()
