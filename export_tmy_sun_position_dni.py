#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from clear_day_analysis import compute_sun_position_columns
from clear_day_analysis.ashrae_clear_day import fit_ashrae_clear_day
from clear_day_analysis.day_classification import add_clear_dni_model
from clear_day_analysis.tmy_reader import read_tmy_csv


def export_sun_position_dni(
    input_csv: Path,
    output_csv: Path | None = None,
    *,
    print_fit_summary: bool = False,
) -> Path:
    input_csv = Path(input_csv)
    if output_csv is not None:
        output_csv = Path(output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"TMY CSV not found: {input_csv}")

    df, md = read_tmy_csv(input_csv, source="auto")

    missing = [col for col in ("datetime", "tmy_datetime_local", "DNI") if col not in df.columns]
    if missing:
        raise ValueError(f"Parsed TMY CSV is missing required column(s): {', '.join(missing)}")

    df = compute_sun_position_columns(
        df,
        datetime_col="datetime",
        lat_deg=md.latitude,
        lon_deg=md.longitude,
        daylight_elevation_deg=0.0,
    )

    fit = fit_ashrae_clear_day(
        df,
        dni_col="DNI",
        elevation_col="sun_elevation_deg",
        alpha_min_deg=5.0,
        confidence=0.95,
        outlier_mode="lower",
        max_iter=25,
        min_points=200,
        enforce_envelope=True,
        envelope_quantile=0.98,
    )
    df = add_clear_dni_model(
        df,
        E0=fit.E0,
        beta=fit.beta,
        dni_col="DNI",
        elevation_col="sun_elevation_deg",
        alpha_min_deg=5.0,
        clear_col="dni_clear_model",
    )

    if print_fit_summary:
        print(
            "ASHRAE clear-day fit: "
            f"E0={fit.E0:.3f}, beta={fit.beta:.6f}, "
            f"n_final={fit.n_final}, converged={fit.converged}"
        )

    datetime_utc = pd.to_datetime(df["datetime"], utc=True)
    tmy_datetime_local = pd.to_datetime(df["tmy_datetime_local"])
    out = pd.DataFrame(
        {
            "datetime": datetime_utc.astype(str),
            "tmy_datetime_local": tmy_datetime_local.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Year": datetime_utc.dt.year,
            "Month": datetime_utc.dt.month,
            "Day": datetime_utc.dt.day,
            "Hour": datetime_utc.dt.hour,
            "Minute": datetime_utc.dt.minute,
            "Second": datetime_utc.dt.second,
            "sun_azimuth_deg": df["sun_azimuth_deg"],
            "sun_elevation_deg": df["sun_elevation_deg"],
            "DNI": df["DNI"],
            "dni_clear_model": df["dni_clear_model"],
            "Sun Azimuth (deg)": df["sun_azimuth_deg"],
            "Sun Elevation (deg)": df["sun_elevation_deg"],
            "DNI (W/m2)": df["DNI"],
            "Clear DNI (W/m2)": df["dni_clear_model"],
        }
    )

    if output_csv is None:
        output_csv = input_csv.parent / f"{input_csv.stem}_sun_position_dni_utc.csv"

    out.to_csv(output_csv, index=False)
    return output_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export normalized UTC/local TMY time, sun position, measured DNI, "
            "and fitted ASHRAE clear-day DNI from a TMY CSV."
        ),
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
        output_csv = export_sun_position_dni(args.input_csv, args.output, print_fit_summary=True)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Wrote: {output_csv}")


if __name__ == "__main__":
    main()
