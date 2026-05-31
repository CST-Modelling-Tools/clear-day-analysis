#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from clear_day_analysis.workflow import ClearDNIModelSpec, run_clear_day_workflow


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

    workflow = run_clear_day_workflow(
        input_csv,
        source="auto",
        daylight_elevation_deg=0.0,
        alpha_min_deg=5.0,
        confidence=0.95,
        outlier_mode="lower",
        max_iter=25,
        min_points=200,
        enforce_envelope=True,
        envelope_quantile=0.98,
        clear_models=(
            ClearDNIModelSpec(
                clear_col="dni_clear_model",
                alpha_min_deg=0.0,
                require_finite_dni=False,
                fill_value=0.0,
            ),
        ),
    )
    df = workflow.df
    fit = workflow.fit

    if print_fit_summary:
        print(
            "ASHRAE clear-day fit: "
            f"E0={fit.E0:.3f}, beta={fit.beta:.6f}, "
            f"n_final={fit.n_final}, converged={fit.converged}"
        )

    datetime_utc = pd.to_datetime(df["datetime"], utc=True)
    tmy_datetime_local = pd.to_datetime(df["tmy_datetime_local"])
    out_data = {
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
        "sun_is_daylight": df["sun_is_daylight"],
    }
    for col in ("DNI", "GHI", "DHI"):
        if col in df.columns:
            out_data[col] = df[col]
    out_data["dni_clear_model"] = df["dni_clear_model"]
    out = pd.DataFrame(out_data)

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
