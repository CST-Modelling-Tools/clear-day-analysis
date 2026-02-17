from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from clear_day_analysis.tmy_reader import read_nsrdb_tmy_csv, read_solargis_tmy60_p50_csv
from clear_day_analysis import compute_sun_position_columns
from clear_day_analysis.ashrae_clear_day import fit_ashrae_clear_day
from clear_day_analysis.day_classification import (
    add_clear_dni_model,
    daily_dni_integral_ratio,
    classify_days_by_ratio,
)
from clear_day_analysis.plots import (
    PlotContext,
    make_plot_paths,
    plot_fit_iterations,
    plot_fit_final_summary,
    plot_clearness_index_timeseries,
    plot_day_examples_grid,
    plot_annual_energy_by_class,
    plot_cumulative_energy,
    plot_seasonal_energy_by_class,
)


def _read_tmy_auto(path: Path):
    """
    Read either NSRDB-style or Solargis TMY60_P50 CSV.
    Detection is based on first header lines.
    """
    try:
        with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
            head = "".join([f.readline() for _ in range(20)])
    except Exception:
        head = ""

    h = head.lower()
    if ("solargis_tmy60_p50" in h) or ("#typical meteorological year" in h and "#data:" in h):
        return read_solargis_tmy60_p50_csv(path)

    return read_nsrdb_tmy_csv(path)


def run_all(
    tmy_csv: Path,
    *,
    dpi: int = 150,
    max_iter_plots: int | None = None,
    alpha_min_deg: float = 5.0,
    confidence: float = 0.95,
    envelope_quantile: float = 0.98,
    outlier_mode: str = "lower",
    location_name: str | None = None,
) -> None:
    # --- 1) Load TMY ---
    df, md = _read_tmy_auto(tmy_csv)

    # --- 2) Solar position (computed using UTC timestamps) ---
    df = compute_sun_position_columns(
        df,
        datetime_col="datetime",
        lat_deg=md.latitude,
        lon_deg=md.longitude,
        daylight_elevation_deg=2.0,
    )

    # --- 3) Create local standard time timestamps for correct day grouping/plots ---
    # NSRDB metadata provides local_time_zone in hours (e.g., -7 for Arizona)
    # We use local standard time (no DST handling) which is appropriate for NSRDB TMY usage.
    df["datetime_local"] = df["datetime"] + pd.to_timedelta(md.local_time_zone, unit="h")

    # --- 3b) Plot context (titles/subtitles) ---
    # PlotContext is optional, but enables the nicer headers in plots.py
    context = PlotContext(
        location_name=location_name or f"Location {md.location_id}",
        latitude=float(md.latitude),
        longitude=float(md.longitude),
        tmy_label=tmy_csv.stem,
        source=str(md.source),
    )

    # --- 4) Fit clear-day envelope (record snapshots for iteration plots) ---
    fit = fit_ashrae_clear_day(
        df,
        dni_col="DNI",
        elevation_col="sun_elevation_deg",
        alpha_min_deg=alpha_min_deg,
        confidence=confidence,
        outlier_mode=outlier_mode,
        max_iter=25,
        min_points=200,
        enforce_envelope=True,
        envelope_quantile=envelope_quantile,
        record_snapshots=True,
    )

    # --- 5) Add two clear-envelope DNI columns using FINAL parameters ---
    # 5a) For fit/integrals/classification
    df = add_clear_dni_model(
        df,
        E0=fit.E0,
        beta=fit.beta,
        dni_col="DNI",
        elevation_col="sun_elevation_deg",
        alpha_min_deg=alpha_min_deg,
        clear_col="dni_clear_model_fit",
    )

    # 5b) For plotting: full daylight curve (sunrise -> sunset)
    df = add_clear_dni_model(
        df,
        E0=fit.E0,
        beta=fit.beta,
        dni_col="DNI",
        elevation_col="sun_elevation_deg",
        alpha_min_deg=0.0,
        clear_col="dni_clear_model_plot",
    )

    # --- 6) Daily integrals + ratio (use LOCAL datetime for day boundaries) ---
    daily = daily_dni_integral_ratio(
        df,
        datetime_col="datetime_local",
        dni_col="DNI",
        clear_col="dni_clear_model_fit",
        alpha_min_deg=alpha_min_deg,
        elevation_col="sun_elevation_deg",
        sort_by_month_day=True,
    )

    # --- 7) Classify days ---
    daily_cls = classify_days_by_ratio(daily)

    # --- 8) Write daily CSV next to TMY ---
    out_csv = tmy_csv.parent / f"{tmy_csv.stem}_daily_classification.csv"
    daily_cls.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")

    # --- 9) Plots (next to TMY) ---
    paths = make_plot_paths(tmy_csv)

    p1 = plot_fit_iterations(fit, paths, context=context, max_plots=max_iter_plots, dpi=dpi)
    print(f"Wrote {len(p1)} iteration plots in: {paths.iterations_dir}")

    p1b = plot_fit_final_summary(fit, paths, context=context, dpi=dpi)
    print(f"Wrote: {p1b}")

    p2 = plot_clearness_index_timeseries(daily_cls, paths, context=context, dpi=dpi)
    print(f"Wrote: {p2}")

    p3 = plot_day_examples_grid(
        df,
        daily_cls,
        paths,
        datetime_col="datetime_local",
        clear_col="dni_clear_model_plot",
        utc_offset_hours=float(md.local_time_zone),
        context=context,
        dpi=dpi,
    )
    print(f"Wrote: {p3}")

    p4 = plot_annual_energy_by_class(daily_cls, paths, context=context, dpi=dpi)
    print(f"Wrote: {p4}")

    p5a = plot_cumulative_energy(daily_cls, paths, context=context, dpi=dpi)
    print(f"Wrote: {p5a}")

    p5b = plot_seasonal_energy_by_class(daily_cls, paths, context=context, dpi=dpi)
    print(f"Wrote: {p5b}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="make_plots",
        description="Generate clear-day fit and DNI classification plots from an NSRDB TMY CSV.",
    )
    p.add_argument("tmy_csv", type=str, help="Path to NSRDB TMY CSV")
    p.add_argument("--dpi", type=int, default=150, help="PNG output dpi")
    p.add_argument(
        "--max-iter-plots",
        type=int,
        default=None,
        help="Limit the number of iteration plots (useful for quick checks).",
    )

    p.add_argument("--alpha-min-deg", type=float, default=5.0, help="Minimum solar elevation used for fit/integrals.")
    p.add_argument("--confidence", type=float, default=0.95, help="Student-t corridor confidence level.")
    p.add_argument("--envelope-quantile", type=float, default=0.98, help="Quantile used to shift fit into an envelope.")
    p.add_argument(
        "--outlier-mode",
        type=str,
        default="lower",
        choices=["lower", "two_sided"],
        help='Outlier rejection mode ("lower" recommended for DNI).',
    )

    # Optional, for prettier slide titles (if you know the human-readable name)
    p.add_argument(
        "--location-name",
        type=str,
        default=None,
        help='Optional human-readable name, e.g. "Hyder, AZ". Defaults to "Location <id>".',
    )

    return p


def main() -> None:
    args = build_parser().parse_args()
    tmy_csv = Path(args.tmy_csv)

    if not tmy_csv.exists():
        raise SystemExit(f"TMY CSV not found: {tmy_csv}")

    run_all(
        tmy_csv,
        dpi=args.dpi,
        max_iter_plots=args.max_iter_plots,
        alpha_min_deg=args.alpha_min_deg,
        confidence=args.confidence,
        envelope_quantile=args.envelope_quantile,
        outlier_mode=args.outlier_mode,
        location_name=args.location_name,
    )


if __name__ == "__main__":
    main()
