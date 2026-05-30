#!/usr/bin/env python

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REFERENCE_DAYS = (
    ("winter_solstice", "Winter solstice", 12, 21, "o"),
    ("spring_equinox", "Spring equinox", 3, 20, "s"),
    ("summer_solstice", "Summer solstice", 6, 21, "^"),
)

DEFAULT_IRRADIANCE_COL = "DNI"
ALLOWED_IRRADIANCE_COLS = ("DNI", "dni_clear_model")


@dataclass(frozen=True)
class ReferenceDaySelection:
    key: str
    label: str
    selected_date: object
    points: pd.DataFrame
    marker: str


def _apply_report_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (13.33, 7.50),
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "text.color": "#222222",
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "grid.color": "#D0D0D0",
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.7,
        }
    )


def _load_export_csv(path: Path, irradiance_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"tmy_datetime_local", "sun_azimuth_deg", "sun_elevation_deg", irradiance_col}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Export CSV is missing required column(s): {', '.join(missing)}")

    out = df.copy()
    out["tmy_datetime_local"] = pd.to_datetime(out["tmy_datetime_local"], errors="coerce")
    out["sun_azimuth_deg"] = pd.to_numeric(out["sun_azimuth_deg"], errors="coerce")
    out["sun_elevation_deg"] = pd.to_numeric(out["sun_elevation_deg"], errors="coerce")
    out[irradiance_col] = pd.to_numeric(out[irradiance_col], errors="coerce")

    if out["tmy_datetime_local"].isna().any():
        n_bad = int(out["tmy_datetime_local"].isna().sum())
        raise ValueError(f"Could not parse {n_bad} tmy_datetime_local value(s).")

    return out


def _nearest_available_date(available_dates: pd.Series, target: pd.Timestamp) -> object:
    candidates = pd.to_datetime(available_dates)
    idx = (candidates - target).abs().argmin()
    return candidates.iloc[int(idx)].date()


def select_reference_day_points(
    df: pd.DataFrame,
    *,
    irradiance_col: str = DEFAULT_IRRADIANCE_COL,
) -> list[ReferenceDaySelection]:
    """
    Select daylight rows for winter solstice, spring equinox, and summer solstice.

    Reference days are selected from tmy_datetime_local. If an exact target date
    is unavailable, the nearest available daylight date is used.
    """
    required = {"tmy_datetime_local", "sun_azimuth_deg", "sun_elevation_deg", irradiance_col}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"DataFrame is missing required column(s): {', '.join(missing)}")

    work = df.copy()
    work["tmy_datetime_local"] = pd.to_datetime(work["tmy_datetime_local"], errors="coerce")
    work["sun_azimuth_deg"] = pd.to_numeric(work["sun_azimuth_deg"], errors="coerce")
    work["sun_elevation_deg"] = pd.to_numeric(work["sun_elevation_deg"], errors="coerce")
    work[irradiance_col] = pd.to_numeric(work[irradiance_col], errors="coerce")

    valid = (
        work["tmy_datetime_local"].notna()
        & np.isfinite(work["sun_azimuth_deg"])
        & np.isfinite(work["sun_elevation_deg"])
        & np.isfinite(work[irradiance_col])
    )
    daylight = valid & (work["sun_elevation_deg"] > 0.0)
    if not daylight.any():
        raise ValueError("No daylight rows found; expected sun_elevation_deg > 0.")

    work = work.loc[valid].copy()
    work["_date"] = work["tmy_datetime_local"].dt.date
    year = int(work.loc[daylight, "tmy_datetime_local"].dt.year.mode().iloc[0])
    available_dates = pd.Series(sorted(work.loc[daylight, "_date"].unique()))

    selections: list[ReferenceDaySelection] = []
    for key, label, month, day, marker in REFERENCE_DAYS:
        target = pd.Timestamp(year=year, month=month, day=day)
        selected_date = _nearest_available_date(available_dates, target)
        points = (
            work.loc[(work["_date"] == selected_date) & (work["sun_elevation_deg"] > 0.0)]
            .sort_values("tmy_datetime_local")
            .drop(columns=["_date"])
            .reset_index(drop=True)
        )
        if points.empty:
            raise ValueError(f"No daylight points found for {label} ({selected_date}).")
        selections.append(
            ReferenceDaySelection(
                key=key,
                label=label,
                selected_date=selected_date,
                points=points,
                marker=marker,
            )
        )

    return selections


def plot_sun_position_reference_days(
    export_csv: Path,
    output_png: Path | None = None,
    *,
    location_name: str | None = None,
    irradiance_col: str = DEFAULT_IRRADIANCE_COL,
    connect_lines: bool = False,
    dpi: int = 200,
) -> Path:
    export_csv = Path(export_csv)
    if output_png is not None:
        output_png = Path(output_png)
    if not export_csv.exists():
        raise FileNotFoundError(f"Export CSV not found: {export_csv}")

    df = _load_export_csv(export_csv, irradiance_col)
    selections = select_reference_day_points(df, irradiance_col=irradiance_col)

    if output_png is None:
        output_png = export_csv.parent / f"{export_csv.stem}_sun_position_reference_days.png"

    _apply_report_style()
    fig, ax = plt.subplots(figsize=(13.33, 7.50))
    values = pd.concat([selection.points[irradiance_col] for selection in selections], ignore_index=True)
    vmin = float(max(0.0, values.min()))
    vmax = float(values.max())
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0

    cmap = "viridis"
    scatter_for_colorbar = None

    for selection in selections:
        pts = selection.points
        if connect_lines:
            ax.plot(
                pts["sun_azimuth_deg"],
                pts["sun_elevation_deg"],
                color="#333333",
                linewidth=1.0,
                alpha=0.22,
                zorder=1,
            )
        scatter = ax.scatter(
            pts["sun_azimuth_deg"],
            pts["sun_elevation_deg"],
            c=pts[irradiance_col],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            marker=selection.marker,
            s=72,
            edgecolor="#1A1A1A",
            linewidth=0.7,
            alpha=0.95,
            zorder=3,
        )
        scatter_for_colorbar = scatter

    title = "Sun Position Reference Days"
    subtitle_base = location_name or export_csv.stem
    subtitle = f"{subtitle_base} - local standard TMY time; marker color shows {irradiance_col}"
    fig.suptitle(title, x=0.5, y=0.985, ha="center", va="top", fontweight="bold")
    fig.text(0.5, 0.952, subtitle, ha="center", va="top", fontsize=12)

    ax.set_xlabel("Sun azimuth angle (deg)")
    ax.set_ylabel("Sun elevation angle (deg)")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.30, linewidth=0.8)
    ax.axhline(0.0, color="#222222", linewidth=0.9, alpha=0.75)
    handles = [
        Line2D(
            [0],
            [0],
            marker=selection.marker,
            color="none",
            markerfacecolor="#BDBDBD",
            markeredgecolor="#1A1A1A",
            markersize=8,
            label=f"{selection.label} ({pd.Timestamp(selection.selected_date).strftime('%b %d')})",
        )
        for selection in selections
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, facecolor="white", edgecolor="#DDDDDD")
    if scatter_for_colorbar is not None:
        label = "DNI (W/m2)" if irradiance_col == "DNI" else "Clear-day DNI model (W/m2)"
        cbar = fig.colorbar(scatter_for_colorbar, ax=ax, pad=0.02)
        cbar.set_label(label)
    fig.subplots_adjust(top=0.88, right=0.90)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_png


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot winter solstice, spring equinox, and summer solstice sun-position reference curves.",
    )
    parser.add_argument(
        "export_csv",
        type=Path,
        help="Path to the CSV generated by export_tmy_sun_position_dni.py.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path. Defaults next to the input CSV.",
    )
    parser.add_argument(
        "--location-name",
        type=str,
        default=None,
        help='Optional human-readable location name, e.g. "Hyder, AZ".',
    )
    parser.add_argument(
        "--irradiance-col",
        type=str,
        default=DEFAULT_IRRADIANCE_COL,
        choices=ALLOWED_IRRADIANCE_COLS,
        help="Column used for marker color.",
    )
    parser.add_argument(
        "--connect-lines",
        action="store_true",
        help="Draw subtle solar-path guide lines behind the irradiance-colored markers.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PNG output dpi.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        output_png = plot_sun_position_reference_days(
            args.export_csv,
            args.output,
            location_name=args.location_name,
            irradiance_col=args.irradiance_col,
            connect_lines=args.connect_lines,
            dpi=args.dpi,
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Wrote: {output_png}")


if __name__ == "__main__":
    main()
