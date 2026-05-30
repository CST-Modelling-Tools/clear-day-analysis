#!/usr/bin/env python

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REFERENCE_DAYS = (
    ("winter_solstice", "Winter solstice", 12, 21, "#2166AC"),
    ("spring_equinox", "Spring equinox", 3, 20, "#238B45"),
    ("summer_solstice", "Summer solstice", 6, 21, "#D73027"),
)


@dataclass(frozen=True)
class ReferenceDaySelection:
    key: str
    label: str
    selected_date: object
    points: pd.DataFrame
    color: str


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


def _load_export_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"tmy_datetime_local", "sun_azimuth_deg", "sun_elevation_deg"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Export CSV is missing required column(s): {', '.join(missing)}")

    out = df.copy()
    out["tmy_datetime_local"] = pd.to_datetime(out["tmy_datetime_local"], errors="coerce")
    out["sun_azimuth_deg"] = pd.to_numeric(out["sun_azimuth_deg"], errors="coerce")
    out["sun_elevation_deg"] = pd.to_numeric(out["sun_elevation_deg"], errors="coerce")

    if out["tmy_datetime_local"].isna().any():
        n_bad = int(out["tmy_datetime_local"].isna().sum())
        raise ValueError(f"Could not parse {n_bad} tmy_datetime_local value(s).")

    return out


def _nearest_available_date(available_dates: pd.Series, target: pd.Timestamp) -> object:
    candidates = pd.to_datetime(available_dates)
    idx = (candidates - target).abs().argmin()
    return candidates.iloc[int(idx)].date()


def select_reference_day_points(df: pd.DataFrame) -> list[ReferenceDaySelection]:
    """
    Select daylight rows for winter solstice, spring equinox, and summer solstice.

    Reference days are selected from tmy_datetime_local. If an exact target date
    is unavailable, the nearest available daylight date is used.
    """
    required = {"tmy_datetime_local", "sun_azimuth_deg", "sun_elevation_deg"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"DataFrame is missing required column(s): {', '.join(missing)}")

    work = df.copy()
    work["tmy_datetime_local"] = pd.to_datetime(work["tmy_datetime_local"], errors="coerce")
    work["sun_azimuth_deg"] = pd.to_numeric(work["sun_azimuth_deg"], errors="coerce")
    work["sun_elevation_deg"] = pd.to_numeric(work["sun_elevation_deg"], errors="coerce")

    valid = (
        work["tmy_datetime_local"].notna()
        & np.isfinite(work["sun_azimuth_deg"])
        & np.isfinite(work["sun_elevation_deg"])
    )
    daylight = valid & (work["sun_elevation_deg"] > 0.0)
    if not daylight.any():
        raise ValueError("No daylight rows found; expected sun_elevation_deg > 0.")

    work = work.loc[valid].copy()
    work["_date"] = work["tmy_datetime_local"].dt.date
    year = int(work.loc[daylight, "tmy_datetime_local"].dt.year.mode().iloc[0])
    available_dates = pd.Series(sorted(work.loc[daylight, "_date"].unique()))

    selections: list[ReferenceDaySelection] = []
    for key, label, month, day, color in REFERENCE_DAYS:
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
                color=color,
            )
        )

    return selections


def plot_sun_position_reference_days(
    export_csv: Path,
    output_png: Path | None = None,
    *,
    location_name: str | None = None,
    dpi: int = 200,
    annotate_hours: bool = True,
) -> Path:
    export_csv = Path(export_csv)
    if output_png is not None:
        output_png = Path(output_png)
    if not export_csv.exists():
        raise FileNotFoundError(f"Export CSV not found: {export_csv}")

    df = _load_export_csv(export_csv)
    selections = select_reference_day_points(df)

    if output_png is None:
        output_png = export_csv.parent / f"{export_csv.stem}_sun_position_reference_days.png"

    _apply_report_style()
    fig, ax = plt.subplots(figsize=(13.33, 7.50))

    for selection in selections:
        pts = selection.points
        label = f"{selection.label} ({pd.Timestamp(selection.selected_date).strftime('%b %d')})"
        ax.plot(
            pts["sun_azimuth_deg"],
            pts["sun_elevation_deg"],
            color=selection.color,
            linewidth=2.2,
            label=label,
            zorder=2,
        )
        ax.scatter(
            pts["sun_azimuth_deg"],
            pts["sun_elevation_deg"],
            s=34,
            facecolor="white",
            edgecolor=selection.color,
            linewidth=1.2,
            zorder=3,
        )

        if annotate_hours:
            hours = pts["tmy_datetime_local"].dt.hour
            annotate = pts.loc[hours.mod(3).eq(0)]
            for _, row in annotate.iterrows():
                ax.annotate(
                    f"{int(row['tmy_datetime_local'].hour):02d}",
                    (row["sun_azimuth_deg"], row["sun_elevation_deg"]),
                    textcoords="offset points",
                    xytext=(5, 4),
                    fontsize=8,
                    color=selection.color,
                    alpha=0.85,
                )

    title = "Sun Position Reference Days"
    subtitle_base = location_name or export_csv.stem
    subtitle = f"{subtitle_base} - local standard TMY time; hourly markers, 3-hour labels"
    fig.suptitle(title, x=0.5, y=0.985, ha="center", va="top", fontweight="bold")
    fig.text(0.5, 0.952, subtitle, ha="center", va="top", fontsize=12)

    ax.set_xlabel("Sun azimuth angle (deg)")
    ax.set_ylabel("Sun elevation angle (deg)")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.30, linewidth=0.8)
    ax.axhline(0.0, color="#222222", linewidth=0.9, alpha=0.75)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#DDDDDD")
    fig.subplots_adjust(top=0.88)
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
    parser.add_argument("--dpi", type=int, default=200, help="PNG output dpi.")
    parser.add_argument(
        "--no-hour-labels",
        action="store_true",
        help="Hide 3-hour labels while keeping hourly markers.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        output_png = plot_sun_position_reference_days(
            args.export_csv,
            args.output,
            location_name=args.location_name,
            dpi=args.dpi,
            annotate_hours=not args.no_hour_labels,
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Wrote: {output_png}")


if __name__ == "__main__":
    main()
