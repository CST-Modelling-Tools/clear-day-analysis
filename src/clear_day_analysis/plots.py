from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .ashrae_clear_day import AshraeFitResult, IterationSnapshot

# ---------------------------------------------------------------------
# Global color palette for day classes (semantic, reused everywhere)
# ---------------------------------------------------------------------

DAY_CLASS_ORDER = [
    "extremely_clear",
    "clear",
    "cloudy",
    "extremely_cloudy",
]

DAY_CLASS_LABELS = {
    "extremely_clear": "Extremely clear",
    "clear": "Clear",
    "cloudy": "Cloudy",
    "extremely_cloudy": "Extremely cloudy",
}

DAY_CLASS_COLORS = {
    "extremely_clear": "#2166AC",   # deep blue
    "clear": "#67A9CF",             # light blue
    "cloudy": "#BDBDBD",            # mid gray
    "extremely_cloudy": "#636363",  # dark gray
}

# A consistent “primary ink” color for lines/axes accents
PRIMARY_INK = "#053061"  # dark blue


# -----------------------------
# Plot context / metadata
# -----------------------------
@dataclass(frozen=True)
class PlotContext:
    location_name: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    tmy_label: str | None = None  # e.g. "TMY-2024"
    source: str | None = None     # e.g. "NSRDB"


@dataclass(frozen=True)
class PlotPaths:
    base_dir: Path
    prefix: str  # usually the TMY stem
    iterations_dir: Path


def make_plot_paths(tmy_path: Path) -> PlotPaths:
    base_dir = tmy_path.parent
    prefix = tmy_path.stem
    iterations_dir = base_dir / f"{prefix}_fit_iterations"
    return PlotPaths(base_dir=base_dir, prefix=prefix, iterations_dir=iterations_dir)


# -----------------------------
# Styling helpers (16:9, theme)
# -----------------------------
# 16:9 figure size (inches). 13.33 x 7.5 prints nicely on widescreen PPT.
FIGSIZE_16_9 = (13.33, 7.50)


def _apply_theme() -> None:
    """Apply a clean, presentation-friendly global style."""
    plt.rcParams.update(
        {
            "figure.figsize": FIGSIZE_16_9,
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


def _format_utc_offset(utc_offset_hours: float | None) -> str:
    if utc_offset_hours is None:
        return "UTC±0"
    if utc_offset_hours < 0:
        return f"UTC\u2212{abs(utc_offset_hours):g}"  # proper minus sign
    return f"UTC+{utc_offset_hours:g}"


def _build_subtitle(paths: PlotPaths, context: PlotContext | None) -> str:
    """
    Build a subtitle:
      "<Location> (Lat ..°, Lon ..°) — <TMY> — <Source>"

    If context is missing, tries to parse from paths.prefix which often looks like:
      "<id>_<lat>_<lon>_tmy-<year>"
    """
    loc = None
    lat = None
    lon = None
    tmy = None
    src = None

    if context is not None:
        loc = context.location_name
        lat = context.latitude
        lon = context.longitude
        tmy = context.tmy_label
        src = context.source

    if (lat is None) or (lon is None) or (tmy is None) or (loc is None):
        import re

        m = re.match(
            r"(?P<id>\d+)_"
            r"(?P<lat>-?\d+(\.\d+)?)_"
            r"(?P<lon>-?\d+(\.\d+)?)_"
            r"(?P<tmy>tmy-\d+)",
            paths.prefix,
        )
        if m:
            if lat is None:
                lat = float(m.group("lat"))
            if lon is None:
                lon = float(m.group("lon"))
            if tmy is None:
                tmy = m.group("tmy").upper()
            if loc is None:
                loc = m.group("id")

    loc_str = f"{loc}" if loc is not None else paths.prefix
    ll_str = f" (Lat {lat:.2f}°, Lon {lon:.2f}°)" if (lat is not None and lon is not None) else ""
    tmy_str = f" — {tmy}" if tmy is not None else ""
    src_str = f" — {src}" if src else ""
    return f"{loc_str}{ll_str}{tmy_str}{src_str}"


def _apply_header(fig: plt.Figure, *, title: str, subtitle: str) -> None:
    """Two-line header. Reserves space at top for consistent layout."""
    fig.suptitle(title, x=0.5, y=0.985, ha="center", va="top", fontweight="bold")
    fig.text(0.5, 0.952, subtitle, ha="center", va="top", fontsize=12)
    fig.subplots_adjust(top=0.88)


def _finalize_and_save(fig: plt.Figure, out: Path, *, dpi: int) -> None:
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _class_patches(alpha: float = 0.30) -> list[Patch]:
    """Legend handles for class colors."""
    return [
        Patch(facecolor=DAY_CLASS_COLORS[c], edgecolor="none", alpha=alpha, label=DAY_CLASS_LABELS[c])
        for c in DAY_CLASS_ORDER
    ]


# -----------------------------
# Plots
# -----------------------------
def plot_fit_iterations(
    fit: AshraeFitResult,
    paths: PlotPaths,
    *,
    max_plots: Optional[int] = None,
    dpi: int = 150,
    context: PlotContext | None = None,
) -> list[Path]:
    """
    Create one PNG per iteration showing:
      points, OLS line, corridor, and outliers.
    Uses fit.snapshots (requires record_snapshots=True).
    """
    _apply_theme()

    if not fit.snapshots:
        raise ValueError("fit.snapshots is empty. Re-run fit_ashrae_clear_day(record_snapshots=True).")

    paths.iterations_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    snaps: Iterable[IterationSnapshot] = fit.snapshots
    if max_plots is not None:
        snaps = list(snaps)[: max_plots]

    subtitle = _build_subtitle(paths, context)

    for s in snaps:
        x = s.x
        y = s.y
        keep = s.keep_mask

        fig = plt.figure(figsize=FIGSIZE_16_9)
        ax = fig.add_subplot(1, 1, 1)

        ax.scatter(x[keep], y[keep], s=14, alpha=0.85, label="Kept points")
        ax.scatter(x[~keep], y[~keep], s=26, marker="x", alpha=0.9, label="Outliers")

        order = np.argsort(x)
        ax.plot(x[order], s.yhat[order], color=PRIMARY_INK, linewidth=2.2, label="OLS fit")
        ax.plot(
            x[order],
            s.lower[order],
            color=PRIMARY_INK,
            linewidth=1.6,
            alpha=0.85,
            label=f"{int(round(fit.confidence * 100))}% Student-t corridor",
        )
        ax.plot(x[order], s.upper[order], color=PRIMARY_INK, linewidth=1.6, alpha=0.85)

        ax.grid(True, alpha=0.25, linewidth=0.8)

        ax.set_xlabel("x = 1/sin(Sun Elevation) (-)")
        ax.set_ylabel("ln[Direct Normal Irradiance (W/m²)] (-)")

        _apply_header(
            fig,
            title=f"Clear-day fit iteration {s.iteration:02d} (OLS + Student-t corridor)",
            subtitle=subtitle,
        )
        ax.legend(loc="best", frameon=True)

        out = paths.iterations_dir / f"iter_{s.iteration:02d}.png"
        _finalize_and_save(fig, out, dpi=dpi)
        written.append(out)

    return written


def plot_clearness_index_timeseries(
    daily: pd.DataFrame,
    paths: PlotPaths,
    *,
    ratio_col: str = "ratio",
    thr_extremely_clear: float = 0.90,
    thr_clear: float = 0.70,
    thr_cloudy: float = 0.40,
    dpi: int = 150,
    context: Optional["PlotContext"] = None,
) -> Path:
    """
    Plot ratio R(d) vs TMY day number (1..N), with colored class bands + legend.
    Uses the global DAY_CLASS_COLORS palette for consistency across the project.
    """
    _apply_theme()

    r = pd.to_numeric(daily[ratio_col], errors="coerce").to_numpy(dtype=float)
    x = np.arange(1, len(r) + 1)

    fig = plt.figure(figsize=FIGSIZE_16_9)
    ax = fig.add_subplot(1, 1, 1)

    # Bands (bottom->top or top->bottom both ok; set zorder so line stays above)
    band_specs = [
        ("extremely_cloudy", 0.0, thr_cloudy),
        ("cloudy", thr_cloudy, thr_clear),
        ("clear", thr_clear, thr_extremely_clear),
        ("extremely_clear", thr_extremely_clear, 1.05),
    ]
    for cls, y0, y1 in band_specs:
        ax.axhspan(
            y0,
            y1,
            facecolor=DAY_CLASS_COLORS[cls],
            alpha=0.22,
            edgecolor="none",
            zorder=0,
        )

    ax.plot(x, r, color=PRIMARY_INK, linewidth=1.8, zorder=2)

    # Threshold lines
    for y in (thr_extremely_clear, thr_clear, thr_cloudy):
        ax.axhline(y, color=PRIMARY_INK, linewidth=1.2, alpha=0.9, zorder=3)

    ax.set_xlabel("TMY day number (-)")
    ax.set_ylabel("Daily clearness index R = H_DNI / H_clear (-)")

    ymax = float(np.nanmax(r)) if np.any(np.isfinite(r)) else 1.0
    ax.set_ylim(0.0, max(1.05, min(1.25, ymax + 0.05)))

    subtitle = _build_subtitle(paths, context)
    _apply_header(
        fig,
        title="Daily clearness index over the year (classification bands)",
        subtitle=subtitle,
    )

    ax.grid(True, alpha=0.25, linewidth=0.8)

    # Legend: explicit patches in semantic order (NOT auto-handles)
    ax.legend(
        handles=_class_patches(alpha=0.22),
        loc="upper right",
        frameon=True,
        ncols=2,
    )

    out = paths.base_dir / f"{paths.prefix}_clearness_index_timeseries.png"
    _finalize_and_save(fig, out, dpi=dpi)
    return out


def plot_annual_energy_by_class(
    daily_cls: pd.DataFrame,
    paths: PlotPaths,
    *,
    class_col: str = "class",
    H_col: str = "H_dni",
    dpi: int = 150,
    context: PlotContext | None = None,
) -> Path:
    """
    Bar chart of annual DNI energy per day class.
    Converts Wh/m² -> kWh/m² (your daily integrals are Wh/m²-based).
    Uses DAY_CLASS_COLORS for consistency.
    """
    _apply_theme()

    grp = daily_cls.groupby(class_col, as_index=False)[H_col].sum()
    mapping = {row[class_col]: float(row[H_col]) for _, row in grp.iterrows()}

    vals_kWh = [(mapping.get(c, 0.0) / 1000.0) for c in DAY_CLASS_ORDER]

    fig = plt.figure(figsize=FIGSIZE_16_9)
    ax = fig.add_subplot(1, 1, 1)

    ax.bar(
        [DAY_CLASS_LABELS[c] for c in DAY_CLASS_ORDER],
        vals_kWh,
        color=[DAY_CLASS_COLORS[c] for c in DAY_CLASS_ORDER],
        edgecolor="none",
    )

    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.set_ylabel("Annual Direct Normal Irradiation (kWh/m²)")
    ax.set_xlabel("Day class (-)")

    subtitle = _build_subtitle(paths, context)
    _apply_header(fig, title="Annual DNI energy by class", subtitle=subtitle)

    out = paths.base_dir / f"{paths.prefix}_annual_energy_by_class.png"
    _finalize_and_save(fig, out, dpi=dpi)
    return out


def plot_cumulative_energy(
    daily: pd.DataFrame,
    paths: PlotPaths,
    *,
    H_col: str = "H_dni",
    H_clear_col: Optional[str] = "H_dni_clear",
    dpi: int = 150,
    context: PlotContext | None = None,
) -> Path:
    """
    Cumulative annual energy along the TMY day number.
    Uses kWh/m² on Y-axis.
    """
    _apply_theme()

    H = pd.to_numeric(daily[H_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    cum_kWh = np.cumsum(H) / 1000.0
    x = np.arange(1, len(cum_kWh) + 1)

    fig = plt.figure(figsize=FIGSIZE_16_9)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(x, cum_kWh, color=PRIMARY_INK, linewidth=2.2, label="Measured DNI")

    if H_clear_col is not None and H_clear_col in daily.columns:
        Hc = pd.to_numeric(daily[H_clear_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, np.cumsum(Hc) / 1000.0, color="#1A1A1A", linewidth=2.0, alpha=0.75, label="Clear-day reference")

    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.set_xlabel("TMY day number (-)")
    ax.set_ylabel("Cumulative Direct Normal Irradiation (kWh/m²)")

    subtitle = _build_subtitle(paths, context)
    _apply_header(fig, title="Cumulative DNI energy over the year", subtitle=subtitle)

    ax.legend(loc="best", frameon=True)

    out = paths.base_dir / f"{paths.prefix}_cumulative_energy.png"
    _finalize_and_save(fig, out, dpi=dpi)
    return out


def plot_seasonal_energy_by_class(
    daily_cls: pd.DataFrame,
    paths: PlotPaths,
    *,
    date_col: str = "date",
    class_col: str = "class",
    H_col: str = "H_dni",
    dpi: int = 150,
    context: Optional["PlotContext"] = None,
) -> Path:
    """
    Seasonal stacked bars (DJF, MAM, JJA, SON) split by class,
    using the global DAY_CLASS_COLORS palette.
    """
    _apply_theme()

    d = pd.to_datetime(daily_cls[date_col])
    month = d.dt.month

    def season(m: int) -> str:
        if m in (12, 1, 2):
            return "DJF"
        if m in (3, 4, 5):
            return "MAM"
        if m in (6, 7, 8):
            return "JJA"
        return "SON"

    tmp = daily_cls.copy()
    tmp["_season"] = [season(int(m)) for m in month.to_numpy()]
    tmp[H_col] = pd.to_numeric(tmp[H_col], errors="coerce").fillna(0.0)

    seasons = ["DJF", "MAM", "JJA", "SON"]
    pivot = (
        tmp.pivot_table(index="_season", columns=class_col, values=H_col, aggfunc="sum", fill_value=0.0)
        .reindex(seasons)
        .reindex(columns=DAY_CLASS_ORDER, fill_value=0.0)
    )

    # Wh/m² -> kWh/m²
    pivot = pivot / 1000.0

    fig = plt.figure(figsize=FIGSIZE_16_9)
    ax = fig.add_subplot(1, 1, 1)

    bottoms = np.zeros(len(seasons), dtype=float)
    x = np.arange(len(seasons))

    for cls in DAY_CLASS_ORDER:
        vals = pivot[cls].to_numpy(dtype=float)
        ax.bar(
            x,
            vals,
            bottom=bottoms,
            color=DAY_CLASS_COLORS[cls],
            edgecolor="none",
            label=DAY_CLASS_LABELS[cls],
        )
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(seasons)
    ax.set_ylabel("Seasonal Direct Normal Irradiation (kWh/m²)")

    subtitle = _build_subtitle(paths, context)
    _apply_header(fig, title="Seasonal DNI energy by class", subtitle=subtitle)

    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(loc="upper right", frameon=True, ncols=2)

    out = paths.base_dir / f"{paths.prefix}_seasonal_energy_by_class.png"
    _finalize_and_save(fig, out, dpi=dpi)
    return out


def plot_day_examples_grid(
    df: pd.DataFrame,
    daily_cls: pd.DataFrame,
    paths: PlotPaths,
    *,
    datetime_col: str = "datetime",
    dni_col: str = "DNI",
    clear_col: str = "dni_clear_model",
    date_col: str = "date",
    ratio_col: str = "ratio",
    class_col: str = "class",
    dpi: int = 150,
    utc_offset_hours: float | None = None,
    context: PlotContext | None = None,
) -> Path:
    """
    4 rows (classes) x 3 cols (min/avg/max ratio within class) showing
    DNI(t) and DNI_clear(t) for those days.
    """
    _apply_theme()

    classes = DAY_CLASS_ORDER

    picks: dict[str, list[object]] = {}
    for c in classes:
        sub = daily_cls[daily_cls[class_col] == c].copy()
        if len(sub) == 0:
            picks[c] = [None, None, None]
            continue

        sub[ratio_col] = pd.to_numeric(sub[ratio_col], errors="coerce")
        sub = sub.dropna(subset=[ratio_col])
        if len(sub) == 0:
            picks[c] = [None, None, None]
            continue

        i_min = sub[ratio_col].idxmin()
        i_max = sub[ratio_col].idxmax()
        mean = float(sub[ratio_col].mean())
        i_mid = (sub[ratio_col] - mean).abs().idxmin()
        picks[c] = [sub.loc[i_min, date_col], sub.loc[i_mid, date_col], sub.loc[i_max, date_col]]

    fig, axes = plt.subplots(4, 3, figsize=FIGSIZE_16_9, sharex=True, sharey=True)

    for ax in axes.ravel():
        ax.set_ylim(0, 1050)
        ax.set_yticks([0, 250, 500, 750, 1000])
        ax.grid(True, alpha=0.25, linewidth=0.8)

    tz_str = _format_utc_offset(utc_offset_hours)
    xlab = f"Hour (Local Standard Time, {tz_str})"

    for r, c in enumerate(classes):
        for col in range(3):
            ax = axes[r, col]
            day = picks[c][col]
            if day is None:
                ax.set_axis_off()
                continue

            day_mask = pd.to_datetime(df[datetime_col]).dt.date == day
            day_df = df.loc[day_mask, [datetime_col, dni_col, clear_col]].copy()
            if len(day_df) == 0:
                ax.set_axis_off()
                continue

            day_df = day_df.sort_values(datetime_col)
            t = pd.to_datetime(day_df[datetime_col])
            hour = t.dt.hour + t.dt.minute / 60.0

            dni = pd.to_numeric(day_df[dni_col], errors="coerce").to_numpy(dtype=float)
            clr = pd.to_numeric(day_df[clear_col], errors="coerce").to_numpy(dtype=float)

            ax.plot(hour, dni, color=PRIMARY_INK, linewidth=2.0, label="DNI")

            clr_plot = clr.copy()
            clr_plot[~np.isfinite(clr_plot)] = 0.0
            clr_plot = np.where(dni <= 0.0, 0.0, clr_plot)
            ax.plot(hour, clr_plot, color="#E66101", linewidth=2.0, label="Clear envelope")

            which = ["min", "avg", "max"][col]
            ax.set_title(f"{DAY_CLASS_LABELS[c]} ({which})\n{day}", fontsize=12)

            if r == 3:
                ax.set_xlabel(xlab)

            if r == 0 and col == 2:
                ax.legend(loc="lower right", frameon=True)

    subtitle = _build_subtitle(paths, context)
    _apply_header(fig, title="Example days by class (measured DNI vs clear-day envelope)", subtitle=subtitle)

    fig.text(0.015, 0.5, "Direct Normal Irradiance (W/m²)", va="center", rotation="vertical", fontsize=12)

    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.08, hspace=0.35, wspace=0.12)

    out = paths.base_dir / f"{paths.prefix}_day_examples_grid.png"
    _finalize_and_save(fig, out, dpi=dpi)
    return out