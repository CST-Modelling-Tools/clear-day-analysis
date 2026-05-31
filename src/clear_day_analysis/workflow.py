from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .ashrae_clear_day import AshraeFitResult, OutlierMode, fit_ashrae_clear_day
from .day_classification import add_clear_dni_model
from .solar_geometry import compute_sun_position_columns
from .tmy_reader import TMYMetadata, read_tmy_csv


@dataclass(frozen=True)
class ClearDNIModelSpec:
    clear_col: str
    alpha_min_deg: float = 5.0
    require_finite_dni: bool = True
    fill_value: float = np.nan


@dataclass(frozen=True)
class ClearDayWorkflowResult:
    df: pd.DataFrame
    metadata: TMYMetadata
    fit: AshraeFitResult


def run_clear_day_workflow(
    tmy_csv: str | Path,
    *,
    source: str | None = "auto",
    datetime_col: str = "datetime",
    dni_col: str = "DNI",
    elevation_col: str = "sun_elevation_deg",
    daylight_elevation_deg: float = 0.0,
    alpha_min_deg: float = 5.0,
    confidence: float = 0.95,
    outlier_mode: OutlierMode = "lower",
    max_iter: int = 25,
    min_points: int = 200,
    enforce_envelope: bool = True,
    envelope_quantile: float = 0.98,
    record_snapshots: bool = False,
    clear_models: Sequence[ClearDNIModelSpec] = (),
) -> ClearDayWorkflowResult:
    """
    Run the shared TMY -> sun position -> ASHRAE fit preparation pipeline.

    Script-specific work such as daily classification, plotting, and export
    formatting stays outside this helper.
    """
    df, md = read_tmy_csv(tmy_csv, source=source)

    missing = [col for col in (datetime_col, "tmy_datetime_local", dni_col) if col not in df.columns]
    if missing:
        raise ValueError(f"Parsed TMY CSV is missing required column(s): {', '.join(missing)}")

    df = compute_sun_position_columns(
        df,
        datetime_col=datetime_col,
        lat_deg=md.latitude,
        lon_deg=md.longitude,
        daylight_elevation_deg=daylight_elevation_deg,
    )

    fit = fit_ashrae_clear_day(
        df,
        dni_col=dni_col,
        elevation_col=elevation_col,
        alpha_min_deg=alpha_min_deg,
        confidence=confidence,
        outlier_mode=outlier_mode,
        max_iter=max_iter,
        min_points=min_points,
        enforce_envelope=enforce_envelope,
        envelope_quantile=envelope_quantile,
        record_snapshots=record_snapshots,
    )

    for spec in clear_models:
        df = add_clear_dni_model(
            df,
            E0=fit.E0,
            beta=fit.beta,
            dni_col=dni_col,
            elevation_col=elevation_col,
            alpha_min_deg=spec.alpha_min_deg,
            clear_col=spec.clear_col,
            require_finite_dni=spec.require_finite_dni,
            fill_value=spec.fill_value,
        )

    return ClearDayWorkflowResult(df=df, metadata=md, fit=fit)
