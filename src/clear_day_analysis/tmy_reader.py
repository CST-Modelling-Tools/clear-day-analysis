from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class TMYMetadata:
    source: str
    location_id: str
    latitude: float
    longitude: float
    elevation_m: float
    time_zone: float
    local_time_zone: float


def read_nsrdb_tmy_csv(path: str | Path) -> tuple[pd.DataFrame, TMYMetadata]:
    """
    Read your NSRDB-style TMY CSV where:
      row 0 = metadata labels
      row 1 = metadata values
      row 2 = data column names
      row 3+ = data
    """
    path = Path(path)

    raw = pd.read_csv(path, header=None)

    # metadata values are row 1, metadata labels are row 0
    meta_labels = raw.iloc[0].tolist()
    meta_values = raw.iloc[1].tolist()
    meta_map = dict(zip(meta_labels, meta_values))

    md = TMYMetadata(
        source=str(meta_map.get("Source", "")),
        location_id=str(meta_map.get("Location ID", "")),
        latitude=float(meta_map["Latitude"]),
        longitude=float(meta_map["Longitude"]),
        elevation_m=float(meta_map["Elevation"]),
        time_zone=float(meta_map["Time Zone"]),
        local_time_zone=float(meta_map["Local Time Zone"]),
    )

    # data header row is row 2, data starts row 3
    header = raw.iloc[2].tolist()
    df = raw.iloc[3:].copy()
    df.columns = header

    # drop columns whose header is NaN (your file has trailing empty headers)
    df = df.loc[:, [c for c in df.columns if isinstance(c, str)]].reset_index(drop=True)

    # convert numeric columns when possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c not in ("Cloud Type",) else df[c]

    # build UTC datetime from Year/Month/Day/Hour/Minute (your file uses UTC)
    df["datetime"] = pd.to_datetime(
        dict(
            year=df["Year"].astype(int),
            month=df["Month"].astype(int),
            day=df["Day"].astype(int),
            hour=df["Hour"].astype(int),
            minute=df["Minute"].astype(int),
        ),
        utc=True,
    )

    return df, md