from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
import re
from typing import Any
import csv


@dataclass(frozen=True)
class TMYMetadata:
    source: str
    location_id: str
    latitude: float
    longitude: float
    elevation_m: float
    time_zone: float
    local_time_zone: float


def _normalize_colname(name: Any) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_first_float(value: Any, *, default: float = np.nan) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float, np.number)):
        out = float(value)
        return out if np.isfinite(out) else default
    m = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    if not m:
        return default
    return float(m.group(0))


def _parse_utc_offset_hours(value: Any) -> float:
    """
    Parse UTC offset in hours from strings like:
      -7
      UTC-7
      UTC+05:30
    Returns NaN if not parseable.
    """
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.number)):
        out = float(value)
        return out if np.isfinite(out) else float("nan")

    s = str(value).strip().upper()
    if not s:
        return float("nan")

    # Plain numeric string
    try:
        return float(s)
    except Exception:
        pass

    m = re.search(r"UTC\s*([+-])\s*(\d{1,2})(?::?(\d{2}))?", s)
    if not m:
        return float("nan")

    sign = -1.0 if m.group(1) == "-" else 1.0
    hh = float(m.group(2))
    mm = float(m.group(3) or 0.0)
    return sign * (hh + mm / 60.0)


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


def read_solargis_tmy60_p50_csv(path: str | Path) -> tuple[pd.DataFrame, TMYMetadata]:
    """
    Read a Solargis TMY60 P50 CSV into a DataFrame + normalized metadata.

    This parser is intentionally tolerant to minor format variations:
      - metadata block before tabular data
      - comma or semicolon delimiter
      - datetime as [Year, Month, Day, Hour, Minute] OR [Date, Time]
    """
    path = Path(path)

    # --- Solargis "report style" CSV ---
    # Metadata and docs are prefixed with '#', tabular data is semicolon-separated
    # under a "Data:" section with columns like Day;Time;GHI;DNI;...
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        lines = [ln.rstrip("\n\r") for ln in f]

    is_report_style = any(ln.startswith("#Data:") for ln in lines) and any(
        ln.startswith("#TYPICAL METEOROLOGICAL YEAR") for ln in lines
    )

    if is_report_style:
        meta_map: dict[str, str] = {}
        utc_offset_h = float("nan")

        for ln in lines:
            if not ln.startswith("#"):
                continue
            body = ln[1:].strip()
            if ":" in body:
                k, v = body.split(":", 1)
                key = _normalize_colname(k)
                val = v.strip()
                if key:
                    meta_map[key] = val

            # Time line example:
            # "Time - Time reference UTC+4, time step 60 min, ..."
            if "time reference utc" in body.lower():
                m = re.search(r"UTC\s*([+-]\d{1,2}(?::\d{2})?)", body, flags=re.IGNORECASE)
                if m:
                    utc_offset_h = _parse_utc_offset_hours(f"UTC{m.group(1)}")

        if not np.isfinite(utc_offset_h):
            utc_offset_h = _parse_utc_offset_hours(
                meta_map.get("local time zone")
                or meta_map.get("time zone")
                or meta_map.get("timezone")
                or meta_map.get("utc offset")
            )
        if not np.isfinite(utc_offset_h):
            utc_offset_h = 0.0

        # pandas skips all '#' lines and starts at Day;Time;...
        df = pd.read_csv(path, sep=";", comment="#")
        df = df.loc[:, [c for c in df.columns if str(c).strip() and not str(c).startswith("Unnamed:")]].copy()

        # Normalize common irradiance aliases if needed
        col_map = {c: _normalize_colname(c) for c in df.columns}
        for c, n in col_map.items():
            if n in {"dni", "bn", "bni"} and "DNI" not in df.columns:
                df = df.rename(columns={c: "DNI"})
            if n in {"ghi", "global horizontal irradiation"} and "GHI" not in df.columns:
                df = df.rename(columns={c: "GHI"})

        # Convert numeric columns where conversion is meaningful.
        for c in df.columns:
            if _normalize_colname(c) == "time":
                continue
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().any():
                df[c] = conv

        # Build datetime from Day-of-year + Time, using a stable non-leap base year.
        norm_cols = {_normalize_colname(c): c for c in df.columns}
        if "day" not in norm_cols or "time" not in norm_cols:
            raise ValueError("Solargis report-style CSV must include Day and Time columns.")

        day = pd.to_numeric(df[norm_cols["day"]], errors="coerce").astype("Int64")
        day_max = int(day.max()) if day.notna().any() else 365
        base_year = 2000 if day_max >= 366 else 2001

        tparts = df[norm_cols["time"]].astype(str).str.split(":", n=1, expand=True)
        hour = pd.to_numeric(tparts[0], errors="coerce").fillna(0.0)
        minute = pd.to_numeric(tparts[1] if tparts.shape[1] > 1 else 0, errors="coerce").fillna(0.0)

        base_date = pd.Timestamp(year=base_year, month=1, day=1)
        dt_local = (
            base_date
            + pd.to_timedelta(day.fillna(1).astype(int) - 1, unit="D")
            + pd.to_timedelta(hour, unit="h")
            + pd.to_timedelta(minute, unit="m")
        )

        df["datetime"] = (dt_local - pd.to_timedelta(utc_offset_h, unit="h")).dt.tz_localize("UTC")

        latitude = _extract_first_float(meta_map.get("latitude") or meta_map.get("lat"))
        longitude = _extract_first_float(meta_map.get("longitude") or meta_map.get("lon") or meta_map.get("lng"))
        elevation = _extract_first_float(meta_map.get("elevation") or meta_map.get("elevation [m]"))

        source = str(meta_map.get("file type") or "Solargis_TMY60_P50")
        location_id = str(meta_map.get("site name") or meta_map.get("site id") or path.stem)

        md = TMYMetadata(
            source=source,
            location_id=location_id,
            latitude=latitude if np.isfinite(latitude) else 0.0,
            longitude=longitude if np.isfinite(longitude) else 0.0,
            elevation_m=elevation if np.isfinite(elevation) else 0.0,
            time_zone=float(utc_offset_h),
            local_time_zone=float(utc_offset_h),
        )
        return df, md

    # Parse once as raw rows to recover metadata block and detect data header row.
    # Use stdlib CSV reader so mixed-width rows (metadata vs table) are accepted.
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            delim = dialect.delimiter
        except Exception:
            # Fallback for mixed-width metadata/table files where Sniffer can fail.
            candidates = [",", ";", "\t"]
            counts = {d: sample.count(d) for d in candidates}
            delim = max(candidates, key=lambda d: counts[d])
            if counts[delim] == 0:
                delim = ","

        reader = csv.reader(f, delimiter=delim)
        raw_rows = [[cell.strip() for cell in row] for row in reader]

    def row_tokens(i: int) -> list[str]:
        vals = [v for v in raw_rows[i] if v is not None and str(v).strip() != ""]
        return [str(v).strip() for v in vals]

    header_idx = None
    for i in range(len(raw_rows)):
        toks = row_tokens(i)
        if not toks:
            continue
        names = {_normalize_colname(t) for t in toks}
        has_ymd = {"year", "month", "day"}.issubset(names)
        has_date_time = ("date" in names) and (("time" in names) or ("hour" in names))
        has_irr = any("dni" in n or "ghi" in n for n in names)
        if (has_ymd or has_date_time) and has_irr:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not locate Solargis TMY data header row.")

    # Build metadata map from key/value rows before header
    meta_map: dict[str, str] = {}
    for i in range(header_idx):
        toks = row_tokens(i)
        if len(toks) < 2:
            continue
        key = _normalize_colname(toks[0])
        if not key:
            continue
        meta_map[key] = toks[1]

    # Parse tabular portion using detected header row.
    df = pd.read_csv(path, skiprows=header_idx, sep=None, engine="python")
    df = df.loc[:, [c for c in df.columns if str(c).strip() and not str(c).startswith("Unnamed:")]].copy()

    col_map = {c: _normalize_colname(c) for c in df.columns}

    # Canonical irradiance names used by the pipeline.
    dni_aliases = {"dni", "bn", "bni", "dni [w/m2]", "dni [w m-2]"}
    ghi_aliases = {"ghi", "ghi [w/m2]", "ghi [w m-2]", "global horizontal irradiance"}
    for c, n in col_map.items():
        if n in dni_aliases and "DNI" not in df.columns:
            df = df.rename(columns={c: "DNI"})
        if n in ghi_aliases and "GHI" not in df.columns:
            df = df.rename(columns={c: "GHI"})

    # Numeric conversion where possible, preserving obvious text columns.
    for c in df.columns:
        n = _normalize_colname(c)
        if n in {"date", "time"}:
            continue
        conv = pd.to_numeric(df[c], errors="coerce")
        # Keep original text column if conversion yields no numeric values.
        if conv.notna().any():
            df[c] = conv

    normalized_cols = {_normalize_colname(c): c for c in df.columns}

    # Build timezone-aware UTC datetime expected by downstream code.
    tz_val = (
        meta_map.get("local time zone")
        or meta_map.get("time zone")
        or meta_map.get("timezone")
        or meta_map.get("utc offset")
    )
    utc_offset_h = _parse_utc_offset_hours(tz_val)
    if not np.isfinite(utc_offset_h):
        utc_offset_h = 0.0

    if {"year", "month", "day"}.issubset(normalized_cols.keys()):
        year = pd.to_numeric(df[normalized_cols["year"]], errors="coerce").astype("Int64")
        month = pd.to_numeric(df[normalized_cols["month"]], errors="coerce").astype("Int64")
        day = pd.to_numeric(df[normalized_cols["day"]], errors="coerce").astype("Int64")
        hour = pd.to_numeric(
            df[normalized_cols["hour"]] if "hour" in normalized_cols else 0,
            errors="coerce",
        ).fillna(0.0)
        minute = pd.to_numeric(
            df[normalized_cols["minute"]] if "minute" in normalized_cols else 0,
            errors="coerce",
        ).fillna(0.0)

        base_date = pd.to_datetime(
            {
                "year": year,
                "month": month,
                "day": day,
            },
            errors="coerce",
        )
        dt_local = base_date + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")
    elif "date" in normalized_cols:
        date_col = df[normalized_cols["date"]].astype(str)
        if "time" in normalized_cols:
            dt_local = pd.to_datetime(date_col + " " + df[normalized_cols["time"]].astype(str), errors="coerce")
        else:
            dt_local = pd.to_datetime(date_col, errors="coerce")
    else:
        raise ValueError("Could not build datetime: expected Year/Month/Day[...] or Date[/Time] columns.")

    # Solargis files are typically in local standard time; convert to UTC.
    df["datetime"] = (dt_local - pd.to_timedelta(utc_offset_h, unit="h")).dt.tz_localize("UTC")

    latitude = _extract_first_float(
        meta_map.get("latitude")
        or meta_map.get("lat"),
    )
    longitude = _extract_first_float(
        meta_map.get("longitude")
        or meta_map.get("lon")
        or meta_map.get("lng"),
    )
    elevation = _extract_first_float(
        meta_map.get("elevation")
        or meta_map.get("elevation [m]")
        or meta_map.get("altitude"),
    )

    location_id = str(
        meta_map.get("site id")
        or meta_map.get("site name")
        or meta_map.get("location")
        or path.stem
    )
    source = str(meta_map.get("source") or "Solargis_TMY60_P50")

    md = TMYMetadata(
        source=source,
        location_id=location_id,
        latitude=latitude if np.isfinite(latitude) else 0.0,
        longitude=longitude if np.isfinite(longitude) else 0.0,
        elevation_m=elevation if np.isfinite(elevation) else 0.0,
        time_zone=float(utc_offset_h),
        local_time_zone=float(utc_offset_h),
    )

    return df, md
