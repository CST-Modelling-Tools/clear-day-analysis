from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
import re
from typing import Any
import csv


TMY_SYNTHETIC_YEAR = 2001
PVGIS_SYNTHETIC_YEAR = TMY_SYNTHETIC_YEAR


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


def _validate_datetime_complete(values: pd.Series, *, provider: str, column: str) -> pd.Series:
    dt = pd.to_datetime(values, errors="coerce", utc=True)
    if dt.isna().any():
        n_bad = int(dt.isna().sum())
        raise ValueError(f"Could not parse {n_bad} {provider} {column} values.")
    return dt


def _normalize_tmy_utc_datetime(
    original: pd.Series,
    *,
    provider: str,
    synthetic_year: int = TMY_SYNTHETIC_YEAR,
) -> pd.Series:
    original = pd.to_datetime(original, errors="coerce", utc=True)
    parts = {
        "year": pd.Series(synthetic_year, index=original.index),
        "month": original.dt.month,
        "day": original.dt.day,
        "hour": original.dt.hour,
        "minute": original.dt.minute,
        "second": original.dt.second,
    }
    normalized = pd.to_datetime(parts, errors="coerce", utc=True)

    invalid = normalized.isna() & original.notna()
    if invalid.any():
        samples = original.loc[invalid].astype(str).head(3).tolist()
        raise ValueError(
            f"Could not normalize {provider} timestamps to a non-leap synthetic TMY calendar. "
            f"Unsupported dates include: {samples}"
        )

    return normalized


def _normalize_tmy_local_datetime(
    original_local: pd.Series,
    *,
    provider: str,
    synthetic_year: int = TMY_SYNTHETIC_YEAR,
) -> pd.Series:
    original_local = pd.to_datetime(original_local, errors="coerce")
    parts = {
        "year": pd.Series(synthetic_year, index=original_local.index),
        "month": original_local.dt.month,
        "day": original_local.dt.day,
        "hour": original_local.dt.hour,
        "minute": original_local.dt.minute,
        "second": original_local.dt.second,
    }
    normalized = pd.to_datetime(parts, errors="coerce")

    invalid = normalized.isna() & original_local.notna()
    if invalid.any():
        samples = original_local.loc[invalid].astype(str).head(3).tolist()
        raise ValueError(
            f"Could not normalize {provider} local timestamps to a non-leap synthetic TMY calendar. "
            f"Unsupported dates include: {samples}"
        )

    return normalized


def _local_standard_time_to_utc(dt_local: pd.Series, utc_offset_h: float) -> pd.Series:
    return (dt_local - pd.to_timedelta(utc_offset_h, unit="h")).dt.tz_localize("UTC")


def read_nsrdb_tmy_csv(path: str | Path) -> tuple[pd.DataFrame, TMYMetadata]:
    """
    Read an NSRDB-style TMY CSV where:
      row 0 = metadata labels
      row 1 = metadata values
      row 2 = data column names
      row 3+ = data

    The parsed source-year UTC timestamp is preserved in nsrdb_datetime_utc.
    The standard datetime column is normalized to a fixed non-leap synthetic
    TMY calendar for analysis.
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

    # Preserve the NSRDB source-year timestamp and expose a normalized TMY
    # calendar for downstream analysis.
    df["nsrdb_datetime_utc"] = pd.to_datetime(
        dict(
            year=df["Year"].astype(int),
            month=df["Month"].astype(int),
            day=df["Day"].astype(int),
            hour=df["Hour"].astype(int),
            minute=df["Minute"].astype(int),
        ),
        errors="coerce",
        utc=True,
    )
    df["nsrdb_datetime_utc"] = _validate_datetime_complete(
        df["nsrdb_datetime_utc"],
        provider="NSRDB",
        column="datetime",
    )
    df["datetime"] = _normalize_tmy_utc_datetime(df["nsrdb_datetime_utc"], provider="NSRDB")
    if not df["datetime"].is_monotonic_increasing:
        raise ValueError("NSRDB normalized TMY datetime is not monotonic increasing.")

    return df, md


def _compact_colname(name: Any) -> str:
    return re.sub(r"\s+", "", _normalize_colname(name))


def _looks_like_pvgis_time(value: Any) -> bool:
    s = str(value).strip()
    return bool(
        re.match(r"^\d{8}:\d{4}$", s)
        or re.match(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{1,2}:?\d{0,2})?", s)
        or re.match(r"^\d{4}/\d{2}/\d{2}(?:[ T]\d{1,2}:?\d{0,2})?", s)
    )


def _parse_pvgis_utc_datetime(values: pd.Series) -> pd.Series:
    s = values.astype(str).str.strip()

    dt = pd.to_datetime(s, format="%Y%m%d:%H%M", errors="coerce", utc=True)

    # Some tools express midnight at the end of a day as 24:00.
    mask_2400 = s.str.match(r"^\d{8}:2400$")
    if mask_2400.any():
        dt.loc[mask_2400] = (
            pd.to_datetime(s.loc[mask_2400].str.slice(0, 8), format="%Y%m%d", errors="coerce", utc=True)
            + pd.Timedelta(days=1)
        )

    # Fall back to pandas' parser for minor variations such as ISO-like dates.
    fallback_strings = s.str.replace(r"^(\d{8}):(\d{2})(\d{2})$", r"\1 \2:\3", regex=True)
    fallback = pd.to_datetime(fallback_strings, errors="coerce", utc=True)
    dt = dt.fillna(fallback)

    return dt


def _normalize_pvgis_tmy_datetime(original: pd.Series, *, synthetic_year: int = PVGIS_SYNTHETIC_YEAR) -> pd.Series:
    return _normalize_tmy_utc_datetime(original, provider="PVGIS", synthetic_year=synthetic_year)


def _metadata_value(meta_map: dict[str, str], needles: list[str]) -> str | None:
    for needle in needles:
        needle_norm = _normalize_colname(needle)
        for key, value in meta_map.items():
            if needle_norm in key:
                return value
    return None


def _metadata_labeled_float(meta_map: dict[str, str], lines: list[str], labels: list[str]) -> float:
    value = _metadata_value(meta_map, labels)
    out = _extract_first_float(value)
    if np.isfinite(out):
        return out

    text = "\n".join(lines)
    for label in labels:
        m = re.search(
            rf"{re.escape(label)}[^\n\r+-]*([-+]?\d+(?:\.\d+)?)",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            return float(m.group(1))
    return float("nan")


def _parse_solargis_selected_years(value: Any) -> dict[int, int]:
    if value is None:
        return {}
    years: dict[int, int] = {}
    for month, year in re.findall(r"\b(\d{1,2})\s*:\s*(\d{4})\b", str(value)):
        month_i = int(month)
        if 1 <= month_i <= 12:
            years[month_i] = int(year)
    return years


def _solargis_source_local_from_selected_years(dt_local: pd.Series, selected_years: dict[int, int]) -> pd.Series:
    source_year = dt_local.dt.month.map(selected_years)
    if source_year.isna().any():
        missing = sorted(int(m) for m in dt_local.loc[source_year.isna()].dt.month.dropna().unique())
        raise ValueError(f"Solargis selected-years metadata is missing month(s): {missing}")

    return pd.to_datetime(
        {
            "year": source_year.astype(int),
            "month": dt_local.dt.month,
            "day": dt_local.dt.day,
            "hour": dt_local.dt.hour,
            "minute": dt_local.dt.minute,
            "second": dt_local.dt.second,
        },
        errors="coerce",
    )


def read_pvgis_tmy_csv(path: str | Path) -> tuple[pd.DataFrame, TMYMetadata]:
    """
    Read a PVGIS 5.x Typical Meteorological Year CSV.

    PVGIS CSV exports contain a metadata/comment block, a data table headed by
    columns like time(UTC), G(h), Gb(n), and Gd(h), and sometimes monthly source
    information before or after the table. This parser locates the data table
    by header names rather than fixed row numbers.

    The original PVGIS source-year timestamp is preserved in
    pvgis_datetime_utc. The standard datetime column is normalized to a fixed
    non-leap synthetic TMY year so downstream daily grouping is monotonic.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        lines = [ln.rstrip("\n\r") for ln in f]

    def tokens(row: list[str]) -> list[str]:
        return [str(cell).strip() for cell in row if str(cell).strip()]

    def is_pvgis_header(row_tokens: list[str]) -> bool:
        names = {_compact_colname(t) for t in row_tokens}
        has_time = any(n in {"time(utc)", "timeutc"} for n in names)
        has_irradiance = any(n in {"g(h)", "gb(n)", "gd(h)"} for n in names)
        return has_time and has_irradiance

    best_match: tuple[int, str, list[list[str]], int] | None = None
    for delim in [",", ";", "\t"]:
        rows = [[cell.strip() for cell in row] for row in csv.reader(lines, delimiter=delim)]
        for i, row in enumerate(rows):
            row_tokens = tokens(row)
            if is_pvgis_header(row_tokens):
                score = len(row_tokens)
                if best_match is None or score > best_match[0]:
                    best_match = (score, delim, rows, i)
                break

    if best_match is None:
        raise ValueError("Could not locate PVGIS TMY data header row with time(UTC) and irradiance columns.")

    _, _, raw_rows, header_idx = best_match
    header_row = [str(cell).strip() for cell in raw_rows[header_idx]]
    valid_col_indices = [i for i, col in enumerate(header_row) if col]
    header = [header_row[i] for i in valid_col_indices]

    time_col_original_idx = None
    for i in valid_col_indices:
        if _compact_colname(header_row[i]) in {"time(utc)", "timeutc"}:
            time_col_original_idx = i
            break
    if time_col_original_idx is None:
        raise ValueError("PVGIS TMY CSV must include a time(UTC) column.")

    data_rows: list[list[str]] = []
    data_line_indices: set[int] = set()
    for i in range(header_idx + 1, len(raw_rows)):
        row = [str(cell).strip() for cell in raw_rows[i]]
        if not any(row):
            continue
        time_value = row[time_col_original_idx] if time_col_original_idx < len(row) else ""
        if not _looks_like_pvgis_time(time_value):
            if data_rows:
                break
            continue

        data_rows.append([row[j] if j < len(row) else "" for j in valid_col_indices])
        data_line_indices.add(i)

    if not data_rows:
        raise ValueError("PVGIS TMY CSV header was found, but no data rows with parseable time(UTC) values followed it.")

    df = pd.DataFrame(data_rows, columns=header)

    normalized_cols = {_compact_colname(c): c for c in df.columns}
    time_col = normalized_cols.get("time(utc)") or normalized_cols.get("timeutc")
    if time_col is None:
        raise ValueError("PVGIS TMY CSV must include a time(UTC) column.")

    rename_targets = {
        "gb(n)": "DNI",
        "g(h)": "GHI",
        "gd(h)": "DHI",
    }
    rename_map: dict[str, str] = {}
    existing_targets = set(df.columns)
    for col in df.columns:
        target = rename_targets.get(_compact_colname(col))
        if target and target not in existing_targets and target not in rename_map.values():
            rename_map[col] = target
    if rename_map:
        df = df.rename(columns=rename_map)

    for col in df.columns:
        if col == time_col:
            continue
        conv = pd.to_numeric(df[col], errors="coerce")
        if conv.notna().any():
            df[col] = conv

    df["pvgis_datetime_utc"] = _parse_pvgis_utc_datetime(df[time_col])
    valid_datetime_fraction = float(df["pvgis_datetime_utc"].notna().mean())
    if valid_datetime_fraction < 0.9:
        raise ValueError(
            "Could not parse most PVGIS time(UTC) values; "
            f"parsed {valid_datetime_fraction:.1%} of rows successfully."
        )

    if df["pvgis_datetime_utc"].isna().any():
        n_bad = int(df["pvgis_datetime_utc"].isna().sum())
        raise ValueError(f"Could not parse {n_bad} PVGIS time(UTC) values.")

    df["datetime"] = _normalize_pvgis_tmy_datetime(df["pvgis_datetime_utc"])
    if not df["datetime"].is_monotonic_increasing:
        raise ValueError("PVGIS normalized TMY datetime is not monotonic increasing.")

    meta_map: dict[str, str] = {}

    def add_meta(key: str, value: str) -> None:
        clean_key = _normalize_colname(key.lstrip("#").strip())
        clean_value = str(value).strip()
        if not clean_key or re.fullmatch(r"[-+]?\d+(?:\.\d+)?", clean_key):
            return
        meta_map.setdefault(clean_key, clean_value)

    for i, row in enumerate(raw_rows):
        if i == header_idx or i in data_line_indices:
            continue
        row_tokens = tokens(row)
        if not row_tokens:
            continue
        if len(row_tokens) == 1:
            body = row_tokens[0].lstrip("#").strip()
            if ":" in body:
                key, value = body.split(":", 1)
                add_meta(key, value)
            elif "=" in body:
                key, value = body.split("=", 1)
                add_meta(key, value)
            continue

        first = row_tokens[0].lstrip("#").strip()
        if ":" in first:
            key, value = first.split(":", 1)
            add_meta(key, value or row_tokens[1])
        elif "=" in first:
            key, value = first.split("=", 1)
            add_meta(key, value or row_tokens[1])
        else:
            add_meta(first, row_tokens[1])

    latitude = _metadata_labeled_float(meta_map, lines, ["latitude", "lat"])
    longitude = _metadata_labeled_float(meta_map, lines, ["longitude", "lon", "lng"])
    elevation = _metadata_labeled_float(meta_map, lines, ["elevation", "altitude"])

    database = _metadata_value(
        meta_map,
        [
            "radiation database",
            "database",
            "meteo database",
            "source",
        ],
    )
    if database:
        database_clean = re.sub(r"\s+", "_", database.strip())
        source = f"PVGIS_TMY_{database_clean}"
    else:
        source = "PVGIS_TMY"

    location_id = str(
        _metadata_value(meta_map, ["location id", "site id", "site name", "location", "name"])
        or path.stem
    )

    md = TMYMetadata(
        source=source,
        location_id=location_id,
        latitude=latitude if np.isfinite(latitude) else 0.0,
        longitude=longitude if np.isfinite(longitude) else 0.0,
        elevation_m=elevation if np.isfinite(elevation) else 0.0,
        time_zone=0.0,
        local_time_zone=0.0,
    )

    return df, md


def read_solargis_tmy60_p50_csv(path: str | Path) -> tuple[pd.DataFrame, TMYMetadata]:
    """
    Read a Solargis TMY60 P50 CSV into a DataFrame + normalized metadata.

    This parser is intentionally tolerant to minor format variations:
      - metadata block before tabular data
      - comma or semicolon delimiter
      - datetime as [Year, Month, Day, Hour, Minute] OR [Date, Time]

    Parsed or reconstructed source-year timestamps are preserved in
    solargis_datetime_utc when source dates are meaningful. The standard
    datetime column is normalized to a fixed non-leap synthetic TMY calendar
    in the file's time reference and converted to UTC for analysis.
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

        # Build datetime from Day-of-year + Time, using the normalized non-leap
        # TMY calendar in the file's local standard time reference.
        norm_cols = {_normalize_colname(c): c for c in df.columns}
        if "day" not in norm_cols or "time" not in norm_cols:
            raise ValueError("Solargis report-style CSV must include Day and Time columns.")

        day = pd.to_numeric(df[norm_cols["day"]], errors="coerce").astype("Int64")
        if day.isna().any() or (day < 1).any() or (day > 365).any():
            raise ValueError("Solargis report-style Day values must fit a non-leap synthetic TMY calendar.")

        tparts = df[norm_cols["time"]].astype(str).str.split(":", n=1, expand=True)
        hour = pd.to_numeric(tparts[0], errors="coerce").fillna(0.0)
        minute = pd.to_numeric(tparts[1] if tparts.shape[1] > 1 else 0, errors="coerce").fillna(0.0)

        base_date = pd.Timestamp(year=TMY_SYNTHETIC_YEAR, month=1, day=1)
        dt_local = (
            base_date
            + pd.to_timedelta(day.fillna(1).astype(int) - 1, unit="D")
            + pd.to_timedelta(hour, unit="h")
            + pd.to_timedelta(minute, unit="m")
        )

        selected_years = _parse_solargis_selected_years(meta_map.get("tmy created from selected years"))
        if selected_years:
            source_local = _solargis_source_local_from_selected_years(dt_local, selected_years)
            df["solargis_datetime_utc"] = _local_standard_time_to_utc(source_local, utc_offset_h)
            df["solargis_datetime_utc"] = _validate_datetime_complete(
                df["solargis_datetime_utc"],
                provider="Solargis",
                column="source datetime",
            )

        df["datetime"] = _local_standard_time_to_utc(dt_local, utc_offset_h)
        if not df["datetime"].is_monotonic_increasing:
            raise ValueError("Solargis normalized TMY datetime is not monotonic increasing.")

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

    # Build source and normalized timezone-aware UTC datetimes expected by downstream code.
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

        dt_local = pd.to_datetime(
            {
                "year": year,
                "month": month,
                "day": day,
            },
            errors="coerce",
        )
        dt_local = dt_local + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")
    elif "date" in normalized_cols:
        date_col = df[normalized_cols["date"]].astype(str)
        if "time" in normalized_cols:
            dt_local = pd.to_datetime(date_col + " " + df[normalized_cols["time"]].astype(str), errors="coerce")
        else:
            dt_local = pd.to_datetime(date_col, errors="coerce")
    else:
        raise ValueError("Could not build datetime: expected Year/Month/Day[...] or Date[/Time] columns.")

    if dt_local.isna().any():
        n_bad = int(dt_local.isna().sum())
        raise ValueError(f"Could not parse {n_bad} Solargis local datetime values.")

    # Solargis files are typically in local standard time. Preserve parsed
    # source-year timestamps, then normalize the analysis calendar.
    df["solargis_datetime_utc"] = _local_standard_time_to_utc(dt_local, utc_offset_h)
    df["solargis_datetime_utc"] = _validate_datetime_complete(
        df["solargis_datetime_utc"],
        provider="Solargis",
        column="source datetime",
    )

    dt_local_normalized = _normalize_tmy_local_datetime(dt_local, provider="Solargis")
    df["datetime"] = _local_standard_time_to_utc(dt_local_normalized, utc_offset_h)
    if not df["datetime"].is_monotonic_increasing:
        raise ValueError("Solargis normalized TMY datetime is not monotonic increasing.")

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


def _detect_tmy_source(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        text = f.read(65536)

    text_l = text.lower()
    lines = text.splitlines()

    for line in lines:
        for delim in [",", ";", "\t"]:
            names = {_compact_colname(cell) for cell in line.split(delim)}
            has_pvgis_time = any(n in {"time(utc)", "timeutc"} for n in names)
            has_pvgis_irradiance = any(n in {"g(h)", "gb(n)", "gd(h)"} for n in names)
            if has_pvgis_time and has_pvgis_irradiance:
                return "pvgis"

    if "solargis" in text_l or "#typical meteorological year" in text_l or "#data:" in text_l:
        return "solargis"

    for line in lines[:5]:
        names = {_normalize_colname(cell) for cell in re.split(r"[,;\t]", line)}
        if {"source", "latitude", "longitude"}.issubset(names) and (
            "time zone" in names or "local time zone" in names
        ):
            return "nsrdb"

    for line in lines:
        for delim in [",", ";", "\t"]:
            names = {_normalize_colname(cell) for cell in line.split(delim)}
            if {"day", "time"}.issubset(names) and (
                {"dni", "ghi"}.issubset(names) or {"bn", "ghi"}.issubset(names)
            ):
                return "solargis"

    raise ValueError(
        "Could not infer TMY CSV format. Pass source='nsrdb', source='solargis', or source='pvgis'."
    )


def read_tmy_csv(path: str | Path, source: str | None = None) -> tuple[pd.DataFrame, TMYMetadata]:
    """
    Read a TMY CSV, dispatching to the NSRDB, Solargis, or PVGIS reader.

    Set source to "nsrdb", "solargis", or "pvgis" to choose explicitly.
    With source=None or source="auto", the format is inferred from file contents.
    """
    path = Path(path)

    if source is None or str(source).strip().lower() == "auto":
        source_key = _detect_tmy_source(path)
    else:
        source_key = str(source).strip().lower().replace("-", "_")

    if source_key in {"nsrdb", "nrel_nsrdb"}:
        return read_nsrdb_tmy_csv(path)
    if source_key in {"solargis", "solargis_tmy60_p50", "tmy60_p50"}:
        return read_solargis_tmy60_p50_csv(path)
    if source_key in {"pvgis", "pvgis_tmy"}:
        return read_pvgis_tmy_csv(path)

    raise ValueError(
        f"Unknown TMY source {source!r}. Expected 'nsrdb', 'solargis', 'pvgis', or None/'auto'."
    )
