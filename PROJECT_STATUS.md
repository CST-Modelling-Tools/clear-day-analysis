# PROJECT_STATUS.md

## Last updated

2026-05-30

## Purpose

Clear Day Analysis provides a reproducible workflow for reading Typical Meteorological Year (TMY) solar-resource files, computing solar position, fitting a clear-day DNI envelope, calculating daily DNI clearness ratios, classifying days by atmospheric clarity, and generating derived plots or exports.

The project supports solar resource assessment and CSP/solar-thermal project development.

## Current Supported TMY Sources

Supported readers are in `src/clear_day_analysis/tmy_reader.py`.

- NSRDB TMY CSV via `read_nsrdb_tmy_csv`.
- Solargis TMY60 P50 CSV via `read_solargis_tmy60_p50_csv`.
- PVGIS 5.x TMY CSV via `read_pvgis_tmy_csv`.
- Generic entry point via `read_tmy_csv(path, source=None)`, with explicit or auto-detected dispatch for NSRDB, Solargis, and PVGIS.

Common downstream expectations are `df["datetime"]`, `df["tmy_datetime_local"]`, `df["DNI"]`, preferably `df["GHI"]`, and `TMYMetadata` with source, location, elevation, and timezone fields.

## Current Analysis Workflow

The standard workflow is:

1. Read TMY data into a normalized DataFrame plus `TMYMetadata`.
2. Compute solar position with `compute_sun_position_columns`.
3. Fit ASHRAE clear-day DNI parameters with `fit_ashrae_clear_day`.
4. Add modeled clear-day DNI with `add_clear_dni_model`.
5. Compute daily DNI integral ratios with `daily_dni_integral_ratio` using `tmy_datetime_local`.
6. Classify days with `classify_days_by_ratio`.
7. Optionally generate plots or CSV exports.

`make_plots.py` is the recommended end-to-end script for daily classification and plots. `export_tmy_sun_position_dni.py` provides the sun-position/measured-DNI/clear-DNI export path. Both use the generic TMY reader.

## Datetime Convention

The standard UTC analysis timestamp column is `df["datetime"]`.

Current expectations:

- timezone-aware UTC
- suitable for solar-position calculation
- suitable for TMY annual ordering and clear-day fitting
- monotonic for normal non-leap 8760-row TMY files

The standard daily grouping timestamp column is `df["tmy_datetime_local"]`.

Current expectations:

- timezone-naive local standard time
- fixed non-leap synthetic TMY year, currently 2001
- suitable for daily DNI integration, day classification, and day-based plots
- 365 complete local dates for normal 8760-row hourly TMY files

Provider-specific source timestamps are preserved where meaningful:

- `nsrdb_datetime_utc` preserves the parsed NSRDB UTC source timestamp.
- `solargis_datetime_utc` preserves reconstructed or parsed Solargis source timestamps when source-year information is available.
- `pvgis_datetime_utc` preserves the original PVGIS source-year UTC timestamp.

`datetime` is normalized to a fixed non-leap synthetic TMY calendar, currently based on 2001. Normalization preserves the provider time reference, month, day, hour, minute, and second while removing source-year discontinuities. For local-time formats such as Solargis report-style files, normalized local standard time is converted to UTC; boundary rows can therefore fall just outside the synthetic local year in UTC. `tmy_datetime_local` wraps local-standard boundary rows back onto the 2001 local TMY calendar so daily classification represents local solar-resource days.

## Current Priorities

- Keep TMY ingestion source-specific only inside `tmy_reader.py`.
- Use `read_tmy_csv()` as the preferred TMY entry point.
- Keep analysis algorithms source-agnostic after ingestion.
- Maintain the normalized DataFrame and metadata schema across TMY sources.
- Expand cross-provider reader validation, especially metadata fallback behavior and real-file coverage.
- Validate UTC datetime behavior, local TMY grouping, and downstream classification.

## Recent Completed Milestones

- Added generic TMY reader dispatch with `read_tmy_csv()`.
- Added PVGIS 5.x TMY CSV support.
- Added PVGIS irradiance normalization: `Gb(n)` to `DNI`, `G(h)` to `GHI`, and `Gd(h)` to `DHI`.
- Preserved original PVGIS source-year timestamps in `pvgis_datetime_utc`.
- Normalized PVGIS `datetime` to a monotonic synthetic non-leap calendar.
- Normalized NSRDB `datetime` and preserved original NSRDB timestamps in `nsrdb_datetime_utc`.
- Normalized Solargis `datetime` and preserved original Solargis timestamps in `solargis_datetime_utc` when source-year information is available.
- Added `tmy_datetime_local` as the standard normalized local TMY timestamp for daily grouping, classification, and day-based plots.
- Updated `make_plots.py` to use generic TMY source selection.
- Updated `export_tmy_sun_position_dni.py` to use generic TMY source selection.
- Removed duplicated `quick_run.py` workflow; `make_plots.py` is now the documented end-to-end analysis entry point.
- Updated README and method documentation for multi-source TMY support, generic ingestion, and normalized datetime policy.
- Added synthetic NSRDB reader coverage and generic auto-detection tests for NSRDB, Solargis, and PVGIS.
- Added PVGIS tests and compact fixture coverage for reader dispatch, timestamp preservation, normalized calendar behavior, 8760-row grouping, and downstream use of normalized TMY timestamps.
- Validated the common workflow on one representative local PVGIS 8760-row TMY file; the file was not committed.
- Validated the common workflow on one representative local NSRDB 8760-row TMY file; the file was not committed. The file confirmed that NSRDB source timestamps can preserve monthly source-year discontinuities, now retained in `nsrdb_datetime_utc`.
- Validated the common workflow on one representative local Solargis 8760-row TMY60 P50 file; the file was not committed.
- Validated generated plots and sun-position/DNI exports on representative local real files for NSRDB UTC-7, Solargis UTC+4, and PVGIS UTC. Validation copies and generated outputs were not committed.
- Updated sun-position/DNI exports to include both `datetime` and `tmy_datetime_local` alongside canonical sun-position, measured DNI, and fitted ASHRAE clear-day DNI columns.

## Known Technical Debt

- Some workflow orchestration remains in scripts rather than reusable library functions.
- Metadata fallback behavior is not fully uniform across TMY sources when files omit location fields.

## Pending Validation

- Validate PVGIS behavior on more export/database variants; current real-file coverage is one local PVGIS file.
- Add focused tests for metadata fallback behavior when provider files omit location fields.
- Run CI after commits that affect reader, datetime, or workflow behavior.

## Recommended Next Milestones

1. Add metadata fallback tests across all TMY readers.
2. Consolidate repeated workflow logic from scripts into reusable library functions when it materially reduces duplication.
