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

Common downstream expectations are `df["datetime"]`, `df["DNI"]`, preferably `df["GHI"]`, and `TMYMetadata` with source, location, elevation, and timezone fields.

## Current Analysis Workflow

The standard workflow is:

1. Read TMY data into a normalized DataFrame plus `TMYMetadata`.
2. Compute solar position with `compute_sun_position_columns`.
3. Fit ASHRAE clear-day DNI parameters with `fit_ashrae_clear_day`.
4. Add modeled clear-day DNI with `add_clear_dni_model`.
5. Compute daily DNI integral ratios with `daily_dni_integral_ratio`.
6. Classify days with `classify_days_by_ratio`.
7. Optionally generate plots or CSV exports.

`quick_run.py`, `make_plots.py`, and `export_tmy_sun_position_dni.py` use the generic TMY reader.

## Datetime Convention

The standard analysis timestamp column is `df["datetime"]`.

Current expectations:

- timezone-aware UTC
- suitable for solar-position calculation
- suitable for TMY annual ordering and daily grouping
- monotonic where the reader provides a synthetic TMY calendar

PVGIS-specific policy:

- `pvgis_datetime_utc` preserves the original PVGIS source-year UTC timestamp.
- `datetime` is normalized to a fixed non-leap synthetic TMY year, currently 2001.
- Normalization preserves month, day, hour, minute, and second while removing source-year discontinuities.

Solargis report-style files already use a stable synthetic base year for day-of-year/time inputs. NSRDB and other Solargis variants currently preserve parsed calendar years; daily classification sorts by month/day to avoid TMY source-year ordering artifacts.

## Current Priorities

- Keep TMY ingestion source-specific only inside `tmy_reader.py`.
- Use `read_tmy_csv()` as the preferred TMY entry point.
- Keep analysis algorithms source-agnostic after ingestion.
- Maintain the normalized DataFrame and metadata schema across TMY sources.
- Expand cross-provider reader validation, especially metadata fallback behavior and real-file coverage.
- Validate datetime behavior for timezone awareness, monotonicity, daily grouping, and downstream classification.

## Recent Completed Milestones

- Added generic TMY reader dispatch with `read_tmy_csv()`.
- Added PVGIS 5.x TMY CSV support.
- Added PVGIS irradiance normalization: `Gb(n)` to `DNI`, `G(h)` to `GHI`, and `Gd(h)` to `DHI`.
- Preserved original PVGIS source-year timestamps in `pvgis_datetime_utc`.
- Normalized PVGIS `datetime` to a monotonic synthetic non-leap calendar.
- Updated `quick_run.py` to use generic TMY source selection.
- Updated `make_plots.py` to use generic TMY source selection.
- Updated `export_tmy_sun_position_dni.py` to use generic TMY source selection.
- Updated README and method documentation for multi-source TMY support, generic ingestion, and normalized datetime policy.
- Added synthetic NSRDB reader coverage and generic auto-detection tests for NSRDB, Solargis, and PVGIS.
- Added PVGIS tests and compact fixture coverage for reader dispatch, timestamp preservation, normalized calendar behavior, 8760-row grouping, and downstream use of normalized `datetime`.

## Known Technical Debt

- Some workflow orchestration remains in scripts rather than reusable library functions.
- Metadata fallback behavior is not fully uniform across TMY sources when files omit location fields.
- Plotting still uses `datetime_local` in some paths; this should be reviewed against the normalized TMY datetime policy before broader PVGIS/Solargis plotting validation.

## Pending Validation

- Validate NSRDB, Solargis, and PVGIS readers against representative real files for each provider.
- Validate PVGIS behavior on more export/database variants.
- Add focused tests for metadata fallback behavior when provider files omit location fields.
- Confirm whether NSRDB and Solargis should also expose explicit original timestamp columns when source years differ from normalized analysis years.
- Decide whether daily classification should standardize on UTC `datetime` or local-standard-time grouping in all user-facing workflows.
- Run CI after commits that affect reader, datetime, or workflow behavior.
- Validate generated plot and export outputs for NSRDB, Solargis, and PVGIS after reader consolidation.

## Recommended Next Milestones

1. Validate representative real-file fixtures for NSRDB, Solargis, and PVGIS reader behavior.
2. Decide whether all readers should provide both normalized `datetime` and source-specific original timestamp columns.
3. Add metadata fallback tests across all TMY readers.
4. Consolidate repeated workflow logic from scripts into reusable library functions when it materially reduces duplication.
