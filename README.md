# clear-day-analysis

Clear-day envelope model and TMY day classification tools for NSRDB, Solargis, and PVGIS Typical Meteorological Year files.

![CI](../../actions/workflows/ci.yml/badge.svg)

## What this project does

Given a supported TMY CSV, this project:

1. Reads and normalizes TMY data from a provider-specific CSV format.
2. Computes solar position (elevation/azimuth) using a C++ implementation wrapped with pybind11.
3. Fits a clear-day DNI envelope using the ASHRAE log-linear formulation and iterative outlier rejection.
4. Builds a clear-day DNI model for each timestamp.
5. Computes daily ratios of measured DNI energy to clear-day DNI energy.
6. Classifies each day into `extremely_clear`, `clear`, `cloudy`, or `extremely_cloudy`.
7. Generates plots, daily classification CSVs, or sun-position/DNI exports.

Classification is based on fixed physical ratio thresholds, not tuned per location.

## Supported TMY Sources

TMY ingestion is centralized in `src/clear_day_analysis/tmy_reader.py`.

- NSRDB TMY CSV: `read_nsrdb_tmy_csv`
- Solargis TMY60 P50 CSV: `read_solargis_tmy60_p50_csv`
- PVGIS 5.x TMY CSV: `read_pvgis_tmy_csv`
- Generic entry point: `read_tmy_csv(path, source=None)`

Use the generic reader for new workflows:

```python
from clear_day_analysis.tmy_reader import read_tmy_csv

df, md = read_tmy_csv("path/to/tmy.csv", source="auto")
```

`source` may be `"auto"`/`None`, `"nsrdb"`, `"solargis"`, or `"pvgis"`.

## Datetime Policy

Readers expose two normalized TMY timestamps with different roles:

- `datetime`: timezone-aware UTC, used for solar-position calculations, clear-day fitting, exports, and row ordering.
- `tmy_datetime_local`: timezone-naive local standard time in the fixed synthetic TMY calendar, used for daily DNI integration, day classification, and day-based plots.

TMY providers may preserve source years for selected months. For stable annual analysis, readers normalize timestamps onto a synthetic TMY calendar. Provider source timestamps are preserved where meaningful:

- NSRDB: `nsrdb_datetime_utc`
- Solargis: `solargis_datetime_utc` when source-year information is available
- PVGIS: `pvgis_datetime_utc`

`datetime` is monotonic for normal non-leap 8760-row TMY files. For local-time providers, UTC timestamps can fall just outside the synthetic local year at the boundaries. `tmy_datetime_local` wraps those boundary rows onto the 2001 non-leap local TMY calendar so daily classification represents local-standard solar-resource days. Do not use source-specific timestamp columns for analysis unless the task is explicitly auditing source data.

## Dependencies

- Python (see `pyproject.toml`)
- A C++ sun-position algorithm included as a git submodule:
  `external/Updated-PSA-sun-position-algorithm`
- Build toolchain, because the C++ extension is compiled during installation:
  - Windows: Visual Studio Build Tools
  - macOS: Xcode Command Line Tools
  - Linux: a C++ compiler toolchain

## Install (developer mode)

Clone including submodules:

```bash
git clone --recurse-submodules <repo-url>
cd clear-day-analysis
```

Create an environment and install editable:

```bash
python -m venv .venv
# activate your venv
pip install -U pip
pip install -e .
```

## Quick Run

Edit `quick_run.py`:

```python
TMY_PATH = Path("path/to/your/tmy.csv")
TMY_SOURCE = "auto"  # "auto", "nsrdb", "solargis", or "pvgis"
```

Then run:

```bash
python quick_run.py
```

Outputs:

- fitted clear-day parameters (`E0`, `beta`)
- daily class counts based on local-standard TMY days
- `<tmy_stem>_daily_classification.csv`

## Plots And Exports

Generate clear-day fit and classification plots:

```bash
python make_plots.py path/to/tmy.csv
```

Export normalized UTC time, normalized local TMY time, sun position, and DNI:

```bash
python export_tmy_sun_position_dni.py path/to/tmy.csv
```

Both scripts use `read_tmy_csv(..., source="auto")`.

## Tests

Install test dependencies and run:

```bash
pip install -e .[test]
pytest
```

## Scientific Method

See:

- `docs/METHOD.md` for the clear-day envelope model, TMY ingestion convention, datetime policy, and day classification definition.
- `docs/DEVELOPMENT.md` for build and CI details.
