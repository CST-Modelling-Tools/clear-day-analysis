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

Analysis workflows use the repository-standard column:

```text
df["datetime"]
```

This column is timezone-aware UTC and is the timestamp used for solar position, fitting, daily grouping, plotting, and exports unless a workflow explicitly documents otherwise.

TMY providers may preserve source years for selected months. For stable annual analysis, readers may normalize timestamps onto a synthetic TMY calendar. PVGIS files keep the original source-year timestamp in `pvgis_datetime_utc` and expose a normalized, monotonic, non-leap UTC TMY calendar in `datetime`.

Do not use source-specific timestamp columns such as `pvgis_datetime_utc` for analysis unless the task is explicitly auditing source data.

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
- daily class counts
- `<tmy_stem>_daily_classification.csv`

## Plots And Exports

Generate clear-day fit and classification plots:

```bash
python make_plots.py path/to/tmy.csv
```

Export normalized UTC time, sun position, and DNI:

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
