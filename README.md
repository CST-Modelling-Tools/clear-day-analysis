# clear-day-analysis

Clear-day envelope model and TMY day classification tools (NSRDB TMY).

![CI](../../actions/workflows/ci.yml/badge.svg)

## What this project does

Given a Typical Meteorological Year (TMY) from NREL NSRDB:

1. Computes solar position (elevation/azimuth) using a C++ implementation (pybind11).
2. Fits a clear-day DNI envelope using the ASHRAE log-linear formulation and iterative outlier rejection.
3. Computes, for each day, the ratio of measured DNI energy to clear-day DNI energy.
4. Classifies each day into:
   - extremely_clear
   - clear
   - cloudy
   - extremely_cloudy

Classification is based on a fixed physical ratio threshold, not tuned per-location.

## Dependencies

- Python (see pyproject.toml)
- A C++ sun-position algorithm included as a git submodule:
  external/Updated-PSA-sun-position-algorithm
- Build toolchain (needed because the C++ extension is compiled during installation):
  - Windows: Visual Studio Build Tools
  - macOS: Xcode Command Line Tools
  - Linux: a C++ compiler toolchain

## Install (developer mode)

Clone including submodules:

git clone --recurse-submodules <repo-url>
cd clear-day-analysis

Create environment and install editable:

python -m venv .venv
# activate your venv
pip install -U pip
pip install -e .

## Quick run

Edit quick_run.py to point to your NSRDB TMY CSV, then run:

python quick_run.py

Outputs:
- Fitted clear-day parameters (E0, beta)
- Daily class counts
- Writes daily_classification.csv

## Tests

Install test dependencies and run:

pip install -e .[test]
pytest

## Scientific method

See:
- docs/METHOD.md for the clear-day envelope model and day classification definition.
- docs/DEVELOPMENT.md for build and CI details.
