# Development notes

## Repository structure

- src/clear_day_analysis/
  Core Python package
- cpp/bindings/
  pybind11 bindings for the C++ sun position library
- external/Updated-PSA-sun-position-algorithm/
  C++ sun position algorithm (git submodule)
- tests/
  pytest-based unit tests
- .github/workflows/ci.yml
  GitHub Actions continuous integration configuration

## Build system

This project includes a compiled Python extension (_sunpos) built from C++ using:

- CMake
- pybind11
- scikit-build-core (PEP 517 build backend)

The extension is built automatically during installation.

Editable install for development:

pip install -e .

## Python versions and platforms

Continuous integration is run on:

- Windows (windows-latest)
- macOS (macos-latest)
- Linux (ubuntu-latest)

for Python versions:

- 3.10
- 3.11
- 3.12

This ensures portability of both the Python and C++ components.

## Running tests

Install test dependencies:

pip install -e .[test]

Run the test suite:

pytest

The tests include:

- Smoke tests for the C++ sun position bindings
- Synthetic-data validation of the ASHRAE clear-day fit
- Daily DNI integral and classification checks

## Continuous integration (CI)

GitHub Actions automatically runs the test suite on every push and pull request.

The CI workflow:

- checks out the repository with submodules
- installs the package in editable mode
- runs pytest

A green CI status indicates the project builds and runs correctly on all supported platforms.

## Notes on packaging

At present, the package is built from source during installation. Pre-built binary wheels are not yet distributed.

Future work may include:

- automated wheel building (e.g. with cibuildwheel)
- publishing wheels to PyPI to avoid requiring a local C++ toolchain for end users.