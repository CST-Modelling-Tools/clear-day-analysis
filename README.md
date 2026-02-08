@"
# clear-day-analysis

Clear-day envelope model and TMY day classification.

## Dependencies
- Python (see pyproject.toml)
- C++ sun position algorithm is included as a git submodule under `external/Updated-PSA-sun-position-algorithm`.

## Clone
git clone --recurse-submodules <repo-url>
"@ | Out-File -Encoding utf8 README.md

git add README.md