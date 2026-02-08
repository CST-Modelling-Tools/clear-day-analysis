from typing import Dict

def sunpos(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    second: float,
    lat_deg: float,
    lon_deg: float,
) -> Dict[str, float]:
    ...