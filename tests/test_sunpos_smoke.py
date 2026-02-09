import math
from clear_day_analysis import sunpos


def test_sunpos_smoke():
    out = sunpos(2024, 6, 1, 12, 0, 0.0, 33.01, -113.38)
    assert isinstance(out, dict)
    for k in ["zenith_rad", "azimuth_rad", "declination_rad", "hour_angle_rad"]:
        assert k in out
        assert math.isfinite(out[k])