#include <pybind11/pybind11.h>
#include "sunpos.h"

namespace py = pybind11;

py::dict sunpos_call(int year, int month, int day,
                     int hour, int minute, double second,
                     double lat_deg, double lon_deg)
{
    cTime t;
    t.iYear = year;
    t.iMonth = month;
    t.iDay = day;
    t.dHours = static_cast<double>(hour);
    t.dMinutes = static_cast<double>(minute);
    t.dSeconds = second;

    cLocation loc;
    loc.dLatitude = lat_deg;
    loc.dLongitude = lon_deg;

    cSunCoordinates out;
    sunpos(t, loc, &out);

    // Return raw outputs (radians) to avoid any convention mismatch.
    // Python can convert to degrees and compute elevation if needed.
    py::dict d;
    d["zenith_rad"] = out.dZenithAngle;
    d["azimuth_rad"] = out.dAzimuth;
    d["declination_rad"] = out.dDeclination;
    d["hour_angle_rad"] = out.dBoundedHourAngle;
    return d;
}

PYBIND11_MODULE(_sunpos, m)
{
    m.doc() = "Python bindings for Updated PSA sun position algorithm";

    m.def("sunpos",
          &sunpos_call,
          py::arg("year"),
          py::arg("month"),
          py::arg("day"),
          py::arg("hour"),
          py::arg("minute"),
          py::arg("second"),
          py::arg("lat_deg"),
          py::arg("lon_deg"),
          R"pbdoc(
Compute sun position using the Updated PSA algorithm.

Inputs:
  - Time is interpreted as UTC/UT (no timezone conversion).
  - Longitude convention: East positive, West negative.
Outputs:
  Dictionary with angles in radians:
    zenith_rad, azimuth_rad, declination_rad, hour_angle_rad
)pbdoc");
}