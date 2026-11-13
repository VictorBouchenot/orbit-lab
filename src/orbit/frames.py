"""Reference-frame conversions, implemented from first principles.

ECI  : Earth-Centered Inertial (TEME output of SGP4, treated as inertial here)
ECEF : Earth-Centered Earth-Fixed (rotates with the Earth)
Topo : topocentric horizon frame at the observer (azimuth/elevation)

The only rotation applied between ECI and ECEF is the sidereal spin about
the z axis (GMST). Polar motion, nutation and the TEME-to-GCRF bias are
neglected: sub-arcminute effects, far below SGP4's own km-level error for
week-old TLEs. The README states this as a limitation.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

# WGS-84 constants
WGS84_A_KM = 6378.137
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def julian_date(dt: datetime) -> float:
    """Julian date of a timezone-aware UTC datetime."""
    dt = dt.astimezone(timezone.utc)
    y, m = dt.year, dt.month
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    day_frac = (dt.hour + (dt.minute + (dt.second + dt.microsecond / 1e6) / 60.0) / 60.0) / 24.0
    return (
        math.floor(365.25 * (y + 4716))
        + math.floor(30.6001 * (m + 1))
        + dt.day
        + day_frac
        + b
        - 1524.5
    )


def gmst_rad(dt: datetime) -> float:
    """Greenwich Mean Sidereal Time (IAU 1982 expression, truncated)."""
    jd = julian_date(dt)
    t = (jd - 2451545.0) / 36525.0
    gmst_deg = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * t * t
        - (t * t * t) / 38710000.0
    )
    return math.radians(gmst_deg % 360.0)


def eci_to_ecef(r_eci: tuple[float, float, float], dt: datetime) -> tuple[float, float, float]:
    """Rotate an ECI vector into ECEF by the GMST angle about z."""
    th = gmst_rad(dt)
    c, s = math.cos(th), math.sin(th)
    x, y, z = r_eci
    return (c * x + s * y, -s * x + c * y, z)


def ecef_to_geodetic(
    r_ecef: tuple[float, float, float],
) -> tuple[float, float, float]:
    """ECEF -> (lat_deg, lon_deg, alt_km), iterative WGS-84 (Bowring-style)."""
    x, y, z = r_ecef
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(10):
        n = WGS84_A_KM / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
        alt = p / math.cos(lat) - n
        new_lat = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + alt)))
        if abs(new_lat - lat) < 1e-12:
            lat = new_lat
            break
        lat = new_lat
    n = WGS84_A_KM / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
    alt = p / math.cos(lat) - n
    return (math.degrees(lat), math.degrees(lon), alt)


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_km: float) -> tuple[float, float, float]:
    """Geodetic observer position -> ECEF vector."""
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    n = WGS84_A_KM / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
    x = (n + alt_km) * math.cos(lat) * math.cos(lon)
    y = (n + alt_km) * math.cos(lat) * math.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt_km) * math.sin(lat)
    return (x, y, z)


def topocentric_az_el(
    sat_ecef: tuple[float, float, float],
    obs_lat_deg: float,
    obs_lon_deg: float,
    obs_alt_km: float,
) -> tuple[float, float, float]:
    """Satellite ECEF -> (azimuth_deg from North eastward, elevation_deg, range_km)."""
    ox, oy, oz = geodetic_to_ecef(obs_lat_deg, obs_lon_deg, obs_alt_km)
    dx, dy, dz = sat_ecef[0] - ox, sat_ecef[1] - oy, sat_ecef[2] - oz
    lat, lon = math.radians(obs_lat_deg), math.radians(obs_lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    # ENU components
    e = -so * dx + co * dy
    n_ = -sl * co * dx - sl * so * dy + cl * dz
    u = cl * co * dx + cl * so * dy + sl * dz
    rng = math.sqrt(e * e + n_ * n_ + u * u)
    el = math.degrees(math.asin(u / rng))
    az = math.degrees(math.atan2(e, n_)) % 360.0
    return (az, el, rng)
