"""Low-precision solar position and satellite illumination.

Purpose: decide whether a satellite is sun-lit (candidate for a visible
naked-eye pass) or in the Earth's shadow.

Solar coordinates: standard low-precision algorithm (accuracy ~0.01 deg,
fine for shadow geometry), same family as the NOAA solar calculator.
Illumination: cylindrical Earth-shadow model in the ECI frame. The
umbral cone correction is neglected (Earth radius used as-is); the
README states this simplification.
"""

from __future__ import annotations

import math
from datetime import datetime

from .frames import julian_date

EARTH_RADIUS_KM = 6378.137
AU_KM = 149_597_870.7


def sun_eci(dt: datetime) -> tuple[float, float, float]:
    """Unit vector toward the Sun in (near-)ECI coordinates."""
    jd = julian_date(dt)
    n = jd - 2451545.0
    mean_long = (280.460 + 0.9856474 * n) % 360.0
    mean_anom = math.radians((357.528 + 0.9856003 * n) % 360.0)
    ecl_long = math.radians(mean_long + 1.915 * math.sin(mean_anom) + 0.020 * math.sin(2 * mean_anom))
    obliq = math.radians(23.439 - 0.0000004 * n)
    x = math.cos(ecl_long)
    y = math.cos(obliq) * math.sin(ecl_long)
    z = math.sin(obliq) * math.sin(ecl_long)
    return (x, y, z)


def sun_ra_dec(dt: datetime) -> tuple[float, float]:
    """Sun right ascension / declination in degrees."""
    x, y, z = sun_eci(dt)
    ra = math.degrees(math.atan2(y, x)) % 360.0
    dec = math.degrees(math.asin(z))
    return (ra, dec)


def is_sunlit(sat_eci: tuple[float, float, float], dt: datetime) -> bool:
    """Cylindrical-shadow test: satellite lit unless behind Earth away from Sun."""
    sx, sy, sz = sun_eci(dt)
    x, y, z = sat_eci
    # projection of satellite position on the Sun direction
    along = x * sx + y * sy + z * sz
    if along >= 0.0:
        return True  # day side of the Earth
    # perpendicular distance from the Earth-Sun axis
    px, py, pz = x - along * sx, y - along * sy, z - along * sz
    dist = math.sqrt(px * px + py * py + pz * pz)
    return dist > EARTH_RADIUS_KM


def sun_elevation_deg(
    dt: datetime, obs_lat_deg: float, obs_lon_deg: float
) -> float:
    """Sun elevation above the observer's horizon (for darkness checks)."""
    from .frames import gmst_rad

    ra_deg, dec_deg = sun_ra_dec(dt)
    lst = math.degrees(gmst_rad(dt)) + obs_lon_deg  # local sidereal time, deg
    ha = math.radians((lst - ra_deg + 180.0) % 360.0 - 180.0)
    dec, lat = math.radians(dec_deg), math.radians(obs_lat_deg)
    sin_el = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(ha)
    return math.degrees(math.asin(sin_el))
