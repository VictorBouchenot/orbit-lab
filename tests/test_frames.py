import math
from datetime import datetime, timezone

from src.orbit.frames import (
    ecef_to_geodetic,
    eci_to_ecef,
    geodetic_to_ecef,
    gmst_rad,
    julian_date,
    topocentric_az_el,
)


def test_julian_date_j2000():
    # 2000-01-01 12:00 UTC is JD 2451545.0 by definition
    assert abs(julian_date(datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc)) - 2451545.0) < 1e-9


def test_gmst_at_j2000():
    # IAU expression at J2000.0: 280.46061837 deg
    got = math.degrees(gmst_rad(datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc)))
    assert abs(got - 280.46061837) < 1e-6


def test_geodetic_roundtrip():
    for lat, lon, alt in [(48.8566, 2.3522, 0.035), (-33.86, 151.2, 0.0), (0.0, 0.0, 400.0)]:
        r = geodetic_to_ecef(lat, lon, alt)
        lat2, lon2, alt2 = ecef_to_geodetic(r)
        assert abs(lat2 - lat) < 1e-9
        assert abs(lon2 - lon) < 1e-9
        assert abs(alt2 - alt) < 1e-6


def test_eci_ecef_is_pure_z_rotation():
    dt = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)
    r = (7000.0, 100.0, -1300.0)
    out = eci_to_ecef(r, dt)
    assert abs(out[2] - r[2]) < 1e-12  # z unchanged
    assert abs(math.hypot(out[0], out[1]) - math.hypot(r[0], r[1])) < 1e-9  # radius in xy kept


def test_topocentric_zenith():
    # A satellite 400 km above the observer along the geodetic normal: elevation 90 deg.
    lat, lon = math.radians(48.8566), math.radians(2.3522)
    obs = geodetic_to_ecef(48.8566, 2.3522, 0.0)
    up = (math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat))
    sat = tuple(o + 400.0 * u for o, u in zip(obs, up))
    az, el, rng = topocentric_az_el(sat, 48.8566, 2.3522, 0.0)
    assert abs(el - 90.0) < 1e-9
    assert abs(rng - 400.0) < 1e-9


def test_radial_zenith_offset_is_small_but_real():
    # On the ellipsoid, geocentric-radial and geodetic-up differ: at Paris the
    # offset is ~0.19 deg. A radial +400 km point is NOT exactly at zenith.
    obs = geodetic_to_ecef(48.8566, 2.3522, 0.0)
    r = math.sqrt(sum(c * c for c in obs))
    sat = tuple(c * (r + 400.0) / r for c in obs)
    _, el, _ = topocentric_az_el(sat, 48.8566, 2.3522, 0.0)
    assert 89.7 < el < 90.0
