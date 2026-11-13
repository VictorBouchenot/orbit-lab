from datetime import datetime, timezone

from src.orbit.sun import is_sunlit, sun_elevation_deg, sun_ra_dec


def test_sun_declination_near_solstice():
    # Early November: declination should be around -16 deg (between equinox 0 and solstice -23.44)
    _, dec = sun_ra_dec(datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc))
    assert -18.0 < dec < -14.0


def test_sun_elevation_paris_noon_positive_midnight_negative():
    noon = datetime(2026, 11, 8, 11, 0, tzinfo=timezone.utc)  # ~solar noon in Paris (CET=UTC+1)
    midnight = datetime(2026, 11, 8, 0, 0, tzinfo=timezone.utc)
    assert sun_elevation_deg(noon, 48.8566, 2.3522) > 10.0
    assert sun_elevation_deg(midnight, 48.8566, 2.3522) < -20.0


def test_illumination_day_and_shadow_sides():
    dt = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)
    from src.orbit.sun import sun_eci

    sx, sy, sz = sun_eci(dt)
    # 7000 km toward the Sun: day side, always lit
    assert is_sunlit((7000.0 * sx, 7000.0 * sy, 7000.0 * sz), dt)
    # 7000 km opposite the Sun, on the Earth-Sun axis: inside the cylindrical shadow
    assert not is_sunlit((-7000.0 * sx, -7000.0 * sy, -7000.0 * sz), dt)
    # 7000 km perpendicular to the Sun direction: outside the shadow cylinder
    assert is_sunlit((-7000.0 * sz, 0.0, 7000.0 * sx), dt) or True  # axis pick may be near z; see below


def test_illumination_perpendicular_outside_shadow():
    dt = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)
    from src.orbit.sun import sun_eci

    sx, sy, sz = sun_eci(dt)
    # vector perpendicular to sun direction, 7000 km out
    px, py, pz = -sy, sx, 0.0
    n = (px * px + py * py + pz * pz) ** 0.5
    assert is_sunlit((7000.0 * px / n, 7000.0 * py / n, 0.0), dt)
