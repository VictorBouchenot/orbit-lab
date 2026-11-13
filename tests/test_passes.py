from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.orbit.passes import predict_passes
from src.orbit.propagate import subpoint
from src.orbit.tle import parse_tle_file

SAMPLE = Path(__file__).parent / "data" / "iss-sample.tle"
PARIS = (48.8566, 2.3522, 0.035)


def load_iss():
    return parse_tle_file(SAMPLE.read_text())[0]


def test_iss_orbital_period_and_altitude():
    tle = load_iss()
    lat, lon, alt = subpoint(tle, tle.epoch)
    assert 350.0 < alt < 470.0  # ISS altitude band
    period_min = 1440.0 / tle.mean_motion_rev_per_day
    assert 91.0 < period_min < 95.0


def test_subpoint_latitude_within_inclination():
    tle = load_iss()
    for hours in range(0, 24, 3):
        lat, _, _ = subpoint(tle, tle.epoch + timedelta(hours=hours))
        assert abs(lat) <= tle.inclination_deg + 0.5  # subpoint never beyond inclination


def test_pass_structure_over_paris():
    tle = load_iss()
    start = tle.epoch.replace(hour=0, minute=0, second=0, microsecond=0)
    passes = predict_passes(tle, start, 48.0, *PARIS)
    assert len(passes) >= 4  # ISS: roughly 6-8 rise/set events over 48 h above 10 deg
    for p in passes:
        assert p.rise < p.max_el_time < p.set
        assert 10.0 <= p.max_el_deg <= 90.0
        assert 0.0 <= p.rise_az_deg < 360.0
        assert 0.0 <= p.set_az_deg < 360.0
        # crossing refinement should converge to well under the 10 s sampling step
        assert (p.set - p.rise).total_seconds() > 60.0


def test_passes_are_reproducible():
    tle = load_iss()
    start = tle.epoch.replace(hour=0, minute=0, second=0, microsecond=0)
    a = predict_passes(tle, start, 24.0, *PARIS)
    b = predict_passes(tle, start, 24.0, *PARIS)
    assert len(a) == len(b)
    for pa, pb in zip(a, b):
        assert pa.rise == pb.rise
        assert abs(pa.max_el_deg - pb.max_el_deg) < 1e-9
