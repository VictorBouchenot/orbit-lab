import pytest

from src.orbit.tle import parse_tle, parse_tle_file
from pathlib import Path

SAMPLE = Path(__file__).parent / "data" / "iss-sample.tle"


def load_iss():
    return parse_tle_file(SAMPLE.read_text())[0]


def test_parse_iss_fields():
    tle = load_iss()
    assert tle.norad_id == 25544
    assert abs(tle.inclination_deg - 51.6322) < 1e-6
    assert 0.0 < tle.eccentricity < 0.01  # ISS is near-circular
    assert 15.4 < tle.mean_motion_rev_per_day < 15.6


def test_epoch_parsed_utc():
    tle = load_iss()
    assert tle.epoch.year == 2026
    assert tle.epoch.tzinfo is not None


def test_checksum_catches_corruption():
    lines = SAMPLE.read_text().splitlines()
    bad = lines[1][:20] + ("9" if lines[1][20] != "9" else "8") + lines[1][21:]
    with pytest.raises(ValueError, match="checksum"):
        parse_tle(lines[0], bad, lines[2])


def test_rejects_swapped_lines():
    lines = SAMPLE.read_text().splitlines()
    with pytest.raises(ValueError):
        parse_tle(lines[0], lines[2], lines[1])
