"""Two-Line Element (TLE) set parsing and validation.

TLE format reference: CelesTrak documentation
https://www.celestrak.org/NORAD/documentation/gp-data-formats.php
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class TLE:
    name: str
    line1: str
    line2: str
    norad_id: int
    epoch: datetime  # UTC, from line 1
    inclination_deg: float
    raan_deg: float
    eccentricity: float
    arg_perigee_deg: float
    mean_anomaly_deg: float
    mean_motion_rev_per_day: float


def _checksum_ok(line: str) -> bool:
    """TLE lines end with a modulo-10 checksum digit."""
    if len(line) < 69:
        return False
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return total % 10 == int(line[68])


def _parse_epoch(line1: str) -> datetime:
    """Epoch: columns 33-43 of line 1, YYDDD.DDDDDDDD (UTC)."""
    year2 = int(line1[18:20])
    year = 2000 + year2 if year2 < 57 else 1900 + year2  # per TLE convention
    day = float(line1[20:32])
    day_int = int(day)
    frac = day - day_int
    base = datetime(year, 1, 1, tzinfo=timezone.utc)
    from datetime import timedelta

    return base + timedelta(days=day_int - 1, seconds=frac * 86400.0)


def parse_tle(name: str, line1: str, line2: str) -> TLE:
    """Parse one TLE. Raises ValueError on structural problems."""
    line1, line2 = line1.rstrip("\n"), line2.rstrip("\n")
    if not line1.startswith("1 ") or not line2.startswith("2 "):
        raise ValueError("lines must start with '1 ' and '2 '")
    if not _checksum_ok(line1) or not _checksum_ok(line2):
        raise ValueError("bad TLE checksum")
    norad1, norad2 = int(line1[2:7]), int(line2[2:7])
    if norad1 != norad2:
        raise ValueError("NORAD id mismatch between lines")

    return TLE(
        name=name.strip(),
        line1=line1,
        line2=line2,
        norad_id=norad1,
        epoch=_parse_epoch(line1),
        inclination_deg=float(line2[8:16]),
        raan_deg=float(line2[17:25]),
        eccentricity=float("0." + line2[26:33].strip()),
        arg_perigee_deg=float(line2[34:42]),
        mean_anomaly_deg=float(line2[43:51]),
        mean_motion_rev_per_day=float(line2[52:63]),
    )


def parse_tle_file(text: str) -> list[TLE]:
    """Parse a CelesTrak-style 3-line (name + 2 lines) TLE file."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    out: list[TLE] = []
    i = 0
    while i + 2 < len(lines) + 1 and i + 2 <= len(lines):
        name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
        if not (l1.startswith("1 ") and l2.startswith("2 ")):
            i += 1
            continue
        out.append(parse_tle(name, l1, l2))
        i += 3
    return out
