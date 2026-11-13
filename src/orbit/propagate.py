"""SGP4 propagation wrapper around the reference `sgp4` package.

We deliberately use the battle-tested SGP4 implementation (Python port of
Vallado's code) rather than re-deriving the perturbation model: SGP4 is a
large, validated standard model, and re-implementing it in a week would add
risk, not insight. Our own maths lives in frames.py / sun.py / passes.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sgp4.api import Satrec, jday

from .frames import ecef_to_geodetic, eci_to_ecef
from .tle import TLE


def _satrec(tle: TLE) -> Satrec:
    return Satrec.twoline2rv(tle.line1, tle.line2)


def position_eci_km(tle: TLE, dt: datetime) -> tuple[float, float, float]:
    """SGP4 position in the TEME frame (treated as ECI here; see frames.py)."""
    dt = dt.astimezone(timezone.utc)
    jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)
    err, r, _v = _satrec(tle).sgp4(jd, fr)
    if err != 0:
        raise RuntimeError(f"SGP4 error code {err}")
    return (r[0], r[1], r[2])


def subpoint(tle: TLE, dt: datetime) -> tuple[float, float, float]:
    """Sub-satellite point: (lat_deg, lon_deg, alt_km)."""
    return ecef_to_geodetic(eci_to_ecef(position_eci_km(tle, dt), dt))


def ground_track(
    tle: TLE, start: datetime, hours: float, step_s: float = 60.0
) -> list[tuple[datetime, float, float, float]]:
    """(time, lat, lon, alt) samples along the orbit for mapping."""
    out = []
    n = int(hours * 3600.0 / step_s)
    for k in range(n + 1):
        t = start + timedelta(seconds=k * step_s)
        lat, lon, alt = subpoint(tle, t)
        out.append((t, lat, lon, alt))
    return out
