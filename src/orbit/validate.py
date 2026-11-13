"""Validation against externally published ISS pass tables (Heavens-Above).

Matching rule: for each published pass, take our predicted pass whose
max-elevation time is closest (must be within 10 minutes, else the pass is
reported unmatched). We then compare max-time offset and max-elevation
difference. Published values are at minute/degree resolution, so tiny
offsets are quantization, not signal.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .frames import eci_to_ecef, topocentric_az_el
from .passes import PassEvent
from .propagate import position_eci_km
from .sun import is_sunlit, sun_elevation_deg

CET = timezone(timedelta(hours=1))  # the published table is in CET


@dataclass
class PassComparison:
    published_date: str
    published_max_time_utc: datetime
    published_max_el_deg: float
    matched: bool
    ours_max_time_utc: datetime | None
    ours_max_el_deg: float | None
    time_offset_s: float | None
    el_offset_deg: float | None
    days_ahead: float


def _az_to_deg(az: str) -> float:
    table = {
        "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5,
        "SE": 135, "SSE": 157.5, "S": 180, "SSW": 202.5, "SW": 225,
        "WSW": 247.5, "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
    }
    return table[az]


def load_published(path: str) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split(",")
            date, mag, st, sa, saz, mt, ma, maz, et, ea, eaz = parts
            dt = datetime.strptime(f"{date} {mt}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=CET)
            rows.append(
                {
                    "date": date,
                    "max_time_utc": dt.astimezone(timezone.utc),
                    "max_el_deg": float(ma),
                }
            )
    return rows


def visible_max_point(
    tle, p: PassEvent, obs_lat: float, obs_lon: float, obs_alt_km: float
) -> tuple[datetime, float]:
    """Highest point of the pass while the satellite is sun-lit AND the
    observer is in darkness (Sun below -6 deg) - the same definition
    Heavens-Above uses for its 'Highest point' column on visible passes.
    If the pass is never properly visible, falls back to the geometric max.
    """
    n = int((p.set - p.rise).total_seconds() / 5.0) + 1
    best_t, best_el = p.max_el_time, p.max_el_deg
    found = False
    for k in range(n + 1):
        t = p.rise + timedelta(seconds=5.0 * k)
        r_eci = position_eci_km(tle, t)
        if not is_sunlit(r_eci, t):
            continue
        if sun_elevation_deg(t, obs_lat, obs_lon) >= -6.0:
            continue
        _, el, _ = topocentric_az_el(eci_to_ecef(r_eci, t), obs_lat, obs_lon, obs_alt_km)
        if not found or el > best_el:
            best_t, best_el, found = t, el, True
    return best_t, best_el


def compare_passes(
    tle,
    published: list[dict],
    ours: list[PassEvent],
    tle_epoch: datetime,
    obs_lat: float,
    obs_lon: float,
    obs_alt_km: float = 0.035,
) -> list[PassComparison]:
    out = []
    for pub in published:
        best, best_dt = None, None
        for p in ours:
            dt = abs((p.max_el_time - pub["max_time_utc"]).total_seconds())
            if best_dt is None or dt < best_dt:
                best, best_dt = p, dt
        if best is not None and best_dt <= 600.0:
            vis_t, vis_el = visible_max_point(tle, best, obs_lat, obs_lon, obs_alt_km)
            out.append(
                PassComparison(
                    published_date=pub["date"],
                    published_max_time_utc=pub["max_time_utc"],
                    published_max_el_deg=pub["max_el_deg"],
                    matched=True,
                    ours_max_time_utc=vis_t,
                    ours_max_el_deg=vis_el,
                    time_offset_s=(vis_t - pub["max_time_utc"]).total_seconds(),
                    el_offset_deg=vis_el - pub["max_el_deg"],
                    days_ahead=(pub["max_time_utc"] - tle_epoch).total_seconds() / 86400.0,
                )
            )
        else:
            out.append(
                PassComparison(
                    published_date=pub["date"],
                    published_max_time_utc=pub["max_time_utc"],
                    published_max_el_deg=pub["max_el_deg"],
                    matched=False,
                    ours_max_time_utc=None,
                    ours_max_el_deg=None,
                    time_offset_s=None,
                    el_offset_deg=None,
                    days_ahead=(pub["max_time_utc"] - tle_epoch).total_seconds() / 86400.0,
                )
            )
    return out


def write_comparison_csv(comparisons: list[PassComparison], path: str) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "date", "published_max_time_utc", "published_max_el_deg",
                "matched", "ours_max_time_utc", "ours_max_el_deg",
                "time_offset_s", "el_offset_deg", "days_ahead",
            ]
        )
        for c in comparisons:
            w.writerow(
                [
                    c.published_date,
                    c.published_max_time_utc.isoformat(),
                    c.published_max_el_deg,
                    c.matched,
                    c.ours_max_time_utc.isoformat() if c.ours_max_time_utc else "",
                    c.ours_max_el_deg if c.ours_max_el_deg is not None else "",
                    c.time_offset_s if c.time_offset_s is not None else "",
                    c.el_offset_deg if c.el_offset_deg is not None else "",
                    round(c.days_ahead, 2),
                ]
            )
