"""Pass prediction: rise/max/set of satellite overflights for an observer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .frames import eci_to_ecef, topocentric_az_el
from .propagate import position_eci_km
from .sun import is_sunlit, sun_elevation_deg
from .tle import TLE


@dataclass
class PassEvent:
    rise: datetime
    rise_az_deg: float
    max_el_time: datetime
    max_el_deg: float
    max_az_deg: float
    set: datetime
    set_az_deg: float
    visible_geometry: bool  # sat sun-lit AND observer in darkness at max


def _elevation(tle: TLE, dt: datetime, lat: float, lon: float, alt_km: float) -> float:
    return topocentric_az_el(eci_to_ecef(position_eci_km(tle, dt), dt), lat, lon, alt_km)[1]


def _refine_crossing(
    tle: TLE, t_lo: datetime, t_hi: datetime, lat: float, lon: float, alt_km: float
) -> datetime:
    """Bisect the horizon crossing between a below and an above sample."""
    for _ in range(40):
        t_mid = t_lo + (t_hi - t_lo) / 2
        if _elevation(tle, t_mid, lat, lon, alt_km) > 0.0:
            t_hi = t_mid
        else:
            t_lo = t_mid
    return t_lo + (t_hi - t_lo) / 2


def predict_passes(
    tle: TLE,
    start: datetime,
    hours: float,
    obs_lat: float,
    obs_lon: float,
    obs_alt_km: float = 0.035,
    min_elevation_deg: float = 10.0,
    step_s: float = 10.0,
) -> list[PassEvent]:
    """All passes above `min_elevation_deg` between start and start+hours.

    Coarse 10 s sampling, then bisection on the 0 deg crossings and a
    parabolic-style refine of the maximum by dense local scan.
    """
    start = start.astimezone(timezone.utc)
    n = int(hours * 3600.0 / step_s)
    els: list[float] = []
    for k in range(n + 1):
        t = start + timedelta(seconds=k * step_s)
        els.append(_elevation(tle, t, obs_lat, obs_lon, obs_alt_km))

    passes: list[PassEvent] = []
    k = 0
    while k < n:
        if els[k] <= 0.0 < els[k + 1]:
            i_lo = k
            # find the corresponding set
            j = k + 1
            while j < n and not (els[j] > 0.0 >= els[j + 1]):
                j += 1
            if j >= n:
                break  # window ends mid-pass
            rise = _refine_crossing(
                tle,
                start + timedelta(seconds=i_lo * step_s),
                start + timedelta(seconds=(i_lo + 1) * step_s),
                obs_lat, obs_lon, obs_alt_km,
            )
            set_ = _refine_crossing(
                tle,
                start + timedelta(seconds=j * step_s),
                start + timedelta(seconds=(j + 1) * step_s),
                obs_lat, obs_lon, obs_alt_km,
            )
            # max elevation: fine 2 s scan inside the pass
            best_t, best_el = rise, -90.0
            span_s = (set_ - rise).total_seconds()
            for m in range(int(span_s / 2.0) + 1):
                t = rise + timedelta(seconds=2.0 * m)
                el = _elevation(tle, t, obs_lat, obs_lon, obs_alt_km)
                if el > best_el:
                    best_el, best_t = el, t
            if best_el >= min_elevation_deg:
                az_rise = topocentric_az_el(
                    eci_to_ecef(position_eci_km(tle, rise), rise), obs_lat, obs_lon, obs_alt_km
                )[0]
                az_max = topocentric_az_el(
                    eci_to_ecef(position_eci_km(tle, best_t), best_t), obs_lat, obs_lon, obs_alt_km
                )[0]
                az_set = topocentric_az_el(
                    eci_to_ecef(position_eci_km(tle, set_), set_), obs_lat, obs_lon, obs_alt_km
                )[0]
                visible = is_sunlit(position_eci_km(tle, best_t), best_t) and (
                    sun_elevation_deg(best_t, obs_lat, obs_lon) < -6.0
                )
                passes.append(
                    PassEvent(
                        rise=rise, rise_az_deg=az_rise,
                        max_el_time=best_t, max_el_deg=best_el, max_az_deg=az_max,
                        set=set_, set_az_deg=az_set,
                        visible_geometry=visible,
                    )
                )
            k = j + 1
        else:
            k += 1
    return passes
