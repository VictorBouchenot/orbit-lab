"""orbit-lab CLI: predict satellite passes over an observer.

Examples:
    python3 src/main.py passes --sat ISS --days 2
    python3 src/main.py passes --sat "CSS (TIANHE)" --lat 48.85 --lon 2.35 --min-el 20
    python3 src/main.py track --sat HST --hours 1.5
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from src.orbit.passes import predict_passes
from src.orbit.propagate import ground_track
from src.orbit.tle import parse_tle_file

ROOT = Path(__file__).resolve().parent.parent


def find_tle(query: str):
    for fname in ("tle-stations.txt", "tle-visual.txt", "tle-starlink.txt"):
        path = ROOT / "data-raw" / fname
        if not path.exists():
            continue
        for tle in parse_tle_file(path.read_text()):
            if query.upper() in tle.name.upper() or query == str(tle.norad_id):
                return tle
    raise SystemExit(f"satellite '{query}' not found in data-raw/ (run scripts_download_data.sh first)")


def main() -> None:
    ap = argparse.ArgumentParser(prog="orbit-lab")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_passes = sub.add_parser("passes", help="predict passes over an observer")
    p_passes.add_argument("--sat", default="ISS")
    p_passes.add_argument("--lat", type=float, default=48.8566)
    p_passes.add_argument("--lon", type=float, default=2.3522)
    p_passes.add_argument("--alt-km", type=float, default=0.035)
    p_passes.add_argument("--days", type=float, default=1.0)
    p_passes.add_argument("--min-el", type=float, default=10.0)

    p_track = sub.add_parser("track", help="print sub-satellite ground track")
    p_track.add_argument("--sat", default="ISS")
    p_track.add_argument("--hours", type=float, default=1.5)

    args = ap.parse_args()
    tle = find_tle(args.sat)
    print(f"{tle.name.strip()} (NORAD {tle.norad_id}), TLE epoch {tle.epoch:%Y-%m-%d %H:%M:%S} UTC")

    now = datetime.now(timezone.utc).replace(microsecond=0)
    if args.cmd == "passes":
        passes = predict_passes(tle, now, args.days * 24.0, args.lat, args.lon, args.alt_km, args.min_el)
        if not passes:
            print(f"no pass above {args.min_el}° in the next {args.days} day(s)")
        for p in passes:
            vis = "VISIBLE (sun-lit, dark sky)" if p.visible_geometry else "not visible"
            print(
                f"{p.rise:%Y-%m-%d %H:%M:%S} rise az {p.rise_az_deg:5.1f}°  "
                f"max {p.max_el_deg:4.1f}° at {p.max_el_time:%H:%M:%S}  "
                f"set {p.set:%H:%M:%S} az {p.set_az_deg:5.1f}°  [{vis}]"
            )
    else:
        for t, lat, lon, alt in ground_track(tle, now, args.hours, step_s=60.0):
            print(f"{t:%H:%M:%S}  {lat:+7.2f}°  {lon:+8.2f}°  {alt:6.1f} km")


if __name__ == "__main__":
    main()
