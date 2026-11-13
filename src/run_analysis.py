"""Runs the orbit-lab analyses and writes results/ (CSVs + figures).

Usage: python3 src/run_analysis.py
Requires: bash scripts_download_data.sh first (data-raw/ is not committed).
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.orbit.passes import predict_passes
from src.orbit.propagate import ground_track, position_eci_km
from src.orbit.sun import is_sunlit, sun_elevation_deg
from src.orbit.tle import parse_tle_file
from src.orbit.validate import compare_passes, load_published, write_comparison_csv

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data-raw"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

PARIS = (48.8566, 2.3522, 0.035)


def load_iss():
    for tle in parse_tle_file((RAW / "tle-stations.txt").read_text()):
        if tle.norad_id == 25544:
            return tle
    raise SystemExit("ISS TLE not found in data-raw/tle-stations.txt")


GROUND_TRACK_SATS = [
    (25544, "ISS"),
    (48274, "CSS (Tianhe)"),
    (20580, "Hubble (HST)"),
    (10967, "Seasat 1 (polar)"),
]
GROUND_TRACK_HOURS = 12.0


def _draw_land(ax) -> None:
    """Light Natural Earth 110 m land polygons as map background."""
    import json

    with open(RAW / "ne110m-land.geojson") as fh:
        geo = json.load(fh)
    for feat in geo["features"]:
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            ring = poly[0]
            xs = [pt[0] for pt in ring]
            ys = [pt[1] for pt in ring]
            ax.fill(xs, ys, color="#e8e8e8", zorder=0, lw=0)


def _find_sat(norad_id: int):
    for fname in ("tle-stations.txt", "tle-visual.txt", "tle-starlink.txt"):
        for tle in parse_tle_file((RAW / fname).read_text()):
            if tle.norad_id == norad_id:
                return tle
    return None


def fig_ground_tracks(now: datetime) -> None:
    sats = []
    for norad_id, label in GROUND_TRACK_SATS:
        tle = _find_sat(norad_id)
        if tle:
            sats.append((tle, label))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, (tle, label) in enumerate(sats):
        color = colors[idx % len(colors)]
        track = ground_track(tle, now, GROUND_TRACK_HOURS, step_s=90.0)
        lons = [t[2] for t in track]
        lats = [t[1] for t in track]
        # split at the antimeridian for clean lines
        xs, ys = [lons[0]], [lats[0]]
        for i in range(1, len(lons)):
            if abs(lons[i] - lons[i - 1]) > 180.0:
                ax.plot(xs, ys, lw=1.2, color=color)
                xs, ys = [], []
            xs.append(lons[i])
            ys.append(lats[i])
        ax.plot(xs, ys, lw=1.2, color=color)
        ax.plot(lons[0], lats[0], "o", ms=5, color=color, label=label)
    _draw_land(ax)
    ax.plot(PARIS[1], PARIS[0], "r*", ms=14, label="Paris", zorder=5)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-90, 91, 30))
    ax.grid(alpha=0.3)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(f"Ground tracks, {GROUND_TRACK_HOURS:.0f} h from " + now.strftime("%Y-%m-%d %H:%M UTC") + " (SGP4)")
    ax.legend(fontsize=8, ncol=3, loc="lower center")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig-ground-tracks.png", dpi=140)
    plt.close(fig)


def fig_best_pass_sky(iss, best) -> None:
    # sky path of the best pass: az/el sampled every 5 s
    n = int((best.set - best.rise).total_seconds() / 5.0) + 1
    azs, els = [], []
    from src.orbit.frames import eci_to_ecef, topocentric_az_el

    for k in range(n + 1):
        t = best.rise + timedelta(seconds=5.0 * k)
        az, el, _ = topocentric_az_el(
            eci_to_ecef(position_eci_km(iss, t), t), PARIS[0], PARIS[1], PARIS[2]
        )
        azs.append(math.radians(az))
        els.append(el)
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.plot(azs, [90.0 - e for e in els], "b-", lw=2)
    ax.plot(azs[0], 90.0 - els[0], "go", ms=8, label="rise " + best.rise.strftime("%H:%M:%S UTC"))
    ax.plot(azs[-1], 90.0 - els[-1], "rs", ms=8, label="set " + best.set.strftime("%H:%M:%S UTC"))
    ax.set_rlim(0, 90)
    ax.set_rticks([15, 30, 45, 60, 75, 90])
    ax.set_yticklabels(["75°", "60°", "45°", "30°", "15°", "0°"])
    ax.set_title(
        f"ISS over Paris, {best.rise.strftime('%Y-%m-%d')} — max elevation {best.max_el_deg:.0f}°\n"
        f"(satellite sun-lit: {is_sunlit(position_eci_km(iss, best.max_el_time), best.max_el_time)}, "
        f"Sun elevation at observer: {sun_elevation_deg(best.max_el_time, PARIS[0], PARIS[1]):.0f}°)"
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig-best-pass-sky.png", dpi=140)
    plt.close(fig)


def fig_validation(comparisons) -> None:
    matched = [c for c in comparisons if c.matched]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    days = [c.days_ahead for c in matched]
    ax1.plot(days, [c.time_offset_s for c in matched], "bo")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_xlabel("Days after TLE epoch")
    ax1.set_ylabel("Max-time offset (s)\nours minus Heavens-Above")
    ax1.set_title("Pass time offset vs prediction horizon")
    ax1.grid(alpha=0.3)
    ax2.plot(days, [c.el_offset_deg for c in matched], "rs")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("Days after TLE epoch")
    ax2.set_ylabel("Max-elevation offset (deg)")
    ax2.set_title("Max elevation offset (published values are integers)")
    ax2.grid(alpha=0.3)
    fig.suptitle("ISS passes over Paris: our SGP4 predictions vs Heavens-Above (fetched 2026-11-08)")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig-validation.png", dpi=140)
    plt.close(fig)


def fig_evening_profile(iss, passes, evening_date) -> None:
    """Elevation vs time for one evening's ISS passes over Paris.

    Shaded spans: satellite inside the Earth shadow (our cylindrical model).
    Dashed line: Sun elevation at the observer (right axis).
    """
    from src.orbit.frames import eci_to_ecef, topocentric_az_el

    day_passes = [p for p in passes if p.rise.date().isoformat() == evening_date]
    if not day_passes:
        return
    t0 = min(p.rise for p in day_passes) - timedelta(minutes=10)
    t1 = max(p.set for p in day_passes) + timedelta(minutes=10)
    n = int((t1 - t0).total_seconds() / 20.0) + 1
    times, els, sun_els, lit = [], [], [], []
    for k in range(n):
        t = t0 + timedelta(seconds=20.0 * k)
        r = position_eci_km(iss, t)
        _, el, _ = topocentric_az_el(eci_to_ecef(r, t), PARIS[0], PARIS[1], PARIS[2])
        times.append(t)
        els.append(el)
        sun_els.append(sun_elevation_deg(t, PARIS[0], PARIS[1]))
        lit.append(is_sunlit(r, t))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax2 = ax.twinx()
    ax2.plot(times, sun_els, color="orange", ls="--", lw=1.2, label="Sun elevation (right)")
    ax2.axhline(-6, color="orange", lw=0.6, alpha=0.5)
    ax2.set_ylabel("Sun elevation at Paris (deg)", color="darkorange")
    ax2.set_ylim(-60, 60)
    # shade shadowed intervals
    in_sh, start_sh = False, None
    for t, is_lit in zip(times, lit):
        if not is_lit and not in_sh:
            in_sh, start_sh = True, t
        elif is_lit and in_sh:
            ax.axvspan(start_sh, t, color="grey", alpha=0.18, lw=0)
            in_sh = False
    if in_sh:
        ax.axvspan(start_sh, times[-1], color="grey", alpha=0.18, lw=0)
    ax.plot(times, els, "b-", lw=1.8, label="ISS elevation (left)")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(10, color="b", lw=0.6, alpha=0.4)
    ax.set_ylim(-20, 90)
    ax.set_ylabel("ISS elevation above horizon (deg)", color="b")
    ax.set_xlabel(f"UTC time, {evening_date}")
    ax.set_title(
        "ISS evening passes over Paris - grey spans: satellite in Earth shadow (our model)\n"
        "Visible geometry = sun-lit satellite while Sun < -6° at observer"
    )
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(RESULTS / "fig-evening-profile.png", dpi=140)
    plt.close(fig)


def main() -> None:
    iss = load_iss()
    now = datetime.now(timezone.utc).replace(microsecond=0)
    print(f"ISS TLE epoch: {iss.epoch.isoformat()}")

    # 1) all passes over Paris for the Heavens-Above search window
    window_start = datetime(2026, 11, 8, 0, 0, tzinfo=timezone.utc)
    passes = predict_passes(iss, window_start, 10 * 24.0, *PARIS, min_elevation_deg=10.0)
    with open(RESULTS / "predicted-passes-iss-paris.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rise_utc", "rise_az_deg", "max_time_utc", "max_el_deg", "max_az_deg", "set_utc", "set_az_deg", "visible_geometry"])
        for p in passes:
            w.writerow([
                p.rise.isoformat(), round(p.rise_az_deg, 1),
                p.max_el_time.isoformat(), round(p.max_el_deg, 1), round(p.max_az_deg, 1),
                p.set.isoformat(), round(p.set_az_deg, 1), p.visible_geometry,
            ])
    print(f"{len(passes)} passes above 10° over Paris in 10 days")

    # 2) validation against the published table
    published = load_published(str(ROOT / "data-validation" / "heavens-above-iss-paris-2026-11-08.csv"))
    comps = compare_passes(iss, published, passes, iss.epoch, PARIS[0], PARIS[1], PARIS[2])
    write_comparison_csv(comps, str(RESULTS / "validation-iss-paris.csv"))
    matched = [c for c in comps if c.matched]
    print(f"matched {len(matched)}/{len(comps)} published visible passes")
    for c in comps:
        if c.matched:
            print(
                f"  {c.published_date}: dt={c.time_offset_s:+.0f}s "
                f"del={c.el_offset_deg:+.1f}° ({c.days_ahead:.1f}d ahead)"
            )
        else:
            print(f"  {c.published_date}: NO MATCH ({c.days_ahead:.1f}d ahead)")

    # 3) figures
    fig_ground_tracks(now)
    upcoming = [p for p in passes if p.max_el_time > now]
    visible_upcoming = [p for p in upcoming if p.visible_geometry]
    best = max(visible_upcoming or upcoming, key=lambda p: p.max_el_deg, default=None)
    if best:
        fig_best_pass_sky(iss, best)
        print(f"best upcoming pass: {best.rise} max el {best.max_el_deg:.0f}°")
    fig_validation(comps)
    fig_evening_profile(iss, passes, "2026-11-17")

    # 4) summary
    summary = {
        "tle_epoch": iss.epoch.isoformat(),
        "generated_utc": now.isoformat(),
        "passes_10d_above_10deg": len(passes),
        "published_visible_passes": len(comps),
        "matched": len(matched),
        "time_offset_s": [c.time_offset_s for c in matched],
        "el_offset_deg": [c.el_offset_deg for c in matched],
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))



if __name__ == "__main__":
    main()
