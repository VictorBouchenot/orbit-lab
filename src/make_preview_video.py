"""Renders the animated orbit-lab preview video (results/preview-orbit-lab.mp4).

One simulated evening: the ISS visible pass over Paris on 2026-11-17
(17:31-17:42 UTC, max el 86 deg - the pass validated against Heavens-Above).

Panels, all driven by the same simulation state at each instant:
  left   - Earth in orthographic projection centered on Europe, with the
           day/night terminator from our own solar model, the ISS ground
           track (trailing 30 min) and the current sub-satellite point.
  middle - elevation of the ISS above the Paris horizon vs time, with a
           moving cursor; grey spans = satellite inside the Earth shadow.
  right  - sky plot (azimuth/elevation, polar) with the ISS position.

Physics: SGP4 (TLE of 2026-11-07), our GMST/frame conversions, our
cylindrical shadow model. No decoration: every pixel shown is computed.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle

from src.orbit.frames import ecef_to_geodetic, eci_to_ecef, topocentric_az_el
from src.orbit.passes import predict_passes
from src.orbit.propagate import position_eci_km, subpoint
from src.orbit.sun import is_sunlit, sun_elevation_deg, sun_eci
from src.orbit.tle import parse_tle_file

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data-raw"
RESULTS = ROOT / "results"

PARIS = (48.8566, 2.3522, 0.035)
PASS_DATE = datetime(2026, 11, 17, tzinfo=timezone.utc)
T_START = datetime(2026, 11, 17, 17, 0, tzinfo=timezone.utc)
T_END = datetime(2026, 11, 17, 18, 10, tzinfo=timezone.utc)
STEP_S = 15.0
FPS = 12

VIEW_LAT, VIEW_LON = 42.0, 12.0  # orthographic view center: Europe


def load_iss():
    for tle in parse_tle_file((RAW / "tle-stations.txt").read_text()):
        if tle.norad_id == 25544:
            return tle
    raise SystemExit("ISS TLE missing")


# ---------- orthographic helpers ----------

def ortho_forward(lat_deg, lon_deg):
    """Project to the view plane; returns (x, y, front-facing mask)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    lat0, lon0 = math.radians(VIEW_LAT), math.radians(VIEW_LON)
    cos_c = np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(lon - lon0)
    x = np.cos(lat) * np.sin(lon - lon0)
    y = np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(lon - lon0)
    return x, y, cos_c > 0.0


def disk_unit_vectors(res=400):
    """Per-pixel ECEF-style unit vectors for the orthographic disk (NaN outside)."""
    lat0, lon0 = math.radians(VIEW_LAT), math.radians(VIEW_LON)
    xs = np.linspace(-1, 1, res)
    ys = np.linspace(1, -1, res)  # image row 0 is top (origin='upper'): y=+1
    gx, gy = np.meshgrid(xs, ys)
    rho2 = gx * gx + gy * gy
    inside = rho2 <= 1.0
    z = np.sqrt(np.where(inside, 1.0 - rho2, 1.0))
    sin_c = np.sqrt(np.minimum(rho2, 1.0))
    cos_c = z
    sc = np.where(sin_c < 1e-9, 1.0, sin_c)
    lat = np.arcsin(np.clip(cos_c * math.sin(lat0) + gy * sin_c * math.cos(lat0) / sc, -1, 1))
    lon = lon0 + np.arctan2(gx * sin_c, sin_c * math.cos(lat0) * cos_c - gy * math.sin(lat0))
    ux = np.cos(lat) * np.cos(lon)
    uy = np.cos(lat) * np.sin(lon)
    uz = np.sin(lat)
    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)
    return ux, uy, uz, inside, lat_deg, lon_deg


def earth_image(sun, disk, land_mask, res=400):
    """RGB image of the Earth disk: ocean/land with a smooth day/night blend."""
    ux, uy, uz, inside, lat_deg, lon_deg = disk
    sx, sy, sz = sun
    illum = ux * sx + uy * sy + uz * sz
    dayw = np.clip((illum + 0.05) * 6.0, 0.0, 1.0)  # 1 = full day
    # land lookup (nearest 1 deg cell)
    lat_i = np.clip((89.5 - lat_deg).astype(int), 0, 179)
    lon_i = np.clip((lon_deg + 179.5).astype(int), 0, 359)
    land = land_mask[lat_i, lon_i]
    # polar cap fallback: longitude is undefined at the poles and the mask
    # lookup jitters there; Arctic cap is ocean, Antarctic cap is land
    land = np.where(lat_deg > 87.0, False, land)
    land = np.where(lat_deg < -87.0, True, land)
    ocean_day = np.array([0.615, 0.722, 0.800])
    ocean_night = np.array([0.035, 0.055, 0.110])
    land_day = np.array([0.875, 0.902, 0.855])
    land_night = np.array([0.075, 0.090, 0.120])
    img = np.zeros((res, res, 3))
    for c in range(3):
        day_col = np.where(land, land_day[c], ocean_day[c])
        night_col = np.where(land, land_night[c], ocean_night[c])
        img[..., c] = night_col + (day_col - night_col) * dayw
    img = np.where(inside[..., None], img, 0.0)
    return img


def build_land_mask():
    """1 deg lat/lon land mask from the Natural Earth polygons."""
    from matplotlib.path import Path as MplPath

    with open(RAW / "ne110m-land.geojson") as fh:
        geo = json.load(fh)
    lats = np.linspace(89.5, -89.5, 180)
    lons = np.linspace(-179.5, 179.5, 360)
    glon, glat = np.meshgrid(lons, lats)
    pts = np.column_stack([glon.ravel(), glat.ravel()])
    mask = np.zeros(len(pts), dtype=bool)
    for feat in geo["features"]:
        geom = feat["geometry"]
        rings = [geom["coordinates"][0]] if geom["type"] == "Polygon" else [p[0] for p in geom["coordinates"]]
        for ring in rings:
            path = MplPath(np.asarray(ring))
            mask |= path.contains_points(pts)
    return mask.reshape(180, 360)


def main() -> None:
    iss = load_iss()
    n_frames = int((T_END - T_START).total_seconds() / STEP_S) + 1
    times = [T_START + timedelta(seconds=STEP_S * k) for k in range(n_frames)]

    # precompute state vectors
    els, azs, lits, sunels, subs = [], [], [], [], []
    for t in times:
        r_eci = position_eci_km(iss, t)
        r_ecef = eci_to_ecef(r_eci, t)
        az, el, _ = topocentric_az_el(r_ecef, *PARIS)
        azs.append(az)
        els.append(el)
        lits.append(is_sunlit(r_eci, t))
        sunels.append(sun_elevation_deg(t, PARIS[0], PARIS[1]))
        subs.append(ecef_to_geodetic(r_ecef))

    disk = disk_unit_vectors()
    land_mask = build_land_mask()

    fig = plt.figure(figsize=(13.2, 5.4), dpi=150)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.35, 1.0], wspace=0.16)
    ax_earth = fig.add_subplot(gs[0])
    ax_el = fig.add_subplot(gs[1])
    ax_sky = fig.add_subplot(gs[2], polar=True)

    fig.patch.set_facecolor("#0d1117")
    for ax in (ax_earth, ax_el, ax_sky):
        ax.set_facecolor("#0d1117")

    # middle panel: static curve + cursor
    ax_el.plot(times, els, color="#58a6ff", lw=2.4)
    ax_el.axhline(0, color="#aab4c0", lw=1.0)
    ax_el.axhline(10, color="#58a6ff", lw=0.8, alpha=0.5)
    in_sh = False
    for k, t in enumerate(times):
        if not lits[k] and not in_sh:
            in_sh, sh0 = True, t
        elif lits[k] and in_sh:
            ax_el.axvspan(sh0, t, color="#8b949e", alpha=0.18, lw=0)
            in_sh = False
    if in_sh:
        ax_el.axvspan(sh0, times[-1], color="#8b949e", alpha=0.18, lw=0)
    ax_el.set_ylim(-20, 90)
    ax_el.set_xlim(times[0], times[-1])
    ax_el.set_ylabel("ISS elevation above Paris horizon (deg)", color="#e6edf3", fontsize=11.5)
    ax_el.tick_params(colors="#b6c2cf", labelsize=10)
    ax_el.set_title("Elevation over Paris", color="#e6edf3", fontsize=12.5)
    for spine in ax_el.spines.values():
        spine.set_color("#3d444d")
    cursor = ax_el.axvline(times[0], color="#ffa657", lw=2.0)
    el_readout = ax_el.text(0.02, 0.04, "", transform=ax_el.transAxes, color="#ffa657", fontsize=11.5)
    import matplotlib.dates as mdates

    ax_el.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone.utc))

    # right panel: polar sky plot
    ax_sky.set_theta_zero_location("N")
    ax_sky.set_theta_direction(-1)
    ax_sky.set_rlim(0, 90)
    ax_sky.set_rticks([30, 60, 90])
    ax_sky.set_yticklabels(["60°", "30°", "horizon"], color="#b6c2cf", fontsize=9)
    ax_sky.tick_params(colors="#b6c2cf", labelsize=9)
    ax_sky.grid(color="#3d444d", lw=0.7)
    ax_sky.set_title("Sky over Paris (azimuth / elevation)", color="#e6edf3", fontsize=12.5, pad=16)
    (sky_path,) = ax_sky.plot([], [], color="#58a6ff", lw=2.0, alpha=0.9)
    (sky_dot,) = ax_sky.plot([], [], "o", ms=11, color="#ffcc33", mec="k")

    time_title = fig.suptitle("", color="#e6edf3", fontsize=13)

    def frame(k):
        t = times[k]
        sun = sun_eci(t)
        ax_earth.clear()
        ax_earth.set_facecolor("#0d1117")
        ax_earth.set_xlim(-1.05, 1.05)
        ax_earth.set_ylim(-1.05, 1.05)
        ax_earth.set_aspect("equal")
        ax_earth.axis("off")
        ax_earth.imshow(
            earth_image(sun, disk, land_mask), extent=(-1, 1, -1, 1), origin="upper", zorder=1,
            interpolation="bilinear",
        )
        ax_earth.add_patch(Circle((0, 0), 1.0, fill=False, color="#33506b", lw=1.2, zorder=3))
        # trailing ground track: last 30 min
        k0 = max(0, k - 60)
        lats = np.array([s[0] for s in subs[k0 : k + 1]])
        lons = np.array([s[1] for s in subs[k0 : k + 1]])
        x, y, front = ortho_forward(lats, lons)
        i = 0
        while i < len(x):
            if front[i]:
                j = i
                while j < len(x) and front[j]:
                    j += 1
                ax_earth.plot(x[i:j], y[i:j], color="#ff7b72", lw=2.2, zorder=4)
                i = j
            else:
                i += 1
        lat, lon, _alt = subs[k]
        xs, ys, fr = ortho_forward(np.array([lat]), np.array([lon]))
        if fr[0]:
            col = "#ffcc33" if lits[k] else "#6e7681"
            ax_earth.plot(xs[0], ys[0], "o", ms=12, color=col, mec="k", mew=0.8, zorder=5)
        px, py, _ = ortho_forward(np.array([PARIS[0]]), np.array([PARIS[1]]))
        ax_earth.plot(px[0], py[0], marker="*", ms=15, color="#ff9f1c", mec="k", mew=0.7, zorder=5)
        ax_earth.annotate("Paris", (px[0], py[0]), xytext=(6, -12), textcoords="offset points",
                          fontsize=10, color="#ffd68a", zorder=5)
        state = "sun-lit" if lits[k] else "in Earth shadow"
        ax_earth.set_title(f"ISS ground track — {state}", color="#e6edf3", fontsize=12.5)

        cursor.set_xdata([t, t])
        el_readout.set_text(
            f"elevation {els[k]:5.1f}°   azimuth {azs[k]:5.1f}°   Sun {sunels[k]:5.1f}°"
        )
        # sky plot path up to now (above horizon only)
        path_az, path_el = [], []
        for m in range(k + 1):
            if els[m] >= 0:
                path_az.append(math.radians(azs[m]))
                path_el.append(90.0 - els[m])
        sky_path.set_data(path_az, path_el)
        if els[k] >= 0:
            sky_dot.set_data([math.radians(azs[k])], [90.0 - els[k]])
        else:
            sky_dot.set_data([], [])
        time_title.set_text(
            f"ISS over Paris — 2026-11-17 {t:%H:%M:%S} UTC  (SGP4, TLE epoch 2026-11-07; "
            f"validated against Heavens-Above)"
        )
        return []

    writer = FFMpegWriter(fps=FPS, bitrate=5000)
    out = RESULTS / "preview-orbit-lab.mp4"
    with writer.saving(fig, str(out), dpi=150):
        for k in range(n_frames):
            frame(k)
            writer.grab_frame()
    plt.close(fig)
    print(f"wrote {out} ({n_frames} frames)")


if __name__ == "__main__":
    main()
