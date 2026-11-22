#!/usr/bin/env python3
"""ERA5 synoptic maps for strong-gust miss events of a gradient set (files from era5_fetch_events.py).

One figure per event: rows = peak - 12 h, peak - 6 h, peak (ERA5 hour after the obs hour ending :56);
columns:
  Surface   2 m temperature (shaded), mean sea-level pressure (contours, 4 hPa), 10 m wind (barbs, kt)
  850 hPa   temperature (shaded), height (contours, 30 m), wind (barbs)
  700 hPa   wind speed (shaded), height (contours, 30 m), wind (barbs)
  500 hPa   wind speed (shaded), height (contours, 60 m), temperature (dashed, 4 C), wind (barbs)
Plus, for SFO-WMC+, composite_peak.png: the mean of the six peak hours (SAN-IPL+ events span seasons, so
no composite). Colour scales are fixed across all figures so events can be compared. Markers: gradient
endpoints (squares) and the set's focus stations (stars). The note under each row gives ERA5 10 m gust and
850/700 hPa wind at the grid point nearest the first focus station, and the ERA5 MSLP difference between
the endpoints (compare with the observed gradient).

-> images/case_studies/<set>/era5/event_YYYYMMDD_HHz.png (+ composite_peak.png for SFO-WMC+)

Usage:
  python plot_era5_event_maps.py                                  # SFO-WMC+
  python plot_era5_event_maps.py --set SAN-IPL+ --dates 2017-05-26 2023-01-03
"""
import argparse
import textwrap
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from era5_fetch_events import ROOT, event_hours
from plot_baseline_models import INK, INK2, SURFACE

HERE = os.path.dirname(os.path.abspath(__file__))
G0, KT = 9.80665, 1.943844
# Per set: gradient endpoints (lon, lat), focus stations (first = the one the row note samples), map
# projection centre and extent, barb spacing (grid steps of 0.25 deg), whether to draw a composite.
SETS = {
    "SFO-WMC+": dict(slug="sfo_wmc", ends={"SFO": (-122.366, 37.620), "WMC": (-117.807, 40.902)},
                     focus={"West Reno cluster": (-119.92, 39.52)}, centre=(-121, 40),
                     extent=(-131.5, -113, 32.5, 48.5), every=8, composite=True),
    # The ERA5 files stop at 30N, so the southern edge is close to the data edge.
    "SAN-IPL+": dict(slug="san_ipl_pos", ends={"SAN": (-117.183, 32.734), "IPL": (-115.579, 32.834)},
                     focus={"Whitewater (WWAC1)": (-116.657, 33.949), "Jawbone (JWBC1)": (-118.227, 35.295),
                            "Mountain Springs (TNSC1)": (-116.094, 32.675), "Lake Palmdale (LPDC1)": (-118.102, 34.537)},
                     centre=(-118, 35), extent=(-125.5, -110.5, 30.2, 40.5), every=6, composite=False),
    # Offshore (Santa Ana) direction; extends north to show the Great Basin high.
    "SAN-IPL-": dict(slug="san_ipl_neg", ends={"SAN": (-117.183, 32.734), "IPL": (-115.579, 32.834)},
                     focus={"Highland Springs (HSGC1)": (-116.953, 33.878), "Chino Hills (CQHC1)": (-117.738, 33.914),
                            "Wiley Ridge (WLYC1)": (-118.841, 34.372), "Miller Ranch (MLXC1)": (-119.011, 34.148)},
                     centre=(-118, 36), extent=(-125.5, -110.5, 30.2, 43), every=6, composite=False),
}
CFG = SETS["SFO-WMC+"]
PROJ = None
TEMP = mcolors.LinearSegmentedColormap.from_list(
    "temp", ["#0d366b", "#3987e5", "#9ec5f4", "#f0efec", "#f4b6b5", "#e34948", "#8f1d1c"])
SPEED = mcolors.LinearSegmentedColormap.from_list(
    "speed", ["#f6f5fd", "#c4bdf4", "#9085e9", "#6250d6", "#3a2a99", "#1f1460"])
SCALE = {"t2m": (np.arange(-10, 36, 2.5), TEMP), "t850": (np.arange(-20, 26, 2.5), TEMP),
         "w700": (np.arange(0, 81, 10), SPEED), "w500": (np.arange(0, 141, 10), SPEED)}
PANELS = [("sfc", "Surface: 2 m T (°C), MSLP (hPa), 10 m wind"), ("850", "850 hPa: T (°C), height, wind"),
          ("700", "700 hPa: wind speed (kt), height, wind"), ("500", "500 hPa: wind speed (kt), height, T, wind")]


def fields(t):
    """All plotted fields at one UTC hour as a dict of 2-D arrays (lat descending)."""
    t = pd.Timestamp(t).tz_localize(None) if pd.Timestamp(t).tzinfo else pd.Timestamp(t)
    s = xr.open_dataset(os.path.join(ROOT, f"era5_sfc_{t:%Y%m%d}.nc")).sel(valid_time=t)
    u = xr.open_dataset(os.path.join(ROOT, f"era5_ua_{t:%Y%m%d}.nc")).sel(valid_time=t)
    lv = lambda v, p: u[v].sel(pressure_level=p).values  # noqa: E731
    f = {"lon": s.longitude.values, "lat": s.latitude.values,
         "t2m": s.t2m.values - 273.15, "msl": s.msl.values / 100, "u10": s.u10.values * KT, "v10": s.v10.values * KT,
         "gust": s.i10fg.values * KT}
    for p in (850, 700, 500):
        f[f"z{p}"] = lv("z", p) / G0
        f[f"t{p}"] = lv("t", p) - 273.15
        f[f"u{p}"] = lv("u", p) * KT
        f[f"v{p}"] = lv("v", p) * KT
    return f


def mean_fields(fs):
    return {k: (fs[0][k] if k in ("lon", "lat") else np.mean([f[k] for f in fs], axis=0)) for k in fs[0]}


def at(f, k, pt):
    i = np.abs(f["lat"] - pt[1]).argmin()
    j = np.abs(f["lon"] - pt[0]).argmin()
    return f[k][i, j]


def base(ax):
    ax.set_extent(CFG["extent"], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.6, edgecolor=INK)
    ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.35, edgecolor=INK2)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.35, edgecolor=INK2)
    for x, y in CFG["ends"].values():
        ax.plot(x, y, "s", ms=5, color=INK, mec=SURFACE, transform=ccrs.PlateCarree(), zorder=8)
    for i, (x, y) in enumerate(CFG["focus"].values()):
        ax.plot(x, y, "*", ms=11 if i == 0 else 8, color="#eda100", mec=INK, mew=0.6, transform=ccrs.PlateCarree(),
                zorder=8)


def contour(ax, f, k, step, **kw):
    v = f[k]
    lv = np.arange(np.floor(v.min() / step) * step, v.max() + step, step)
    cs = ax.contour(f["lon"], f["lat"], v, levels=lv, transform=ccrs.PlateCarree(), **kw)
    ax.clabel(cs, fontsize=6, fmt="%d", inline=True)


def barbs(ax, f, uk, vk):
    every = CFG["every"]
    s = (slice(None, None, every), slice(None, None, every))
    lon, lat = np.meshgrid(f["lon"], f["lat"])
    ax.barbs(lon[s], lat[s], f[uk][s], f[vk][s], length=4.6, lw=0.45, color=INK, transform=ccrs.PlateCarree(),
             zorder=6)


def shade(ax, f, v, key):
    lv, cmap = SCALE[key]
    return ax.contourf(f["lon"], f["lat"], v, levels=lv, cmap=cmap, extend="both", transform=ccrs.PlateCarree(),
                       zorder=1)


def draw_panel(ax, f, which):
    base(ax)
    if which == "sfc":
        m = shade(ax, f, f["t2m"], "t2m")
        contour(ax, f, "msl", 4, colors=INK, linewidths=0.8, zorder=4)
        barbs(ax, f, "u10", "v10")
    elif which == "850":
        m = shade(ax, f, f["t850"], "t850")
        contour(ax, f, "z850", 30, colors=INK, linewidths=0.8, zorder=4)
        barbs(ax, f, "u850", "v850")
    elif which == "700":
        m = shade(ax, f, np.hypot(f["u700"], f["v700"]), "w700")
        contour(ax, f, "z700", 30, colors=INK, linewidths=0.8, zorder=4)
        barbs(ax, f, "u700", "v700")
    else:
        m = shade(ax, f, np.hypot(f["u500"], f["v500"]), "w500")
        contour(ax, f, "z500", 60, colors=INK, linewidths=0.8, zorder=4)
        contour(ax, f, "t500", 4, colors="#e34948", linewidths=0.7, linestyles="--", zorder=4)
        barbs(ax, f, "u500", "v500")
    return m


def wind(f, p, pt):
    u, v = at(f, f"u{p}", pt), at(f, f"v{p}", pt)
    return f"{(np.degrees(np.arctan2(-u, -v)) + 360) % 360:03.0f}°/{np.hypot(u, v):.0f} kt"


def point_note(f):
    name, pt = next(iter(CFG["focus"].items()))
    (a, pa), (b, pb) = CFG["ends"].items()
    dp = at(f, "msl", pa) - at(f, "msl", pb)
    return (f"{name.split(' (')[0]} pt: 10 m gust {at(f, 'gust', pt):.0f} kt · 850 hPa {wind(f, 850, pt)} · "
            f"700 hPa {wind(f, 700, pt)} · ERA5 {a}−{b} MSLP {dp:+.1f} hPa")


def figure(rows, title, sub, path):
    """rows: list of (row label, fields dict)."""
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(20, 4.6 * n + 1.4), facecolor=SURFACE,
                             subplot_kw={"projection": PROJ}, squeeze=False)
    mappables = {}
    for i, (lbl, f) in enumerate(rows):
        for j, (which, ptitle) in enumerate(PANELS):
            ax = axes[i, j]
            mappables[which] = draw_panel(ax, f, which)
            if i == 0:
                ax.set_title(ptitle, loc="left", fontsize=10, color=INK)
        axes[i, 0].text(-0.04, 0.5, lbl, transform=axes[i, 0].transAxes, rotation=90, va="center", ha="right",
                        fontsize=10.5, color=INK)
        axes[i, 3].text(1.0, -0.03, point_note(f), transform=axes[i, 3].transAxes, ha="right", va="top",
                        fontsize=8, color=INK2)
    fig.subplots_adjust(left=0.03, right=0.995, top=1 - 0.95 / fig.get_figheight(), bottom=0.9 / fig.get_figheight(),
                        wspace=0.03, hspace=0.1)
    for j, (which, _) in enumerate(PANELS):
        pos = axes[-1, j].get_position()
        cax = fig.add_axes([pos.x0 + 0.1 * pos.width, 0.45 / fig.get_figheight(), 0.8 * pos.width,
                            0.12 / fig.get_figheight()])
        cb = fig.colorbar(mappables[which], cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=7.5)
        cb.set_label({"sfc": "2 m T (°C)", "850": "850 hPa T (°C)", "700": "700 hPa wind (kt)",
                      "500": "500 hPa wind (kt)"}[which], fontsize=8.5, color=INK2)
    fig.suptitle(title, x=0.01, ha="left", y=0.995, fontsize=14, color=INK)
    fig.text(0.01, 1 - 0.45 / fig.get_figheight(), textwrap.fill(sub, 230), fontsize=9, color=INK2, va="top")
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main():
    global CFG, PROJ
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="SFO-WMC+", choices=sorted(SETS))
    ap.add_argument("--dates", nargs="*", help="UTC dates of the set's events (worst miss hour on each)")
    a = ap.parse_args()
    CFG = SETS[a.set]
    PROJ = ccrs.LambertConformal(central_longitude=CFG["centre"][0], central_latitude=CFG["centre"][1],
                                 standard_parallels=(33, 45))
    out = os.path.normpath(os.path.join(HERE, "..", "images", "case_studies", CFG["slug"], "era5"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
    os.makedirs(out, exist_ok=True)
    ends = " / ".join(CFG["ends"])
    stars = ", ".join(CFG["focus"])
    peaks = []
    for peak, hrs in event_hours(a.set, a.dates):
        rows = [(f"{'peak' if h == peak else f'peak − {int((peak - h).total_seconds() // 3600)} h'} · "
                 f"{h:%Y-%m-%d %H}Z", fields(h)) for h in hrs]
        peaks.append(rows[-1][1])
        figure(rows, f"{a.set} strong-gust miss, peak {peak:%Y-%m-%d %H}Z: ERA5",
               f"★ {stars}, ■ {ends}. Barbs in kt (every {CFG['every'] / 4:g}°). MSLP every 4 hPa, 850/700 hPa "
               "height every 30 m, 500 hPa every 60 m, 500 hPa T dashed every 4 °C. Fixed colour scales. 850 hPa lies below "
               "ground over the Sierra and Great Basin (ERA5 extrapolates there).",
               os.path.join(out, f"event_{peak:%Y%m%d_%H}z.png"))
        print(f"event {peak:%Y-%m-%d %H}Z done")
    if not CFG["composite"]:
        print(f"Wrote figures to {out}")
        return
    figure([("mean of the 6 peak hours", mean_fields(peaks))], "SFO-WMC+ strong-gust misses: ERA5 composite of "
           "the six peak hours", "Simple mean of the six peak-hour fields (no anomalies). Same scales as the event "
           "figures.", os.path.join(out, "composite_peak.png"))
    print(f"Wrote figures to {out}")


if __name__ == "__main__":
    main()
