#!/usr/bin/env python3
"""ERA5 synoptic maps for the worst SFO-WMC+ strong-gust misses (files from era5_fetch_events.py).

One figure per event: rows = peak - 12 h, peak - 6 h, peak (ERA5 hour after the obs hour ending :56);
columns:
  Surface   2 m temperature (shaded), mean sea-level pressure (contours, 4 hPa), 10 m wind (barbs, kt)
  850 hPa   temperature (shaded), height (contours, 30 m), wind (barbs)
  700 hPa   wind speed (shaded), height (contours, 30 m), wind (barbs)
  500 hPa   wind speed (shaded), height (contours, 60 m), temperature (dashed, 4 C), wind (barbs)
Plus composite_peak.png: the mean of the six peak hours, same four panels.
Colour scales are fixed across all figures so events can be compared. Markers: SFO and WMC (squares),
West Reno station cluster (star). Panel titles give ERA5 10 m gust and 700 hPa wind at the grid point
nearest the cluster.

-> images/case_studies/sfo_wmc/era5/event_YYYYMMDD_HHz.png, composite_peak.png

Usage:
  python plot_era5_event_maps.py
"""
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
OUT = os.path.normpath(os.path.join(HERE, "..", "images", "case_studies", "sfo_wmc", "era5"))
G0, KT = 9.80665, 1.943844
PTS = {"SFO": (-122.366, 37.620), "WMC": (-117.807, 40.902)}
RENO = (-119.92, 39.52)   # West Reno / Verdi cluster centre
PROJ = ccrs.LambertConformal(central_longitude=-121, central_latitude=40, standard_parallels=(33, 45))
EXTENT = (-131.5, -113, 32.5, 48.5)
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


def at_reno(f, k):
    i = np.abs(f["lat"] - RENO[1]).argmin()
    j = np.abs(f["lon"] - RENO[0]).argmin()
    return f[k][i, j]


def base(ax):
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.6, edgecolor=INK)
    ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.35, edgecolor=INK2)
    for name, (x, y) in PTS.items():
        ax.plot(x, y, "s", ms=5, color=INK, mec=SURFACE, transform=ccrs.PlateCarree(), zorder=8)
    ax.plot(*RENO, "*", ms=11, color="#eda100", mec=INK, mew=0.6, transform=ccrs.PlateCarree(), zorder=8)


def contour(ax, f, k, step, **kw):
    v = f[k]
    lv = np.arange(np.floor(v.min() / step) * step, v.max() + step, step)
    cs = ax.contour(f["lon"], f["lat"], v, levels=lv, transform=ccrs.PlateCarree(), **kw)
    ax.clabel(cs, fontsize=6, fmt="%d", inline=True)


def barbs(ax, f, uk, vk, every=8):
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


def reno_note(f):
    w700 = np.hypot(at_reno(f, "u700"), at_reno(f, "v700"))
    d700 = (np.degrees(np.arctan2(-at_reno(f, "u700"), -at_reno(f, "v700"))) + 360) % 360
    return f"Reno pt: 10 m gust {at_reno(f, 'gust'):.0f} kt · 700 hPa {d700:03.0f}°/{w700:.0f} kt"


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
        axes[i, 3].text(1.0, -0.03, reno_note(f), transform=axes[i, 3].transAxes, ha="right", va="top",
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
    fig.text(0.01, 1 - 0.55 / fig.get_figheight(), sub, fontsize=9, color=INK2)
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
    os.makedirs(OUT, exist_ok=True)
    peaks = []
    for peak, hrs in event_hours():
        rows = [(f"{'peak' if h == peak else f'peak − {int((peak - h).total_seconds() // 3600)} h'} · "
                 f"{h:%Y-%m-%d %H}Z", fields(h)) for h in hrs]
        peaks.append(rows[-1][1])
        figure(rows, f"SFO-WMC+ strong-gust miss, peak {peak:%Y-%m-%d %H}Z: ERA5",
               "★ West Reno station cluster, ■ SFO / WMC. Barbs in kt (every 2°). MSLP every 4 hPa, 850/700 hPa "
               "height every 30 m, 500 hPa every 60 m, 500 hPa T dashed every 4 °C. Fixed colour scales. 850 hPa lies below "
               "ground over the Sierra and Great Basin (ERA5 extrapolates there).",
               os.path.join(OUT, f"event_{peak:%Y%m%d_%H}z.png"))
        print(f"event {peak:%Y-%m-%d %H}Z done")
    figure([("mean of the 6 peak hours", mean_fields(peaks))], "SFO-WMC+ strong-gust misses: ERA5 composite of "
           "the six peak hours", "Simple mean of the six peak-hour fields (no anomalies). Same scales as the event "
           "figures.", os.path.join(OUT, "composite_peak.png"))
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
