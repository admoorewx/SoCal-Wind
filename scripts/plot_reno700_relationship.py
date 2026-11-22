#!/usr/bin/env python3
"""SFO-WMC+ gust forecast error vs ERA5 700 hPa wind over West Reno (diagnostic, not a model input).

Uses the hourly 700 hPa u/v series from era5_fetch_reno700.py (grid point 39.50N 120.00W, nearest the
West Reno / Verdi cluster that sets the SFO-WMC+ gust target) and the out-of-fold GBT (gbt_all) gust
forecasts. Strong = target > 50 mph; missed = forecast more than 5 mph too low (as in plot_strong_misses.py).
Obs hours end at :51-:56, so each is matched to the next full ERA5 hour.

One figure, four panels:
  A  share of hours with target > 50 mph, observed vs GBT forecast, by 700 hPa speed
  B  cumulative distribution of 700 hPa speed: all hours, strong hours, missed strong hours
  C  mean forecast error (forecast - obs) by |SFO-WMC gradient| x 700 hPa speed (cells < 30 h blank)
  D  700 hPa wind direction and speed: all hours (density) and missed strong hours (dots)

-> images/case_studies/sfo_wmc/reno700/reno700_vs_error.png
-> data/v2/derived/models/case_studies/sfo_wmc_reno700_bins.csv   (panel A/C numbers)

Usage:
  python plot_reno700_relationship.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from plot_baseline_models import GRID, HALO, INK, INK2, SURFACE, THR_COLORS, style
from plot_set_stations import load

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "images", "case_studies", "sfo_wmc", "reno700"))
TOUT = os.path.normpath(os.path.join(HERE, "..", "data", "v2", "derived", "models", "case_studies"))
SRC = "/mnt/d/ERA5/reno700/era5_ua700_*.nc"
PT = dict(latitude=39.5, longitude=-120.0)
KT = 1.943844
STRONG, MISS_ERR, MIN_CELL = 50, -5, 30
WBINS = [0, 20, 30, 40, 50, 60, 200]
WLAB = ["<20", "20–30", "30–40", "40–50", "50–60", "≥60"]
GBINS = [0, 2, 4, 6, 8, 99]
GLAB = ["0–2", "2–4", "4–6", "6–8", "≥8"]
C_OBS, C_FC = THR_COLORS[0], THR_COLORS[1]          # categorical slots 1, 2
C_ALL = "#9a998f"
DIVERGE = mcolors.LinearSegmentedColormap.from_list(  # red = forecast too low, blue = too high, grey = 0
    "div", ["#8f1d1c", "#e34948", "#f4b6b5", "#f0efec", "#9ec5f4", "#3987e5", "#0d366b"])


def era5_series():
    ds = xr.concat([xr.open_dataset(f)[["u", "v"]].sel(**PT).load() for f in sorted(glob.glob(SRC))], dim="valid_time")
    u, v = ds.u.values.ravel() * KT, ds.v.values.ravel() * KT
    return pd.DataFrame({"time": pd.to_datetime(ds.valid_time.values).tz_localize("UTC"), "w700": np.hypot(u, v),
                         "d700": (np.degrees(np.arctan2(-u, -v)) + 360) % 360})


def matched():
    *_, G, T, _, _ = load("SFO-WMC+")
    d = (T.assign(time=T.time_utc.dt.ceil("h")).merge(era5_series(), on="time")
         .merge(G[["time_utc", "gradient_hpa"]], on="time_utc"))
    d["err"] = d.pred_gbt_all - d.obs
    d["strong"] = d.obs > STRONG
    d["missed"] = d.strong & (d.err < MISS_ERR)
    d["wbin"] = pd.cut(d.w700, WBINS, right=False, labels=WLAB)
    d["gbin"] = pd.cut(d.gradient_hpa.abs(), GBINS, right=False, labels=GLAB)
    return d


def panel_rate(ax, d):
    style(ax)
    g = d.groupby("wbin", observed=False)
    obs, fc, n = 100 * g.strong.mean(), 100 * g.apply(lambda x: (x.pred_gbt_all > STRONG).mean()), g.size()
    x, w = np.arange(len(WLAB)), 0.38
    ax.bar(x - w / 2 - 0.01, obs.values, w, color=C_OBS, label="observed target > 50 mph", zorder=3)
    ax.bar(x + w / 2 + 0.01, fc.values, w, color=C_FC, label="GBT forecast > 50 mph", zorder=3)
    for xi, o, f in zip(x, obs.values, fc.values):
        if o >= 1:
            ax.annotate(f"{o:.0f}%", (xi - w / 2, o), xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=8, color=INK)
        if f >= 1:
            ax.annotate(f"{f:.0f}%", (xi + w / 2, f), xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=8, color=INK)
    ax.set_xticks(x, [f"{l}\n{c:,} h" for l, c in zip(WLAB, n.values)], fontsize=8.5)
    ax.set_xlabel("ERA5 700 hPa wind speed at Reno (kt)", fontsize=9)
    ax.set_ylabel("Share of hours (%)", fontsize=9)
    ax.set_ylim(0, max(obs.max(), fc.max()) * 1.15)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)
    ax.set_title("A  Strong gusts become common above ~40 kt aloft; the forecast lags",
                 loc="left", fontsize=10.5, color=INK)
    return pd.DataFrame({"hours": n, "obs_gt50_pct": obs, "gbt_gt50_pct": fc,
                         "mean_err_mph": g.err.mean()})


def panel_cdf(ax, d):
    style(ax)
    groups = [("all hours", d, C_ALL, 1.6), ("strong hours", d[d.strong], C_OBS, 2.0),
              ("missed strong hours", d[d.missed], C_FC, 2.0)]
    for lbl, g, c, lw in groups:
        s = np.sort(g.w700.to_numpy())
        y = 100 * np.arange(1, len(s) + 1) / len(s)
        share = 100 * (s >= 40).mean()
        ax.plot(s, y, color=c, lw=lw, zorder=4, label=f"{lbl} ({len(s):,} h): {share:.0f}% at ≥ 40 kt")
    ax.axvline(40, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 100)
    ax.set_xlabel("ERA5 700 hPa wind speed at Reno (kt)", fontsize=9)
    ax.set_ylabel("Cumulative share of hours (%)", fontsize=9)
    ax.set_title("B  Strong and missed hours sit at high 700 hPa speeds", loc="left", fontsize=10.5, color=INK)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK2)
    ax.annotate("40 kt", (40, 97), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK2, va="top")


def panel_heat(ax, d, fig):
    p = d.pivot_table(index="gbin", columns="wbin", values="err", aggfunc="mean", observed=False)
    n = d.pivot_table(index="gbin", columns="wbin", values="err", aggfunc="size", observed=False).fillna(0)
    v = p.where(n >= MIN_CELL)
    im = ax.imshow(v.values, cmap=DIVERGE, norm=mcolors.TwoSlopeNorm(0, -10, 10), origin="lower", aspect="auto")
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.isfinite(v.values[i, j]):
                ax.text(j, i, f"{v.values[i, j]:+.1f}\n{int(n.values[i, j]):,} h", ha="center", va="center",
                        fontsize=7.5, color=SURFACE if abs(v.values[i, j]) > 6 else INK)
    ax.set_xticks(range(len(WLAB)), WLAB, fontsize=8.5)
    ax.set_yticks(range(len(GLAB)), GLAB, fontsize=8.5)
    ax.set_xlabel("ERA5 700 hPa wind speed at Reno (kt)", fontsize=9)
    ax.set_ylabel("|SFO−WMC gradient| (hPa)", fontsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Mean forecast − obs (mph)", fontsize=8.5, color=INK2)
    cb.ax.tick_params(labelsize=7.5, length=0)
    cb.outline.set_visible(False)
    ax.set_title("C  At any gradient, the forecast drops lower as 700 hPa wind rises", loc="left",
                 fontsize=10.5, color=INK)
    return v, n


def panel_polar(ax, d):
    ax.set_facecolor(SURFACE)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    th, r = np.radians(d.d700), d.w700
    H, te, re = np.histogram2d(th, r, bins=[np.linspace(0, 2 * np.pi, 37), np.arange(0, 85, 5)])
    T, R = np.meshgrid(te, re, indexing="ij")
    grey = mcolors.LinearSegmentedColormap.from_list("g", [SURFACE, "#d6d5cd", "#9a998f", "#52514e"])
    ax.pcolormesh(T, R, np.where(H > 0, H, np.nan), cmap=grey, norm=mcolors.LogNorm(1, H.max()), zorder=1)
    m = d[d.missed]
    ax.scatter(np.radians(m.d700), m.w700, s=14, color=C_FC, edgecolor=SURFACE, linewidth=0.6, zorder=4,
               label=f"missed strong hours ({len(m):,})")
    leg_kw = dict(markerscale=1.8)
    ax.set_rlim(0, 80)
    ax.set_rticks([20, 40, 60, 80])
    ax.set_rlabel_position(135)
    ax.tick_params(labelsize=8, colors=INK2)
    ax.set_xticks(np.radians(np.arange(0, 360, 45)), ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    ax.grid(color=GRID, lw=0.6)
    ax.spines["polar"].set_color(GRID)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.25, -0.12), frameon=False, fontsize=8.5, labelcolor=INK2, **leg_kw)
    ax.text(1.12, -0.08, "grey: all hours (log density)\nradius: speed (kt), angle: direction from",
            transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
    ax.set_title("D  Missed strong hours: SW flow aloft at 30–65 kt", loc="left", fontsize=10.5, color=INK,
                 pad=18, x=-0.25)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TOUT, exist_ok=True)
    d = matched()
    fig = plt.figure(figsize=(16, 12.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.2, left=0.06, right=0.97, top=0.9, bottom=0.07)
    a = panel_rate(fig.add_subplot(gs[0, 0]), d)
    panel_cdf(fig.add_subplot(gs[0, 1]), d)
    v, n = panel_heat(fig.add_subplot(gs[1, 0]), d, fig)
    panel_polar(fig.add_subplot(gs[1, 1], projection="polar"), d)
    r = np.corrcoef(d.err, d.w700)[0, 1]
    fig.suptitle("SFO-WMC+ gust forecast error vs ERA5 700 hPa wind over West Reno, 2015–2025",
                 x=0.01, ha="left", y=0.985, fontsize=14, color=INK)
    fig.text(0.01, 0.945, f"{len(d):,} SFO-WMC+ gust hours; GBT (gbt_all) out-of-fold forecasts. Strong = target "
             f"> {STRONG} mph ({d.strong.sum():,} h); missed = forecast > 5 mph too low ({d.missed.sum():,} h). ERA5 grid "
             f"point 39.50N 120.00W. Correlation of forecast error with 700 hPa speed: {r:.2f}. Diagnostic only: "
             "700 hPa wind is not a model input.", fontsize=9, color=INK2, wrap=True)
    path = os.path.join(OUT, "reno700_vs_error.png")
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    heat = v.stack().rename("mean_err_mph").to_frame().join(n.stack().rename("hours"))
    a.to_csv(os.path.join(TOUT, "sfo_wmc_reno700_bins.csv"), float_format="%.2f")
    heat.to_csv(os.path.join(TOUT, "sfo_wmc_reno700_grad_x_w700.csv"), float_format="%.2f")
    print(a.round(1).to_string())
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
