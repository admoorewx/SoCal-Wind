#!/usr/bin/env python3
"""SFO-WMC+ gust set: are the strong-wind misses local high-wind sites or errant stations?

Station set (SFO-WMC+, gust): HVRSG HF006 NV013 DDWAR LVYN2 NV051 NV052 NV011 NNSC DNMOG. Gusts are the
QC'd hourly max from gradient_pipeline match (data/v2/derived/gradients/matched/KSFO). Misses = strong hours
(target > 50 mph) with |GBT error| > 5 mph, from plot_strong_misses.py.

Figures -> images/case_studies/sfo_wmc/
  station_timeseries_full.png    daily max gust per station 2015-2026 with the SFO-WMC gradient (daily
                                 max / min) and the missed hours marked; shows when each station reports
  station_timeseries_events.png  the 6 worst miss events (distinct by > 24 h): hourly gust per station,
                                 the set target (k-th highest gust), the GBT forecast, and the gradient
  station_map.png                terrain (AWS Terrarium tiles) with the stations coloured by their 99th
                                 percentile gust: region, Tahoe-Reno, and a West Reno close-up

Usage:
  python plot_sfo_wmc_stations.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_baseline_models import GRID, HALO, INK, INK2, SURFACE, style
from verify_suspects import dem

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data", "v2"))
MDIR = os.path.join(DATA, "derived", "models")
OUT = os.path.normpath(os.path.join(HERE, "..", "images", "case_studies", "sfo_wmc"))
SG, GRAD = "SFO-WMC+", "SFO-WMC"
MPH = 2.23694
# colour follows the station: the seven reporting since 2022 take categorical slots 1-7
COLOR = {"NV013": "#eb6834", "NV052": "#2a78d6", "NV011": "#1baf7a", "NV051": "#eda100",
         "DNMOG": "#e87ba4", "NNSC": "#008300", "LVYN2": "#6250d6", "HVRSG": "#e34948",
         "HF006": "#8c8b85", "DDWAR": "#8c8b85"}
N_EVENTS = 6
WINDOW_H = 36   # hours either side of an event's worst hour


def load():
    S = pd.read_csv(os.path.join(MDIR, "station_sets.csv"))
    st = S[(S.signed_gradient == SG) & (S.variable == "gust")].stid.tolist()
    meta = pd.read_parquet(os.path.join(DATA, "stations.parquet")).set_index("stid")
    fs = sorted(glob.glob(os.path.join(DATA, "derived", "gradients", "matched", "KSFO", "*.parquet")))
    D = pd.concat([pd.read_parquet(f, filters=[("stid", "in", st)], columns=["stid", "time_utc", "gust_max_mps"])
                   for f in fs], ignore_index=True)
    D["stid"] = D.stid.astype(str)
    D["gust"] = D.gust_max_mps * MPH
    D = D.dropna(subset=["gust"])
    G = pd.read_parquet(os.path.join(DATA, "derived", "gradients", "gradient_timeseries.parquet"),
                        columns=["gradient", "time_utc", "gradient_hpa"], filters=[("gradient", "==", GRAD)])
    T = pd.read_parquet(os.path.join(MDIR, "gbt", "oof_predictions.parquet"),
                        columns=["time_utc", "obs", "pred_gbt_all"],
                        filters=[("signed_gradient", "==", SG), ("variable", "==", "gust"), ("kind", "==", "linear")])
    M = pd.read_parquet(os.path.join(MDIR, "strong_misses", "strong_hours.parquet"))
    M = M[(M.signed_gradient == SG) & (M.variable == "gust") & M.miss]
    return st, meta, D, G, T, M


def order(st, D):
    """Stations by first report, so the full-record panels read top-down in time."""
    first = D.groupby("stid").time_utc.min()
    return sorted(st, key=lambda s: first.get(s, pd.Timestamp.max.tz_localize("UTC")))


# ---------------------------------------------------------------- full record
def full_record(st, meta, D, G, M):
    st = order(st, D)
    fig, axes = plt.subplots(len(st) + 1, 1, figsize=(17, 2 + 1.25 * (len(st) + 1)), sharex=True,
                             facecolor=SURFACE, gridspec_kw={"height_ratios": [1.6] + [1] * len(st)})
    gd = G.set_index("time_utc").gradient_hpa.resample("D").agg(["max", "min"])
    ax = axes[0]
    style(ax)
    ax.fill_between(gd.index, gd["min"], gd["max"], color=INK2, alpha=0.35, lw=0, step="mid")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("SFO−WMC\n(hPa)", fontsize=9)
    ax.set_title("SFO − WMC sea-level pressure gradient, daily range (positive = higher at SFO)", loc="left",
                 fontsize=10, color=INK)
    miss_days = M.time_utc.dt.floor("D").unique()
    ymax = D.groupby("stid").gust.quantile(0.999).max()
    daily = D.set_index("time_utc").groupby("stid").gust.resample("D").max()
    for ax, s in zip(axes[1:], st):
        style(ax)
        d = daily.loc[s]
        ax.vlines(miss_days, 0, ymax * 1.05, color=GRID, lw=0.8, zorder=1)
        ax.plot(d.index, d.values, color=COLOR[s], lw=0.7, zorder=3)
        ax.set_ylim(0, ymax * 1.05)
        r = meta.loc[s]
        ax.text(0.003, 0.93, f"{s} · {r['name']} · {r.elevation_m:,.0f} m · {r.network_shortname}",
                transform=ax.transAxes, va="top", fontsize=8.5, color=INK, path_effects=HALO)
        ax.text(0.997, 0.93, f"p99 {D[D.stid == s].gust.quantile(0.99):.0f} mph", transform=ax.transAxes,
                va="top", ha="right", fontsize=8.5, color=INK, path_effects=HALO)
        ax.set_ylabel("mph", fontsize=8)
        ax.tick_params(labelsize=7.5)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlim(pd.Timestamp("2015-01-01", tz="UTC"), D.time_utc.max())
    fig.suptitle(f"{SG} gust set: daily max gust per station, 2015–2026", x=0.01, ha="left", fontsize=14,
                 color=INK)
    fig.text(0.01, 1 - 0.75 / fig.get_figheight(), "Grey vertical lines: days with a strong-gust miss (target > "
             "50 mph, GBT error > 5 mph). Stations ordered by first report. Shared y axis.",
             fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.9 / fig.get_figheight()))
    fig.savefig(os.path.join(OUT, "station_timeseries_full.png"), dpi=110, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- event close-ups
def pick_events(M):
    chosen = []
    for r in M.sort_values("err").itertuples():
        if all(abs((r.time_utc - c.time_utc).total_seconds()) > 86400 for c in chosen):
            chosen.append(r)
        if len(chosen) == N_EVENTS:
            break
    return chosen


def events(st, D, G, T, M):
    ev = pick_events(M)
    fig = plt.figure(figsize=(18, 13), facecolor=SURFACE)
    outer = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.18)
    used = set()
    for e, r in enumerate(ev):
        t0, t1 = r.time_utc - pd.Timedelta(hours=WINDOW_H), r.time_utc + pd.Timedelta(hours=WINDOW_H)
        inner = outer[e // 3, e % 3].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        ax = fig.add_subplot(inner[0])
        gx = fig.add_subplot(inner[1], sharex=ax)
        style(ax)
        style(gx)
        d = D[(D.time_utc >= t0) & (D.time_utc <= t1)]
        for s in st:
            g = d[d.stid == s]
            if len(g):
                used.add(s)
                ax.plot(g.time_utc, g.gust, color=COLOR[s], lw=1.5, zorder=3)
        t = T[(T.time_utc >= t0) & (T.time_utc <= t1)]
        ax.plot(t.time_utc, t.obs, color=INK, lw=2.2, ls=(0, (1, 1.5)), zorder=5)
        ax.plot(t.time_utc, t.pred_gbt_all, color=INK, lw=1.8, ls=(0, (6, 2)), zorder=5)
        ax.axvline(r.time_utc, color=INK2, lw=0.8, zorder=1)
        ax.axhline(50, color=GRID, lw=1, zorder=1)
        ax.set_title(f"{r.time_utc:%Y-%m-%d %H:%M} UTC · target {r.obs:.0f}, GBT {r.pred_gbt_all:.0f} mph",
                     loc="left", fontsize=9.5, color=INK)
        ax.set_ylabel("Gust (mph)", fontsize=8.5)
        ax.tick_params(axis="x", labelbottom=False)
        g = G[(G.time_utc >= t0) & (G.time_utc <= t1)]
        gx.plot(g.time_utc, g.gradient_hpa, color=INK2, lw=1.6)
        gx.axhline(0, color=GRID, lw=1)
        gx.axvline(r.time_utc, color=INK2, lw=0.8)
        gx.set_ylabel("SFO−WMC\n(hPa)", fontsize=8)
        gx.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
        gx.xaxis.set_major_formatter(mdates.DateFormatter("%d/%H"))
        gx.tick_params(labelsize=7.5)
        gx.set_xlim(t0, t1)
    hs = [Line2D([], [], color=COLOR[s], lw=2) for s in st if s in used]
    ls = [s for s in st if s in used]
    hs += [Line2D([], [], color=INK, lw=2.2, ls=(0, (1, 1.5))), Line2D([], [], color=INK, lw=1.8, ls=(0, (6, 2)))]
    ls += ["set target (k-th highest gust)", "GBT forecast"]
    fig.legend(hs, ls, loc="lower center", ncol=len(ls), frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle(f"{SG}: the {N_EVENTS} worst strong-gust misses, ±{WINDOW_H} h", x=0.01, ha="left",
                 fontsize=14, color=INK)
    fig.text(0.01, 0.945, "Hourly max gust per station; x axis day/hour UTC. Vertical line: the event's worst hour. "
             "Target and forecast only drawn for qualifying hours (≥ 50% of stations reporting).",
             fontsize=9, color=INK2)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.91, bottom=0.08)
    fig.savefig(os.path.join(OUT, "station_timeseries_events.png"), dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return ev


# ---------------------------------------------------------------- terrain maps
def terrain(ax, box, z):
    E, lons, lats = dem(*box, z=z)
    E = np.where(E < -50, 0, E)
    ls = mcolors.LightSource(azdeg=315, altdeg=40)
    shade = ls.hillshade(E, vert_exag=2 if z <= 8 else 0.6, dx=1, dy=1)
    ext = (lons[0], lons[-1], lats[-1], lats[0])
    ax.imshow(E, extent=ext, cmap="Greys", vmin=-1500, vmax=4500, zorder=0, interpolation="bilinear")
    ax.imshow(shade, extent=ext, cmap="gray", alpha=0.35, zorder=1, interpolation="bilinear")
    cs = ax.contour(lons, lats, E, levels=np.arange(500, 4001, 500), colors="#ffffff", linewidths=0.4,
                    alpha=0.7, zorder=2)
    ax.clabel(cs, fontsize=6, fmt="%d m", colors=INK2)
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_aspect(1 / np.cos(np.radians((box[2] + box[3]) / 2)))
    ax.tick_params(labelsize=7.5)


def station_map(st, meta, D, M):
    p99 = D.groupby("stid").gust.quantile(0.99)
    miss_med = D.merge(M[["time_utc"]], on="time_utc").groupby("stid").gust.median()
    cmap = mcolors.LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    norm = mcolors.Normalize(30, 60)
    boxes = [("Region: SFO to WMC", (-123.0, -117.2, 37.3, 41.3), 8, True),
             ("Tahoe – Reno", (-120.35, -119.55, 38.85, 39.75), 10, False),
             ("West Reno / Verdi close-up", (-120.02, -119.76, 39.46, 39.60), 12, False)]
    fig, axes = plt.subplots(1, 3, figsize=(19, 7.8), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.25, 1, 1.15]})
    for ax, (title, box, z, region) in zip(axes, boxes):
        terrain(ax, box, z)
        if region:
            for a in ("KSFO", "KWMC"):
                r = meta.loc[a]
                ax.plot(r.lon, r.lat, marker="s", ms=8, color=INK, mec=SURFACE, zorder=6)
                ax.annotate(a[1:], (r.lon, r.lat), xytext=(6, -10), textcoords="offset points", fontsize=9,
                            color=INK, path_effects=HALO, zorder=7)
            x0, x1, y0, y1 = boxes[1][1]
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=INK, lw=1, zorder=5)
        else:
            x0, x1, y0, y1 = boxes[2][1]
            if title.startswith("Tahoe"):
                ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=INK, lw=1, zorder=5)
        for s in st:
            r = meta.loc[s]
            if not (box[0] <= r.lon <= box[1] and box[2] <= r.lat <= box[3]):
                continue
            ax.scatter(r.lon, r.lat, s=150 if not region else 70, c=[cmap(norm(p99.get(s, np.nan)))],
                       edgecolors=SURFACE, linewidths=1.8, zorder=8)
            if region and box is boxes[0][1] and s not in ("HF006",):
                continue  # region panel labels only the outlying site; the cluster is labelled in the zooms
            cx0, cx1, cy0, cy1 = boxes[2][1]
            if title.startswith("Tahoe") and cx0 <= r.lon <= cx1 and cy0 <= r.lat <= cy1:
                continue  # labelled in the close-up
            lbl = f"{s}\n{r.elevation_m:,.0f} m · p99 {p99.get(s, np.nan):.0f}"
            if s in miss_med:
                lbl += f" · miss med {miss_med[s]:.0f}"
            ax.annotate(lbl, (r.lon, r.lat), xytext=(8, 6), textcoords="offset points", fontsize=7.5,
                        color=INK, path_effects=HALO, zorder=9)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.04, pad=0.08, aspect=50)
    cb.set_label("Station 99th percentile hourly max gust over its record (mph)", fontsize=9, color=INK2)
    fig.suptitle(f"{SG} gust stations on the terrain", x=0.01, ha="left", fontsize=14, color=INK)
    fig.text(0.01, 0.925, "Labels: elevation · p99 gust (mph) · median gust during the set's missed strong hours "
             "(when reporting). Terrain: AWS Terrarium, 500 m contours.", fontsize=9, color=INK2)
    fig.savefig(os.path.join(OUT, "station_map.png"), dpi=120, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return p99, miss_med


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    os.makedirs(OUT, exist_ok=True)
    st, meta, D, G, T, M = load()
    full_record(st, meta, D, G, M)
    ev = events(st, D, G, T, M)
    p99, mm = station_map(st, meta, D, M)
    print("events:", [f"{r.time_utc:%Y-%m-%d %H}Z err {r.err:.0f}" for r in ev])
    print(pd.DataFrame({"p99": p99, "miss_median": mm}).round(1).sort_values("p99").to_string())
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
