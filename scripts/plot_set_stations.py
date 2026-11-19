#!/usr/bin/env python3
"""Station diagnostics for one gradient set's strong-gust misses (any set size).

For a signed gradient (gust set), using the QC'd hourly max gusts from gradient_pipeline match and the
strong-hour misses from plot_strong_misses.py (target > 50 mph, |GBT error| > 5 mph):

  TARGET SETTERS  in each hour the target is the k-th highest gust, k = max(2, ceil(0.1 * n_reporting));
                  the stations ranked 1..k "set" the target. For each station: the share of missed hours in
                  which it was a setter.

Figures -> images/case_studies/<set>/
  station_record.png    per station (rows, south to north) and month: 99th percentile gust (blank = not
                        reporting); top strip = missed strong hours per month
  station_events.png    6 worst misses (distinct by > 24 h), +-36 h: the target setters at the worst hour in
                        colour, other stations thin grey, set target, GBT forecast, gradient
  station_map.png       all stations on terrain coloured by p99 gust, ring size = setter share in misses;
                        plus a close-up of the main cluster
-> data/v2/derived/models/case_studies/<set>_stations.csv   per-station table (distance, rho, n, p99,
                                                            setter share)

Usage:
  python plot_set_stations.py SAN-IPL+
"""
import argparse
import glob
import math
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
MPH = 2.23694
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
SEQ = mcolors.LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
N_EVENTS, WINDOW_H = 6, 36


def km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def load(sg):
    grad = sg[:-1]
    st = pd.read_csv(os.path.join(MDIR, "station_sets.csv"))
    st = st[(st.signed_gradient == sg) & (st.variable == "gust")][["stid", "rho", "n"]]
    meta = pd.read_parquet(os.path.join(DATA, "stations.parquet")).set_index("stid")
    a, b = ("K" + x for x in grad.split("-"))
    fs = sorted(glob.glob(os.path.join(DATA, "derived", "gradients", "matched", a, "*.parquet")))
    D = pd.concat([pd.read_parquet(f, filters=[("stid", "in", st.stid.tolist())],
                                   columns=["stid", "time_utc", "gust_max_mps"]) for f in fs], ignore_index=True)
    D["stid"] = D.stid.astype(str)
    D["gust"] = D.gust_max_mps * MPH
    D = D.dropna(subset=["gust"]).drop(columns="gust_max_mps")
    missing = set(st.stid) - set(D.stid)
    if missing:
        print(f"WARNING no matched gusts for {sorted(missing)}")
    G = pd.read_parquet(os.path.join(DATA, "derived", "gradients", "gradient_timeseries.parquet"),
                        columns=["time_utc", "gradient_hpa"], filters=[("gradient", "==", grad)])
    T = pd.read_parquet(os.path.join(MDIR, "gbt", "oof_predictions.parquet"), columns=["time_utc", "obs", "pred_gbt_all"],
                        filters=[("signed_gradient", "==", sg), ("variable", "==", "gust"), ("kind", "==", "linear")])
    M = pd.read_parquet(os.path.join(MDIR, "strong_misses", "strong_hours.parquet"))
    M = M[(M.signed_gradient == sg) & (M.variable == "gust") & M.miss]
    mid = ((meta.loc[a].lat + meta.loc[b].lat) / 2, (meta.loc[a].lon + meta.loc[b].lon) / 2)
    st = st.merge(meta[["name", "lat", "lon", "elevation_m", "network_shortname"]], left_on="stid", right_index=True)
    st["dist_km"] = km(st.lat, st.lon, *mid)
    return st, meta, D, G, T, M, (a, b)


def setters(D, times):
    """Stations ranked 1..k at each given hour (k from the hour's reporting count)."""
    d = D[D.time_utc.isin(times)].copy()
    d["n_rep"] = d.groupby("time_utc").stid.transform("size")
    d["k"] = np.maximum(2, np.ceil(0.1 * d.n_rep)).astype(int)
    d["rank"] = d.groupby("time_utc").gust.rank(ascending=False, method="first")
    return d[d["rank"] <= d.k]


def table(st, D, M):
    p99 = D.groupby("stid").gust.quantile(0.99)
    s = setters(D, M.time_utc.unique())
    share = s.groupby("stid").size() / M.time_utc.nunique()
    rep = D[D.time_utc.isin(M.time_utc)].groupby("stid").size() / M.time_utc.nunique()
    t = st.assign(p99=st.stid.map(p99), setter_share=st.stid.map(share).fillna(0),
                  reporting_in_misses=st.stid.map(rep).fillna(0),
                  first=st.stid.map(D.groupby("stid").time_utc.min()), last=st.stid.map(D.groupby("stid").time_utc.max()))
    return t.sort_values("setter_share", ascending=False)


def record(sg, t, D, M, out):
    order = t.sort_values("lat").stid.tolist()
    D = D.assign(month=D.time_utc.dt.tz_localize(None).dt.to_period("M"))
    P = D.groupby(["stid", "month"]).gust.quantile(0.99).unstack()
    months = pd.period_range(D.month.min(), D.month.max(), freq="M")
    P = P.reindex(index=order, columns=months)
    mm = M.time_utc.dt.tz_localize(None).dt.to_period("M").value_counts().reindex(months, fill_value=0)
    n = len(order)
    fig, (ax0, ax) = plt.subplots(2, 1, figsize=(17, 2.4 + 0.28 * n), facecolor=SURFACE, sharex=True,
                                  gridspec_kw={"height_ratios": [1.1, 0.28 * n]})
    style(ax0)
    ax0.bar(np.arange(len(months)), mm.values, width=0.9, color="#eb6834", zorder=3)
    ax0.set_ylabel("missed\nstrong h", fontsize=8)
    ax0.set_title("Missed strong hours per month", loc="left", fontsize=10, color=INK)
    im = ax.imshow(P.values, aspect="auto", cmap=SEQ, norm=mcolors.Normalize(20, 70), interpolation="nearest")
    lab = [f"{s} · {t.set_index('stid').loc[s, 'name'][:24]} · {t.set_index('stid').loc[s, 'dist_km']:.0f} km"
           for s in order]
    ax.set_yticks(range(n), lab, fontsize=7)
    yrs = [i for i, m in enumerate(months) if m.month == 1]
    ax.set_xticks(yrs, [str(months[i].year) for i in yrs], fontsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=[ax0, ax], orientation="horizontal", fraction=0.025, pad=0.04, aspect=60)
    cb.set_label("Monthly 99th percentile hourly max gust (mph); blank = not reporting", fontsize=8.5, color=INK2)
    fig.suptitle(f"{sg} gust set: station record (rows south → north; label = distance from gradient midpoint)",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.savefig(os.path.join(out, "station_record.png"), dpi=110, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def pick_events(M):
    chosen = []
    for r in M.sort_values("err").itertuples():
        if all(abs((r.time_utc - c.time_utc).total_seconds()) > 86400 for c in chosen):
            chosen.append(r)
        if len(chosen) == N_EVENTS:
            break
    return chosen


def events(sg, t, D, G, T, M, out):
    ev = pick_events(M)
    info = t.set_index("stid")
    fig = plt.figure(figsize=(18, 13), facecolor=SURFACE)
    outer = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.2)
    for e, r in enumerate(ev):
        t0, t1 = r.time_utc - pd.Timedelta(hours=WINDOW_H), r.time_utc + pd.Timedelta(hours=WINDOW_H)
        inner = outer[e // 3, e % 3].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        ax = fig.add_subplot(inner[0])
        gx = fig.add_subplot(inner[1], sharex=ax)
        style(ax)
        style(gx)
        d = D[(D.time_utc >= t0) & (D.time_utc <= t1)]
        top = setters(D, [r.time_utc]).sort_values("rank").stid.tolist()[:len(SLOTS)]
        for s, g in d.groupby("stid"):
            if s not in top:
                ax.plot(g.time_utc, g.gust, color=GRID, lw=0.8, zorder=2)
        for s, c in zip(top, SLOTS):
            g = d[d.stid == s]
            ax.plot(g.time_utc, g.gust, color=c, lw=1.8, zorder=4,
                    label=f"{s} ({info.loc[s, 'dist_km']:.0f} km, {info.loc[s, 'name'][:16]})")
        w = T[(T.time_utc >= t0) & (T.time_utc <= t1)]
        ax.plot(w.time_utc, w.obs, color=INK, lw=2.2, ls=(0, (1, 1.5)), zorder=5)
        ax.plot(w.time_utc, w.pred_gbt_all, color=INK, lw=1.8, ls=(0, (6, 2)), zorder=5)
        ax.axvline(r.time_utc, color=INK2, lw=0.8, zorder=1)
        n_rep = int((D.time_utc == r.time_utc).sum())
        ax.set_title(f"{r.time_utc:%Y-%m-%d %H:%M}Z · target {r.obs:.0f}, GBT {r.pred_gbt_all:.0f} mph · "
                     f"{n_rep} reporting", loc="left", fontsize=9, color=INK)
        ax.legend(loc="upper left", frameon=False, fontsize=7, labelcolor=INK2, title="setters at worst hour",
                  title_fontsize=7)
        ax.set_ylabel("Gust (mph)", fontsize=8.5)
        ax.tick_params(axis="x", labelbottom=False)
        g = G[(G.time_utc >= t0) & (G.time_utc <= t1)]
        gx.plot(g.time_utc, g.gradient_hpa, color=INK2, lw=1.6)
        gx.axhline(0, color=GRID, lw=1)
        gx.axvline(r.time_utc, color=INK2, lw=0.8)
        gx.set_ylabel(f"{sg[:-1]}\n(hPa)", fontsize=8)
        gx.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
        gx.xaxis.set_major_formatter(mdates.DateFormatter("%d/%H"))
        gx.tick_params(labelsize=7.5)
        gx.set_xlim(t0, t1)
    fig.legend([Line2D([], [], color=GRID, lw=1), Line2D([], [], color=INK, lw=2.2, ls=(0, (1, 1.5))),
                Line2D([], [], color=INK, lw=1.8, ls=(0, (6, 2)))],
               ["other stations", "set target (k-th highest gust)", "GBT forecast"], loc="lower center", ncol=3,
               frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle(f"{sg}: the {N_EVENTS} worst strong-gust misses, ±{WINDOW_H} h", x=0.01, ha="left", fontsize=14,
                 color=INK)
    fig.text(0.01, 0.945, "Coloured: the stations that set the target at the worst hour (km = distance from the "
             "gradient midpoint). x axis day/hour UTC.", fontsize=9, color=INK2)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.91, bottom=0.07)
    fig.savefig(os.path.join(out, "station_events.png"), dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return ev


def terrain(ax, box, z):
    E, lons, lats = dem(*box, z=z)
    E = np.where(E < -100, 0, E)
    shade = mcolors.LightSource(azdeg=315, altdeg=40).hillshade(E, vert_exag=2 if z <= 7 else 0.8, dx=1, dy=1)
    ext = (lons[0], lons[-1], lats[-1], lats[0])
    ax.imshow(E, extent=ext, cmap="Greys", vmin=-1500, vmax=4500, zorder=0, interpolation="bilinear")
    ax.imshow(shade, extent=ext, cmap="gray", alpha=0.35, zorder=1, interpolation="bilinear")
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_aspect(1 / np.cos(np.radians((box[2] + box[3]) / 2)))
    ax.tick_params(labelsize=7.5)


def zoom_for(box_span):
    """Tile zoom keeping the mosaic to a few dozen tiles."""
    return int(np.clip(np.floor(np.log2(360 * 5 / max(box_span, 0.05))), 6, 11))


def station_map(sg, t, meta, ends, out):
    pad = 0.4
    lon0, lon1 = min(t.lon.min(), *(meta.loc[e].lon for e in ends)) - pad, max(t.lon.max(), *(meta.loc[e].lon for e in ends)) + pad
    lat0, lat1 = min(t.lat.min(), *(meta.loc[e].lat for e in ends)) - pad, max(t.lat.max(), *(meta.loc[e].lat for e in ends)) + pad
    region = (lon0, lon1, lat0, lat1)
    top = t.nlargest(1, "setter_share").iloc[0]
    near = t[km(t.lat, t.lon, top.lat, top.lon) < 120]
    cl = (near.lon.min() - 0.15, near.lon.max() + 0.15, near.lat.min() - 0.12, near.lat.max() + 0.12)
    norm = mcolors.Normalize(30, 70)
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1, 1.25]})
    for ax, box, title, full in ((axes[0], region, "All stations", True),
                                 (axes[1], cl, f"Cluster around the main target setter ({top.stid})", False)):
        terrain(ax, box, zoom_for(max(box[1] - box[0], box[3] - box[2])))
        for e in ends:
            r = meta.loc[e]
            if box[0] <= r.lon <= box[1] and box[2] <= r.lat <= box[3]:
                ax.plot(r.lon, r.lat, "s", ms=8, color=INK, mec=SURFACE, zorder=6)
                ax.annotate(e[1:], (r.lon, r.lat), xytext=(6, -10), textcoords="offset points", fontsize=9,
                            color=INK, path_effects=HALO, zorder=7)
        if full:
            ax.plot([cl[0], cl[1], cl[1], cl[0], cl[0]], [cl[2], cl[2], cl[3], cl[3], cl[2]], color=INK, lw=1, zorder=5)
        for r in t.itertuples():
            if not (box[0] <= r.lon <= box[1] and box[2] <= r.lat <= box[3]):
                continue
            size = 40 + 900 * r.setter_share
            ax.scatter(r.lon, r.lat, s=size, c=[SEQ(norm(r.p99))] if np.isfinite(r.p99) else ["#ffffff"],
                       edgecolors=INK if r.setter_share >= 0.1 else SURFACE, linewidths=1.2, zorder=8, alpha=0.95)
            if (not full) or r.setter_share >= 0.1 or r.dist_km > 300:
                ax.annotate(f"{r.stid} {r.setter_share:.0%}" + (f" · {r.dist_km:.0f} km" if full else
                                                                 f"\n{r.elevation_m:,.0f} m · p99 {r.p99:.0f}"),
                            (r.lon, r.lat), xytext=(7, 5), textcoords="offset points", fontsize=7.5, color=INK,
                            path_effects=HALO, zorder=9)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQ), ax=axes, orientation="horizontal", fraction=0.035,
                      pad=0.07, aspect=50)
    cb.set_label("Station 99th percentile hourly max gust (mph)", fontsize=9, color=INK2)
    fig.suptitle(f"{sg} gust stations: where are the target setters?", x=0.01, ha="left", fontsize=14, color=INK)
    fig.text(0.01, 0.93, "Dot size and label % = share of missed strong hours in which the station set the target "
             "(ranked within the top k); dark ring = ≥ 10%. Left: km from gradient midpoint (stations ≥ 10% or > 300 km "
             "labelled).", fontsize=9, color=INK2)
    fig.savefig(os.path.join(out, "station_map.png"), dpi=115, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signed_gradient")
    sg = ap.parse_args().signed_gradient
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    slug = sg[:-1].lower().replace("-", "_") + ("_pos" if sg.endswith("+") else "_neg")
    out = os.path.normpath(os.path.join(HERE, "..", "images", "case_studies", slug))
    tout = os.path.join(MDIR, "case_studies")
    os.makedirs(out, exist_ok=True)
    os.makedirs(tout, exist_ok=True)
    st, meta, D, G, T, M, ends = load(sg)
    t = table(st, D, M)
    t.to_csv(os.path.join(tout, f"{slug}_stations.csv"), index=False, float_format="%.3f")
    print(f"{sg}: {len(t)} stations, {M.time_utc.nunique()} missed strong hours")
    print(t[["stid", "name", "dist_km", "rho", "n", "p99", "setter_share", "reporting_in_misses", "first", "last"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    record(sg, t, D, M, out)
    ev = events(sg, t, D, G, T, M, out)
    station_map(sg, t, meta, ends, out)
    print("events:", [f"{r.time_utc:%Y-%m-%d %H}Z err {r.err:.0f}" for r in ev])
    print(f"Wrote figures to {out}")


if __name__ == "__main__":
    main()
