#!/usr/bin/env python3
"""Sanity-check the bulk data: wind speed/gust distributions by network, and the
pressure-gradient distributions per pair.

Steps:
  hist   Stream every data/v2/obs_raw file and accumulate wind histograms per
         network x year (WWG networks excluded).
         -> data/v2/derived/qc/wind_hist.npz, wind_stats_by_network.csv
  plots  Figures -> images/distributions/
           wind_overview.png          top networks overlaid (sustained, gust)
           wind_by_network.png        one panel per network
           gradient_hist.png          one panel per gradient pair
           gradient_monthly.png       monthly p1/p50/p99 per pair (shifts over time)
         -> data/v2/derived/qc/gradient_stats.csv

Usage:
  python explore_distributions.py all
  python explore_distributions.py plots
"""
import argparse
import glob
import os
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data", "v2"))
QC = os.path.join(DATA, "derived", "qc")
IMAGES = os.path.normpath(os.path.join(HERE, "..", "images", "distributions"))
EXCLUDE_NETWORK_PREFIXES = ("WWG",)

BIN = 0.1                      # m/s, storage resolution
NBIN = 1000                    # 0 .. 100 m/s; last bin also gets everything >= 100
VARS = ["wind_speed_mps", "wind_gust_mps"]
YEARS = list(range(2015, 2027))

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
OTHER = "#8a8a85"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"

_net_of = None


def _init(net_of):
    global _net_of
    _net_of = net_of


def stations():
    s = pd.read_parquet(os.path.join(DATA, "stations.parquet")).drop_duplicates("stid")
    s["network_shortname"] = s.network_shortname.fillna("UNKNOWN")
    return s[~s.network_shortname.str.startswith(EXCLUDE_NETWORK_PREFIXES)]


def hist_file(path):
    """-> (year, {var: (nnet, NBIN) counts}, {var: (nnet,) negative counts}, nan counts)"""
    year = int(path.split(os.sep)[-3])
    t = pq.read_table(path, columns=["stid"] + VARS).to_pandas()
    code = t.stid.map(_net_of)
    keep = code.notna().to_numpy()
    code = code.to_numpy()[keep].astype(np.int64)
    nnet = max(_net_of.values()) + 1
    out, neg = {}, {}
    for v in VARS:
        x = t[v].to_numpy()[keep].astype(np.float64)
        ok = np.isfinite(x)
        neg[v] = np.bincount(code[ok & (x < 0)], minlength=nnet)
        m = ok & (x >= 0)
        b = np.minimum((x[m] / BIN + 1e-6).astype(np.int64), NBIN - 1)
        out[v] = np.bincount(code[m] * NBIN + b, minlength=nnet * NBIN).reshape(nnet, NBIN)
    return year, out, neg


def step_hist(workers):
    st = stations()
    nets = sorted(st.network_shortname.unique())
    idx = {n: i for i, n in enumerate(nets)}
    net_of = dict(zip(st.stid, st.network_shortname.map(idx)))
    files = sorted(glob.glob(os.path.join(DATA, "obs_raw", "*", "*", "*.parquet")))
    print(f"{len(files)} files, {len(nets)} networks, {len(net_of)} stations (WWG excluded)")
    H = {v: np.zeros((len(nets), len(YEARS), NBIN), np.int64) for v in VARS}
    N = {v: np.zeros(len(nets), np.int64) for v in VARS}
    with Pool(workers, initializer=_init, initargs=(net_of,)) as p:
        for i, (year, out, neg) in enumerate(p.imap_unordered(hist_file, files, chunksize=2)):
            for v in VARS:
                H[v][:, YEARS.index(year)] += out[v]
                N[v] += neg[v]
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(files)}")
    os.makedirs(QC, exist_ok=True)
    np.savez_compressed(os.path.join(QC, "wind_hist.npz"), networks=np.array(nets), years=np.array(YEARS),
                        bin=BIN, **{f"{v}": H[v] for v in VARS}, **{f"{v}_neg": N[v] for v in VARS})

    centers = (np.arange(NBIN) + 0.5) * BIN
    rows = []
    for i, n in enumerate(nets):
        r = {"network": n, "stations": int((st.network_shortname == n).sum())}
        for v, short in zip(VARS, ["spd", "gust"]):
            h = H[v][i].sum(0)
            tot = h.sum()
            r[f"{short}_n"] = int(tot)
            r[f"{short}_neg"] = int(N[v][i])
            if tot == 0:
                continue
            c = np.cumsum(h) / tot
            for q in (0.5, 0.99, 0.9999):
                r[f"{short}_p{q * 100:g}"] = round(float(centers[np.searchsorted(c, q)]), 1)
            r[f"{short}_frac_zero"] = round(float(h[0] / tot), 3)
            r[f"{short}_max"] = round(float(centers[np.nonzero(h)[0].max()]), 1)
            r[f"{short}_n_gt50"] = int(h[int(50 / BIN):].sum())
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("spd_n", ascending=False)
    df.to_csv(os.path.join(QC, "wind_stats_by_network.csv"), index=False)
    print(df.to_string(index=False))


# ----------------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------------

def style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def rebin(h, k):
    """Sum consecutive groups of k fine bins (0.1 m/s -> k*0.1 m/s)."""
    n = h.shape[-1] // k * k
    return h[..., :n].reshape(*h.shape[:-1], -1, k).sum(-1)


def density_line(ax, h, k, xmax, **kw):
    hb = rebin(h, k).astype(float)
    tot = hb.sum()
    if tot == 0:
        return
    edges = np.arange(hb.size + 1) * BIN * k
    y = np.where(hb > 0, hb / tot / (BIN * k), np.nan)
    m = edges[:-1] < xmax
    ax.stairs(y[m], edges[:m.sum() + 1], baseline=None, **kw)


def plot_wind():
    z = np.load(os.path.join(QC, "wind_hist.npz"))
    nets = list(z["networks"])
    H = {v: z[v].sum(1) for v in VARS}          # (nnet, NBIN), summed over years
    order = np.argsort(-H[VARS[0]].sum(1))
    top = [i for i in order[:7]]
    rest = [i for i in order[7:]]
    labels = {"wind_speed_mps": "Sustained wind", "wind_gust_mps": "Gust"}

    # 1. overview
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, v in zip(axes, VARS):
        style(ax)
        for c, i in zip(SERIES, top):
            density_line(ax, H[v][i], 5, 45, color=c, lw=2, label=f"{nets[i]} ({H[v][i].sum() / 1e6:.0f}M)")
        o = H[v][rest].sum(0)
        density_line(ax, o, 5, 45, color=OTHER, lw=2, ls="--", label=f"Other {len(rest)} networks ({o.sum() / 1e6:.0f}M)")
        ax.set_yscale("log")
        ax.set_xlim(0, 45)
        ax.set_title(labels[v], color=INK, fontsize=11, loc="left")
        ax.set_xlabel("m/s (0.5 m/s bins)", color=MUTED, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    axes[0].set_ylabel("Probability density (log)", color=MUTED, fontsize=9)
    fig.suptitle("Wind distribution by network, 2015-2026, all hours, WWG excluded", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES, "wind_overview.png"), dpi=130)
    plt.close(fig)

    # 2. small multiples, every network with data; fine (0.1 m/s) bins to expose quantisation/units
    show = [i for i in order if H[VARS[0]][i].sum() + H[VARS[1]][i].sum() > 0]
    ncol = 6
    nrow = int(np.ceil(len(show) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.3 * nrow), sharex=True, sharey=True)
    for ax in axes.flat[len(show):]:
        ax.set_visible(False)
    for ax, i in zip(axes.flat, show):
        style(ax)
        density_line(ax, H[VARS[0]][i], 2, 50, color=SERIES[0], lw=1.2, label="Sustained")
        density_line(ax, H[VARS[1]][i], 2, 50, color=SERIES[1], lw=1.2, label="Gust")
        ax.set_yscale("log")
        ax.set_xlim(0, 50)
        ax.set_ylim(1e-8, 5)
        ns, ng = H[VARS[0]][i].sum(), H[VARS[1]][i].sum()
        ax.set_title(f"{nets[i]}  n={ns / 1e6:.1f}M / {ng / 1e6:.1f}M", fontsize=8.5, color=INK, loc="left")
    axes.flat[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Wind distribution per network (0.2 m/s bins, log density); n = sustained / gust obs",
                 color=INK, x=0.01, ha="left")
    fig.supxlabel("m/s", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES, "wind_by_network.png"), dpi=110)
    plt.close(fig)


def plot_gradients():
    g = pd.read_parquet(os.path.join(DATA, "derived", "gradients", "gradient_timeseries.parquet"))
    pairs = list(dict.fromkeys(g.gradient))
    rows = []
    for p, d in g.groupby("gradient", sort=False):
        x = d.gradient_hpa
        rows.append({"pair": p, "n": len(x), "first": d.time_utc.min().date(), "last": d.time_utc.max().date(),
                     "min": x.min(), "p0.1": x.quantile(.001), "p1": x.quantile(.01), "p50": x.median(),
                     "p99": x.quantile(.99), "p99.9": x.quantile(.999), "max": x.max(),
                     "mean_offset_min": round(d.station2_offset_min.abs().mean(), 1)})
    stats = pd.DataFrame(rows).round(2)
    stats.to_csv(os.path.join(QC, "gradient_stats.csv"), index=False)
    print(stats.to_string(index=False))

    ncol = 7
    nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.5 * nrow), sharey=True)
    edges = np.arange(-25, 25.01, 0.5)
    for ax, p in zip(axes.flat, pairs):
        style(ax)
        x = g.loc[g.gradient == p, "gradient_hpa"].to_numpy()
        h, _ = np.histogram(x, edges)
        y = np.where(h > 0, h / h.sum(), np.nan)
        ax.stairs(np.where(edges[:-1] >= 0, y, np.nan), edges, color=SERIES[0], lw=1.4, baseline=None)
        ax.stairs(np.where(edges[:-1] < 0, y, np.nan), edges, color=SERIES[1], lw=1.4, baseline=None)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yscale("log")
        lo, hi = np.floor(x.min()) - 1, np.ceil(x.max()) + 1
        ax.set_xlim(lo, hi)
        ax.set_title(f"{p}  n={len(x) / 1e3:.0f}k", fontsize=8.5, color=INK, loc="left")
    for ax in axes.flat[len(pairs):]:
        ax.set_visible(False)
    fig.suptitle("Pressure gradient distributions (SLP station1 - station2, hPa, 0.5 hPa bins, log fraction); "
                 "blue = positive, orange = negative", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES, "gradient_hist.png"), dpi=110)
    plt.close(fig)

    # monthly percentiles -> station moves, sensor changes, step shifts
    g["month"] = g.time_utc.dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    q = g.groupby(["gradient", "month"]).gradient_hpa.quantile([.01, .5, .99]).unstack()
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.3 * nrow), sharex=True)
    for ax, p in zip(axes.flat, pairs):
        style(ax)
        d = q.loc[p]
        ax.fill_between(d.index, d[.01], d[.99], color=SERIES[0], alpha=0.18, lw=0)
        ax.plot(d.index, d[.5], color=SERIES[0], lw=1.2)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(p, fontsize=8.5, color=INK, loc="left")
        ax.tick_params(axis="x", labelrotation=90)
    for ax in axes.flat[len(pairs):]:
        ax.set_visible(False)
    fig.suptitle("Monthly gradient median (line) and 1st-99th percentile (band), hPa", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES, "gradient_monthly.png"), dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["hist", "plots", "all"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(IMAGES, exist_ok=True)
    if a.step in ("hist", "all"):
        step_hist(a.workers)
    if a.step in ("plots", "all"):
        plot_wind()
        plot_gradients()


if __name__ == "__main__":
    main()
