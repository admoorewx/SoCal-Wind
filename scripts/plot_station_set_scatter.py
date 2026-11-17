#!/usr/bin/env python3
"""Density (KDE) of |gradient| vs wind at each signed gradient's correlated stations (model station sets).

One figure per signed gradient, panels for sustained wind and gust. Every point is one station-hour
from the station set (derived/models/station_sets.csv), training period only (before 2026-01-01), at
gradient times of the matching sign. Gust panels use reported gusts only (as in the correlations).
Panel titles give the mean Spearman rho of the set's stations (from station_sets.csv). Shading is a
Gaussian KDE (2-D histogram convolved with a Gaussian kernel, Scott bandwidth, floored at the data's
quantisation of 0.1 hPa / 1 mph), on a log colour scale so the sparse strong-wind tail stays visible.
Lines show the 10th, 25th, 50th, 75th and 90th percentiles of wind in |gradient| bins.

-> images/models/station_set_scatter/<signed_gradient>.png

Usage:
  python plot_station_set_scatter.py
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import pandas as pd

import gradient_pipeline as gp

HERE = os.path.dirname(os.path.abspath(__file__))
DER = os.path.normpath(os.path.join(HERE, "..", "data", "v2", "derived"))
OUT = os.path.normpath(os.path.join(HERE, "..", "images", "models", "station_set_scatter"))
CUT = pd.Timestamp("2026-01-01", tz="UTC")
MPS2MPH = 2.236936
COLS = {"sustained": "sustained_max_mps", "gust": "gust_max_mps"}
LABEL = {"sustained": "Sustained wind (mph)", "gust": "Gust (mph)"}
MIN_BIN = 100
NGRID = 220
DMIN = 1e-3  # shade 3 decades of relative density
QUANTUM = (0.1, 1.0)  # data resolution: hPa, mph
PCTS = [(10, 1.0, (0, (3, 2))), (25, 1.4, "-"), (50, 2.6, "-"), (75, 1.4, "-"), (90, 1.0, (0, (3, 2)))]

SURFACE, INK, INK2, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#2a78d6"


def binned(x, y, width):
    """Percentiles of y in |gradient| bins with at least MIN_BIN points."""
    edges = np.arange(0, np.nanmax(x) + width, width)
    idx = np.digitize(x, edges) - 1
    mid, q = [], []
    for i in range(len(edges) - 1):
        v = y[idx == i]
        if len(v) >= MIN_BIN:
            mid.append(edges[i] + width / 2)
            q.append(np.percentile(v, [p for p, _, _ in PCTS]))
    return np.array(mid), np.array(q)


def gauss_smooth(h, sx, sy):
    """Separable Gaussian convolution of a 2-D histogram; sigmas in bins."""
    for axis, sig in ((0, sx), (1, sy)):
        r = max(1, int(np.ceil(4 * sig)))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
        k /= k.sum()
        h = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), axis, h)
    return h


def kde(x, y, xmax, ymax):
    n = len(x)
    f = n ** (-1 / 6)  # Scott's factor in 2-D
    bx = max(np.std(x) * f, QUANTUM[0])
    by = max(np.std(y) * f, QUANTUM[1])
    xe, ye = np.linspace(0, xmax, NGRID + 1), np.linspace(0, ymax, NGRID + 1)
    h, _, _ = np.histogram2d(x, y, bins=(xe, ye))
    d = gauss_smooth(h, bx / (xe[1] - xe[0]), by / (ye[1] - ye[0]))
    d /= d.max()
    return xe, ye, np.ma.masked_less(d.T, DMIN)


def panel(ax, x, y, var, info):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    if info is None:
        ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
        ax.text(0.5, 0.5, "Not modelled (fewer than 10 stations)", transform=ax.transAxes,
                ha="center", va="center", color=INK2)
        ax.set_title(f"{var.capitalize()}: no station set", loc="left", fontsize=11, color=INK)
        return None
    xmax = np.nanpercentile(x, 99.9) * 1.05
    ymax = max(np.nanpercentile(y, 99.95) * 1.05, 10)
    xe, ye, d = kde(x, y, xmax, ymax)
    mesh = ax.pcolormesh(xe, ye, d, cmap="Blues", norm=LogNorm(DMIN, 1), shading="flat",
                         rasterized=True, zorder=1)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.6, zorder=2)
    width = 0.5 if np.nanmax(x) > 6 else 0.25
    keep = x <= xmax  # lines cover the same range as the shading
    mid, q = binned(x[keep], y[keep], width)
    halo = [pe.Stroke(linewidth=4, foreground=SURFACE, alpha=0.85), pe.Normal()]
    for j, (p, lw, ls) in enumerate(PCTS):
        ax.plot(mid, q[:, j], color=INK, lw=lw, ls=ls, zorder=4,
                path_effects=[pe.Stroke(linewidth=lw + 2.5, foreground=SURFACE, alpha=0.85), pe.Normal()])
        ax.annotate("Median" if p == 50 else f"P{p}", (mid[-1], q[-1, j]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK,
                    path_effects=halo, zorder=5)
    ax.set_xlim(0, max(xmax, mid[-1]) * 1.1)  # room for the line labels
    ax.set_ylim(0, ymax)
    ax.set_xlabel("|Pressure gradient| (hPa)")
    ax.set_ylabel(LABEL[var])
    ax.set_title(f"{var.capitalize()}: mean ρ = {info['mean_rho']:.2f}  ({info['n']} stations)",
                 loc="left", fontsize=11, color=INK)
    ax.text(0.99, 0.98, f"{len(x):,} station-hours", transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=INK2, path_effects=halo, zorder=5)
    return mesh


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    sets = pd.read_csv(os.path.join(DER, "models", "station_sets.csv"))
    info = (sets.groupby(["signed_gradient", "variable"])
            .agg(mean_rho=("rho", "mean"), n=("stid", "size"), stids=("stid", list)).to_dict("index"))
    pairs = gp.load_pairs(gp.DEFAULT_GRADIENTS)
    ts = pd.read_parquet(os.path.join(DER, "gradients", "gradient_timeseries.parquet"),
                         columns=["gradient", "time_utc", "gradient_hpa"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    os.makedirs(OUT, exist_ok=True)

    for s1 in sorted({p[0] for p in pairs.values()}):
        sgs = sorted(sg for sg in sets.signed_gradient.unique() if pairs[sg[:-1]][0] == s1)
        if not sgs:
            continue
        need = sorted(set(sets[sets.signed_gradient.isin(sgs)].stid))
        files = sorted(f for f in glob.glob(os.path.join(DER, "gradients", "matched", s1, "*.parquet"))
                       if os.path.basename(f) < "2026")
        m = pd.concat([pd.read_parquet(f, columns=["stid", "time_utc", *COLS.values()],
                                       filters=[("stid", "in", need)]) for f in files])
        m["stid"] = m.stid.astype(str)
        for sg in sgs:
            g = ts[ts.gradient == sg[:-1]][["time_utc", "gradient_hpa"]]
            g = g[(g.gradient_hpa > 0) if sg.endswith("+") else (g.gradient_hpa < 0)]
            g = g[g.time_utc < CUT]
            d = m.merge(g, on="time_utc")
            a, b = pairs[sg[:-1]]
            fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), facecolor=SURFACE)
            mesh = None
            for ax, var in zip(axes, ("sustained", "gust")):
                inf = info.get((sg, var))
                if inf is None:
                    panel(ax, None, None, var, None)
                    continue
                dv = d[d.stid.isin(inf["stids"])].dropna(subset=[COLS[var]])
                mesh = panel(ax, np.abs(dv.gradient_hpa.to_numpy()), dv[COLS[var]].to_numpy() * MPS2MPH,
                             var, inf) or mesh
            sign = ">" if sg.endswith("+") else "<"
            fig.suptitle(f"{sg} station set", x=0.01, ha="left", fontsize=13, color=INK)
            fig.text(0.01, 0.905, f"|{a} − {b}| SLP gradient when {a} − {b} {sign} 0, vs wind at the "
                     f"correlated stations, 2015–2025", fontsize=10, color=INK2)
            fig.tight_layout(rect=(0, 0, 0.94, 0.89))
            if mesh is not None:
                cax = fig.add_axes((0.95, 0.14, 0.012, 0.62))
                cb = fig.colorbar(mesh, cax=cax)
                cb.outline.set_visible(False)
                cb.ax.tick_params(length=0, labelsize=8)
                cb.set_label("Relative density (log)", color=INK2, fontsize=9)
            path = os.path.join(OUT, f"{sg}.png")
            fig.savefig(path, dpi=130, facecolor=SURFACE)
            plt.close(fig)
            print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
