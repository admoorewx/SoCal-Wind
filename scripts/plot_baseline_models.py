#!/usr/bin/env python3
"""Figures for the step-3 baseline models (fit_baseline_models.py), 2015-2025 cross-validation.

One figure per signed gradient and variable (sustained / gust), four panels:
  A  Event frequency vs |gradient|: observed rate in 0.5 hPa bins (dots) and the |gradient|-only
     logistic curve fitted on all of 2015-2025 (line), one colour per threshold.
  B  Reliability of the full model (|grad| + hour + day of year), out-of-fold: mean forecast
     probability vs observed frequency in probability bins, against the 1:1 line.
  C  Linear model, out-of-fold: predicted vs observed target (KDE shading, 1:1 line, binned median).
  D  Skill by predictor set: Brier skill score per threshold and linear R2.
Plus an overview heatmap of full-model skill for every gradient.

-> images/models/baseline/<signed_gradient>_<variable>.png
-> images/models/baseline/overview_skill.png

Usage:
  python plot_baseline_models.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from plot_station_set_scatter import kde

HERE = os.path.dirname(os.path.abspath(__file__))
MDIR = os.path.normpath(os.path.join(HERE, "..", "data", "v2", "derived", "models"))
OUT = os.path.normpath(os.path.join(HERE, "..", "images", "models", "baseline"))
FULL = "grad+hour+doy"
SETS = ["grad", "grad+hour", FULL]
SET_LABEL = {"grad": "|grad|", "grad+hour": "+ hour", FULL: "+ hour + day of year"}
SET_COLOR = {"grad": "#b9b8b0", "grad+hour": "#7d7c76", FULL: "#0b0b0b"}
THR_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, direct-labelled
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
MIN_BIN = 50
HALO = [pe.Stroke(linewidth=3, foreground=SURFACE), pe.Normal()]


def style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(length=0)


def end_label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8,
                color=INK, path_effects=HALO, zorder=6)


def panel_frequency(ax, d, coefs, thresholds):
    style(ax)
    x = d.abs_gradient_hpa.to_numpy()
    xmax = np.nanpercentile(x, 99.9)
    edges = np.arange(0, xmax + 0.5, 0.5)
    idx = np.digitize(x, edges) - 1
    grid = np.linspace(0, xmax, 200)
    ends = []
    for T, c in zip(thresholds, THR_COLORS):
        y = d[f"ev_{T}"].to_numpy(float)
        mids, rates = [], []
        for i in range(len(edges) - 1):
            s = idx == i
            if s.sum() >= MIN_BIN:
                mids.append(edges[i] + 0.25)
                rates.append(y[s].mean())
        ax.scatter(mids, rates, s=26, color=c, edgecolor=SURFACE, linewidth=1, zorder=4)
        b = coefs[(coefs.threshold_mph == T) & (coefs.model == "grad")].iloc[0]
        p = 1 / (1 + np.exp(-(b.intercept + b.abs_gradient_hpa * grid)))
        ax.plot(grid, p, color=c, lw=2, zorder=3)
        ends.append([p[-1], f"≥{T} mph"])
    # spread end labels so they never overlap (min gap 0.05 in probability units)
    ends.sort(key=lambda e: -e[0])
    for i in range(1, len(ends)):
        ends[i][0] = min(ends[i][0], ends[i - 1][0] - 0.05)
    for y, t in ends:
        ax.annotate(t, (grid[-1], y), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8, color=INK, path_effects=HALO, zorder=6)
    ax.set_xlim(0, xmax * 1.15)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("|Pressure gradient| (hPa)")
    ax.set_ylabel("Event frequency")
    ax.set_title("A  Event frequency vs |gradient|", loc="left", fontsize=11, color=INK)
    ax.text(0.02, 0.97, "dots: observed (0.5 hPa bins)\nline: |grad|-only logistic fit", transform=ax.transAxes,
            va="top", fontsize=8, color=INK2)


def panel_reliability(ax, oof, thresholds, cvl):
    style(ax)
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=(0, (3, 2)), zorder=2)
    edges = np.array([0, .02, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001])
    lines = []
    for T, c in zip(thresholds, THR_COLORS):
        o = oof[oof.threshold_mph == T]
        p, y = o[f"pred_{FULL}"].to_numpy(), o.obs.to_numpy()
        idx = np.digitize(p, edges) - 1
        mp, fr = [], []
        for i in range(len(edges) - 1):
            s = idx == i
            if s.sum() >= MIN_BIN:
                mp.append(p[s].mean())
                fr.append(y[s].mean())
        ax.plot(mp, fr, color=c, lw=2, marker="o", ms=5, mec=SURFACE, zorder=4)
        m = cvl[(cvl.threshold_mph == T) & (cvl.model == FULL)].iloc[0]
        lines.append((c, f"≥{T} mph   BSS {m.bss:.2f}   AUC {m.auc:.2f}   ({int(m.event_hours):,} events)"))
    for i, (c, t) in enumerate(lines):
        ax.text(0.03, 0.97 - 0.065 * i, "●", transform=ax.transAxes, va="top", fontsize=9, color=c)
        ax.text(0.07, 0.97 - 0.065 * i, t, transform=ax.transAxes, va="top", fontsize=8, color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Forecast probability (out of fold)")
    ax.set_ylabel("Observed frequency")
    ax.set_title("B  Reliability, full model", loc="left", fontsize=11, color=INK)


def panel_linear(ax, o, lin):
    style(ax)
    p, y = o[f"pred_{FULL}"].to_numpy(), o.obs.to_numpy()
    hi = max(np.nanpercentile(y, 99.9), np.nanpercentile(p, 99.9)) * 1.05
    xe, ye, dens = kde(p, y, hi, hi)
    ax.pcolormesh(xe, ye, dens, cmap="Blues", norm=LogNorm(1e-3, 1), shading="flat", rasterized=True, zorder=1)
    ax.plot([0, hi], [0, hi], color=INK2, lw=1, ls=(0, (3, 2)), zorder=3)
    edges = np.arange(0, hi + 2, 2)
    idx = np.digitize(p, edges) - 1
    mids, med = [], []
    for i in range(len(edges) - 1):
        s = idx == i
        if s.sum() >= MIN_BIN:
            mids.append(edges[i] + 1)
            med.append(np.median(y[s]))
    ax.plot(mids, med, color=INK, lw=2, zorder=4,
            path_effects=[pe.Stroke(linewidth=4.5, foreground=SURFACE, alpha=0.85), pe.Normal()])
    end_label(ax, mids[-1], med[-1], "Observed median", INK)
    m = lin[lin.model == FULL].iloc[0]
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted (mph, out of fold)")
    ax.set_ylabel("Observed (mph)")
    ax.set_title(f"C  Linear, full model: R² {m.r2:.2f}, RMSE {m.rmse:.1f} mph", loc="left", fontsize=11, color=INK)


def panel_skill(ax, cvl, lin, thresholds):
    style(ax)
    cats = [f"≥{T} mph\nBSS" for T in thresholds] + ["Linear\nR²"]
    for j, sname in enumerate(SETS):
        vals = [cvl[(cvl.threshold_mph == T) & (cvl.model == sname)].bss.iloc[0] for T in thresholds]
        vals.append(lin[lin.model == sname].r2.iloc[0])
        xs = np.arange(len(cats)) + (j - 1) * 0.22
        ax.scatter(xs, vals, s=46, color=SET_COLOR[sname], edgecolor=SURFACE, linewidth=1.2, zorder=4,
                   label=SET_LABEL[sname])
    ax.axhline(0, color=INK2, lw=0.8, zorder=2)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylim(min(-0.05, ax.get_ylim()[0]), 1)
    ax.set_ylabel("Skill vs climatology (out of fold)")
    ax.set_title("D  Skill by predictor set", loc="left", fontsize=11, color=INK)
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=INK2)


def overview(L, R):
    cols = [("sustained", 15), ("sustained", 20), ("sustained", 30), ("sustained", "R²"),
            ("gust", 30), ("gust", 40), ("gust", 50), ("gust", 60), ("gust", "R²")]
    sgs = sorted(set(L.signed_gradient) | set(R.signed_gradient))
    M = np.full((len(sgs), len(cols)), np.nan)
    for i, sg in enumerate(sgs):
        for j, (v, t) in enumerate(cols):
            if t == "R²":
                x = R[(R.signed_gradient == sg) & (R.variable == v) & (R.model == FULL)].r2
            else:
                x = L[(L.signed_gradient == sg) & (L.variable == v) & (L.threshold_mph == t) & (L.model == FULL)].bss
            if len(x):
                M[i, j] = x.iloc[0]
    fig, ax = plt.subplots(figsize=(10, 12), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(np.clip(M, 0, None), cmap="Blues", vmin=0, vmax=0.8, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.text(j, i, "–", ha="center", va="center", fontsize=8, color=INK2)
            else:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color=SURFACE if M[i, j] > 0.45 else INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{'Sust.' if v == 'sustained' else 'Gust'}\n{'≥' + str(t) + ' mph' if t != 'R²' else 'linear R²'}"
                        for v, t in cols], fontsize=8)
    ax.set_yticks(range(len(sgs)))
    ax.set_yticklabels(sgs, fontsize=8)
    ax.axvline(3.5, color=SURFACE, lw=4)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("Skill vs climatology (BSS or R²), clipped at 0", color=INK2, fontsize=9)
    fig.suptitle("Baseline model skill", x=0.02, ha="left", fontsize=13, color=INK)
    fig.text(0.02, 0.955, "Full model (|grad| + hour + day of year), leave-one-year-out CV 2015–2025. "
             "– = not modelled (< 100 event hours or no station set)", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(OUT, "overview_skill.png")
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    bdir = os.path.join(MDIR, "baseline")
    L = pd.read_csv(os.path.join(bdir, "cv_logistic.csv"))
    R = pd.read_csv(os.path.join(bdir, "cv_linear.csv"))
    C = pd.read_csv(os.path.join(bdir, "coefficients.csv"))
    O = pd.read_parquet(os.path.join(bdir, "oof_predictions.parquet"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    os.makedirs(OUT, exist_ok=True)
    overview(L, R)
    for var in ("sustained", "gust"):
        T = pd.read_parquet(os.path.join(MDIR, "targets", f"{var}_train.parquet"))
        T = T[T.qualifies & T.target_mph.notna()]
        for sg, d in T.groupby("signed_gradient"):
            cvl = L[(L.signed_gradient == sg) & (L.variable == var)]
            lin = R[(R.signed_gradient == sg) & (R.variable == var)]
            thresholds = sorted(cvl.threshold_mph.unique())
            o = O[(O.signed_gradient == sg) & (O.variable == var)]
            coefs = C[(C.signed_gradient == sg) & (C.variable == var) & (C.kind == "logistic")]
            fig, axes = plt.subplots(2, 2, figsize=(13, 11), facecolor=SURFACE)
            if thresholds:
                panel_frequency(axes[0, 0], d, coefs, thresholds)
                panel_reliability(axes[0, 1], o[o.kind == "logistic"], thresholds, cvl)
            else:
                for ax in axes[0]:
                    style(ax)
                    ax.text(0.5, 0.5, "No threshold with ≥ 100 event hours", transform=ax.transAxes,
                            ha="center", va="center", color=INK2)
            panel_linear(axes[1, 0], o[o.kind == "linear"], lin)
            panel_skill(axes[1, 1], cvl, lin, thresholds)
            n = int(lin.n_hours.iloc[0])
            what = "sustained 75th percentile" if var == "sustained" else \
                "gust: k-th highest, k = max(2, ⌈0.1·n⌉)"
            fig.suptitle(f"{sg} {var} models", x=0.01, ha="left", fontsize=14, color=INK)
            fig.text(0.01, 0.948, f"Target: {what}. {n:,} qualifying hours, 2015–2025, "
                     f"leave-one-year-out CV", fontsize=10, color=INK2)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            path = os.path.join(OUT, f"{sg}_{var}.png")
            fig.savefig(path, dpi=120, facecolor=SURFACE)
            plt.close(fig)
            print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
