#!/usr/bin/env python3
"""Figures for the step-4b add-on models (fit_gradient_addon.py) vs the step-3 baseline.

Baseline = |grad| + hour + day of year; add-on = baseline + up to 3 other pairs' signed gradients
(chosen inside each CV fold). Both are out-of-fold, leave-one-year-out 2015-2025, identical hours.

One figure per signed gradient and variable:
  top row   reliability per threshold (small multiples): baseline (grey) vs add-on (colour), with the
            top-end ratio (observed rate / forecast in the top 2% of forecasts) for both
  bottom    linear add-on: predicted vs observed (KDE), and a skill comparison (BSS per threshold,
            linear R2) with the pairs chosen in the final fit
Plus an overview heatmap of the skill change (add-on minus baseline).

-> images/models/gradient_addon/<signed_gradient>_<variable>.png
-> images/models/gradient_addon/overview_skill_change.png

Usage:
  python plot_gradient_addon.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.gridspec import GridSpec

from plot_baseline_models import GRID, HALO, INK, INK2, MIN_BIN, SURFACE, THR_COLORS, style
from plot_station_set_scatter import kde

HERE = os.path.dirname(os.path.abspath(__file__))
ADIR = os.path.normpath(os.path.join(HERE, "..", "data", "v2", "derived", "models", "gradient_addon"))
OUT = os.path.normpath(os.path.join(HERE, "..", "images", "models", "gradient_addon"))
BASE_C = "#9a9992"
EDGES = np.array([0, .02, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001])


def reliability(p, y):
    idx = np.digitize(p, EDGES) - 1
    mp, fr = [], []
    for i in range(len(EDGES) - 1):
        s = idx == i
        if s.sum() >= MIN_BIN:
            mp.append(p[s].mean())
            fr.append(y[s].mean())
    return mp, fr


def panel_rel(ax, o, row, color, T):
    style(ax)
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=(0, (3, 2)), zorder=2)
    y = o.obs.to_numpy()
    for col, c, lw, lab in (("pred_base", BASE_C, 1.6, "baseline"), ("pred_addon", color, 2.2, "add-on")):
        mp, fr = reliability(o[col].to_numpy(), y)
        ax.plot(mp, fr, color=c, lw=lw, marker="o", ms=4, mec=SURFACE, zorder=4 if lab == "add-on" else 3,
                label=lab)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"≥{T} mph   top ratio {row.top_ratio_base:.2f} → {row.top_ratio_addon:.2f}",
                 loc="left", fontsize=10, color=INK)
    ax.text(0.03, 0.97, f"BSS {row.bss_base:.2f} → {row.bss_addon:.2f}\nAUC {row.auc_base:.2f} → "
            f"{row.auc_addon:.2f}\n{int(row.event_hours):,} events", transform=ax.transAxes, va="top",
            fontsize=8, color=INK, path_effects=HALO)
    ax.legend(loc="lower right", frameon=False, fontsize=7, labelcolor=INK2)
    ax.set_xlabel("Forecast probability", fontsize=9)


def panel_linear(ax, o, row):
    style(ax)
    p, y = o.pred_addon.to_numpy(), o.obs.to_numpy()
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
    ax.annotate("Observed median", (mids[-1], med[-1]), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8, color=INK, path_effects=HALO)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted (mph, out of fold)")
    ax.set_ylabel("Observed (mph)")
    ax.set_title(f"Linear add-on: R² {row.r2_base:.2f} → {row.r2_addon:.2f}, RMSE {row.rmse_base:.1f} → "
                 f"{row.rmse_addon:.1f} mph", loc="left", fontsize=11, color=INK)


def panel_skill(ax, lrows, lin, chosen):
    style(ax)
    cats, base, add = [], [], []
    for r in lrows.itertuples():
        cats.append(f"≥{r.threshold_mph} mph\nBSS")
        base.append(r.bss_base)
        add.append(r.bss_addon)
    cats.append("Linear\nR²")
    base.append(lin.r2_base)
    add.append(lin.r2_addon)
    x = np.arange(len(cats))
    for i in range(len(cats)):
        ax.plot([x[i], x[i]], [base[i], add[i]], color=GRID, lw=3, zorder=2)
    ax.scatter(x, base, s=50, color=BASE_C, edgecolor=SURFACE, linewidth=1.2, zorder=4, label="baseline")
    ax.scatter(x, add, s=50, color=INK, edgecolor=SURFACE, linewidth=1.2, zorder=5, label="add-on")
    for i in range(len(cats)):
        d = add[i] - base[i]
        ax.annotate(f"{d:+.2f}", (x[i], max(add[i], base[i])), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylim(min(-0.05, min(base + add) - 0.05), 1)
    ax.set_ylabel("Skill vs climatology (out of fold)")
    ax.set_title("Skill: baseline → add-on", loc="left", fontsize=11, color=INK)
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=INK2)
    txt = "Pairs added (final fit):\n" + "\n".join(chosen)
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", fontsize=8, color=INK, path_effects=HALO)


def overview(L, R):
    cols = [("sustained", 15), ("sustained", 20), ("sustained", 30), ("sustained", "R²"),
            ("gust", 30), ("gust", 40), ("gust", 50), ("gust", 60), ("gust", "R²")]
    sgs = sorted(set(L.signed_gradient) | set(R.signed_gradient))
    M = np.full((len(sgs), len(cols)), np.nan)
    for i, sg in enumerate(sgs):
        for j, (v, t) in enumerate(cols):
            if t == "R²":
                x = R[(R.signed_gradient == sg) & (R.variable == v)]
                if len(x):
                    M[i, j] = x.r2_addon.iloc[0] - x.r2_base.iloc[0]
            else:
                x = L[(L.signed_gradient == sg) & (L.variable == v) & (L.threshold_mph == t)]
                if len(x):
                    M[i, j] = x.bss_addon.iloc[0] - x.bss_base.iloc[0]
    fig, ax = plt.subplots(figsize=(10, 12), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(M, cmap="RdBu", norm=TwoSlopeNorm(0, -0.25, 0.25), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, "–" if np.isnan(v) else f"{v:+.2f}", ha="center", va="center", fontsize=8,
                    color=INK2 if np.isnan(v) else (SURFACE if v > 0.17 else INK))
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
    cb.set_label("Skill change, add-on − baseline (BSS or R²)", color=INK2, fontsize=9)
    fig.suptitle("Skill change from adding other gradients", x=0.02, ha="left", fontsize=13, color=INK)
    fig.text(0.02, 0.955, "Up to 3 other pairs chosen inside each CV fold; leave-one-year-out 2015–2025. "
             "Blue = better, red = worse, – = not modelled", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "overview_skill_change.png"), dpi=140, facecolor=SURFACE)
    plt.close(fig)


def main():
    L = pd.read_csv(os.path.join(ADIR, "cv_logistic.csv"))
    R = pd.read_csv(os.path.join(ADIR, "cv_linear.csv"))
    O = pd.read_parquet(os.path.join(ADIR, "oof_predictions.parquet"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    os.makedirs(OUT, exist_ok=True)
    overview(L, R)
    for (sg, var), lin in R.groupby(["signed_gradient", "variable"]):
        lin = lin.iloc[0]
        lrows = L[(L.signed_gradient == sg) & (L.variable == var)].sort_values("threshold_mph")
        o = O[(O.signed_gradient == sg) & (O.variable == var)]
        fig = plt.figure(figsize=(16, 11), facecolor=SURFACE)
        gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 1.35])
        for k in range(4):
            ax = fig.add_subplot(gs[0, k])
            if k < len(lrows):
                r = lrows.iloc[k]
                panel_rel(ax, o[(o.kind == "logistic") & (o.threshold_mph == r.threshold_mph)], r,
                          THR_COLORS[k], int(r.threshold_mph))
                if k == 0:
                    ax.set_ylabel("Observed frequency")
            else:
                ax.axis("off")
        panel_linear(fig.add_subplot(gs[1, :2]), o[o.kind == "linear"], lin)
        chosen = [c.removeprefix("g_") for c in str(lin.final_chosen).split("|") if c and c != "nan"]
        lch = {int(r.threshold_mph): [c.removeprefix("g_") for c in str(r.final_chosen).split("|") if c and c != "nan"]
               for r in lrows.itertuples()}
        lines = [f"linear: {', '.join(chosen) or 'none'}"] + [f"≥{t} mph: {', '.join(c) or 'none'}" for t, c in lch.items()]
        panel_skill(fig.add_subplot(gs[1, 2:]), lrows, lin, lines)
        fig.suptitle(f"{sg} {var}: baseline vs add-on (other gradients)", x=0.01, ha="left", fontsize=14,
                     color=INK)
        fig.text(0.01, 0.948, "Reliability, skill and linear fit, out of fold (leave-one-year-out 2015–2025), "
                 "identical hours. Top ratio = observed rate / mean forecast in the top 2% of forecasts "
                 "(1 = calibrated).", fontsize=10, color=INK2)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(os.path.join(OUT, f"{sg}_{var}.png"), dpi=115, facecolor=SURFACE)
        plt.close(fig)
        print(f"Wrote {sg}_{var}.png", flush=True)


if __name__ == "__main__":
    main()
