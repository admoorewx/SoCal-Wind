#!/usr/bin/env python3
"""Figures comparing the three model families, and calibration of the tuned GBT.

Models (all out of fold, leave-one-year-out 2015-2025, identical hours):
  add-on     logistic / linear: |grad| + hour + doy + up to 3 BIC-selected other gradients (step 4b)
  GBT        LightGBM, all other gradients, fixed shape (step 5, gbt_all)
  tuned GBT  LightGBM, all other gradients, shape tuned by inner year-group CV (step 5b);
             run for the rare thresholds only (gust 50/60, sustained 30)

-> images/models/gbt_compare/overview_compare.png        BSS, top-end ratio and linear R2 by threshold
-> images/models/gbt_compare/tuned_calibration_pooled.png reliability pooled over all gradient sets
-> images/models/gbt_compare/tuned_calibration_<variable>_<T>.png  reliability per gradient set

Usage:
  python plot_gbt_compare.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_baseline_models import GRID, HALO, INK, INK2, SURFACE, style

HERE = os.path.dirname(os.path.abspath(__file__))
MDIR = os.path.normpath(os.path.join(HERE, "..", "data", "v2", "derived", "models"))
OUT = os.path.normpath(os.path.join(HERE, "..", "images", "models", "gbt_compare"))
# reference palette, categorical slots 1-3 (validated all-pairs, light surface)
MODELS = [("addon", "Add-on (logistic/linear)", "#2a78d6"),
          ("gbt_all", "GBT", "#eb6834"),
          ("gbt_tuned", "Tuned GBT", "#1baf7a")]
COLOR = {k: c for k, _, c in MODELS}
LABEL = {k: l for k, l, _ in MODELS}
ROWS = [("sustained", 15), ("sustained", 20), ("sustained", 30),
        ("gust", 30), ("gust", 40), ("gust", 50), ("gust", 60)]
RARE = [("gust", 50), ("gust", 60), ("sustained", 30)]
EDGES = np.array([0, .01, .02, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001])
MIN_BIN = 50
BAND = (0.85, 1.15)


def row_label(v, t):
    return f"{'Sustained' if v == 'sustained' else 'Gust'} ≥{t} mph"


def reliability(p, y, min_bin=MIN_BIN):
    idx = np.digitize(p, EDGES) - 1
    out = []
    for i in range(len(EDGES) - 1):
        s = idx == i
        if s.sum() >= min_bin:
            out.append((p[s].mean(), y[s].mean(), int(s.sum())))
    return np.array(out).reshape(-1, 3)


def legend(ax, keys, loc="lower right"):
    h = [Line2D([], [], color=COLOR[k], marker="o", lw=0, ms=8, mec=SURFACE) for k in keys]
    ax.legend(h, [LABEL[k] for k in keys], loc=loc, frameon=False, fontsize=8, labelcolor=INK2)


def overview(L, Lt, R):
    """Three panels: median BSS (dots, IQR whisker), top-end ratio per model (strip), linear R2."""
    L = L.merge(Lt[["signed_gradient", "variable", "threshold_mph", "bss_gbt_tuned", "top_ratio_gbt_tuned"]],
                on=["signed_gradient", "variable", "threshold_mph"], how="left")
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.6), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.1, 1.4, 0.8]})
    off = {"addon": -0.22, "gbt_all": 0.0, "gbt_tuned": 0.22}
    ylab = [row_label(v, t) for v, t in ROWS]
    # A: median BSS with IQR
    ax = axes[0]
    style(ax)
    for i, (v, t) in enumerate(ROWS):
        g = L[(L.variable == v) & (L.threshold_mph == t)]
        for k, _, c in MODELS:
            x = g[f"bss_{k}"].dropna()
            if x.empty:
                continue
            q1, med, q3 = np.percentile(x, [25, 50, 75])
            y = i + off[k]
            ax.plot([q1, q3], [y, y], color=c, lw=2, alpha=0.45, solid_capstyle="round", zorder=3)
            ax.scatter(med, y, s=64, color=c, edgecolor=SURFACE, linewidth=2, zorder=4)
    ns = [len(L[(L.variable == v) & (L.threshold_mph == t)]) for v, t in ROWS]
    ax.set_yticks(range(len(ROWS)), [f"{l}\n{n} gradient sets" for l, n in zip(ylab, ns)], fontsize=9)
    ax.invert_yaxis()
    ax.axhline(2.5, color=GRID, lw=1.2)
    ax.set_xlabel("Brier skill score vs climatology (median, bar = middle 50% of gradient sets)")
    ax.set_title("A  Probability skill", loc="left", fontsize=11, color=INK)
    legend(ax, [k for k, _, _ in MODELS])
    # B: top-end ratio, one dot per model
    ax = axes[1]
    style(ax)
    ax.axvspan(*BAND, color=GRID, alpha=0.55, lw=0, zorder=1)
    ax.axvline(1, color=INK2, lw=1, ls=(0, (3, 2)), zorder=2)
    rng = np.random.default_rng(3)
    for i, (v, t) in enumerate(ROWS):
        g = L[(L.variable == v) & (L.threshold_mph == t)]
        for k, _, c in MODELS:
            x = g[f"top_ratio_{k}"].dropna().to_numpy()
            if x.size == 0:
                continue
            y = i + off[k] + rng.uniform(-0.07, 0.07, x.size)
            ax.scatter(np.clip(x, 0.3, 1.8), y, s=26, color=c, edgecolor=SURFACE, linewidth=0.8, zorder=4)
            nlo, nhi = int((x < BAND[0]).sum()), int((x > BAND[1]).sum())
            if nlo or nhi:
                ax.text(1.83, i + off[k], f"{nlo} low / {nhi} high", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(ROWS)), [""] * len(ROWS))
    ax.invert_yaxis()
    ax.axhline(2.5, color=GRID, lw=1.2)
    ax.set_xlim(0.3, 2.15)
    ax.set_xticks([0.4, 0.6, 0.85, 1.0, 1.15, 1.4, 1.6, 1.8])
    ax.set_xlabel("Top-end ratio = observed rate / mean forecast, top 2% of forecasts\n"
                  "← over-forecast        shaded 0.85–1.15 ≈ calibrated        under-forecast →")
    ax.set_title("B  Top-end calibration, one dot per gradient set", loc="left", fontsize=11, color=INK)
    # C: linear R2
    ax = axes[2]
    style(ax)
    for i, v in enumerate(("sustained", "gust")):
        g = R[R.variable == v]
        for k, _, c in MODELS[:2]:
            q1, med, q3 = np.percentile(g[f"r2_{k}"], [25, 50, 75])
            y = i + off[k]
            ax.plot([q1, q3], [y, y], color=c, lw=2, alpha=0.45, solid_capstyle="round", zorder=3)
            ax.scatter(med, y, s=64, color=c, edgecolor=SURFACE, linewidth=2, zorder=4)
            ax.annotate(f"{med:.2f}", (med, y), xytext=(0, 8), textcoords="offset points", ha="center",
                        fontsize=8, color=INK, path_effects=HALO)
    ax.set_yticks([0, 1], ["Sustained\n(regional P75)", "Gust\n(P90, 2-stn floor)"], fontsize=9)
    ax.set_ylim(1.6, -0.6)
    ax.set_xlabel("Out-of-fold R² (median, bar = middle 50%)")
    ax.set_title("C  Wind speed (linear target)", loc="left", fontsize=11, color=INK)
    legend(ax, ["addon", "gbt_all"])
    fig.suptitle("Add-on regression vs gradient boosted trees", x=0.01, ha="left", fontsize=14, color=INK)
    fig.text(0.01, 0.925, "Leave-one-year-out 2015–2025, identical hours for every model. Tuned GBT was run "
             "for the rare thresholds only (gust ≥50/60, sustained ≥30). 2026 holdout untouched.",
             fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(os.path.join(OUT, "overview_compare.png"), dpi=140, facecolor=SURFACE)
    plt.close(fig)


def rel_panel(ax, o, keys, lw_main=2.4, min_bin=MIN_BIN, lim=1.0):
    style(ax)
    ax.plot([0, lim], [0, lim], color=INK2, lw=1, ls=(0, (3, 2)), zorder=2)
    y = o.obs.to_numpy(float)
    for k in keys:
        r = reliability(o[f"pred_{k}"].to_numpy(float), y, min_bin)
        main = k == "gbt_tuned"
        ax.plot(r[:, 0], r[:, 1], color=COLOR[k], lw=lw_main if main else 1.4, marker="o",
                ms=5 if main else 3.5, mec=SURFACE, alpha=1 if main else 0.85, zorder=5 if main else 4)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")


def pooled(Ot):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.4), facecolor=SURFACE, sharex="col",
                             gridspec_kw={"height_ratios": [3, 1]})
    keys = ["addon", "gbt_all", "gbt_tuned"]
    for j, (v, t) in enumerate(RARE):
        o = Ot[(Ot.variable == v) & (Ot.threshold_mph == t)]
        ax = axes[0, j]
        rel_panel(ax, o, keys)
        ax.set_title(f"{row_label(v, t)}  ·  {o.signed_gradient.nunique()} gradient sets, "
                     f"{int(o.obs.sum()):,} event h", loc="left", fontsize=10.5, color=INK)
        if j == 0:
            ax.set_ylabel("Observed frequency")
            legend(ax, keys, loc="upper left")
        # forecast counts (log) for the tuned model
        hx = axes[1, j]
        style(hx)
        cnt, _ = np.histogram(o.pred_gbt_tuned, bins=np.linspace(0, 1, 41))
        hx.bar(np.linspace(0.0125, 0.9875, 40), np.maximum(cnt, 0.8), width=0.021, color=COLOR["gbt_tuned"],
               edgecolor=SURFACE, linewidth=0.5, zorder=3)
        hx.set_yscale("log")
        hx.set_xlabel("Forecast probability")
        if j == 0:
            hx.set_ylabel("Hours (tuned GBT)")
    fig.suptitle("Tuned GBT calibration, pooled over all gradient sets", x=0.01, ha="left", fontsize=14,
                 color=INK)
    fig.text(0.01, 0.935, f"Out-of-fold reliability (bins with ≥{MIN_BIN} h); dashed = perfect calibration. "
             "Pooling weights large gradient sets more; per-set panels are in tuned_calibration_*.png.",
             fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT, "tuned_calibration_pooled.png"), dpi=140, facecolor=SURFACE)
    plt.close(fig)


def per_set(Ot, Lt):
    for v, t in RARE:
        o = Ot[(Ot.variable == v) & (Ot.threshold_mph == t)]
        sgs = sorted(o.signed_gradient.unique())
        nc = 6
        nr = int(np.ceil(len(sgs) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(2.6 * nc, 2.75 * nr + 0.9), facecolor=SURFACE,
                                 squeeze=False)
        for ax, sg in zip(axes.flat, sgs):
            g = o[o.signed_gradient == sg]
            rel_panel(ax, g, ["addon", "gbt_tuned"], lw_main=2.0, min_bin=30)
            r = Lt[(Lt.signed_gradient == sg) & (Lt.variable == v) & (Lt.threshold_mph == t)].iloc[0]
            ax.set_title(sg, loc="left", fontsize=9.5, color=INK)
            ax.text(0.04, 0.96, f"{int(r.event_hours)} event h\ntop ratio {r.top_ratio_addon:.2f} → "
                    f"{r.top_ratio_gbt_tuned:.2f}\nBSS {r.bss_addon:.2f} → {r.bss_gbt_tuned:.2f}",
                    transform=ax.transAxes, va="top", fontsize=7, color=INK, path_effects=HALO)
            ax.tick_params(labelsize=7)
        for ax in list(axes.flat)[len(sgs):]:
            ax.axis("off")
        legend(axes.flat[len(sgs) - 1] if len(sgs) % nc == 0 else list(axes.flat)[len(sgs)],
               ["addon", "gbt_tuned"], loc="center")
        fig.suptitle(f"{row_label(v, t)}: tuned GBT vs add-on calibration by gradient set", x=0.01, ha="left",
                     fontsize=13, color=INK)
        fig.text(0.01, 1 - 0.55 / fig.get_figheight(),
                 "x = forecast probability, y = observed frequency (bins with ≥30 h), out of fold. "
                 "Numbers: add-on → tuned GBT.", fontsize=9, color=INK2)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
        fig.savefig(os.path.join(OUT, f"tuned_calibration_{v}_{t}.png"), dpi=120, facecolor=SURFACE)
        plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    os.makedirs(OUT, exist_ok=True)
    L = pd.read_csv(os.path.join(MDIR, "gbt", "cv_logistic.csv"))
    R = pd.read_csv(os.path.join(MDIR, "gbt", "cv_linear.csv"))
    Lt = pd.read_csv(os.path.join(MDIR, "gbt_tuned", "cv_logistic.csv"))
    Ot = pd.read_parquet(os.path.join(MDIR, "gbt_tuned", "oof_predictions.parquet"))
    overview(L, Lt, R)
    pooled(Ot)
    per_set(Ot, Lt)
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
