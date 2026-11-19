#!/usr/bin/env python3
"""Speed-model error distributions as box plots, to look into strong-wind under-prediction.

Error = forecast - observed (mph), so negative = under-prediction. Model: fixed-shape GBT (gbt_all),
out of fold (leave-one-year-out 2015-2025); the tuned GBT and linear model show the same pattern.
All gradient sets pooled.

Errors are broken down three ways (one row each), sustained and gust side by side:
  observed speed         5 mph bins (top bin open-ended), the set statistic the model predicts
  |pressure gradient|    1 hPa bins (top bin open-ended), the hour's gradient for that set
  wind direction         16 x 22.5 deg sectors of the set's circular mean direction (wdir_mean_deg,
                         stations with sustained >= 1.5 m/s); hours with no direction are dropped

Boxes: 25-75th percentile, line = median, diamond = mean (bias), whiskers = 5th-95th percentile,
no outliers drawn. Bins with fewer than MIN_N hours are not drawn. A bar strip under each panel
shows hours per bin (log scale).

-> images/models/error_boxes/error_boxes_pooled.png
-> data/v2/derived/models/error_boxes/error_box_stats.csv   the box statistics

Usage:
  python plot_error_boxes.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter

from plot_baseline_models import GRID, INK, INK2, SURFACE, style

HERE = os.path.dirname(os.path.abspath(__file__))
MDIR = os.path.normpath(os.path.join(HERE, "..", "data", "v2", "derived", "models"))
OUT = os.path.normpath(os.path.join(HERE, "..", "images", "models", "error_boxes"))
TOUT = os.path.join(MDIR, "error_boxes")
MODEL, MODEL_LABEL = "pred_gbt_all", "GBT"
COLOR = {"sustained": "#2a78d6", "gust": "#eb6834"}
TITLE = {"sustained": "Sustained (set P75)", "gust": "Gust (set P90)"}
TOP_SPEED = {"sustained": 40, "gust": 70}  # last 5 mph bin is >= this
TOP_GRAD = 12                               # last 1 hPa bin is >= this
MIN_N = 30
SECTORS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
KEYS = ["signed_gradient", "time_utc"]


def open_bins(top, width):
    e = np.r_[np.arange(0, top + 1, width), np.inf]
    labels = [f"{a:g}–{b:g}" if np.isfinite(b) else f"≥{a:g}" for a, b in zip(e[:-1], e[1:])]
    return e, labels


def assign_bins(d, var):
    """Bin index columns for the three breakdowns; returns {dimension: (column, labels, xlabel)}."""
    e, ls = open_bins(TOP_SPEED[var], 5)
    d["b_obs"] = np.digitize(d.obs, e) - 1
    e, lg = open_bins(TOP_GRAD, 1)
    d["b_grad"] = np.digitize(d.abs_gradient_hpa, e) - 1
    d["b_dir"] = np.where(d.wdir_mean_deg.notna(),
                          np.floor(((d.wdir_mean_deg.fillna(0) + 11.25) % 360) / 22.5), -1).astype(int)
    return {"obs": ("b_obs", ls, "Observed speed (mph)"),
            "grad": ("b_grad", lg, "|Pressure gradient| (hPa)"),
            "dir": ("b_dir", SECTORS, "Observed wind direction (set circular mean, from)")}


def box_stats(err, bins, nbins):
    out = []
    for i in range(nbins):
        x = err[bins == i]
        n = len(x)
        if n == 0:
            out.append({"bin": i, "n": 0})
            continue
        p5, q1, med, q3, p95 = np.percentile(x, [5, 25, 50, 75, 95])
        out.append({"bin": i, "n": n, "p5": p5, "q1": q1, "median": med, "q3": q3, "p95": p95,
                    "mean": x.mean(), "rmse": np.sqrt(np.mean(x ** 2))})
    return pd.DataFrame(out)


def draw(ax, hx, s, labels, color, xlabel):
    style(ax)
    style(hx)
    ax.axhline(0, color=INK2, lw=1, zorder=2)
    keep = s[s.n >= MIN_N]
    bxp = [{"med": r.median, "q1": r.q1, "q3": r.q3, "whislo": r.p5, "whishi": r.p95, "fliers": []}
           for r in keep.itertuples()]
    ax.bxp(bxp, positions=keep.bin.to_numpy(), widths=0.6, showfliers=False, patch_artist=True,
           boxprops=dict(facecolor=color, alpha=0.35, edgecolor=color, lw=1.2),
           medianprops=dict(color=color, lw=2), whiskerprops=dict(color=color, lw=1.2),
           capprops=dict(color=color, lw=1.2), manage_ticks=False, zorder=3)
    ax.scatter(keep.bin, keep["mean"], marker="D", s=22, color=INK, zorder=5)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_xticks(range(len(labels)))
    ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel("Forecast − observed (mph)")
    hx.bar(s.bin, np.maximum(s.n, 0.8), width=0.7, color=GRID, edgecolor=SURFACE, zorder=3)
    hx.set_yscale("log")
    hx.yaxis.set_major_locator(LogLocator(base=10))
    hx.yaxis.set_minor_formatter(NullFormatter())
    hx.set_xlim(ax.get_xlim())
    hx.set_xticks(range(len(labels)), labels, fontsize=8 if len(labels) < 15 else 7.5)
    hx.set_xlabel(xlabel, fontsize=9)
    hx.set_ylabel("Hours", fontsize=8)
    hx.tick_params(axis="y", labelsize=7)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TOUT, exist_ok=True)
    O = pd.read_parquet(os.path.join(MDIR, "gbt", "oof_predictions.parquet"),
                        columns=KEYS + ["variable", "obs", MODEL], filters=[("kind", "==", "linear")])
    dims = ["obs", "grad", "dir"]
    fig = plt.figure(figsize=(17, 15), facecolor=SURFACE)
    gs = fig.add_gridspec(3 * 2, 2, height_ratios=[3, 0.8] * 3, hspace=0.18, wspace=0.12)
    rows = []
    for j, var in enumerate(("sustained", "gust")):
        T = pd.read_parquet(os.path.join(MDIR, "targets", f"{var}_train.parquet"),
                            columns=KEYS + ["abs_gradient_hpa", "wdir_mean_deg"])
        d = O[O.variable == var].merge(T, on=KEYS, how="left", validate="one_to_one")
        assert d.abs_gradient_hpa.notna().all(), f"{var}: predictions without a target row"
        err = (d[MODEL] - d.obs).to_numpy()
        spec = assign_bins(d, var)
        for i, dim in enumerate(dims):
            col, labels, xlabel = spec[dim]
            s = box_stats(err, d[col].to_numpy(), len(labels))
            rows.append(s.assign(variable=var, breakdown=dim, bin_label=[labels[b] for b in s.bin]))
            ax = fig.add_subplot(gs[2 * i, j])
            hx = fig.add_subplot(gs[2 * i + 1, j])
            draw(ax, hx, s, labels, COLOR[var], xlabel)
            if i == 0:
                ax.set_title(f"{TITLE[var]} · {len(d):,} h", loc="left", fontsize=11.5, color=INK)
            if dim == "dir":
                nd = int((d.b_dir < 0).sum())
                hx.text(0.99, 0.85, f"{nd:,} h with no direction left out", transform=hx.transAxes,
                        ha="right", fontsize=7.5, color=INK2)
    fig.suptitle(f"Where the {MODEL_LABEL} speed forecasts go wrong: error by observed speed, "
                 "pressure gradient and wind direction", x=0.01, ha="left", y=0.995, fontsize=14, color=INK)
    fig.text(0.01, 0.972, "Out of fold, leave-one-year-out 2015–2025, all gradient sets pooled. Negative = "
             "under-prediction. Box 25–75th percentile, line median, ◆ mean (bias), whiskers 5–95th "
             f"percentile. Bins with < {MIN_N} h not drawn.", fontsize=9, color=INK2)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.955, bottom=0.05)
    fig.savefig(os.path.join(OUT, "error_boxes_pooled.png"), dpi=120, facecolor=SURFACE)
    plt.close(fig)
    pd.concat(rows, ignore_index=True)[["variable", "breakdown", "bin", "bin_label", "n", "p5", "q1", "median",
                                        "q3", "p95", "mean", "rmse"]].round(3).to_csv(
        os.path.join(TOUT, "error_box_stats.csv"), index=False)
    print(f"Wrote {OUT}/error_boxes_pooled.png and {TOUT}/error_box_stats.csv")


if __name__ == "__main__":
    main()
