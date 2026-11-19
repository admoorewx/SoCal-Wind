#!/usr/bin/env python3
"""Speed (point-forecast) models: linear add-on vs GBT vs tuned GBT, error by 5 mph wind-speed bin.

Targets: sustained = the station set's 75th percentile; gust = nearest-rank P90 with 2-station floor
(same values the event models threshold). All predictions are out of fold (leave-one-year-out
2015-2025) on identical hours; 2026 holdout untouched.

Models:
  linear     add-on linear regression (|grad| + hour + doy + up to 3 BIC-chosen other gradients)
  GBT        LightGBM, all other gradients, fixed shape (gbt_all)
  tuned GBT  LightGBM, all other gradients, shape tuned by inner year-group CV (gbt_tuned_speed)

Error is binned two ways, both 5 mph wide:
  by OBSERVED speed  how large are the errors when the wind really was X mph (strong-event view;
                     a least-squares model always under-predicts the top bins here)
  by FORECAST speed  when the model says X mph, how far off is it (user view, calibration)

-> images/models/speed_compare/rmse_by_speed_pooled.png       all gradient sets pooled
-> images/models/speed_compare/rmse_by_speed_<variable>.png   one panel per gradient set
-> data/v2/derived/models/speed_compare/error_by_bin.csv       the binned numbers (pooled and per set)

Usage:
  python plot_speed_compare.py
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
OUT = os.path.normpath(os.path.join(HERE, "..", "images", "models", "speed_compare"))
TOUT = os.path.join(MDIR, "speed_compare")
MODELS = [("addon", "Linear (add-on)", "#2a78d6"),
          ("gbt_all", "GBT", "#eb6834"),
          ("gbt_tuned", "Tuned GBT", "#1baf7a")]
COLOR = {k: c for k, _, c in MODELS}
LABEL = {k: l for k, l, _ in MODELS}
TOP = {"sustained": 40, "gust": 70}  # last bin is open-ended: >= TOP
MIN_N = 30  # bins with fewer hours are not plotted
KEYS = ["signed_gradient", "variable", "kind", "threshold_mph", "time_utc"]


def bin_edges(var):
    return np.r_[np.arange(0, TOP[var] + 1, 5), np.inf]


def bin_labels(var):
    e = bin_edges(var)
    return [f"{int(a)}–{int(b)}" if np.isfinite(b) else f"≥{int(a)}" for a, b in zip(e[:-1], e[1:])]


def binned(d, var, by):
    """RMSE, bias, MAE and hours per 5 mph bin of `by` ('obs' or a model's forecast)."""
    e = bin_edges(var)
    rows = []
    for k, _, _ in MODELS:
        x = d.obs if by == "obs" else d[f"pred_{k}"]
        idx = np.digitize(x.to_numpy(), e) - 1
        err = (d[f"pred_{k}"] - d.obs).to_numpy()
        for i in range(len(e) - 1):
            s = idx == i
            n = int(s.sum())
            rows.append({"model": k, "bin": i, "n": n,
                         "rmse": np.sqrt(np.mean(err[s] ** 2)) if n else np.nan,
                         "bias": err[s].mean() if n else np.nan,
                         "mae": np.abs(err[s]).mean() if n else np.nan})
    return pd.DataFrame(rows)


def series(ax, t, metric, labels):
    for k, _, c in MODELS:
        g = t[(t.model == k) & (t.n >= MIN_N)]
        ax.plot(g.bin, g[metric], color=c, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.2,
                zorder=5 if k == "gbt_tuned" else 4)
    ax.set_xticks(range(len(labels)), labels, fontsize=8, rotation=0)
    ax.set_xlim(-0.5, len(labels) - 0.5)


def legend(ax, loc="upper left"):
    h = [Line2D([], [], color=c, marker="o", lw=2, ms=6, mec=SURFACE) for _, _, c in MODELS]
    ax.legend(h, [l for _, l, _ in MODELS], loc=loc, frameon=False, fontsize=8.5, labelcolor=INK2)


def pooled(O, rows_out):
    fig, axes = plt.subplots(3, 4, figsize=(19, 11.5), facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [2.2, 1.6, 0.9]})
    for j, (var, by) in enumerate([("sustained", "obs"), ("sustained", "pred"),
                                   ("gust", "obs"), ("gust", "pred")]):
        d = O[O.variable == var]
        labels = bin_labels(var)
        t = binned(d, var, by)
        rows_out.append(t.assign(signed_gradient="ALL", variable=var, binned_by=by,
                                 bin_label=[labels[i] for i in t.bin]))
        what = "observed" if by == "obs" else "forecast"
        ax = axes[0, j]
        style(ax)
        series(ax, t, "rmse", labels)
        tot = {k: np.sqrt(np.mean((d[f"pred_{k}"] - d.obs) ** 2)) for k, _, _ in MODELS}
        ax.set_title(f"{'Sustained (set P75)' if var == 'sustained' else 'Gust (set P90)'}"
                     f" · binned by {what} speed", loc="left", fontsize=10.5, color=INK)
        ax.text(0.02, 0.97, "overall RMSE\n" + "\n".join(f"{LABEL[k]}: {v:.2f} mph" for k, v in tot.items()),
                transform=ax.transAxes, va="top", fontsize=8, color=INK, path_effects=HALO)
        if j == 0:
            ax.set_ylabel("RMSE (mph)")
        if j == 3:
            legend(ax, loc="upper right")
        ax.set_ylim(bottom=0)
        # bias
        bx = axes[1, j]
        style(bx)
        bx.axhline(0, color=INK2, lw=1, zorder=2)
        series(bx, t, "bias", labels)
        if j == 0:
            bx.set_ylabel("Bias, forecast − observed (mph)")
        # hours
        hx = axes[2, j]
        style(hx)
        n = t[t.model == ("addon" if by == "obs" else "gbt_tuned")]
        hx.bar(n.bin, np.maximum(n.n, 0.8), width=0.8, color=GRID if by == "obs" else COLOR["gbt_tuned"],
               edgecolor=SURFACE, zorder=3)
        hx.set_yscale("log")
        hx.set_xticks(range(len(labels)), labels, fontsize=8)
        hx.set_xlim(-0.5, len(labels) - 0.5)
        hx.set_xlabel(f"{what.capitalize()} speed bin (mph)")
        if j == 0:
            hx.set_ylabel("Hours")
        if by == "pred":
            hx.text(0.98, 0.9, "hours by tuned-GBT forecast", transform=hx.transAxes, ha="right",
                    fontsize=7.5, color=INK2)
    fig.suptitle("Wind-speed forecasts: error by 5 mph bin, all gradient sets pooled", x=0.01, ha="left",
                 fontsize=14, color=INK)
    fig.text(0.01, 0.948, "Out of fold, leave-one-year-out 2015–2025, identical hours. Binned by observed speed: "
             "error during strong winds (least squares always under-predicts the top bins). Binned by forecast "
             f"speed: how far off a given forecast is. Bins with < {MIN_N} h not shown.",
             fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "rmse_by_speed_pooled.png"), dpi=130, facecolor=SURFACE)
    plt.close(fig)


def per_set(O, rows_out):
    for var in ("sustained", "gust"):
        d = O[O.variable == var]
        labels = bin_labels(var)
        sgs = sorted(d.signed_gradient.unique())
        nc = 5
        nr = int(np.ceil(len(sgs) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(3.6 * nc, 2.6 * nr + 1), facecolor=SURFACE, squeeze=False,
                                 sharex=True)
        ymax = 0
        for ax, sg in zip(axes.flat, sgs):
            g = d[d.signed_gradient == sg]
            t = binned(g, var, "obs")
            rows_out.append(t.assign(signed_gradient=sg, variable=var, binned_by="obs",
                                     bin_label=[labels[i] for i in t.bin]))
            style(ax)
            series(ax, t, "rmse", labels)
            ymax = max(ymax, t[t.n >= MIN_N].rmse.max())
            ax.set_title(sg, loc="left", fontsize=9.5, color=INK)
            tot = {k: np.sqrt(np.mean((g[f"pred_{k}"] - g.obs) ** 2)) for k, _, _ in MODELS}
            ax.text(0.03, 0.95, " / ".join(f"{v:.1f}" for v in tot.values()) + " mph",
                    transform=ax.transAxes, va="top", fontsize=7.5, color=INK, path_effects=HALO)
            ax.tick_params(labelsize=6.5)
            ax.set_xticks(range(len(labels)), labels, fontsize=6, rotation=90)
        for ax in axes.flat:
            ax.set_ylim(0, ymax * 1.05)
        spare = list(axes.flat)[len(sgs):]
        for ax in spare:
            ax.axis("off")
        legend(spare[0] if spare else axes.flat[0], loc="center" if spare else "upper left")
        fh = fig.get_figheight()
        fig.suptitle(f"{'Sustained (set P75)' if var == 'sustained' else 'Gust (set P90)'}: RMSE by observed "
                     "speed bin, per gradient set", x=0.01, y=1 - 0.15 / fh, ha="left", va="top",
                     fontsize=13, color=INK)
        fig.text(0.01, 1 - 0.55 / fh, "y = RMSE (mph), x = observed 5 mph bin; shared y axis. Corner numbers: "
                 "overall RMSE, linear / GBT / tuned GBT. Bins with < 30 h not shown.", fontsize=9, color=INK2)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / fh))
        fig.savefig(os.path.join(OUT, f"rmse_by_speed_{var}.png"), dpi=120, facecolor=SURFACE)
        plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TOUT, exist_ok=True)
    O = pd.read_parquet(os.path.join(MDIR, "gbt", "oof_predictions.parquet"),
                        columns=KEYS + ["obs", "pred_addon", "pred_gbt_all"], filters=[("kind", "==", "linear")])
    Ot = pd.read_parquet(os.path.join(MDIR, "gbt_tuned_speed", "oof_predictions.parquet"),
                         columns=KEYS + ["obs", "pred_gbt_tuned"], filters=[("kind", "==", "linear")])
    O = O.merge(Ot, on=KEYS, how="inner", suffixes=("", "_t"))
    assert (O.obs == O.obs_t).all(), "observations differ between gbt and gbt_tuned_speed runs"
    O = O.drop(columns="obs_t")
    rows = []
    pooled(O, rows)
    per_set(O, rows)
    pd.concat(rows, ignore_index=True)[["variable", "signed_gradient", "binned_by", "bin", "bin_label", "model",
                                        "n", "rmse", "bias", "mae"]].round(3).to_csv(
        os.path.join(TOUT, "error_by_bin.csv"), index=False)
    print(f"Wrote figures to {OUT} and {TOUT}/error_by_bin.csv")


if __name__ == "__main__":
    main()
