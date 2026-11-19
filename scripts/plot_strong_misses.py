#!/usr/bin/env python3
"""Strong-wind misses of the GBT speed models, and event-model Brier skill by observed speed and gradient.

Strong hours: sustained (set P75) > 40 mph, gust (set P90) > 50 mph (observed). A strong hour is a
MISS when |forecast - observed| > 5 mph. Speed model: fixed-shape GBT (gbt_all), out of fold
(leave-one-year-out 2015-2025), all gradient sets pooled. Hour of day is local standard time (UTC-8).

Figures:
  strong_miss_breakdown.png  share of strong hours that are misses, and strong-hour counts split into
                             misses / others, by |gradient|, hour of day and month
  strong_miss_by_set.png     per signed gradient: mean error and miss share over its strong hours
  bss_by_speed_gradient.png  Brier skill score vs climatology of the chosen event models (tuned GBT for
                             sustained 30 / gust 50, 60; fixed GBT for the other thresholds, which were
                             not tuned), binned by observed speed and by |gradient|

Note on BSS by observed speed: within a bin the outcome is almost fixed (all events above the threshold,
none below), so the score measures how far the probability moves away from climatology toward the right
answer for hours with that observed speed. It is conditional on the outcome, like the error-by-observed-
speed plots.

-> images/models/strong_misses/*.png
-> data/v2/derived/models/strong_misses/{strong_hours.parquet, by_set.csv, bss_by_bin.csv}

Usage:
  python plot_strong_misses.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter

from plot_baseline_models import GRID, HALO, INK, INK2, SURFACE, style

HERE = os.path.dirname(os.path.abspath(__file__))
MDIR = os.path.normpath(os.path.join(HERE, "..", "data", "v2", "derived", "models"))
OUT = os.path.normpath(os.path.join(HERE, "..", "images", "models", "strong_misses"))
TOUT = os.path.join(MDIR, "strong_misses")
KEYS = ["signed_gradient", "time_utc"]
STRONG = {"sustained": 40, "gust": 50}
MISS_MPH = 5
TITLE = {"sustained": "Sustained (set P75)", "gust": "Gust (set P90)"}
MISS_C, OTHER_C, LINE_C = "#eb6834", GRID, "#2a78d6"
RAMP = {"sustained": ["#86b6ef", "#2a78d6", "#104281"],             # ordinal blue, 15 / 20 / 30 mph
        "gust": ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]}       # 30 / 40 / 50 / 60 mph
TUNED = {("sustained", 30), ("gust", 50), ("gust", 60)}
MIN_N_SHARE = 10   # strong hours needed to plot a miss share
MIN_N_BSS = 30     # hours needed to plot a BSS
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def targets(var):
    T = pd.read_parquet(os.path.join(MDIR, "targets", f"{var}_train.parquet"),
                        columns=KEYS + ["abs_gradient_hpa", "hour_utc", "target_mph"])
    T["hour_lst"] = (T.hour_utc.astype(int) - 8) % 24
    T["month"] = T.time_utc.dt.month
    return T


def strong_hours():
    O = pd.read_parquet(os.path.join(MDIR, "gbt", "oof_predictions.parquet"),
                        columns=KEYS + ["variable", "obs", "pred_gbt_all"], filters=[("kind", "==", "linear")])
    parts = []
    for var in ("sustained", "gust"):
        d = O[O.variable == var].merge(targets(var), on=KEYS, how="left", validate="one_to_one")
        assert np.allclose(d.obs, d.target_mph, atol=1e-4), f"{var}: obs and target_mph differ"
        d = d[d.obs > STRONG[var]].copy()
        d["err"] = d.pred_gbt_all - d.obs
        d["miss"] = d.err.abs() > MISS_MPH
        parts.append(d.drop(columns="target_mph"))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- figure 1: breakdown of strong hours
def breakdown(S):
    dims = [("grad", "|Pressure gradient| (hPa)"), ("hour", "Hour of day (local standard time)"),
            ("month", "Month")]
    fig = plt.figure(figsize=(17, 14), facecolor=SURFACE)
    gs = fig.add_gridspec(6, 2, height_ratios=[2, 1.3] * 3, hspace=0.22, wspace=0.13)
    for j, var in enumerate(("sustained", "gust")):
        d = S[S.variable == var]
        gtop = int(np.ceil(d.abs_gradient_hpa.quantile(0.99)))
        ge = np.r_[np.arange(0, gtop + 1), np.inf]
        glab = [f"{a:g}–{b:g}" if np.isfinite(b) else f"≥{a:g}" for a, b in zip(ge[:-1], ge[1:])]
        bins = {"grad": (np.digitize(d.abs_gradient_hpa, ge) - 1, glab),
                "hour": (d.hour_lst.to_numpy(), [str(h) for h in range(24)]),
                "month": (d.month.to_numpy() - 1, MONTHS)}
        for i, (dim, xlabel) in enumerate(dims):
            b, labels = bins[dim]
            k = np.arange(len(labels))
            n = np.bincount(b, minlength=len(labels))
            m = np.bincount(b, weights=d.miss.to_numpy(float), minlength=len(labels))
            share = np.where(n >= MIN_N_SHARE, 100 * m / np.maximum(n, 1), np.nan)
            ax = fig.add_subplot(gs[2 * i, j])
            hx = fig.add_subplot(gs[2 * i + 1, j])
            for a in (ax, hx):
                style(a)
                a.set_xlim(-0.6, len(labels) - 0.4)
                a.set_xticks(k)
            overall = 100 * d.miss.mean()
            ax.axhline(overall, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
            ax.text(len(labels) - 0.5, overall, f"all strong hours {overall:.0f}%", ha="right", va="bottom",
                    fontsize=8, color=INK2, path_effects=HALO)
            ax.plot(k, share, color=LINE_C, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.2, zorder=4)
            ax.set_ylim(0, 100)
            ax.set_ylabel("Misses (% of strong h)")
            ax.tick_params(axis="x", labelbottom=False)
            hx.bar(k, n - m, width=0.75, color=OTHER_C, edgecolor=SURFACE, lw=1, zorder=3)
            hx.bar(k, m, bottom=n - m, width=0.75, color=MISS_C, edgecolor=SURFACE, lw=1, zorder=3)
            hx.set_ylabel("Strong hours")
            hx.set_xticks(k, labels, fontsize=7.5 if len(labels) > 14 else 8.5)
            hx.set_xlabel(xlabel, fontsize=9)
            if i == 0:
                nu = int((d.miss & (d.err < 0)).sum())
                no = int((d.miss & (d.err > 0)).sum())
                ax.set_title(f"{TITLE[var]} > {STRONG[var]} mph · {len(d):,} strong h, {int(d.miss.sum()):,} "
                             f"misses ({nu:,} under, {no:,} over)", loc="left", fontsize=11, color=INK)
            if i == 0 and j == 1:
                hx.legend([Patch(color=MISS_C), Patch(color=OTHER_C)],
                          [f"miss (|error| > {MISS_MPH} mph)", f"|error| ≤ {MISS_MPH} mph"],
                          loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.suptitle("Strong-wind hours the GBT speed model misses by more than 5 mph: when do they happen?",
                 x=0.01, ha="left", y=0.995, fontsize=14, color=INK)
    fig.text(0.01, 0.972, "Out of fold 2015–2025, all gradient sets pooled. Line: share of strong hours in "
             f"the bin that are misses (bins with < {MIN_N_SHARE} strong h not drawn); dashed: share over all "
             "strong hours. Bars: strong hours per bin.", fontsize=9, color=INK2)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.955, bottom=0.045)
    fig.savefig(os.path.join(OUT, "strong_miss_breakdown.png"), dpi=120, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- figure 2: per gradient set
def by_set(S):
    t = (S.groupby(["variable", "signed_gradient"])
         .agg(n_strong=("err", "size"), mean_err=("err", "mean"), median_err=("err", "median"),
              miss_share=("miss", "mean"), n_under_miss=("err", lambda e: int((e < -MISS_MPH).sum())),
              mean_obs=("obs", "mean"), mean_abs_grad=("abs_gradient_hpa", "mean"))
         .reset_index())
    t.to_csv(os.path.join(TOUT, "by_set.csv"), index=False, float_format="%.3f")
    fig, axes = plt.subplots(1, 2, figsize=(16, 10), facecolor=SURFACE)
    for ax, var in zip(axes, ("sustained", "gust")):
        g = t[(t.variable == var) & (t.n_strong >= MIN_N_SHARE)].sort_values("mean_err", ascending=False)
        dropped = t[(t.variable == var) & (t.n_strong < MIN_N_SHARE)].signed_gradient.tolist()
        style(ax)
        y = np.arange(len(g))
        ax.barh(y, g.mean_err, height=0.65, color=MISS_C, edgecolor=SURFACE, zorder=3)
        ax.axvline(0, color=INK2, lw=1, zorder=2)
        ax.set_yticks(y, g.signed_gradient, fontsize=9)
        ax.set_ylim(-0.7, len(g) - 0.3)
        lo = g.mean_err.min()
        ax.set_xlim(lo * 1.7, max(1, g.mean_err.max() * 1.2))
        for yy, r in zip(y, g.itertuples()):
            ax.text(r.mean_err - 0.3, yy, f"{r.n_strong:,} h · {100 * r.miss_share:.0f}% miss · "
                    f"|grad| {r.mean_abs_grad:.1f}", ha="right", va="center", fontsize=7.5, color=INK2,
                    path_effects=HALO)
        ax.set_xlabel("Mean error over strong hours, forecast − observed (mph)")
        ax.set_title(f"{TITLE[var]} > {STRONG[var]} mph", loc="left", fontsize=11.5, color=INK)
        if dropped:
            ax.text(0.0, -0.075, f"Fewer than {MIN_N_SHARE} strong h, not shown: " + ", ".join(dropped),
                    transform=ax.transAxes, fontsize=7.5, color=INK2, wrap=True)
    fig.suptitle("Which gradient sets under-forecast their strong winds the most?", x=0.01, ha="left",
                 fontsize=14, color=INK)
    fig.text(0.01, 0.935, "GBT speed model, out of fold 2015–2025. Bar: mean error over the set's strong hours "
             "(most under-forecast at the bottom). Label: strong hours · share missed by > 5 mph · mean "
             "|gradient| (hPa) in those hours.", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(os.path.join(OUT, "strong_miss_by_set.png"), dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return t


# ---------------------------------------------------------------- figure 3: BSS by observed speed / gradient
def event_predictions():
    cols = KEYS + ["variable", "threshold_mph", "obs", "clim"]
    E = pd.read_parquet(os.path.join(MDIR, "gbt", "oof_predictions.parquet"), columns=cols + ["pred_gbt_all"],
                        filters=[("kind", "==", "logistic")])
    Et = pd.read_parquet(os.path.join(MDIR, "gbt_tuned", "oof_predictions.parquet"),
                         columns=cols + ["pred_gbt_tuned"], filters=[("kind", "==", "logistic")])
    E = E.merge(Et, on=KEYS + ["variable", "threshold_mph"], how="left", suffixes=("", "_t"))
    t = E.pred_gbt_tuned.notna()
    assert (E.obs[t] == E.obs_t[t]).all(), "event outcomes differ between gbt and gbt_tuned"
    assert set(map(tuple, E.loc[t, ["variable", "threshold_mph"]].drop_duplicates().to_numpy())) == TUNED
    E["p"] = np.where(t, E.pred_gbt_tuned, E.pred_gbt_all)
    return E.drop(columns=["obs_t", "clim_t", "pred_gbt_tuned", "pred_gbt_all"])


def bss_table(E):
    rows = []
    for var in ("sustained", "gust"):
        T = targets(var)[KEYS + ["abs_gradient_hpa", "target_mph"]]
        d = E[E.variable == var].merge(T, on=KEYS, how="left", validate="many_to_one")
        top = 60 if var == "sustained" else 90
        se = np.r_[np.arange(0, top + 1, 5), np.inf]
        ge = np.r_[np.arange(0, 15), np.inf]
        for by, e, x in (("obs_speed", se, d.target_mph), ("abs_gradient", ge, d.abs_gradient_hpa)):
            d["bin"] = np.digitize(x, e) - 1
            bs = ((d.p - d.obs) ** 2).groupby([d.threshold_mph, d.bin]).sum()
            br = ((d.clim - d.obs) ** 2).groupby([d.threshold_mph, d.bin]).sum()
            g = d.groupby(["threshold_mph", "bin"]).agg(n=("p", "size"), events=("obs", "sum"),
                                                        mean_p=("p", "mean"), mean_clim=("clim", "mean"))
            g["bss"] = 1 - bs / br
            g = g.reset_index()
            g["lo"], g["hi"] = e[g.bin], e[g.bin + 1]
            rows.append(g.assign(variable=var, binned_by=by))
    return pd.concat(rows, ignore_index=True)


def bss_figure(B):
    fig, axes = plt.subplots(3, 2, figsize=(17, 15), facecolor=SURFACE)
    YMIN = -1.0
    for j, var in enumerate(("sustained", "gust")):
        ths = sorted(B[B.variable == var].threshold_mph.unique())
        for i, (by, xlabel) in enumerate((("obs_speed", "Observed speed (mph), 5 mph bins"),
                                          ("abs_gradient", "|Pressure gradient| (hPa), 1 hPa bins"))):
            ax = axes[i, j]
            style(ax)
            ax.axhline(0, color=INK2, lw=1, zorder=2)
            for th, c in zip(ths, RAMP[var]):
                g = B[(B.variable == var) & (B.binned_by == by) & (B.threshold_mph == th) & (B.n >= MIN_N_BSS)]
                x = (g.lo + np.where(np.isfinite(g.hi), g.hi, g.lo + 5 if by == "obs_speed" else g.lo + 1)) / 2
                y = g.bss.to_numpy()
                ax.plot(x, np.maximum(y, YMIN), color=c, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.2, zorder=4)
                clipped = y < YMIN
                ax.scatter(x[clipped], np.full(clipped.sum(), YMIN), marker="v", s=40, color=c, zorder=5)
                if by == "obs_speed":
                    ax.axvline(th, color=c, lw=1, ls=(0, (3, 3)), zorder=1)
                lbl = f"{th} mph" + (" (tuned)" if (var, th) in TUNED else "")
                last = np.isfinite(y)
                if last.any():
                    k = np.flatnonzero(last)[-1]
                    ax.annotate(lbl, (x.iloc[k], max(y[k], YMIN)), xytext=(6, 0), textcoords="offset points",
                                va="center", fontsize=8, color=INK, path_effects=HALO)
            ax.set_ylim(YMIN - 0.05, 1.02)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel("Brier skill score vs climatology")
            if i == 0:
                ax.set_title(f"{TITLE[var]} event models", loc="left", fontsize=11.5, color=INK)
                ax.text(0.99, 0.03, "dotted lines: event threshold", transform=ax.transAxes, ha="right",
                        fontsize=8, color=INK2)
        # mean forecast probability by observed speed: explains the top row
        ax = axes[2, j]
        style(ax)
        for th, c in zip(ths, RAMP[var]):
            g = B[(B.variable == var) & (B.binned_by == "obs_speed") & (B.threshold_mph == th) & (B.n >= MIN_N_BSS)]
            x = (g.lo + np.where(np.isfinite(g.hi), g.hi, g.lo + 5)) / 2
            ax.plot(x, g.mean_p, color=c, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.2, zorder=4)
            ax.axvline(th, color=c, lw=1, ls=(0, (3, 3)), zorder=1)
            ax.annotate(f"{th} mph", (x.iloc[-1], g.mean_p.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK, path_effects=HALO)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Observed speed (mph), 5 mph bins", fontsize=9)
        ax.set_ylabel("Mean forecast probability")
        ax.text(0.01, 0.97, "mean event probability given the hour's observed speed (ideal: 0 left of the "
                "threshold, 1 right of it)", transform=ax.transAxes, va="top", fontsize=8, color=INK2)
        axes[0, j].legend([Line2D([], [], color=c, lw=2, marker="o", mec=SURFACE) for c in RAMP[var]],
                          [f"≥ {t} mph" for t in ths], loc="lower left", frameon=False, fontsize=8.5,
                          labelcolor=INK2, title="threshold", title_fontsize=8.5)
    fig.suptitle("GBT event forecasts: Brier skill by observed speed and by pressure gradient", x=0.01,
                 ha="left", y=0.995, fontsize=14, color=INK)
    fig.text(0.01, 0.962, "Out of fold 2015–2025, all gradient sets pooled (BSS = 1 − ΣBS / ΣBS_clim per bin). "
             "Tuned GBT for sustained 30 and gust 50/60, fixed GBT otherwise. Top row is conditional on the "
             f"outcome (see script notes). Values below {YMIN:g} drawn as ▼ at {YMIN:g}. Bins < {MIN_N_BSS} h "
             "not drawn. Bottom row: mean forecast probability by observed speed.", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "bss_by_speed_gradient.png"), dpi=120, facecolor=SURFACE)
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TOUT, exist_ok=True)
    S = strong_hours()
    S.to_parquet(os.path.join(TOUT, "strong_hours.parquet"), index=False)
    breakdown(S)
    by_set(S)
    B = bss_table(event_predictions())
    B[["variable", "threshold_mph", "binned_by", "bin", "lo", "hi", "n", "events", "mean_p", "mean_clim",
       "bss"]].to_csv(os.path.join(TOUT, "bss_by_bin.csv"), index=False, float_format="%.4f")
    bss_figure(B)
    print(f"Wrote figures to {OUT} and tables to {TOUT}")


if __name__ == "__main__":
    main()
