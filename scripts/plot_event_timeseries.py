#!/usr/bin/env python3
"""Event time series: observed set targets vs GBT (gbt_all) speed forecasts for several gradient pairs.

For each pair, both signed sets are combined into one series (each hour belongs to the set matching the
gradient's sign; the two signs have different station sets, so the target switches set when the sign
changes). Forecasts are the out-of-fold ones, i.e. for 2025 from models trained without 2025.

Per pair, three stacked panels:
  1  gust and sustained targets (solid) and GBT forecasts (dashed), mph
  2  error = forecast - obs (mph), gust and sustained
  3  signed gradient (hPa), shaded by sign; hours with no model for that sign are marked
Panel titles: window bias / RMSE and peak obs vs peak forecast (gust).
Plus a summary figure: peak obs vs peak forecast and gust bias / RMSE by signed set.

-> images/case_studies/events/<YYYYMMDD>/timeseries.png, summary.png
-> data/v2/derived/models/case_studies/event_<YYYYMMDD>_stats.csv

Usage:
  python plot_event_timeseries.py                                   # Jan 2025 Santa Ana, SoCal pairs
  python plot_event_timeseries.py --start "2025-01-06 04:00" --end "2025-01-10 04:00" --pairs LAX-DAG SAN-IPL
  (times are local standard time, PST = UTC-8)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_baseline_models import GRID, HALO, INK, INK2, SURFACE, THR_COLORS, style

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data", "v2"))
MDIR = os.path.join(DATA, "derived", "models")
SOCAL = ["LAX-DAG", "LAX-TPH", "LAX-PMD", "LAX-WJF", "LAX-BFL", "SBA-SMX", "SAN-LAS", "SAN-IPL"]
C_GUST, C_SUS = THR_COLORS[0], THR_COLORS[1]
C_POS, C_NEG = "#3987e5", "#e34948"            # diverging pair for the gradient sign
PST = pd.Timedelta(hours=8)
MARKS = [("2025-01-07 10:30", "Palisades fire"), ("2025-01-07 18:18", "Eaton fire")]   # local times


def load(pairs, t0, t1):
    sets = [p + s for p in pairs for s in "+-"]
    P = pd.read_parquet(os.path.join(MDIR, "gbt", "oof_predictions.parquet"),
                        columns=["signed_gradient", "variable", "time_utc", "obs", "pred_gbt_all"],
                        filters=[("kind", "==", "linear"), ("signed_gradient", "in", sets),
                                 ("time_utc", ">=", t0), ("time_utc", "<", t1)])
    G = pd.read_parquet(os.path.join(DATA, "derived", "gradients", "gradient_timeseries.parquet"),
                        columns=["gradient", "time_utc", "gradient_hpa"],
                        filters=[("gradient", "in", pairs), ("time_utc", ">=", t0), ("time_utc", "<", t1)])
    for d in (P, G):
        d["t"] = (d.time_utc - PST).dt.tz_localize(None)
    P["pair"] = P.signed_gradient.str[:-1]
    P["err"] = P.pred_gbt_all - P.obs
    return P, G


def stats(P):
    rows = []
    for (sg, v), g in P.groupby(["signed_gradient", "variable"]):
        rows.append({"signed_gradient": sg, "variable": v, "hours": len(g), "bias_mph": g.err.mean(),
                     "rmse_mph": np.sqrt((g.err ** 2).mean()), "peak_obs_mph": g.obs.max(),
                     "peak_fc_mph": g.pred_gbt_all.max(), "worst_err_mph": g.err.min(),
                     "worst_err_time_pst": g.loc[g.err.idxmin(), "t"]})
    return pd.DataFrame(rows)


def marks(ax, label=False):
    for t, name in MARKS:
        ax.axvline(pd.Timestamp(t), color=INK2, lw=0.8, ls=(0, (2, 2)), zorder=1)
        if label:
            ax.annotate(name, (pd.Timestamp(t), 1), xycoords=("data", "axes fraction"), xytext=(3, -3),
                        textcoords="offset points", fontsize=7.5, color=INK2, va="top")


def pair_panels(fig, cell, pair, P, G, t0, t1):
    inner = cell.subgridspec(3, 1, height_ratios=[3, 1.5, 1.2], hspace=0.08)
    ax, ex, gx = (fig.add_subplot(inner[i]) for i in range(3))
    for a in (ax, ex, gx):
        style(a)
        a.set_xlim(t0, t1)
    p = P[P.pair == pair]
    for v, c in (("gust", C_GUST), ("sustained", C_SUS)):
        for sg, g in p[p.variable == v].groupby("signed_gradient"):
            g = g.set_index("t").sort_index().asfreq("h") if len(g) else g   # gaps break the line
            ax.plot(g.index, g.obs, color=c, lw=1.8, zorder=4)
            ax.plot(g.index, g.pred_gbt_all, color=c, lw=1.6, ls=(0, (5, 2)), zorder=5)
            ex.plot(g.index, g.err, color=c, lw=1.5, zorder=4)
    ex.axhline(0, color=INK2, lw=0.8, zorder=2)
    ex.set_ylabel("fc − obs\n(mph)", fontsize=8)
    ax.set_ylabel("mph", fontsize=8.5)
    g = G[G.gradient == pair].sort_values("t")
    gx.fill_between(g.t, 0, g.gradient_hpa, where=g.gradient_hpa >= 0, color=C_POS, alpha=0.35, lw=0, step="mid")
    gx.fill_between(g.t, 0, g.gradient_hpa, where=g.gradient_hpa < 0, color=C_NEG, alpha=0.35, lw=0, step="mid")
    gx.plot(g.t, g.gradient_hpa, color=INK2, lw=1.1, drawstyle="steps-mid")
    gx.axhline(0, color=INK2, lw=0.6)
    gx.set_ylabel("grad\n(hPa)", fontsize=8)
    have = set(p.signed_gradient.unique())
    nomodel = [s for s in ("+", "-") if pair + s not in have]
    for a in (ax, ex, gx):
        marks(a, label=a is ax)
    for a in (ax, ex):
        a.tick_params(axis="x", labelbottom=False)
    gx.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    gx.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    gx.tick_params(labelsize=7.5)
    gu = p[p.variable == "gust"]
    su = p[p.variable == "sustained"]
    parts = [pair]
    if len(gu):
        parts.append(f"gust bias {gu.err.mean():+.1f}, RMSE {np.sqrt((gu.err ** 2).mean()):.1f}; "
                     f"peak obs {gu.obs.max():.0f} vs fc {gu.pred_gbt_all.max():.0f}")
    if len(su):
        parts.append(f"sust bias {su.err.mean():+.1f}")
    ax.set_title(" · ".join(parts), loc="left", fontsize=9.5, color=INK)
    if nomodel:
        gx.text(0.995, 0.92, "no model for " + " / ".join(f"{pair}{s}" for s in nomodel) + " hours",
                transform=gx.transAxes, ha="right", va="top", fontsize=7, color=INK2, path_effects=HALO)


def summary(S, out, title):
    g = S[S.variable == "gust"].sort_values("bias_mph").reset_index(drop=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 0.42 * len(g) + 2.2), facecolor=SURFACE, sharey=True)
    y = np.arange(len(g))
    style(a1)
    a1.hlines(y, g.peak_fc_mph, g.peak_obs_mph, color=GRID, lw=2.5, zorder=2)
    a1.scatter(g.peak_obs_mph, y, s=60, color=C_GUST, edgecolor=SURFACE, lw=1.2, zorder=4, label="peak observed target")
    a1.scatter(g.peak_fc_mph, y, s=60, color="#9a998f", edgecolor=SURFACE, lw=1.2, zorder=4, label="peak GBT forecast")
    a1.set_yticks(y, [f"{s} ({h} h)" for s, h in zip(g.signed_gradient, g.hours)], fontsize=8.5)
    a1.set_xlabel("Gust (mph)", fontsize=9)
    a1.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK2)
    a1.set_title("Peak gust target vs peak forecast in the window", loc="left", fontsize=10.5, color=INK)
    style(a2)
    a2.barh(y - 0.2, g.bias_mph, 0.38, color=C_NEG, label="bias (fc − obs)", zorder=3)
    a2.barh(y + 0.2, g.rmse_mph, 0.38, color="#9a998f", label="RMSE", zorder=3)
    a2.axvline(0, color=INK2, lw=0.8)
    a2.set_xlabel("mph", fontsize=9)
    a2.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK2)
    a2.set_title("Gust bias and RMSE over the window", loc="left", fontsize=10.5, color=INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-01-06 04:00", help="local standard time (PST)")
    ap.add_argument("--end", default="2025-01-10 04:00")
    ap.add_argument("--pairs", nargs="*", default=SOCAL)
    a = ap.parse_args()
    t0, t1 = pd.Timestamp(a.start), pd.Timestamp(a.end)
    u0, u1 = ((t + PST).tz_localize("UTC") for t in (t0, t1))
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    tag = f"{t0:%Y%m%d}"
    out = os.path.normpath(os.path.join(HERE, "..", "images", "case_studies", "events", tag))
    tout = os.path.join(MDIR, "case_studies")
    os.makedirs(out, exist_ok=True)
    os.makedirs(tout, exist_ok=True)
    P, G = load(a.pairs, u0, u1)
    S = stats(P)
    S.to_csv(os.path.join(tout, f"event_{tag}_stats.csv"), index=False, float_format="%.2f")
    n = len(a.pairs)
    rows = (n + 1) // 2
    fig = plt.figure(figsize=(20, 6.2 * rows + 1.6), facecolor=SURFACE)
    outer = fig.add_gridspec(rows, 2, hspace=0.3, wspace=0.12, left=0.045, right=0.99,
                             top=1 - 1.25 / fig.get_figheight(), bottom=0.5 / fig.get_figheight())
    for i, pair in enumerate(a.pairs):
        pair_panels(fig, outer[i // 2, i % 2], pair, P, G, t0, t1)
    fig.suptitle(f"GBT speed forecasts vs observed set targets, {t0:%b %d %H:%M} – {t1:%b %d %H:%M} PST {t0:%Y}",
                 x=0.01, ha="left", y=1 - 0.25 / fig.get_figheight(), fontsize=15, color=INK)
    fig.text(0.01, 1 - 0.62 / fig.get_figheight(),
             "Targets: sustained = set P75, gust = set P90 (k-th highest). Out-of-fold gbt_all forecasts (2025 predicted by "
             "models trained without 2025). Each sign of a pair has its own station set; the gradient panel is shaded "
             "blue (+) / red (−). Dotted lines: fire ignitions.", fontsize=9, color=INK2)
    fig.legend([Line2D([], [], color=C_GUST, lw=1.8), Line2D([], [], color=C_GUST, lw=1.6, ls=(0, (5, 2))),
                Line2D([], [], color=C_SUS, lw=1.8), Line2D([], [], color=C_SUS, lw=1.6, ls=(0, (5, 2)))],
               ["gust target", "gust forecast", "sustained target", "sustained forecast"],
               loc="upper right", bbox_to_anchor=(0.99, 1 - 0.15 / fig.get_figheight()), ncol=4, frameon=False,
               fontsize=9.5, labelcolor=INK2)
    fig.savefig(os.path.join(out, "timeseries.png"), dpi=100, facecolor=SURFACE)
    plt.close(fig)
    summary(S, os.path.join(out, "summary.png"),
            f"Gust forecast performance by signed set, {t0:%b %d} – {t1:%b %d} {t0:%Y} (PST)")
    print(S.round(1).to_string(index=False))
    print(f"Wrote figures to {out}")


if __name__ == "__main__":
    main()
