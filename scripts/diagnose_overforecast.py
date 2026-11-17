#!/usr/bin/env python3
"""Step 4a: why do some baseline logistic models over-forecast at high probabilities?

Uses the out-of-fold predictions of the full baseline model (|grad| + hour + day of year) from
fit_baseline_models.py; 2015-2025 only, the 2026 holdout is not touched.

1. Top-end calibration for every logistic model: among the hours with the highest TOP_FRAC of
   forecast probabilities (at least MIN_TOP hours), the observed event rate / mean forecast
   probability. A ratio below FLAG_RATIO = over-forecast at the top end.
2. For flagged models with enough hits and false alarms in that top group: which variables separate
   the hits (forecast high, event happened) from the false alarms (forecast high, no event)?
   True misses (event happened, forecast low) are not examined here: they concern under-forecasting.
   Separation is the AUC of each
   variable for hit vs false alarm (0.5 = no separation). Candidates:
     own |gradient|, n_reporting (stations reporting), observed set-mean wind direction and its
     station agreement (wdir_r), hour, day of year, and the signed gradient of every OTHER pair.
   Circular variables (direction, hour, day of year) use cos(angle - mean angle of the hits), so
   their AUC is slightly optimistic. noise_auc = 2 standard errors of an AUC under no separation.

-> derived/models/diagnostics/top_calibration.csv   all logistic models
-> derived/models/diagnostics/separation.csv        flagged models x candidate variables
-> images/models/diagnostics/<signed_gradient>_<variable>_<T>.png, separation_overview.png

Usage:
  python diagnose_overforecast.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DER = os.path.normpath(os.path.join(HERE, "..", "data", "v2", "derived"))
MDIR = os.path.join(DER, "models")
OUT = os.path.join(MDIR, "diagnostics")
IMG = os.path.normpath(os.path.join(HERE, "..", "images", "models", "diagnostics"))
FULL = "pred_grad+hour+doy"
TOP_FRAC, MIN_TOP = 0.02, 100
FLAG_RATIO = 0.85
MIN_HITS = MIN_FALSE_ALARMS = 20
MIN_COVERAGE = 0.8  # other gradients: share of top hours with a value

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
HIT, FA = "#2a78d6", "#eb6834"


def auc(y, x):
    ok = ~np.isnan(x)
    y, x = y[ok], x[ok]
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 < 5 or n0 < 5:
        return np.nan
    r = pd.Series(x).rank().to_numpy()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def circ_score(deg, hits, period=360.0):
    """cos(angle - circular mean angle of the hits): high = close to where hits cluster."""
    a = 2 * np.pi * deg / period
    ok = ~np.isnan(a)
    m = np.arctan2(np.nanmean(np.sin(a[hits & ok])), np.nanmean(np.cos(a[hits & ok])))
    return np.cos(a - m), (np.degrees(m) % 360) * period / 360


def style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(length=0)


def top_group(o):
    n = max(MIN_TOP, int(np.ceil(TOP_FRAC * len(o))))
    return o.nlargest(n, FULL)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(IMG, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    O = pd.read_parquet(os.path.join(MDIR, "baseline", "oof_predictions.parquet"))
    O = O[O.kind == "logistic"]
    ts = pd.read_parquet(os.path.join(DER, "gradients", "gradient_timeseries.parquet"),
                         columns=["gradient", "time_utc", "gradient_hpa"])
    ts["hour"] = ts.time_utc.dt.floor("h")
    G = ts.pivot_table(index="hour", columns="gradient", values="gradient_hpa", aggfunc="first")
    G.index = G.index.astype("datetime64[ms, UTC]")
    targets = {v: pd.read_parquet(os.path.join(MDIR, "targets", f"{v}_train.parquet"),
                                  columns=["signed_gradient", "time_utc", "abs_gradient_hpa", "n_reporting",
                                           "wdir_mean_deg", "wdir_r", "hour_utc", "doy"])
               for v in ("sustained", "gust")}

    cal, sep = [], []
    for (sg, var, T), o in O.groupby(["signed_gradient", "variable", "threshold_mph"]):
        top = top_group(o)
        hits = int(top.obs.sum())
        ratio = top.obs.mean() / top[FULL].mean()
        flagged = ratio < FLAG_RATIO and hits >= MIN_HITS and len(top) - hits >= MIN_FALSE_ALARMS
        cal.append({"signed_gradient": sg, "variable": var, "threshold_mph": T, "n_hours": len(o),
                    "top_hours": len(top), "top_min_prob": top[FULL].min(), "top_mean_prob": top[FULL].mean(),
                    "top_event_rate": top.obs.mean(), "ratio": ratio, "hits": hits,
                    "false_alarms": len(top) - hits, "flagged": flagged})
        if not flagged:
            continue
        t = targets[var]
        d = top.merge(t[t.signed_gradient == sg].drop(columns="signed_gradient"), on="time_utc", how="left")
        d["hour"] = d.time_utc.dt.floor("h")
        d = d.join(G, on="hour")
        y = d.obs.to_numpy() > 0.5
        n1, n0 = y.sum(), (~y).sum()
        noise = 2 * np.sqrt((n1 + n0 + 1) / (12 * n1 * n0))
        cand = {"own |gradient|": d.abs_gradient_hpa.to_numpy(float),
                "stations reporting": d.n_reporting.to_numpy(float),
                "wind-direction agreement": d.wdir_r.to_numpy(float)}
        circ = {}
        for name, col, period in (("wind direction", "wdir_mean_deg", 360.0), ("hour of day", "hour_utc", 24.0),
                                  ("day of year", "doy", 365.25)):
            cand[name], circ[name] = circ_score(d[col].to_numpy(float), y, period)
        own = sg[:-1]
        for g in G.columns:
            if g != own and d[g].notna().mean() >= MIN_COVERAGE:
                cand[f"gradient {g}"] = d[g].to_numpy(float)
        rows = []
        for name, x in cand.items():
            a = auc(y.astype(int), x)
            rows.append({"signed_gradient": sg, "variable": var, "threshold_mph": T, "candidate": name,
                         "auc": a, "separation": abs(a - 0.5) if not np.isnan(a) else np.nan,
                         "higher_in": "hits" if a > 0.5 else "false_alarms", "noise_auc": noise,
                         "hits_center": circ.get(name), "hits": n1, "false_alarms": n0})
        rows = pd.DataFrame(rows).sort_values("separation", ascending=False)
        sep.append(rows)
        figure(sg, var, T, d, y, rows, circ, ratio, noise)

    cal = pd.DataFrame(cal).round(4)
    cal.to_csv(os.path.join(OUT, "top_calibration.csv"), index=False)
    sep = pd.concat(sep).round(4)
    sep.to_csv(os.path.join(OUT, "separation.csv"), index=False)
    overview(sep)
    with open(os.path.join(OUT, "diagnostics_meta.json"), "w") as f:
        json.dump({"model": FULL, "top_frac": TOP_FRAC, "min_top": MIN_TOP, "flag_ratio": FLAG_RATIO,
                   "min_hits": MIN_HITS, "min_false_alarms": MIN_FALSE_ALARMS,
                   "created_utc": pd.Timestamp.now(tz="UTC").isoformat()}, f, indent=2)
    print(f"{cal.flagged.sum()} of {len(cal)} logistic models flagged; wrote {OUT} and {IMG}")


def figure(sg, var, T, d, y, rows, circ, ratio, noise):
    fig = plt.figure(figsize=(14, 10), facecolor=SURFACE)
    # A: separation ranking
    ax = fig.add_subplot(2, 2, 1)
    style(ax)
    r = rows.head(12).iloc[::-1]
    ax.barh(r.candidate, r.auc - 0.5, color=[HIT if a > 0.5 else FA for a in r.auc], height=0.6, zorder=3)
    ax.axvspan(-noise, noise, color=GRID, alpha=0.7, zorder=1)
    ax.axvline(0, color=INK2, lw=0.8)
    lim = max(0.35, np.nanmax(np.abs(r.auc - 0.5)) * 1.1)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("AUC − 0.5   (right: higher in hits, left: higher in false alarms)")
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title("A  What separates hits from false alarms (top 12)", loc="left", fontsize=11, color=INK)
    ax.text(0.99, 0.02, "grey band: noise (±2 SE)", transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
    # B: wind direction rose
    ax = fig.add_subplot(2, 2, 2, projection="polar")
    ax.set_facecolor(SURFACE)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    bins = np.radians(np.arange(0, 361, 22.5))
    for mask, c, lab in ((y, HIT, "hits"), (~y, FA, "false alarms")):
        w = d.wdir_mean_deg.to_numpy(float)[mask]
        w = w[~np.isnan(w)]
        h, _ = np.histogram(np.radians(w), bins=bins)
        h = h / max(h.sum(), 1)
        ax.plot(np.r_[bins[:-1] + np.diff(bins) / 2, bins[0] + np.diff(bins)[0] / 2], np.r_[h, h[0]],
                color=c, lw=2, label=f"{lab} (n={mask.sum()})")
    ax.tick_params(labelsize=8, colors=INK2)
    ax.grid(color=GRID)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=False, fontsize=8)
    ax.set_title("B  Observed set-mean wind direction (fraction)", loc="left", fontsize=11, color=INK, pad=18)
    # C: month
    ax = fig.add_subplot(2, 2, 3)
    style(ax)
    mon = d.time_utc.dt.month.to_numpy()
    for k, (mask, c, lab) in enumerate(((y, HIT, "hits"), (~y, FA, "false alarms"))):
        cnt = np.bincount(mon[mask], minlength=13)[1:]
        ax.bar(np.arange(1, 13) + (k - 0.5) * 0.38, cnt, width=0.36, color=c, label=lab, zorder=3)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(list("JFMAMJJASOND"))
    ax.set_ylabel("Hours")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("C  Month of top-forecast hours", loc="left", fontsize=11, color=INK)
    # D: best other gradient, or own gradient
    ax = fig.add_subplot(2, 2, 4)
    style(ax)
    best = rows[rows.candidate.str.startswith("gradient ")].head(1)
    name = best.candidate.iloc[0] if len(best) else None
    if name:
        g = name.split(" ", 1)[1]
        x = d[g].to_numpy(float)
        lo, hi = np.nanpercentile(x, [1, 99])
        b = np.linspace(lo, hi, 25)
        for mask, c, lab in ((y, HIT, "hits"), (~y, FA, "false alarms")):
            ax.hist(x[mask & ~np.isnan(x)], bins=b, density=True, histtype="step", lw=2, color=c, label=lab)
        ax.set_xlabel(f"{g} gradient (hPa, signed)")
        ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"D  Best-separating other gradient: {g} (AUC {best.auc.iloc[0]:.2f})", loc="left",
                     fontsize=11, color=INK)
    fig.suptitle(f"{sg} {var} ≥{T} mph: top-forecast hours", x=0.01, ha="left", fontsize=14, color=INK)
    fig.text(0.01, 0.945, f"Highest {len(y):,} forecasts (full model, out of fold): observed rate is "
             f"{ratio:.2f}× the forecast. Hits = event happened ({y.sum()}), false alarms = no event ({(~y).sum()}).",
             fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(IMG, f"{sg}_{var}_{T}.png"), dpi=115, facecolor=SURFACE)
    plt.close(fig)


def overview(sep):
    fam = sep.copy()
    fam["family"] = np.where(fam.candidate.str.startswith("gradient "), "best other gradient", fam.candidate)
    piv = fam.groupby(["signed_gradient", "variable", "threshold_mph", "family"]).separation.max().unstack()
    noise = sep.groupby(["signed_gradient", "variable", "threshold_mph"]).noise_auc.first()
    cols = ["own |gradient|", "best other gradient", "wind direction", "wind-direction agreement",
            "day of year", "hour of day", "stations reporting"]
    piv = piv.reindex(columns=cols)
    labels = [f"{s} {v[:4]}. ≥{t}" for s, v, t in piv.index]
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(piv) + 2), facecolor=SURFACE)
    im = ax.imshow(piv.to_numpy(), cmap="Blues", vmin=0, vmax=0.35, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if not np.isnan(v):
                weak = v < noise.iloc[i]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color=INK2 if weak else (SURFACE if v > 0.2 else INK),
                        fontstyle="italic" if weak else "normal")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("|AUC − 0.5| for hits vs false alarms", color=INK2, fontsize=9)
    fig.suptitle("What separates hits from false alarms in the top-forecast hours", x=0.02, ha="left",
                 fontsize=12, color=INK)
    fig.text(0.02, 1 - 0.9 / (0.32 * len(piv) + 2), "Over-forecasting models only. Grey italic = within noise (±2 SE).",
             fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 1 - 1.0 / (0.32 * len(piv) + 2)))
    fig.savefig(os.path.join(IMG, "separation_overview.png"), dpi=130, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
