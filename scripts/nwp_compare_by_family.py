#!/usr/bin/env python
"""Compare forecast families: raw NWP wind vs bias-corrected NWP wind vs GBT driven by NWP SLP.

Reads the scores already written by nwp_compare_nbm.py (no recomputation):
  model_comparison/nbm_vs_guidance_scores.csv       common rows native to NBM/GFS/ECMWF/GEM, days 0-7
  model_comparison/nbm_vs_guidance_hrrr_scores.csv  the subset that also has HRRR, days 0-1

The HRRR markers (days 0-1) are scored on the HRRR subset, which is about 4% smaller than the common rows.
NBM has no GBT member because it is wind-only.

Outputs:
  images/verification/model_comparison/rmse_by_family.png
  images/verification/model_comparison/bias_by_family.png
  data/v2/derived/verification/model_comparison/rmse_by_family_table.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("/mnt/d/socal")
CMP = ROOT / "data/v2/derived/verification/model_comparison"
IMG = ROOT / "images/verification/model_comparison"

FAMILIES = [("raw", "Raw model wind"), ("bc", "Bias-corrected model wind"), ("gbt", "GBT driven by model SLP")]
MODELS = {"nbm": ("NBM", "#444444", "^"), "gfs": ("GFS", "#e8603c", "s"), "ecmwf": ("ECMWF", "#2b7bd6", "o"),
          "gem": ("GEM", "#6a4fd1", "D"), "hrrr": ("HRRR", "#2a9d5c", "v")}
FAM_STYLE = {"raw": ("#9a9a9a", ":"), "bc": ("#555555", "--"), "gbt": ("#2b7bd6", "-")}
GROUPS = [("all", "all"), ("offshore", "offshore"), (">=30 mph obs", ">=30 mph obs"),
          ("all, 00/12Z (equal lead)", "all, 00/12Z (equal lead)")]


def split(df):
    df = df.copy()
    df[["family", "model"]] = df["forecast"].str.split("_", n=1, expand=True)
    return df


def main():
    s = split(pd.read_csv(CMP / "nbm_vs_guidance_scores.csv"))
    h = split(pd.read_csv(CMP / "nbm_vs_guidance_hrrr_scores.csv"))
    h = h[h.model == "hrrr"]

    # Best member of each family per group/lead on the common rows (HRRR excluded: it covers days 0-1 only).
    rows = []
    for (g, lead, fam), d in s.groupby(["group", "lead_day", "family"]):
        b = d.loc[d.rmse.idxmin()]
        rows.append(dict(group=g, lead_day=lead, family=fam, best_model=b.model, rmse=b.rmse, bias=b.bias, n=b.n))
    best = pd.DataFrame(rows)
    table = best.pivot_table(index=["group", "lead_day"], columns="family",
                             values=["rmse", "best_model"], aggfunc="first")
    table.columns = [f"{fam}_{v}" for v, fam in table.columns]
    table = table[[f"{f}_{v}" for f, _ in FAMILIES for v in ("best_model", "rmse")]]
    table.to_csv(CMP / "rmse_by_family_table.csv")
    with pd.option_context("display.width", 200):
        print(table.to_string())

    for metric, fname in (("rmse", "rmse_by_family.png"), ("bias", "bias_by_family.png")):
        fig, axes = plt.subplots(len(GROUPS), 4, figsize=(19, 4.1 * len(GROUPS)), sharex=True)
        for r, (g, gtitle) in enumerate(GROUPS):
            sg, hg = s[s.group == g], h[h.group == g]
            n = sg.groupby("lead_day").n.first()
            for c, (fam, ftitle) in enumerate(FAMILIES):
                ax = axes[r, c]
                for m, (lab, col, mk) in MODELS.items():
                    if m == "hrrr":
                        d = hg[hg.family == fam]
                        if len(d):
                            ax.plot(d.lead_day, d[metric], color=col, marker=mk, ms=8, lw=1.5,
                                    label=f"{lab} (HRRR rows)")
                        continue
                    d = sg[(sg.family == fam) & (sg.model == m)].sort_values("lead_day")
                    if len(d):
                        ax.plot(d.lead_day, d[metric], color=col, marker=mk, lw=2, label=lab)
                if r == 0:
                    ax.set_title(ftitle, fontsize=12, fontweight="bold")
                ax.legend(fontsize=8, loc="upper left" if metric == "rmse" else "lower left")
            ax = axes[r, 3]
            for fam, ftitle in FAMILIES:
                d = best[(best.group == g) & (best.family == fam)].sort_values("lead_day")
                col, ls = FAM_STYLE[fam]
                ax.plot(d.lead_day, d[metric], color=col, ls=ls, marker="o", lw=2.4, label=f"best {fam}")
                for _, row in d.iterrows():
                    ax.annotate(MODELS[row.best_model][0][:2], (row.lead_day, row[metric]), fontsize=6.5,
                                color=col, xytext=(0, 5), textcoords="offset points", ha="center")
            if r == 0:
                ax.set_title("Best member of each family", fontsize=12, fontweight="bold")
            ax.legend(fontsize=8, loc="upper left" if metric == "rmse" else "lower left")
            axes[r, 0].set_ylabel(f"{gtitle}\n{metric.upper()} (mph)\nn {n.min():,}-{n.max():,} per lead", fontsize=10)
            ys = [a.get_ylim() for a in axes[r]]
            lo, hi = min(y[0] for y in ys), max(y[1] for y in ys)
            for a in axes[r]:
                a.set_ylim(0 if metric == "rmse" else lo, hi)
                a.grid(alpha=0.3)
                if metric == "bias":
                    a.axhline(0, color="k", lw=0.8)
        for a in axes[-1]:
            a.set_xlabel("lead day")
            a.set_xticks(range(8))
        fig.suptitle(f"Sustained wind {metric.upper()} by forecast family, 2026 holdout. Common rows native to NBM, "
                     f"GFS, ECMWF and GEM; HRRR markers (days 0-1) use the HRRR subset. NBM has no GBT (wind-only). "
                     f"Bias correction is an oracle fit on the holdout.", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(IMG / fname, dpi=110)
        plt.close(fig)
        print("wrote", IMG / fname)


if __name__ == "__main__":
    main()
