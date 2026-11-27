"""Figures for NBM vs all guidance (tables from nwp_compare_nbm.py).
1. images/verification/model_comparison/nbm_vs_guidance_by_lead.png
   rows: all sets / offshore / >=30 mph obs (left: RMSE by lead) ; right: RMSE differences with 95% CIs
   (GBT-<m> minus NBM, bias-corrected NBM minus raw NBM), last right panel = the same at 00/12Z (equal lead).
2. images/verification/model_comparison/nbm_vs_gbt_ecmwf_events/<set>.png
   GBT driven by ECMWF SLP (3-hourly native) vs raw NBM (hourly), days 1 (solid) and 5 (dashed), same top-5 event
   days as nwp_verify.py; window RMSE on common times -> model_comparison/nbm_vs_gbt_ecmwf_event_windows.csv
Usage: python nwp_compare_nbm_plots.py"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
sys.path.insert(0, "/mnt/d/socal/scripts"); import nwp_config as C

ROOT = "/mnt/d/socal"
D = f"{ROOT}/data/v2/derived/verification"
IMGC = f"{ROOT}/images/verification/model_comparison"
IMG = f"{IMGC}/nbm_vs_gbt_ecmwf_events"
os.makedirs(IMG, exist_ok=True)
INK, MUTED, GRID, INK2 = "#0b0b0b", "#8a8986", "#e5e4e0", "#52514e"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
MOD = {"nbm": ("NBM", INK2, "^"), "ecmwf": ("ECMWF", BLUE, "o"), "gem": ("GEM", VIOLET, "D"), "gfs": ("GFS", ORANGE, "s")}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "font.size": 9, "axes.titlesize": 10, "legend.frameon": False, "lines.linewidth": 2})

# ---- 1. by-lead ----
S = pd.read_csv(f"{D}/model_comparison/nbm_vs_guidance_scores.csv")
DF = pd.read_csv(f"{D}/model_comparison/nbm_vs_guidance_diffs.csv")
fig, axes = plt.subplots(3, 2, figsize=(12, 12.5))
for i, grp in enumerate(("all", "offshore", ">=30 mph obs")):
    ax = axes[i, 0]
    s = S[S.group == grp]
    get = lambda f: s[s.forecast == f].sort_values("lead_day")
    for m in ("ecmwf", "gem", "gfs"):
        lab, c, mk = MOD[m]
        x = get(f"gbt_{m}"); ax.plot(x.lead_day, x.rmse, color=c, marker=mk, ms=5, label=f"GBT, {lab} SLP")
        x = get(f"bc_{m}"); ax.plot(x.lead_day, x.rmse, color=c, ls="--", lw=1.3, marker=mk, ms=4, mfc="white",
                                    label=f"bias-corrected raw {lab}")
    lab, c, mk = MOD["nbm"]
    x = get("bc_nbm"); ax.plot(x.lead_day, x.rmse, color=c, ls="--", lw=2, marker=mk, ms=6, mfc="white",
                               label="bias-corrected NBM")
    x = get("raw_nbm"); ax.plot(x.lead_day, x.rmse, color=c, lw=2, marker=mk, ms=6, label="raw NBM")
    n = get("raw_nbm").n
    ax.set(title=f"RMSE by lead day, common rows: {grp} (n {n.min():,}-{n.max():,} per lead)", xlabel="lead day",
           ylabel="RMSE (mph)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7.5, ncol=2, loc="lower right" if i < 2 else "upper left")
STY = {("gbt_ecmwf", "raw_nbm"): (BLUE, ":", "o", "GBT-ECMWF minus raw NBM"),
       ("gbt_ecmwf", "bc_nbm"): (BLUE, "-", "o", "GBT-ECMWF minus bias-corrected NBM"),
       ("gbt_gem", "bc_nbm"): (VIOLET, "-", "D", "GBT-GEM minus bias-corrected NBM"),
       ("gbt_gfs", "bc_nbm"): (ORANGE, "-", "s", "GBT-GFS minus bias-corrected NBM"),
       ("bc_nbm", "bc_ecmwf"): (INK2, "--", "^", "bias-corrected NBM minus bias-corrected ECMWF")}
for i, grp in enumerate(("all", "offshore", "all, 00/12Z (equal lead)")):
    ax = axes[i, 1]
    for (a_, b_), (c, ls, mk, lab) in STY.items():
        x = DF[(DF.group == grp) & (DF.a == a_) & (DF.b == b_)].sort_values("lead_day")
        ax.fill_between(x.lead_day, x.ci_lo, x.ci_hi, color=c, alpha=0.12, lw=0)
        ax.plot(x.lead_day, x["diff"], color=c, ls=ls, marker=mk, ms=5, label=lab)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set(title=f"RMSE difference, 95% day-block CI: {grp} (below 0 = first better)", xlabel="lead day",
           ylabel="mph")
    ax.legend(fontsize=7.5, loc="best", framealpha=0.85, frameon=True, edgecolor="none")
for ax in axes.flat:
    ax.set_xticks(range(8))
fig.suptitle("NBM vs all guidance, 2026 holdout: times native to NBM, GFS, ECMWF and GEM; rows selected by all three "
             "GBTs' corrected signs", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(f"{IMGC}/nbm_vs_guidance_by_lead.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# ---- 2. event plots ----
LEADS = (1, 5)
cols = ["signed_gradient", "valid_utc", "lead_day", "lead_h", "target_mph", "gbt_obs"]
E = pd.read_parquet(f"{D}/ecmwf/forecasts_sustained.parquet", columns=cols + ["gbt_nwp_corr"],
                    filters=[("lead_day", "in", LEADS)])
E = E[C.is_native(C.MODELS["ecmwf"], E.lead_h.to_numpy())]
N = pd.read_parquet(f"{D}/nbm/forecasts_sustained.parquet", columns=cols + ["nwp_p75_rep"],
                    filters=[("lead_day", "in", LEADS)])
EV = pd.read_csv(f"{D}/ecmwf/scores_native/event_days.csv", parse_dates=["peak_valid_utc"])
lt = lambda s: s.dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
rows = []
for sg, evs in EV.groupby("signed_gradient"):
    fig, axes = plt.subplots(len(evs), 1, figsize=(11, 2.6 * len(evs)), squeeze=False)
    for ax, (_, e) in zip(axes[:, 0], evs.sort_values("rank").iterrows()):
        t0, t1 = e.peak_valid_utc - pd.Timedelta(hours=24), e.peak_valid_utc + pd.Timedelta(hours=24)
        sel = lambda d: d[(d.signed_gradient == sg) & d.valid_utc.between(t0, t1)]
        n_, e_ = sel(N), sel(E)
        o = n_[n_.lead_day == LEADS[0]].sort_values("valid_utc")
        ax.plot(lt(o.valid_utc), o.target_mph, color=INK, lw=2.2, label="observed (set P75, hourly)")
        ax.plot(lt(o.valid_utc), o.gbt_obs, color=AQUA, lw=1.2, ls=":", label="GBT, observed SLP")
        for k, ls, lw in ((LEADS[0], "-", 2), (LEADS[1], "--", 1.3)):
            ek = e_[e_.lead_day == k].sort_values("valid_utc"); nk = n_[n_.lead_day == k].sort_values("valid_utc")
            ax.plot(lt(ek.valid_utc), ek.gbt_nwp_corr, color=BLUE, ls=ls, lw=lw, marker="o", ms=2.5,
                    label=f"GBT, ECMWF SLP day {k} (3-hourly)")
            ax.plot(lt(nk.valid_utc), nk.nwp_p75_rep, color=INK2, ls=ls, lw=lw, alpha=0.8,
                    label=f"raw NBM day {k} (hourly)")
            mm = nk.merge(ek[["valid_utc", "gbt_nwp_corr"]], on="valid_utc").dropna(subset=["target_mph", "nwp_p75_rep"])
            rows.append(dict(signed_gradient=sg, rank=e["rank"], lead_day=k, n=len(mm), peak_obs=e.peak_obs_mph,
                             rmse_gbt_ecmwf=np.sqrt(((mm.gbt_nwp_corr - mm.target_mph) ** 2).mean()),
                             rmse_raw_nbm=np.sqrt(((mm.nwp_p75_rep - mm.target_mph) ** 2).mean()),
                             bias_raw_nbm=(mm.nwp_p75_rep - mm.target_mph).mean(),
                             peak_gbt_ecmwf=ek.gbt_nwp_corr.max(), peak_raw_nbm=nk.nwp_p75_rep.max()))
        ax.set_title(f"#{e['rank']}  {e.local_day}  peak observed {e.peak_obs_mph:.0f} mph", loc="left")
        ax.set_ylabel("mph")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
        ax.set_ylim(bottom=0)
    axes[0, 0].legend(ncol=3, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.45))
    fig.suptitle(f"{sg} sustained wind: GBT (ECMWF SLP) vs raw NBM, top 5 event days (time PT; all hours shown, "
                 f"no sign selection)", x=0.01, ha="left", y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{IMG}/{sg}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
W = pd.DataFrame(rows)
W.round(2).to_csv(f"{D}/model_comparison/nbm_vs_gbt_ecmwf_event_windows.csv", index=False)
s = W.groupby("lead_day")[["rmse_gbt_ecmwf", "rmse_raw_nbm", "bias_raw_nbm"]].median().round(2)
w = W.assign(gbt_better=W.rmse_gbt_ecmwf < W.rmse_raw_nbm).groupby("lead_day").gbt_better.agg(["sum", "size"])
print("median event-window RMSE (common 3-hourly times):"); print(s.join(w).to_string())
print(f"wrote {IMGC}/nbm_vs_guidance_by_lead.png and {IMG}/ ({W.signed_gradient.nunique()} sets)")
