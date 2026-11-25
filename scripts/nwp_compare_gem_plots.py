"""Figures for the GEM vs GFS vs ECMWF comparison.
1. by-lead figure from nwp_compare_gem.py's common-row table (ECMWF native times, all three corrected signs):
   RMSE of GBT and bias-corrected raw wind per driving model, and GBT-GEM minus GBT-ECMWF / GBT-GFS with 95% CIs.
   -> images/verification/model_comparison/gem_vs_gfs_ecmwf_by_lead.png
2. side-by-side event plots: GBT driven by GEM vs by ECMWF SLP, days 1 (solid) and 5 (dashed), each at its own native
   times; same top-5 event days per set as nwp_verify.py. Window RMSE is on common times (ECMWF native).
   -> images/verification/model_comparison/gem_vs_gbt_ecmwf_events/<set>.png
   -> derived/verification/model_comparison/gem_vs_gbt_ecmwf_event_windows.csv
Usage: python nwp_compare_gem_plots.py"""
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
IMG = f"{IMGC}/gem_vs_gbt_ecmwf_events"
os.makedirs(IMG, exist_ok=True)
INK, MUTED, GRID = "#0b0b0b", "#8a8986", "#e5e4e0"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
MOD = {"gfs": ("GFS", ORANGE, "s"), "ecmwf": ("ECMWF", BLUE, "o"), "gem": ("GEM", VIOLET, "D")}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "font.size": 9, "axes.titlesize": 10, "legend.frameon": False, "lines.linewidth": 2})

# ---- 1. by-lead comparison ----
R = pd.read_csv(f"{D}/model_comparison/gem_vs_gfs_ecmwf_common_native.csv")
ci = lambda s: np.array([[float(v) for v in x.strip("[]").split(",")] for x in s])
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
for i, grp in enumerate(("all", "offshore")):
    x = R[R.group == grp].sort_values("lead")
    ax = axes[i, 0]
    for m, (lab, col, mk) in MOD.items():
        ax.plot(x.lead, x[f"gbt_nwp_corr_{m}"], color=col, marker=mk, ms=5, label=f"GBT, {lab} SLP")
        ax.plot(x.lead, x[f"nwp_bc_{m}"], color=col, ls="--", lw=1.3, marker=mk, ms=4, mfc="white",
                label=f"bias-corrected raw {lab}")
    ax.set(title=f"RMSE by lead day, common rows ({'all sets' if grp == 'all' else 'offshore sets'})",
           xlabel="lead day", ylabel="RMSE (mph)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7.5, ncol=2, loc="lower right")
    ax = axes[i, 1]
    for other, ls in (("ecmwf", "-"), ("gfs", "--")):
        col = f"gbt_gem-gbt_{other}"
        b = ci(x[f"ci {col}"])
        lab, c, mk = MOD[other]
        ax.fill_between(x.lead, b[:, 0], b[:, 1], color=c, alpha=0.15, lw=0)
        ax.plot(x.lead, x[col], color=c, ls=ls, marker=mk, ms=5, label=f"GBT-GEM minus GBT-{lab}")
    b = ci(x["ci gbt_gem-nwp_bc_gem"])
    ax.fill_between(x.lead, b[:, 0], b[:, 1], color=VIOLET, alpha=0.12, lw=0)
    ax.plot(x.lead, x["gbt_gem-nwp_bc_gem"], color=VIOLET, ls=":", marker="D", ms=4,
            label="GBT-GEM minus bias-corrected GEM")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set(title="RMSE difference, 95% day-block bootstrap CI (below 0 = GBT-GEM better)", xlabel="lead day",
           ylabel="mph")
    ax.legend(fontsize=8)
for ax in axes.flat:
    ax.set_xticks(range(8))
fig.suptitle("GEM vs GFS vs ECMWF, 2026 holdout: times native to all three models (ECMWF's native set), "
             "rows selected by all three corrected signs", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(f"{IMGC}/gem_vs_gfs_ecmwf_by_lead.png", dpi=110, bbox_inches="tight")
plt.close(fig)

# ---- 2. event plots ----
LEADS = (1, 5)
cols = ["signed_gradient", "valid_utc", "lead_day", "lead_h", "target_mph", "gbt_obs", "gbt_nwp_corr"]
F = {}
for m in ("ecmwf", "gem"):
    x = pd.read_parquet(f"{D}/{m}/forecasts_sustained.parquet", columns=cols, filters=[("lead_day", "in", LEADS)])
    F[m] = x[C.is_native(C.MODELS[m], x.lead_h.to_numpy())]
OBS = pd.read_parquet(f"{D}/gem/forecasts_sustained.parquet", columns=cols, filters=[("lead_day", "==", 1)])
EV = pd.read_csv(f"{D}/gem/scores_native/event_days.csv", parse_dates=["peak_valid_utc"])
lt = lambda s: s.dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
rows = []
for sg, evs in EV.groupby("signed_gradient"):
    fig, axes = plt.subplots(len(evs), 1, figsize=(11, 2.6 * len(evs)), squeeze=False)
    for ax, (_, e) in zip(axes[:, 0], evs.sort_values("rank").iterrows()):
        t0, t1 = e.peak_valid_utc - pd.Timedelta(hours=24), e.peak_valid_utc + pd.Timedelta(hours=24)
        sel = lambda d: d[(d.signed_gradient == sg) & d.valid_utc.between(t0, t1)]
        o = sel(OBS).sort_values("valid_utc")
        ax.plot(lt(o.valid_utc), o.target_mph, color=INK, lw=2.2, label="observed (set P75, hourly)")
        ax.plot(lt(o.valid_utc), o.gbt_obs, color=AQUA, lw=1.2, ls=":", label="GBT, observed SLP")
        for k, ls, lw in ((LEADS[0], "-", 2), (LEADS[1], "--", 1.3)):
            ek = sel(F["ecmwf"])[lambda d: d.lead_day == k].sort_values("valid_utc")
            gk = sel(F["gem"])[lambda d: d.lead_day == k].sort_values("valid_utc")
            for m, d in (("ecmwf", ek), ("gem", gk)):
                lab, c, mk = MOD[m]
                ax.plot(lt(d.valid_utc), d.gbt_nwp_corr, color=c, ls=ls, lw=lw, marker=mk, ms=2.5,
                        label=f"GBT, {lab} SLP day {k}")
            mm = ek.merge(gk[["valid_utc", "gbt_nwp_corr"]], on="valid_utc", suffixes=("_ec", "_gem"))
            mm = mm.dropna(subset=["target_mph"])
            rows.append(dict(signed_gradient=sg, rank=e["rank"], lead_day=k, n=len(mm), peak_obs=e.peak_obs_mph,
                             rmse_gbt_ecmwf=np.sqrt(((mm.gbt_nwp_corr_ec - mm.target_mph) ** 2).mean()),
                             rmse_gbt_gem=np.sqrt(((mm.gbt_nwp_corr_gem - mm.target_mph) ** 2).mean()),
                             peak_gbt_ecmwf=ek.gbt_nwp_corr.max(), peak_gbt_gem=gk.gbt_nwp_corr.max()))
        ax.set_title(f"#{e['rank']}  {e.local_day}  peak observed {e.peak_obs_mph:.0f} mph", loc="left")
        ax.set_ylabel("mph")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
        ax.set_ylim(bottom=0)
    axes[0, 0].legend(ncol=3, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.45))
    fig.suptitle(f"{sg} sustained wind: GBT driven by GEM vs ECMWF SLP, top 5 event days (time PT; each model at its "
                 f"native times; no sign selection)", x=0.01, ha="left", y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{IMG}/{sg}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
W = pd.DataFrame(rows)
W.round(2).to_csv(f"{D}/model_comparison/gem_vs_gbt_ecmwf_event_windows.csv", index=False)
s = W.groupby("lead_day")[["rmse_gbt_ecmwf", "rmse_gbt_gem"]].median().round(2)
w = W.assign(gem_better=W.rmse_gbt_gem < W.rmse_gbt_ecmwf).groupby("lead_day").gem_better.agg(["sum", "size"])
print("median event-window RMSE (common ECMWF-native times):"); print(s.join(w).to_string())
print(f"wrote {IMGC}/gem_vs_gfs_ecmwf_by_lead.png and {IMG}/ ({W.signed_gradient.nunique()} sets)")
