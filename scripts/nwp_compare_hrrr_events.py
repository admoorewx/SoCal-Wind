"""Side-by-side event plots: GBT driven by corrected ECMWF pressure vs raw HRRR wind, day 1 (and day 0 dashed).
Event days are the shared top-5 per set from nwp_verify.py (event_days.csv, picked from hourly obs, identical for all
models). ECMWF-driven GBT is drawn at its native 3-hourly times; HRRR is hourly (native).
Outputs: images/verification/model_comparison/hrrr_vs_gbt_ecmwf_events/<set>.png
Usage: python nwp_compare_hrrr_events.py"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
sys.path.insert(0, "/mnt/d/socal/scripts"); import nwp_config as C

ROOT = "/mnt/d/socal"
D = f"{ROOT}/data/v2/derived/verification"
IMG = f"{ROOT}/images/verification/model_comparison/hrrr_vs_gbt_ecmwf_events"
os.makedirs(IMG, exist_ok=True)
INK, MUTED, GRID = "#0b0b0b", "#8a8986", "#e5e4e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "font.size": 9, "axes.titlesize": 10, "legend.frameon": False, "lines.linewidth": 2})
cols = ["signed_gradient", "valid_utc", "lead_day", "lead_h", "target_mph", "gbt_obs", "gbt_nwp_corr", "nwp_p75_rep"]
E = pd.read_parquet(f"{D}/ecmwf/forecasts_sustained.parquet", columns=cols, filters=[("lead_day", "<=", 1)])
H = pd.read_parquet(f"{D}/hrrr/forecasts_sustained.parquet", columns=cols, filters=[("lead_day", "<=", 1)])
E = E[C.is_native(C.MODELS["ecmwf"], E.lead_h.to_numpy())]
EV = pd.read_csv(f"{D}/hrrr/scores_native/event_days.csv", parse_dates=["peak_valid_utc"])
lt = lambda s: s.dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
rows = []
for sg, evs in EV.groupby("signed_gradient"):
    fig, axes = plt.subplots(len(evs), 1, figsize=(11, 2.6 * len(evs)), squeeze=False)
    for ax, (_, e) in zip(axes[:, 0], evs.sort_values("rank").iterrows()):
        t0, t1 = e.peak_valid_utc - pd.Timedelta(hours=24), e.peak_valid_utc + pd.Timedelta(hours=24)
        h = H[(H.signed_gradient == sg) & H.valid_utc.between(t0, t1)]
        ec = E[(E.signed_gradient == sg) & E.valid_utc.between(t0, t1)]
        o = h[h.lead_day == 1].sort_values("valid_utc")
        ax.plot(lt(o.valid_utc), o.target_mph, color=INK, lw=2.2, label="observed (set P75, hourly)")
        ax.plot(lt(o.valid_utc), o.gbt_obs, color=AQUA, lw=1.2, ls=":", label="GBT, observed SLP")
        for k, ls, lw in ((1, "-", 2), (0, "--", 1.2)):
            ek = ec[ec.lead_day == k].sort_values("valid_utc"); hk = h[h.lead_day == k].sort_values("valid_utc")
            ax.plot(lt(ek.valid_utc), ek.gbt_nwp_corr, color=BLUE, ls=ls, lw=lw, marker="o", ms=2.5,
                    label=f"GBT, ECMWF SLP day {k} (3-hourly)")
            ax.plot(lt(hk.valid_utc), hk.nwp_p75_rep, color=ORANGE, ls=ls, lw=lw,
                    label=f"raw HRRR day {k}" + (" (analysis)" if k == 0 else ""))
            m = hk.merge(ek[["valid_utc", "gbt_nwp_corr"]], on="valid_utc", suffixes=("", "_ec")).dropna(subset=["target_mph"])
            rows.append(dict(signed_gradient=sg, rank=e["rank"], lead_day=k, n=len(m), peak_obs=e.peak_obs_mph,
                             rmse_gbt_ecmwf=np.sqrt(((m.gbt_nwp_corr_ec - m.target_mph) ** 2).mean()),
                             rmse_raw_hrrr=np.sqrt(((m.nwp_p75_rep - m.target_mph) ** 2).mean()),
                             peak_gbt_ecmwf=ek.gbt_nwp_corr.max(), peak_raw_hrrr=hk.nwp_p75_rep.max()))
        ax.set_title(f"#{e['rank']}  {e.local_day}  peak observed {e.peak_obs_mph:.0f} mph", loc="left")
        ax.set_ylabel("mph")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
        ax.set_ylim(bottom=0)
    axes[0, 0].legend(ncol=3, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.45))
    fig.suptitle(f"{sg} sustained wind: GBT (ECMWF SLP) vs raw HRRR, top 5 event days (time PT; all hours shown, "
                 f"no sign selection)", x=0.01, ha="left", y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{IMG}/{sg}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
R = pd.DataFrame(rows)
R.round(2).to_csv(f"{D}/model_comparison/hrrr_vs_gbt_ecmwf_event_windows.csv", index=False)
s = R.groupby("lead_day")[["rmse_gbt_ecmwf", "rmse_raw_hrrr"]].median().round(2)
w = R.assign(gbt_better=R.rmse_gbt_ecmwf < R.rmse_raw_hrrr).groupby("lead_day").gbt_better.agg(["sum", "size"])
print("median event-window RMSE (3-hourly common times):"); print(s.join(w).to_string())
print(f"wrote {IMG}/ ({R.signed_gradient.nunique()} sets)")
