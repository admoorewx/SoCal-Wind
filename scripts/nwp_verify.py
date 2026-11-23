#!/usr/bin/env python3
"""Phase 5: score GBT-NWP vs raw NWP (sustained), docs/nwp_verification_plan.md.

Main rows: qualifying set-hours chosen by the CORRECTED forecast sign (sel_nwp_corr). On these rows:
  gbt_nwp_corr   GBT driven by SLP-convention-corrected NWP pressure (the main forecast)
  nwp_p75_rep    raw NWP: 75th pct of NWP 10 m wind at the target's reporting stations (agreed baseline)
  nwp_bc         raw NWP minus its per-set, per-lead mean bias over these 2026 rows (ORACLE, generous to NWP)
  clim           per-set training (2015-2025) mean target
Also: gbt_nwp_raw on its own sign rows (uncorrected pressure) and gbt_obs on observed-sign rows (perfect pressure).

Outputs -> derived/verification/<model>/scores/, figures -> images/verification/<model>/
  scores_by_lead.csv, bootstrap_by_lead.csv (paired day-block bootstrap, 1,000 resamples, 95% CI), verdict.csv,
  scores_by_speed_bin.csv, scores_by_sign.csv, scores_by_season.csv, scores_by_set.csv, events_brier_csi.csv,
  csi_cutoffs_oof.csv, pressure_station_errors.csv, pressure_gradient_errors.csv, pressure_loss_decomposition.csv,
  event_days.csv, event_window_scores.csv
Usage: python nwp_verify.py --model gfs
"""
import argparse, glob, json, os, warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap

import nwp_config as C
import gradient_pipeline as gp

warnings.simplefilter("ignore", RuntimeWarning)
ROOT = "/mnt/d/socal"
DER = f"{ROOT}/data/v2/derived"
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gfs")
ap.add_argument("--points-tag", default="20260101_20261031")
ap.add_argument("--calib", default="coefficients_2025.json")
ap.add_argument("--n-boot", type=int, default=1000)
a = ap.parse_args()
VDIR = f"{DER}/verification/{a.model}"
OUT = f"{VDIR}/scores"
IMG = f"{ROOT}/images/verification/{a.model}"
for d in (OUT, IMG, f"{IMG}/events"):
    os.makedirs(d, exist_ok=True)
pd.set_option("display.width", 220)
MODEL_LABEL = a.model.upper()

# offshore (Santa Ana-type: inland high vs coast) + sundowner; user to confirm the grouping
OFFSHORE = ["LAX-BFL-", "LAX-DAG-", "LAX-PMD-", "LAX-TPH-", "LAX-WJF-", "SAN-IPL-", "SAN-LAS-", "SBA-SMX-"]
BINS = [0, 10, 20, 30, np.inf]
BIN_LAB = ["<10", "10-20", "20-30", ">=30"]

# palette (dataviz reference instance, light mode)
INK, MUTED, GRID = "#0b0b0b", "#8a8986", "#e5e4e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "font.size": 9, "axes.titlesize": 10, "legend.frameon": False, "lines.linewidth": 2})
DIVERGING = LinearSegmentedColormap.from_list("div", [ORANGE, "#f3f2ee", BLUE])

# ------------------------------------------------------------------ data
F = pd.read_parquet(f"{VDIR}/forecasts_sustained.parquet")
tr = pd.read_parquet(f"{DER}/models/targets/sustained_train.parquet", columns=["signed_gradient", "target_mph", "qualifies"])
clim = tr[tr.qualifies].groupby("signed_gradient").target_mph.mean()
F["clim"] = F.signed_gradient.map(clim)
Q = F[F.qualifies & F.target_mph.notna()].copy()
S = Q[Q.sel_nwp_corr].copy()
S["nwp_bias"] = S.groupby(["signed_gradient", "lead_day"]).apply(
    lambda x: (x.nwp_p75_rep - x.target_mph).mean(), include_groups=False).reindex(
    pd.MultiIndex.from_frame(S[["signed_gradient", "lead_day"]])).to_numpy()
S["nwp_bc"] = S.nwp_p75_rep - S.nwp_bias
S["sign_right"] = S.obs_sign == S.signed_gradient.str[-1]
S["day"] = S.valid_utc.dt.floor("D")
S["speed_bin"] = pd.cut(S.target_mph, BINS, labels=BIN_LAB, right=False)
S["season"] = S.valid_utc.dt.month.map({12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                                        6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"})
S["group"] = np.where(S.signed_gradient.isin(OFFSHORE), "offshore", "other")
MAIN = {"gbt_nwp_corr": "GBT (corrected SLP)", "nwp_p75_rep": f"raw {MODEL_LABEL}",
        "nwp_bc": f"{MODEL_LABEL} bias-corrected (oracle)", "clim": "climatology"}


def met(x, col):
    e = x[col] - x.target_mph
    return pd.Series({"n": len(e), "bias": e.mean(), "mae": e.abs().mean(), "rmse": np.sqrt((e ** 2).mean())})


def table(df, by, cols):
    rows = []
    for k, x in df.groupby(by, observed=True):
        k = k if isinstance(k, tuple) else (k,)
        for c in cols:
            rows.append({**dict(zip(by, k)), "forecast": c, **met(x, c)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 1. by lead
t_main = pd.concat([table(S, ["lead_day"], list(MAIN)).assign(group="all"),
                    table(S[S.group == "offshore"], ["lead_day"], list(MAIN)).assign(group="offshore")])
Sraw = Q[Q.sel_nwp_raw]
Sobs = Q[Q.sel_obs]
t_extra = pd.concat([table(Sraw, ["lead_day"], ["gbt_nwp_raw"]).assign(group="all"),
                     table(Sraw[Sraw.signed_gradient.isin(OFFSHORE)], ["lead_day"], ["gbt_nwp_raw"]).assign(group="offshore"),
                     table(Sobs, ["lead_day"], ["gbt_obs"]).assign(group="all"),
                     table(Sobs[Sobs.signed_gradient.isin(OFFSHORE)], ["lead_day"], ["gbt_obs"]).assign(group="offshore")])
by_lead = pd.concat([t_main, t_extra], ignore_index=True)
by_lead.round(3).to_csv(f"{OUT}/scores_by_lead.csv", index=False)

# ------------------------------------------------------------------ 2. paired day-block bootstrap
rng = np.random.default_rng(20261123)
boot_rows = []
for grp, sub in (("all", S), ("offshore", S[S.group == "offshore"])):
    for k, x in sub.groupby("lead_day"):
        agg = x.assign(se_g=(x.gbt_nwp_corr - x.target_mph) ** 2, se_n=(x.nwp_p75_rep - x.target_mph) ** 2,
                       se_b=(x.nwp_bc - x.target_mph) ** 2).groupby("day")[["se_g", "se_n", "se_b"]].agg(["sum", "count"])
        sg_, sn_, sb_ = agg[("se_g", "sum")].to_numpy(), agg[("se_n", "sum")].to_numpy(), agg[("se_b", "sum")].to_numpy()
        n_ = agg[("se_g", "count")].to_numpy()
        nd = len(n_)
        cnt = rng.multinomial(nd, np.full(nd, 1 / nd), size=a.n_boot)
        N = cnt @ n_
        rg, rn, rb = np.sqrt(cnt @ sg_ / N), np.sqrt(cnt @ sn_ / N), np.sqrt(cnt @ sb_ / N)
        for lab, ref, refb in (("vs_raw", np.sqrt(sn_.sum() / n_.sum()), rn), ("vs_biascorr", np.sqrt(sb_.sum() / n_.sum()), rb)):
            d = rg - refb
            boot_rows.append(dict(group=grp, lead_day=k, comparison=lab, n_days=nd, n=int(n_.sum()),
                                  rmse_gbt=np.sqrt(sg_.sum() / n_.sum()), rmse_ref=ref,
                                  diff=np.sqrt(sg_.sum() / n_.sum()) - ref,
                                  ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5),
                                  skill=1 - sg_.sum() / (sn_.sum() if lab == "vs_raw" else sb_.sum())))
boot = pd.DataFrame(boot_rows)
boot["gbt_better_sig"] = boot.ci_hi < 0
boot.round(3).to_csv(f"{OUT}/bootstrap_by_lead.csv", index=False)
verdict = boot.groupby(["group", "comparison"]).agg(leads_sig_better=("gbt_better_sig", "sum"),
                                                    n_leads=("gbt_better_sig", "size")).reset_index()
verdict["criterion_met"] = verdict.leads_sig_better == verdict.n_leads
verdict.to_csv(f"{OUT}/verdict.csv", index=False)

# ------------------------------------------------------------------ 3. splits
cols3 = ["gbt_nwp_corr", "nwp_p75_rep", "nwp_bc"]
by_bin = pd.concat([table(S, ["lead_day", "speed_bin"], cols3).assign(group="all"),
                    table(S[S.group == "offshore"], ["lead_day", "speed_bin"], cols3).assign(group="offshore")])
by_bin.round(3).to_csv(f"{OUT}/scores_by_speed_bin.csv", index=False)
by_sign = table(S, ["lead_day", "sign_right"], cols3)
by_sign.round(3).to_csv(f"{OUT}/scores_by_sign.csv", index=False)
by_season = table(S, ["lead_day", "season"], cols3)
by_season.round(3).to_csv(f"{OUT}/scores_by_season.csv", index=False)
by_set = table(S, ["signed_gradient", "lead_day"], cols3)
bs = by_set.pivot_table(index=["signed_gradient", "lead_day"], columns="forecast", values="rmse")
bs["skill_vs_raw"] = 1 - bs.gbt_nwp_corr ** 2 / bs.nwp_p75_rep ** 2
bs["skill_vs_bc"] = 1 - bs.gbt_nwp_corr ** 2 / bs.nwp_bc ** 2
bs.round(3).to_csv(f"{OUT}/scores_by_set.csv")

# ------------------------------------------------------------------ 4. events: Brier, CSI
oof = pd.read_parquet(f"{DER}/models/gbt/oof_predictions.parquet", columns=["signed_gradient", "variable", "kind",
                                                                            "threshold_mph", "obs", "pred_gbt_all"])
oof = oof[(oof.variable == "sustained") & (oof.kind == "logistic")]
cut_rows = []
grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
for (sg, T), x in oof.groupby(["signed_gradient", "threshold_mph"]):
    y, p = x.obs.to_numpy() > 0.5, x.pred_gbt_all.to_numpy()
    best = max(grid, key=lambda c: ((p >= c) & y).sum() / max(((p >= c) | y).sum(), 1))
    cut_rows.append(dict(signed_gradient=sg, threshold_mph=int(T), cutoff=best))
cuts = pd.DataFrame(cut_rows)
cuts.to_csv(f"{OUT}/csi_cutoffs_oof.csv", index=False)


def cat_scores(fc, ob):
    h, m_, f_ = (fc & ob).sum(), (~fc & ob).sum(), (fc & ~ob).sum()
    return dict(pod=h / max(h + m_, 1), far=f_ / max(h + f_, 1), csi=h / max(h + m_ + f_, 1))


ev_rows = []
for T in (15, 20, 30):
    pc = f"p{T}_nwp_corr"
    x = S[S[pc].notna()].merge(cuts[cuts.threshold_mph == T][["signed_gradient", "cutoff"]], on="signed_gradient")
    for grp, xg in (("all", x), ("offshore", x[x.group == "offshore"])):
        for k, xx in xg.groupby("lead_day"):
            ob = (xx.target_mph >= T).to_numpy().astype(float); obb = ob > 0.5
            r = dict(threshold_mph=T, group=grp, lead_day=k, n=len(xx), base_rate=ob.mean(),
                     brier_gbt=np.mean((xx[pc] - ob) ** 2),
                     brier_nwp=np.mean(((xx.nwp_p75_rep >= T).to_numpy(float) - ob) ** 2),
                     brier_nwp_bc=np.mean(((xx.nwp_bc >= T).to_numpy(float) - ob) ** 2),
                     brier_clim=np.mean((ob.mean() - ob) ** 2))
            for lab, fc in (("gbt", (xx[pc] >= xx.cutoff).to_numpy()), ("nwp", (xx.nwp_p75_rep >= T).to_numpy()),
                            ("nwp_bc", (xx.nwp_bc >= T).to_numpy())):
                r.update({f"{m}_{lab}": v for m, v in cat_scores(fc, obb).items()})
            ev_rows.append(r)
ev = pd.DataFrame(ev_rows)
ev.round(4).to_csv(f"{OUT}/events_brier_csi.csv", index=False)

# ------------------------------------------------------------------ 5. NWP pressure error (documentation)
pairs = gp.load_pairs(gp.DEFAULT_GRADIENTS)
pres_st = sorted({s for p in pairs.values() for s in p})
P = pd.read_parquet(f"{C.OPEN_METEO_ROOT}/points/{a.model}/points_{a.points_tag}.parquet",
                    columns=["valid_utc", "stid", "lead_day", "pressure_msl_hpa"], filters=[("stid", "in", pres_st)])
P["valid_utc"] = P.valid_utc.astype("datetime64[us, UTC]")
coef = json.load(open(f"{VDIR}/pressure_calibration/{a.calib}"))["coefficients"]
P["p_corr"] = np.nan
for s, idx in P.groupby("stid").groups.items():
    P.loc[idx, "p_corr"] = P.loc[idx, "pressure_msl_hpa"] - C.slp_correction_design(P.loc[idx, "valid_utc"]) @ np.array(coef[s])
g = pd.read_parquet(f"{DER}/gradients/gradient_timeseries.parquet")
g = g[(g.time_utc >= P.valid_utc.min() - pd.Timedelta(hours=1)) & (g.time_utc < P.valid_utc.max())].copy()
g["valid_utc"] = g.time_utc.dt.ceil("h").astype("datetime64[us, UTC]")
o = pd.concat([g[["station1", "valid_utc", "slp1_hpa"]].set_axis(["stid", "valid_utc", "slp"], axis=1),
               g[["station2", "valid_utc", "slp2_hpa"]].set_axis(["stid", "valid_utc", "slp"], axis=1)])
o = o.dropna().groupby(["stid", "valid_utc"]).slp.first().reset_index()
PM = P.merge(o, on=["stid", "valid_utc"])
st_rows = []
for (s, k), x in PM.groupby(["stid", "lead_day"]):
    for lab, c in (("raw", "pressure_msl_hpa"), ("corrected", "p_corr")):
        e = x[c] - x.slp
        st_rows.append(dict(stid=s, lead_day=k, version=lab, n=len(e), bias=e.mean(), sd=e.std(), rmse=np.sqrt((e ** 2).mean())))
pd.DataFrame(st_rows).round(3).to_csv(f"{OUT}/pressure_station_errors.csv", index=False)
W = PM.pivot_table(index=["lead_day", "valid_utc"], columns="stid", values=["pressure_msl_hpa", "p_corr", "slp"])
gr_rows = []
for pn, (s1, s2) in pairs.items():
    go = W["slp"][s1] - W["slp"][s2]
    for lab, c in (("raw", "pressure_msl_hpa"), ("corrected", "p_corr")):
        gn = W[c][s1] - W[c][s2]
        d = pd.DataFrame(dict(gn=gn, go=go)).dropna()
        for k, x in d.groupby(level="lead_day"):
            e = x.gn - x.go
            gr_rows.append(dict(gradient=pn, lead_day=k, version=lab, n=len(x), bias=e.mean(),
                                rmse=np.sqrt((e ** 2).mean()), r=x.gn.corr(x.go),
                                sign_agree=(np.sign(x.gn) == np.sign(x.go)).mean()))
gerr = pd.DataFrame(gr_rows)
gerr.round(3).to_csv(f"{OUT}/pressure_gradient_errors.csv", index=False)
# loss decomposition on rows where both observed and corrected-forecast signs pick the set
common = Q[Q.sel_obs & Q.sel_nwp_corr]
dec = common.groupby("lead_day").apply(lambda x: pd.Series(dict(
    n=len(x), frac_of_main=len(x) / max((S.lead_day == x.name).sum(), 1),
    rmse_gbt_obs=np.sqrt(((x.gbt_obs - x.target_mph) ** 2).mean()),
    rmse_gbt_corr=np.sqrt(((x.gbt_nwp_corr - x.target_mph) ** 2).mean()))), include_groups=False)
dec["median_gradient_rmse_corr"] = gerr[gerr.version == "corrected"].groupby("lead_day").rmse.median()
dec["median_gradient_rmse_raw"] = gerr[gerr.version == "raw"].groupby("lead_day").rmse.median()
dec.round(3).to_csv(f"{OUT}/pressure_loss_decomposition.csv")

# ------------------------------------------------------------------ 6. top event days per set
L0 = F[(F.lead_day == 0) & F.qualifies & F.target_mph.notna()].copy()
L0["local_day"] = L0.valid_utc.dt.tz_convert("America/Los_Angeles").dt.date
ev_days = []
for sg, x in L0[L0.obs_sign == L0.signed_gradient.str[-1]].groupby("signed_gradient"):
    daily = x.loc[x.groupby("local_day").target_mph.idxmax()].sort_values("target_mph", ascending=False)
    chosen = []
    for _, r in daily.iterrows():
        if all(abs((pd.Timestamp(r.local_day) - pd.Timestamp(c.local_day)).days) >= 3 for c in chosen):
            chosen.append(r)
        if len(chosen) == 5:
            break
    for rank, r in enumerate(chosen, 1):
        ev_days.append(dict(signed_gradient=sg, rank=rank, local_day=r.local_day, peak_valid_utc=r.valid_utc,
                            peak_obs_mph=r.target_mph))
EV = pd.DataFrame(ev_days)

PLOT_LEADS = (1, 3)
os.makedirs(f"{IMG}/events_d5d7", exist_ok=True)


def plot_events(plot_leads, subdir):
    """One figure per set: top-5 event windows, obs vs GBT and raw NWP at two lead days."""
    rows = []
    k0 = plot_leads[0]
    for sg, evs in EV.groupby("signed_gradient"):
        fig, axes = plt.subplots(len(evs), 1, figsize=(11, 2.6 * len(evs)), squeeze=False)
        for ax, (_, e) in zip(axes[:, 0], evs.iterrows()):
            t0, t1 = e.peak_valid_utc - pd.Timedelta(hours=24), e.peak_valid_utc + pd.Timedelta(hours=24)
            x = F[(F.signed_gradient == sg) & F.valid_utc.between(t0, t1)]
            lt = lambda s: s.dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
            d1 = x[x.lead_day == k0].sort_values("valid_utc")
            # shade hours where the first plotted lead's corrected forecast sign does NOT pick this set
            for v in d1.valid_utc[~d1.sel_nwp_corr]:
                vv = v.tz_convert("America/Los_Angeles").tz_localize(None)
                ax.axvspan(vv - pd.Timedelta(minutes=30), vv + pd.Timedelta(minutes=30), color="#efeeea", lw=0)
            ax.plot(lt(d1.valid_utc), d1.target_mph, color=INK, lw=2.2, label="observed (set P75)")
            ax.plot(lt(d1.valid_utc), d1.gbt_obs, color=AQUA, lw=1.2, ls=":", label="GBT, observed SLP")
            for k, ls in zip(plot_leads, ("-", "--")):
                dk = x[x.lead_day == k].sort_values("valid_utc")
                ax.plot(lt(dk.valid_utc), dk.gbt_nwp_corr, color=BLUE, ls=ls, label=f"GBT, {MODEL_LABEL} day {k}")
                ax.plot(lt(dk.valid_utc), dk.nwp_p75_rep, color=ORANGE, ls=ls, label=f"raw {MODEL_LABEL} day {k}")
                s_ = dk[dk.sel_nwp_corr & dk.qualifies]
                rows.append(dict(signed_gradient=sg, rank=e["rank"], lead_day=k, n=len(s_),
                                 rmse_gbt=np.sqrt(((s_.gbt_nwp_corr - s_.target_mph) ** 2).mean()),
                                 rmse_nwp=np.sqrt(((s_.nwp_p75_rep - s_.target_mph) ** 2).mean()),
                                 peak_obs=dk.target_mph.max(), peak_gbt=dk.gbt_nwp_corr.max(), peak_nwp=dk.nwp_p75_rep.max(),
                                 peak_gbt_obs=dk.gbt_obs.max()))
            ax.set_title(f"#{e['rank']}  {e.local_day}  peak observed {e.peak_obs_mph:.0f} mph", loc="left")
            ax.set_ylabel("mph")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %H:%M"))
            ax.set_ylim(bottom=0)
        axes[0, 0].legend(ncol=3, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.45))
        fig.suptitle(f"{sg} sustained wind: top 5 event days, lead days {plot_leads[0]} and {plot_leads[1]} "
                     f"(time PT; grey = day-{k0} forecast sign picks the other set)", x=0.01, ha="left", y=1.0, fontsize=11)
        fig.tight_layout()
        fig.savefig(f"{IMG}/{subdir}/{sg}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    return rows


win_rows = plot_events(PLOT_LEADS, "events") + plot_events((5, 7), "events_d5d7")
win = pd.DataFrame(win_rows)
EV.merge(win[win.lead_day == 1][["signed_gradient", "rank", "peak_gbt", "peak_nwp", "peak_gbt_obs"]],
         on=["signed_gradient", "rank"]).round({"peak_obs_mph": 1, "peak_gbt": 1, "peak_nwp": 1, "peak_gbt_obs": 1}).to_csv(f"{OUT}/event_days.csv", index=False)
win.round(2).to_csv(f"{OUT}/event_window_scores.csv", index=False)

# pooled event-window scores, all leads (rows chosen by corrected sign, within +/-24 h of each event peak)
ew = []
for _, e in EV.iterrows():
    x = S[(S.signed_gradient == e.signed_gradient) &
          S.valid_utc.between(e.peak_valid_utc - pd.Timedelta(hours=24), e.peak_valid_utc + pd.Timedelta(hours=24))]
    ew.append(x.assign(event=f"{e.signed_gradient}#{e['rank']}"))
EW = pd.concat(ew)
EW["group"] = np.where(EW.signed_gradient.isin(OFFSHORE), "offshore", "other")
ew_tab = pd.concat([table(EW, ["lead_day"], cols3).assign(group="all events"),
                    table(EW[EW.group == "offshore"], ["lead_day"], cols3).assign(group="offshore events")])
ew_tab.round(3).to_csv(f"{OUT}/event_window_scores_by_lead.csv", index=False)

# ------------------------------------------------------------------ figures
leads = sorted(S.lead_day.unique())
STY = {"gbt_nwp_corr": (BLUE, "-", "o"), "gbt_nwp_raw": (BLUE, "--", None), "gbt_obs": (AQUA, ":", None),
       "nwp_p75_rep": (ORANGE, "-", "o"), "nwp_bc": (ORANGE, "--", None), "clim": (MUTED, "-", None)}
LBL = {**MAIN, "gbt_nwp_raw": "GBT (uncorrected SLP)", "gbt_obs": "GBT (observed SLP)"}

fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
for i, grp in enumerate(("all", "offshore")):
    ax = axes[i, 0]
    for c in ["clim", "gbt_obs", "gbt_nwp_raw", "nwp_bc", "nwp_p75_rep", "gbt_nwp_corr"]:
        x = by_lead[(by_lead.group == grp) & (by_lead.forecast == c)].sort_values("lead_day")
        col, ls, mk = STY[c]
        ax.plot(x.lead_day, x.rmse, color=col, ls=ls, marker=mk, ms=5, label=LBL[c])
    ax.set(title=f"RMSE by lead day ({'all sets' if grp == 'all' else 'offshore sets'})", xlabel="lead day",
           ylabel="RMSE (mph)")
    ax.set_ylim(bottom=0)
    ax = axes[i, 1]
    for comp, ls, lab in (("vs_raw", "-", f"GBT - raw {MODEL_LABEL}"), ("vs_biascorr", "--", f"GBT - bias-corrected {MODEL_LABEL}")):
        x = boot[(boot.group == grp) & (boot.comparison == comp)].sort_values("lead_day")
        ax.fill_between(x.lead_day, x.ci_lo, x.ci_hi, color=BLUE, alpha=0.15, lw=0)
        ax.plot(x.lead_day, x["diff"], color=BLUE, ls=ls, marker="o", ms=5, label=lab)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set(title="RMSE difference, 95% day-block bootstrap CI (below 0 = GBT better)", xlabel="lead day", ylabel="mph")
    ax.legend(fontsize=8)
for ax in axes[:, 0]:
    ax.legend(fontsize=7.5, loc="lower right")
fig.tight_layout()
fig.savefig(f"{IMG}/rmse_by_lead.png", dpi=130)
plt.close(fig)

# skill by set heatmaps
fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
order = sorted(bs.index.get_level_values(0).unique(), key=lambda s: (s not in OFFSHORE, s))
for ax, col, ttl in zip(axes, ("skill_vs_raw", "skill_vs_bc"), (f"vs raw {MODEL_LABEL}", f"vs bias-corrected {MODEL_LABEL} (oracle)")):
    M = bs[col].unstack("lead_day").reindex(order)
    im = ax.imshow(M.to_numpy(), cmap=DIVERGING, vmin=-0.6, vmax=0.6, aspect="auto")
    for (r, c), v in np.ndenumerate(M.to_numpy()):
        ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color=INK)
    ax.set_xticks(range(M.shape[1]), M.columns)
    ax.set_yticks(range(M.shape[0]), [f"{s} *" if s in OFFSHORE else s for s in M.index])
    ax.set(title=f"MSE skill of GBT {ttl}", xlabel="lead day")
    ax.grid(False)
fig.colorbar(im, ax=axes, shrink=0.6, label="skill = 1 - MSE(GBT)/MSE(ref)  (blue = GBT better)")
fig.text(0.01, 0.01, "* offshore group", fontsize=8, color="#52514e")
fig.savefig(f"{IMG}/skill_by_set.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# by observed-speed bin (strong winds)
for grp in ("all", "offshore"):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for j, k in enumerate(PLOT_LEADS):
        x = by_bin[(by_bin.group == grp) & (by_bin.lead_day == k)]
        for c in cols3:
            xc = x[x.forecast == c].set_index("speed_bin").reindex(BIN_LAB)
            col, ls, mk = STY[c]
            axes[0, j].plot(BIN_LAB, xc.rmse, color=col, ls=ls, marker="o", ms=5, label=LBL[c])
            axes[1, j].plot(BIN_LAB, xc.bias, color=col, ls=ls, marker="o", ms=5, label=LBL[c])
        n_ = x[x.forecast == "gbt_nwp_corr"].set_index("speed_bin").reindex(BIN_LAB).n
        axes[0, j].set_title(f"RMSE by observed speed, lead day {k}  (n: {', '.join(f'{int(v):,}' for v in n_)})", fontsize=9)
        axes[1, j].set_title(f"bias by observed speed, lead day {k}", fontsize=9)
        axes[1, j].axhline(0, color=INK, lw=0.8)
        axes[1, j].set_xlabel("observed set P75 (mph)")
    axes[0, 0].set_ylabel("RMSE (mph)")
    axes[1, 0].set_ylabel("forecast - observed (mph)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Strong-wind performance ({'all sets' if grp == 'all' else 'offshore sets'})", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(f"{IMG}/rmse_by_speed_bin_{grp}.png", dpi=130)
    plt.close(fig)

# pressure error and loss
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for lab, ls in (("raw", "--"), ("corrected", "-")):
    x = gerr[gerr.version == lab]
    for pn, xp in x.groupby("gradient"):
        axes[0].plot(xp.lead_day, xp.rmse, color=BLUE if lab == "corrected" else ORANGE, lw=0.6, alpha=0.35)
    med = x.groupby("lead_day").rmse.median()
    axes[0].plot(med.index, med.values, color=BLUE if lab == "corrected" else ORANGE, ls=ls, lw=2.4,
                 label=f"median over pairs, {lab} SLP")
axes[0].set(title=f"{MODEL_LABEL} gradient error vs METAR gradient (thin = individual pairs)", xlabel="lead day",
            ylabel="gradient RMSE (hPa)")
axes[0].legend(fontsize=8)
axes[1].plot(dec.index, dec.rmse_gbt_obs, color=AQUA, marker="o", ms=5, label="GBT, observed SLP")
axes[1].plot(dec.index, dec.rmse_gbt_corr, color=BLUE, marker="o", ms=5, label=f"GBT, corrected {MODEL_LABEL} SLP")
axes[1].set(title="Wind RMSE on hours where both signs pick the set", xlabel="lead day", ylabel="RMSE (mph)")
axes[1].set_ylim(bottom=0)
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{IMG}/pressure_error_and_loss.png", dpi=130)
plt.close(fig)

# Brier by lead
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, T in zip(axes, (15, 20, 30)):
    x = ev[(ev.threshold_mph == T) & (ev.group == "all")]
    ax.plot(x.lead_day, x.brier_gbt, color=BLUE, marker="o", ms=5, label="GBT probability")
    ax.plot(x.lead_day, x.brier_nwp, color=ORANGE, marker="o", ms=5, label=f"raw {MODEL_LABEL} yes/no")
    ax.plot(x.lead_day, x.brier_nwp_bc, color=ORANGE, ls="--", label=f"bias-corr. {MODEL_LABEL} yes/no")
    ax.plot(x.lead_day, x.brier_clim, color=MUTED, label="climatology (2026 base rate)")
    ax.set(title=f">= {T} mph (base rate {x.base_rate.mean():.3f})", xlabel="lead day", ylabel="Brier score")
    ax.set_ylim(bottom=0)
axes[0].legend(fontsize=7.5)
fig.tight_layout()
fig.savefig(f"{IMG}/brier_by_lead.png", dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ print summary
print("== RMSE by lead (mph) ==")
print(by_lead.pivot_table(index=["group", "lead_day"], columns="forecast", values="rmse").round(2).to_string())
print("\n== bias by lead (mph) ==")
print(by_lead.pivot_table(index=["group", "lead_day"], columns="forecast", values="bias").round(2).to_string())
print("\n== bootstrap ==")
print(boot[["group", "comparison", "lead_day", "n_days", "rmse_gbt", "rmse_ref", "diff", "ci_lo", "ci_hi", "skill",
            "gbt_better_sig"]].round(3).to_string(index=False))
print("\n== verdict (criterion: GBT significantly better at every lead) ==")
print(verdict.to_string(index=False))
print("\n== by observed speed bin, lead 1 and 3 ==")
print(by_bin[by_bin.lead_day.isin(PLOT_LEADS)].pivot_table(index=["group", "lead_day", "speed_bin"], columns="forecast",
      values=["rmse", "bias"], observed=True).round(2).to_string())
print("\n== by sign right/wrong (rmse) ==")
print(by_sign.pivot_table(index=["lead_day", "sign_right"], columns="forecast", values="rmse").round(2).to_string())
print("\n== by season (rmse), lead 1 ==")
print(by_season[by_season.lead_day == 1].pivot_table(index="season", columns="forecast", values=["rmse", "n"]).round(2).to_string())
print("\n== events: Brier and CSI, all sets ==")
print(ev[ev.group == "all"][["threshold_mph", "lead_day", "base_rate", "brier_gbt", "brier_nwp", "brier_nwp_bc", "brier_clim",
                             "csi_gbt", "csi_nwp", "csi_nwp_bc", "pod_gbt", "pod_nwp", "far_gbt", "far_nwp"]].round(3).to_string(index=False))
print("\n== pressure loss decomposition ==")
print(dec.round(2).to_string())
print("\n== event windows (+/-24 h of top-5 peaks), rmse by lead ==")
print(ew_tab.pivot_table(index=["group", "lead_day"], columns="forecast", values=["rmse", "bias"]).round(2).to_string())
print(f"\nwrote {OUT}/ and {IMG}/")
