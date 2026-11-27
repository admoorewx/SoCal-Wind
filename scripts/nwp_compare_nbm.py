"""NBM (raw 10 m wind, wind-only) vs all guidance on common rows, 2026 holdout, sustained.
Rows: qualifying set-hours native to NBM, GFS, ECMWF and GEM (each model's own lead_h/native_steps; in practice
ECMWF's native set) and selected by all three GBTs' corrected signs (NBM has no pressure, hence no sign).
Forecasts (mph):
  gbt_<m>    GBT driven by corrected <m> SLP (m = gfs, ecmwf, gem; + hrrr on the HRRR table)
  raw_<m>    raw <m> wind, set P75 at the reporting stations (nwp_p75_rep); m also = nbm
  bc_<m>     raw_<m> minus its per-set, per-lead mean bias over these rows (oracle)
Tables (long form: group, lead_day, forecast, n, rmse, bias):
  main     days 0-7, groups all / offshore / >=30 mph obs / all at 00+12Z (equal lead: every model's lead = 24N h,
           NBM's otherwise is 24N while GFS/ECMWF are 24N+0-5 h and GEM 24N+0-11 h)
  hrrr     days 0-1, the main rows where HRRR also has a forecast (HRRR day 0 is an analysis)
Paired day-block bootstrap (1,000) for the key differences.
-> derived/verification/model_comparison/nbm_vs_guidance_{scores,diffs}.csv, nbm_vs_guidance_hrrr_scores.csv
Usage: python nwp_compare_nbm.py"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "/mnt/d/socal/scripts"); import nwp_config as C

D = "/mnt/d/socal/data/v2/derived/verification"
OUT = f"{D}/model_comparison"
OFF = ["LAX-BFL-", "LAX-DAG-", "LAX-PMD-", "LAX-TPH-", "LAX-WJF-", "SAN-IPL-", "SAN-LAS-", "SBA-SMX-"]
k = ["signed_gradient", "valid_utc", "lead_day"]


def load(m, gbt=True, sel=True):
    cols = k + ["lead_h", "target_mph", "qualifies", "nwp_p75_rep"] + (["gbt_nwp_corr", "sel_nwp_corr"] if gbt else [])
    x = pd.read_parquet(f"{D}/{m}/forecasts_sustained.parquet", columns=cols)
    keep = x.qualifies & x.target_mph.notna() & C.is_native(C.MODELS[m], x.lead_h.to_numpy())
    if sel:
        keep &= x.sel_nwp_corr
    x = x[keep].rename(columns={"nwp_p75_rep": f"raw_{m}", "gbt_nwp_corr": f"gbt_{m}"})
    return x.drop(columns=[c for c in ("qualifies", "sel_nwp_corr", "lead_h") if c in x])


M = load("nbm", gbt=False, sel=False).dropna(subset=["raw_nbm"])
for m in ("gfs", "ecmwf", "gem"):
    M = M.merge(load(m).drop(columns="target_mph"), on=k)
M = M.dropna(subset=["gbt_gfs", "gbt_ecmwf", "gbt_gem", "raw_gfs", "raw_ecmwf", "raw_gem"])
H = load("hrrr").drop(columns="target_mph")
MH = M[M.lead_day <= 1].merge(H, on=k).dropna(subset=["gbt_hrrr", "raw_hrrr"])


def add_bc(X, models):
    g = X.groupby(["signed_gradient", "lead_day"])
    for m in models:
        X[f"bc_{m}"] = X[f"raw_{m}"] - (g[f"raw_{m}"].transform("mean") - g.target_mph.transform("mean"))
    X["day"] = X.valid_utc.dt.floor("D")
    return X


M = add_bc(M, ("nbm", "gfs", "ecmwf", "gem"))
MH = add_bc(MH, ("nbm", "gfs", "ecmwf", "gem", "hrrr"))
FC = ["raw_nbm", "bc_nbm"] + [f"{p}_{m}" for m in ("ecmwf", "gem", "gfs") for p in ("gbt", "bc", "raw")]
FCH = FC + ["gbt_hrrr", "bc_hrrr", "raw_hrrr"]
GROUPS = lambda X: (("all", X), ("offshore", X[X.signed_gradient.isin(OFF)]), (">=30 mph obs", X[X.target_mph >= 30]),
                    ("all, 00/12Z (equal lead)", X[X.valid_utc.dt.hour.isin([0, 12])]))


def scores(X, fcs):
    rows = []
    for grp, Y in GROUPS(X):
        for d, y in Y.groupby("lead_day"):
            for c in fcs:
                e = y[c] - y.target_mph
                rows.append(dict(group=grp, lead_day=d, forecast=c, n=len(y), rmse=np.sqrt((e ** 2).mean()),
                                 bias=e.mean()))
    return pd.DataFrame(rows)


rng = np.random.default_rng(1)


def boot(x, a, b, n=1000):
    ea = ((x[a] - x.target_mph) ** 2).groupby(x.day).sum()
    eb = ((x[b] - x.target_mph) ** 2).groupby(x.day).sum()
    nn = x.groupby("day").size()
    out = []
    for _ in range(n):
        s = rng.choice(len(nn), len(nn))
        out.append(np.sqrt(ea.iloc[s].sum() / nn.iloc[s].sum()) - np.sqrt(eb.iloc[s].sum() / nn.iloc[s].sum()))
    return np.percentile(out, [2.5, 97.5])


PAIRS = [("gbt_ecmwf", "raw_nbm"), ("gbt_ecmwf", "bc_nbm"), ("gbt_gem", "bc_nbm"), ("gbt_gfs", "bc_nbm"),
         ("bc_nbm", "bc_ecmwf"), ("raw_nbm", "bc_nbm")]
drows = []
for grp, Y in GROUPS(M):
    if grp.startswith(">=30"):
        continue
    for d, y in Y.groupby("lead_day"):
        for a_, b_ in PAIRS:
            lo, hi = boot(y, a_, b_)
            diff = np.sqrt(((y[a_] - y.target_mph) ** 2).mean()) - np.sqrt(((y[b_] - y.target_mph) ** 2).mean())
            drows.append(dict(group=grp, lead_day=d, a=a_, b=b_, n=len(y), diff=diff, ci_lo=lo, ci_hi=hi,
                              sig=("a better" if hi < 0 else "b better" if lo > 0 else "tie")))
for d, y in MH.groupby("lead_day"):
    for a_, b_ in (("gbt_ecmwf", "raw_hrrr"), ("bc_nbm", "raw_hrrr"), ("bc_nbm", "bc_hrrr"), ("raw_nbm", "raw_hrrr")):
        lo, hi = boot(y, a_, b_)
        diff = np.sqrt(((y[a_] - y.target_mph) ** 2).mean()) - np.sqrt(((y[b_] - y.target_mph) ** 2).mean())
        drows.append(dict(group="all (HRRR rows)", lead_day=d, a=a_, b=b_, n=len(y), diff=diff, ci_lo=lo, ci_hi=hi,
                          sig=("a better" if hi < 0 else "b better" if lo > 0 else "tie")))
S, SH, DF = scores(M, FC), scores(MH, FCH), pd.DataFrame(drows)
S.round(3).to_csv(f"{OUT}/nbm_vs_guidance_scores.csv", index=False)
SH.round(3).to_csv(f"{OUT}/nbm_vs_guidance_hrrr_scores.csv", index=False)
DF.round(3).to_csv(f"{OUT}/nbm_vs_guidance_diffs.csv", index=False)
pd.set_option("display.width", 250)
for grp in S.group.unique():
    print(f"\n== RMSE, {grp} (n per lead: {S[(S.group == grp) & (S.forecast == 'raw_nbm')].n.tolist()}) ==")
    print(S[S.group == grp].pivot(index="lead_day", columns="forecast", values="rmse")[FC].round(2).to_string())
print("\n== bias, all ==")
print(S[S.group == "all"].pivot(index="lead_day", columns="forecast", values="bias")[FC].round(2).to_string())
print("\n== HRRR rows (days 0-1), RMSE all ==")
print(SH[SH.group == "all"].pivot(index="lead_day", columns="forecast", values="rmse")[FCH].round(2).to_string())
print("\n== differences (a - b), 95% CI ==")
DF["txt"] = DF.apply(lambda r: f"{r['diff']:+.2f} [{r.ci_lo:+.2f},{r.ci_hi:+.2f}]", axis=1)
print(DF.pivot_table(index=["group", "a", "b"], columns="lead_day", values="txt", aggfunc="first").to_string())
