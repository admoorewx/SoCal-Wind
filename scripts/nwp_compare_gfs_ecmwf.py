"""GFS vs ECMWF head-to-head on common rows: ECMWF native valid times, both models' corrected-sign selection.
Usage: python nwp_compare_gfs_ecmwf.py <out.csv>  (derived/verification/model_comparison/gfs_vs_ecmwf_common_native.csv)"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/mnt/d/socal/scripts"); import nwp_config as C
D = "/mnt/d/socal/data/v2/derived/verification"
OFF = ["LAX-BFL-", "LAX-DAG-", "LAX-PMD-", "LAX-TPH-", "LAX-WJF-", "SAN-IPL-", "SAN-LAS-", "SBA-SMX-"]
cols = ["signed_gradient", "valid_utc", "lead_day", "lead_h", "target_mph", "qualifies", "sel_nwp_corr", "gbt_nwp_corr", "nwp_p75_rep"]
F = {}
for m in ("gfs", "ecmwf"):
    x = pd.read_parquet(f"{D}/{m}/forecasts_sustained.parquet", columns=cols)
    x = x[x.qualifies & x.target_mph.notna() & x.sel_nwp_corr & C.is_native(C.MODELS["ecmwf"], C.lead_hours(x.valid_utc.dt.hour.to_numpy(), x.lead_day.to_numpy(), np.where(x.lead_day >= 6, 12, 6)))]
    F[m] = x.drop(columns=["qualifies", "sel_nwp_corr", "lead_h"])
k = ["signed_gradient", "valid_utc", "lead_day"]
M = F["gfs"].merge(F["ecmwf"].drop(columns="target_mph"), on=k, suffixes=("_gfs", "_ecmwf"))
# per-set, per-lead oracle bias correction of raw wind on the common rows
for m in ("gfs", "ecmwf"):
    b = M.groupby(["signed_gradient", "lead_day"])[f"nwp_p75_rep_{m}"].transform("mean") - M.groupby(["signed_gradient", "lead_day"]).target_mph.transform("mean")
    M[f"nwp_bc_{m}"] = M[f"nwp_p75_rep_{m}"] - b
M["day"] = M.valid_utc.dt.floor("D")
fc = [f"{f}_{m}" for f in ("gbt_nwp_corr", "nwp_bc", "nwp_p75_rep") for m in ("gfs", "ecmwf")]
rng = np.random.default_rng(1)
def boot(x, a, b, n=1000):
    e = x.assign(d=(x[a] - x.target_mph) ** 2 - (x[b] - x.target_mph) ** 2).groupby("day").agg(s=("d", "sum"), sa=(a, "size"))
    # rmse difference via resampled days
    ea = x.assign(q=(x[a] - x.target_mph) ** 2).groupby("day").q.sum(); eb = x.assign(q=(x[b] - x.target_mph) ** 2).groupby("day").q.sum()
    nn = x.groupby("day").size(); days = nn.index.to_numpy()
    out = []
    for _ in range(n):
        s = rng.choice(len(days), len(days))
        out.append(np.sqrt(ea.iloc[s].sum() / nn.iloc[s].sum()) - np.sqrt(eb.iloc[s].sum() / nn.iloc[s].sum()))
    return np.percentile(out, [2.5, 97.5])
rows = []
for grp, X in (("all", M), ("offshore", M[M.signed_gradient.isin(OFF)]), (">=30 mph obs", M[M.target_mph >= 30])):
    for d, x in X.groupby("lead_day"):
        r = dict(group=grp, lead=d, n=len(x), **{c: np.sqrt(((x[c] - x.target_mph) ** 2).mean()) for c in fc})
        if grp != ">=30 mph obs":
            lo, hi = boot(x, "gbt_nwp_corr_ecmwf", "gbt_nwp_corr_gfs"); r.update(gbt_e_minus_g=r["gbt_nwp_corr_ecmwf"] - r["gbt_nwp_corr_gfs"], ci=f"[{lo:+.2f},{hi:+.2f}]")
        rows.append(r)
R = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(R.round(2).to_string(index=False))
R.round(3).to_csv(sys.argv[1], index=False)
