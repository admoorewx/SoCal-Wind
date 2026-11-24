"""Raw HRRR wind vs GBT driven by GFS / ECMWF / HRRR pressure, days 0-1, common rows.
Rows: ECMWF native valid times (3-hourly), qualifying, selected by all three models' corrected sign. The oracle bias
correction of raw winds is refitted per set and lead on these rows. Paired day-block bootstrap (1,000) vs raw HRRR and
vs bias-corrected HRRR.
Usage: python nwp_compare_hrrr.py <out.csv>"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/mnt/d/socal/scripts"); import nwp_config as C
D = "/mnt/d/socal/data/v2/derived/verification"
OFF = ["LAX-BFL-", "LAX-DAG-", "LAX-PMD-", "LAX-TPH-", "LAX-WJF-", "SAN-IPL-", "SAN-LAS-", "SBA-SMX-"]
cols = ["signed_gradient", "valid_utc", "lead_day", "target_mph", "qualifies", "sel_nwp_corr", "gbt_nwp_corr", "nwp_p75_rep"]
k = ["signed_gradient", "valid_utc", "lead_day"]
M = None
for m in ("gfs", "ecmwf", "hrrr"):
    x = pd.read_parquet(f"{D}/{m}/forecasts_sustained.parquet", columns=cols, filters=[("lead_day", "<=", 1)])
    x = x[x.qualifies & x.target_mph.notna() & x.sel_nwp_corr & (x.valid_utc.dt.hour % 3 == 0)]
    x = x.drop(columns=["qualifies", "sel_nwp_corr"]).rename(columns={"gbt_nwp_corr": f"gbt_{m}", "nwp_p75_rep": f"raw_{m}"})
    M = x if M is None else M.merge(x.drop(columns="target_mph"), on=k)
for m in ("gfs", "ecmwf", "hrrr"):
    g = M.groupby(["signed_gradient", "lead_day"])
    M[f"bc_{m}"] = M[f"raw_{m}"] - (g[f"raw_{m}"].transform("mean") - g.target_mph.transform("mean"))
M["day"] = M.valid_utc.dt.floor("D")
fc = ["gbt_ecmwf", "gbt_gfs", "gbt_hrrr", "raw_hrrr", "bc_hrrr", "bc_ecmwf", "raw_ecmwf", "raw_gfs"]
rng = np.random.default_rng(1)
def boot(x, a, b, n=1000):
    ea = ((x[a] - x.target_mph) ** 2).groupby(x.day).sum(); eb = ((x[b] - x.target_mph) ** 2).groupby(x.day).sum()
    nn = x.groupby("day").size()
    out = []
    for _ in range(n):
        s = rng.choice(len(nn), len(nn))
        out.append(np.sqrt(ea.iloc[s].sum() / nn.iloc[s].sum()) - np.sqrt(eb.iloc[s].sum() / nn.iloc[s].sum()))
    return np.percentile(out, [2.5, 97.5])
rows = []
for grp, X in (("all", M), ("offshore", M[M.signed_gradient.isin(OFF)]), (">=30 mph obs", M[M.target_mph >= 30]),
               ("20-30 mph obs", M[M.target_mph.between(20, 30, inclusive="left")])):
    for d, x in X.groupby("lead_day"):
        r = dict(group=grp, lead=d, n=len(x), **{c: np.sqrt(((x[c] - x.target_mph) ** 2).mean()) for c in fc},
                 bias_gbt_ecmwf=(x.gbt_ecmwf - x.target_mph).mean(), bias_raw_hrrr=(x.raw_hrrr - x.target_mph).mean())
        if grp in ("all", "offshore"):
            for ref in ("raw_hrrr", "bc_hrrr"):
                lo, hi = boot(x, "gbt_ecmwf", ref)
                r[f"gbtE_minus_{ref}"] = r["gbt_ecmwf"] - r[ref]; r[f"ci_{ref}"] = f"[{lo:+.2f},{hi:+.2f}]"
        rows.append(r)
R = pd.DataFrame(rows)
pd.set_option("display.width", 300, "display.max_columns", 30)
print(R.round(2).to_string(index=False))
R.round(3).to_csv(sys.argv[1], index=False)
