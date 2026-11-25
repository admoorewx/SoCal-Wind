"""GEM vs GFS vs ECMWF on common rows. Rows: valid times native to every model (each model's own lead_h and
native_steps) and selected by all three models' corrected sign. In practice the common set is ECMWF's native set.
Paired day-block bootstrap (1,000 resamples) of RMSE differences.
Usage: python nwp_compare_gem.py <out.csv>  (derived/verification/model_comparison/gem_vs_gfs_ecmwf_common_native.csv)"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "/mnt/d/socal/scripts"); import nwp_config as C
D = "/mnt/d/socal/data/v2/derived/verification"
OFF = ["LAX-BFL-", "LAX-DAG-", "LAX-PMD-", "LAX-TPH-", "LAX-WJF-", "SAN-IPL-", "SAN-LAS-", "SBA-SMX-"]
MODELS = ("gfs", "ecmwf", "gem")
cols = ["signed_gradient", "valid_utc", "lead_day", "lead_h", "target_mph", "qualifies", "sel_nwp_corr", "gbt_nwp_corr",
        "nwp_p75_rep"]
k = ["signed_gradient", "valid_utc", "lead_day"]
M = None
for m in MODELS:
    x = pd.read_parquet(f"{D}/{m}/forecasts_sustained.parquet", columns=cols)
    x = x[x.qualifies & x.target_mph.notna() & x.sel_nwp_corr & C.is_native(C.MODELS[m], x.lead_h.to_numpy())]
    x = x.drop(columns=["qualifies", "sel_nwp_corr", "lead_h"]).rename(
        columns={"gbt_nwp_corr": f"gbt_nwp_corr_{m}", "nwp_p75_rep": f"nwp_p75_rep_{m}"})
    M = x if M is None else M.merge(x.drop(columns="target_mph"), on=k)
# per-set, per-lead oracle bias correction of raw wind on the common rows
for m in MODELS:
    g = M.groupby(["signed_gradient", "lead_day"])
    M[f"nwp_bc_{m}"] = M[f"nwp_p75_rep_{m}"] - (g[f"nwp_p75_rep_{m}"].transform("mean") - g.target_mph.transform("mean"))
M["day"] = M.valid_utc.dt.floor("D")
fc = [f"{f}_{m}" for f in ("gbt_nwp_corr", "nwp_bc", "nwp_p75_rep") for m in MODELS]
rng = np.random.default_rng(1)


def boot(x, a, b, n=1000):
    ea = x.assign(q=(x[a] - x.target_mph) ** 2).groupby("day").q.sum()
    eb = x.assign(q=(x[b] - x.target_mph) ** 2).groupby("day").q.sum()
    nn = x.groupby("day").size()
    out = []
    for _ in range(n):
        s = rng.choice(len(nn), len(nn))
        out.append(np.sqrt(ea.iloc[s].sum() / nn.iloc[s].sum()) - np.sqrt(eb.iloc[s].sum() / nn.iloc[s].sum()))
    return np.percentile(out, [2.5, 97.5])


PAIRS = [("gbt_nwp_corr_gem", "gbt_nwp_corr_ecmwf"), ("gbt_nwp_corr_gem", "gbt_nwp_corr_gfs"),
         ("gbt_nwp_corr_gem", "nwp_bc_gem"), ("gbt_nwp_corr_gem", "nwp_p75_rep_gem")]
rows = []
for grp, X in (("all", M), ("offshore", M[M.signed_gradient.isin(OFF)]), (">=30 mph obs", M[M.target_mph >= 30])):
    for d, x in X.groupby("lead_day"):
        r = dict(group=grp, lead=d, n=len(x), **{c: np.sqrt(((x[c] - x.target_mph) ** 2).mean()) for c in fc})
        r.update({f"bias_{c}": (x[c] - x.target_mph).mean() for c in ("gbt_nwp_corr_gem", "gbt_nwp_corr_ecmwf",
                                                                       "nwp_p75_rep_gem")})
        if grp != ">=30 mph obs":
            for a, b in PAIRS:
                lo, hi = boot(x, a, b)
                tag = f"{a.replace('_nwp_corr', '').replace('_rep', '')}-{b.replace('_nwp_corr', '').replace('_rep', '')}"
                r[tag] = r[a] - r[b]
                r[f"ci {tag}"] = f"[{lo:+.2f},{hi:+.2f}]"
        rows.append(r)
R = pd.DataFrame(rows)
pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 50)
print(R.round(2).to_string(index=False))
R.round(3).to_csv(sys.argv[1], index=False)
