"""Data-quality checks on the fetched Open-Meteo point forecasts (before Phase 4).

Checks: completeness; value ranges; stuck/constant runs; lead-day consistency (distinct lead days, stable
climatology); grid-cell distance/height offsets; quick day-0 sanity vs observations (METAR SLP at gradient
stations; matched max sustained wind at set stations).

Usage: python nwp_qc_points.py --model gfs --tag 20260101_20261031
-> derived/verification/<model>/qc/*.csv, printed summary
"""
import argparse, glob, os

import numpy as np
import pandas as pd

import nwp_config as C

ROOT = "/mnt/d/socal"
DER = f"{ROOT}/data/v2/derived"

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gfs")
ap.add_argument("--tag", default="20260101_20261031")
a = ap.parse_args()
OUT = f"{DER}/verification/{a.model}/qc"
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 220)

P = pd.read_parquet(f"{C.OPEN_METEO_ROOT}/points/{a.model}/points_{a.tag}.parquet")
cells = pd.read_csv(f"{C.OPEN_METEO_ROOT}/points/{a.model}/grid_cells.csv")
PV, WV = "pressure_msl_hpa", "wind_speed_10m_mps"
pres_st = cells.stid[cells.role.isin(["gradient", "both"])].tolist()
wind_st = cells.stid[cells.role.isin(["set", "both"])].tolist()

# ---- 1. completeness ----
print("== 1. completeness ==")
exp_hours = P.valid_utc.nunique()
cnt = P.groupby(["stid", "lead_day"]).size()
print(f"stations {P.stid.nunique()}, valid hours {exp_hours} ({P.valid_utc.min()} .. {P.valid_utc.max()}), "
      f"station x lead groups with != {exp_hours} rows: {(cnt != exp_hours).sum()}")
print(f"missing pressure at pressure stations: {P[P.stid.isin(pres_st)][PV].isna().sum()}, "
      f"missing wind at wind stations: {P[P.stid.isin(wind_st)][WV].isna().sum()}")

# ---- 2. ranges ----
print("\n== 2. value ranges ==")
pp = P[P.stid.isin(pres_st)].groupby("stid")[PV].describe(percentiles=[0.01, 0.99])[["min", "1%", "mean", "99%", "max"]]
print("pressure (hPa) overall min/max:", round(pp["min"].min(), 1), round(pp["max"].max(), 1))
print(pp.round(1).sort_values("min").head(5).to_string())
ww = P[P.stid.isin(wind_st)].groupby("stid")[WV].describe(percentiles=[0.99])[["min", "mean", "99%", "max"]]
print(f"wind (m/s): min {ww['min'].min():.2f}, negative values {(P[WV] < 0).sum()}, "
      f"overall max {ww['max'].max():.1f}; stations with max > 30 m/s: {(ww['max'] > 30).sum()}")
print("highest-max wind stations:\n" + ww.sort_values("max", ascending=False).head(5).round(1).to_string())
pp.round(2).to_csv(f"{OUT}/pressure_ranges.csv"); ww.round(2).to_csv(f"{OUT}/wind_ranges.csv")

# ---- 3. stuck / constant runs (day 0 and day 7) ----
print("\n== 3. stuck values ==")
def longest_run(x):
    v = x.to_numpy()
    if len(v) == 0:
        return 0
    change = np.r_[True, v[1:] != v[:-1]]
    ids = np.cumsum(change)
    return np.bincount(ids).max()
rows = []
for var, sts in ((PV, pres_st), (WV, wind_st)):
    sub = P[P.stid.isin(sts) & P.lead_day.isin([0, 7])].sort_values(["stid", "lead_day", "valid_utc"])
    for (s, k), x in sub.groupby(["stid", "lead_day"]):
        rows.append(dict(var=var, stid=s, lead_day=k, longest_identical_run_h=longest_run(x[var]),
                         n_unique=x[var].nunique(), frac_zero=float((x[var] == 0).mean())))
stuck = pd.DataFrame(rows)
stuck.to_csv(f"{OUT}/stuck_runs.csv", index=False)
for var in (PV, WV):
    s = stuck[stuck["var"] == var]
    print(f"{var}: longest identical run max {s.longest_identical_run_h.max()} h "
          f"(median {s.longest_identical_run_h.median():.0f}); min unique values {s.n_unique.min()}; "
          f"max frac exactly zero {s.frac_zero.max():.3f}")
print(stuck.sort_values("longest_identical_run_h", ascending=False).head(6).to_string(index=False))

# ---- 4. lead-day consistency ----
print("\n== 4. lead-day consistency ==")
w = P.pivot_table(index=["stid", "valid_utc"], columns="lead_day", values=[PV, WV])
lc = []
for var, sts in ((PV, pres_st), (WV, wind_st)):
    x = w[var].loc[sts]
    for k in range(8):
        r = dict(var=var, lead_day=k, mean=x[k].mean(), sd=x[k].std())
        if k < 7:
            r["frac_identical_to_next"] = float((x[k] == x[k + 1]).mean())
            r["corr_with_next"] = x[k].corr(x[k + 1])
        lc.append(r)
lc = pd.DataFrame(lc)
lc.round(4).to_csv(f"{OUT}/lead_consistency.csv", index=False)
print(lc.round(3).to_string(index=False))
# days with an entire lead identical to the next lead at all stations (a duplicated run)
d = (w[WV].loc[wind_st][0] == w[WV].loc[wind_st][1]).groupby(level="valid_utc").mean()
print(f"valid hours where >= 90% of wind stations have day0 == day1: {(d >= 0.9).sum()}")

# ---- 5. grid cells ----
print("\n== 5. grid cells ==")
lat1, lon1, lat2, lon2 = map(np.radians, (cells.stn_lat, cells.stn_lon, cells.grid_lat, cells.grid_lon))
cells["dist_km"] = 6371 * 2 * np.arcsin(np.sqrt(np.sin((lat2 - lat1) / 2) ** 2
                                                + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2))
cells["dz_m"] = cells.grid_elev_m - cells.stn_elev_m
cells.round(2).drop(columns="note").to_csv(f"{OUT}/grid_cells_offsets.csv", index=False)
print(cells[["dist_km", "dz_m"]].describe(percentiles=[0.5, 0.9, 0.99]).round(1).T.to_string())
print(f"dist > 15 km: {(cells.dist_km > 15).sum()}; |dz| > 300 m: {(cells.dz_m.abs() > 300).sum()}")
print(cells.reindex(cells.dz_m.abs().sort_values(ascending=False).index)
      [["stid", "role", "stn_elev_m", "grid_elev_m", "dz_m", "dist_km"]].head(8).round(1).to_string(index=False))

# ---- 6a. day-0 pressure vs METAR SLP ----
print("\n== 6a. day-0 pressure vs METAR SLP (gradient stations) ==")
g = pd.read_parquet(f"{DER}/gradients/gradient_timeseries.parquet")
g = g[(g.time_utc >= "2025-12-31 23:00") & (g.time_utc < "2026-11-01")].copy()
g["valid_utc"] = g.time_utc.dt.ceil("h")
obs = pd.concat([g[["station1", "valid_utc", "slp1_hpa"]].set_axis(["stid", "valid_utc", "slp"], axis=1),
                 g[["station2", "valid_utc", "slp2_hpa"]].set_axis(["stid", "valid_utc", "slp"], axis=1)])
obs = obs.dropna().groupby(["stid", "valid_utc"]).slp.first().reset_index()
m = P[(P.lead_day == 0) & P.stid.isin(pres_st)].merge(obs, on=["stid", "valid_utc"])
m["e"] = m[PV] - m.slp
ps = m.groupby("stid").agg(n=("e", "size"), bias=("e", "mean"), sd=("e", "std"),
                           r=("slp", lambda s: s.corr(m.loc[s.index, PV])), max_abs=("e", lambda e: e.abs().max()))
ps.round(2).to_csv(f"{OUT}/day0_pressure_vs_obs.csv")
print(ps.round(2).sort_values("sd").to_string())
print("by month (pooled bias / sd):")
print(m.groupby(m.valid_utc.dt.month).e.agg(["mean", "std"]).round(2).T.to_string())

# ---- 6b. day-0 wind vs matched max sustained wind ----
print("\n== 6b. day-0 wind vs observed matched max sustained (set stations) ==")
need = set(wind_st)
parts = []
for d_ in sorted(glob.glob(f"{DER}/gradients/matched/*/")):
    for f in sorted(glob.glob(d_ + "2026-*.parquet")):
        parts.append(pd.read_parquet(f, columns=["stid", "time_utc", "sustained_max_mps"],
                                     filters=[("stid", "in", sorted(need))]))
M = pd.concat(parts)
M["stid"] = M.stid.astype(str)
M["valid_utc"] = M.time_utc.dt.ceil("h").astype("datetime64[us, UTC]")
M = M.dropna().groupby(["stid", "valid_utc"]).sustained_max_mps.first().reset_index()
mw = P[(P.lead_day == 0) & P.stid.isin(wind_st)].merge(M, on=["stid", "valid_utc"])
wsum = mw.groupby("stid").apply(lambda x: pd.Series(dict(n=len(x), obs_mean=x.sustained_max_mps.mean(),
                                                         nwp_mean=x[WV].mean(),
                                                         r=x[WV].corr(x.sustained_max_mps))), include_groups=False)
wsum = wsum.join(cells.set_index("stid")[["dz_m", "dist_km"]])
wsum.round(3).to_csv(f"{OUT}/day0_wind_vs_obs.csv")
print(f"stations matched: {len(wsum)} of {len(wind_st)}; r quantiles:",
      wsum.r.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(2).to_dict())
print(f"NWP/obs mean ratio median {(wsum.nwp_mean / wsum.obs_mean).median():.2f}; stations with r < 0.2: "
      f"{(wsum.r < 0.2).sum()}")
print(wsum.sort_values("r").head(8).round(2).to_string())
print(f"\nwrote {OUT}/")
