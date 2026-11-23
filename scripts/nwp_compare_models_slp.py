#!/usr/bin/env python3
"""Q2: compare the SLP-convention bias of several NWP models (Open-Meteo day 0, fit year) against METAR SLP.

Inputs (no API calls): per model, the cached raw day-0 responses in /mnt/d/NWP/open_meteo/raw/<model>_calib/ and the
outputs of nwp_pressure_calibration.py in derived/verification/<model>/pressure_calibration/.

Outputs -> derived/verification/model_comparison/
  slp_bias_by_month_<year>.csv          station x month day-0 bias (pressure_msl - METAR SLP), per model
  station_summary_<year>.csv            per station and model: SLP bias/SD/seasonal/diurnal range, station-pressure
                                        check (bias, seasonal, diurnal range)
  cv_gradient_rmse_<year>.csv           leave-one-month-out CV day-0 gradient RMSE raw vs corrected, per pair/model
  gradient_bias_<year>.csv              day-0 gradient bias per pair, annual and JJA, per model
figures -> images/verification/model_comparison/
Usage: python nwp_compare_models_slp.py --models gfs ecmwf gem --year 2025
"""
import argparse, gzip, json, os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nwp_config as C

ROOT = "/mnt/d/socal"
DER = f"{ROOT}/data/v2/derived"
ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["gfs", "ecmwf", "gem"])
ap.add_argument("--year", type=int, default=2025)
a = ap.parse_args()
Y = a.year
OUT = f"{DER}/verification/model_comparison"
IMG = f"{ROOT}/images/verification/model_comparison"
os.makedirs(OUT, exist_ok=True)
os.makedirs(IMG, exist_ok=True)
pd.set_option("display.width", 220)

INK, MUTED, GRID = "#0b0b0b", "#8a8986", "#e5e4e0"
MCOL = {"gfs": "#eb6834", "ecmwf": "#2a78d6", "gem": "#1baf7a"}          # orange, blue, aqua (3-slot all-pairs set)
HILITE = {"KTPH": "#2a78d6", "KWMC": "#eb6834", "KRNO": "#1baf7a"}
LABEL = {m: m.upper() for m in a.models}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.edgecolor": MUTED, "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "font.size": 9, "axes.titlesize": 10, "legend.frameon": False, "lines.linewidth": 2})
KEY = ["LAX-TPH", "SFO-WMC", "SAC-RNO", "MFR-LMT"]

g = pd.read_parquet(f"{DER}/gradients/gradient_timeseries.parquet")
pairs = g[["gradient", "station1", "station2"]].drop_duplicates()
pairs = pairs[pairs.gradient != "SAC-RDD"]   # duplicate (sign-flipped) of RDD-SAC
stations = sorted(set(pairs.station1) | set(pairs.station2))
meta = pd.read_parquet(f"{ROOT}/data/v2/stations.parquet").drop_duplicates("stid").set_index("stid").loc[stations]
gy = g[(g.time_utc >= pd.Timestamp(f"{Y - 1}-12-31 23:00", tz="UTC")) & (g.time_utc < pd.Timestamp(f"{Y + 1}-01-01", tz="UTC"))].copy()
gy["valid_utc"] = gy.time_utc.dt.ceil("h")
obs = pd.concat([gy[["station1", "valid_utc", "slp1_hpa"]].set_axis(["stid", "valid_utc", "slp_obs"], axis=1),
                 gy[["station2", "valid_utc", "slp2_hpa"]].set_axis(["stid", "valid_utc", "slp_obs"], axis=1)])
obs = obs.dropna().groupby(["stid", "valid_utc"]).slp_obs.first().reset_index()
obs["valid_utc"] = obs.valid_utc.astype("datetime64[us, UTC]")

# ---------------------------------------------------------------- per-model day-0 errors
E = {}
for mdl in a.models:
    fr = []
    for s in stations:
        d = json.load(gzip.open(f"{C.OPEN_METEO_ROOT}/raw/{mdl}_calib/{s}_{Y}_day0.json.gz", "rt"))
        h = pd.DataFrame(d["hourly"])
        h["valid_utc"] = pd.to_datetime(h.time).dt.tz_localize("UTC").astype("datetime64[us, UTC]")
        h["stid"] = s
        fr.append(h[["stid", "valid_utc", "pressure_msl"]])
    m = pd.concat(fr).merge(obs, on=["stid", "valid_utc"])
    m["err"] = m.pressure_msl - m.slp_obs
    E[mdl] = m

# monthly bias table
mb = pd.concat({mdl: m.groupby(["stid", m.valid_utc.dt.month]).err.mean().unstack() for mdl, m in E.items()},
               names=["model", "stid"])
mb.round(2).to_csv(f"{OUT}/slp_bias_by_month_{Y}.csv")

# station summary: SLP stats + station-pressure check from the calibration outputs
rows = []
for mdl, m in E.items():
    chk = pd.read_csv(f"{DER}/verification/{mdl}/pressure_calibration/station_pressure_check_{Y}.csv", index_col=0)
    for s, x in m.groupby("stid"):
        rows.append(dict(model=mdl, stid=s, elev_m=meta.elevation_m[s], slp_bias=x.err.mean(), slp_sd=x.err.std(),
                         slp_seasonal_range=np.ptp(x.groupby(x.valid_utc.dt.month).err.mean()),
                         slp_diurnal_range=np.ptp(x.groupby(x.valid_utc.dt.hour).err.mean()),
                         stnp_bias=chk.stnp_bias[s], stnp_sd=chk.stnp_sd[s],
                         stnp_seasonal_range=chk.stnp_seasonal_range[s], stnp_diurnal_range=chk.stnp_diurnal_range[s]))
ss = pd.DataFrame(rows)
ss.round(3).to_csv(f"{OUT}/station_summary_{Y}.csv", index=False)

# CV gradient RMSE (from calibration outputs)
cvs = []
for mdl in a.models:
    cv = pd.read_csv(f"{DER}/verification/{mdl}/pressure_calibration/cv_gradient_errors_{Y}.csv", header=[0, 1], index_col=0)
    cv = cv.drop(index=[i for i in cv.index if not isinstance(i, str) or i in ("gradient", "SAC-RDD")], errors="ignore")
    cvs.append(pd.DataFrame({"model": mdl, "gradient": cv.index, "rmse_raw": cv[("rmse", "pressure_msl")].astype(float).values,
                             "rmse_corrected_cv": cv[("rmse", "corr_cv")].astype(float).values}))
CV = pd.concat(cvs, ignore_index=True)
CV.round(3).to_csv(f"{OUT}/cv_gradient_rmse_{Y}.csv", index=False)

# gradient bias annual / JJA
gb = []
for mdl, m in E.items():
    w = m.pivot_table(index="valid_utc", columns="stid", values=["pressure_msl", "slp_obs"])
    for _, (gn, s1, s2) in pairs.iterrows():
        e = ((w["pressure_msl"][s1] - w["pressure_msl"][s2]) - (w["slp_obs"][s1] - w["slp_obs"][s2])).dropna()
        jja = e[e.index.month.isin([6, 7, 8])]
        gb.append(dict(model=mdl, gradient=gn, bias_annual=e.mean(), bias_jja=jja.mean(), rmse_annual=np.sqrt((e ** 2).mean())))
GB = pd.DataFrame(gb)
GB.round(3).to_csv(f"{OUT}/gradient_bias_{Y}.csv", index=False)

# ---------------------------------------------------------------- figures
hi = meta.index[meta.elevation_m > 580].tolist()
hi = sorted(hi, key=lambda s: meta.elevation_m[s])

# 1. monthly SLP bias at high stations, one panel per model
fig, axes = plt.subplots(1, len(a.models), figsize=(5 * len(a.models), 4.3), sharey=True)
for ax, mdl in zip(np.atleast_1d(axes), a.models):
    t = mb.loc[mdl]
    for s in hi:
        if s in HILITE:
            continue
        ax.plot(t.columns, t.loc[s], color=MUTED, lw=0.9, alpha=0.8)
    for s, col in HILITE.items():
        ax.plot(t.columns, t.loc[s], color=col, lw=2, marker="o", ms=4, label=s)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set(title=f"{LABEL[mdl]}: day-0 pressure_msl - METAR SLP, {Y}", xlabel="month", xticks=range(1, 13))
np.atleast_1d(axes)[0].set_ylabel("hPa")
np.atleast_1d(axes)[0].plot([], [], color=MUTED, lw=0.9, label="other stations > 580 m")
np.atleast_1d(axes)[0].legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(f"{IMG}/high_station_slp_bias_by_month_{Y}.png", dpi=130)
plt.close(fig)

# 2. per-station small multiples, models overlaid
n = len(hi)
nc = 3
nr = int(np.ceil(n / nc))
fig, axes = plt.subplots(nr, nc, figsize=(12, 3 * nr), sharex=True, sharey=True)
for ax, s in zip(axes.flat, hi):
    for mdl in a.models:
        ax.plot(mb.loc[mdl].columns, mb.loc[mdl].loc[s], color=MCOL[mdl], marker="o", ms=3, lw=1.8, label=LABEL[mdl])
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_title(f"{s} ({meta.elevation_m[s]:.0f} m)", fontsize=9)
for ax in axes.flat[n:]:
    ax.set_visible(False)
axes.flat[0].legend(fontsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel("SLP bias (hPa)")
for ax in axes[-1, :]:
    ax.set_xlabel("month")
fig.suptitle(f"Day-0 SLP bias vs METAR by month, {Y}", x=0.01, ha="left")
fig.tight_layout()
fig.savefig(f"{IMG}/slp_bias_by_month_per_station_{Y}.png", dpi=130)
plt.close(fig)

# 3. seasonal and diurnal range of SLP bias, plus station-pressure seasonal range, by station
order = sorted(stations, key=lambda s: meta.elevation_m[s])
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
off = {m: (i - (len(a.models) - 1) / 2) * 0.22 for i, m in enumerate(a.models)}
for ax, col, ttl in zip(axes, ("slp_seasonal_range", "slp_diurnal_range", "stnp_seasonal_range"),
                        ("seasonal range of monthly SLP bias", "diurnal range of hourly SLP bias",
                         "seasonal range of monthly STATION-pressure bias (Open-Meteo surface_pressure vs obs)")):
    for mdl in a.models:
        x = ss[ss.model == mdl].set_index("stid").reindex(order)
        ax.scatter(np.arange(len(order)) + off[mdl], x[col], color=MCOL[mdl], s=28, label=LABEL[mdl], zorder=3)
    ax.set(title=ttl, ylabel="hPa")
    ax.set_ylim(bottom=0)
axes[0].legend(fontsize=8, ncol=3)
axes[-1].set_xticks(range(len(order)), [f"{s}\n{meta.elevation_m[s]:.0f} m" for s in order], rotation=90, fontsize=7.5)
fig.tight_layout()
fig.savefig(f"{IMG}/bias_ranges_by_station_{Y}.png", dpi=130)
plt.close(fig)

# 4. CV gradient RMSE raw vs corrected (dumbbells)
rowsp = ["median"] + KEY
med = CV.groupby("model")[["rmse_raw", "rmse_corrected_cv"]].median().reset_index().assign(gradient="median")
D = pd.concat([med, CV[CV.gradient.isin(KEY)]], ignore_index=True)
fig, ax = plt.subplots(figsize=(8, 5.5))
yt, yl = [], []
for i, gname in enumerate(rowsp):
    for j, mdl in enumerate(a.models):
        y = i * (len(a.models) + 1) + j
        r = D[(D.gradient == gname) & (D.model == mdl)].iloc[0]
        ax.plot([r.rmse_raw, r.rmse_corrected_cv], [y, y], color=MCOL[mdl], lw=2)
        ax.scatter(r.rmse_raw, y, s=40, facecolor="white", edgecolor=MCOL[mdl], lw=1.8, zorder=3)
        ax.scatter(r.rmse_corrected_cv, y, s=40, color=MCOL[mdl], zorder=3)
        yt.append(y)
        yl.append(f"{gname}  {LABEL[mdl]}")
ax.set_yticks(yt, yl, fontsize=8)
ax.invert_yaxis()
ax.set(xlabel="day-0 gradient RMSE (hPa)", title=f"Gradient RMSE vs METAR, {Y}: raw (open) -> corrected, month-held-out CV (filled)")
ax.set_xlim(left=0)
fig.tight_layout()
fig.savefig(f"{IMG}/cv_gradient_rmse_{Y}.png", dpi=130)
plt.close(fig)

# 5. gradient bias per pair, annual and JJA
po = GB[GB.model == "gfs"].sort_values("bias_jja").gradient.tolist() if "gfs" in a.models else sorted(GB.gradient.unique())
fig, axes = plt.subplots(1, 2, figsize=(11, 6.5), sharey=True)
offy = {m: (i - (len(a.models) - 1) / 2) * 0.25 for i, m in enumerate(a.models)}
for ax, col, ttl in zip(axes, ("bias_annual", "bias_jja"), ("annual", "June-August")):
    for mdl in a.models:
        x = GB[GB.model == mdl].set_index("gradient").reindex(po)
        ax.scatter(x[col], np.arange(len(po)) + offy[mdl], color=MCOL[mdl], s=26, label=LABEL[mdl], zorder=3)
    ax.axvline(0, color=INK, lw=0.7)
    ax.set(title=f"day-0 gradient bias (NWP - METAR), {ttl} {Y}", xlabel="hPa")
axes[0].set_yticks(range(len(po)), po)
axes[0].invert_yaxis()
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{IMG}/gradient_bias_by_pair_{Y}.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- print summary
print("== monthly SLP bias, selected high stations (Jan / Apr / Jul / Aug / Oct) ==")
print(mb.loc[(slice(None), ["KTPH", "KWMC", "KRNO", "KLMT", "KLAS", "KNID", "KLAX"]), [1, 4, 7, 8, 10]].round(1).to_string())
h = ss[ss.elev_m > 580]
print("\n== high stations (> 580 m), median over stations ==")
print(h.groupby("model")[["slp_bias", "slp_sd", "slp_seasonal_range", "slp_diurnal_range", "stnp_bias", "stnp_sd",
                          "stnp_seasonal_range", "stnp_diurnal_range"]].median().loc[a.models].round(2).to_string())
print("\n== low stations (< 160 m), median ==")
print(ss[ss.elev_m < 160].groupby("model")[["slp_bias", "slp_sd", "slp_seasonal_range", "slp_diurnal_range"]].median()
      .loc[a.models].round(2).to_string())
print("\n== CV gradient RMSE (median over pairs and key pairs) ==")
print(D.pivot_table(index="gradient", columns="model", values=["rmse_raw", "rmse_corrected_cv"]).loc[rowsp].round(2).to_string())
print("\n== gradient bias annual / JJA, key pairs and median |bias| ==")
print(GB[GB.gradient.isin(KEY)].pivot_table(index="gradient", columns="model", values=["bias_annual", "bias_jja"]).round(2).to_string())
print(GB.assign(aa=GB.bias_annual.abs(), aj=GB.bias_jja.abs()).groupby("model")[["aa", "aj", "rmse_annual"]].median().round(2).to_string())
print(f"\nwrote {OUT}/ and {IMG}/")
