"""Pilot pressure check: Open-Meteo GFS pressure_msl (previous-runs, lead days 0-7) vs METAR SLP, Jan 2025.

Fetches pressure_msl_previous_day{0..7} for the gradient stations (about 40 weighted calls), matches obs SLP
(obs at :50-:56 -> next full hour), and reports per-station and per-pair errors by lead day.

Usage: ~/.venv/analysis_env/bin/python nwp_pressure_check_pilot.py [--start 2025-01-01 --end 2025-01-31]
"""
import argparse, gzip, json, os, time, urllib.parse, urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/mnt/d/socal"
RAW = "/mnt/d/NWP/open_meteo/pilot/pressure_check"
OUT = f"{ROOT}/data/v2/derived/verification/gfs/pressure_check_pilot"
IMG = f"{ROOT}/images/verification/gfs/pressure_check_pilot"
URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODEL = "gfs_global"
DAYS = range(8)

ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2025-01-01")
ap.add_argument("--end", default="2025-01-31")
a = ap.parse_args()
tag = f"{a.start.replace('-', '')}_{a.end.replace('-', '')}"
for d in (RAW, OUT, IMG):
    os.makedirs(d, exist_ok=True)

# ---- observed SLP per station, hourly (obs at :50-:56 -> ceil to the hour) ----
g = pd.read_parquet(f"{ROOT}/data/v2/derived/gradients/gradient_timeseries.parquet")
t0, t1 = pd.Timestamp(a.start, tz="UTC"), pd.Timestamp(a.end, tz="UTC") + pd.Timedelta(days=1)
g = g[(g.time_utc >= t0 - pd.Timedelta(hours=1)) & (g.time_utc < t1)].copy()
g["valid"] = g.time_utc.dt.ceil("h")
pairs = g[["gradient", "station1", "station2"]].drop_duplicates()
obs = pd.concat([g[["station1", "valid", "slp1_hpa"]].set_axis(["stid", "valid", "slp_obs"], axis=1),
                 g[["station2", "valid", "slp2_hpa"]].set_axis(["stid", "valid", "slp_obs"], axis=1)])
obs = obs.dropna().groupby(["stid", "valid"]).slp_obs.first().reset_index()
stations = sorted(set(pairs.station1) | set(pairs.station2))
meta = pd.read_parquet(f"{ROOT}/data/v2/stations.parquet").set_index("stid").loc[stations]

# ---- fetch NWP (cached per station) ----
rows, cost_total = [], 0.0
for s in stations:
    fn = f"{RAW}/{s}_{tag}.json.gz"
    if not os.path.exists(fn):
        p = dict(latitude=meta.lat[s], longitude=meta.lon[s], models=MODEL, timezone="GMT",
                 start_date=a.start, end_date=a.end,
                 hourly=",".join("pressure_msl" if d == 0 else f"pressure_msl_previous_day{d}" for d in DAYS))
        with urllib.request.urlopen(URL + "?" + urllib.parse.urlencode(p), timeout=120) as r:
            body = r.read(); cost_total += float(r.headers.get("X-RateLimit-Cost", "nan"))
        with gzip.open(fn + ".part", "wb") as f:
            f.write(body)
        os.replace(fn + ".part", fn)
        time.sleep(0.5)
    d = json.load(gzip.open(fn, "rt"))
    h = pd.DataFrame(d["hourly"])
    h["valid"] = pd.to_datetime(h.time).dt.tz_localize("UTC")
    for k in DAYS:
        col = "pressure_msl" if k == 0 else f"pressure_msl_previous_day{k}"
        rows.append(pd.DataFrame(dict(stid=s, valid=h.valid, lead_day=k, slp_nwp=h[col],
                                      grid_lat=d["latitude"], grid_lon=d["longitude"], grid_elev_m=d["elevation"])))
print(f"weighted calls used this run: {cost_total:.1f}")
nwp = pd.concat(rows)
m = nwp.merge(obs, on=["stid", "valid"], how="inner").dropna(subset=["slp_nwp", "slp_obs"])
m["err"] = m.slp_nwp - m.slp_obs

# ---- station errors ----
st = (m.groupby(["stid", "lead_day"]).err
      .agg(n="size", bias="mean", sd="std", rmse=lambda e: np.sqrt((e ** 2).mean())).reset_index())
corr = m.groupby(["stid", "lead_day"]).apply(lambda x: x.slp_nwp.corr(x.slp_obs), include_groups=False)
st = st.merge(corr.rename("r").reset_index(), on=["stid", "lead_day"])
st = st.merge(meta[["elevation_m"]].rename(columns={"elevation_m": "stn_elev_m"}), left_on="stid", right_index=True)
st = st.merge(nwp.groupby("stid").grid_elev_m.first(), left_on="stid", right_index=True)
st.round(3).to_csv(f"{OUT}/station_errors_{tag}.csv", index=False)

# diurnal bias at day 0 (temperature-dependent reduction shows up here)
diur = m[m.lead_day == 0].assign(hour=lambda x: x.valid.dt.hour).groupby(["stid", "hour"]).err.mean().unstack()
diur.round(2).to_csv(f"{OUT}/station_bias_by_hour_day0_{tag}.csv")

# ---- pair (gradient) errors: grad = slp(station1) - slp(station2) ----
w = m.pivot_table(index=["valid", "lead_day"], columns="stid", values=["slp_nwp", "slp_obs"])
pr = []
for _, (gname, s1, s2) in pairs.iterrows():
    gn = w["slp_nwp"][s1] - w["slp_nwp"][s2]
    go = w["slp_obs"][s1] - w["slp_obs"][s2]
    df = pd.DataFrame(dict(gn=gn, go=go)).dropna().reset_index()
    for k, x in df.groupby("lead_day"):
        e = x.gn - x.go
        pr.append(dict(gradient=gname, lead_day=k, n=len(x), obs_sd=x.go.std(), bias=e.mean(), sd=e.std(),
                       rmse=np.sqrt((e ** 2).mean()), r=x.gn.corr(x.go),
                       skill_vs_clim=1 - (e ** 2).mean() / x.go.var(), sign_agree=(np.sign(x.gn) == np.sign(x.go)).mean()))
pr = pd.DataFrame(pr)
pr.round(3).to_csv(f"{OUT}/pair_errors_{tag}.csv", index=False)

# ---- print summary ----
pd.set_option("display.width", 200)
print("\nStation errors at day 0 and day 3 (hPa, NWP - obs):")
tab = st[st.lead_day.isin([0, 3])].pivot(index="stid", columns="lead_day", values=["bias", "sd", "r"])
tab.columns = [f"{v}_d{k}" for v, k in tab.columns]
print(tab.join(st.groupby("stid")[["stn_elev_m", "grid_elev_m"]].first()).round(2).sort_values("stn_elev_m").to_string())
print("\nDay-0 bias diurnal range (max-min of hourly mean bias), hPa:")
print((diur.max(axis=1) - diur.min(axis=1)).round(2).sort_values(ascending=False).to_string())
print("\nGradient errors at day 0 (hPa):")
print(pr[pr.lead_day == 0].set_index("gradient")[["n", "obs_sd", "bias", "sd", "rmse", "r", "skill_vs_clim", "sign_agree"]]
      .round(2).sort_values("rmse").to_string())
print("\nGradient RMSE by lead day (median over pairs) and pooled r:")
print(pr.groupby("lead_day").agg(rmse_med=("rmse", "median"), sd_err_med=("sd", "median"),
                                 bias_abs_med=("bias", lambda b: b.abs().median()), r_med=("r", "median"),
                                 skill_med=("skill_vs_clim", "median")).round(2).to_string())

# ---- figures ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for gname, x in pr.groupby("gradient"):
    ax[0].plot(x.lead_day, x.rmse, color="0.7", lw=0.8)
med = pr.groupby("lead_day").rmse.median()
ax[0].plot(med.index, med.values, color="C0", lw=2.5, label="median over pairs")
ax[0].set(xlabel="lead day", ylabel="gradient RMSE (hPa)", title="NWP gradient error vs METAR SLP gradient")
ax[0].legend()
d0 = pr[pr.lead_day == 0].sort_values("rmse")
ax[1].barh(d0.gradient, d0.obs_sd, color="0.85", label="obs gradient SD")
ax[1].barh(d0.gradient, d0.rmse, color="C0", height=0.5, label="day-0 RMSE")
ax[1].scatter(d0.bias, d0.gradient, color="C3", zorder=3, s=15, label="day-0 bias")
ax[1].axvline(0, color="k", lw=0.5)
ax[1].set(xlabel="hPa", title=f"Day-0 gradient error, {a.start} to {a.end}")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{IMG}/gradient_error_{tag}.png", dpi=130)

# ---- correction potential: fit on first half of the period, test on second half ----
# corrections are per station, estimated from day-0 errors and applied to all lead days
mid = m.valid.min() + (m.valid.max() - m.valid.min()) / 2
fit, test = m[(m.valid < mid) & (m.lead_day == 0)], m[m.valid >= mid].copy()
test["hour"] = test.valid.dt.hour
b_st = fit.groupby("stid").err.mean()
b_hr = fit.assign(hour=fit.valid.dt.hour).groupby(["stid", "hour"]).err.mean()
test["c_station"] = test.slp_nwp - test.stid.map(b_st)
test["c_station_hour"] = test.slp_nwp - b_hr.reindex(pd.MultiIndex.from_frame(test[["stid", "hour"]])).values
wt = test.pivot_table(index=["valid", "lead_day"], columns="stid", values=["slp_nwp", "c_station", "c_station_hour", "slp_obs"])
cr = []
for _, (gname, s1, s2) in pairs.iterrows():
    go = wt["slp_obs"][s1] - wt["slp_obs"][s2]
    for v in ["slp_nwp", "c_station", "c_station_hour"]:
        e = (wt[v][s1] - wt[v][s2] - go).dropna()
        for k, x in e.groupby(level="lead_day"):
            cr.append(dict(gradient=gname, lead_day=k, version=v, rmse=np.sqrt((x ** 2).mean())))
cr = pd.DataFrame(cr).pivot_table(index=["gradient", "lead_day"], columns="version", values="rmse").reset_index()
cr.round(3).to_csv(f"{OUT}/correction_potential_{tag}.csv", index=False)
print("\nCorrection potential (fit first half, test second half), gradient RMSE hPa, median over pairs:")
print(cr.groupby("lead_day")[["slp_nwp", "c_station", "c_station_hour"]].median().round(2).to_string())
print("\nDay 0, per pair (test half):")
print(cr[cr.lead_day == 0].set_index("gradient")[["slp_nwp", "c_station", "c_station_hour"]].round(2)
      .sort_values("slp_nwp").to_string())
print(f"\nwrote {OUT}/ and {IMG}/")
