"""SLP-convention calibration for NWP pressure (GBT inputs) + station-pressure check.

Background (docs/nwp_verification_plan.md, "Fetch and data QC"): GFS pressure_msl at high stations is biased low
vs METAR SLP, seasonally and diurnally (current-temperature reduction vs METAR's 12-h mean + plateau correction).

Steps
 1. Fetch day-0 pressure_msl, surface_pressure (elevation = station elevation) and temperature_2m for the
    gradient stations over the fit year (cached raw JSON).
 2. Station-pressure check: Open-Meteo surface_pressure at station height vs observed station pressure
    (obs_hourly pressure_mean_hpa; hour bin H is centred ~H:30, compared with the mean of NWP H and H+1).
 3. Fit per station: err = pressure_msl - METAR SLP ~ seasonal harmonics (2) + diurnal harmonics (2)
    + first seasonal x first diurnal interaction. Form fixed before fitting.
    Leave-one-month-out CV within the fit year -> decision rule: use the correction if CV gradient RMSE is lower.
 4. Apply to the verification-period points (all lead days; the bias depends on valid time, not lead) and report
    raw vs corrected gradient error per pair and lead. Report only; nothing is tuned on the verification period.

Usage: python nwp_pressure_calibration.py --model gfs --fit-year 2025 --points-tag 20260101_20261031
"""
import argparse, gzip, json, os, time, urllib.parse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nwp_config as C
from nwp_fetch_openmeteo import request, log

ROOT = "/mnt/d/socal"
DER = f"{ROOT}/data/v2/derived"

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gfs")
ap.add_argument("--fit-year", type=int, default=2025)
ap.add_argument("--points-tag", default="20260101_20261031")
a = ap.parse_args()
cfg = C.MODELS[a.model]
Y = a.fit_year
RAW = f"{C.OPEN_METEO_ROOT}/raw/{a.model}_calib"
OUT = f"{DER}/verification/{a.model}/pressure_calibration"
IMG = f"{ROOT}/images/verification/{a.model}/pressure_calibration"
for d in (RAW, OUT, IMG):
    os.makedirs(d, exist_ok=True)
pd.set_option("display.width", 220)

g = pd.read_parquet(f"{DER}/gradients/gradient_timeseries.parquet")
pairs = g[["gradient", "station1", "station2"]].drop_duplicates()
stations = sorted(set(pairs.station1) | set(pairs.station2))
meta = pd.read_parquet(f"{ROOT}/data/v2/stations.parquet").drop_duplicates("stid").set_index("stid").loc[stations]

# ---------------- 1. fetch ----------------
spent = 0.0
frames = []
for s in stations:
    fn = f"{RAW}/{s}_{Y}_day0.json.gz"
    if not os.path.exists(fn):
        p = dict(latitude=meta.lat[s], longitude=meta.lon[s], models=cfg["open_meteo_model"], timezone="GMT",
                 hourly="pressure_msl,surface_pressure,temperature_2m", elevation=round(meta.elevation_m[s], 1),
                 start_date=f"{Y}-01-01", end_date=f"{Y}-12-31")
        body, cost, _ = request(C.PREVIOUS_RUNS_URL, p)
        with gzip.open(fn + ".part", "wb") as f:
            f.write(body)
        os.replace(fn + ".part", fn)
        spent += cost
        log(f"{s} fetched cost={cost:.2f}")
        time.sleep(cost * 3600 / 4500)
    d = json.load(gzip.open(fn, "rt"))
    h = pd.DataFrame(d["hourly"])
    h["valid_utc"] = pd.to_datetime(h.time).dt.tz_localize("UTC")
    h["stid"] = s
    frames.append(h.drop(columns="time"))
log(f"weighted calls this run: {spent:.1f}")
N = pd.concat(frames)
assert N[["pressure_msl", "surface_pressure"]].notna().all().all(), "missing NWP values"

# ---------------- METAR SLP (training convention) ----------------
gy = g[(g.time_utc >= pd.Timestamp(f"{Y - 1}-12-31 23:00", tz="UTC")) & (g.time_utc < pd.Timestamp(f"{Y + 1}-01-01", tz="UTC"))].copy()
gy["valid_utc"] = gy.time_utc.dt.ceil("h")
obs = pd.concat([gy[["station1", "valid_utc", "slp1_hpa"]].set_axis(["stid", "valid_utc", "slp_obs"], axis=1),
                 gy[["station2", "valid_utc", "slp2_hpa"]].set_axis(["stid", "valid_utc", "slp_obs"], axis=1)])
obs = obs.dropna().groupby(["stid", "valid_utc"]).slp_obs.first().reset_index()
m = N.merge(obs, on=["stid", "valid_utc"])
m["err"] = m.pressure_msl - m.slp_obs

# ---------------- 2. station-pressure check ----------------
sp = []
for mon in range(1, 13):
    f = f"{ROOT}/data/v2/obs_hourly/{Y}/{mon:02d}"
    for st_file in os.listdir(f):
        sp.append(pd.read_parquet(f"{f}/{st_file}", columns=["stid", "time_utc", "pressure_mean_hpa"],
                                  filters=[("stid", "in", stations)]))
SP = pd.concat(sp).dropna()
SP["stid"] = SP.stid.astype(str)
SP["time_utc"] = SP.time_utc.astype("datetime64[us, UTC]")
Ns = N.set_index(["stid", "valid_utc"]).surface_pressure
Nmid = (Ns + Ns.groupby(level="stid").shift(-1)) / 2       # mean of H and H+1, keyed on H
SP = SP.join(Nmid.rename("sfc_nwp"), on=["stid", "time_utc"]).dropna()
SP["err"] = SP.sfc_nwp - SP.pressure_mean_hpa
spc = SP.groupby("stid").err.agg(n="size", bias="mean", sd="std")
spc["seasonal_range"] = SP.groupby(["stid", SP.time_utc.dt.month]).err.mean().groupby("stid").agg(lambda x: x.max() - x.min())
spc["diurnal_range"] = SP.groupby(["stid", SP.time_utc.dt.hour]).err.mean().groupby("stid").agg(lambda x: x.max() - x.min())
slpc = m.groupby("stid").err.agg(slp_bias="mean", slp_sd="std")
slpc["slp_seasonal_range"] = m.groupby(["stid", m.valid_utc.dt.month]).err.mean().groupby("stid").agg(lambda x: x.max() - x.min())
slpc["slp_diurnal_range"] = m.groupby(["stid", m.valid_utc.dt.hour]).err.mean().groupby("stid").agg(lambda x: x.max() - x.min())
chk = spc.add_prefix("stnp_").join(slpc).join(meta.elevation_m.rename("elev_m")).sort_values("elev_m")
chk.round(2).to_csv(f"{OUT}/station_pressure_check_{Y}.csv")
print(f"\n== station pressure (Open-Meteo surface_pressure at station elev) vs observed station pressure, {Y} ==")
print("   compared with the SLP error (pressure_msl vs METAR SLP), same year")
print(chk.round(2).to_string())
SP.assign(month=SP.time_utc.dt.month).groupby(["stid", "month"]).err.mean().unstack().round(2) \
    .to_csv(f"{OUT}/station_pressure_bias_by_month_{Y}.csv")
m.assign(month=m.valid_utc.dt.month).groupby(["stid", "month"]).err.mean().unstack().round(2) \
    .to_csv(f"{OUT}/slp_bias_by_month_{Y}.csv")


# ---------------- 3. fit (form fixed in advance) ----------------
def design(t):
    doy = 2 * np.pi * t.dt.day_of_year.to_numpy() / 365.25
    hr = 2 * np.pi * t.dt.hour.to_numpy() / 24
    cols = [np.ones(len(t)), np.cos(doy), np.sin(doy), np.cos(2 * doy), np.sin(2 * doy),
            np.cos(hr), np.sin(hr), np.cos(2 * hr), np.sin(2 * hr)]
    cols += [np.cos(doy) * np.cos(hr), np.cos(doy) * np.sin(hr), np.sin(doy) * np.cos(hr), np.sin(doy) * np.sin(hr)]
    return np.column_stack(cols)


TERMS = ["const", "cos_doy", "sin_doy", "cos_2doy", "sin_2doy", "cos_hr", "sin_hr", "cos_2hr", "sin_2hr",
         "cdoy_chr", "cdoy_shr", "sdoy_chr", "sdoy_shr"]


def fit(df):
    X = design(df.valid_utc)
    beta, *_ = np.linalg.lstsq(X, df.err.to_numpy(), rcond=None)
    return beta


coef = {s: fit(x).tolist() for s, x in m.groupby("stid")}
with open(f"{OUT}/coefficients_{Y}.json", "w") as f:
    json.dump({"model": a.model, "fit_year": Y, "target": "pressure_msl - METAR SLP (hPa)", "terms": TERMS,
               "apply": "corrected = pressure_msl - X(valid_utc) @ beta, all lead days",
               "coefficients": coef}, f, indent=1)

# leave-one-month-out CV in the fit year
m["corr_cv"] = np.nan
for mon in range(1, 13):
    te = m.valid_utc.dt.month == mon
    for s, x in m[~te].groupby("stid"):
        b = fit(x)
        idx = te & (m.stid == s)
        m.loc[idx, "corr_cv"] = m.loc[idx, "pressure_msl"] - design(m.loc[idx, "valid_utc"]) @ b


def pair_errors(df, raw_col, cols, obs_col="slp_obs", by=None):
    rows = []
    w = df.pivot_table(index=[c for c in ["valid_utc", by] if c], columns="stid", values=[raw_col, *cols, obs_col])
    for _, (gn, s1, s2) in pairs.iterrows():
        go = w[obs_col][s1] - w[obs_col][s2]
        for c in [raw_col, *cols]:
            e = (w[c][s1] - w[c][s2] - go).dropna()
            grp = e.groupby(level=by) if by else [(None, e)]
            for k, x in grp:
                rows.append(dict(gradient=gn, lead_day=k, version=c, bias=x.mean(), rmse=np.sqrt((x ** 2).mean()),
                                 n=len(x)))
    return pd.DataFrame(rows)


cv = pair_errors(m, "pressure_msl", ["corr_cv"])
cvt = cv.pivot_table(index="gradient", columns="version", values=["bias", "rmse"])
cvt.round(2).to_csv(f"{OUT}/cv_gradient_errors_{Y}.csv")
med = cv.groupby("version")[["rmse"]].median()
print(f"\n== leave-one-month-out CV in {Y}, day-0 gradient error (hPa) ==")
print(cvt.round(2).sort_values(("rmse", "pressure_msl"), ascending=False).to_string())
print(f"median gradient RMSE: raw {med.loc['pressure_msl', 'rmse']:.2f}, corrected (CV) {med.loc['corr_cv', 'rmse']:.2f}; "
      f"pairs improved {(cvt['rmse']['corr_cv'] < cvt['rmse']['pressure_msl']).sum()}/{len(cvt)}")
use = med.loc["corr_cv", "rmse"] < med.loc["pressure_msl", "rmse"]
print(f"DECISION RULE (CV gradient RMSE lower): {'USE correction' if use else 'do NOT use correction'}")

# ---------------- 4. apply to verification period (report only) ----------------
PTS = f"{C.OPEN_METEO_ROOT}/points/{a.model}/points_{a.points_tag}.parquet"
HAVE_PTS = os.path.exists(PTS)   # no verification-period points yet (e.g. a model fetched for calibration only)
if not HAVE_PTS:
    print(f"\nno points file {PTS}: skipping the verification-period report")
else:
    P = pd.read_parquet(PTS,
                        columns=["valid_utc", "stid", "lead_day", "pressure_msl_hpa"], filters=[("stid", "in", stations)])
    P["p_corr"] = np.nan
    for s, idx in P.groupby("stid").groups.items():
        P.loc[idx, "p_corr"] = P.loc[idx, "pressure_msl_hpa"] - design(P.loc[idx, "valid_utc"]) @ np.array(coef[s])
    g2 = g[g.time_utc >= P.valid_utc.min() - pd.Timedelta(hours=1)].copy()
    g2["valid_utc"] = g2.time_utc.dt.ceil("h")
    o2 = pd.concat([g2[["station1", "valid_utc", "slp1_hpa"]].set_axis(["stid", "valid_utc", "slp_obs"], axis=1),
                    g2[["station2", "valid_utc", "slp2_hpa"]].set_axis(["stid", "valid_utc", "slp_obs"], axis=1)])
    o2 = o2.dropna().groupby(["stid", "valid_utc"]).slp_obs.first().reset_index()
    o2["valid_utc"] = o2.valid_utc.astype("datetime64[us, UTC]")
    P2 = P.merge(o2, on=["stid", "valid_utc"])
    ver = pair_errors(P2, "pressure_msl_hpa", ["p_corr"], by="lead_day")
    ver.round(3).to_csv(f"{OUT}/verification_period_gradient_errors.csv", index=False)
    print("\n== verification period (report only): gradient error by lead day, median over pairs ==")
    print(ver.pivot_table(index="lead_day", columns="version", values="rmse", aggfunc="median").round(2)
          .join(ver.assign(ab=ver.bias.abs()).pivot_table(index="lead_day", columns="version", values="ab",
                                                          aggfunc="median").round(2), rsuffix="_absbias").to_string())
    print("\nday 0 per pair (verification period):")
    print(ver[ver.lead_day == 0].pivot_table(index="gradient", columns="version", values=["bias", "rmse"]).round(2)
          .sort_values(("rmse", "pressure_msl_hpa"), ascending=False).to_string())
    sb = P2[P2.lead_day == 0].assign(e_raw=lambda x: x.pressure_msl_hpa - x.slp_obs, e_cor=lambda x: x.p_corr - x.slp_obs)
    sbm = sb.groupby(["stid", sb.valid_utc.dt.month])[["e_raw", "e_cor"]].mean()
    sbm.round(2).to_csv(f"{OUT}/verification_period_station_bias_by_month_day0.csv")

# ---------------- figures ----------------
hi = chk.index[chk.elev_m > 580].tolist()
fig, ax = plt.subplots(1, 3 if HAVE_PTS else 2, figsize=(16 if HAVE_PTS else 11, 4.5), sharey=True)
bm = m.groupby(["stid", m.valid_utc.dt.month]).err.mean().unstack()
spm = SP.groupby(["stid", SP.time_utc.dt.month]).err.mean().unstack()
for s in hi:
    ax[0].plot(bm.columns, bm.loc[s], marker="o", ms=3, label=s)
    ax[1].plot(spm.columns, spm.loc[s], marker="o", ms=3, label=s)
    if HAVE_PTS:
        x = sbm.loc[s]
        ax[2].plot(x.index, x.e_cor, marker="o", ms=3, label=s)
ax[0].set(title=f"SLP: pressure_msl - METAR SLP ({Y})", xlabel="month", ylabel="hPa")
ax[1].set(title=f"Station pressure: Open-Meteo sfc - obs ({Y})", xlabel="month")
if HAVE_PTS:
    ax[2].set(title="SLP after correction, verification period day 0", xlabel="month")
for x_ in ax:
    x_.axhline(0, color="k", lw=0.6)
ax[0].legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(f"{IMG}/high_station_bias_by_month.png", dpi=130)
print(f"\nwrote {OUT}/ and {IMG}/")
