"""Free test of a model-agnostic, METAR-style SLP reduction from Open-Meteo station pressure (no API calls).

Uses the cached 2025 day-0 GFS data from nwp_pressure_calibration.py (surface_pressure at station elevation,
temperature_2m at station elevation, pressure_msl) and compares against METAR SLP.

Versions:
  raw          Open-Meteo pressure_msl
  manual_now   ps * exp(g h / (Rd Tm)), Tm = T_now + gamma h / 2    (should reproduce raw: round-trip check)
  manual_12h   same with T = (T_now + T_12h_ago) / 2                 (METAR-like temperature)
  manual_12h_off  manual_12h minus a per-station constant offset (leave-one-month-out), a stand-in for the
                  station-specific plateau correction
  harmonic_cv  the fitted seasonal/diurnal correction (nwp_pressure_calibration.py), leave-one-month-out
Metrics: per-station bias / SD / seasonal and diurnal range of the error; per-pair day-0 gradient bias and RMSE.

Usage: python nwp_slp_reduction_test.py --model gfs --year 2025
"""
import argparse, gzip, json, os

import numpy as np
import pandas as pd

import nwp_config as C

ROOT = "/mnt/d/socal"
DER = f"{ROOT}/data/v2/derived"
G, RD, GAMMA = 9.80665, 287.05, 0.0065

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gfs")
ap.add_argument("--year", type=int, default=2025)
a = ap.parse_args()
Y = a.year
RAW = f"{C.OPEN_METEO_ROOT}/raw/{a.model}_calib"
OUT = f"{DER}/verification/{a.model}/pressure_calibration"
pd.set_option("display.width", 220)

g = pd.read_parquet(f"{DER}/gradients/gradient_timeseries.parquet")
pairs = g[["gradient", "station1", "station2"]].drop_duplicates()
stations = sorted(set(pairs.station1) | set(pairs.station2))
meta = pd.read_parquet(f"{ROOT}/data/v2/stations.parquet").drop_duplicates("stid").set_index("stid").loc[stations]

fr = []
for s in stations:
    d = json.load(gzip.open(f"{RAW}/{s}_{Y}_day0.json.gz", "rt"))
    h = pd.DataFrame(d["hourly"])
    h["valid_utc"] = pd.to_datetime(h.time).dt.tz_localize("UTC")
    h["stid"], h["elev"] = s, d["elevation"]
    h = h.sort_values("valid_utc")
    tk = h.temperature_2m + 273.15
    t12 = tk.shift(12)
    for name, T in (("manual_now", tk), ("manual_12h", (tk + t12) / 2)):
        h[name] = h.surface_pressure * np.exp(G * h.elev / (RD * (T + GAMMA * h.elev / 2)))
    fr.append(h)
N = pd.concat(fr).rename(columns={"pressure_msl": "raw"})

gy = g[(g.time_utc >= pd.Timestamp(f"{Y - 1}-12-31 23:00", tz="UTC")) & (g.time_utc < pd.Timestamp(f"{Y + 1}-01-01", tz="UTC"))].copy()
gy["valid_utc"] = gy.time_utc.dt.ceil("h")
obs = pd.concat([gy[["station1", "valid_utc", "slp1_hpa"]].set_axis(["stid", "valid_utc", "slp_obs"], axis=1),
                 gy[["station2", "valid_utc", "slp2_hpa"]].set_axis(["stid", "valid_utc", "slp_obs"], axis=1)])
obs = obs.dropna().groupby(["stid", "valid_utc"]).slp_obs.first().reset_index()
m = N.merge(obs, on=["stid", "valid_utc"]).dropna(subset=["manual_12h"])
m["month"] = m.valid_utc.dt.month


# leave-one-month-out versions
def design(t):
    doy = 2 * np.pi * t.dt.day_of_year.to_numpy() / 365.25
    hr = 2 * np.pi * t.dt.hour.to_numpy() / 24
    cols = [np.ones(len(t)), np.cos(doy), np.sin(doy), np.cos(2 * doy), np.sin(2 * doy),
            np.cos(hr), np.sin(hr), np.cos(2 * hr), np.sin(2 * hr),
            np.cos(doy) * np.cos(hr), np.cos(doy) * np.sin(hr), np.sin(doy) * np.cos(hr), np.sin(doy) * np.sin(hr)]
    return np.column_stack(cols)


m["manual_12h_off"] = np.nan
m["harmonic_cv"] = np.nan
for mon in range(1, 13):
    te = m.month == mon
    for s in stations:
        tr, ix = ~te & (m.stid == s), te & (m.stid == s)
        off = (m.loc[tr, "manual_12h"] - m.loc[tr, "slp_obs"]).mean()
        m.loc[ix, "manual_12h_off"] = m.loc[ix, "manual_12h"] - off
        b, *_ = np.linalg.lstsq(design(m.loc[tr, "valid_utc"]), (m.loc[tr, "raw"] - m.loc[tr, "slp_obs"]).to_numpy(),
                                rcond=None)
        m.loc[ix, "harmonic_cv"] = m.loc[ix, "raw"] - design(m.loc[ix, "valid_utc"]) @ b

V = ["raw", "manual_now", "manual_12h", "manual_12h_off", "harmonic_cv"]
# round-trip check
rt = (m.manual_now - m.raw).abs()
print(f"round-trip check |manual_now - pressure_msl|: median {rt.median():.2f}, p99 {rt.quantile(.99):.2f}, max {rt.max():.2f} hPa")

# station errors
rows = []
for s, x in m.groupby("stid"):
    for v in V:
        e = x[v] - x.slp_obs
        rows.append(dict(stid=s, elev=meta.elevation_m[s], version=v, bias=e.mean(), sd=e.std(),
                         seas=e.groupby(x.month).mean().agg(lambda z: z.max() - z.min()),
                         diur=e.groupby(x.valid_utc.dt.hour).mean().agg(lambda z: z.max() - z.min())))
st = pd.DataFrame(rows)
st.round(3).to_csv(f"{OUT}/reduction_test_station_{Y}.csv", index=False)
hi = st[st.elev > 580]
print(f"\n== station SLP error vs METAR, {Y} day 0: high stations (> 580 m) ==")
print(hi.pivot_table(index=["stid", "elev"], columns="version", values=["bias", "seas"])[[("bias", v) for v in V] + [("seas", v) for v in V]]
      .round(2).sort_index(level="elev").to_string())
print("\nmedian over high stations:")
print(hi.groupby("version")[["bias", "sd", "seas", "diur"]].agg(lambda z: z.abs().median()).loc[V].round(2).to_string())

# monthly bias at a few stations
print("\nmonthly bias, manual_12h (hPa):")
print(m.assign(e=m.manual_12h - m.slp_obs)[m.stid.isin(["KTPH", "KWMC", "KRNO", "KLMT", "KLAS", "KLAX"])]
      .pivot_table(index="stid", columns="month", values="e").round(1).to_string())

# gradients
w = m.pivot_table(index="valid_utc", columns="stid", values=V + ["slp_obs"])
pr = []
for _, (gn, s1, s2) in pairs.iterrows():
    go = w["slp_obs"][s1] - w["slp_obs"][s2]
    for v in V:
        e = (w[v][s1] - w[v][s2] - go).dropna()
        pr.append(dict(gradient=gn, version=v, bias=e.mean(), rmse=np.sqrt((e ** 2).mean())))
pr = pd.DataFrame(pr)
pr.round(3).to_csv(f"{OUT}/reduction_test_gradient_{Y}.csv", index=False)
print(f"\n== day-0 gradient RMSE (bias), {Y} ==")
t = pr.pivot_table(index="gradient", columns="version", values="rmse")[V]
tb = pr.pivot_table(index="gradient", columns="version", values="bias")[V]
print(t.round(2).astype(str).add(" (").add(tb.round(2).astype(str)).add(")").loc[t.raw.sort_values(ascending=False).index].to_string())
print("\nmedian gradient RMSE:", t.median().round(2).to_dict())
print("median |gradient bias|:", tb.abs().median().round(2).to_dict())
print(f"\nwrote {OUT}/reduction_test_*_{Y}.csv")
