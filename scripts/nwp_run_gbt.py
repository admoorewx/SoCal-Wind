#!/usr/bin/env python3
"""Phase 4: run the frozen gbt_all sustained models on NWP and observed gradients; compute raw-NWP set targets.

For every signed set, valid hour (2026 holdout) and lead day, one row with:
  target_mph, qualifies, n_reporting, obs_sign        observed set target (sign-free, Phase 3)
  grad_obs, grad_nwp_raw, grad_nwp_corr               own-pair gradient (hPa); corr = SLP-convention corrected
  sel_obs, sel_nwp_raw, sel_nwp_corr                  True where that gradient's sign picks THIS set
  gbt_obs, gbt_nwp_raw, gbt_nwp_corr                  linear GBT forecast (mph), computed at every row
  p<T>_obs, p<T>_nwp_raw, p<T>_nwp_corr               logistic GBT event probabilities (thresholds per set)
  nwp_p75_rep, n_nwp_rep                              raw NWP: 75th pct of NWP 10 m wind at the set stations that
                                                      reported at that hour (same stations as the obs target)
  nwp_p75_all                                         same over all set stations (diagnostic)
Score a variant X on rows where sel_X & qualifies. gbt_obs/grad_obs do not depend on lead (repeated per lead).

Feature construction reproduces training exactly (checked on 2025 rows, see --check):
  valid hour V <-> obs time in hour V-1 (obs at :50-:56): hour/doy features and other-pair gradients use V-1h.
  Feature order is read from each model file. Missing obs gradients stay NaN (LightGBM native).

-> derived/verification/<model>/forecasts_sustained.parquet
-> derived/verification/model_checksums.txt
Wind-only models (nwp_config variables without pressure_msl, e.g. NBM): GBT/gradient NWP columns are skipped (NaN
sel_nwp_*); only obs columns, nwp_p75_rep, nwp_p75_all, n_nwp_rep.
Usage: python nwp_run_gbt.py --model gfs [--check]
"""
import argparse, glob, hashlib, json, os, re

import lightgbm as lgb
import numpy as np
import pandas as pd

import nwp_config as C
import gradient_pipeline as gp

ROOT = "/mnt/d/socal"
DER = f"{ROOT}/data/v2/derived"
MDIR = f"{DER}/models/gbt/models"
MPS2MPH = 2.236936
VAR = "sustained"

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gfs")
ap.add_argument("--points-tag", default="20260101_20261031")
ap.add_argument("--calib", default="coefficients_2025.json")
ap.add_argument("--check", action="store_true", help="only verify feature construction against training (2025)")
a = ap.parse_args()
OUT = f"{DER}/verification/{a.model}"
os.makedirs(OUT, exist_ok=True)
CFG = C.MODELS[a.model]
# wind-only models (NBM): no pressure, so no NBM-driven GBT; only the raw-wind set target nwp_p75_rep (+ obs columns)
WIND_ONLY = "pressure_msl" not in CFG.get("variables", ("pressure_msl",))
pairs = gp.load_pairs(gp.DEFAULT_GRADIENTS)                     # name -> (station1, station2)
PAIRS = list(pairs)


def time_features(valid):
    t = valid - pd.Timedelta(hours=1)                           # obs time lies in hour V-1
    hour, doy = t.dt.hour.to_numpy(), t.dt.day_of_year.to_numpy()
    return pd.DataFrame({"hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
                         "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25)},
                        index=valid.index)


def obs_gradients(start, end):
    """Observed signed gradients, one per pair and valid hour (first obs in hour V-1, as in training)."""
    ts = pd.read_parquet(f"{DER}/gradients/gradient_timeseries.parquet", columns=["gradient", "time_utc", "gradient_hpa"])
    ts = ts[ts.gradient.isin(PAIRS) & (ts.time_utc >= start - pd.Timedelta(hours=1)) & (ts.time_utc < end)]
    ts["valid_utc"] = ts.time_utc.dt.floor("h") + pd.Timedelta(hours=1)
    G = ts.pivot_table(index="valid_utc", columns="gradient", values="gradient_hpa", aggfunc="first")
    G.index = G.index.astype("datetime64[us, UTC]")
    return G


def model_files(sg):
    out = {}
    for f in glob.glob(f"{MDIR}/{sg}_{VAR}_*__gbt_all.txt"):
        kind, T = re.search(rf"_{VAR}_(linear|logistic)_(\d+)__", f).groups()
        out["lin" if kind == "linear" else f"p{T}"] = f
    return dict(sorted(out.items()))


def predict(booster, own_grad, G, valid):
    """Build the feature matrix in the model's own column order and predict."""
    cols = booster.feature_name()
    X = pd.DataFrame(index=valid.index)
    X["abs_gradient_hpa"] = np.abs(own_grad)
    X = X.join(time_features(valid))
    Gv = G.reindex(valid.to_numpy())
    for c in cols:
        if c.startswith("g_"):
            X[c] = Gv[c[2:]].to_numpy()
    return booster.predict(X[cols].to_numpy(float)), X[cols]


sets = pd.read_csv(f"{DER}/models/station_sets.csv")
SETS = sorted(sets[sets.variable == VAR].signed_gradient.unique())

# ---------------------------------------------------------------- feature check vs training (2025)
if a.check:
    import build_targets  # noqa: F401  (same environment as training)
    T = pd.read_parquet(f"{DER}/models/targets/{VAR}_train.parquet")
    T = T[T.qualifies & T.target_mph.notna() & (T.year == 2025)]
    ts = pd.read_parquet(f"{DER}/gradients/gradient_timeseries.parquet", columns=["gradient", "time_utc", "gradient_hpa"])
    ts = ts[ts.gradient.isin(PAIRS)]
    ts["hour"] = ts.time_utc.dt.floor("h")
    Gt = ts.pivot_table(index="hour", columns="gradient", values="gradient_hpa", aggfunc="first")
    Gt.index = Gt.index.astype("datetime64[ms, UTC]")
    Gt.columns = [f"g_{c}" for c in Gt.columns]
    G = obs_gradients(pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC"))
    worst = 0.0
    for sg in SETS:
        d = T[T.signed_gradient == sg].reset_index(drop=True)
        d = d.assign(hour=d.time_utc.dt.floor("h")).join(Gt, on="hour")
        b = lgb.Booster(model_file=model_files(sg)["lin"])
        cols = b.feature_name()
        ref = d[cols].to_numpy(float)
        valid = (d.time_utc.dt.floor("h") + pd.Timedelta(hours=1)).astype("datetime64[us, UTC]")
        own = G[sg[:-1]].reindex(valid.to_numpy()).to_numpy()
        p_new, Xn = predict(b, own, G, valid)
        diff = np.nanmax(np.abs(np.where(np.isnan(ref) & np.isnan(Xn.to_numpy()), 0, ref - Xn.to_numpy())))
        nan_mis = int((np.isnan(ref) != np.isnan(Xn.to_numpy())).sum())
        p_ref = b.predict(ref)
        worst = max(worst, diff)
        print(f"{sg}: rows {len(d)}, max feature diff {diff:.2e}, NaN mismatches {nan_mis}, "
              f"max prediction diff {np.max(np.abs(p_ref - p_new)):.2e} mph")
    print(f"worst feature diff {worst:.2e}")
    raise SystemExit

# ---------------------------------------------------------------- inputs
with open(f"{DER}/verification/model_checksums.txt", "w") as f:
    for p in sorted(glob.glob(f"{MDIR}/*_{VAR}_*__gbt_all.txt")):
        f.write(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {os.path.basename(p)}\n")

tgt = pd.read_parquet(f"{DER}/verification/holdout_targets_signfree_{VAR}.parquet",
                      columns=["signed_gradient", "valid_utc", "time_utc", "target_mph", "qualifies", "n_reporting",
                               "obs_sign"])
tgt["valid_utc"] = tgt.valid_utc.astype("datetime64[us, UTC]")
START, END = tgt.valid_utc.min(), tgt.valid_utc.max() + pd.Timedelta(hours=1)
G_obs = obs_gradients(START, END)

P = pd.read_parquet(f"{C.OPEN_METEO_ROOT}/points/{a.model}/points_{a.points_tag}.parquet")
P["valid_utc"] = P.valid_utc.astype("datetime64[us, UTC]")
pres_st = sorted({s for p in pairs.values() for s in p})
if WIND_ONLY:
    lead_h = P.drop_duplicates(["lead_day", "valid_utc"]).set_index(["lead_day", "valid_utc"]).lead_h
PP = pd.DataFrame() if WIND_ONLY else P[P.stid.isin(pres_st)][["valid_utc", "stid", "lead_day", "lead_h", "pressure_msl_hpa"]].copy()
G_nwp = {}
if not WIND_ONLY:
    coef = json.load(open(f"{OUT}/pressure_calibration/{a.calib}"))["coefficients"]
    PP["p_corr"] = np.nan
    for s, idx in PP.groupby("stid").groups.items():
        PP.loc[idx, "p_corr"] = PP.loc[idx, "pressure_msl_hpa"] - C.slp_correction_design(PP.loc[idx, "valid_utc"]) @ np.array(coef[s])
    for lab, col in (("nwp_raw", "pressure_msl_hpa"), ("nwp_corr", "p_corr")):
        w = PP.pivot_table(index=["lead_day", "valid_utc"], columns="stid", values=col)
        G_nwp[lab] = pd.DataFrame({pn: w[s1] - w[s2] for pn, (s1, s2) in pairs.items()})
    lead_h = PP.drop_duplicates(["lead_day", "valid_utc"]).set_index(["lead_day", "valid_utc"]).lead_h

# NWP wind (mph) per lead: valid x stid
W = P[P.stid.isin(set(sets[sets.variable == VAR].stid))][["valid_utc", "stid", "lead_day", "wind_speed_10m_mps"]]
W = W.assign(w=W.wind_speed_10m_mps * MPS2MPH).pivot_table(index=["lead_day", "valid_utc"], columns="stid", values="w")
del P

# station reporting mask (same matched data as the targets): per station1 grid, valid x stid
s1_of = {pn: s1 for pn, (s1, _) in pairs.items()}
rep = {}
for s1 in sorted(set(s1_of.values())):
    need = sorted(set(sets[(sets.variable == VAR) & sets.signed_gradient.str[:-1].map(s1_of).eq(s1)].stid))
    if not need:
        continue
    files = [f for f in sorted(glob.glob(f"{DER}/gradients/matched/{s1}/*.parquet")) if os.path.basename(f) >= "2025-12"]
    M = pd.concat([pd.read_parquet(f, columns=["stid", "time_utc", "sustained_max_mps"], filters=[("stid", "in", need)])
                   for f in files])
    M["stid"] = M.stid.astype(str)
    M["time_utc"] = M.time_utc.astype("datetime64[us, UTC]")
    rep[s1] = M.dropna().pivot_table(index="time_utc", columns="stid", values="sustained_max_mps", aggfunc="first").notna()

# ---------------------------------------------------------------- run
out = []
leads = sorted(W.index.get_level_values("lead_day").unique())
for sg in SETS:
    pn, sign = sg[:-1], sg[-1]
    stids = sets[(sets.signed_gradient == sg) & (sets.variable == VAR)].stid.tolist()
    t = tgt[tgt.signed_gradient == sg].set_index("valid_utc").sort_index()
    valid = pd.Series(t.index, index=t.index)
    boosters = {k: lgb.Booster(model_file=f) for k, f in model_files(sg).items()}
    # observed-gradient GBT (lead-independent)
    own_obs = G_obs[pn].reindex(t.index).to_numpy() if pn in G_obs else np.full(len(t), np.nan)
    obs_pred = {k: predict(b, own_obs, G_obs, valid)[0] for k, b in boosters.items()}
    # reporting mask aligned to target rows (target time_utc -> station columns)
    R = rep[s1_of[pn]].reindex(index=t.time_utc.to_numpy(), columns=stids, fill_value=False).to_numpy()
    for k in leads:
        if WIND_ONLY:
            Gk = Gc = None
            own_raw = own_cor = np.full(len(t), np.nan)
        else:
            Gk = G_nwp["nwp_raw"].loc[k]
            Gc = G_nwp["nwp_corr"].loc[k]
            own_raw = Gk[pn].reindex(t.index).to_numpy()
            own_cor = Gc[pn].reindex(t.index).to_numpy()
        row = pd.DataFrame({"signed_gradient": sg, "valid_utc": t.index, "lead_day": np.int8(k),
                            "lead_h": lead_h.loc[k].reindex(t.index).to_numpy(),
                            "target_mph": t.target_mph.to_numpy(), "qualifies": t.qualifies.to_numpy(),
                            "n_reporting": t.n_reporting.to_numpy(), "obs_sign": t.obs_sign.to_numpy(),
                            "grad_obs": own_obs, "grad_nwp_raw": own_raw, "grad_nwp_corr": own_cor})
        sgn = 1 if sign == "+" else -1
        for lab, g_ in (("obs", own_obs), ("nwp_raw", own_raw), ("nwp_corr", own_cor)):
            row[f"sel_{lab}"] = np.sign(g_) == sgn
        for kk, b in boosters.items():
            name = "gbt" if kk == "lin" else kk
            row[f"{name}_obs"] = obs_pred[kk]
            if not WIND_ONLY:
                row[f"{name}_nwp_raw"] = predict(b, own_raw, Gk, valid)[0]
                row[f"{name}_nwp_corr"] = predict(b, own_cor, Gc, valid)[0]
        Wk = W.loc[k].reindex(index=t.index, columns=stids).to_numpy()
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                row["nwp_p75_rep"] = np.nanpercentile(np.where(R, Wk, np.nan), 75, axis=1)
                row["nwp_p75_all"] = np.nanpercentile(Wk, 75, axis=1)
        row["n_nwp_rep"] = (R & ~np.isnan(Wk)).sum(1).astype("int16")
        out.append(row)
    print(f"{sg}: {len(t):,} hours x {len(leads)} leads, models {list(boosters)}", flush=True)

F = pd.concat(out, ignore_index=True)
F.to_parquet(f"{OUT}/forecasts_{VAR}.parquet", index=False)
print(f"wrote {OUT}/forecasts_{VAR}.parquet: {len(F):,} rows")

# quick summary (not the Phase 5 scoring): day-0 and day-3 RMSE on chosen, qualifying rows
q = F[F.qualifies & F.target_mph.notna()]
rows = []
for k, x in q.groupby("lead_day"):
    r = {"lead_day": k}
    for lab in ("obs",) if WIND_ONLY else ("obs", "nwp_raw", "nwp_corr"):
        s = x[x[f"sel_{lab}"]]
        r[f"n_{lab}"] = len(s)
        r[f"rmse_gbt_{lab}"] = np.sqrt(((s[f"gbt_{lab}"] - s.target_mph) ** 2).mean())
        r[f"rmse_rawnwp_on_{lab}"] = np.sqrt(((s.nwp_p75_rep - s.target_mph) ** 2).mean())
    rows.append(r)
print(pd.DataFrame(rows).round(2).to_string(index=False))
