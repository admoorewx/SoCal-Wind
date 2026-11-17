#!/usr/bin/env python3
"""Event and target tables for the gradient wind models (step 2 of the model plan).

Definitions are in docs/wind_model_plan.md. For every signed gradient with a station set
(derived/models/station_sets.csv) and every gradient time of that sign (one per hour):

  n_set, n_reporting   stations in the set / stations with a value of the modelled variable
  qualifies            n_reporting >= 50% of n_set (models use only these rows)
  target_mph           sustained: the set's 75th percentile (numpy linear)
                       gust: the k-th highest reported gust, k = max(2, ceil(0.1 * n_reporting))
                       (nearest-rank 90th percentile with a two-station floor; no gap filling,
                       NaN when fewer than k gusts are reported)
  k                    gust only: the rank used
  max_mph              highest station value (diagnostic only)
  ev_<T>               target_mph >= T mph (sustained 15/20/30, gust 30/40/50/60)
  wdir_mean_deg        circular mean of the direction reported with each station's max sustained wind,
  wdir_r, n_wdir       over set stations with sustained >= CALM_MPS (directions in calm air are noise);
                       wdir_r is the mean resultant length (1 = all stations agree, 0 = no agreement).
                       Stored for later use; not a predictor in the first models.
  abs_gradient_hpa, hour_utc, doy and their sin/cos encodings

-> derived/models/targets/<variable>_train.parquet     2015-2025
-> derived/models/targets/<variable>_holdout.parquet   2026 (held out)
-> derived/models/targets/event_counts.csv             qualifying hours and event hours per
                                                        (signed gradient, variable, threshold) and
                                                        whether it has enough events to model

Usage:
  python build_targets.py
"""
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd

import gradient_pipeline as gp

HERE = os.path.dirname(os.path.abspath(__file__))
DER = os.path.normpath(os.path.join(HERE, "..", "data", "v2", "derived"))
OUT = os.path.join(DER, "models", "targets")
HOLDOUT_START = pd.Timestamp("2026-01-01", tz="UTC")
MPS2MPH = 2.236936
MIN_REPORTING_FRAC = 0.5
CALM_MPS = 1.5  # ~3 kt, the ASOS calm threshold
MIN_EVENT_HOURS = 100
THRESHOLDS = {"sustained": [15, 20, 30], "gust": [30, 40, 50, 60]}
COLS = {"sustained": "sustained_max_mps", "gust": "gust_max_mps"}
WDIR = "wdir_at_sustained_max_deg"


def gust_stat(X, n_rep):
    """k-th highest per row, k = max(2, ceil(0.1 * n_rep)); NaN where fewer than k values."""
    k = np.maximum(2, np.ceil(0.1 * n_rep).astype(int))
    srt = -np.sort(-np.where(np.isnan(X), -np.inf, X), axis=1)
    val = srt[np.arange(len(X)), np.minimum(k, X.shape[1]) - 1]
    val = np.where((n_rep >= k) & np.isfinite(val), val, np.nan)
    return val, k


def mean_direction(D, S):
    ok = ~np.isnan(D) & (S >= CALM_MPS)
    r = np.radians(np.where(ok, D, 0.0))
    n = ok.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sm, cm = (np.sin(r) * ok).sum(1) / n, (np.cos(r) * ok).sum(1) / n
    mean = np.where(n > 0, np.degrees(np.arctan2(sm, cm)) % 360, np.nan)
    return mean, np.where(n > 0, np.hypot(sm, cm), np.nan), n


def table_for(sg, var, stids, g, M):
    """g: gradient rows of this sign (time_utc, gradient_hpa); M: matched rows for the s1 grid."""
    m = M[M.stid.isin(stids)]
    piv = lambda col: m.pivot_table(index="time_utc", columns="stid", values=col, aggfunc="first") \
        .reindex(index=g.time_utc, columns=stids).to_numpy(dtype=float)
    X = piv(COLS[var]) * MPS2MPH
    S, D = piv("sustained_max_mps"), piv(WDIR)
    n_set = len(stids)
    n_rep = (~np.isnan(X)).sum(1)
    out = pd.DataFrame({"signed_gradient": sg, "time_utc": g.time_utc.to_numpy(),
                        "gradient_hpa": g.gradient_hpa.to_numpy(),
                        "abs_gradient_hpa": g.gradient_hpa.abs().to_numpy(),
                        "n_set": n_set, "n_reporting": n_rep,
                        "qualifies": n_rep >= MIN_REPORTING_FRAC * n_set})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["max_mph"] = np.nanmax(X, axis=1)
    if var == "sustained":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows -> NaN
            out["target_mph"] = np.nanpercentile(X, 75, axis=1)
    else:
        out["target_mph"], out["k"] = gust_stat(X, n_rep)
    for T in THRESHOLDS[var]:
        out[f"ev_{T}"] = out.target_mph >= T
    out["wdir_mean_deg"], out["wdir_r"], out["n_wdir"] = mean_direction(D, S)
    return out[out.n_reporting > 0]


def add_time(df):
    t = df.time_utc.dt
    df["hour_utc"] = t.hour.astype("int8")
    df["doy"] = t.day_of_year.astype("int16")
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * df.hour_utc / 24), np.cos(2 * np.pi * df.hour_utc / 24)
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * df.doy / 365.25), np.cos(2 * np.pi * df.doy / 365.25)
    df["year"] = t.year.astype("int16")
    return df


def main():
    sets = pd.read_csv(os.path.join(DER, "models", "station_sets.csv"))
    pairs = gp.load_pairs(gp.DEFAULT_GRADIENTS)
    ts = pd.read_parquet(os.path.join(DER, "gradients", "gradient_timeseries.parquet"),
                         columns=["gradient", "time_utc", "gradient_hpa"])
    ts = ts.assign(hour=ts.time_utc.dt.floor("h")).drop_duplicates(["gradient", "hour"]).drop(columns="hour")
    tables = {"sustained": [], "gust": []}
    for s1 in sorted({p[0] for p in pairs.values()}):
        sgs = sorted(sg for sg in sets.signed_gradient.unique() if pairs[sg[:-1]][0] == s1)
        if not sgs:
            continue
        need = sorted(set(sets[sets.signed_gradient.isin(sgs)].stid))
        files = sorted(glob.glob(os.path.join(DER, "gradients", "matched", s1, "*.parquet")))
        M = pd.concat([pd.read_parquet(f, columns=["stid", "time_utc", *COLS.values(), WDIR],
                                       filters=[("stid", "in", need)]) for f in files])
        M["stid"] = M.stid.astype(str)
        M["time_utc"] = M.time_utc.astype("datetime64[ms, UTC]")
        for sg in sgs:
            g = ts[ts.gradient == sg[:-1]]
            g = g[(g.gradient_hpa > 0) if sg.endswith("+") else (g.gradient_hpa < 0)]
            g = g.assign(time_utc=g.time_utc.astype("datetime64[ms, UTC]"))
            for var in ("sustained", "gust"):
                stids = sets[(sets.signed_gradient == sg) & (sets.variable == var)].stid.tolist()
                if stids:
                    tables[var].append(table_for(sg, var, stids, g, M))
            print(f"{sg} done", flush=True)

    os.makedirs(OUT, exist_ok=True)
    counts = []
    for var, parts in tables.items():
        df = add_time(pd.concat(parts, ignore_index=True))
        df["n_reporting"] = df.n_reporting.astype("int16")
        df["n_wdir"] = df.n_wdir.astype("int16")
        for split, part in (("train", df[df.time_utc < HOLDOUT_START]),
                            ("holdout", df[df.time_utc >= HOLDOUT_START])):
            part.to_parquet(os.path.join(OUT, f"{var}_{split}.parquet"), index=False)
            print(f"Wrote {var}_{split}.parquet: {len(part):,} rows, {part.qualifies.sum():,} qualifying")
        q = df[df.qualifies]
        for (sg, ho), x in q.groupby(["signed_gradient", q.time_utc >= HOLDOUT_START]):
            for T in THRESHOLDS[var]:
                counts.append({"signed_gradient": sg, "variable": var, "threshold_mph": T,
                               "split": "holdout" if ho else "train", "qualifying_hours": len(x),
                               "event_hours": int(x[f"ev_{T}"].sum())})
    c = pd.DataFrame(counts)
    c = c.pivot_table(index=["signed_gradient", "variable", "threshold_mph"], columns="split",
                      values=["qualifying_hours", "event_hours"], fill_value=0)
    c.columns = [f"{s}_{v}" for v, s in c.columns]
    c = c.reset_index()
    c["train_event_pct"] = (100 * c.train_event_hours / c.train_qualifying_hours).round(2)
    c["modelled"] = c.train_event_hours >= MIN_EVENT_HOURS
    c = c[["signed_gradient", "variable", "threshold_mph", "train_qualifying_hours", "train_event_hours",
           "train_event_pct", "holdout_qualifying_hours", "holdout_event_hours", "modelled"]]
    c.to_csv(os.path.join(OUT, "event_counts.csv"), index=False)
    with open(os.path.join(OUT, "targets_meta.json"), "w") as f:
        json.dump({"holdout_start": str(HOLDOUT_START), "min_reporting_frac": MIN_REPORTING_FRAC,
                   "calm_mps": CALM_MPS, "min_event_hours": MIN_EVENT_HOURS, "thresholds_mph": THRESHOLDS,
                   "sustained_target": "75th percentile (numpy linear) of reporting set stations",
                   "gust_target": "k-th highest reported gust, k = max(2, ceil(0.1 * n_reporting)), no gap filling",
                   "created_utc": pd.Timestamp.now(tz="UTC").isoformat()}, f, indent=2)
    print(f"Wrote event_counts.csv and targets_meta.json in {OUT}")


if __name__ == "__main__":
    main()
