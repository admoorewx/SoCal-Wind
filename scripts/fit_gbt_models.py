#!/usr/bin/env python3
"""Step 5: gradient boosted trees (LightGBM) on the same rows, targets and folds as steps 3-4b.

This script only READS the logistic/linear outputs (targets/, gradient_addon/) and writes to its own
folder, derived/models/gbt/, so the regression models stay untouched and the comparison is clean.

Two tree variants per (signed gradient, variable, kind, threshold):
  gbt_sel  base features (|grad|, hour sin/cos, doy sin/cos) + the SAME other-pair gradients that the
           step-4b BIC selection chose in that fold (read from gradient_addon/selections.csv).
           Same information as the add-on model -> isolates the effect of model form.
  gbt_all  base features + ALL other pairs' signed gradients (trees pick their own).

Leakage control: folds are calendar years (leave-one-year-out, 2015-2025). The number of boosting
rounds is chosen by early stopping on an INNER validation set of training years only (nearest
training years to the held-out year, added until it holds >= MIN_VAL_EVENTS events / at least 2 years),
then the model is refit on all training years with that number of rounds. The held-out year is never
used for any choice. The 2026 holdout is not touched. Final models (all 2015-2025, rounds = median of
the fold choices, features = final add-on selection for gbt_sel) are saved for later holdout/NWP use.

Missing other-pair gradients are left as NaN (LightGBM routes them natively); the regression models
filled them with the training-fold mean. |grad| has a monotone-increasing constraint (physical prior).

Metrics (pooled out of fold) on rows identical to the add-on run (checked: same times and obs):
  logistic: BSS vs training-fold climatology, AUC, log loss, top-end ratio (same as step 4a/4b)
  linear:   R2, RMSE

-> derived/models/gbt/cv_logistic.csv, cv_linear.csv   addon vs gbt_sel vs gbt_all
-> derived/models/gbt/oof_predictions.parquet
-> derived/models/gbt/importance.csv                    gain importance, final models
-> derived/models/gbt/models/<key>__<variant>.txt       final LightGBM models

Usage:
  python fit_gbt_models.py [--workers 14] [--only LAX-WJF-]
"""
import argparse
import json
import os
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np
import pandas as pd

import fit_baseline_models as fb
import fit_gradient_addon as fa
import gradient_pipeline as gp

HERE = os.path.dirname(os.path.abspath(__file__))
DER = os.path.normpath(os.path.join(HERE, "..", "data", "v2", "derived"))
MDIR = os.path.join(DER, "models")
ADDON = os.path.join(MDIR, "gradient_addon")
OUT = os.path.join(MDIR, "gbt")
BASE = fb.FEATURES["grad+hour+doy"]
MIN_VAL_EVENTS = 30
MAX_ROUNDS, EARLY = 3000, 100
PARAMS = {
    "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 100, "lambda_l2": 1.0,
    "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
    "num_threads": 1, "verbose": -1, "seed": 7, "deterministic": True,
}


def params(kind, cols):
    p = dict(PARAMS)
    p["objective"] = "binary" if kind == "logistic" else "regression"
    p["metric"] = "binary_logloss" if kind == "logistic" else "l2"
    p["monotone_constraints"] = [1 if c == "abs_gradient_hpa" else 0 for c in cols]
    return p


def inner_val_years(train_years, held, y_by_year, kind):
    """Training years nearest the held-out year, until enough events (logistic) and >= 2 years."""
    order = sorted(train_years, key=lambda t: (abs(t - held), t))
    val, ev = [], 0
    for t in order:
        val.append(t)
        ev += y_by_year.get(t, 0)
        if len(val) >= 2 and (kind == "linear" or ev >= MIN_VAL_EVENTS):
            break
        if len(val) >= max(2, len(train_years) // 3):
            break
    return val


def fit_fold(X, y, years, tr_years, held, kind, cols):
    p = params(kind, cols)
    ybyyr = pd.Series(y).groupby(years).sum().to_dict() if kind == "logistic" else {}
    val = inner_val_years(tr_years, held, ybyyr, kind)
    itr = np.isin(years, tr_years) & ~np.isin(years, val)
    iva = np.isin(years, val)
    dtr = lgb.Dataset(X[itr], y[itr], feature_name=cols, free_raw_data=False)
    dva = lgb.Dataset(X[iva], y[iva], reference=dtr)
    m = lgb.train(p, dtr, MAX_ROUNDS, valid_sets=[dva], callbacks=[lgb.early_stopping(EARLY, verbose=False)])
    n_rounds = max(m.best_iteration, 10)
    tr = np.isin(years, tr_years)
    m = lgb.train(p, lgb.Dataset(X[tr], y[tr], feature_name=cols), n_rounds)
    return m, n_rounds


def run(job):
    sg, var, kind, T, d, cands, sel = job
    y = (d[f"ev_{T}"] if kind == "logistic" else d.target_mph).to_numpy(float)
    years = d.year.to_numpy()
    uy = np.unique(years)
    p_clim = np.full(len(d), np.nan)
    preds = {v: np.full(len(d), np.nan) for v in ("gbt_sel", "gbt_all")}
    rounds = {v: [] for v in preds}
    for yr in uy:
        te = years == yr
        p_clim[te] = y[~te].mean()
        tr_years = [t for t in uy if t != yr]
        for v in preds:
            cols = BASE + (sel[str(yr)] if v == "gbt_sel" else cands)
            X = d[cols].to_numpy(float)
            m, n = fit_fold(X, y, years, tr_years, yr, kind, cols)
            preds[v][te] = m.predict(X[te])
            rounds[v].append(n)
    key = {"signed_gradient": sg, "variable": var, "kind": kind, "threshold_mph": T}
    # final models on all 2015-2025
    os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
    imp = []
    for v in preds:
        cols = BASE + (sel["final"] if v == "gbt_sel" else cands)
        n = int(np.median(rounds[v]))
        m = lgb.train(params(kind, cols), lgb.Dataset(d[cols].to_numpy(float), y, feature_name=cols), n)
        m.save_model(os.path.join(OUT, "models", f"{sg}_{var}_{kind}_{T}__{v}.txt"))
        g = m.feature_importance("gain")
        g = g / g.sum() if g.sum() > 0 else g
        imp += [{**key, "variant": v, "feature": c, "gain_frac": float(x)} for c, x in zip(cols, g)]
    m = {"n_hours": len(y), "rounds_sel_median": int(np.median(rounds["gbt_sel"])),
         "rounds_all_median": int(np.median(rounds["gbt_all"]))}
    oof = pd.DataFrame({**key, "time_utc": d.time_utc.to_numpy(), "obs": y.astype("float32"),
                        "clim": p_clim.astype("float32"),
                        **{f"pred_{v}": p.astype("float32") for v, p in preds.items()}})
    return {**key, **m}, imp, oof


def metrics(g, kind):
    """g: one model's rows with obs, clim, pred_addon, pred_gbt_sel, pred_gbt_all."""
    y, c = g.obs.to_numpy(float), g.clim.to_numpy(float)
    out = {}
    labs = ("addon", "gbt_sel", "gbt_all")
    if kind == "logistic":
        bc = np.mean((c - y) ** 2)
        out.update(event_hours=int(y.sum()), base_rate=y.mean())
        for lab in labs:
            p = g[f"pred_{lab}"].to_numpy(float)
            out[f"bss_{lab}"] = 1 - np.mean((p - y) ** 2) / bc
            out[f"auc_{lab}"] = fb.auc(y, p)
            out[f"logloss_{lab}"] = fb.logloss(y, p)
            out[f"top_ratio_{lab}"] = fa.top_ratio(y, p)
    else:
        cm = np.mean((c - y) ** 2)
        for lab in labs:
            p = g[f"pred_{lab}"].to_numpy(float)
            out[f"r2_{lab}"] = 1 - np.mean((p - y) ** 2) / cm
            out[f"rmse_{lab}"] = np.sqrt(np.mean((p - y) ** 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--only", help="run one signed gradient (test)")
    args = ap.parse_args()
    pairs = list(gp.load_pairs(gp.DEFAULT_GRADIENTS))
    ts = pd.read_parquet(os.path.join(DER, "gradients", "gradient_timeseries.parquet"),
                         columns=["gradient", "time_utc", "gradient_hpa"])
    ts = ts[ts.gradient.isin(pairs)]
    ts["hour"] = ts.time_utc.dt.floor("h")
    G = ts.pivot_table(index="hour", columns="gradient", values="gradient_hpa", aggfunc="first")
    G.index = G.index.astype("datetime64[ms, UTC]")
    G.columns = [f"g_{c}" for c in G.columns]
    counts = pd.read_csv(os.path.join(MDIR, "targets", "event_counts.csv"))
    sels = pd.read_csv(os.path.join(ADDON, "selections.csv"), keep_default_na=False)
    selmap = {k: dict(zip(g.fold.astype(str), [c.split("|") if c else [] for c in g.chosen]))
              for k, g in sels.groupby(["signed_gradient", "variable", "kind", "threshold_mph"])}
    jobs = []
    for var in ("sustained", "gust"):
        T = pd.read_parquet(os.path.join(MDIR, "targets", f"{var}_train.parquet"))
        T = T[T.qualifies & T.target_mph.notna()]
        T = T.assign(hour=T.time_utc.dt.floor("h")).join(G, on="hour")
        for sg, d in T.groupby("signed_gradient", sort=True):
            if args.only and sg != args.only:
                continue
            d = d.reset_index(drop=True)
            cands = [f"g_{p}" for p in pairs if p != sg[:-1]]
            mod = counts[(counts.signed_gradient == sg) & (counts.variable == var) & counts.modelled]
            for kind, t in [("linear", 0)] + [("logistic", int(t)) for t in mod.threshold_mph]:
                jobs.append((sg, var, kind, t, d, cands, selmap[(sg, var, kind, t)]))
    print(f"{len(jobs)} models", flush=True)
    res, imps, oofs = [], [], []
    with Pool(args.workers) as pool:
        for i, (r, im, o) in enumerate(pool.imap_unordered(run, jobs), 1):
            res.append(r)
            imps += im
            oofs.append(o)
            print(f"[{i}/{len(jobs)}] {r['signed_gradient']} {r['variable']} {r['kind']} {r['threshold_mph']}: "
                  f"rounds sel/all {r['rounds_sel_median']}/{r['rounds_all_median']}", flush=True)
    oof = pd.concat(oofs, ignore_index=True)
    # join the add-on out-of-fold predictions and check the rows are identical
    keys = ["signed_gradient", "variable", "kind", "threshold_mph", "time_utc"]
    add = pd.read_parquet(os.path.join(ADDON, "oof_predictions.parquet"))
    if args.only:
        add = add[add.signed_gradient == args.only]
    m = oof.merge(add[keys + ["obs", "pred_addon"]], on=keys, how="outer", suffixes=("", "_addon"),
                  indicator=True)
    bad = (m._merge != "both").sum()
    mism = (m.obs != m.obs_addon).sum()
    assert bad == 0 and mism == 0, f"row mismatch with add-on run: {bad} unmatched, {mism} obs differ"
    oof = m.drop(columns=["_merge", "obs_addon"])
    os.makedirs(OUT, exist_ok=True)
    oof.to_parquet(os.path.join(OUT, "oof_predictions.parquet"), index=False)
    met = pd.DataFrame([{**dict(zip(keys[:4], k)), **metrics(g, k[2])}
                        for k, g in oof.groupby(keys[:4], sort=True)])
    met = met.merge(pd.DataFrame(res), on=keys[:4])
    os.makedirs(OUT, exist_ok=True)
    met[met.kind == "logistic"].dropna(axis=1, how="all").round(4).to_csv(
        os.path.join(OUT, "cv_logistic.csv"), index=False)
    met[met.kind == "linear"].dropna(axis=1, how="all").round(4).to_csv(
        os.path.join(OUT, "cv_linear.csv"), index=False)
    pd.DataFrame(imps).round(4).to_csv(os.path.join(OUT, "importance.csv"), index=False)
    with open(os.path.join(OUT, "gbt_meta.json"), "w") as f:
        json.dump({"lightgbm": lgb.__version__, "params": PARAMS, "max_rounds": MAX_ROUNDS,
                   "early_stopping": EARLY, "min_val_events": MIN_VAL_EVENTS,
                   "variants": {"gbt_sel": "base + add-on fold selections",
                                "gbt_all": "base + all other pairs' signed gradients"},
                   "monotone": "abs_gradient_hpa increasing", "missing": "NaN (native)",
                   "only": args.only, "created_utc": pd.Timestamp.now(tz="UTC").isoformat()}, f, indent=2)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
