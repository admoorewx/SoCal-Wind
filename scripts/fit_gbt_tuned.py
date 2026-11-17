#!/usr/bin/env python3
"""Step 5b: tune the tree shape of the gbt_all models, aimed at the rare thresholds.

The first tree run (fit_gbt_models.py) used fixed params and chose the number of rounds by early stopping
on the 2+ training years nearest the held-out year. For rare events (~100-250 event hours) that inner
set holds few events, stopping came early and the trees under-forecast at the top.

Here, inside each outer leave-one-year-out fold, the training years are split into 3 interleaved
year groups (inner CV, so each validation group has 3-4 years of events). For every tree shape in GRID
(num_leaves x min_data_in_leaf), lgb.cv early-stops on the mean inner log loss (l2 for linear); the
shape + rounds with the best inner score are refit on all training years and predict the held-out year.
The held-out year is never used for any choice. Final models (all 2015-2025) repeat the inner tuning.

Variant: gbt_all features (base + all other pairs' signed gradients), as in fit_gbt_models.py.
Rows / folds / targets identical (asserted against gbt/oof_predictions.parquet).
Default scope: rare thresholds (gust 50, gust 60, sustained 30); --all for every model.

Reads (never writes) derived/models/gbt/ and gradient_addon/.
-> derived/models/gbt_tuned/cv_logistic.csv, cv_linear.csv   addon vs gbt_all vs gbt_tuned
-> derived/models/gbt_tuned/choices.csv                       shape + rounds chosen per fold
-> derived/models/gbt_tuned/oof_predictions.parquet, models/

Usage:
  python fit_gbt_tuned.py [--workers 14] [--all] [--only LAX-WJF-]
"""
import argparse
import itertools
import json
import os
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np
import pandas as pd

import fit_baseline_models as fb
import fit_gbt_models as fg
import gradient_pipeline as gp

OUT = os.path.join(fg.MDIR, "gbt_tuned")
GBT = os.path.join(fg.MDIR, "gbt")
GRID = list(itertools.product([4, 7, 15], [20, 50, 100]))  # (num_leaves, min_data_in_leaf)
RARE = {("gust", 50), ("gust", 60), ("sustained", 30)}
N_INNER = 3


def tune(X, y, years, tr_years, kind, cols):
    """Inner year-group CV over GRID on the given training years; returns (leaves, min_leaf, rounds, score)."""
    tr_years = sorted(tr_years)
    tr_idx = np.flatnonzero(np.isin(years, tr_years))
    yrs = years[tr_idx]
    groups = {t: i % N_INNER for i, t in enumerate(tr_years)}
    g = np.array([groups[t] for t in yrs])
    folds = [(np.flatnonzero(g != k), np.flatnonzero(g == k)) for k in range(N_INNER)]
    ds = lgb.Dataset(X[tr_idx], y[tr_idx], feature_name=cols, free_raw_data=False)
    best = None
    for leaves, mleaf in GRID:
        p = fg.params(kind, cols)
        p.update(num_leaves=leaves, min_data_in_leaf=mleaf)
        r = lgb.cv(p, ds, fg.MAX_ROUNDS, folds=folds, stratified=False,
                   callbacks=[lgb.early_stopping(fg.EARLY, verbose=False)])
        curve = r[f"valid {p['metric']}-mean"]
        n = int(np.argmin(curve)) + 1
        if best is None or curve[n - 1] < best[3]:
            best = (leaves, mleaf, max(n, 10), float(curve[n - 1]))
    return best


def fit(X, y, idx, kind, cols, leaves, mleaf, n):
    p = fg.params(kind, cols)
    p.update(num_leaves=leaves, min_data_in_leaf=mleaf)
    return lgb.train(p, lgb.Dataset(X[idx], y[idx], feature_name=cols), n)


def run(job):
    sg, var, kind, T, d, cands = job
    cols = fg.BASE + cands
    X = d[cols].to_numpy(float)
    y = (d[f"ev_{T}"] if kind == "logistic" else d.target_mph).to_numpy(float)
    years = d.year.to_numpy()
    uy = np.unique(years)
    pred = np.full(len(d), np.nan)
    key = {"signed_gradient": sg, "variable": var, "kind": kind, "threshold_mph": T}
    ch = []
    for yr in uy:
        te = years == yr
        tr_years = [t for t in uy if t != yr]
        leaves, mleaf, n, sc = tune(X, y, years, tr_years, kind, cols)
        pred[te] = fit(X, y, ~te, kind, cols, leaves, mleaf, n).predict(X[te])
        ch.append({**key, "fold": str(yr), "num_leaves": leaves, "min_data_in_leaf": mleaf,
                   "rounds": n, "inner_score": sc})
    leaves, mleaf, n, sc = tune(X, y, years, list(uy), kind, cols)
    m = fit(X, y, np.ones(len(y), bool), kind, cols, leaves, mleaf, n)
    os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
    m.save_model(os.path.join(OUT, "models", f"{sg}_{var}_{kind}_{T}__gbt_tuned.txt"))
    ch.append({**key, "fold": "final", "num_leaves": leaves, "min_data_in_leaf": mleaf,
               "rounds": n, "inner_score": sc})
    oof = pd.DataFrame({**key, "time_utc": d.time_utc.to_numpy(), "obs": y.astype("float32"),
                        "pred_gbt_tuned": pred.astype("float32")})
    return key, ch, oof


def metrics(g, kind, labs):
    y, c = g.obs.to_numpy(float), g.clim.to_numpy(float)
    out = {}
    if kind == "logistic":
        bc = np.mean((c - y) ** 2)
        out.update(event_hours=int(y.sum()), base_rate=y.mean())
        for lab in labs:
            p = g[f"pred_{lab}"].to_numpy(float)
            out[f"bss_{lab}"] = 1 - np.mean((p - y) ** 2) / bc
            out[f"auc_{lab}"] = fb.auc(y, p)
            out[f"logloss_{lab}"] = fb.logloss(y, p)
            out[f"top_ratio_{lab}"] = fg.fa.top_ratio(y, p)
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
    ap.add_argument("--all", action="store_true", help="tune every model, not only rare thresholds")
    ap.add_argument("--only", help="one signed gradient (test)")
    args = ap.parse_args()
    pairs = list(gp.load_pairs(gp.DEFAULT_GRADIENTS))
    ts = pd.read_parquet(os.path.join(fg.DER, "gradients", "gradient_timeseries.parquet"),
                         columns=["gradient", "time_utc", "gradient_hpa"])
    ts = ts[ts.gradient.isin(pairs)]
    ts["hour"] = ts.time_utc.dt.floor("h")
    G = ts.pivot_table(index="hour", columns="gradient", values="gradient_hpa", aggfunc="first")
    G.index = G.index.astype("datetime64[ms, UTC]")
    G.columns = [f"g_{c}" for c in G.columns]
    counts = pd.read_csv(os.path.join(fg.MDIR, "targets", "event_counts.csv"))
    jobs = []
    for var in ("sustained", "gust"):
        T = pd.read_parquet(os.path.join(fg.MDIR, "targets", f"{var}_train.parquet"))
        T = T[T.qualifies & T.target_mph.notna()]
        T = T.assign(hour=T.time_utc.dt.floor("h")).join(G, on="hour")
        for sg, d in T.groupby("signed_gradient", sort=True):
            if args.only and sg != args.only:
                continue
            d = d.reset_index(drop=True)
            cands = [f"g_{p}" for p in pairs if p != sg[:-1]]
            mod = counts[(counts.signed_gradient == sg) & (counts.variable == var) & counts.modelled]
            for kind, t in [("linear", 0)] + [("logistic", int(t)) for t in mod.threshold_mph]:
                if args.all or (var, t) in RARE:
                    jobs.append((sg, var, kind, t, d, cands))
    print(f"{len(jobs)} models, grid {GRID}", flush=True)
    chs, oofs = [], []
    with Pool(args.workers) as pool:
        for i, (k, ch, o) in enumerate(pool.imap_unordered(run, jobs), 1):
            chs += ch
            oofs.append(o)
            f = ch[-1]
            print(f"[{i}/{len(jobs)}] {k['signed_gradient']} {k['variable']} {k['kind']} {k['threshold_mph']}: "
                  f"final leaves {f['num_leaves']} min_leaf {f['min_data_in_leaf']} rounds {f['rounds']}", flush=True)
    keys = ["signed_gradient", "variable", "kind", "threshold_mph", "time_utc"]
    oof = pd.concat(oofs, ignore_index=True)
    ref = pd.read_parquet(os.path.join(GBT, "oof_predictions.parquet"),
                          columns=keys + ["obs", "clim", "pred_addon", "pred_gbt_all"])
    ref = ref.merge(oof[keys[:4]].drop_duplicates(), on=keys[:4])
    m = oof.merge(ref, on=keys, how="outer", suffixes=("", "_ref"), indicator=True)
    bad, mism = (m._merge != "both").sum(), (m.obs != m.obs_ref).sum()
    assert bad == 0 and mism == 0, f"row mismatch with gbt run: {bad} unmatched, {mism} obs differ"
    oof = m.drop(columns=["_merge", "obs_ref"])
    os.makedirs(OUT, exist_ok=True)
    oof.to_parquet(os.path.join(OUT, "oof_predictions.parquet"), index=False)
    pd.DataFrame(chs).to_csv(os.path.join(OUT, "choices.csv"), index=False)
    labs = ("addon", "gbt_all", "gbt_tuned")
    met = pd.DataFrame([{**dict(zip(keys[:4], k)), "n_hours": len(g), **metrics(g, k[2], labs)}
                        for k, g in oof.groupby(keys[:4], sort=True)])
    met[met.kind == "logistic"].dropna(axis=1, how="all").round(4).to_csv(
        os.path.join(OUT, "cv_logistic.csv"), index=False)
    met[met.kind == "linear"].dropna(axis=1, how="all").round(4).to_csv(
        os.path.join(OUT, "cv_linear.csv"), index=False)
    with open(os.path.join(OUT, "gbt_tuned_meta.json"), "w") as f:
        json.dump({"lightgbm": lgb.__version__, "base_params": fg.PARAMS, "grid_leaves_minleaf": GRID,
                   "inner_cv": f"{N_INNER} interleaved year groups of the training years",
                   "selection": "best mean inner logloss (l2 linear), early stopping",
                   "scope": "all" if args.all else sorted(map(list, RARE)), "only": args.only,
                   "created_utc": pd.Timestamp.now(tz="UTC").isoformat()}, f, indent=2)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
