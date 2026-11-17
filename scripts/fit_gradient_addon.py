#!/usr/bin/env python3
"""Step 4b: add other pairs' pressure gradients to the baseline models, leave-one-year-out CV.

Base model: |own gradient| + hour (sin/cos) + day of year (sin/cos), i.e. the step-3 full model.
Add-on: up to MAX_ADD other pairs' SIGNED gradients (pairs from gradients.json, own pair excluded),
chosen by forward selection on BIC using ONLY the training years of each fold (so the held-out year
never influences which pairs are picked). The final model repeats the selection on all of 2015-2025.

Rows are exactly the step-3 rows (qualifying hours with a target). Other gradients are joined by
hour; a missing value is filled with that pair's mean over the training years of the fold.
The 2026 holdout is not touched.

Metrics (pooled out of fold), base vs add-on on identical rows:
  logistic: BSS (vs flat climatology), AUC, top-end calibration ratio (observed rate / mean forecast
            in the top TOP_FRAC of forecasts, >= MIN_TOP hours; same definition as step 4a)
  linear:   R2, RMSE

-> derived/models/gradient_addon/cv_logistic.csv, cv_linear.csv
-> derived/models/gradient_addon/selections.csv      pairs chosen in each fold and in the final fit
-> derived/models/gradient_addon/coefficients.csv    final fits
-> derived/models/gradient_addon/oof_predictions.parquet

Usage:
  python fit_gradient_addon.py [--workers 12]
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

import fit_baseline_models as fb
import gradient_pipeline as gp

HERE = os.path.dirname(os.path.abspath(__file__))
DER = os.path.normpath(os.path.join(HERE, "..", "data", "v2", "derived"))
MDIR = os.path.join(DER, "models")
OUT = os.path.join(MDIR, "gradient_addon")
BASE = fb.FEATURES["grad+hour+doy"]
MAX_ADD = 3
TOP_FRAC, MIN_TOP = 0.02, 100
CUT = pd.Timestamp("2026-01-01", tz="UTC")


def loglik(X, y, b, kind):
    if kind == "logistic":
        return fb.loglik(X, y, b)
    r = y - X @ b
    return -0.5 * len(y) * np.log(np.mean(r ** 2))  # Gaussian, up to a constant


def fit(X, y, kind):
    return fb.fit_logistic(X, y) if kind == "logistic" else fb.fit_linear(X, y)


def predict(X, b, kind):
    return fb.predict_logistic(X, b) if kind == "logistic" else X @ b


def matrix(d, cols, means):
    X = [np.ones(len(d))]
    for c in cols:
        v = d[c].to_numpy(float)
        X.append(np.where(np.isnan(v), means.get(c, 0.0), v))
    return np.column_stack(X)


def forward_select(d, y, cands, kind):
    """Greedy BIC forward selection of other gradients on top of BASE; returns chosen list."""
    means = {c: d[c].mean() for c in cands}
    n = len(y)
    chosen = []
    X = matrix(d, BASE, means)
    best_bic = -2 * loglik(X, y, fit(X, y, kind), kind) + X.shape[1] * np.log(n)
    for _ in range(MAX_ADD):
        trial = []
        for c in cands:
            if c in chosen:
                continue
            Xc = matrix(d, BASE + chosen + [c], means)
            b = fit(Xc, y, kind)
            trial.append((-2 * loglik(Xc, y, b, kind) + Xc.shape[1] * np.log(n), c))
        if not trial:
            break
        bic, c = min(trial)
        if bic >= best_bic:
            break
        best_bic = bic
        chosen.append(c)
    return chosen


def top_ratio(y, p):
    n = max(MIN_TOP, int(np.ceil(TOP_FRAC * len(p))))
    idx = np.argsort(-p)[:n]
    return y[idx].mean() / p[idx].mean()


def run(job):
    sg, var, kind, T, d, cands = job
    y = (d[f"ev_{T}"] if kind == "logistic" else d.target_mph).to_numpy(float)
    years = d.year.to_numpy()
    p_base, p_add = np.full(len(d), np.nan), np.full(len(d), np.nan)
    p_clim = np.full(len(d), np.nan)
    sel = []
    for yr in np.unique(years):
        tr, te = years != yr, years == yr
        dtr, dte = d[tr], d[te]
        means = {c: dtr[c].mean() for c in cands}
        p_clim[te] = (y[tr].mean())
        Xb_tr, Xb_te = matrix(dtr, BASE, means), matrix(dte, BASE, means)
        p_base[te] = predict(Xb_te, fit(Xb_tr, y[tr], kind), kind)
        chosen = forward_select(dtr, y[tr], cands, kind)
        Xa_tr, Xa_te = matrix(dtr, BASE + chosen, means), matrix(dte, BASE + chosen, means)
        p_add[te] = predict(Xa_te, fit(Xa_tr, y[tr], kind), kind)
        sel.append({"fold": int(yr), "chosen": "|".join(chosen)})
    final = forward_select(d, y, cands, kind)
    means = {c: d[c].mean() for c in cands}
    b = fit(matrix(d, BASE + final, means), y, kind)
    sel.append({"fold": "final", "chosen": "|".join(final)})
    key = {"signed_gradient": sg, "variable": var, "kind": kind, "threshold_mph": T}
    if kind == "logistic":
        bc = np.mean((p_clim - y) ** 2)
        m = {"n_hours": len(y), "event_hours": int(y.sum()), "base_rate": y.mean()}
        for lab, p in (("base", p_base), ("addon", p_add)):
            m[f"bss_{lab}"] = 1 - np.mean((p - y) ** 2) / bc
            m[f"auc_{lab}"] = fb.auc(y, p)
            m[f"top_ratio_{lab}"] = top_ratio(y, p)
    else:
        cm = np.mean((p_clim - y) ** 2)
        m = {"n_hours": len(y)}
        for lab, p in (("base", p_base), ("addon", p_add)):
            m[f"r2_{lab}"] = 1 - np.mean((p - y) ** 2) / cm
            m[f"rmse_{lab}"] = np.sqrt(np.mean((p - y) ** 2))
    coefs = {**key, **dict(zip(["intercept"] + BASE + final, b))}
    oof = pd.DataFrame({**key, "time_utc": d.time_utc.to_numpy(), "obs": y.astype("float32"),
                        "pred_base": p_base.astype("float32"), "pred_addon": p_add.astype("float32")})
    return {**key, **m, "final_chosen": "|".join(final)}, [{**key, **s} for s in sel], coefs, oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    pairs = list(gp.load_pairs(gp.DEFAULT_GRADIENTS))
    ts = pd.read_parquet(os.path.join(DER, "gradients", "gradient_timeseries.parquet"),
                         columns=["gradient", "time_utc", "gradient_hpa"])
    ts = ts[ts.gradient.isin(pairs)]  # drops stale series not in gradients.json (e.g. SAC-RDD)
    ts["hour"] = ts.time_utc.dt.floor("h")
    G = ts.pivot_table(index="hour", columns="gradient", values="gradient_hpa", aggfunc="first")
    G.index = G.index.astype("datetime64[ms, UTC]")
    G.columns = [f"g_{c}" for c in G.columns]
    counts = pd.read_csv(os.path.join(MDIR, "targets", "event_counts.csv"))
    jobs = []
    for var in ("sustained", "gust"):
        T = pd.read_parquet(os.path.join(MDIR, "targets", f"{var}_train.parquet"))
        T = T[T.qualifies & T.target_mph.notna()]
        T = T.assign(hour=T.time_utc.dt.floor("h")).join(G, on="hour")
        for sg, d in T.groupby("signed_gradient", sort=True):
            d = d.reset_index(drop=True)
            cands = [f"g_{p}" for p in pairs if p != sg[:-1]]
            jobs.append((sg, var, "linear", 0, d, cands))
            mod = counts[(counts.signed_gradient == sg) & (counts.variable == var) & counts.modelled]
            for t in mod.threshold_mph:
                jobs.append((sg, var, "logistic", int(t), d, cands))
    print(f"{len(jobs)} models", flush=True)
    res, sels, coefs, oofs = [], [], [], []
    with Pool(args.workers) as pool:
        for i, (r, s, c, o) in enumerate(pool.imap_unordered(run, jobs), 1):
            res.append(r)
            sels.extend(s)
            coefs.append(c)
            oofs.append(o)
            print(f"[{i}/{len(jobs)}] {r['signed_gradient']} {r['variable']} {r['kind']} {r['threshold_mph']}: "
                  f"{r['final_chosen'] or '-'}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    res = pd.DataFrame(res).sort_values(["kind", "variable", "threshold_mph", "signed_gradient"])
    res[res.kind == "logistic"].dropna(axis=1, how="all").round(4).to_csv(
        os.path.join(OUT, "cv_logistic.csv"), index=False)
    res[res.kind == "linear"].dropna(axis=1, how="all").round(4).to_csv(
        os.path.join(OUT, "cv_linear.csv"), index=False)
    pd.DataFrame(sels).to_csv(os.path.join(OUT, "selections.csv"), index=False)
    pd.DataFrame(coefs).round(5).to_csv(os.path.join(OUT, "coefficients.csv"), index=False)
    pd.concat(oofs, ignore_index=True).to_parquet(os.path.join(OUT, "oof_predictions.parquet"), index=False)
    with open(os.path.join(OUT, "addon_meta.json"), "w") as f:
        json.dump({"base": BASE, "max_add": MAX_ADD, "selection": "forward, BIC, within training folds",
                   "candidates": "signed gradients of the other pairs in gradients.json",
                   "missing": "filled with training-fold mean", "top_frac": TOP_FRAC, "min_top": MIN_TOP,
                   "created_utc": pd.Timestamp.now(tz="UTC").isoformat()}, f, indent=2)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
