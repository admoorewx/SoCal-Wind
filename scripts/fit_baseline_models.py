#!/usr/bin/env python3
"""Baseline gradient wind models with leave-one-year-out CV (step 3 of the model plan).

Reads derived/models/targets/<variable>_train.parquet (2015-2025, qualifying hours only) and
event_counts.csv (which (signed gradient, threshold) pairs have enough events to model).
The 2026 holdout is NOT touched.

Nested predictor sets (hour and day of year enter as sin/cos pairs):
  clim            intercept only (reference: training-fold base rate / mean)
  grad            |gradient|
  grad+hour       |gradient|, hour_sin, hour_cos
  grad+hour+doy   |gradient|, hour_sin, hour_cos, doy_sin, doy_cos

Logistic (one per signed gradient, variable, modelled threshold): event = ev_<T>.
  Metrics on pooled out-of-fold predictions: Brier, Brier skill score vs clim (BSS), ROC AUC,
  log loss; plus per-year BSS (min / median) for stability.
Linear (one per signed gradient, variable): target_mph.
  Metrics: RMSE, MAE, bias, R2 (= 1 - MSE / MSE of clim, out of fold), per-year R2 (min / median).

Fitting is plain numpy: damped-Newton IRLS for logistic (step halving, tiny ridge), least squares for linear.
Coefficients are also refit on all of 2015-2025 for inspection.

-> derived/models/baseline/cv_logistic.csv, cv_linear.csv      pooled CV metrics
-> derived/models/baseline/coefficients.csv                    full-period fits
-> derived/models/baseline/oof_predictions.parquet             out-of-fold predictions

Usage:
  python fit_baseline_models.py
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TDIR = os.path.normpath(os.path.join(HERE, "..", "data", "v2", "derived", "models", "targets"))
OUT = os.path.normpath(os.path.join(HERE, "..", "data", "v2", "derived", "models", "baseline"))
FEATURES = {
    "clim": [],
    "grad": ["abs_gradient_hpa"],
    "grad+hour": ["abs_gradient_hpa", "hour_sin", "hour_cos"],
    "grad+hour+doy": ["abs_gradient_hpa", "hour_sin", "hour_cos", "doy_sin", "doy_cos"],
}
RIDGE = 1e-6
EPS = 1e-12


def design(df, cols):
    return np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in cols])


def loglik(X, y, b):
    z = X @ b
    return np.sum(y * z - np.logaddexp(0, z)) - 0.5 * RIDGE * b @ b


def fit_logistic(X, y, iters=100, tol=1e-8):
    """Damped Newton (IRLS): a step is halved until the penalised log-likelihood improves,
    which keeps rare-event fits from overshooting."""
    b = np.zeros(X.shape[1])
    p0 = np.clip(y.mean(), 1e-6, 1 - 1e-6)
    b[0] = np.log(p0 / (1 - p0))
    ll = loglik(X, y, b)
    for _ in range(iters):
        p = predict_logistic(X, b)
        w = np.clip(p * (1 - p), 1e-10, None)
        H = X.T @ (X * w[:, None]) + RIDGE * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - RIDGE * b)
        t = 1.0
        while t > 1e-6:
            nb = b + t * step
            nll = loglik(X, y, nb)
            if nll >= ll - 1e-12:
                break
            t /= 2
        else:
            break  # no improving step: converged as far as numerically possible
        done = abs(nll - ll) < tol * (abs(ll) + 1) and np.max(np.abs(t * step)) < 1e-6
        b, ll = nb, nll
        if done:
            break
    return b


def predict_logistic(X, b):
    z = X @ b
    return np.exp(-np.logaddexp(0, -z))  # stable sigmoid


def fit_linear(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def auc(y, p):
    r = pd.Series(p).rank().to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def logloss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def loyo(df, ycol, kind):
    """Out-of-fold predictions for every predictor set; folds are calendar years."""
    years = df.year.to_numpy()
    y = df[ycol].to_numpy(float)
    oof = {m: np.full(len(df), np.nan) for m in FEATURES}
    for yr in np.unique(years):
        tr, te = years != yr, years == yr
        for m, cols in FEATURES.items():
            Xtr, Xte = design(df[tr], cols), design(df[te], cols)
            if kind == "logistic":
                if y[tr].sum() == 0:
                    oof[m][te] = 0.0
                    continue
                oof[m][te] = predict_logistic(Xte, fit_logistic(Xtr, y[tr]))
            else:
                oof[m][te] = Xte @ fit_linear(Xtr, y[tr])
    return y, oof


def logistic_metrics(y, oof, years):
    clim_brier = np.mean((oof["clim"] - y) ** 2)
    rows = []
    for m, p in oof.items():
        brier = np.mean((p - y) ** 2)
        per = []
        for yr in np.unique(years):
            s = years == yr
            cb = np.mean((oof["clim"][s] - y[s]) ** 2)
            per.append(1 - np.mean((p[s] - y[s]) ** 2) / cb if cb > 0 else np.nan)
        rows.append({"model": m, "brier": brier, "bss": 1 - brier / clim_brier, "auc": auc(y, p),
                     "logloss": logloss(y, p), "bss_year_min": np.nanmin(per),
                     "bss_year_median": np.nanmedian(per)})
    return rows, per


def linear_metrics(y, oof, years):
    clim_mse = np.mean((oof["clim"] - y) ** 2)
    rows = []
    for m, p in oof.items():
        e = p - y
        per = []
        for yr in np.unique(years):
            s = years == yr
            cm = np.mean((oof["clim"][s] - y[s]) ** 2)
            per.append(1 - np.mean(e[s] ** 2) / cm if cm > 0 else np.nan)
        rows.append({"model": m, "rmse": np.sqrt(np.mean(e ** 2)), "mae": np.mean(np.abs(e)),
                     "bias": np.mean(e), "r2": 1 - np.mean(e ** 2) / clim_mse,
                     "r2_year_min": np.nanmin(per), "r2_year_median": np.nanmedian(per)})
    return rows


def main():
    counts = pd.read_csv(os.path.join(TDIR, "event_counts.csv"))
    os.makedirs(OUT, exist_ok=True)
    log_rows, lin_rows, coef_rows, oof_parts = [], [], [], []
    for var in ("sustained", "gust"):
        df = pd.read_parquet(os.path.join(TDIR, f"{var}_train.parquet"))
        df = df[df.qualifies & df.target_mph.notna()]
        for sg, d in df.groupby("signed_gradient", sort=True):
            d = d.reset_index(drop=True)
            years = d.year.to_numpy()
            keep = {"signed_gradient": sg, "variable": var, "time_utc": d.time_utc}
            # linear
            y, oof = loyo(d, "target_mph", "linear")
            for r in linear_metrics(y, oof, years):
                lin_rows.append({"signed_gradient": sg, "variable": var, "n_hours": len(d),
                                 "target_mean": y.mean(), "target_sd": y.std(), **r})
            for m, cols in FEATURES.items():
                b = fit_linear(design(d, cols), y)
                coef_rows.append({"signed_gradient": sg, "variable": var, "kind": "linear",
                                  "threshold_mph": None, "model": m,
                                  **dict(zip(["intercept"] + cols, b))})
            oof_parts.append(pd.DataFrame({**keep, "kind": "linear", "threshold_mph": 0,
                                           "obs": y, **{f"pred_{m}": oof[m] for m in FEATURES}}))
            # logistic, modelled thresholds only
            mod = counts[(counts.signed_gradient == sg) & (counts.variable == var) & counts.modelled]
            for T in mod.threshold_mph:
                y, oof = loyo(d, f"ev_{T}", "logistic")
                rows, _ = logistic_metrics(y, oof, years)
                for r in rows:
                    log_rows.append({"signed_gradient": sg, "variable": var, "threshold_mph": T,
                                     "n_hours": len(d), "event_hours": int(y.sum()),
                                     "base_rate": y.mean(), **r})
                for m, cols in FEATURES.items():
                    b = fit_logistic(design(d, cols), y)
                    coef_rows.append({"signed_gradient": sg, "variable": var, "kind": "logistic",
                                      "threshold_mph": T, "model": m,
                                      **dict(zip(["intercept"] + cols, b))})
                oof_parts.append(pd.DataFrame({**keep, "kind": "logistic", "threshold_mph": T,
                                               "obs": y, **{f"pred_{m}": oof[m] for m in FEATURES}}))
            print(f"{var:9s} {sg:9s} {len(d):7,} h, {len(mod)} thresholds", flush=True)

    r = lambda df: df.round(4)
    pd.DataFrame(log_rows).pipe(r).to_csv(os.path.join(OUT, "cv_logistic.csv"), index=False)
    pd.DataFrame(lin_rows).pipe(r).to_csv(os.path.join(OUT, "cv_linear.csv"), index=False)
    pd.DataFrame(coef_rows).pipe(r).to_csv(os.path.join(OUT, "coefficients.csv"), index=False)
    oof = pd.concat(oof_parts, ignore_index=True)
    for c in oof.columns:
        if c.startswith("pred_") or c == "obs":
            oof[c] = oof[c].astype("float32")
    oof.to_parquet(os.path.join(OUT, "oof_predictions.parquet"), index=False)
    with open(os.path.join(OUT, "baseline_meta.json"), "w") as f:
        json.dump({"features": FEATURES, "cv": "leave-one-year-out, 2015-2025", "ridge": RIDGE,
                   "rows": "qualifying hours with a target", "created_utc": pd.Timestamp.now(tz="UTC").isoformat()},
                  f, indent=2)
    print(f"Wrote cv_logistic.csv, cv_linear.csv, coefficients.csv, oof_predictions.parquet in {OUT}")


if __name__ == "__main__":
    main()
