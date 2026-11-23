#!/usr/bin/env python3
"""Sign-independent 2026 holdout targets for NWP verification (Phase 3, docs/nwp_verification_plan.md).

The training/holdout targets (build_targets.py) exist only at hours where the OBSERVED gradient has the set's
sign. In verification the FORECAST sign picks the model, so each signed set needs its target at every hour.
Here every signed set gets a target at every hour of its station1 time grid (the METAR times used for
matching winds), whatever the observed sign, and even when this pair's observed gradient is missing.

Rules are identical to training because build_targets.table_for / add_time are reused:
  sustained: 75th percentile (numpy linear) of reporting set stations
  gust:      k-th highest, k = max(2, ceil(0.1 * n_reporting)), NaN if fewer than k gusts
  qualifies: n_reporting >= 50% of n_set

Extra columns:
  valid_utc            obs time (:50-:56) rounded up to the full hour, the key for matching NWP valid times
  gradient_hpa         observed gradient of the pair (NaN when missing at that hour)
  obs_sign             '+', '-', '0' or '' (missing)
  obs_sign_matches     observed sign equals the set's sign (these rows reproduce the training-style holdout)

-> derived/verification/holdout_targets_signfree_<variable>.parquet
-> derived/verification/holdout_targets_signfree_check.csv   agreement with targets/<variable>_holdout.parquet

Usage:
  python build_holdout_targets_signfree.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

import build_targets as bt
import gradient_pipeline as gp

DER = bt.DER
OUT = os.path.join(DER, "verification")
START, END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-11-01", tz="UTC")


def main():
    sets = pd.read_csv(os.path.join(DER, "models", "station_sets.csv"))
    pairs = gp.load_pairs(gp.DEFAULT_GRADIENTS)
    ts = pd.read_parquet(os.path.join(DER, "gradients", "gradient_timeseries.parquet"),
                         columns=["gradient", "time_utc", "gradient_hpa"])
    ts = ts[(ts.time_utc >= START - pd.Timedelta(hours=1)) & (ts.time_utc < END)]
    ts = ts.assign(time_utc=ts.time_utc.astype("datetime64[ms, UTC]"))
    tables = {"sustained": [], "gust": []}
    for s1 in sorted({p[0] for p in pairs.values()}):
        sgs = sorted(sg for sg in sets.signed_gradient.unique() if pairs[sg[:-1]][0] == s1)
        if not sgs:
            continue
        need = sorted(set(sets[sets.signed_gradient.isin(sgs)].stid))
        files = [f for f in sorted(glob.glob(os.path.join(DER, "gradients", "matched", s1, "*.parquet")))
                 if os.path.basename(f) >= "2025-12"]
        M = pd.concat([pd.read_parquet(f, columns=["stid", "time_utc", *bt.COLS.values(), bt.WDIR],
                                       filters=[("stid", "in", need)]) for f in files])
        M["stid"] = M.stid.astype(str)
        M["time_utc"] = M.time_utc.astype("datetime64[ms, UTC]")
        M = M[(M.time_utc >= START - pd.Timedelta(hours=1)) & (M.time_utc < END)]
        # station1 time grid, one time per valid hour (first, as in build_targets)
        grid = pd.DataFrame({"time_utc": np.sort(M.time_utc.unique())})
        grid["valid_utc"] = grid.time_utc.dt.ceil("h")
        grid = grid.drop_duplicates("valid_utc")
        grid = grid[(grid.valid_utc >= START) & (grid.valid_utc < END)]
        for sg in sgs:
            pair = sg[:-1]
            obs = ts[ts.gradient == pair][["time_utc", "gradient_hpa"]].drop_duplicates("time_utc")
            g = grid.merge(obs, on="time_utc", how="left")
            for var in ("sustained", "gust"):
                stids = sets[(sets.signed_gradient == sg) & (sets.variable == var)].stid.tolist()
                if not stids:
                    continue
                t = bt.table_for(sg, var, stids, g, M)
                t = t.merge(g[["time_utc", "valid_utc"]], on="time_utc", how="left")
                gh = t.gradient_hpa
                t["obs_sign"] = np.select([gh > 0, gh < 0, gh == 0], ["+", "-", "0"], "")
                t["obs_sign_matches"] = t.obs_sign == sg[-1]
                t["abs_gradient_hpa"] = gh.abs()
                tables[var].append(t)
            print(f"{sg}: {len(grid):,} grid hours", flush=True)

    os.makedirs(OUT, exist_ok=True)
    checks = []
    for var, parts in tables.items():
        df = bt.add_time(pd.concat(parts, ignore_index=True))
        df["n_reporting"] = df.n_reporting.astype("int16")
        df["n_wdir"] = df.n_wdir.astype("int16")
        lead = ["signed_gradient", "valid_utc", "time_utc", "gradient_hpa", "obs_sign", "obs_sign_matches"]
        df = df[lead + [c for c in df.columns if c not in lead]]
        df.to_parquet(os.path.join(OUT, f"holdout_targets_signfree_{var}.parquet"), index=False)
        print(f"Wrote holdout_targets_signfree_{var}.parquet: {len(df):,} rows, {df.qualifies.sum():,} qualifying, "
              f"{df.obs_sign_matches.sum():,} with observed sign = set sign")

        # consistency with the training-style holdout (same-sign hours must reproduce it exactly)
        ref = pd.read_parquet(os.path.join(DER, "models", "targets", f"{var}_holdout.parquet"),
                              columns=["signed_gradient", "time_utc", "target_mph", "n_reporting", "qualifies"])
        ref = ref.assign(time_utc=ref.time_utc.astype("datetime64[ms, UTC]"))
        j = ref.merge(df[["signed_gradient", "time_utc", "target_mph", "n_reporting", "qualifies"]],
                      on=["signed_gradient", "time_utc"], how="left", suffixes=("_ref", "_new"))
        both = j.target_mph_ref.notna() & j.target_mph_new.notna()
        checks.append(dict(variable=var, ref_rows=len(ref), found_in_new=int(j.n_reporting_new.notna().sum()),
                           target_max_abs_diff=float((j.target_mph_ref - j.target_mph_new)[both].abs().max()),
                           target_nan_mismatch=int((j.target_mph_ref.isna() != j.target_mph_new.isna()).sum()),
                           n_reporting_mismatch=int((j.n_reporting_ref != j.n_reporting_new).sum()),
                           qualifies_mismatch=int((j.qualifies_ref != j.qualifies_new).sum())))
    c = pd.DataFrame(checks)
    c.to_csv(os.path.join(OUT, "holdout_targets_signfree_check.csv"), index=False)
    print(c.to_string(index=False))
    with open(os.path.join(OUT, "holdout_targets_signfree_meta.json"), "w") as f:
        json.dump({"period": [str(START), str(END)], "rules": "build_targets.table_for (identical to training)",
                   "time_grid": "station1 matched METAR times, one per valid hour (ceil)",
                   "created_utc": pd.Timestamp.now(tz="UTC").isoformat()}, f, indent=2)


if __name__ == "__main__":
    main()
