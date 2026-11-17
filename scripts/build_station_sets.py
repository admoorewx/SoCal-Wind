#!/usr/bin/env python3
"""Station sets for the gradient wind models (step 1 of the model plan).

Hybrid rule, per signed gradient and variable (sustained, gust), using correlations computed on the
training period only (gradient_pipeline.py correlate --before 2026-01-01), stations with n >= MIN_N:
  core   every station with rho >= RHO_CORE
  fill   if core has fewer than MIN_STATIONS, add the next-best stations with rho >= RHO_FILL
         until there are MIN_STATIONS
  skip   if there are still fewer than MIN_STATIONS, the signed gradient/variable is not modelled

-> derived/models/station_sets.csv         one row per (signed gradient, variable, station) used
-> derived/models/station_set_summary.csv  one row per (signed gradient, variable): counts, rho range,
                                           status (modelled / skipped), and overlap with the
                                           all-years correlations (to show what holding out 2026 changed)

Usage:
  python build_station_sets.py [--corr correlations_before2026-01-01.parquet]
"""
import argparse
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GDIR = os.path.normpath(os.path.join(HERE, "..", "data", "v2", "derived", "gradients"))
MDIR = os.path.normpath(os.path.join(HERE, "..", "data", "v2", "derived", "models"))
MIN_N = 50
RHO_CORE, RHO_FILL, MIN_STATIONS = 0.5, 0.4, 10


def select(g):
    g = g.sort_values("rho", ascending=False)
    core = g[g.rho >= RHO_CORE].assign(tier="core")
    fill = g[(g.rho >= RHO_FILL) & (g.rho < RHO_CORE)].head(max(0, MIN_STATIONS - len(core))).assign(tier="fill")
    return pd.concat([core, fill])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr", default="correlations_before2026-01-01.parquet")
    args = ap.parse_args()
    path = os.path.join(GDIR, args.corr)
    c = pd.read_parquet(path, columns=["gradient", "sign", "signed_gradient", "variable", "stid", "n", "rho"])
    c = c[(c.n >= MIN_N) & c.rho.notna()]

    allyr = pd.read_parquet(os.path.join(GDIR, "correlations.parquet"),
                            columns=["signed_gradient", "variable", "stid", "n", "rho"])
    allyr_core = allyr[(allyr.n >= MIN_N) & (allyr.rho >= RHO_CORE)].groupby(
        ["signed_gradient", "variable"]).stid.apply(set)

    sets, summ = [], []
    for (sg, var), g in c.groupby(["signed_gradient", "variable"], sort=False):
        s = select(g)
        status = "modelled" if len(s) >= MIN_STATIONS else "skipped"
        core = set(s.loc[s.tier == "core", "stid"])
        prev = allyr_core.get((sg, var), set())
        summ.append({"signed_gradient": sg, "gradient": g.gradient.iloc[0], "sign": g.sign.iloc[0],
                     "variable": var, "status": status, "n_core": len(core),
                     "n_fill": int((s.tier == "fill").sum()), "n_total": len(s),
                     "rho_max": round(s.rho.max(), 3) if len(s) else None,
                     "rho_min": round(s.rho.min(), 3) if len(s) else None,
                     "rho_median": round(s.rho.median(), 3) if len(s) else None,
                     "core_also_core_all_years": len(core & prev), "core_all_years": len(prev)})
        if status == "modelled":
            s = s.assign(rank=range(1, len(s) + 1))
            sets.append(s[["signed_gradient", "gradient", "sign", "variable", "stid", "tier", "rank", "rho", "n"]])

    os.makedirs(MDIR, exist_ok=True)
    sets = pd.concat(sets)
    summ = pd.DataFrame(summ).sort_values(["variable", "status", "n_total"], ascending=[True, True, False])
    sets.to_csv(os.path.join(MDIR, "station_sets.csv"), index=False)
    summ.to_csv(os.path.join(MDIR, "station_set_summary.csv"), index=False)
    with open(os.path.join(MDIR, "station_sets_meta.json"), "w") as f:
        json.dump({"correlations": args.corr, "min_n": MIN_N, "rho_core": RHO_CORE, "rho_fill": RHO_FILL,
                   "min_stations": MIN_STATIONS, "created_utc": pd.Timestamp.now(tz="UTC").isoformat()}, f, indent=2)

    with pd.option_context("display.width", 220, "display.max_rows", 100):
        for var in ("sustained", "gust"):
            x = summ[summ.variable == var]
            print(f"\n=== {var}: {(x.status == 'modelled').sum()} modelled, {(x.status == 'skipped').sum()} skipped ===")
            print(x.drop(columns=["gradient", "sign", "variable"]).to_string(index=False))
    print(f"\nWrote {MDIR}/station_sets.csv, station_set_summary.csv, station_sets_meta.json")


if __name__ == "__main__":
    main()
