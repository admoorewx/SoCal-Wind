#!/usr/bin/env python3
"""Rank signed gradients by their top-K station correlations instead of the single best station.

For each signed gradient and variable (sustained, gust), take the K stations with the highest
Spearman rho (n >= MIN_N) from correlations.parquet and summarise them:
  topK_mean_rho   mean rho of the top K stations (the ranking metric)
  rho_kth         rho of the K-th best station (how deep the strong signal goes)
  max_rho         best single station (the old view)
  topK_sep_{min,median,mean,max}_km
                  pairwise great-circle distances between the top K stations
                  (small median = the strong correlations sit in one local area)
                  Stations without valid coordinates (missing, or lat/lon == 0 placeholders on
                  portable RAWS) are left out of the distances but still count in the rho stats.
Ranks by max_rho and by topK_mean_rho are compared.

With --min-sep-km D (> 0) the top K are picked greedily in descending rho, skipping any station
within D km of one already picked, so one local cluster cannot fill the top K. Stations without
coordinates are skipped in that mode. The output also gives rank_top{K}_nosep (the plain top-K
rank) and rank_change_vs_nosep for comparison.

-> derived/gradients/top10_summary.csv            (K=10, no separation; the default)
-> derived/gradients/top10_sep25km_summary.csv    (--min-sep-km 25)

Usage:
  python explore_top10.py [--k 10] [--min-n 50] [--min-sep-km 25]
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GDIR = os.path.normpath(os.path.join(HERE, "..", "data", "v2", "derived", "gradients"))


def km_to_many(lat, lon, lats, lons):
    la, lo, las, los = np.radians(lat), np.radians(lon), np.radians(lats), np.radians(lons)
    a = np.sin((las - la) / 2) ** 2 + np.cos(la) * np.cos(las) * np.sin((los - lo) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def pick_separated(g, k, min_km):
    """Greedy: highest rho first, skip stations within min_km of any already picked."""
    g = g.dropna(subset=["lat", "lon"]).sort_values("rho", ascending=False)
    lat, lon = g.lat.to_numpy(), g.lon.to_numpy()
    picked = []
    for i in range(len(g)):
        if not picked or km_to_many(lat[i], lon[i], lat[picked], lon[picked]).min() >= min_km:
            picked.append(i)
            if len(picked) == k:
                break
    return g.iloc[picked]


def pairwise_km(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    i, j = np.triu_indices(len(la), 1)
    a = (np.sin((la[i] - la[j]) / 2) ** 2
         + np.cos(la[i]) * np.cos(la[j]) * np.sin((lo[i] - lo[j]) / 2) ** 2)
    d = 2 * 6371.0 * np.arcsin(np.sqrt(a))
    return d[~np.isnan(d)]  # pairs with a station lacking coordinates are dropped


def sep_stats(top, k):
    d = pairwise_km(top.lat.to_numpy(), top.lon.to_numpy())
    stats = {"min": np.min, "median": np.median, "mean": np.mean, "max": np.max}
    return {f"top{k}_sep_{name}_km": (float(f(d)) if len(d) else np.nan) for name, f in stats.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--min-n", type=int, default=50)
    ap.add_argument("--min-sep-km", type=float, default=0.0)
    args = ap.parse_args()
    k = args.k

    c = pd.read_parquet(os.path.join(GDIR, "correlations.parquet"),
                        columns=["signed_gradient", "variable", "stid", "n", "rho", "lat", "lon"])
    c = c[(c.n >= args.min_n) & c.rho.notna()]
    # Some portable RAWS have placeholder coordinates (lat or lon == 0): treat as unknown location.
    bad = (c.lat == 0) | (c.lon == 0)
    c.loc[bad, ["lat", "lon"]] = np.nan

    rows = []
    for (sg, var), g in c.groupby(["signed_gradient", "variable"]):
        plain = g.nlargest(k, "rho")
        top = pick_separated(g, k, args.min_sep_km) if args.min_sep_km > 0 else plain
        rows.append({"signed_gradient": sg, "variable": var, "n_picked": len(top),
                     "nosep_mean_rho": plain.rho.mean(),
                     f"top{k}_mean_rho": top.rho.mean(), "rho_kth": top.rho.iloc[-1],
                     "max_rho": top.rho.iloc[0], "n_rho_ge_0.5": int((g.rho >= 0.5).sum()),
                     **sep_stats(top, k),
                     f"top{k}_stations": " ".join(top.stid)})
    s = pd.DataFrame(rows)
    for var, g in s.groupby("variable"):
        s.loc[g.index, "rank_max"] = g.max_rho.rank(ascending=False, method="min")
        s.loc[g.index, f"rank_top{k}"] = g[f"top{k}_mean_rho"].rank(ascending=False, method="min")
        s.loc[g.index, f"rank_top{k}_nosep"] = g.nosep_mean_rho.rank(ascending=False, method="min")
    s["rank_change"] = s.rank_max - s[f"rank_top{k}"]  # positive = moves up under top-K
    rank_cols = ["rank_max", f"rank_top{k}", "rank_change"]
    if args.min_sep_km > 0:
        s["rank_change_vs_nosep"] = s[f"rank_top{k}_nosep"] - s[f"rank_top{k}"]
        rank_cols += [f"rank_top{k}_nosep", "rank_change_vs_nosep"]
    else:
        s = s.drop(columns=["nosep_mean_rho", f"rank_top{k}_nosep", "n_picked"])
    s = s.sort_values(["variable", f"rank_top{k}"]).round(3)
    for col in rank_cols:
        s[col] = s[col].astype(int)

    sep = f"_sep{args.min_sep_km:g}km" if args.min_sep_km > 0 else ""
    path = os.path.join(GDIR, f"top{k}{sep}_summary.csv")
    s.to_csv(path, index=False)
    show = ["signed_gradient", f"rank_top{k}", "rank_max", "rank_change", f"top{k}_mean_rho",
            "rho_kth", "max_rho", "n_rho_ge_0.5"] + [f"top{k}_sep_{x}_km" for x in ("min", "median", "mean", "max")]
    if args.min_sep_km > 0:
        show[3:3] = [f"rank_top{k}_nosep", "rank_change_vs_nosep"]
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        for var in ("sustained", "gust"):
            print(f"\n=== {var} ===")
            print(s.loc[s.variable == var, show].to_string(index=False))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
