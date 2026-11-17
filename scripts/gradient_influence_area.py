#!/usr/bin/env python3
"""Area influenced by each signed gradient: union of circular influence zones around its stations.

Each station that represents a gradient gets a circle of radius R; overlapping circles are merged
(union), so a tight cluster of stations covers about as much area as one station, while widely
spaced stations each add their own area. Circles are NOT clipped to land (offshore wind matters).

Station sets (per signed gradient and variable, from correlations.parquet, n >= MIN_N):
  top10   the 10 stations with the highest rho
  rho05   every station with rho >= RHO_MIN
  rho05w  every station with rho > RHO_MIN, weighted by its excess over the threshold:
          w = (rho - RHO_MIN) / (1 - RHO_MIN)   (0 at the threshold, 1 at rho = 1)
          Where circles overlap, each point counts at the HIGHEST weight covering it (weights never
          stack), so area_km2 = integral of the max weight = the area the gradient would cover if
          every station were perfectly correlated. footprint_km2 is the unweighted union.
          (Chosen 2026-11-13 over w = rho, which barely differs from unweighted because the
          threshold confines weights to 0.5-0.8, and over top-10 sets, which cap the area.)

Radii: 10, 15, ..., 50 km. Areas are computed in CONUS Albers equal-area (EPSG:5070).
Stations without valid coordinates (missing, or lat/lon == 0 placeholders) are skipped.

-> derived/gradients/influence_area.csv          long: signed_gradient, variable, method, radius_km,
                                                 n_stations, area_km2 (ranking metric; weighted for
                                                 rho05w), footprint_km2, mean_rho, rank
-> derived/gradients/influence_area_ranks.csv    wide: rank (and area) at each radius, per method/variable
-> derived/gradients/influence_area_stability.csv
                                                 Spearman correlation of the gradient ranking at each
                                                 radius with the ranking at 25 km, per method/variable

Usage:
  python gradient_influence_area.py [--min-n 50] [--rho-min 0.5] [--k 10]
"""
import argparse
import os

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GDIR = os.path.normpath(os.path.join(HERE, "..", "data", "v2", "derived", "gradients"))
RADII_KM = list(range(10, 51, 5))
REF_RADIUS_KM = 25
CRS_EQUAL_AREA = "EPSG:5070"  # CONUS Albers equal-area, metres
QUAD_SEGS = 32                # segments per quarter circle (area error < 0.1%)


def excess_weight(rho, rho_min):
    return (np.asarray(rho, dtype=float) - rho_min) / (1 - rho_min)


def station_sets(c, k, rho_min):
    """Yield (signed_gradient, variable, method, frame of stations, weights or None)."""
    tag = f"rho{rho_min:g}".replace(".", "")
    for (sg, var), g in c.groupby(["signed_gradient", "variable"]):
        yield sg, var, f"top{k}", g.nlargest(k, "rho"), None
        yield sg, var, tag, g[g.rho >= rho_min], None
        gw = g[g.rho > rho_min]
        yield sg, var, tag + "w", gw, excess_weight(gw.rho, rho_min)


def load_correlations(min_n, verbose=True):
    """correlations.parquet rows with n >= min_n and valid coordinates, plus every station projected
    once to CRS_EQUAL_AREA (GeoSeries indexed by stid)."""
    c = pd.read_parquet(os.path.join(GDIR, "correlations.parquet"),
                        columns=["signed_gradient", "gradient", "sign", "variable", "stid", "n", "rho", "lat", "lon"])
    c = c[(c.n >= min_n) & c.rho.notna()]
    bad = c.lat.isna() | c.lon.isna() | (c.lat == 0) | (c.lon == 0)
    if bad.any() and verbose:
        print(f"Skipping {c.loc[bad, 'stid'].nunique()} stations without valid coordinates: "
              f"{', '.join(sorted(c.loc[bad, 'stid'].unique()))}")
    c = c[~bad]
    st = c.drop_duplicates("stid")[["stid", "lat", "lon"]]
    pts = gpd.GeoSeries(gpd.points_from_xy(st.lon, st.lat), index=st.stid, crs="EPSG:4326").to_crs(CRS_EQUAL_AREA)
    return c, pts


def footprint(pts, stids, radius_km):
    """Union of radius_km circles around the given stations (CRS_EQUAL_AREA geometry, or None)."""
    if len(stids) == 0:
        return None
    return pts.loc[list(stids)].buffer(radius_km * 1000, quad_segs=QUAD_SEGS).union_all()


def weighted_pieces(pts, stids, weights, radius_km):
    """Split the footprint into pieces that each carry the highest weight covering them.

    Circles are added strongest first; each contributes only its not-yet-covered part.
    Returns [(geometry in CRS_EQUAL_AREA, weight), ...], strongest first.
    """
    stids, weights = list(stids), np.asarray(weights, dtype=float)
    circles = pts.loc[stids].buffer(radius_km * 1000, quad_segs=QUAD_SEGS).to_numpy()
    pieces, covered = [], None
    for i in np.argsort(-weights, kind="stable"):
        if weights[i] <= 0:
            break
        new = circles[i] if covered is None else circles[i].difference(covered)
        if not new.is_empty:
            pieces.append((new, float(weights[i])))
        covered = circles[i] if covered is None else covered.union(circles[i])
    return pieces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=50)
    ap.add_argument("--rho-min", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    c, pts = load_correlations(args.min_n)

    rows = []
    for sg, var, method, g, w in station_sets(c, args.k, args.rho_min):
        for r in RADII_KM:
            fp = footprint(pts, g.stid, r)
            fp_area = fp.area / 1e6 if fp is not None else 0.0
            if w is None:
                area = fp_area
            else:
                area = sum(geom.area * wt for geom, wt in weighted_pieces(pts, g.stid, w, r)) / 1e6
            rows.append({"signed_gradient": sg, "variable": var, "method": method, "radius_km": r,
                         "n_stations": len(g), "area_km2": round(area, 1), "footprint_km2": round(fp_area, 1),
                         "mean_rho": round(g.rho.mean(), 3) if len(g) else np.nan})
    out = pd.DataFrame(rows)
    out["rank"] = (out.groupby(["method", "variable", "radius_km"]).area_km2
                   .rank(ascending=False, method="min").astype(int))
    out = out.sort_values(["method", "variable", "radius_km", "rank"])
    path = os.path.join(GDIR, "influence_area.csv")
    out.to_csv(path, index=False)

    # wide view: rank and area at each radius
    wide = out.pivot_table(index=["method", "variable", "signed_gradient", "n_stations"],
                           columns="radius_km", values=["rank", "area_km2"])
    wide.columns = [f"{v}_{r}km" for v, r in wide.columns]
    rank_cols = [f"rank_{r}km" for r in RADII_KM]
    wide = wide.reset_index()
    wide["rank_min"] = wide[rank_cols].min(axis=1).astype(int)
    wide["rank_max"] = wide[rank_cols].max(axis=1).astype(int)
    for col in rank_cols:
        wide[col] = wide[col].astype(int)
    wide = wide.sort_values(["method", "variable", f"rank_{REF_RADIUS_KM}km"])
    wpath = os.path.join(GDIR, "influence_area_ranks.csv")
    wide.to_csv(wpath, index=False)

    # how stable is the ranking across radii (vs the 25 km ranking)?
    stab = []
    for (m, v), g in out.groupby(["method", "variable"]):
        p = g.pivot(index="signed_gradient", columns="radius_km", values="area_km2")
        for r in RADII_KM:  # Spearman = Pearson on ranks (scipy not installed)
            stab.append({"method": m, "variable": v, "radius_km": r,
                         f"spearman_vs_{REF_RADIUS_KM}km": round(p[r].rank().corr(p[REF_RADIUS_KM].rank()), 3)})
    stab = pd.DataFrame(stab)
    spath = os.path.join(GDIR, "influence_area_stability.csv")
    stab.to_csv(spath, index=False)

    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(stab.pivot_table(index=["method", "variable"], columns="radius_km",
                               values=f"spearman_vs_{REF_RADIUS_KM}km"))
        show = ["signed_gradient", "n_stations"] + rank_cols + ["rank_min", "rank_max",
                f"area_km2_10km", f"area_km2_{REF_RADIUS_KM}km", "area_km2_50km"]
        for (m, v), g in wide.groupby(["method", "variable"], sort=False):
            print(f"\n=== {m} / {v} ===")
            print(g[show].to_string(index=False))
    print(f"\nWrote {path}\nWrote {wpath}\nWrote {spath}")


if __name__ == "__main__":
    main()
