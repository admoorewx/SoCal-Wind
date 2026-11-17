#!/usr/bin/env python3
"""Influence footprints (rho >= 0.5 stations, 25 km circles, merged) and area vs mean rho.

Uses the "rho05" station set from gradient_influence_area.py: every station with n >= 50 and
rho >= 0.5 for that signed gradient and variable. Circles are not clipped to land.
With --weighted, uses "rho05w": each station weighted by w = (rho - 0.5) / 0.5, overlaps count at
the highest weight; the map shades each part of the footprint by the rho that governs it.

-> images/gradients/footprints/<PAIR>_<pos|neg>_footprint25km[_weighted].png
                                         one map per signed gradient, sustained and gust panels
-> images/gradients/influence_area[_weighted]_vs_rho_25km.png
                                         scatter, one dot per signed gradient

Usage:
  python plot_influence_25km.py [--radius 25] [--weighted] [--no-maps]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

import gradient_influence_area as gia
from plot_top10_scatter import GRID, INK, INK2, SERIES, SURFACE, place_labels

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.normpath(os.path.join(HERE, "..", "images", "gradients"))
RHO_MIN, MIN_N = 0.5, 50
EXTENT = [-125.0, -113.5, 31.8, 49.3]  # same domain as gradient_pipeline maps
VAR_TITLE = {"sustained": "Sustained wind", "gust": "Gust"}
SIGN_SYM = {"pos": "+", "neg": "-"}


def load_pairs():
    with open(os.path.join(HERE, "gradients.json"), encoding="utf-8") as f:
        return {k: (v["station1"], v["station2"]) for k, v in json.load(f).items()}


def draw_maps(c, pts, radius, area, weighted=False):
    st = pd.read_parquet(os.path.join(gia.GDIR, "..", "..", "stations.parquet"),
                         columns=["stid", "lat", "lon"]).drop_duplicates("stid").set_index("stid")
    cmap = LinearSegmentedColormap.from_list("rho", ["#9ec5f0", "#2a78d6", "#0b3b7a"])
    norm = Normalize(vmin=RHO_MIN, vmax=0.85)
    proj = ccrs.PlateCarree()
    out = os.path.join(IMG, "footprints")
    os.makedirs(out, exist_ok=True)
    for name, (s1, s2) in load_pairs().items():
        for sign, sym in SIGN_SYM.items():
            sg = name + sym
            fig, axes = plt.subplots(1, 2, figsize=(11.5, 9.4), subplot_kw={"projection": proj},
                                     facecolor=SURFACE)
            fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.1, wspace=0.05)
            for ax, var in zip(axes, ("sustained", "gust")):
                ax.set_extent(EXTENT, crs=proj)
                ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f4f3ef")
                ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#e6eef5")
                ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="#8a8984", linewidth=0.5)
                ax.coastlines("50m", color="#8a8984", linewidth=0.5)
                g = c[(c.signed_gradient == sg) & (c.variable == var) & (c.rho >= RHO_MIN)].sort_values("rho")
                if weighted:
                    g = g[g.rho > RHO_MIN]
                fp = gia.footprint(pts, g.stid, radius)
                if fp is not None and weighted:
                    pieces = gia.weighted_pieces(pts, g.stid, gia.excess_weight(g.rho, RHO_MIN), radius)
                    geo = gpd.GeoSeries([p for p, _ in pieces], crs=gia.CRS_EQUAL_AREA).to_crs("EPSG:4326")
                    rho_of_piece = [RHO_MIN + w * (1 - RHO_MIN) for _, w in pieces]
                    geo.plot(ax=ax, transform=proj, color=[cmap(norm(r)) for r in rho_of_piece], alpha=0.8,
                             edgecolor="none", zorder=2)
                    gpd.GeoSeries([fp], crs=gia.CRS_EQUAL_AREA).to_crs("EPSG:4326").plot(
                        ax=ax, transform=proj, facecolor="none", edgecolor=SERIES, linewidth=0.7, zorder=2.5)
                    ax.scatter(g.lon, g.lat, s=5, c=INK, linewidths=0, transform=proj, zorder=3)
                elif fp is not None:
                    gpd.GeoSeries([fp], crs=gia.CRS_EQUAL_AREA).to_crs("EPSG:4326").plot(
                        ax=ax, transform=proj, facecolor="#9ec5f0", alpha=0.75, edgecolor=SERIES,
                        linewidth=0.8, zorder=2)
                    sc = ax.scatter(g.lon, g.lat, c=g.rho, cmap=cmap, norm=norm, s=16,
                                    edgecolors="white", linewidths=0.4, transform=proj, zorder=3)
                for stid, marker in ((s1, "^"), (s2, "v")):
                    ax.scatter(st.at[stid, "lon"], st.at[stid, "lat"], marker=marker, s=90, c=INK,
                               edgecolors="white", linewidths=1.0, transform=proj, zorder=4)
                    ax.annotate(stid, (st.at[stid, "lon"], st.at[stid, "lat"]), xytext=(6, 4),
                                textcoords="offset points", fontsize=8.5, fontweight="bold", color=INK,
                                zorder=5)
                a = area[(area.signed_gradient == sg) & (area.variable == var)].iloc[0]
                rho_txt = f"mean ρ {a.mean_rho:.3f}" if a.n_stations else "no stations"
                area_txt = (f"{a.area_km2:,.0f} km² weighted of {a.footprint_km2:,.0f} km²" if weighted
                            else f"{a.area_km2:,.0f} km²")
                ax.set_title(f"{VAR_TITLE[var]}\n{area_txt} · {a.n_stations} stations · {rho_txt}",
                             loc="left", fontsize=10.5, color=INK)
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes((0.25, 0.055, 0.5, 0.018)),
                              orientation="horizontal", extend="max")
            cb.set_label("Spearman ρ, |gradient| vs wind" + ("   (weight = (ρ − 0.5) / 0.5: 0 at 0.5, 0.5 at 0.75)"
                                                               if weighted else ""), color=INK2)
            cb.outline.set_visible(False)
            direction = f"{s1} SLP > {s2} SLP" if sign == "pos" else f"{s1} SLP < {s2} SLP"
            fig.suptitle(f"{sg} influence footprint ({direction}; {s1} ▲, {s2} ▼)", x=0.02, y=0.975,
                         ha="left", fontsize=13, color=INK)
            sub = (f"{radius:g} km circles around every station with ρ > {RHO_MIN} (n ≥ {MIN_N}), shaded by the "
                   f"strongest station covering each point; not clipped to land" if weighted else
                   f"{radius:g} km circles around every station with ρ ≥ {RHO_MIN} "
                   f"(n ≥ {MIN_N}), overlaps merged; not clipped to land")
            fig.text(0.02, 0.945, sub, fontsize=9.5, color=INK2)
            path = os.path.join(out, f"{name}_{sign}_footprint{radius:g}km{'_weighted' if weighted else ''}.png")
            fig.savefig(path, dpi=120, facecolor=SURFACE)
            plt.close(fig)
            print(f"Wrote {path}")


def draw_scatter(area, radius, weighted=False):
    d = area[area.n_stations > 0]
    dropped = area[area.n_stations == 0]
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, var in zip(axes, ("sustained", "gust")):
        v = d[d.variable == var]
        ax.set_facecolor(SURFACE)
        ax.scatter(v.mean_rho, v.area_km2 / 1000, s=48, color=SERIES, edgecolor=SURFACE, linewidth=1.5,
                   zorder=3)
        ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)
        none = ", ".join(dropped[dropped.variable == var].signed_gradient)
        ax.set_title(VAR_TITLE[var], loc="left", fontsize=12, color=INK, pad=10)
        ax.set_xlabel(f"Mean Spearman ρ of the footprint stations (ρ {'>' if weighted else '≥'} {RHO_MIN})")
        if none:
            ax.text(0.0, -0.17, f"Not shown (no stations with ρ ≥ {RHO_MIN}): {none}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=8, color=INK2)
    axes[0].set_ylim(0, d.area_km2.max() / 1000 * 1.08)  # shared y; set after both panels so neither clips
    if weighted:
        axes[0].set_ylabel(f"Weighted footprint area at {radius:g} km radius (1,000 km²)")
        fig.suptitle("Weighted Influence Area vs. Avg. Correlation", x=0.01, ha="left", fontsize=13, color=INK)
        fig.text(0.01, 0.925, f"{radius:g} km footprint of all stations with ρ > {RHO_MIN}, weighted by "
                 f"(ρ − {RHO_MIN}) / {1 - RHO_MIN:g}, vs. their average correlation", fontsize=10, color=INK2)
    else:
        axes[0].set_ylabel(f"Footprint area at {radius:g} km radius (1,000 km²)")
        fig.suptitle("Influence Area vs. Avg. Correlation", x=0.01, ha="left", fontsize=13, color=INK)
        fig.text(0.01, 0.925, f"Merged {radius:g} km footprint of all stations with ρ ≥ {RHO_MIN} vs. their "
                 f"average correlation", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.91))
    for ax, var in zip(axes, ("sustained", "gust")):
        v = d[d.variable == var]
        lab = pd.concat([v.nlargest(8, "area_km2"), v.nlargest(4, "mean_rho")]).drop_duplicates("signed_gradient")
        lab = lab.sort_values("area_km2", ascending=False)
        place_labels(ax, list(zip(lab.mean_rho, lab.area_km2 / 1000)), lab.signed_gradient)
    path = os.path.join(IMG, f"influence_area{'_weighted' if weighted else ''}_vs_rho_{radius:g}km.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=25)
    ap.add_argument("--no-maps", action="store_true")
    ap.add_argument("--weighted", action="store_true", help="use the rho05w (excess-over-0.5) weighting")
    args = ap.parse_args()
    method = "rho05w" if args.weighted else "rho05"
    area = pd.read_csv(os.path.join(gia.GDIR, "influence_area.csv"))
    area = area[(area.method == method) & (area.radius_km == args.radius)]
    if area.empty:
        raise SystemExit(f"No {method} rows at {args.radius} km in influence_area.csv; run gradient_influence_area.py")
    if not args.no_maps:
        c, pts = gia.load_correlations(MIN_N)
        draw_maps(c, pts, args.radius, area, args.weighted)
    draw_scatter(area, args.radius, args.weighted)


if __name__ == "__main__":
    main()
