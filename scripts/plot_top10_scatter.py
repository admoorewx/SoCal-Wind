#!/usr/bin/env python3
"""Scatter of top-10 mean rho vs median separation of those 10 stations, one panel per variable.

Reads derived/gradients/top10_summary.csv (from explore_top10.py).
-> images/gradients/top10_rho_vs_separation.png

Upper right = strong AND widespread; upper left = strong but local.
Labels: the strongest LABEL_N gradients per panel, plus any above LOCAL_RHO with median
separation below LOCAL_KM (strong but clustered).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, NullLocator

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.normpath(os.path.join(HERE, "..", "data", "v2", "derived", "gradients", "top10_summary.csv"))
OUT = os.path.normpath(os.path.join(HERE, "..", "images", "gradients", "top10_rho_vs_separation.png"))

LABEL_N = 8
LOCAL_RHO, LOCAL_KM = 0.55, 60

SURFACE, INK, INK2, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#2a78d6"


def place_labels(ax, pts, names):
    """Greedy label placement with leader lines: for each point (strongest first) try offsets on rings
    of growing radius and keep the first label box that stays inside the axes and overlaps no dot
    or earlier label."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axb = ax.get_window_extent(renderer)
    boxes = []
    for line in ax.collections:  # every dot is an obstacle, labelled or not
        for x, y in ax.transData.transform(line.get_offsets()):
            boxes.append((x - 10, y - 10, x + 10, y + 10))
    angles = np.radians([30, -30, 150, -150, 90, -90, 0, 180, 60, -60, 120, -120])
    for (x, y), name in zip(pts, names):
        done = False
        for r in (16, 26, 38, 52, 68):
            for a in angles:
                dx, dy = r * np.cos(a), r * np.sin(a)
                ha = "left" if dx > 1 else "right" if dx < -1 else "center"
                va = "bottom" if dy > 1 else "top" if dy < -1 else "center"
                t = ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                                fontsize=8, color=INK2,
                                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=1, shrinkB=4)
                                )
                t.update_positions(renderer)  # resolve the offset-point position before measuring
                bb = Text.get_window_extent(t, renderer).expanded(1.04, 1.1)  # text only, not the leader line
                b = (bb.x0, bb.y0, bb.x1, bb.y1)
                inside = b[0] >= axb.x0 and b[2] <= axb.x1 and b[1] >= axb.y0 and b[3] <= axb.y1
                if inside and not any(b[0] < o[2] and b[2] > o[0] and b[1] < o[3] and b[3] > o[1]
                                      for o in boxes):
                    boxes.append(b)
                    done = True
                    break
                t.remove()
            if done:
                break


def main():
    s = pd.read_csv(CSV)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True, facecolor=SURFACE)
    xmin = s.top10_sep_median_km.min() * 0.8
    xmax = s.top10_sep_median_km.max() * 1.25
    for ax, var in zip(axes, ("sustained", "gust")):
        d = s[s.variable == var]
        ax.set_facecolor(SURFACE)
        ax.scatter(d.top10_sep_median_km, d.top10_mean_rho, s=48, color=SERIES,
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(xmin, xmax)
        ticks = [20, 50, 100, 200, 500]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([str(t) for t in ticks])
        ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(f"{var.capitalize()} wind", loc="left", fontsize=12, color=INK, pad=10)
        ax.set_xlabel("Median separation between the top 10 stations (km, log scale)")

    axes[0].set_ylabel("Mean Spearman ρ of the top 10 stations")
    fig.suptitle("Avg. Correlation vs. Separation", x=0.01, ha="left", fontsize=13, color=INK)
    fig.text(0.01, 0.925, "Average correlation of top 10 stations vs. median station separation",
             fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    # labels are placed after the layout is final, so their pixel positions hold
    for ax, var in zip(axes, ("sustained", "gust")):
        d = s[s.variable == var]
        top = d.nlargest(LABEL_N, "top10_mean_rho")
        local = d[(d.top10_mean_rho >= LOCAL_RHO) & (d.top10_sep_median_km < LOCAL_KM)]
        lab = pd.concat([top, local]).drop_duplicates("signed_gradient").sort_values("top10_mean_rho",
                                                                                     ascending=False)
        place_labels(ax, list(zip(lab.top10_sep_median_km, lab.top10_mean_rho)), lab.signed_gradient)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
