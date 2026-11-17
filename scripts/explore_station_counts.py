"""Plot the number of reporting stations per month for each state.

A station counts toward a month if it has at least one hourly row in that
month's obs_hourly/YYYY/MM/<STATE>.parquet file.
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.normpath(os.path.join(HERE, "..", "data", "v2"))
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "images", "station_counts_by_state.png"))
STATES = ["CA", "OR", "WA", "NV"]
COLORS = {"CA": "#2a6fdb", "OR": "#1f9e6e", "WA": "#d9822b", "NV": "#b8439b"}


def count_stations(data_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(data_dir, "obs_hourly", "*", "*", "*.parquet"))):
        parts = path.split(os.sep)
        year, month, state = parts[-3], parts[-2], parts[-1].split(".")[0]
        stids = pq.read_table(path, columns=["stid"]).column("stid")
        rows.append({"month": pd.Timestamp(f"{year}-{month}-01"), "state": state,
                     "n_stations": len(stids.unique())})
    return pd.DataFrame(rows).pivot(index="month", columns="state", values="n_stations").sort_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    counts = count_stations(args.data)
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    counts.to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for st in STATES:
        if st in counts:
            ax.plot(counts.index, counts[st], color=COLORS[st], lw=2, label=st)
    ax.set_title("Reporting stations per month by state")
    ax.set_ylabel("Stations with ≥1 hourly observation")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}\nWrote {csv_path}")
    print(counts.iloc[[0, len(counts) // 2, -1]])
    print("\nMin / max per state:\n", counts.agg(["min", "max"]))


if __name__ == "__main__":
    main()
