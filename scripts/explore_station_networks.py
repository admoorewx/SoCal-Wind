"""Break monthly reporting-station counts down by network, one panel per state.

A station counts toward a month if it has at least one hourly row in that
month's obs_hourly/YYYY/MM/<STATE>.parquet file. Network comes from
stations.parquet (network_shortname). The three Western Weather Group networks
(WWG-SONOMA, WWG-LLWC, WWG-MISC) are combined as "WWG".

Outputs (next to --out):
  <out>.png              stacked-area small multiples
  <out>_long.csv         month, state, network, n_stations (all networks, uncombined)
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.normpath(os.path.join(HERE, "..", "data", "v2"))
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "images", "station_counts_by_network.png"))

# Categorical slots (fixed order, adjacent pairs CVD-validated).
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
OTHER = "#b5b3ad"

# Named networks per state, bottom of stack first. RAWS and ASOS/AWOS take slots 1-2
# everywhere; AGRIMET is third in both OR and WA so it gets the same color. Everything
# else folds into "Other".
PANELS = {
    "CA": ["RAWS", "ASOS/AWOS", "CIMIS", "SGXWFO", "CW3E", "WWG"],
    "OR": ["RAWS", "ASOS/AWOS", "AGRIMET", "PGN", "USBR HYDRO"],
    "WA": ["RAWS", "ASOS/AWOS", "AGRIMET", "HMMN", "PSE"],
    "NV": ["RAWS", "ASOS/AWOS", "DRI", "CEMP", "LAS VEGAS", "ARL SORD", "NV-ENERGY"],
}


def station_months(data_dir):
    meta = (pd.read_parquet(os.path.join(data_dir, "stations.parquet"),
                            columns=["stid", "network_shortname"])
            .drop_duplicates("stid"))
    frames = []
    for path in sorted(glob.glob(os.path.join(data_dir, "obs_hourly", "*", "*", "*.parquet"))):
        parts = path.split(os.sep)
        stids = pq.read_table(path, columns=["stid"]).column("stid").unique().to_pylist()
        frames.append(pd.DataFrame({"month": pd.Timestamp(f"{parts[-3]}-{parts[-2]}-01"),
                                    "state": parts[-1].split(".")[0], "stid": stids}))
    df = pd.concat(frames, ignore_index=True).merge(meta, on="stid", how="left")
    df["network"] = df["network_shortname"].fillna("UNKNOWN")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    df = station_months(args.data)
    base = os.path.splitext(args.out)[0]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    long = df.groupby(["month", "state", "network"]).size().rename("n_stations").reset_index()
    long.to_csv(base + "_long.csv", index=False)

    df["group"] = df["network"].where(~df["network"].str.startswith("WWG"), "WWG")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for ax, (state, named) in zip(axes.flat, PANELS.items()):
        sub = df[df["state"] == state].copy()
        sub["group"] = sub["group"].where(sub["group"].isin(named), "Other")
        wide = (sub.groupby(["month", "group"]).size().unstack(fill_value=0)
                .reindex(columns=named + ["Other"], fill_value=0).sort_index())
        colors = SLOTS[:len(named)] + [OTHER]
        ax.stackplot(wide.index, wide.T.values, labels=wide.columns, colors=colors,
                     edgecolor="white", linewidth=0.6)
        ax.set_title(f"{state} — {int(wide.iloc[-1].sum())} stations in {wide.index[-1]:%b %Y}",
                     loc="left", fontsize=12)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], frameon=False, fontsize=9,
                  loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for ax in axes[:, 0]:
        ax.set_ylabel("Reporting stations")
    fig.suptitle("Reporting stations per month by network (y-axes differ by state)", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}\nWrote {base}_long.csv")


if __name__ == "__main__":
    main()
