#!/usr/bin/env python3
"""Convert the saved raw Synoptic responses into full-resolution parquet tables.

Reads   data/v2/raw/YYYY-MM/<STATE>_batch*.json.gz
Writes  data/v2/obs_raw/YYYY/MM/<STATE>.parquet   one row per station per observation time

Uses the same rules as synoptic_bulk_download.py: primary variable set preferred
(SET_PRIORITY), Synoptic-QC-flagged values removed, implausible pressures removed, odd value
types cleaned with clean_value(). Rows where every variable is missing are dropped, and
duplicate (stid, time) rows are collapsed.

Columns: stid, time_utc, wind_speed_mps, wind_direction_deg, wind_gust_mps,
         pressure_hpa, sea_level_pressure_hpa

Usage:
  python build_obs_raw.py                     # all months, skip ones already written
  python build_obs_raw.py --months 2024-01    # specific months
  python build_obs_raw.py --workers 6 --overwrite
"""
import argparse
import glob
import gzip
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synoptic_bulk_download as sbd  # noqa: E402  (reuse QC constants and value cleaning)

DEFAULT_DATA = os.path.normpath(os.path.join(HERE, "..", "data", "v2"))

# (API base variable, output column, scale applied to the metric value)
VARIABLES = [
    ("wind_speed", "wind_speed_mps", 1.0),
    ("wind_direction", "wind_direction_deg", 1.0),
    ("wind_gust", "wind_gust_mps", 1.0),
    ("pressure", "pressure_hpa", 0.01),
    ("sea_level_pressure", "sea_level_pressure_hpa", 0.01),
]
SCHEMA = pa.schema(
    [("stid", pa.string()), ("time_utc", pa.timestamp("s", tz="UTC"))]
    + [(col, pa.float32()) for _, col, _ in VARIABLES]
)


def to_array(values, base):
    """Convert one API value list to float64 with NaN for missing/unusable values."""
    out = np.empty(len(values), dtype=np.float64)
    for i, x in enumerate(values):
        t = type(x)
        if t is float or t is int:
            out[i] = x
        else:
            v = sbd.clean_value(x, base)
            out[i] = np.nan if v is None else v
    return out


def station_arrays(st):
    """Return (times[datetime64[s]], {column: float32 array}) for one station object."""
    obs = st.get("OBSERVATIONS") or {}
    times = obs.get("date_time") or []
    if not times:
        return None
    qc_all = st.get("QC") or {}
    chosen = {}
    for key in obs:
        base, sep, set_name = key.rpartition("_set_")
        if not sep or base not in sbd.API_VARS:
            continue
        rank = sbd.SET_PRIORITY.get(set_name, 9)
        if base not in chosen or rank < chosen[base][0]:
            chosen[base] = (rank, key)

    t = np.array([s[:-1] for s in times], dtype="datetime64[s]")  # strip trailing "Z"
    cols = {}
    for base, col, scale in VARIABLES:
        arr = np.full(len(times), np.nan)
        if base in chosen:
            key = chosen[base][1]
            values = obs[key]
            if len(values) == len(times):
                arr = to_array(values, base)
                flags = qc_all.get(key)
                if flags is not None and len(flags) == len(times):
                    arr[np.array([f is not None for f in flags])] = np.nan
                if base in sbd.PLAUSIBLE_PA:
                    lo, hi = sbd.PLAUSIBLE_PA[base]
                    with np.errstate(invalid="ignore"):
                        arr[(arr < lo) | (arr > hi)] = np.nan
        cols[col] = (arr * scale).astype(np.float32)
    return t, cols


def process(job):
    month, state, data_dir, overwrite = job
    out_path = os.path.join(data_dir, "obs_raw", month[:4], month[5:], f"{state}.parquet")
    if os.path.exists(out_path) and not overwrite:
        return month, state, "skip", 0
    files = sorted(glob.glob(os.path.join(data_dir, "raw", month, f"{state}_batch*.json.gz")))
    stids, times, cols = [], [], {c: [] for _, c, _ in VARIABLES}
    for f in files:
        with gzip.open(f, "rb") as fh:
            d = json.load(fh)
        for st in d.get("STATION") or []:
            res = station_arrays(st)
            if res is None:
                continue
            t, c = res
            stids.append(np.full(len(t), st["STID"], dtype=object))
            times.append(t)
            for k in cols:
                cols[k].append(c[k])
    if not times:
        return month, state, "empty", 0
    stid = np.concatenate(stids)
    t = np.concatenate(times)
    data = {k: np.concatenate(v) for k, v in cols.items()}
    keep = ~np.all(np.isnan(np.vstack(list(data.values()))), axis=0)
    # sort by (stid, time) and drop duplicate keys (a station can appear in a retried batch)
    order = np.lexsort((t, stid))
    order = order[keep[order]]
    stid, t = stid[order], t[order]
    data = {k: v[order] for k, v in data.items()}
    dup = np.zeros(len(t), dtype=bool)
    dup[1:] = (stid[1:] == stid[:-1]) & (t[1:] == t[:-1])
    stid, t = stid[~dup], t[~dup]
    data = {k: v[~dup] for k, v in data.items()}

    table = pa.Table.from_arrays(
        [pa.array(stid, pa.string()), pa.array(t, pa.timestamp("s", tz="UTC"))]
        + [pa.array(data[c], pa.float32()) for _, c, _ in VARIABLES],
        schema=SCHEMA,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    pq.write_table(table, tmp, compression="zstd", row_group_size=1_000_000)
    os.replace(tmp, out_path)
    return month, state, "ok", table.num_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--months", nargs="*", help="YYYY-MM list (default: every month in raw/)")
    ap.add_argument("--states", nargs="*", default=sbd.STATES)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    months = args.months or sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.data, "raw", "*")))
    jobs = [(m, s, args.data, args.overwrite) for m in months for s in args.states]
    t0 = time.time()
    total = 0
    with Pool(args.workers) as pool:
        for i, (m, s, status, n) in enumerate(pool.imap_unordered(process, jobs), 1):
            total += n
            print(f"[{i}/{len(jobs)}] {m} {s}: {status} {n:,} rows  ({time.time() - t0:.0f}s)", flush=True)
    print(f"Done: {total:,} rows in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
