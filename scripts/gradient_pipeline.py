#!/usr/bin/env python3
"""Find the stations whose winds track each operational pressure gradient.

Steps (run in order, or `all`):

  timeseries  For each pair in gradients.json, gradient = SLP(station1) - SLP(station2) in hPa (mb).
              Times are station1's measured (set_1, hourly METAR) sea-level-pressure times;
              station2's reading must be within +/-MATCH_MINUTES of it (nearest is used).
              -> derived/gradients/gradient_timeseries.parquet

  match       For every station (WWG networks, portable stations and wind-QC-excluded stations
              removed; see portable_stations) and every
              station1 time grid, the max sustained wind and the max gust from observations
              within +/-MATCH_MINUTES of each gradient time. Sustained and gust are matched
              independently. Wind QC (wind_qc.py, docs/wind_qc_method.md) is applied first:
              readings flagged by a masked rule are dropped before the max is taken.
              Also stores the wind direction reported with the window's max sustained speed
              (wdir_at_sustained_max_deg; NaN if outside 0-360 or the speed was QC-masked).
              -> derived/gradients/matched/<STATION1>/<YYYY-MM>.parquet

  correlate   Each pair is split into two signed gradients: "+" (gradient > 0) and "-"
              (gradient < 0). The predictor is the gradient magnitude |gradient|. For each
              signed gradient, station and variable (sustained, gust), Spearman rho between
              magnitude and wind, over hours where both exist. A station is retained when
              n >= MIN_N and rho >= MIN_RHO. Sustained and gust are evaluated separately.
              -> derived/gradients/correlations.parquet   (every station with n >= 2)
              -> derived/gradients/retained_summary.csv
              With --before YYYY-MM-DD only gradient times before that date are used and the outputs
              get a suffix (correlations_before2026-01-01.parquet, ...), e.g. to pick model stations
              without seeing a held-out validation period.

  maps        One figure per signed gradient: sustained and gust panels, retained stations
              coloured by rho. -> images/gradients/<PAIR>_<pos|neg>.png

Usage:
  python gradient_pipeline.py all
  python gradient_pipeline.py correlate --min-rho 0.5 --min-n 50
"""
import argparse
import glob
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

import wind_qc

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.normpath(os.path.join(HERE, "..", "data", "v2"))
DEFAULT_IMAGES = os.path.normpath(os.path.join(HERE, "..", "images", "gradients"))
DEFAULT_GRADIENTS = os.path.join(HERE, "gradients.json")

MATCH_MINUTES = 30
MIN_N = 50
MIN_RHO = 0.5
EXCLUDE_NETWORK_PREFIXES = ("WWG",)  # Western Weather Group: intermittent, dropped from all analyses
VARS = {"sustained": "wind_speed_mps", "gust": "wind_gust_mps"}
QC_COL = {"sustained": "qc_spd", "gust": "qc_gust"}
DIR_COL = "wind_direction_deg"  # matched as the direction reported with the window's max sustained speed
SIGNS = {"pos": "+", "neg": "-"}


def out_dir(data):
    return os.path.join(data, "derived", "gradients")


def load_pairs(path):
    with open(path, encoding="utf-8") as f:
        g = json.load(f)
    return {name: (v["station1"], v["station2"]) for name, v in g.items()}


def obs_raw_files(data):
    return sorted(glob.glob(os.path.join(data, "obs_raw", "*", "*", "*.parquet")))


def load_stations(data):
    s = pd.read_parquet(os.path.join(data, "stations.parquet")).drop_duplicates("stid")
    return s


PORTABLE_RAWS_PATTERN = r"(?:\b|_)PORT(?:_|\s*#|\d|\b)"  # "MEDFORD PORT #1", "CAYIA_PORT1", not AIRPORT


def portable_stations(data):
    """Station IDs that are portable units: one stored location can't describe where their data came from.

    A station ID is portable if ANY of its stations.parquet rows has "PORTABLE" in the name (any network),
    or is a RAWS station whose name has a "PORT" token (RAWS abbreviation for portable).
    """
    s = pd.read_parquet(os.path.join(data, "stations.parquet"), columns=["stid", "name", "network_shortname"])
    n = s.name.fillna("").str.upper()
    m = n.str.contains("PORTABLE") | ((s.network_shortname == "RAWS") & n.str.contains(PORTABLE_RAWS_PATTERN))
    return set(s.loc[m, "stid"])


# ----------------------------------------------------------------------------
# Step 1: gradient timeseries
# ----------------------------------------------------------------------------

SPIKE_HPA = 4.0        # an SLP reading this far from BOTH neighbours (same direction) is dropped
SPIKE_MAX_GAP_H = 3.0  # ... when both neighbours are within this many hours


def clean_slp(d):
    """One station's SLP series -> one reading per hour, isolated spikes removed.

    Hourly thinning: the station's modal reporting minute m0 defines nominal times HH:m0;
    each reading is assigned to the nearest nominal time and the closest one is kept.
    (Some stations, e.g. KOTH in 2015-2017, report several times an hour.)
    """
    d = d.sort_values("time_utc")
    m0 = int(d.time_utc.dt.minute.mode().iloc[0])
    shift = pd.Timedelta(minutes=m0)
    nominal = (d.time_utc - shift).dt.round("h") + shift
    d = (d.assign(_dist=(d.time_utc - nominal).abs(), _nom=nominal)
         .sort_values(["_nom", "_dist"]).drop_duplicates("_nom").sort_values("time_utc"))
    v = d.sea_level_pressure_hpa.astype("float64").to_numpy()
    t = d.time_utc
    gap_prev = t.diff().dt.total_seconds().to_numpy() / 3600
    gap_next = -t.diff(-1).dt.total_seconds().to_numpy() / 3600
    a = v - np.r_[np.nan, v[:-1]]
    b = v - np.r_[v[1:], np.nan]
    spike = ((np.abs(a) > SPIKE_HPA) & (np.abs(b) > SPIKE_HPA) & (np.sign(a) == np.sign(b))
             & (gap_prev <= SPIKE_MAX_GAP_H) & (gap_next <= SPIKE_MAX_GAP_H))
    stid = d.stid.iloc[0]
    print(f"  {stid}: modal minute :{m0:02d}, {len(d):,} hourly readings, {int(spike.sum())} spikes removed")
    return d.loc[~spike, ["stid", "time_utc", "sea_level_pressure_hpa"]]


def step_timeseries(args):
    pairs = load_pairs(args.gradients)
    stids = sorted({s for p in pairs.values() for s in p})
    t0 = time.time()
    slp = (ds.dataset(obs_raw_files(args.data), format="parquet")
           .to_table(columns=["stid", "time_utc", "sea_level_pressure_hpa"],
                     filter=ds.field("stid").isin(stids))
           .to_pandas())
    # NaN is "valid" to Arrow, so drop missing pressures here, after conversion
    slp = slp.dropna(subset=["sea_level_pressure_hpa"])
    slp = slp.drop_duplicates(["stid", "time_utc"])
    print(f"Loaded {len(slp):,} SLP obs for {slp.stid.nunique()} of {len(stids)} pair stations "
          f"({time.time() - t0:.0f}s)")
    slp = pd.concat([clean_slp(d) for _, d in slp.groupby("stid")]).sort_values("time_utc")
    missing = sorted(set(stids) - set(slp.stid))
    if missing:
        print(f"WARNING: no SLP data for {missing}")

    tol = pd.Timedelta(minutes=MATCH_MINUTES)
    frames = []
    for name, (s1, s2) in pairs.items():
        a = slp[slp.stid == s1][["time_utc", "sea_level_pressure_hpa"]].rename(
            columns={"sea_level_pressure_hpa": "slp1_hpa"})
        b = slp[slp.stid == s2][["time_utc", "sea_level_pressure_hpa"]].rename(
            columns={"sea_level_pressure_hpa": "slp2_hpa"})
        b["time2_utc"] = b["time_utc"]
        m = pd.merge_asof(a, b, on="time_utc", direction="nearest", tolerance=tol)
        m = m.dropna(subset=["slp2_hpa"])
        m["station2_offset_min"] = ((m["time2_utc"] - m["time_utc"]).dt.total_seconds() / 60).astype("int16")
        m["gradient_hpa"] = (m["slp1_hpa"].astype("float64") - m["slp2_hpa"].astype("float64")).round(2)
        m.insert(0, "gradient", name)
        m.insert(1, "station1", s1)
        m.insert(2, "station2", s2)
        frames.append(m.drop(columns="time2_utc"))
        print(f"  {name:8s} {len(m):7,} times  "
              f"(+ {int((m.gradient_hpa > 0).sum()):,} / - {int((m.gradient_hpa < 0).sum()):,} / 0 {int((m.gradient_hpa == 0).sum()):,})  "
              f"range {m.gradient_hpa.min():.1f} to {m.gradient_hpa.max():.1f} hPa")
    ts = pd.concat(frames, ignore_index=True)
    os.makedirs(out_dir(args.data), exist_ok=True)
    path = os.path.join(out_dir(args.data), "gradient_timeseries.parquet")
    ts.to_parquet(path, index=False)
    print(f"Wrote {path}")


# ----------------------------------------------------------------------------
# Step 2: match station winds to gradient times
# ----------------------------------------------------------------------------

def epoch_s(t):
    """Seconds since 1970 (int64 numpy) for a tz-aware datetime Series or DatetimeIndex."""
    return np.asarray((t - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1), dtype=np.int64)


def window_max(keys, vals, qlo, qhi, aux=None):
    """Max of vals over each sorted-key window [qlo, qhi]; NaN and count 0 where empty.

    With aux (same length as vals), also returns aux at the window's max (first max wins).
    """
    lo = np.searchsorted(keys, qlo, "left")
    hi = np.searchsorted(keys, qhi, "right")
    cnt = hi - lo
    out = np.full(len(qlo), np.nan, dtype=np.float32)
    at = np.full(len(qlo), np.nan, dtype=np.float32) if aux is not None else None
    for k in range(int(cnt.max()) if len(cnt) else 0):
        m = cnt > k
        v = vals[lo[m] + k]
        if aux is not None:
            better = ~np.isnan(v) & (np.isnan(out[m]) | (v > out[m]))
            idx = np.flatnonzero(m)[better]
            at[idx] = aux[lo[idx] + k]
        out[m] = np.fmax(out[m], v)
    return (out, cnt, at) if aux is not None else (out, cnt)


def match_month(job):
    month_start, files_by_month, grids, exclude, dest, overwrite = job
    tag = f"{month_start:%Y-%m}"
    paths = {s1: os.path.join(dest, s1, f"{tag}.parquet") for s1 in grids}
    if not overwrite and all(os.path.exists(p) for p in paths.values()):
        return tag, "skip", 0
    month_end = month_start + pd.offsets.MonthBegin(1)
    pad = pd.Timedelta(minutes=MATCH_MINUTES)
    want = [month_start - pd.offsets.MonthBegin(1), month_start, month_end]
    files = [f for m in want for f in files_by_month.get(m, [])]
    flt = ((ds.field("time_utc") >= pa.scalar(month_start - pad, pa.timestamp("s", "UTC")))
           & (ds.field("time_utc") <= pa.scalar(month_end + pad, pa.timestamp("s", "UTC"))))
    obs = (ds.dataset(files, format="parquet")
           .to_table(columns=["stid", "time_utc", *VARS.values(), DIR_COL], filter=flt).to_pandas())
    d = obs[DIR_COL]
    obs.loc[(d < 0) | (d > 360), DIR_COL] = np.nan
    obs = obs[~obs.stid.isin(exclude)].drop_duplicates(["stid", "time_utc"])
    obs = apply_wind_qc(obs, [m for m in want if m in files_by_month])
    codes, uniques = pd.factorize(obs.stid, sort=True)
    secs = epoch_s(obs.time_utc)
    big = np.int64(10**10)
    base_key = codes.astype(np.int64) * big + secs

    series = {}
    for var, col in VARS.items():
        v = obs[col].to_numpy(np.float32)
        ok = ~np.isnan(v)
        k, vv = base_key[ok], v[ok]
        order = np.argsort(k, kind="stable")
        # direction rides with the sustained speed: QC-masked speeds (NaN) drop their direction too
        dd = obs[DIR_COL].to_numpy(np.float32)[ok][order] if var == "sustained" else None
        series[var] = (k[order], vv[order], np.unique(codes[ok]), dd)

    total = 0
    pad_s = MATCH_MINUTES * 60
    for s1, times in grids.items():
        g = times[(times >= month_start) & (times < month_end)]
        if len(g) == 0:
            continue
        gs = epoch_s(g)
        stations = np.union1d(series["sustained"][2], series["gust"][2])
        qs = np.repeat(stations.astype(np.int64), len(gs)) * big + np.tile(gs, len(stations))
        res = {}
        for var, (k, vv, _, dd) in series.items():
            if dd is None:
                res[var], res[f"n_{var}"] = window_max(k, vv, qs - pad_s, qs + pad_s)
            else:
                res[var], res[f"n_{var}"], res["wdir"] = window_max(k, vv, qs - pad_s, qs + pad_s, aux=dd)
        keep = (res["n_sustained"] > 0) | (res["n_gust"] > 0)
        st_idx = np.repeat(stations, len(gs))[keep]
        table = pa.table({
            "stid": pa.array(uniques[st_idx].astype(str)).dictionary_encode(),
            "time_utc": pa.array(np.tile(gs, len(stations))[keep].astype("datetime64[s]"),
                                 pa.timestamp("s", "UTC")),
            "sustained_max_mps": pa.array(res["sustained"][keep]),
            "gust_max_mps": pa.array(res["gust"][keep]),
            "wdir_at_sustained_max_deg": pa.array(res["wdir"][keep]),
            "n_sustained_obs": pa.array(np.minimum(res["n_sustained"][keep], 255).astype(np.uint8)),
            "n_gust_obs": pa.array(np.minimum(res["n_gust"][keep], 255).astype(np.uint8)),
        })
        os.makedirs(os.path.dirname(paths[s1]), exist_ok=True)
        tmp = paths[s1] + ".tmp"
        pq.write_table(table, tmp, compression="zstd", row_group_size=200_000)
        os.replace(tmp, paths[s1])
        total += table.num_rows
    return tag, "ok", total


def apply_wind_qc(obs, months):
    """Set sustained/gust readings flagged by a masked wind-QC rule to NaN."""
    flags = []
    for m in months:
        fp = os.path.join(wind_qc.FLAGS, f"{m:%Y}", f"{m:%m}.parquet")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"wind QC flags missing for {m:%Y-%m}: run wind_qc.py apply")
        flags.append(pd.read_parquet(fp))
    fl = pd.concat(flags).drop_duplicates(["stid", "time_utc"])
    fl["time_utc"] = fl.time_utc.astype("datetime64[ms, UTC]")
    obs["time_utc"] = obs.time_utc.astype("datetime64[ms, UTC]")
    obs = obs.merge(fl, on=["stid", "time_utc"], how="left")
    bits = wind_qc.mask_bits()
    for var, col in VARS.items():
        q = obs[QC_COL[var]].fillna(0).astype(np.int64)
        obs.loc[(q & bits) > 0, col] = np.nan
    return obs.drop(columns=list(QC_COL.values()))


def qc_metadata():
    return {"wind_qc_config_version": wind_qc.CONFIG["version"], "wind_qc_mask_rules": wind_qc.CONFIG["mask_rules"],
            "wind_qc_excluded_stations": sorted(wind_qc.excluded_stations()),
            "wind_qc_manual_exclusions": [{k: e[k] for k in ("stid", "start", "end")}
                                          for e in wind_qc.CONFIG["manual_exclusions"]]}


def step_match(args):
    pairs = load_pairs(args.gradients)
    ts = pd.read_parquet(os.path.join(out_dir(args.data), "gradient_timeseries.parquet"))
    grids = {s1: pd.DatetimeIndex(sorted(ts.loc[ts.station1 == s1, "time_utc"].unique()))
             for s1 in sorted({p[0] for p in pairs.values()})}
    st = load_stations(args.data)
    exclude = set(st.loc[st.network_shortname.fillna("").str.startswith(EXCLUDE_NETWORK_PREFIXES), "stid"])
    qc_ex = wind_qc.excluded_stations()
    if not qc_ex:
        raise SystemExit("wind QC station_exclusions.csv missing: run wind_qc.py evaluate")
    print(f"Excluding {len(qc_ex)} wind-QC stations: {', '.join(sorted(qc_ex))}; masking rules {wind_qc.CONFIG['mask_rules']}")
    exclude |= qc_ex
    portable = portable_stations(args.data)
    print(f"Excluding {len(portable)} portable stations")
    exclude |= portable
    with open(os.path.join(out_dir(args.data), "match_qc.json"), "w") as f:
        json.dump({**qc_metadata(), "excluded_portable_stations": sorted(portable),
                   "created_utc": pd.Timestamp.now(tz="UTC").isoformat()}, f, indent=2)
    print(f"Excluding {len(exclude)} stations in total (networks {EXCLUDE_NETWORK_PREFIXES} + portable + wind QC); "
          f"{len(grids)} station1 time grids: {', '.join(grids)}")
    files_by_month = {}
    for f in obs_raw_files(args.data):
        y, m = f.split(os.sep)[-3:-1]
        files_by_month.setdefault(pd.Timestamp(f"{y}-{m}-01", tz="UTC"), []).append(f)
    dest = os.path.join(out_dir(args.data), "matched")
    jobs = [(m, files_by_month, grids, exclude, dest, args.overwrite) for m in sorted(files_by_month)]
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (tag, status, n) in enumerate(pool.imap_unordered(match_month, jobs), 1):
            print(f"[{i}/{len(jobs)}] {tag}: {status} {n:,} rows ({time.time() - t0:.0f}s)", flush=True)


# ----------------------------------------------------------------------------
# Step 3: Spearman correlations
# ----------------------------------------------------------------------------

def rankdata(a):
    """Average ranks (1-based) with ties sharing the mean rank, like scipy.stats.rankdata."""
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), dtype=np.int64)
    inv[sorter] = np.arange(len(a))
    a_s = a[sorter]
    obs = np.r_[True, a_s[1:] != a_s[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else np.nan


def load_matched(files):
    """Read one time grid's matched files into compact arrays sorted by station.

    Returns (stids, starts, ends, time_s, sustained, gust): rows for stids[i] are
    starts[i]:ends[i] of the row arrays.
    """
    code_of, names = {}, []
    codes, times, sus, gus = [], [], [], []
    for f in files:
        t = pq.read_table(f, columns=["stid", "time_utc", "sustained_max_mps", "gust_max_mps"])
        col = t.column("stid").combine_chunks()
        if not isinstance(col.type, pa.DictionaryType):
            col = col.dictionary_encode()
        remap = np.array([code_of.setdefault(s, len(code_of)) for s in col.dictionary.to_pylist()],
                         dtype=np.int16)
        codes.append(remap[col.indices.to_numpy()])
        # parquet stores timestamp[s] as [ms]; convert explicitly to epoch seconds
        times.append(t.column("time_utc").cast(pa.timestamp("s", "UTC")).cast(pa.int64()).to_numpy())
        sus.append(t.column("sustained_max_mps").to_numpy(zero_copy_only=False))
        gus.append(t.column("gust_max_mps").to_numpy(zero_copy_only=False))
    names = np.array(list(code_of), dtype=object)
    codes = np.concatenate(codes)
    order = np.argsort(codes, kind="stable")  # keeps time order within station
    codes = codes[order]
    time_s = np.concatenate(times)[order]
    sus = np.concatenate(sus)[order]
    gus = np.concatenate(gus)[order]
    uniq, starts = np.unique(codes, return_index=True)
    ends = np.r_[starts[1:], len(codes)]
    return names[uniq], starts, ends, time_s, sus, gus


def step_correlate(args):
    suffix = f"_before{args.before}" if args.before else ""
    cutoff_s = int(pd.Timestamp(args.before, tz="UTC").timestamp()) if args.before else None
    pairs = load_pairs(args.gradients)
    ts = pd.read_parquet(os.path.join(out_dir(args.data), "gradient_timeseries.parquet"))
    st = load_stations(args.data).set_index("stid")
    rows = []
    t0 = time.time()
    for s1 in sorted({p[0] for p in pairs.values()}):
        files = sorted(glob.glob(os.path.join(out_dir(args.data), "matched", s1, "*.parquet")))
        if not files:
            print(f"WARNING: no matched files for {s1}")
            continue
        stids, starts, ends, time_s, sus, gus = load_matched(files)
        print(f"{s1}: {len(time_s):,} matched rows, {len(stids)} stations ({time.time() - t0:.0f}s)", flush=True)
        for name, (a, _) in pairs.items():
            if a != s1:
                continue
            g = ts[ts.gradient == name]
            gt = epoch_s(g.time_utc)
            gv = g.gradient_hpa.to_numpy()
            order = np.argsort(gt)
            gt, gv = gt[order], gv[order]
            idx = np.searchsorted(gt, time_s)
            idx_c = np.minimum(idx, len(gt) - 1)
            x = np.where(gt[idx_c] == time_s, gv[idx_c], np.nan)  # NaN where station2 missing
            if args.before:
                x[time_s >= cutoff_s] = np.nan  # held-out period: not used for correlations
            for i, stid in enumerate(stids):
                sl = slice(starts[i], ends[i])
                xs = x[sl]
                for var, ys in (("sustained", sus[sl]), ("gust", gus[sl])):
                    ok = ~np.isnan(xs) & ~np.isnan(ys)
                    for sign, m in (("pos", ok & (xs > 0)), ("neg", ok & (xs < 0))):
                        n = int(m.sum())
                        if n < 2:
                            continue
                        rows.append((name, sign, name + SIGNS[sign], var, stid, n,
                                     spearman(np.abs(xs[m]), ys[m].astype(np.float64))))
            print(f"  {name:8s} done ({time.time() - t0:.0f}s)", flush=True)
        del time_s, sus, gus

    out = pd.DataFrame(rows, columns=["gradient", "sign", "signed_gradient", "variable", "stid", "n", "rho"])
    out["retained"] = (out.n >= args.min_n) & (out.rho >= args.min_rho)
    meta = st[["name", "state", "network_shortname", "lat", "lon", "elevation_m"]]
    out = out.merge(meta, left_on="stid", right_index=True, how="left")
    table = pa.Table.from_pandas(out, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), b"selection": json.dumps(
        {"min_n": args.min_n, "min_rho": args.min_rho, "match_minutes": MATCH_MINUTES,
         "predictor": "abs(gradient_hpa) within sign", "method": "spearman, average ranks",
         "before": args.before,
         "excluded_networks": list(EXCLUDE_NETWORK_PREFIXES),
         "matched_with": json.load(open(os.path.join(out_dir(args.data), "match_qc.json")))
         if os.path.exists(os.path.join(out_dir(args.data), "match_qc.json")) else None,
         "created_utc": pd.Timestamp.now(tz="UTC").isoformat()}).encode()})
    path = os.path.join(out_dir(args.data), f"correlations{suffix}.parquet")
    pq.write_table(table, path)

    ev = out[out.n >= args.min_n]
    summ = (ev.groupby(["signed_gradient", "variable"])
            .agg(stations_evaluated=("stid", "size"), stations_retained=("retained", "sum"),
                 max_rho=("rho", "max"), median_rho=("rho", "median"))
            .unstack("variable"))
    summ.columns = [f"{v}_{c}" for c, v in summ.columns]
    order = [f"{n}{s}" for n in pairs for s in SIGNS.values()]
    summ = summ.reindex([o for o in order if o in summ.index]).round(3)
    spath = os.path.join(out_dir(args.data), f"retained_summary{suffix}.csv")
    summ.to_csv(spath)
    print(f"Wrote {path}\nWrote {spath}")
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(summ[[c for c in summ.columns if "retained" in c or "max_rho" in c or "evaluated" in c]])


# ----------------------------------------------------------------------------
# Step 4: maps
# ----------------------------------------------------------------------------

def step_maps(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    pairs = load_pairs(args.gradients)
    corr = pd.read_parquet(os.path.join(out_dir(args.data), "correlations.parquet"))
    st = load_stations(args.data).set_index("stid")
    os.makedirs(args.images, exist_ok=True)
    # single-hue sequential ramp (light -> dark) starting visibly above white
    cmap = LinearSegmentedColormap.from_list("rho", ["#9ec5f0", "#2a78d6", "#0b3b7a"])
    norm = Normalize(vmin=args.min_rho, vmax=1.0)
    proj = ccrs.PlateCarree()
    extent_all = [-125.0, -113.5, 31.8, 49.3]

    for name, (s1, s2) in pairs.items():
        for sign, sym in SIGNS.items():
            sub = corr[(corr.gradient == name) & (corr.sign == sign) & (corr.n >= args.min_n)]
            keep = sub[sub.retained]
            pts = st.loc[[s1, s2], ["lat", "lon"]]
            if len(keep):
                lon0, lon1 = min(keep.lon.min(), pts.lon.min()), max(keep.lon.max(), pts.lon.max())
                lat0, lat1 = min(keep.lat.min(), pts.lat.min()), max(keep.lat.max(), pts.lat.max())
            else:
                lon0, lon1, lat0, lat1 = pts.lon.min(), pts.lon.max(), pts.lat.min(), pts.lat.max()
            cx, cy = (lon0 + lon1) / 2, (lat0 + lat1) / 2
            half = max(lon1 - lon0, (lat1 - lat0) * 1.25, 4.0) / 2 + 0.8
            ext = [max(cx - half, extent_all[0]), min(cx + half, extent_all[1]),
                   max(cy - half / 1.25, extent_all[2]), min(cy + half / 1.25, extent_all[3])]

            fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), subplot_kw={"projection": proj})
            for ax, var in zip(axes, VARS):
                ax.set_extent(ext, crs=proj)
                ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f4f3ef")
                ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#e6eef5")
                ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="#8a8984", linewidth=0.6)
                ax.coastlines("50m", color="#8a8984", linewidth=0.6)
                ev = sub[sub.variable == var]
                kv = ev[ev.retained].sort_values("rho")
                ax.scatter(ev.lon, ev.lat, s=6, c="#c3c2b7", linewidths=0, transform=proj, zorder=2,
                           label=f"evaluated, rho < {args.min_rho} ({len(ev) - len(kv)})")
                sc = ax.scatter(kv.lon, kv.lat, c=kv.rho, cmap=cmap, norm=norm, s=34,
                                edgecolors="white", linewidths=0.6, transform=proj, zorder=3,
                                label=f"retained, rho >= {args.min_rho} ({len(kv)})")
                for stid, marker in ((s1, "^"), (s2, "v")):
                    ax.scatter(st.at[stid, "lon"], st.at[stid, "lat"], marker=marker, s=110, c="#0b0b0b",
                               edgecolors="white", linewidths=1.0, transform=proj, zorder=4)
                    ax.annotate(stid, (st.at[stid, "lon"], st.at[stid, "lat"]), xytext=(7, 5),
                                textcoords="offset points", fontsize=9, fontweight="bold", zorder=5)
                for _, r in kv.nlargest(5, "rho").iterrows():
                    ax.annotate(f"{r.stid} {r.rho:.2f}", (r.lon, r.lat), xytext=(5, -9),
                                textcoords="offset points", fontsize=7.5, color="#52514e", zorder=5)
                ax.set_title(f"{'Max sustained (2-min)' if var == 'sustained' else 'Max gust'}: "
                             f"{len(kv)} of {len(ev)} stations retained", loc="left", fontsize=11)
                ax.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.9)
            cb = fig.colorbar(sc, ax=axes, orientation="horizontal", fraction=0.04, pad=0.04, aspect=50)
            cb.set_label(f"Spearman rho, |gradient| vs wind (stations with n >= {args.min_n})")
            direction = f"{s1} SLP > {s2} SLP" if sign == "pos" else f"{s1} SLP < {s2} SLP"
            fig.suptitle(f"{name}{sym}  ({direction}; {s1} ▲, {s2} ▼)", fontsize=14)
            path = os.path.join(args.images, f"{name}_{sign}.png")
            fig.savefig(path, dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["timeseries", "match", "correlate", "maps", "all"])
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--gradients", default=DEFAULT_GRADIENTS)
    ap.add_argument("--images", default=DEFAULT_IMAGES)
    ap.add_argument("--min-n", type=int, default=MIN_N)
    ap.add_argument("--min-rho", type=float, default=MIN_RHO)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true", help="match: redo months already written")
    ap.add_argument("--before", default=None, help="correlate: use only times before this date (YYYY-MM-DD)")
    args = ap.parse_args()
    steps = ["timeseries", "match", "correlate", "maps"] if args.step == "all" else [args.step]
    for s in steps:
        print(f"=== {s} ===")
        globals()[f"step_{s}"](args)


if __name__ == "__main__":
    main()
