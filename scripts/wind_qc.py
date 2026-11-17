#!/usr/bin/env python3
"""Wind observation QC: each station is judged on its OWN time series and statistics.

Neighbour comparisons can't tell a biased sensor from a station sited to sample a local
terrain effect (common in mountains), so the rules that remove data (CONFIG["mask_rules"])
use only the station's own record. Neighbour diagnostics (NEIGH, DECOR) are still computed
and written, for inspection only.

Two levels:
  1. Observation flags (bitmask per reading, below).
  2. Station exclusion: a station whose share of sustained readings masked by the automatic
     rules (everything in mask_rules except MANUAL) is
     >= station_exclude_share is listed in station_exclusions.csv and dropped from analysis
     (load_month_qc(..., drop_excluded=True)).

Rules (applied to sustained `wind_speed_mps` and `wind_gust_mps` separately unless noted).
Readings are a station's consecutive non-missing reports; two reports are "consecutive" only
if they are <= max_gap_min apart. Bits marked [mask] remove data by default.

  CONST  (bit 1, mask)  Constant high value. A run of consecutive identical reports (rounded to
                  0.01 m/s) is flagged when the value >= const_min_mps, the run has
                  >= const_min_reports reports and lasts >= const_min_minutes.
  SPIKE  (bit 2, mask)  Jump between consecutive reports. thr = the station's spike_pct percentile of
                  |change between consecutive reports|, clipped to [spike_floor_mps,
                  spike_cap_mps]; percentiles are fitted once by `fit` and then frozen.
                  The floor matters: station percentiles are 1.5-4 m/s and without it the
                  rule mostly removes real gusty variability (images/wind_qc/
                  spike_calibration.png). The cap matters at stations that spike often:
                  their own spikes inflate the percentile (LALN2 p99.9 = 42 m/s).
                    spike_mode "prior":     |v_i - v_prev| > thr
                    spike_mode "two_sided": v_i departs from BOTH the previous and the next
                                            report by > thr in the same direction (up-and-back
                                            spike or drop-out). The reading after a spike is
                                            not flagged.
                  Only readings >= spike_min_mps are flagged.
  NOISE  (bit 64, mask)  Erratic day, per station and UTC day, from sustained readings: the
                  day has >= noise_min_jumps jumps between consecutive reports larger than the
                  station's spike threshold, and they are >= noise_min_share of the day's
                  consecutive pairs. All readings that day (sustained and gust) are flagged.
                  Catches bursts of noise that aren't single up-and-back spikes (MCUES,
                  CPPW, PVPN2).
  SHIFT  (bit 32, mask)  Level shift, per station and calendar month: the month's median
                  sustained wind (>= shift_min_reports readings) is > shift_ratio x the
                  station's typical monthly median (median of its monthly medians, from
                  `fit`) and > shift_min_mps. Catches months-long stretches where a sensor
                  reads far above its own normal (MDTCA 2016-17, DDRIO/DDBUF 2023-24).
  LEVEL  (bit 128, mask)  Implausible station: the station's typical monthly median >
                  level_max_mps (all its readings). Catches stations that are bad for their
                  whole record, so SHIFT has no normal period to compare with (CLE, BEUO,
                  KEE, OWY: 22-84 m/s; the next highest is the 4342 m WMSRS summit at 11).
  RANGE  (bit 8, mask)  Physical limit: sustained >= range_max_spd_mps or gust >=
                  range_max_gust_mps.
  MANUAL (bit 256, mask)  Station periods in CONFIG["manual_exclusions"], each with a reason,
                  added after visual review. Not counted toward station exclusion.

  Neighbour diagnostics (info only; not in mask_rules).
  Neighbours: up to neigh_max stations within neigh_km and +/- neigh_dz_m elevation. The
  station's hourly MEAN sustained wind is compared with the median of its neighbours' hourly
  means (means, so 1-min and hourly reporters compare fairly). Every reading in a flagged
  station-day / station-month (sustained AND gust) gets the bit; untestable periods don't.

  NEIGH  (bit 4, info)  Implausibly windy vs neighbours, per UTC day (>= neigh_mag_min_hours joint
                  hours): the station's p95 > neigh_mag_min_mps AND > neigh_mag_ratio x the
                  neighbours' p95. Catches broken sensors whose garbage still rises and falls
                  with storms (MDTCA 2016-17, monthly rho 0.3-0.5) and noise bursts lasting
                  hours to days (MCUES). Calibrated on known-good and known-bad stations:
                  flags 66% of days at the bad ones and 5% at real exposed ridge stations.
  DECOR  (bit 16, info)  Decorrelated from neighbours, per calendar month (>= neigh_min_hours joint
                  hours): Spearman rho < neigh_rho_min AND the station's p95 >
                  neigh_decor_ratio x the neighbours' p95 (calmer, decorrelated sheltered
                  stations are never flagged). INFORMATIONAL by default (not in mask_rules):
                  of 1,287 flagged station-months only 63 were at known-bad stations and 9 of
                  those weren't caught by other rules; the rest were real exposed sites with
                  local wind regimes (San Marcos Pass, Round Mountain, Eagle Peak...). See
                  neigh_calibration.png and inspect_decor_examples.png.

Steps:
  fit       One pass over obs_raw: per-station percentiles of |consecutive change| (spike
            thresholds), typical monthly median (SHIFT, LEVEL), constant-run calibration
            tables, neighbour table. Frozen until fit is rerun; stations that appear later
            have no SHIFT/LEVEL reference (network-median spike threshold) until then.
            -> data/v2/derived/qc/wind/{thresholds,station_level,neighbours}.parquet
  apply     Per month: flags -> data/v2/qc/wind/YYYY/MM.parquet (flagged readings only:
            stid, time_utc, qc_spd, qc_gust bitmasks), station-month neighbour stats and
            before/after histograms. Months already done are skipped (--overwrite to redo),
            so new months can be appended with frozen thresholds.
  evaluate  Station summary and exclusion list (station_summary.csv, station_exclusions.csv),
            calibration and utility plots -> images/wind_qc/
  inspect   Time series of chosen stations with flags by rule -> images/wind_qc/inspect_*.png
  gallery   inspect for a fixed set of known-bad and known-good stations -> inspect_gallery.png

Downstream code: `load_month_qc(year, month, drop_excluded=True)` returns obs_raw rows for the
month with values flagged by any rule in CONFIG["mask_rules"] set to NaN, excluded stations
removed, and the bitmasks attached.

Usage:
  python wind_qc.py fit
  python wind_qc.py apply [--months 2026-11 ...] [--overwrite]
  python wind_qc.py evaluate
  python wind_qc.py inspect MDTCA SLIDE [--start 2016-06 --end 2018-06]
  python wind_qc.py gallery

To change a rule: edit CONFIG, then `apply --overwrite` and `evaluate`. Thresholds from `fit`
stay frozen until `fit` is rerun. config.json in the model directory records the CONFIG used.
"""
import argparse
import glob
import json
import os
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data", "v2"))
MODEL = os.path.join(DATA, "derived", "qc", "wind")      # fitted thresholds, tables
FLAGS = os.path.join(DATA, "qc", "wind")                 # per-month flag files
IMAGES = os.path.normpath(os.path.join(HERE, "..", "images", "wind_qc"))

CONFIG = {
    "version": 1,
    "exclude_network_prefixes": ["WWG"],
    "max_gap_min": 120,
    # CONST
    "const_min_mps": 20.0,
    "const_min_reports": 3,
    "const_min_minutes": 60,
    # SPIKE
    "spike_pct": 99.9,
    "spike_floor_mps": 10.0,
    "spike_cap_mps": 15.0,
    "spike_mode": "two_sided",
    "spike_min_mps": 0.0,
    # NOISE
    "noise_min_jumps": 5,
    "noise_min_share": 0.05,
    # SHIFT / LEVEL
    "shift_min_reports": 100,
    "shift_ratio": 4.0,
    "shift_min_mps": 10.0,
    "level_max_mps": 15.0,
    # station exclusion
    "station_exclude_share": 0.05,
    "station_exclude_min_reports": 1000,
    # NEIGH / DECOR (info only)
    "neigh_km": 75,
    "neigh_dz_m": 400,
    "neigh_max": 5,
    "neigh_min_hours": 200,
    "neigh_rho_min": 0.2,
    "neigh_decor_ratio": 3.0,
    "neigh_mag_ratio": 4.0,
    "neigh_mag_min_mps": 25.0,
    "neigh_mag_min_hours": 6,
    # RANGE
    "range_max_spd_mps": 60.0,
    "range_max_gust_mps": 80.0,
    # which rules remove data (load_month_qc, before/after plots)
    "mask_rules": ["CONST", "SPIKE", "NOISE", "SHIFT", "LEVEL", "RANGE", "MANUAL"],
    # MANUAL: station periods removed after visual review (start inclusive, end exclusive, None = open)
    "manual_exclusions": [
        {"stid": "KWSIX", "start": "2025-12-01", "end": None,
         "reason": "Real ridge wind through mid-2025 (vs HVRTM rho 0.63, ratio 1.15), then degraded: "
                   "Dec 2025 median 36 m/s vs SLIDE 7 m/s; Jan-Mar 2026 medians 17-21 m/s (~2x normal) "
                   "pass SHIFT. Reviewed 2026-11-12 (images/wind_qc/verify/KWSIX.png)."},
    ],
}
VARS = {"spd": "wind_speed_mps", "gust": "wind_gust_mps"}
CONST, SPIKE, NEIGH, RANGE, DECOR, SHIFT, NOISE, LEVEL, MANUAL = 1, 2, 4, 8, 16, 32, 64, 128, 256
QC_DTYPE = np.uint16
RULES = {"CONST": CONST, "SPIKE": SPIKE, "NOISE": NOISE, "SHIFT": SHIFT, "LEVEL": LEVEL, "MANUAL": MANUAL,
         "RANGE": RANGE, "NEIGH": NEIGH, "DECOR": DECOR}


def auto_mask_bits(cfg=CONFIG):
    """Masked automatic rules only; MANUAL does not count toward station exclusion."""
    return mask_bits(cfg) & ~MANUAL


def mask_bits(cfg=CONFIG):
    return int(sum(RULES[r] for r in cfg["mask_rules"]))
PCTS = [90, 95, 99, 99.5, 99.9]
DBIN, NDBIN = 0.1, 1000                       # |change| histogram: 0.1 m/s up to 100 m/s
RUN_V_EDGES = np.r_[np.arange(0, 62.5, 2.5), np.inf]
RUN_D_EDGES = np.array([0, 15, 30, 60, 120, 180, 360, 720, 1440, np.inf])
HIST_EDGES = np.r_[np.arange(0, 100.01, 0.5), np.inf]


# ============================================================================ data access

def stations():
    s = pd.read_parquet(os.path.join(DATA, "stations.parquet")).drop_duplicates("stid")
    s["network_shortname"] = s.network_shortname.fillna("UNKNOWN")
    ex = tuple(CONFIG["exclude_network_prefixes"])
    return s[~s.network_shortname.str.startswith(ex)].sort_values("stid").reset_index(drop=True)


def month_files(ym):
    y, m = ym.split("-")
    return sorted(glob.glob(os.path.join(DATA, "obs_raw", y, m, "*.parquet")))


def all_months():
    return sorted({"-".join(p.split(os.sep)[-3:-1]) for p in glob.glob(os.path.join(DATA, "obs_raw", "*", "*", "*.parquet"))})


def load_raw(paths, code_of):
    """obs_raw rows for known stations, sorted by station then time, with int minute times."""
    tabs = [pq.read_table(p, columns=["stid", "time_utc", *VARS.values()]) for p in paths]
    t = pa.concat_tables(tabs).to_pandas()
    t["code"] = t.stid.map(code_of)
    t = t[t.code.notna()]
    t["code"] = t.code.astype(np.int32)
    t["tmin"] = t.time_utc.astype("datetime64[ms, UTC]").astype("int64") // 60000
    return t.sort_values(["code", "tmin"], kind="stable").reset_index(drop=True)


def valid(df, col):
    """Positions, station codes, minute times and values of the non-missing readings of col."""
    v = df[col].to_numpy(np.float64)
    pos = np.flatnonzero(np.isfinite(v) & (v >= 0))
    return pos, df.code.to_numpy()[pos], df.tmin.to_numpy()[pos], v[pos]


def links(code, t, max_gap):
    """same_prev[i]: reading i-1 is the same station and within max_gap minutes."""
    sp = np.r_[False, (code[1:] == code[:-1]) & (np.diff(t) <= max_gap)]
    return sp, np.r_[sp[1:], False]


# ============================================================================ the rules

def rule_const(code, t, v, cfg=CONFIG):
    """-> per-reading flag, and the run table (value, n, duration) for calibration."""
    vr = np.round(v, 2)
    sp, _ = links(code, t, cfg["max_gap_min"])
    brk = ~sp | np.r_[True, vr[1:] != vr[:-1]]
    rid = np.cumsum(brk) - 1
    starts = np.flatnonzero(brk)
    ends = np.r_[starts[1:], len(v)] - 1
    n = ends - starts + 1
    dur = t[ends] - t[starts]
    val = vr[starts]
    flag_run = (val >= cfg["const_min_mps"]) & (n >= cfg["const_min_reports"]) & (dur >= cfg["const_min_minutes"])
    return flag_run[rid], (val, n, dur, code[starts], t[starts])


def rule_spike(code, t, v, thr, mode=None, floor=None, cfg=CONFIG):
    """thr: per-reading threshold (NaN = untested)."""
    mode = mode or cfg["spike_mode"]
    if floor is None:
        thr = np.clip(thr, cfg["spike_floor_mps"], cfg["spike_cap_mps"])
    else:
        thr = np.maximum(thr, floor)
    sp, sn = links(code, t, cfg["max_gap_min"])
    dp = v - np.r_[np.nan, v[:-1]]
    dn = v - np.r_[v[1:], np.nan]
    with np.errstate(invalid="ignore"):
        if mode == "prior":
            f = sp & (np.abs(dp) > thr)
        elif mode == "two_sided":
            f = sp & sn & (((dp > thr) & (dn > thr)) | ((dp < -thr) & (dn < -thr)))
        else:
            raise ValueError(mode)
        return f & (v >= cfg["spike_min_mps"])


def hourly_matrix(df, nst, agg="mean"):
    """(nst, nhours) hourly mean (or max) sustained wind, and the first hour index."""
    pos, code, t, v = valid(df, VARS["spd"])
    h = t // 60
    h0 = h.min()
    H = np.full((nst, h.max() - h0 + 1), np.nan, np.float32)
    k = code.astype(np.int64) * H.shape[1] + (h - h0)
    order = np.argsort(k, kind="stable")
    k, vv = k[order], v[order]
    first = np.r_[True, k[1:] != k[:-1]]
    idx = np.flatnonzero(first)
    if agg == "max":
        H.flat[k[first]] = np.maximum.reduceat(vv, idx)
    else:
        H.flat[k[first]] = np.add.reduceat(vv, idx) / np.diff(np.r_[idx, len(vv)])
    return H, h0


def neighbour_ref(H, nbr):
    """nbr: (nst, neigh_max) neighbour codes, -1 padded -> median/max over neighbours, n available."""
    pad = np.vstack([H, np.full((1, H.shape[1]), np.nan, np.float32)])
    stack = pad[np.where(nbr < 0, len(H), nbr)]           # (nst, k, nh)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(stack, 1), np.nanmax(stack, 1), np.isfinite(stack).sum(1)


def rankdata(a):
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), dtype=np.int64)
    inv[sorter] = np.arange(len(a))
    a_s = a[sorter]
    obs = np.r_[True, a_s[1:] != a_s[:-1]]
    dense = obs.cumsum()[inv]
    cnt = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (cnt[dense] + cnt[dense - 1] + 1)


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else np.nan


def rule_neigh(H, ref, cfg=CONFIG):
    """-> per-station rho, joint hours, p95 of station and reference, and flag."""
    nst = len(H)
    rho = np.full(nst, np.nan)
    nj = np.zeros(nst, np.int64)
    p95s = np.full(nst, np.nan)
    p95r = np.full(nst, np.nan)
    for i in range(nst):
        m = np.isfinite(H[i]) & np.isfinite(ref[i])
        nj[i] = m.sum()
        if nj[i] >= cfg["neigh_min_hours"]:
            x, y = H[i, m].astype(np.float64), ref[i, m].astype(np.float64)
            rho[i] = spearman(x, y)
            p95s[i], p95r[i] = np.percentile(x, 95), np.percentile(y, 95)
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = nj >= cfg["neigh_min_hours"]
        ratio = p95s / np.maximum(p95r, 0.5)
        decor = ok & (rho < cfg["neigh_rho_min"]) & (ratio > cfg["neigh_decor_ratio"])
    return rho, nj, p95s, p95r, decor


def rule_neigh_daily(H, ref, h0, cfg=CONFIG):
    """-> (nst, ndays) magnitude flag, and the first day index. Days are UTC days."""
    d0 = h0 // 24
    nd = (h0 + H.shape[1] - 1) // 24 - d0 + 1
    flag = np.zeros((len(H), nd), bool)
    day = (h0 + np.arange(H.shape[1])) // 24 - d0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for d in range(nd):
            c = day == d
            a, b = H[:, c], ref[:, c]
            j = np.isfinite(a) & np.isfinite(b)
            a, b = np.where(j, a, np.nan), np.where(j, b, np.nan)
            n = j.sum(1)
            ps, pr = np.nanpercentile(a, 95, axis=1), np.nanpercentile(b, 95, axis=1)
            with np.errstate(invalid="ignore"):
                flag[:, d] = ((n >= cfg["neigh_mag_min_hours"]) & (ps > cfg["neigh_mag_min_mps"])
                              & (ps > cfg["neigh_mag_ratio"] * np.maximum(pr, 0.5)))
    return flag, d0


# ============================================================================ fit

def neighbour_table(st, cfg=CONFIG):
    lat, lon, z = (st[c].to_numpy(np.float64) for c in ("lat", "lon", "elevation_m"))
    p = np.pi / 180
    rows = []
    nbr = np.full((len(st), cfg["neigh_max"]), -1, np.int64)
    for i in range(len(st)):
        if not np.isfinite(z[i]):
            continue
        a = np.sin((lat - lat[i]) * p / 2) ** 2 + np.cos(lat[i] * p) * np.cos(lat * p) * np.sin((lon - lon[i]) * p / 2) ** 2
        d = 12742 * np.arcsin(np.sqrt(a))
        ok = (d <= cfg["neigh_km"]) & (np.abs(z - z[i]) <= cfg["neigh_dz_m"])
        ok[i] = False
        j = np.flatnonzero(ok)
        j = j[np.argsort(d[j])][:cfg["neigh_max"]]
        nbr[i, :len(j)] = j
        rows += [{"stid": st.stid.iloc[i], "neighbour": st.stid.iloc[k], "dist_km": round(d[k], 1),
                  "dz_m": round(z[k] - z[i])} for k in j]
    return nbr, pd.DataFrame(rows)


_G = {}


def _init(g):
    _G.update(g)


def fit_file(path):
    df = load_raw([path], _G["code_of"])
    nst, cfg = _G["nst"], CONFIG
    out = {}
    for short, col in VARS.items():
        pos, code, t, v = valid(df, col)
        sp, _ = links(code, t, cfg["max_gap_min"])
        d = np.abs(np.diff(v))[sp[1:]]
        c = code[1:][sp[1:]]
        b = np.minimum((d / DBIN + 1e-6).astype(np.int64), NDBIN - 1)
        k, n = np.unique(c.astype(np.int64) * NDBIN + b, return_counts=True)
        gaps = np.diff(t)[sp[1:]]
        gk, gn = np.unique(c.astype(np.int64) * 1000 + np.minimum(gaps, 999), return_counts=True)
        _, (val, rn, dur, rcode, rt0) = rule_const(code, t, v)
        m = rn >= 2
        rh, _, _ = np.histogram2d(val[m], dur[m], [RUN_V_EDGES, RUN_D_EDGES])
        long = m & (val >= 10) & (dur >= 60)
        runs = pd.DataFrame({"code": rcode[long], "value": val[long], "n": rn[long], "dur_min": dur[long],
                             "start_min": rt0[long]})
        out[short] = (k, n, gk, gn, rh, runs)
    pos, code, t, v = valid(df, VARS["spd"])
    y, m = path.split(os.sep)[-3:-1]
    mm = pd.DataFrame({"code": code, "v": v}).groupby("code").v.agg(["size", "median"]).reset_index()
    mm["month"] = f"{y}-{m}"
    return out, mm


def step_fit(workers):
    os.makedirs(MODEL, exist_ok=True)
    st = stations()
    nst = len(st)
    code_of = dict(zip(st.stid, st.index))
    nbr, nb = neighbour_table(st)
    nb.to_parquet(os.path.join(MODEL, "neighbours.parquet"))
    print(f"{nst} stations; {int((nbr[:, 0] >= 0).sum())} have >= 1 neighbour")
    D = {s: np.zeros(nst * NDBIN, np.int64) for s in VARS}
    G = {s: {} for s in VARS}
    RH = {s: np.zeros((len(RUN_V_EDGES) - 1, len(RUN_D_EDGES) - 1)) for s in VARS}
    RUNS = {s: [] for s in VARS}
    files = sorted(glob.glob(os.path.join(DATA, "obs_raw", "*", "*", "*.parquet")))
    with Pool(workers, initializer=_init, initargs=({"code_of": code_of, "nst": nst},)) as p:
        MM = []
        for i, (out, mm) in enumerate(p.imap_unordered(fit_file, files, chunksize=2)):
            MM.append(mm)
            for s, (k, n, gk, gn, rh, runs) in out.items():
                np.add.at(D[s], k, n)
                for a, b in zip(gk, gn):
                    G[s][a] = G[s].get(a, 0) + b
                RH[s] += rh
                RUNS[s].append(runs)
            if (i + 1) % 100 == 0:
                print(f"  fit {i + 1}/{len(files)}")
    centers = (np.arange(NDBIN) + 1) * DBIN       # upper bin edge: conservative threshold
    rows = []
    for s in VARS:
        Hd = D[s].reshape(nst, NDBIN)
        n = Hd.sum(1)
        c = np.cumsum(Hd, 1) / np.maximum(n, 1)[:, None]
        g = pd.Series(G[s])
        gdf = pd.DataFrame({"code": g.index // 1000, "gap": g.index % 1000, "n": g.values})
        modal = gdf.sort_values("n").groupby("code").gap.last()
        for i in range(nst):
            if n[i] == 0:
                continue
            r = {"stid": st.stid.iloc[i], "network": st.network_shortname.iloc[i], "var": s,
                 "n_pairs": int(n[i]), "modal_gap_min": int(modal.get(i, -1))}
            for q in PCTS:
                r[f"p{q:g}"] = float(centers[min(np.searchsorted(c[i], q / 100), NDBIN - 1)])
            rows.append(r)
        np.save(os.path.join(MODEL, f"dchange_hist_{s}.npy"), Hd)
        np.save(os.path.join(MODEL, f"const_runs_hist_{s}.npy"), RH[s])
        runs = pd.concat(RUNS[s])
        runs["stid"] = st.stid.to_numpy()[runs.code]
        runs.drop(columns="code").to_parquet(os.path.join(MODEL, f"const_long_runs_{s}.parquet"))
    thr = pd.DataFrame(rows)
    thr.to_parquet(os.path.join(MODEL, "thresholds.parquet"))
    mm = pd.concat(MM)
    mm["stid"] = st.stid.to_numpy()[mm.code]
    mm.drop(columns="code").to_parquet(os.path.join(MODEL, "station_month_medians.parquet"))
    ok = mm[mm["size"] >= CONFIG["shift_min_reports"]]
    lvl = (ok.groupby("stid").agg(typical_month_median=("median", "median"), months=("median", "size"))
           .join(mm.groupby("stid")["size"].sum().rename("n_spd")))
    lvl.reset_index().to_parquet(os.path.join(MODEL, "station_level.parquet"))
    with open(os.path.join(MODEL, "fit_info.json"), "w") as f:
        json.dump({"months": [all_months()[0], all_months()[-1]], "config_at_fit": CONFIG}, f, indent=2)
    print(thr.groupby("var")[[f"p{q:g}" for q in PCTS]].median())


def load_station_level(st):
    """typical monthly median per station code (NaN if not fitted)."""
    lvl = pd.read_parquet(os.path.join(MODEL, "station_level.parquet")).set_index("stid")
    return st.stid.map(lvl.typical_month_median).to_numpy(np.float64)


def load_thresholds(st, pct):
    """Per-variable threshold array indexed by station code; network median as fallback."""
    thr = pd.read_parquet(os.path.join(MODEL, "thresholds.parquet"))
    col = f"p{pct:g}"
    out = {}
    for s in VARS:
        t = thr[(thr["var"] == s) & (thr.n_pairs >= 500)]
        a = st.stid.map(t.set_index("stid")[col]).to_numpy(np.float64)
        net = st.network_shortname.map(t.groupby("network")[col].median()).to_numpy(np.float64)
        a = np.where(np.isfinite(a), a, net)
        out[s] = np.where(np.isfinite(a), a, np.nanmedian(t[col]))
    return out


# ============================================================================ apply

def rule_noise(code, t, v, thr, nst, cfg=CONFIG):
    """-> (nst, ndays) erratic-day flag, first day index, and per-station-day jump counts."""
    sp, _ = links(code, t, cfg["max_gap_min"])
    jump = sp & (np.abs(v - np.r_[np.nan, v[:-1]]) > np.clip(thr, cfg["spike_floor_mps"], cfg["spike_cap_mps"]))
    day = t // 1440
    d0 = day.min()
    nd = day.max() - d0 + 1
    k = code.astype(np.int64) * nd + (day - d0)
    pairs = np.bincount(k[sp], minlength=nst * nd).reshape(nst, nd)
    jumps = np.bincount(k[jump], minlength=nst * nd).reshape(nst, nd)
    flag = (jumps >= cfg["noise_min_jumps"]) & (jumps >= cfg["noise_min_share"] * np.maximum(pairs, 1))
    return flag, d0, jumps, pairs


def rule_shift(code, v, typical, nst, cfg=CONFIG):
    """-> per-station month median, n readings, and level-shift flag."""
    n = np.bincount(code, minlength=nst)
    med = np.full(nst, np.nan)
    if len(v):
        o = np.lexsort((v, code))
        c, vv = code[o], v[o]
        start = np.r_[0, np.cumsum(n)[:-1]]
        has = n > 0
        lo = start[has] + (n[has] - 1) // 2
        hi = start[has] + n[has] // 2
        med[has] = 0.5 * (vv[lo] + vv[hi])
    with np.errstate(invalid="ignore"):
        flag = ((n >= cfg["shift_min_reports"]) & (med > cfg["shift_ratio"] * np.maximum(typical, 0.5))
                & (med > cfg["shift_min_mps"]))
    return med, n, flag


def day_lookup(flag, d0, tmin):
    """Row-aligned lookup into a (nst, ndays) day-flag array; days outside the array are False."""
    def f(codes):
        idx = tmin // 1440 - d0
        ok = (idx >= 0) & (idx < flag.shape[1])
        out = np.zeros(len(codes), bool)
        out[ok] = flag[codes[ok], idx[ok]]
        return out
    return f


def flag_month(df, H, h0, nbr, thr, typical, cfg=CONFIG):
    """-> (qc bitmask per var aligned with df rows, station-month table)."""
    nst = len(H)
    ref, _, navail = neighbour_ref(H, nbr)
    rho, nj, p95s, p95r, decor = rule_neigh(H, ref, cfg)
    dflag, d0 = rule_neigh_daily(H, ref, h0, cfg)
    rows_code = df.code.to_numpy()
    tmin = df.tmin.to_numpy()
    row_neigh = day_lookup(dflag, d0, tmin)(rows_code)
    row_decor = decor[rows_code]
    mag = dflag.any(1)
    # self-based station-level rules, from sustained readings
    _, scode, st_, sv = valid(df, VARS["spd"])
    nflag, nd0, jumps, pairs = rule_noise(scode, st_, sv, thr["spd"][scode], nst, cfg)
    jc, jd = np.nonzero(jumps)
    jdays = pd.DataFrame({"code": jc, "day": jd + nd0, "jumps": jumps[jc, jd], "pairs": pairs[jc, jd],
                          "noise": nflag[jc, jd]})
    row_noise = day_lookup(nflag, nd0, tmin)(rows_code)
    med, nread, shift = rule_shift(scode, sv, typical, nst, cfg)
    with np.errstate(invalid="ignore"):
        level = typical > cfg["level_max_mps"]
    row_shift = shift[rows_code]
    row_level = level[rows_code]
    row_manual = manual_rows(df, cfg)
    qc = {}
    for s, col in VARS.items():
        q = np.zeros(len(df), QC_DTYPE)
        pos, code, t, v = valid(df, col)
        fc, _ = rule_const(code, t, v, cfg)
        fs = rule_spike(code, t, v, thr[s][code], cfg=cfg)
        has = np.zeros(len(df), bool)
        has[pos] = True                      # station/day-level bits only on readings that exist
        q[pos[fc]] |= CONST
        q[pos[fs]] |= SPIKE
        q[row_neigh & has] |= NEIGH
        q[row_decor & has] |= DECOR
        q[row_noise & has] |= NOISE
        q[row_shift & has] |= SHIFT
        q[row_level & has] |= LEVEL
        q[row_manual & has] |= MANUAL
        lim = cfg["range_max_spd_mps"] if s == "spd" else cfg["range_max_gust_mps"]
        q[pos[v >= lim]] |= RANGE
        qc[s] = q
    sm = pd.DataFrame({"code": np.arange(len(H)), "rho": rho, "joint_hours": nj, "p95_station": p95s,
                       "p95_neigh": p95r, "neigh_flag": mag | (decor if "DECOR" in cfg["mask_rules"] else False),
                       "neigh_decor": decor, "neigh_mag": mag, "neigh_mag_days": dflag.sum(1),
                       "mean_neigh_avail": np.where(nj > 0, navail.mean(1), 0),
                       "n_spd": nread, "month_median": med, "typical_median": typical,
                       "shift": shift, "level": level, "noise_days": nflag.sum(1),
                       "max_day_jumps": jumps.max(1)})
    return qc, sm[(nj > 0) | (nread > 0)], jdays


def apply_month(ym):
    st, nbr, thr, typical = _G["st"], _G["nbr"], _G["thr"], _G["typical"]
    code_of = dict(zip(st.stid, st.index))
    df = load_raw(month_files(ym), code_of)
    if df.empty:
        return ym, 0
    H, h0 = hourly_matrix(df, len(st), "mean")
    qc, sm, jdays = flag_month(df, H, h0, nbr, thr, typical)
    os.makedirs(os.path.join(MODEL, "jump_days"), exist_ok=True)
    jdays.insert(0, "stid", st.stid.to_numpy()[jdays.code])
    jdays.drop(columns="code").to_parquet(os.path.join(MODEL, "jump_days", f"{ym}.parquet"))
    y, m = ym.split("-")
    os.makedirs(os.path.join(FLAGS, y), exist_ok=True)
    keep = (qc["spd"] > 0) | (qc["gust"] > 0)
    out = pd.DataFrame({"stid": df.stid[keep].to_numpy(), "time_utc": df.time_utc[keep].to_numpy(),
                        "qc_spd": qc["spd"][keep], "qc_gust": qc["gust"][keep]})
    out.to_parquet(os.path.join(FLAGS, y, f"{m}.parquet"), index=False)
    sm.insert(0, "stid", st.stid.to_numpy()[sm.code])
    sm.insert(1, "month", ym)
    os.makedirs(os.path.join(MODEL, "station_month"), exist_ok=True)
    sm.drop(columns="code").to_parquet(os.path.join(MODEL, "station_month", f"{ym}.parquet"))
    # before/after histograms per network
    net = st.network_shortname.to_numpy()[df.code.to_numpy()]
    nets = _G["nets"]
    ni = pd.Series(np.arange(len(nets)), index=nets)[net].to_numpy()
    hist = {}
    for s, col in VARS.items():
        v = df[col].to_numpy(np.float64)
        ok = np.isfinite(v) & (v >= 0)
        b = np.searchsorted(HIST_EDGES, v[ok], side="right") - 1
        for tag, m_ in (("raw", ok), ("qc", ok & ((qc[s] & mask_bits()) == 0))):
            sel = m_[ok]
            hist[f"{s}_{tag}"] = np.bincount(ni[ok][sel] * (len(HIST_EDGES) - 1) + b[sel],
                                             minlength=len(nets) * (len(HIST_EDGES) - 1)).reshape(len(nets), -1)
        for r, bit in RULES.items():
            sel = (qc[s][ok] & bit) > 0
            hist[f"{s}_{r}"] = np.bincount(ni[ok][sel] * (len(HIST_EDGES) - 1) + b[sel],
                                           minlength=len(nets) * (len(HIST_EDGES) - 1)).reshape(len(nets), -1)
    os.makedirs(os.path.join(MODEL, "hist"), exist_ok=True)
    np.savez_compressed(os.path.join(MODEL, "hist", f"{ym}.npz"), networks=np.array(nets), **hist)
    return ym, int(keep.sum())


def step_apply(months, overwrite, workers):
    st = stations()
    nbr, _ = neighbour_table(st)
    thr = load_thresholds(st, CONFIG["spike_pct"])
    with open(os.path.join(MODEL, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)
    months = months or all_months()
    if not overwrite:
        months = [m for m in months if not os.path.exists(os.path.join(FLAGS, *m.split("-")) + ".parquet")]
    print(f"applying QC to {len(months)} months")
    g = {"st": st, "nbr": nbr, "thr": thr, "typical": load_station_level(st),
         "nets": sorted(st.network_shortname.unique())}
    with Pool(workers, initializer=_init, initargs=(g,)) as p:
        for i, (ym, n) in enumerate(p.imap_unordered(apply_month, months)):
            if (i + 1) % 20 == 0 or len(months) < 20:
                print(f"  {ym}: {n:,} flagged readings ({i + 1}/{len(months)})")


def manual_rows(df, cfg=CONFIG):
    """Rows of df inside a CONFIG manual_exclusions period."""
    out = np.zeros(len(df), bool)
    for e in cfg["manual_exclusions"]:
        m = (df.stid == e["stid"]).to_numpy().copy()
        if e.get("start"):
            m &= (df.time_utc >= pd.Timestamp(e["start"], tz="UTC")).to_numpy()
        if e.get("end"):
            m &= (df.time_utc < pd.Timestamp(e["end"], tz="UTC")).to_numpy()
        out |= m
    return out


def excluded_stations():
    p = os.path.join(MODEL, "station_exclusions.csv")
    return set(pd.read_csv(p).stid) if os.path.exists(p) else set()


def load_month_qc(year, month, columns=None, drop_excluded=True):
    """obs_raw for one month with QC applied: masked values -> NaN, excluded stations dropped
    (drop_excluded), bitmasks in qc_spd/qc_gust."""
    paths = sorted(glob.glob(os.path.join(DATA, "obs_raw", f"{year:04d}", f"{month:02d}", "*.parquet")))
    df = pa.concat_tables([pq.read_table(p, columns=columns) for p in paths]).to_pandas()
    fp = os.path.join(FLAGS, f"{year:04d}", f"{month:02d}.parquet")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"no QC flags for {year}-{month:02d}; run wind_qc.py apply")
    fl = pd.read_parquet(fp)
    df = df.merge(fl, on=["stid", "time_utc"], how="left")
    for s, col in VARS.items():
        df[f"qc_{s}"] = df[f"qc_{s}"].fillna(0).astype(QC_DTYPE)
        if col in df:
            df.loc[(df[f"qc_{s}"] & mask_bits()) > 0, col] = np.nan
    if drop_excluded:
        df = df[~df.stid.isin(excluded_stations())]
    return df


# ============================================================================ plots

INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
RAWC = "#b9b8b0"
RULE_COLOR = {"CONST": "#eb6834", "SPIKE": "#6250d6", "NOISE": "#e87ba4", "SHIFT": "#e34948",
              "LEVEL": "#1f1f1e", "RANGE": "#eda100", "MANUAL": "#86b6ef", "NEIGH": "#8a8a85", "DECOR": "#1baf7a"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


EVAL_MONTHS = ["2015-12", "2016-02", "2017-01", "2018-03", "2019-11", "2020-01",
               "2021-01", "2022-12", "2023-02", "2024-02", "2025-01", "2026-01"]
SPEED_EDGES = np.array([0, 5, 10, 15, 20, 25, 30, 40, np.inf])
SPEED_LABELS = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-40", "40+"]


def eval_month(ym):
    """Spike-rule variants on one month: flagged counts and neighbour corroboration by speed."""
    st, nbr = _G["st"], _G["nbr"]
    code_of = dict(zip(st.stid, st.index))
    df = load_raw(month_files(ym), code_of)
    H, h0 = hourly_matrix(df, len(st), "max")
    _, refmax, _ = neighbour_ref(H, nbr)
    pos, code, t, v = valid(df, VARS["spd"])
    nbmax = refmax[code, t // 60 - h0]
    corrob = nbmax >= 0.5 * v
    has_nb = np.isfinite(nbmax)
    b = np.searchsorted(SPEED_EDGES, v, side="right") - 1
    rows = []
    variants = {"none": np.zeros(len(v), bool)}
    for pct in (95, 99.9):
        th = _G["thr"][pct]["spd"][code]
        for mode in ("prior", "two_sided"):
            variants[f"{mode} p{pct:g}"] = rule_spike(code, t, v, th, mode=mode, floor=0)
    th = _G["thr"][CONFIG["spike_pct"]]["spd"][code]
    variants["CONFIG"] = rule_spike(code, t, v, th)
    for name, f in variants.items():
        for i, lab in enumerate(SPEED_LABELS):
            m = b == i
            rows.append({"month": ym, "variant": name, "speed": lab, "n": int(m.sum()),
                         "flagged": int((m & f).sum()),
                         "flagged_with_nb": int((m & f & has_nb).sum()),
                         "flagged_corrob": int((m & f & has_nb & corrob).sum()),
                         "unflagged_with_nb": int((m & ~f & has_nb).sum()),
                         "unflagged_corrob": int((m & ~f & has_nb & corrob).sum())})
    return pd.DataFrame(rows)


def step_evaluate(workers):
    plt = _plt()
    os.makedirs(IMAGES, exist_ok=True)
    st = stations()
    nbr, _ = neighbour_table(st)
    thr_all = {p: load_thresholds(st, p) for p in {95, 99.9, CONFIG["spike_pct"]}}
    thr = pd.read_parquet(os.path.join(MODEL, "thresholds.parquet"))

    # ---------------------------------------------------------------- 1. spike rule
    g = {"st": st, "nbr": nbr, "thr": thr_all}
    with Pool(min(workers, len(EVAL_MONTHS)), initializer=_init, initargs=(g,)) as p:
        ev = pd.concat(p.map(eval_month, EVAL_MONTHS))
    ev.to_csv(os.path.join(MODEL, "eval_spike_variants.csv"), index=False)
    agg = ev.groupby(["variant", "speed"], sort=False).sum(numeric_only=True).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    style(ax)
    s = thr[(thr["var"] == "spd") & (thr.n_pairs >= 500)]
    bins = np.arange(0, 25.5, 0.5)
    ax.axvspan(CONFIG["spike_floor_mps"], CONFIG["spike_cap_mps"], color="#e34948", alpha=0.12,
               label=f"threshold clipped to [{CONFIG['spike_floor_mps']:g}, {CONFIG['spike_cap_mps']:g}] m/s")
    for c, q in zip(SERIES, (95, 99, 99.9)):
        ax.hist(s[f"p{q:g}"], bins, histtype="step", lw=2, color=c, label=f"p{q:g}  (median {s[f'p{q:g}'].median():.1f} m/s)")
    ax.set_title("Station spike thresholds: percentile of |change|\nbetween consecutive sustained reports",
                 loc="left", fontsize=10, color=INK)
    ax.set_xlabel("m/s", color=MUTED, fontsize=9)
    ax.set_ylabel("stations", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    style(ax)
    x = np.arange(len(SPEED_LABELS))
    names = [n for n in agg.variant.unique() if n != "none"]
    cfg_lab = (f"CONFIG: two-sided, p{CONFIG['spike_pct']:g} clipped to [{CONFIG['spike_floor_mps']:g}, {CONFIG['spike_cap_mps']:g}] m/s"
               if CONFIG["spike_mode"] == "two_sided" else "CONFIG")
    labels = {"prior p95": "prior report, p95 (no floor)", "two_sided p95": "two-sided, p95 (no floor)",
              "prior p99.9": "prior report, p99.9 (no floor)", "two_sided p99.9": "two-sided, p99.9 (no floor)",
              "CONFIG": cfg_lab}
    cols = {"prior p95": SERIES[1], "two_sided p95": SERIES[3], "prior p99.9": "#f3a27f",
            "two_sided p99.9": "#f5cf80", "CONFIG": SERIES[0]}
    for n in names:
        a = agg[agg.variant == n]
        ax.plot(x, 100 * a.flagged / a.n.clip(lower=1), marker="o", ms=4, lw=2.4 if n == "CONFIG" else 1.4,
                color=cols[n], label=labels[n])
    ax.set_xticks(x, SPEED_LABELS)
    ax.set_yscale("log")
    ax.set_title("Share of readings flagged by the spike rule,\nby reading speed (12 winter months)",
                 loc="left", fontsize=10, color=INK)
    ax.set_xlabel("sustained reading (m/s)", color=MUTED, fontsize=9)
    ax.set_ylabel("% flagged (log)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)

    ax = axes[2]
    style(ax)
    a0 = agg[agg.variant == "none"]
    ax.plot(x, 100 * a0.unflagged_corrob / a0.unflagged_with_nb.clip(lower=1), color=RAWC, lw=3,
            label="all readings")
    for n in ("prior p95", "two_sided p95", "CONFIG"):
        a = agg[agg.variant == n]
        y = np.where(a.flagged_with_nb >= 20, 100 * a.flagged_corrob / a.flagged_with_nb.clip(lower=1), np.nan)
        ax.plot(x, y, marker="o", ms=4, lw=2.4 if n == "CONFIG" else 1.4, color=cols[n], label=f"flagged: {labels[n]}")
    ax.set_xticks(x, SPEED_LABELS)
    ax.set_ylim(0, 100)
    ax.set_title("Corroboration: % of readings where a neighbour (hourly max)\nreached >= half the value in the same hour"
                 "\n(a useful rule's flagged line sits far below grey)",
                 loc="left", fontsize=10, color=INK)
    ax.set_xlabel("sustained reading (m/s)", color=MUTED, fontsize=9)
    ax.set_ylabel("% corroborated", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("SPIKE rule calibration: does the rule remove readings that neighbours do NOT support?",
                 color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES, "spike_calibration.png"), dpi=120)
    plt.close(fig)

    # ---------------------------------------------------------------- 2. constant rule
    rh = np.load(os.path.join(MODEL, "const_runs_hist_spd.npy"))
    runs = pd.read_parquet(os.path.join(MODEL, "const_long_runs_spd.parquet"))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    style(ax)
    ax.grid(False)
    im = ax.imshow(np.log10(np.where(rh.T > 0, rh.T, np.nan)), origin="lower", aspect="auto", cmap="Blues")
    ax.set_xticks(np.arange(len(RUN_V_EDGES) - 1)[::2], [f"{e:g}" for e in RUN_V_EDGES[:-1][::2]])
    dl = ["0", "15m", "30m", "1h", "2h", "3h", "6h", "12h", "24h+"]
    ax.set_yticks(np.arange(len(RUN_D_EDGES) - 1), dl)
    cv = np.searchsorted(RUN_V_EDGES, CONFIG["const_min_mps"]) - 0.5
    cd = np.searchsorted(RUN_D_EDGES, CONFIG["const_min_minutes"]) - 0.5
    ax.plot([cv, cv, len(RUN_V_EDGES) - 1.5], [len(RUN_D_EDGES) - 1.5, cd, cd], color="#e34948", lw=2)
    ax.text(cv + 0.3, len(RUN_D_EDGES) - 1.8, "flagged", color="#e34948", fontsize=9)
    fig.colorbar(im, ax=ax, label="log10 runs")
    ax.set_xlabel("run value, sustained (m/s, bin start)", color=MUTED, fontsize=9)
    ax.set_ylabel("run duration (first to last identical report)", color=MUTED, fontsize=9)
    ax.set_title("Runs of >= 2 identical consecutive reports, all stations 2015-2026",
                 loc="left", fontsize=10, color=INK)
    ax = axes[1]
    style(ax)
    r = runs[(runs.n >= CONFIG["const_min_reports"])]
    flagged = (r.value >= CONFIG["const_min_mps"]) & (r.dur_min >= CONFIG["const_min_minutes"])
    by = (r[flagged].groupby("stid").agg(runs=("n", "size"), readings=("n", "sum"), value=("value", "median"))
          .sort_values("readings", ascending=False).head(25))
    ax.barh(np.arange(len(by))[::-1], by.readings, color=RULE_COLOR["CONST"], height=0.7)
    ax.set_yticks(np.arange(len(by))[::-1], [f"{i}  ({v:.0f} m/s)" for i, v in zip(by.index, by.value)], fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("readings flagged by CONST (log)", color=MUTED, fontsize=9)
    ax.set_title("Stations with the most CONST-flagged readings\n(median flagged value)", loc="left", fontsize=10, color=INK)
    fig.suptitle(f"CONST rule calibration: flag runs >= {CONFIG['const_min_mps']:g} m/s lasting "
                 f">= {CONFIG['const_min_minutes']} min and >= {CONFIG['const_min_reports']} reports",
                 color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES, "const_calibration.png"), dpi=120)
    plt.close(fig)

    # ---------------------------------------------------------------- 3. neighbour rule
    sm = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(MODEL, "station_month", "*.parquet"))])
    sm.to_parquet(os.path.join(MODEL, "station_month_all.parquet"))
    t = sm[sm.joint_hours >= CONFIG["neigh_min_hours"]].copy()
    t["ratio"] = t.p95_station / t.p95_neigh.clip(lower=0.5)
    c = CONFIG
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    style(ax)
    ax.scatter(t.rho, t.ratio, s=2, color=RAWC, alpha=0.25, lw=0, label="passed")
    for col, lab, colr in (("neigh_decor", "DECOR month (info)", RULE_COLOR["DECOR"]),
                           ("neigh_mag", "month with NEIGH days", RULE_COLOR["NEIGH"])):
        d = t[t[col]]
        ax.scatter(d.rho, d.ratio, s=4, color=colr, alpha=0.6, lw=0, label=f"{lab} ({len(d):,})")
    ax.plot([-1, c["neigh_rho_min"], c["neigh_rho_min"]], [c["neigh_decor_ratio"]] * 2 + [200], color=RULE_COLOR["DECOR"], lw=1)
    ax.set_yscale("log")
    ax.set_xlim(-0.8, 1.02)
    ax.set_xlabel("monthly Spearman rho, station vs neighbour median (hourly means)", color=MUTED, fontsize=9)
    ax.set_ylabel("station p95 / neighbour p95 (log)", color=MUTED, fontsize=9)
    ax.set_title(f"{len(t):,} testable station-months: {t.neigh_mag.mean() * 100:.2f}% have NEIGH days, "
                 f"{t.neigh_decor.mean() * 100:.2f}% are DECOR",
                 loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, markerscale=4, loc="upper left")
    ax = axes[1]
    style(ax)
    lo = t[(t.rho < c["neigh_rho_min"])]
    bins = np.logspace(-1.5, 2, 50)
    ax.hist(lo.ratio, bins, color=RAWC, label=f"rho < {c['neigh_rho_min']} ({len(lo):,} station-months)")
    ax.hist(lo.ratio[lo.neigh_decor], bins, color=RULE_COLOR["DECOR"], label="... and DECOR")
    ax.set_xscale("log")
    ax.axvline(c["neigh_decor_ratio"], color=RULE_COLOR["DECOR"], lw=1)
    ax.set_xlabel("station p95 / neighbour p95 (log)", color=MUTED, fontsize=9)
    ax.set_ylabel("station-months", color=MUTED, fontsize=9)
    ax.set_title("Decorrelated months (rho < threshold): most are CALMER than\ntheir neighbours (sheltered sites); never flagged",
                 loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[2]
    style(ax)
    per = t.groupby("stid").agg(months=("rho", "size"), flagged=("neigh_mag", "sum"),
                                decor=("neigh_decor", "sum"), mag_days=("neigh_mag_days", "sum"))
    per["share"] = per.flagged / per.months
    top = per[per.flagged > 0].sort_values("flagged", ascending=False).head(25)
    ax.barh(np.arange(len(top))[::-1], top.flagged, color=RULE_COLOR["NEIGH"], height=0.7)
    top = per[per.mag_days > 0].sort_values("mag_days", ascending=False).head(25)
    ax.barh(np.arange(len(top))[::-1], top.mag_days, color=RULE_COLOR["NEIGH"], height=0.7)
    ax.set_yticks(np.arange(len(top))[::-1], [f"{i}  ({r.decor} DECOR months)" for i, r in top.iterrows()], fontsize=7.5)
    ax.set_xlabel("NEIGH-flagged days", color=MUTED, fontsize=9)
    ax.set_title(f"Stations with the most NEIGH days ({int((per.mag_days > 0).sum())} stations have any)",
                 loc="left", fontsize=10, color=INK)
    fig.suptitle(f"NEIGH: day flagged if p95 > {c['neigh_mag_min_mps']:g} m/s and > {c['neigh_mag_ratio']:g}x neighbours.  "
                 f"DECOR (info only): month with rho < {c['neigh_rho_min']} and p95 > {c['neigh_decor_ratio']:g}x neighbours.  "
                 f"Neighbours <= {c['neigh_km']} km, ±{c['neigh_dz_m']} m",
                 color=INK, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES, "neigh_calibration.png"), dpi=120)
    plt.close(fig)
    per.sort_values("share", ascending=False).to_csv(os.path.join(MODEL, "neigh_flag_by_station.csv"))

    # ---------------------------------------------------------------- 4. before / after
    hs = None
    for f in sorted(glob.glob(os.path.join(MODEL, "hist", "*.npz"))):
        z = np.load(f)
        if hs is None:
            nets = list(z["networks"])
            hs = {k: z[k].astype(np.int64) for k in z.files if k != "networks"}
        else:
            for k in hs:
                hs[k] += z[k]
    order = np.argsort(-hs["spd_raw"].sum(1))
    show = [i for i in order if hs["spd_raw"][i].sum() > 0][:24]
    centers = HIST_EDGES[:-1]
    ncol = 6
    for s, lab in (("spd", "sustained"), ("gust", "gust")):
        fig, axes = plt.subplots(4, ncol, figsize=(3.2 * ncol, 9.5), sharex=True, sharey=True)
        for ax, i in zip(axes.flat, show):
            style(ax)
            raw, qc = hs[f"{s}_raw"][i], hs[f"{s}_qc"][i]
            if raw.sum() == 0:
                ax.set_visible(False)
                continue
            tot = raw.sum()
            ax.stairs(np.where(raw > 0, raw / tot / 0.5, np.nan), HIST_EDGES[:-1].tolist() + [100.5],
                      color=RAWC, lw=2.2, baseline=None, label="raw")
            ax.stairs(np.where(qc > 0, qc / tot / 0.5, np.nan), HIST_EDGES[:-1].tolist() + [100.5],
                      color=SERIES[0], lw=1.2, baseline=None, label="after QC")
            ax.set_yscale("log")
            ax.set_xlim(0, 101)
            ax.set_ylim(1e-9, 2)
            share = 100 * (1 - qc.sum() / tot)
            gt40r, gt40q = raw[centers >= 40].sum(), qc[centers >= 40].sum()
            ax.set_title(f"{nets[i]}  {share:.2f}% flagged\n>40 m/s: {gt40r:,} -> {gt40q:,}", fontsize=8, color=INK, loc="left")
        axes.flat[0].legend(frameon=False, fontsize=7.5)
        fig.supxlabel("m/s", color=MUTED, fontsize=9)
        fig.suptitle(f"Before / after QC, {lab} wind, 24 largest networks (density relative to raw total, log)",
                     color=INK, x=0.01, ha="left", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(IMAGES, f"before_after_{s}.png"), dpi=110)
        plt.close(fig)

    # summary table per network and rule
    rows = []
    for i, n in enumerate(nets):
        for s in VARS:
            raw = hs[f"{s}_raw"][i]
            if raw.sum() == 0:
                continue
            r = {"network": n, "var": s, "n": int(raw.sum()), "flagged_pct": round(100 * (1 - hs[f"{s}_qc"][i].sum() / raw.sum()), 3)}
            for rule in RULES:
                r[f"{rule}_pct"] = round(100 * hs[f"{s}_{rule}"][i].sum() / raw.sum(), 3)
            r["gt40_raw"] = int(raw[centers >= 40].sum())
            r["gt40_after"] = int(hs[f"{s}_qc"][i][centers >= 40].sum())
            r["gt25_raw"] = int(raw[centers >= 25].sum())
            r["gt25_after"] = int(hs[f"{s}_qc"][i][centers >= 25].sum())
            rows.append(r)
    summ = pd.DataFrame(rows).sort_values(["var", "n"], ascending=[False, False])
    summ.to_csv(os.path.join(MODEL, "summary_by_network.csv"), index=False)
    pd.set_option("display.width", 200)
    print(summ.head(30).to_string(index=False))
    tot = summ.groupby("var")[["n", "gt25_raw", "gt25_after", "gt40_raw", "gt40_after"]].sum()
    print(tot)

    # ---------------------------------------------------------------- 5. self-based rules
    c = CONFIG
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    style(ax)
    t = sm[sm.n_spd >= c["shift_min_reports"]].dropna(subset=["month_median", "typical_median"])
    x, y = t.typical_median.clip(lower=0.05), t.month_median.clip(lower=0.05)
    ax.scatter(x[~t["shift"]], y[~t["shift"]], s=2, color=RAWC, alpha=0.3, lw=0, label="passed")
    ax.scatter(x[t["shift"]], y[t["shift"]], s=8, color=RULE_COLOR["SHIFT"], lw=0, label=f"SHIFT ({int(t['shift'].sum())} station-months)")
    xx = np.logspace(-1.3, 2, 100)
    ax.plot(xx, np.maximum(c["shift_ratio"] * np.maximum(xx, 0.5), c["shift_min_mps"]), color=RULE_COLOR["SHIFT"], lw=1)
    ax.axvline(c["level_max_mps"], color=RULE_COLOR["LEVEL"], lw=1, ls="--", label=f"LEVEL: typical > {c['level_max_mps']:g} m/s")
    for sid, d in t[t["shift"]].groupby("stid"):
        ax.annotate(sid, (d.typical_median.iloc[0], d.month_median.max()), fontsize=7, color=INK, xytext=(3, 0), textcoords="offset points")
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel("station's typical monthly median (m/s, log)", color=MUTED, fontsize=9)
    ax.set_ylabel("this month's median (m/s, log)", color=MUTED, fontsize=9)
    ax.set_title(f"SHIFT: month median > {c['shift_ratio']:g}x typical and > {c['shift_min_mps']:g} m/s",
                 loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, markerscale=3, loc="upper left")

    ax = axes[1]
    style(ax)
    jd = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(MODEL, "jump_days", "*.parquet"))])
    share = jd.jumps / jd.pairs.clip(lower=1)
    jit = np.random.default_rng(0).uniform(-0.25, 0.25, len(jd))
    ax.scatter(jd.jumps[~jd.noise] + jit[~jd.noise], share[~jd.noise], s=3, color=RAWC, alpha=0.4, lw=0, label="passed")
    ax.scatter(jd.jumps[jd.noise] + jit[jd.noise], share[jd.noise], s=6, color=RULE_COLOR["NOISE"], lw=0,
               label=f"NOISE ({int(jd.noise.sum())} station-days, {jd[jd.noise].stid.nunique()} stations)")
    ax.axvline(c["noise_min_jumps"] - 0.5, color=RULE_COLOR["NOISE"], lw=1)
    ax.axhline(c["noise_min_share"], color=RULE_COLOR["NOISE"], lw=1)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel("jumps > spike threshold in the day (log)", color=MUTED, fontsize=9)
    ax.set_ylabel("share of the day's consecutive pairs (log)", color=MUTED, fontsize=9)
    ax.set_title(f"NOISE: >= {c['noise_min_jumps']} jumps and >= {c['noise_min_share']:.0%} of pairs\n"
                 f"(station-days with >= 1 jump)", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, markerscale=3, loc="lower right")

    ax = axes[2]
    style(ax)
    lvl = pd.read_parquet(os.path.join(MODEL, "station_level.parquet")).sort_values("typical_month_median", ascending=False).head(20)
    colr = [RULE_COLOR["LEVEL"] if v > c["level_max_mps"] else SERIES[0] for v in lvl.typical_month_median]
    ax.barh(np.arange(len(lvl))[::-1], lvl.typical_month_median, color=colr, height=0.7)
    ax.set_yticks(np.arange(len(lvl))[::-1], lvl.stid, fontsize=7.5)
    ax.axvline(c["level_max_mps"], color=RULE_COLOR["LEVEL"], lw=1, ls="--")
    ax.set_xlabel("typical monthly median, sustained (m/s)", color=MUTED, fontsize=9)
    ax.set_title("LEVEL: the 20 stations with the highest typical level", loc="left", fontsize=10, color=INK)
    fig.suptitle("Self-based station rules: each station judged against its own record", color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES, "self_rules_calibration.png"), dpi=120)
    plt.close(fig)

    # ---------------------------------------------------------------- 6. station summary / exclusion
    fl = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(FLAGS, "*", "*.parquet"))])
    q = fl.qc_spd.astype(np.int64)
    lvl = pd.read_parquet(os.path.join(MODEL, "station_level.parquet")).set_index("stid")
    summ = pd.DataFrame(index=st.stid)
    summ["network"] = st.set_index("stid").network_shortname
    summ["n_spd"] = lvl.n_spd.reindex(summ.index).fillna(0).astype(np.int64)
    summ = summ[summ.n_spd > 0]
    m = (q & auto_mask_bits()) > 0
    summ["masked"] = fl[m].groupby("stid").size().reindex(summ.index).fillna(0).astype(np.int64)
    summ["manual"] = fl[(q & MANUAL) > 0].groupby("stid").size().reindex(summ.index).fillna(0).astype(np.int64)
    for r in c["mask_rules"]:
        summ[r] = fl[(q & RULES[r]) > 0].groupby("stid").size().reindex(summ.index).fillna(0).astype(np.int64)
    summ["share"] = summ.masked / summ.n_spd
    summ["few_reports"] = summ.n_spd < c["station_exclude_min_reports"]
    summ["excluded"] = ~summ.few_reports & (summ.share >= c["station_exclude_share"])
    summ = summ.sort_values("share", ascending=False)
    summ.to_csv(os.path.join(MODEL, "station_summary.csv"))
    summ[summ.excluded].reset_index()[["stid", "network", "n_spd", "masked", "share"]].to_csv(
        os.path.join(MODEL, "station_exclusions.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    style(ax)
    ok = summ[~summ.few_reports]
    sh = ok.share[ok.share > 0]
    ax.hist(sh, np.logspace(-7, 0, 57), color=SERIES[0])
    ax.axvline(c["station_exclude_share"], color=RULE_COLOR["SHIFT"], lw=1.5)
    ax.set_xscale("log")
    lines = [f"{len(ok):,} stations with >= {c['station_exclude_min_reports']:,} sustained readings",
             f"{int((ok.share == 0).sum()):,} with nothing masked (not shown)"]
    lines += [f"share >= {th:g}: {int((ok.share >= th).sum())}" for th in (0.001, 0.01, 0.05, 0.1, 0.5)]
    ax.text(0.02, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=8.5, color=INK)
    ax.set_xlabel("share of the station's sustained readings masked (log)", color=MUTED, fontsize=9)
    ax.set_ylabel("stations", color=MUTED, fontsize=9)
    ax.set_title(f"Station exclusion: masked share >= {c['station_exclude_share']:.0%}  "
                 f"({int(summ.excluded.sum())} stations excluded)", loc="left", fontsize=10, color=INK)
    ax = axes[1]
    style(ax)
    top = ok.head(30)
    left = np.zeros(len(top))
    yy = np.arange(len(top))[::-1]
    for r in [r for r in c["mask_rules"] if r != "MANUAL"]:
        w = top[r] / top.n_spd
        ax.barh(yy, w, left=left, color=RULE_COLOR[r], height=0.72, label=r)
        left += w.to_numpy()
    ax.axvline(c["station_exclude_share"], color=RULE_COLOR["SHIFT"], lw=1.5)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 3)
    ax.set_yticks(yy, [f"{i} ({r.network})" + ("  EXCLUDED" if r.excluded else "") for i, r in top.iterrows()], fontsize=7.5)
    ax.set_xlabel("share of sustained readings flagged, by rule (stacked; a reading can carry several rules)",
                  color=MUTED, fontsize=9)
    ax.set_title("30 stations with the largest masked share", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES, "station_exclusion.png"), dpi=120)
    plt.close(fig)
    print(f"stations with any masked reading: {int((ok.masked > 0).sum())} of {len(ok)}; "
          f"excluded: {int(summ.excluded.sum())}")
    print(summ[~summ.few_reports].head(30).round(4).to_string())


GALLERY = [  # (stid, start, end, why it is here)
    ("MDTCA", "2016-09", "2018-03", "broken sensor for 14 months"),
    ("DDRIO", "2023-04", "2025-01", "DRI deployment reading 25-45 m/s for a year"),
    ("CLE", "2024-10", "2026-04", "USBR: nearly all readings 60-100 m/s"),
    ("OWY", "2023-10", "2025-01", "USBR: stuck near 25 m/s"),
    ("LALN2", "2016-01", "2017-01", "Las Vegas net: isolated ~100 m/s spikes"),
    ("OOSC1", "2018-10", "2019-04", "LOXWFO: multi-day bursts"),
    ("KWSIX", "2025-10", "2026-05", "Kirkwood ridge: suspect scaling"),
    ("SLIDE", "2016-10", "2017-04", "real: Slide Mtn 2941 m storm winds"),
    ("HVRTM", "2022-11", "2023-04", "real: Heavenly ridge 2992 m"),
    ("CGVC1", "2021-01", "2021-04", "real: sheltered canyon station, decorrelated but calm"),
]


def step_gallery():
    step_inspect([g[0] for g in GALLERY], None, None, "gallery",
                 ranges={g[0]: (g[1], g[2], g[3]) for g in GALLERY})


def step_inspect(stids, start, end, name, ranges=None):
    plt = _plt()
    os.makedirs(IMAGES, exist_ok=True)
    st = stations().set_index("stid")
    nb = pd.read_parquet(os.path.join(MODEL, "neighbours.parquet"))
    want = set(stids) | set(nb[nb.stid.isin(stids)].neighbour)
    filt = [("stid", "in", list(want))]
    obs = ds_read(filt)
    ranges = ranges or {s_: (start, end, "") for s_ in stids}
    nbr_of = nb.groupby("stid").neighbour.apply(list).to_dict()
    keep = np.zeros(len(obs), bool)
    for s_, (a, b, _) in ranges.items():
        m = obs.stid.isin([s_] + nbr_of.get(s_, [])).to_numpy().copy()
        if a:
            m &= (obs.time_utc >= pd.Timestamp(a, tz="UTC")).to_numpy()
        if b:
            m &= (obs.time_utc < pd.Timestamp(b, tz="UTC")).to_numpy()
        keep |= m
    obs = obs[keep]
    ms = sorted(obs.time_utc.dt.strftime("%Y/%m").unique())
    fl = pd.concat([pd.read_parquet(os.path.join(FLAGS, f"{m}.parquet"), filters=[("stid", "in", list(stids))])
                    for m in ms if os.path.exists(os.path.join(FLAGS, f"{m}.parquet"))])
    obs = obs.merge(fl, on=["stid", "time_utc"], how="left")
    obs[["qc_spd", "qc_gust"]] = obs[["qc_spd", "qc_gust"]].fillna(0).astype(QC_DTYPE)
    n = len(stids)
    fig, axes = plt.subplots(n, 1, figsize=(15, 2.6 * n), squeeze=False)
    for ax, s in zip(axes[:, 0], stids):
        style(ax)
        a_, b_, why = ranges[s]
        win = np.ones(len(obs), bool)
        if a_:
            win &= (obs.time_utc >= pd.Timestamp(a_, tz="UTC")).to_numpy()
        if b_:
            win &= (obs.time_utc < pd.Timestamp(b_, tz="UTC")).to_numpy()
        d = obs[win & (obs.stid == s).to_numpy()].dropna(subset=["wind_speed_mps"])
        nbs = nb[nb.stid == s].neighbour.tolist()
        if nbs:
            o = obs[win & obs.stid.isin(nbs).to_numpy()].dropna(subset=["wind_speed_mps"])
            ref = (o.assign(h=o.time_utc.dt.floor("h")).groupby(["h", "stid"]).wind_speed_mps.mean()
                   .groupby("h").median())
            ax.plot(ref.index, ref.values, color=RAWC, lw=0.6, label=f"neighbour median, hourly mean ({len(nbs)})", zorder=1)
        ok = (d.qc_spd & mask_bits()) == 0
        ax.plot(d.time_utc[ok], d.wind_speed_mps[ok], ".", ms=1.2, color=SERIES[0], label="passed", zorder=2)
        for r, bit in RULES.items():
            m = (d.qc_spd & bit) > 0
            if not m.any():
                continue
            if r in CONFIG["mask_rules"]:
                ax.plot(d.time_utc[m], d.wind_speed_mps[m], ".", ms=3, color=RULE_COLOR[r],
                        label=f"{r} ({m.sum():,})", zorder=3)
            else:  # informational rule: ring, data kept
                ax.plot(d.time_utc[m], d.wind_speed_mps[m], "o", ms=3, mfc="none", mew=0.5,
                        color=RULE_COLOR[r], label=f"{r} info, kept ({m.sum():,})", zorder=2.5)
        m = st.loc[s] if s in st.index else None
        lab = f"{s}  {m.network_shortname}, {m.elevation_m:.0f} m — {m['name']}" if m is not None else s
        pct = 100 * ((d.qc_spd & mask_bits()) > 0).mean() if len(d) else 0
        if s in excluded_stations():
            lab += "  [STATION EXCLUDED]"
        ax.set_title(f"{lab}   ({pct:.1f}% of sustained readings flagged)" + (f"  —  {why}" if why else ""),
                     loc="left", fontsize=9, color=INK)
        ax.set_ylabel("m/s", color=MUTED, fontsize=8)
        ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=5, markerscale=4)
    fig.tight_layout()
    out = os.path.join(IMAGES, f"inspect_{name or '_'.join(stids[:4])}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(out)


def ds_read(filters):
    import pyarrow.dataset as ds
    d = ds.dataset(os.path.join(DATA, "obs_raw"), format="parquet")
    import pyarrow.compute as pc
    expr = pc.field("stid").isin(filters[0][2])
    return d.to_table(columns=["stid", "time_utc", *VARS.values()], filter=expr).to_pandas()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["fit", "apply", "evaluate", "inspect", "gallery"])
    ap.add_argument("stids", nargs="*")
    ap.add_argument("--months", nargs="*")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--name")
    a = ap.parse_args()
    if a.step == "fit":
        step_fit(a.workers)
    elif a.step == "apply":
        step_apply(a.months, a.overwrite, a.workers)
    elif a.step == "evaluate":
        step_evaluate(a.workers)
    elif a.step == "gallery":
        step_gallery()
    else:
        step_inspect(a.stids, a.start, a.end, a.name)


if __name__ == "__main__":
    main()
