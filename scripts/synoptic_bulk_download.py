#!/usr/bin/env python3
"""
Bulk-download hourly wind and pressure observations from the Synoptic Weather API
for surface stations (ASOS/AWOS + mesonets) in CA, NV, WA, and OR.

Output layout (under data/v2 by default):

  stations.parquet                              one row per station (lat, lon, elevation, network, ...)
  obs_hourly/YYYY/MM/XX.parquet  one row per station per UTC hour
  raw/YYYY-MM/XX_batchNNN.json.gz               untouched API responses (for reproducibility)
  manifest.jsonl                                one line per (state, month) run

Hourly fields per station:
  wind_speed_max_mps          max of 5-min wind_speed in the hour (sustained)
  wind_speed_mean_mps         mean wind_speed in the hour
  wind_direction_at_max_deg   wind_direction at the observation with the max speed
  wind_direction_mean_deg     circular mean wind_direction in the hour
  wind_gust_max_mps           max wind_gust in the hour
  pressure_mean_hpa           station pressure, mean in the hour
  sea_level_pressure_mean_hpa sea-level pressure, mean in the hour
  n_obs                       number of raw observations in the hour
  n_flagged_<var>             observations removed by Synoptic QC for each variable

Values that Synoptic QC flagged are excluded from the aggregates but counted
in n_flagged_<var>, so the QC decision stays visible.

Usage:
  python synoptic_bulk_download.py --start 2024-01-01 --end 2024-01-02
"""

import argparse
import csv
import datetime as dt
import gzip
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAVE_PYARROW = True
except ImportError:
    HAVE_PYARROW = False

API_BASE = "https://api.synopticdata.com/v2"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TOKEN_FILE = os.path.join(HERE, "synoptic_api.txt.txt")
DEFAULT_OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "data", "v2"))

STATES = ["CA", "NV", "WA", "OR"]

# ASOS/AWOS is network 1. The other networks are selected by their Synoptic CATEGORY
# (from /v2/networks): 1 = agricultural mesonets, 4 = federal/official surface networks,
# 6 = local/regional mesonets. Edit these to narrow or widen the station set.
# RAWS (fire-weather stations) is network 2, included explicitly.
ALWAYS_INCLUDE_NETWORK_IDS = {"1", "2"}
INCLUDE_NETWORK_CATEGORIES = {"1", "4", "6"}

# Variables requested from the timeseries endpoint (names verified against the live API).
API_VARS = ["wind_speed", "wind_direction", "wind_gust", "pressure", "sea_level_pressure"]

# Synoptic returns the same variable in several "sets". Prefer the primary set.
SET_PRIORITY = {"1": 0, "1d": 1}

# Physically plausible pressure range in Pa (metric output). Values outside it are treated
# as missing. Station pressure is wide because it depends on elevation.
PLAUSIBLE_PA = {
    "pressure": (30000.0, 110000.0),
    "sea_level_pressure": (87000.0, 109000.0),
}
IMPLAUSIBLE_COUNTS = defaultdict(int)
# QC flag arrays whose length did not match date_time (should stay zero).
MISALIGNED_QC_COUNTS = defaultdict(int)
# Values the API sent in an unexpected type, keyed by (variable, description).
ODD_VALUE_COUNTS = defaultdict(int)


def clean_value(x, base):
    """Return x as a float, or None if it is not a usable number.

    The API occasionally sends numbers as strings (e.g. "6.6"), and booleans (false) as
    placeholders. Strings that parse are converted; booleans and unparseable values are missing.
    Each unusual case is counted in ODD_VALUE_COUNTS so it shows up in the run output.
    """
    if x is None:
        return None
    if isinstance(x, bool):
        ODD_VALUE_COUNTS[(base, "boolean -> missing")] += 1
        return None
    if isinstance(x, (int, float)):
        return None if math.isnan(x) else float(x)
    if isinstance(x, str):
        try:
            value = float(x)
        except ValueError:
            ODD_VALUE_COUNTS[(base, f"unparseable string {x[:20]!r} -> missing")] += 1
            return None
        ODD_VALUE_COUNTS[(base, "numeric string -> float")] += 1
        return None if math.isnan(value) else value
    ODD_VALUE_COUNTS[(base, f"{type(x).__name__} -> missing")] += 1
    return None

STATIONS_PER_REQUEST = 50
REQUEST_SLEEP_SECONDS = 0.5
# After a pass with failed batches, retry them this many more times, waiting this long first.
BATCH_RETRY_PASSES = 2
BATCH_RETRY_WAIT_SECONDS = 60

# Stations never downloaded, because they return no data for this account in every month tested
# (2016-01 to 2016-03 and 2024-01). Add to this set to exclude more.
EXCLUDED_STIDS = {"OCRO", "PHL"}

# Synoptic's error text when none of the requested stations has data in the window
# (the same text it uses when the account lacks access, so the cause is not visible from here).
NO_DATA_MARKER = "No stations found for this request"


# ----------------------------------------------------------------------------
# API access
# ----------------------------------------------------------------------------

def load_token(path):
    with open(path, encoding="utf-8") as f:
        token = f.read().strip()
    if not token:
        sys.exit(f"Token file is empty: {path}")
    return token


def api_get(endpoint, params, token, retries=4):
    """GET an API endpoint and return (parsed_json, raw_bytes).

    The URL contains the token, so error messages below only report status codes
    and exception types, never the URL.
    """
    query = urllib.parse.urlencode({**params, "token": token})
    url = f"{API_BASE}/{endpoint}?{query}"
    last_error = "unknown"
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "socal-synoptic-downloader/1.0"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = resp.read()
            return json.loads(body), body
        except urllib.error.HTTPError as e:
            last_error = f"HTTP {e.code}"
            if e.code in (400, 401, 403, 404):
                break  # not retryable
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_error = type(e).__name__
        if attempt < retries:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{endpoint} request failed after {attempt} attempt(s): {last_error}")


def fetch_networks(token):
    data, _ = api_get("networks", {}, token)
    return {n["ID"]: n for n in data.get("MNET", [])}


def fetch_station_metadata(token, states):
    data, _ = api_get("stations/metadata", {"state": ",".join(states)}, token)
    code = data.get("SUMMARY", {}).get("RESPONSE_CODE")
    if code != 1:
        raise RuntimeError(f"metadata error: {data.get('SUMMARY', {}).get('RESPONSE_MESSAGE')}")
    return data.get("STATION", [])


# ----------------------------------------------------------------------------
# Station selection
# ----------------------------------------------------------------------------

def parse_iso(s):
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


def selected_network_ids(networks):
    ids = set(ALWAYS_INCLUDE_NETWORK_IDS)
    for net_id, net in networks.items():
        if str(net.get("CATEGORY")) in INCLUDE_NETWORK_CATEGORIES:
            ids.add(net_id)
    return ids


def to_float(value, scale=1.0):
    """Metadata numbers arrive as strings (or None). Return a float scaled by `scale`, or None."""
    if value is None or value == "":
        return None
    try:
        return float(value) * scale
    except (TypeError, ValueError):
        return None


def station_record(st, networks):
    """Flatten a metadata STATION object into one row. Elevation is converted to meters."""
    units = st.get("UNITS") or {}
    to_m = 0.3048 if units.get("elevation") == "ft" else 1.0
    elev = to_float(st.get("ELEVATION"), to_m)
    elev_dem = to_float(st.get("ELEV_DEM"), to_m)
    lat = to_float(st.get("LATITUDE"))
    lon = to_float(st.get("LONGITUDE"))
    por = st.get("PERIOD_OF_RECORD") or {}
    net = networks.get(str(st.get("MNET_ID")), {})
    return {
        "stid": st["STID"],
        "name": st.get("NAME"),
        "state": st.get("STATE"),
        "county": st.get("COUNTY"),
        "network_id": str(st.get("MNET_ID")),
        "network_shortname": net.get("SHORTNAME"),
        "network_category": net.get("CATEGORY"),
        "lat": lat,
        "lon": lon,
        "elevation_m": elev,
        "elevation_dem_m": elev_dem,
        "timezone": st.get("TIMEZONE"),
        "status": st.get("STATUS"),
        "por_start_utc": por.get("start"),
        "por_end_utc": por.get("end"),
    }


def overlaps(station, win_start, win_end):
    por_start = station.get("por_start_utc")
    por_end = station.get("por_end_utc")
    if not por_start or not por_end:
        return False
    return parse_iso(por_start) <= win_end and parse_iso(por_end) >= win_start


# ----------------------------------------------------------------------------
# Parsing and hourly aggregation
# ----------------------------------------------------------------------------

def station_observations(st):
    """Return a list of per-observation dicts for one station response object.

    Each dict: {"time": datetime, "ws", "wd", "gust", "p", "slp", "flags": {var: bool}}
    with None for missing or QC-flagged values.
    """
    obs = st.get("OBSERVATIONS") or {}
    times = obs.get("date_time") or []
    qc_all = st.get("QC") or {}

    # Pick one key per base variable, preferring the primary set.
    chosen = {}
    for key in obs:
        if key == "date_time":
            continue
        base, sep, set_name = key.rpartition("_set_")
        if not sep or base not in API_VARS:
            continue
        rank = SET_PRIORITY.get(set_name, 9)
        if base not in chosen or rank < chosen[base][0]:
            chosen[base] = (rank, key)

    series = {}
    for base, (_, key) in chosen.items():
        values = obs[key]
        flags = qc_all.get(key)
        if len(values) != len(times):
            continue  # malformed series; treat as missing
        if flags is not None and len(flags) != len(times):
            MISALIGNED_QC_COUNTS[base] += 1
            flags = None  # cannot align flags; leave values unflagged (counted and reported)
        series[base] = (values, flags)

    out = []
    for i, t in enumerate(times):
        row = {"time": parse_iso(t), "flags": {}}
        for base in API_VARS:
            if base not in series:
                row[base] = None
                row["flags"][base] = False
                continue
            values, flags = series[base]
            is_flagged = flags is not None and flags[i] is not None
            row["flags"][base] = is_flagged
            v = None if is_flagged else clean_value(values[i], base)
            if v is not None and base in PLAUSIBLE_PA and not (PLAUSIBLE_PA[base][0] <= v <= PLAUSIBLE_PA[base][1]):
                # Synoptic does not flag these; sensors report 0.0 for missing pressure.
                IMPLAUSIBLE_COUNTS[base] += 1
                v = None
            row[base] = v
        out.append(row)
    return out


def circular_mean_deg(directions):
    if not directions:
        return None
    s = sum(math.sin(math.radians(d)) for d in directions)
    c = sum(math.cos(math.radians(d)) for d in directions)
    if math.hypot(s, c) < 1e-9:
        return None
    return round(math.degrees(math.atan2(s, c)) % 360.0, 1)


def aggregate_hour(obs_in_hour):
    ws_vals = [o["wind_speed"] for o in obs_in_hour if o["wind_speed"] is not None]
    gust_vals = [o["wind_gust"] for o in obs_in_hour if o["wind_gust"] is not None]
    p_vals = [o["pressure"] for o in obs_in_hour if o["pressure"] is not None]
    slp_vals = [o["sea_level_pressure"] for o in obs_in_hour if o["sea_level_pressure"] is not None]
    wd_vals = [o["wind_direction"] for o in obs_in_hour if o["wind_direction"] is not None]

    wd_at_max = None
    if ws_vals:
        best = None
        for o in obs_in_hour:
            if o["wind_speed"] is None:
                continue
            if best is None or o["wind_speed"] > best["wind_speed"]:
                best = o
        wd_at_max = best["wind_direction"]

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    def hpa_mean(xs):
        # Synoptic metric output is in Pa; convert to hPa.
        return round(sum(xs) / len(xs) / 100.0, 2) if xs else None

    flag_counts = defaultdict(int)
    for o in obs_in_hour:
        for var, is_flagged in o["flags"].items():
            if is_flagged:
                flag_counts[var] += 1

    return {
        "wind_speed_max_mps": round(max(ws_vals), 3) if ws_vals else None,
        "wind_speed_mean_mps": mean(ws_vals),
        "wind_direction_at_max_deg": wd_at_max,
        "wind_direction_mean_deg": circular_mean_deg(wd_vals),
        "wind_gust_max_mps": round(max(gust_vals), 3) if gust_vals else None,
        "pressure_mean_hpa": hpa_mean(p_vals),
        "sea_level_pressure_mean_hpa": hpa_mean(slp_vals),
        "n_obs": len(obs_in_hour),
        "n_flagged_wind_speed": flag_counts["wind_speed"],
        "n_flagged_wind_direction": flag_counts["wind_direction"],
        "n_flagged_wind_gust": flag_counts["wind_gust"],
        "n_flagged_pressure": flag_counts["pressure"],
        "n_flagged_sea_level_pressure": flag_counts["sea_level_pressure"],
    }


def hourly_rows_for_station(st):
    buckets = defaultdict(list)
    for o in station_observations(st):
        hour = o["time"].replace(minute=0, second=0, microsecond=0)
        buckets[hour].append(o)
    rows = []
    for hour in sorted(buckets):
        row = {"stid": st["STID"], "time_utc": hour}
        row.update(aggregate_hour(buckets[hour]))
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

OBS_COLUMNS = [
    ("stid", "string"),
    ("time_utc", "timestamp"),
    ("wind_speed_max_mps", "float"),
    ("wind_speed_mean_mps", "float"),
    ("wind_direction_at_max_deg", "float"),
    ("wind_direction_mean_deg", "float"),
    ("wind_gust_max_mps", "float"),
    ("pressure_mean_hpa", "float"),
    ("sea_level_pressure_mean_hpa", "float"),
    ("n_obs", "int"),
    ("n_flagged_wind_speed", "int"),
    ("n_flagged_wind_direction", "int"),
    ("n_flagged_wind_gust", "int"),
    ("n_flagged_pressure", "int"),
    ("n_flagged_sea_level_pressure", "int"),
]

STATION_COLUMNS = [
    "stid", "name", "state", "county", "network_id", "network_shortname", "network_category",
    "lat", "lon", "elevation_m", "elevation_dem_m", "timezone", "status",
    "por_start_utc", "por_end_utc",
]


def write_table(rows, path, columns):
    """Write rows to `path` (a .parquet or .csv.gz path from output_ext()), atomically.

    The data goes to a .tmp file first and is renamed into place only when complete, so a
    file at its final name always holds a full result.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    if path.endswith(".parquet"):
        if not HAVE_PYARROW:
            sys.exit("pyarrow is required to write .parquet files")
        type_map = {"string": pa.string(), "float": pa.float64(), "int": pa.int64(),
                    "timestamp": pa.timestamp("s", tz="UTC")}
        if columns and isinstance(columns[0], tuple):
            schema = pa.schema([(name, type_map[t]) for name, t in columns])
            names = [name for name, _ in columns]
        else:
            schema = None
            names = columns
        arrays = {n: [r.get(n) for r in rows] for n in names}
        table = pa.table(arrays, schema=schema) if schema is not None else pa.table(arrays)
        pq.write_table(table, tmp, compression="zstd")
    else:
        names = [c[0] if isinstance(c, tuple) else c for c in columns]
        with gzip.open(tmp, "wt", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=names)
            writer.writeheader()
            for r in rows:
                writer.writerow({n: ("" if r.get(n) is None else r.get(n)) for n in names})
    os.replace(tmp, path)


def output_ext():
    return ".parquet" if HAVE_PYARROW else ".csv.gz"


def obs_path(out_dir, win_start, state):
    return os.path.join(
        out_dir, "obs_hourly",
        f"{win_start:%Y}", f"{win_start:%m}",
        f"{state}{output_ext()}",
    )


def append_manifest(out_dir, entry):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def save_raw(out_dir, win_start, state, batch_idx, suffix, body):
    path = os.path.join(out_dir, "raw", f"{win_start:%Y-%m}", f"{state}_batch{batch_idx:03d}{suffix}.json.gz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(body)


# ----------------------------------------------------------------------------
# Main download loop
# ----------------------------------------------------------------------------

def month_windows(start_date, end_date):
    """Yield (window_start, window_end) datetimes in UTC, one per calendar month, inclusive."""
    cur = start_date.replace(day=1)
    while cur <= end_date:
        nxt = (cur.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        w_start = max(cur, start_date)
        w_end = min(nxt - dt.timedelta(days=1), end_date)
        yield (
            dt.datetime(w_start.year, w_start.month, w_start.day, tzinfo=dt.timezone.utc),
            dt.datetime(w_end.year, w_end.month, w_end.day, 23, 59, tzinfo=dt.timezone.utc),
        )
        cur = nxt


def fetch_batch(token, stids, win_start, win_end):
    params = {
        "stid": ",".join(stids),
        "start": win_start.strftime("%Y%m%d%H%M"),
        "end": win_end.strftime("%Y%m%d%H%M"),
        "vars": ",".join(API_VARS),
        "units": "metric",
        "qc": "on",
        "qc_flags": "on",
        "qc_remove_data": "mark",
        "output": "json",
    }
    data, body = api_get("stations/timeseries", params, token)
    summary = data.get("SUMMARY", {})
    if summary.get("RESPONSE_CODE") != 1:
        raise RuntimeError(f"timeseries error: {summary.get('RESPONSE_MESSAGE')}")
    return data.get("STATION", []), body


def fetch_batch_split(token, stids, win_start, win_end):
    """Fetch a batch. If Synoptic reports that none of its stations has data, fetch each
    station alone so the stations with data are kept and the empty ones are recorded.

    Returns (stations, responses, no_data_stids). responses is a list of (suffix, body)
    for raw saving: suffix "" for a whole-batch response, "_<STID>" for a single-station one.
    Any other error is raised.
    """
    try:
        stations, body = fetch_batch(token, stids, win_start, win_end)
        return stations, [("", body)], []
    except RuntimeError as e:
        if NO_DATA_MARKER not in str(e):
            raise
        if len(stids) == 1:
            return [], [], list(stids)

    stations, responses, no_data = [], [], []
    for stid in stids:
        time.sleep(REQUEST_SLEEP_SECONDS)
        try:
            st, body = fetch_batch(token, [stid], win_start, win_end)
        except RuntimeError as e:
            if NO_DATA_MARKER not in str(e):
                raise
            no_data.append(stid)
            continue
        stations.extend(st)
        responses.append((f"_{stid}", body))
    return stations, responses, no_data


def run(args):
    start_date = dt.date.fromisoformat(args.start)
    end_date = dt.date.fromisoformat(args.end)
    if end_date < start_date:
        sys.exit("--end must be on or after --start")
    token = load_token(args.token_file)
    out_dir = args.out
    print(f"Output: {out_dir}  (format: {'parquet' if HAVE_PYARROW else 'csv.gz (pyarrow not installed)'})")

    networks = fetch_networks(token)
    net_ids = selected_network_ids(networks)
    print(f"Selected {len(net_ids)} networks (ASOS/AWOS + categories {sorted(INCLUDE_NETWORK_CATEGORIES)})")

    meta = fetch_station_metadata(token, args.states)
    stations = [station_record(st, networks) for st in meta if str(st.get("MNET_ID")) in net_ids]
    print(f"Metadata: {len(meta)} stations in {args.states}, {len(stations)} in selected networks")

    stations_path = os.path.join(out_dir, "stations" + output_ext())
    write_table(stations, stations_path, STATION_COLUMNS)
    print(f"Wrote station metadata: {stations_path}")

    any_failed = False
    for win_start, win_end in month_windows(start_date, end_date):
        for state in args.states:
            out_path = obs_path(out_dir, win_start, state)
            if os.path.exists(out_path) and not args.overwrite:
                print(f"skip existing {out_path}")
                continue

            state_stations = [s for s in stations
                              if s["state"] == state and overlaps(s, win_start, win_end)]
            excluded = sorted(s["stid"] for s in state_stations if s["stid"] in EXCLUDED_STIDS)
            stids = [s["stid"] for s in state_stations if s["stid"] not in EXCLUDED_STIDS]
            print(f"{win_start:%Y-%m} {state}: {len(stids)} stations"
                  + (f" ({len(excluded)} excluded: {', '.join(excluded)})" if excluded else ""))

            # Each batch is fetched; batches that fail are retried in later passes.
            # The output file is written only if every batch succeeded.
            batches = [(batch_idx, stids[i:i + STATIONS_PER_REQUEST])
                       for batch_idx, i in enumerate(range(0, len(stids), STATIONS_PER_REQUEST))]
            rows_by_batch = {}
            no_data_stids = []
            pending = batches
            last_error = {}
            for pass_no in range(1 + BATCH_RETRY_PASSES):
                if not pending:
                    break
                if pass_no > 0:
                    print(f"  retrying {len(pending)} failed batch(es), pass {pass_no}")
                    time.sleep(BATCH_RETRY_WAIT_SECONDS)
                still_failing = []
                for batch_idx, batch in pending:
                    try:
                        api_stations, responses, empty = fetch_batch_split(token, batch, win_start, win_end)
                        batch_rows = []
                        for st in api_stations:
                            batch_rows.extend(hourly_rows_for_station(st))
                        if args.save_raw:
                            for suffix, body in responses:
                                save_raw(out_dir, win_start, state, batch_idx, suffix, body)
                        rows_by_batch[batch_idx] = batch_rows
                        no_data_stids.extend(empty)
                        if empty:
                            print(f"  batch {batch_idx}: {len(empty)} station(s) have no data in this window: "
                                  + ", ".join(empty))
                    except RuntimeError as e:
                        last_error[batch_idx] = str(e)
                        still_failing.append((batch_idx, batch))
                        print(f"  batch {batch_idx} failed: {e}")
                    time.sleep(REQUEST_SLEEP_SECONDS)
                pending = still_failing

            failed = [{"batch": idx, "error": last_error[idx]} for idx, _ in pending]
            rows = [r for idx in sorted(rows_by_batch) for r in rows_by_batch[idx]]
            if failed:
                status = "failed"
                print(f"  NOT writing {out_path}: {len(failed)} batch(es) still failing after retries")
                any_failed = True
            else:
                status = "complete"
                write_table(rows, out_path, OBS_COLUMNS)
                print(f"  wrote {len(rows)} station-hours -> {out_path}")
            if IMPLAUSIBLE_COUNTS:
                print(f"  implausible pressure values set to missing so far: {dict(IMPLAUSIBLE_COUNTS)}")
                IMPLAUSIBLE_COUNTS.clear()
            if MISALIGNED_QC_COUNTS:
                print(f"  WARNING: QC arrays not aligned with timestamps: {dict(MISALIGNED_QC_COUNTS)}")
                MISALIGNED_QC_COUNTS.clear()
            if ODD_VALUE_COUNTS:
                print(f"  unusual values converted or dropped: {dict(ODD_VALUE_COUNTS)}")
                ODD_VALUE_COUNTS.clear()
            append_manifest(out_dir, {
                "state": state,
                "month": f"{win_start:%Y-%m}",
                "status": status,
                "stations_requested": len(stids),
                "stations_excluded": excluded,
                "stations_without_data": sorted(no_data_stids),
                "station_hours": len(rows),
                "failed_batches": failed,
                "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            })
    return 1 if any_failed else 0


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", required=True, help="First UTC date to download, YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="Last UTC date to download, YYYY-MM-DD (inclusive)")
    p.add_argument("--states", nargs="+", default=STATES, help="State codes (default: CA NV WA OR)")
    p.add_argument("--out", default=DEFAULT_OUT_DIR, help="Output root (default: data/v2)")
    p.add_argument("--token-file", default=DEFAULT_TOKEN_FILE, help="File containing the Synoptic API token")
    p.add_argument("--overwrite", action="store_true", help="Re-download state-months that already exist")
    p.add_argument("--no-raw", dest="save_raw", action="store_false", help="Do not keep raw API responses")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(parse_args(sys.argv[1:])))
