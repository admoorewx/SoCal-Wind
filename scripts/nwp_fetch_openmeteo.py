"""Fetch Open-Meteo Previous Runs point forecasts (lead days 0-7) for the verification stations.

Stations: the gradient stations (pressure_msl) and every station in station_sets.csv (wind_speed_10m);
stations in both get both variables. One request per station covering the whole period.

Outputs (under /mnt/d/NWP/open_meteo):
  raw/<model>/<STID>_<start>_<end>.json.gz   exact API response
  manifest_<model>.csv                       one row per station (status, cost, missing counts)
  points/<model>/points_<tag>.parquet        long table: valid_utc, stid, lead_day, lead_h, pressure_msl_hpa,
                                             wind_speed_10m_mps
  points/<model>/grid_cells.csv              station vs returned grid cell

Resumable: stations with a raw file and manifest status 'ok' are skipped.
Usage:
  python nwp_fetch_openmeteo.py --model gfs --start 2026-01-01 --end 2026-10-31 [--limit-stations N] [--build-only]
"""
import argparse, gzip, json, os, sys, time, urllib.error, urllib.parse, urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import nwp_config as C

ROOT = "/mnt/d/socal"
MAX_PER_HOUR = 4500      # stay under the 5,000/hour free limit
MAX_RETRIES = 6


def log(msg):
    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {msg}", flush=True)


def station_table():
    g = pd.read_parquet(f"{ROOT}/data/v2/derived/gradients/gradient_timeseries.parquet", columns=["station1", "station2"])
    grad = set(g.station1) | set(g.station2)
    sets = set(pd.read_csv(f"{ROOT}/data/v2/derived/models/station_sets.csv").stid)
    meta = pd.read_parquet(f"{ROOT}/data/v2/stations.parquet").drop_duplicates("stid").set_index("stid")
    rows = []
    for s in sorted(grad | sets, key=lambda s: (s not in grad, s)):
        role = "both" if s in grad and s in sets else ("gradient" if s in grad else "set")
        var = {"gradient": ["pressure_msl"], "set": ["wind_speed_10m"], "both": ["pressure_msl", "wind_speed_10m"]}[role]
        rows.append(dict(stid=s, role=role, variables=",".join(var), lat=meta.lat[s], lon=meta.lon[s],
                         elevation_m=meta.elevation_m[s]))
    return pd.DataFrame(rows)


def request(url, params):
    """GET with retries. Returns (body_bytes, weighted_cost, http_status)."""
    q = url + "?" + urllib.parse.urlencode(params)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(q, timeout=180) as r:
                return r.read(), float(r.headers.get("X-RateLimit-Cost") or "nan"), r.status
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:300]
            if e.code == 429:
                now = time.time()
                if "inute" in body:
                    wait = 65
                elif "ourly" in body:
                    wait = 3600 - now % 3600 + 60
                elif "aily" in body:
                    wait = 86400 - now % 86400 + 60
                else:
                    wait = 120
                log(f"  429 rate limit ({body}); sleeping {wait / 60:.1f} min")
                time.sleep(wait)
            elif e.code >= 500:
                log(f"  HTTP {e.code} ({body}); retry {attempt} in {30 * attempt}s")
                time.sleep(30 * attempt)
            else:
                raise RuntimeError(f"HTTP {e.code}: {body}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            log(f"  network error {e}; retry {attempt} in {30 * attempt}s")
            time.sleep(30 * attempt)
    raise RuntimeError(f"gave up after {MAX_RETRIES} attempts")


def fetch(args, cfg, stations, raw_dir, man_path):
    man = pd.read_csv(man_path) if os.path.exists(man_path) else pd.DataFrame(columns=["stid"])
    done = set(man.loc[man.get("status", pd.Series(dtype=str)) == "ok", "stid"])
    days = range(cfg["max_lead_day"] + 1)
    n_hours_expected = 24 * ((pd.Timestamp(args.end) - pd.Timestamp(args.start)).days + 1)
    todo = stations[~stations.stid.isin(done)]
    if args.limit_stations:
        todo = todo.head(args.limit_stations)
    log(f"{len(stations)} stations, {len(done)} already ok, fetching {len(todo)}")
    spent = 0.0
    t_start = time.time()
    for i, r in enumerate(todo.itertuples(), 1):
        fn = f"{raw_dir}/{r.stid}_{args.start.replace('-', '')}_{args.end.replace('-', '')}.json.gz"
        cols = [C.column_name(v, d) for v in r.variables.split(",") for d in days]
        params = dict(latitude=r.lat, longitude=r.lon, models=cfg["open_meteo_model"], hourly=",".join(cols),
                      start_date=args.start, end_date=args.end, timezone="GMT", wind_speed_unit="ms")
        url = C.PREVIOUS_RUNS_URL + "?" + urllib.parse.urlencode(params)
        row = dict(stid=r.stid, role=r.role, variables=r.variables, request_url=url, http_status=None,
                   weighted_cost=None, n_hours=None, n_missing_per_column=None,
                   fetched_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), status="error", error="")
        t0 = time.time()
        try:
            body, cost, code = request(C.PREVIOUS_RUNS_URL, params)
            d = json.loads(body)
            h = d["hourly"]
            missing_cols = [c for c in cols if c not in h]
            if missing_cols:
                raise RuntimeError(f"columns missing from response: {missing_cols}")
            if len(h["time"]) != n_hours_expected:
                raise RuntimeError(f"{len(h['time'])} hours, expected {n_hours_expected}")
            with gzip.open(fn + ".part", "wb") as f:
                f.write(body)
            os.replace(fn + ".part", fn)
            nmiss = {c: int(sum(v is None for v in h[c])) for c in cols}
            row.update(http_status=code, weighted_cost=cost, n_hours=len(h["time"]),
                       n_missing_per_column=json.dumps({k: v for k, v in nmiss.items() if v}), status="ok")
            spent += cost
            log(f"{i}/{len(todo)} {r.stid} ({r.role}) ok cost={cost:.2f} total={spent:.0f} "
                f"missing={sum(nmiss.values())}")
        except Exception as e:
            row["error"] = str(e)[:300]
            cost = 0
            log(f"{i}/{len(todo)} {r.stid} ERROR {e}")
        man = pd.concat([man[man.stid != r.stid], pd.DataFrame([row])], ignore_index=True)
        man.to_csv(man_path + ".part", index=False)
        os.replace(man_path + ".part", man_path)
        # pace to MAX_PER_HOUR on average
        wait = (cost or 1) * 3600 / MAX_PER_HOUR - (time.time() - t0)
        if wait > 0 and i < len(todo):
            time.sleep(wait)
    log(f"fetch finished: {spent:.0f} weighted calls in {(time.time() - t_start) / 60:.1f} min")


def build(args, cfg, stations, raw_dir, pts_dir, tag):
    days = range(cfg["max_lead_day"] + 1)
    tables, cells = [], []
    for r in stations.itertuples():
        fn = f"{raw_dir}/{r.stid}_{args.start.replace('-', '')}_{args.end.replace('-', '')}.json.gz"
        if not os.path.exists(fn):
            continue
        d = json.load(gzip.open(fn, "rt"))
        h = d["hourly"]
        valid = pd.to_datetime(h["time"]).tz_localize("UTC")
        n = len(valid)
        for k in days:
            out = dict(valid_utc=valid, stid=np.full(n, r.stid), lead_day=np.full(n, k, dtype="int8"),
                       lead_h=np.asarray(C.lead_hours(valid.hour.values, k, C.cycle_hours_for(cfg, k)), dtype="int16"))
            for v, colname in C.VARIABLES.items():
                c = C.column_name(v, k)
                out[colname] = (np.array([np.nan if x is None else x for x in h[c]], dtype="float32")
                                if c in h else np.full(n, np.nan, dtype="float32"))
            tables.append(pa.table(out))
        cells.append(dict(stid=r.stid, role=r.role, stn_lat=r.lat, stn_lon=r.lon, stn_elev_m=r.elevation_m,
                          grid_lat=d["latitude"], grid_lon=d["longitude"], grid_elev_m=d["elevation"],
                          note=cfg["grid_note"]))
    if not tables:
        log("no raw files to build from")
        return
    t = pa.concat_tables(tables)
    out = f"{pts_dir}/points_{tag}.parquet"
    pq.write_table(t, out + ".part", compression="zstd")
    os.replace(out + ".part", out)
    pd.DataFrame(cells).to_csv(f"{pts_dir}/grid_cells.csv", index=False)
    log(f"built {out}: {t.num_rows:,} rows from {len(cells)} stations ({os.path.getsize(out) / 1e6:.0f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(C.MODELS))
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--limit-stations", type=int, default=0, help="fetch at most N pending stations (testing)")
    ap.add_argument("--build-only", action="store_true", help="rebuild the points table from raw files, no calls")
    args = ap.parse_args()
    cfg = C.MODELS[args.model]
    if pd.Timestamp(args.start) < pd.Timestamp(cfg["archive_start"]).normalize():
        sys.exit(f"start {args.start} precedes the {args.model} archive start {cfg['archive_start']}")
    tag = f"{args.start.replace('-', '')}_{args.end.replace('-', '')}"
    raw_dir = f"{C.OPEN_METEO_ROOT}/raw/{args.model}"
    pts_dir = f"{C.OPEN_METEO_ROOT}/points/{args.model}"
    for d in (raw_dir, pts_dir, f"{C.OPEN_METEO_ROOT}/logs"):
        os.makedirs(d, exist_ok=True)
    man_path = f"{C.OPEN_METEO_ROOT}/manifest_{args.model}.csv"
    stations = station_table()
    if not args.build_only:
        fetch(args, cfg, stations, raw_dir, man_path)
    build(args, cfg, stations, raw_dir, pts_dir, tag)


if __name__ == "__main__":
    main()
