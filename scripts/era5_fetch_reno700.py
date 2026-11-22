#!/usr/bin/env python3
"""Download an hourly ERA5 700 hPa u/v series around the West Reno point, 2015-2025.

One CDS request per month (pressure levels, 700 hPa, u and v, all hours) on a 3x3 0.25 deg box centred
on the ERA5 grid point nearest the West Reno / Verdi cluster (39.52N, 119.92W -> 39.50N, 120.00W), so the
centre point can be compared with its neighbours. Files go to /mnt/d/ERA5/reno700/era5_ua700_YYYYMM.nc,
kept apart from the other files in /mnt/d/ERA5. Existing files are skipped; each file is written to a
.part name and renamed when complete, so an interrupted run can simply be restarted.

Needs ~/.cdsapirc and the accepted ERA5 licence on the CDS site.

Usage:
  python era5_fetch_reno700.py                    # 2015-01 .. 2025-12
  python era5_fetch_reno700.py --start 2020-01 --end 2020-12 --workers 4
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

OUT = "/mnt/d/ERA5/reno700"
AREA = [39.75, -120.25, 39.25, -119.75]   # N/W/S/E: 3x3 points around 39.50N, 120.00W


def target(m):
    return os.path.join(OUT, f"era5_ua700_{m:%Y%m}.nc")


def fetch(m):
    t = target(m)
    if os.path.exists(t):
        return m, "exists"
    import cdsapi
    c = cdsapi.Client(quiet=True, progress=False)
    days = pd.date_range(m, m + pd.offsets.MonthEnd(0), freq="D")
    req = {"product_type": ["reanalysis"], "variable": ["u_component_of_wind", "v_component_of_wind"],
           "pressure_level": ["700"], "year": [f"{m:%Y}"], "month": [f"{m:%m}"],
           "day": [f"{d:%d}" for d in days], "time": [f"{h:02d}:00" for h in range(24)],
           "area": AREA, "data_format": "netcdf", "download_format": "unarchived"}
    t0 = time.time()
    c.retrieve("reanalysis-era5-pressure-levels", req, t + ".part")
    os.replace(t + ".part", t)
    return m, f"wrote {os.path.getsize(t) / 1e3:.0f} kB in {(time.time() - t0) / 60:.1f} min"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01")
    ap.add_argument("--end", default="2025-12")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    months = list(pd.date_range(a.start, a.end, freq="MS"))
    todo = [m for m in months if not os.path.exists(target(m))]
    print(f"{len(months)} months, {len(months) - len(todo)} already present, {len(todo)} to fetch -> {OUT}",
          flush=True)
    t0, done, failed = time.time(), 0, []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fetch, m): m for m in todo}
        for f in as_completed(futs):
            m = futs[f]
            try:
                _, msg = f.result()
                done += 1
                print(f"[{done}/{len(todo)}] {m:%Y-%m} {msg}  (elapsed {(time.time() - t0) / 60:.0f} min)",
                      flush=True)
            except Exception as e:  # keep going; rerun picks up what's missing
                failed.append(m)
                print(f"FAILED {m:%Y-%m}: {e}", flush=True)
    print(f"done in {(time.time() - t0) / 60:.0f} min; {len(failed)} failed"
          + (": " + ", ".join(f"{m:%Y-%m}" for m in failed) if failed else ""), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
