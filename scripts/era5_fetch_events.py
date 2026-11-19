#!/usr/bin/env python3
"""Download ERA5 hourly fields for the worst SFO-WMC+ gust-miss events (or any given dates).

One file per UTC day and dataset, all 24 hours, in /mnt/d/ERA5 using the folder's existing naming:
  era5_sfc_YYYYMMDD.nc   single levels: 2 m temperature, mean sea-level pressure, 10 m u/v wind,
                         instantaneous 10 m wind gust
  era5_ua_YYYYMMDD.nc    pressure levels 850 / 700 / 500 hPa: geopotential, temperature, u/v wind
Domain 30-50N, 135-110W, 0.25 deg (CDS area = N/W/S/E). Existing files are never overwritten (the folder
also holds files from other work); a file is written to a .part name and renamed when complete.

Days: by default the UTC days covering each event's peak hour and the 12 h before it, for the events
chosen by plot_sfo_wmc_stations.pick_events (6 worst SFO-WMC+ strong-gust misses, distinct by > 24 h).
Obs hours end at :56, so an event's ERA5 hour is the next full hour.

Needs ~/.cdsapirc and accepted ERA5 licences on the CDS site.

Usage:
  python era5_fetch_events.py                 # SFO-WMC+ events
  python era5_fetch_events.py 2024-12-16 ...  # explicit UTC days
"""
import argparse
import os
import sys

import pandas as pd

ROOT = "/mnt/d/ERA5"
AREA = [50, -135, 30, -110]
SFC = ["2m_temperature", "mean_sea_level_pressure", "10m_u_component_of_wind", "10m_v_component_of_wind",
       "instantaneous_10m_wind_gust"]
UA = ["geopotential", "temperature", "u_component_of_wind", "v_component_of_wind"]
LEVELS = ["500", "700", "850"]
LEAD_H = (12, 6, 0)


def event_hours():
    """ERA5 analysis hours (UTC) for each SFO-WMC+ event: peak, -6 h, -12 h."""
    from plot_sfo_wmc_stations import load, pick_events
    *_, M = load()
    out = []
    for r in pick_events(M):
        peak = r.time_utc.ceil("h")
        out.append((peak, [peak - pd.Timedelta(hours=h) for h in LEAD_H]))
    return out


def request(c, dataset, day, extra, target):
    if os.path.exists(target):
        print(f"  exists, skipped: {os.path.basename(target)}")
        return
    req = {"product_type": ["reanalysis"], "year": [f"{day:%Y}"], "month": [f"{day:%m}"], "day": [f"{day:%d}"],
           "time": [f"{h:02d}:00" for h in range(24)], "area": AREA, "data_format": "netcdf",
           "download_format": "unarchived", **extra}
    part = target + ".part"
    c.retrieve(dataset, req, part)
    os.replace(part, target)
    print(f"  wrote {os.path.basename(target)} ({os.path.getsize(target) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("days", nargs="*", help="UTC days YYYY-MM-DD (default: SFO-WMC+ events)")
    a = ap.parse_args()
    if a.days:
        days = sorted({pd.Timestamp(d) for d in a.days})
    else:
        ev = event_hours()
        for peak, hrs in ev:
            print(f"event peak {peak:%Y-%m-%d %H}Z, maps at " + ", ".join(f"{h:%d %H}Z" for h in hrs))
        days = sorted({h.tz_localize(None).normalize() for _, hrs in ev for h in hrs})
    print(f"{len(days)} days -> {ROOT}: " + ", ".join(f"{d:%Y-%m-%d}" for d in days))
    import cdsapi
    c = cdsapi.Client()
    for d in days:
        print(f"{d:%Y-%m-%d}")
        request(c, "reanalysis-era5-single-levels", d, {"variable": SFC},
                os.path.join(ROOT, f"era5_sfc_{d:%Y%m%d}.nc"))
        request(c, "reanalysis-era5-pressure-levels", d, {"variable": UA, "pressure_level": LEVELS},
                os.path.join(ROOT, f"era5_ua_{d:%Y%m%d}.nc"))
    print("done")


if __name__ == "__main__":
    sys.exit(main())
