"""NWP model table for the Open-Meteo verification pipeline. Every nwp_* script takes --model <key>.

Facts below are from the Phase 0 pilot (docs/nwp_verification_plan.md, "Pilot results", 2026-11-22).
"""

OPEN_METEO_ROOT = "/mnt/d/NWP/open_meteo"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# Free-tier limits (weighted calls). Weighted cost per request = max(1, n_columns/10 * n_days/14),
# where each lead day counts as one column. Measured via the X-RateLimit-Cost response header.
LIMITS = dict(per_minute=600, per_hour=5000, per_day=10000)

MODELS = {
    "gfs": dict(
        open_meteo_model="gfs_global",  # pressure/80 m/gust from ncep_gfs025, 10 m wind from ncep_gfs013
        api="previous_runs",
        max_lead_day=7,
        cycle_hours=6,                  # 00/06/12/18Z; lead_h = 24*N + (valid hour mod 6)
        native_steps=((120, 1), (None, 3)),  # GFS output: hourly to 120 h, 3-hourly beyond
        archive_start="2024-01-18T12:00",
        grid_note="returned lat/lon/elevation is the 0.13 deg wind cell; pressure_msl comes from the "
                  "nearest 0.25 deg cell, which the response does not report",
    ),
    # Added 2026-11-23 (Q2). Lead definition checked against single runs 2026-11-23 (KLAX, valid 2026-06-08):
    # days 0-5 use the latest 00/06/12/18Z run (lead 24N + hour mod 6); days 6-7 use only 00/12Z runs, because the
    # 06/18Z runs stop at 146 h (lead 24N + hour mod 12). Native output is 3-hourly to 144 h and 6-hourly beyond;
    # off-native hours are Open-Meteo interpolations of the stitched series (differ from the single run by <= 0.6 hPa
    # / 0.7 m/s at days 0-6, up to 2.1 hPa at day 7).
    "ecmwf": dict(
        open_meteo_model="ecmwf_ifs025",
        api="previous_runs",
        max_lead_day=7,
        cycle_hours=6,
        cycle_hours_by_day={6: 12, 7: 12},
        native_steps=((144, 3), (None, 6)),  # ECMWF open data: 3-hourly to 144 h, 6-hourly beyond
        archive_start="2024-02-03T00:00",  # previous_day1 empty 2024-02-01, present 2024-03-01; docs say 2024-02-03
        grid_note="ECMWF IFS 0.25 deg; returned lat/lon is the 0.25 deg cell; surface_pressure is derived by "
                  "Open-Meteo from pressure_msl and the requested elevation (elevation=0 -> equals msl)",
    ),
    # Added 2026-11-24. Lead definition checked against single runs (KLAX, valid 2026-06-08): day 0 = lead 0 of the
    # hourly run (an analysis, unlike GFS/ECMWF day 0 = 0-5 h); day 1 = the 00/06/12/18Z 48-h runs, lead 24 + hour
    # mod 6 (exact match every hour). No day 2. Output is hourly, so every hour is native (no native_steps).
    # Open-Meteo marks the model discontinued 2026-11-30. Single-runs archive keeps only every 3rd run.
    "hrrr": dict(
        open_meteo_model="ncep_hrrr_conus",
        api="previous_runs",
        max_lead_day=1,
        cycle_hours=6,
        cycle_hours_by_day={0: 1},
        archive_start="2025-01-01T00:00",  # 2025 available per the Q2 probe; earliest date not checked
        grid_note="HRRR 3 km CONUS; returned lat/lon is the model cell; pressure_msl is HRRR's own reduction "
                  "(MSLMA, MAPS method, per the AWS GRIB inventory)",
    ),
    # Lead definition checked 2026-11-24 against single runs (KLAX, valid 2026-10-28, 17 runs at full 240 h horizon;
    # the single-runs archive is empty for June/Aug 2026): every day 0-7 comes from the 00/12Z runs, lead 24N + hour
    # mod 12, exact match at every hour (wind within 0.05 m/s rounding). No 06/18Z runs exist. Native output is hourly
    # to 84 h, 3-hourly beyond; off-native hours are Open-Meteo Hermite interpolations (residual <= 0.05 after 84 h).
    # So days 0-3 are native at all hours, days 4-7 at 00/03/../21Z.
    "gem": dict(
        open_meteo_model="gem_global",
        api="previous_runs",
        max_lead_day=7,
        cycle_hours=12,                 # 00/12Z runs only; lead_h = 24*N + (valid hour mod 12)
        native_steps=((84, 1), (None, 3)),  # GEM GDPS: hourly to 84 h, 3-hourly beyond
        archive_start="2024-02-01T00:00",  # previous_day1 empty 2024-01-10, present 2024-02-01 (approximate)
        grid_note="GEM GDPS 15 km (cmc_gem_gdps_15km); returned lat/lon is the model cell; surface_pressure is "
                  "derived by Open-Meteo from pressure_msl and the requested elevation (elevation=0 -> equals msl)",
    ),
    # Added 2026-11-25. Wind only (Open-Meteo NBM has no pressure_msl), so no NBM-driven GBT. Lead definition checked
    # against single runs (KLAX, valid 2026-06-08; the single-runs archive keeps only 00/03/../21Z cycles, each to
    # ~262 h, first output at +1 h): days 1-7 = the hourly run exactly 24N h before valid (exact match where that run
    # is archived; off-3h hours are not interpolations of the stitched series, so they come from the unarchived
    # hourly runs); day 0 = lead 1 h. Native output hourly to 36 h, 3-hourly to ~192 h, so every lead (1, 24N <= 168)
    # is native. NBM is fresher than GFS/ECMWF (24N + 0-5 h) and GEM (24N + 0-11 h) at the same lead day.
    "nbm": dict(
        open_meteo_model="ncep_nbm_conus",
        api="previous_runs",
        variables=("wind_speed_10m",),
        max_lead_day=7,
        cycle_hours=1,                  # hourly runs; lead_h = 24*N
        lead_h_add={0: 1},              # no f000: day 0 is the +1 h forecast of the previous hour's run
        native_steps=((36, 1), (192, 3), (None, 6)),
        archive_start="2024-10-08T00:00",  # Historical Forecast table (plan doc); previous-runs start not checked
        grid_note="NBM CONUS 2.5 km; returned lat/lon is the model cell; wind only",
    ),
}

# Open-Meteo variable -> (column name in points table, request unit parameter)
VARIABLES = {
    "pressure_msl": "pressure_msl_hpa",
    "wind_speed_10m": "wind_speed_10m_mps",
}


def column_name(var, lead_day):
    """Open-Meteo hourly column for a variable at a lead day (day 0 has no suffix)."""
    return var if lead_day == 0 else f"{var}_previous_day{lead_day}"


def slp_correction_design(valid_utc):
    """Design matrix of the SLP-convention correction (nwp_pressure_calibration.py; 13 terms, same order as
    coefficients_<year>.json). valid_utc: pandas Series of tz-aware datetimes."""
    import numpy as np
    doy = 2 * np.pi * valid_utc.dt.day_of_year.to_numpy() / 365.25
    hr = 2 * np.pi * valid_utc.dt.hour.to_numpy() / 24
    return np.column_stack([np.ones(len(valid_utc)), np.cos(doy), np.sin(doy), np.cos(2 * doy), np.sin(2 * doy),
                            np.cos(hr), np.sin(hr), np.cos(2 * hr), np.sin(2 * hr),
                            np.cos(doy) * np.cos(hr), np.cos(doy) * np.sin(hr),
                            np.sin(doy) * np.cos(hr), np.sin(doy) * np.sin(hr)])


def cycle_hours_for(cfg, lead_day):
    """Spacing (h) of the runs that feed previous_day<lead_day> for a model config."""
    return cfg.get("cycle_hours_by_day", {}).get(lead_day, cfg["cycle_hours"])


def is_native(cfg, lead_h):
    """True where the forecast lead is an output step of the model itself (not an Open-Meteo interpolation).
    native_steps: ((max_lead_h, step_h), ..., (None, step_h)), checked in order. Models without the key: all True."""
    import numpy as np
    lead_h = np.asarray(lead_h)
    steps = cfg.get("native_steps")
    if not steps:
        return np.ones(lead_h.shape, bool)
    out = np.zeros(lead_h.shape, bool)
    done = np.zeros(lead_h.shape, bool)
    for max_h, step in steps:
        here = ~done & (lead_h <= max_h if max_h is not None else True)
        out |= here & (lead_h % step == 0)
        done |= here
    return out


def lead_hours(valid_hour, lead_day, cycle_hours=6):
    """Nominal forecast lead (h) for previous_dayN, from the pilot's run-matching check."""
    return 24 * lead_day + valid_hour % cycle_hours


def lead_hours_for(cfg, valid_hour, lead_day):
    """lead_hours with a model's per-day run spacing and any fixed offset (NBM day 0 = +1 h)."""
    return lead_hours(valid_hour, lead_day, cycle_hours_for(cfg, lead_day)) + cfg.get("lead_h_add", {}).get(lead_day, 0)
