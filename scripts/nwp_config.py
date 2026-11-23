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
        archive_start="2024-01-18T12:00",
        grid_note="returned lat/lon/elevation is the 0.13 deg wind cell; pressure_msl comes from the "
                  "nearest 0.25 deg cell, which the response does not report",
    ),
    # Added 2026-11-23 (Q2). Lead definition and cycle_hours: verify in Q3 (single-runs check as for gfs).
    "ecmwf": dict(
        open_meteo_model="ecmwf_ifs025",
        api="previous_runs",
        max_lead_day=7,
        cycle_hours=6,                  # 00/06/12/18Z runs exist in single-runs (checked 2026-06-01); verify in Q3
        archive_start="2024-02-03T00:00",  # previous_day1 empty 2024-02-01, present 2024-03-01; docs say 2024-02-03
        grid_note="ECMWF IFS 0.25 deg; returned lat/lon is the 0.25 deg cell; surface_pressure is derived by "
                  "Open-Meteo from pressure_msl and the requested elevation (elevation=0 -> equals msl)",
    ),
    "gem": dict(
        open_meteo_model="gem_global",
        api="previous_runs",
        max_lead_day=7,
        cycle_hours=12,                 # only 00/12Z runs in single-runs (06Z/18Z not available); verify in Q3
        archive_start="2024-02-01T00:00",  # previous_day1 empty 2024-01-10, present 2024-02-01 (approximate)
        grid_note="GEM GDPS 15 km (cmc_gem_gdps_15km); returned lat/lon is the model cell; surface_pressure is "
                  "derived by Open-Meteo from pressure_msl and the requested elevation (elevation=0 -> equals msl)",
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


def lead_hours(valid_hour, lead_day, cycle_hours=6):
    """Nominal forecast lead (h) for previous_dayN, from the pilot's run-matching check."""
    return 24 * lead_day + valid_hour % cycle_hours
