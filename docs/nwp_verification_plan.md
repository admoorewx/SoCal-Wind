# NWP verification plan: GBT vs numerical models on the 2026 holdout

Status: Phase 0 pilot done 2026-11-22 (GO). Scope revised 2026-11-22: sustained wind only, 2026 data only, no wind
direction (see "Scope revision" below). Next: Phase 1 fetcher.

## Goal and success criterion

**Question:** at each lead time, is the GBT forecast more accurate than the NWP model's own wind forecast? The GBT forecast is driven by the NWP model's sea-level pressure. Both are scored against the observed station-set targets.

**Success criterion (agreed, first target):** pooled GBT-NWP RMSE beats raw NWP RMSE in every lead-day bin, with the 95% bootstrap interval of the difference excluding 0. Sustained and gust get separate verdicts. The project counts as worthwhile if GBT wins against GFS.

## Decisions already made (2026-11-21)

| Topic | Decision |
|---|---|
| Model stays gradient-only | Yes. Upper-air inputs were ruled out to keep deployment simple. The flow-aloft limitation is documented (see the Jan 2025 case in wind_model_plan.md). |
| Data source | **Open-Meteo**, not raw GRIB. Open-Meteo handles point extraction and supplies `pressure_msl`. |
| Models | The pipeline must work for **any Open-Meteo model** through a config table. GFS comes first. |
| Lead times | Day 0 to **Day 7** at minimum. |
| Storage | All data on `/mnt/d` (see Storage). |
| Models scored | The frozen final `gbt_all` models in `data/v2/derived/models/gbt/models/` (trained 2015-2025). They are never retrained. |
| Holdout window | 2026-01-01 to 2026-10-31. Observations currently end on 2026-10-31. |

## What Open-Meteo provides (docs read 2026-11-21)

- **Previous Runs API.** Variables take a suffix `_previous_dayN` with N = 0..7. Day 0 is the latest run. `_previous_day1` is "the value predicted 24 hours before valid time", day 2 is 48 h before, and so on up to day 7. This gives **lead-day bins of hourly valid times**, not individual runs. Day 7 is the maximum. Most models are archived from Jan 2024. The Historical Forecast table gives GFS from 2021-03-23, ECMWF IFS 0.25° from 2024-02-03 and NBM from 2024-10-08.
- **Single Runs API** (`run=YYYY-MM-DDTHH:MM`). Gives the full horizon of one initialisation. This is the fallback if previous-runs is missing a variable or we want leads beyond Day 7.
- **Historical Forecast API.** Stitches only the first hours of each run, so it can't be used for lead-time verification.
- **Variables.** `pressure_msl`, `wind_speed_10m` (instantaneous), `wind_direction_10m`, `wind_gusts_10m` ("maximum of the preceding hour", which matches our hourly-max gust target better than an instantaneous gust).
- **Point selection.** `cell_selection` = `land` (default: a land cell of similar elevation, using a 90 m DEM), `sea` or `nearest`. `elevation` sets statistical downscaling, and `elevation=nan` turns it off. The docs don't say which variables downscaling changes. The response returns the coordinates of the chosen grid cell.
- **Free tier.** Limits are 600 calls/min, 5,000/hour, 10,000/day and 300,000/month. A request with more than 10 variables or more than 2 weeks of data counts as several calls. From the doc examples the weight is about (variables/10) × (weeks/2), with each factor at least 1. Several coordinates can go in one request, comma-separated, but how multi-location requests are counted isn't documented. The free tier is **non-commercial only**. Data is **CC BY 4.0**, so attribution is required.

### Risks to settle in the Phase 0 pilot, before writing the main pipeline

1. **Free access to the archive.** The pricing table marks Previous Runs as free, but the FAQ says historical APIs need a paid plan. The pilot must test real requests without a key.
2. **Gusts in Previous Runs.** No wind-gust variable appears in the previous-runs variable list. Test `wind_gusts_10m_previous_day1`. If it fails, take gusts from the Single Runs API: one 00Z run per day, leads cut into the same day bins.
3. **Pressure reduction is NOT necessarily solved by Open-Meteo.** `pressure_msl` is the model's own sea-level reduction at a grid cell, and its method isn't documented. Our models were trained on METAR SLP, which is reduced differently. That matters most at the 7 stations above 580 m: TPH 1,654 m, RNO 1,342 m, WMC 1,311 m, LMT 1,244 m, PMD 775 m, WJF 711 m and NID 696 m. Phase 2 checks this on 2024-2025 and applies a correction only if needed.
4. **Lead definition.** Confirm with a known case that `previous_dayN` behaves as documented (the value issued N×24 h before valid time). Record which run hours feed it.
5. **Downscaling and cell choice.** For one mountain station (e.g. TPH) and one RAWS site, compare `elevation` default vs `nan` and `cell_selection` land vs nearest, to see whether pressure or wind changes.
6. **Model availability.** HRRR is marked "Discontinued 2026-11-30", and ICON Global is too. Check whether their archives stay reachable after that date. If HRRR or ICON is wanted as an extra baseline, fetch it **before 2026-11-30**.

## Storage (all on /mnt/d)

```
/mnt/d/NWP/open_meteo/                       # new; separate from /mnt/d/ERA5 and /mnt/d/Verification (other work)
  raw/<model>/<stid>.json.gz                 # one cached API response per station: resume and audit
  points/<model>/points.parquet              # long table: valid_utc, stid, lead_day (0-7), pressure_msl_hpa,
                                             #   wind_speed_mph, grid_lat, grid_lon, grid_elev_m
  manifest_<model>.csv                       # stid, request url (no key), status, weighted calls, fetched_at
/mnt/d/socal/data/v2/derived/verification/   # derived tables (already on /mnt/d)
  holdout_targets_signfree.parquet           # targets for every signed set at every hour (Phase 3)
  <model>/forecasts.parquet                  # GBT-NWP, GBT-obs, raw-NWP and obs per set, valid time and lead day
  <model>/scores_*.csv, bootstrap_*.csv
/mnt/d/socal/images/verification/<model>/    # figures
```

## Model config (makes the pipeline work for multiple models)

A single table in `scripts/nwp_config.py`:

| key | open_meteo_model | api | max_lead_day | gusts_from | notes |
|---|---|---|---|---|---|
| gfs | GFS 0.25° global id (confirm the exact string in the pilot; NOT the "seamless" blend, which mixes in HRRR over the US) | previous_runs | 7 | previous_runs or single_runs | first model |
| ecmwf | ECMWF IFS 0.25° | previous_runs | 7 | ... | optional later |
| nbm | NCEP NBM CONUS | previous_runs | 7 | ... | strongest operational bar; archive from 2024-10-08 |
| hrrr | NCEP HRRR CONUS | previous_runs | 2 | ... | 0-48 h only; discontinued 2026-11-30 |

Every script takes `--model <key>`. Nothing downstream hard-codes GFS.

## Phases

### Phase 0: pilot (about 1 hour, a few hundred calls)
- Hit the Previous Runs API without a key for KLAX and KTPH plus 2 RAWS stations, January 2026, days 0-7.
- Settle risks 1-6 above. Record the exact model identifier string and the real weighted-call count per request.
- **Output:** a short "Pilot results" section appended to this file, and an exact call budget (see the estimate below).
- **Go / no-go:** if the archive needs a paid plan or a key variable is missing, stop and decide with the user: paid plan, Single Runs API, or AWS GRIB.

### Phase 1: fetcher, `scripts/nwp_fetch_openmeteo.py --model gfs --start 2026-01-01 --end 2026-10-31`
- Points:
  - The 22 gradient stations: `pressure_msl`, days 0-7.
  - Every station in `station_sets.csv` (425): `wind_speed_10m`, days 0-7. No direction (not used by targets or
    raw-NWP statistics). Gust and 80 m wind deferred (sustained first).
- Everything covers 2026-01-01 to 2026-10-31 only (revised 2026-11-22; was 2024 onward for pressure).
- One location per request, all lead days, 2-week chunks if needed. Throttle below the per-minute, hourly and daily limits.
- Resumable: a station is skipped if its raw cache exists and the manifest says OK. Responses are written to `.part` files and renamed when complete.
- Units: request mph and hPa, timezone GMT. Record the returned grid-cell lat, lon and elevation.

**Call budget (revised 2026-11-22, measured cost formula from the pilot):** 22 x 17.4 = ~380 (pressure) + 425 x 17.4 =
~7,400 (speed) = **~7,800 calls**, under one day's 10,000 limit. The 5,000/hour limit means about 2 hours of fetching.

### Phase 2: pressure check and correction, `scripts/nwp_check_pressure.py --model gfs` (2026, leave-one-month-out)
- For each gradient station, compare NWP `pressure_msl` at day 0 with METAR SLP (`slp1_hpa` / `slp2_hpa` in gradient_timeseries.parquet) at matched hours.
- Report bias, SD and dependence on temperature and season. Report the resulting **gradient** error per pair, by lead day.
- Revised 2026-11-22: there is no 2024-2025 NWP data, so the check uses 2026. It compares NWP pressure only with
  observed SLP and never sees wind observations.
- **Correction, leave-one-month-out:** each 2026 month is corrected with per-station coefficients fitted on the other 9
  months, so no hour is corrected with a fit that included it. Linear, optionally with a seasonal term.
- **Decision rule** (fixed now): use the correction only if the leave-one-month-out gradient RMSE is lower than
  uncorrected.
  - Correcting per station rather than per pair keeps all 24 GBT inputs consistent.
- Output: `verification/<model>/pressure_correction.json` (coefficients, or "none") and figures.

### Phase 3: sign-independent holdout targets, extending `scripts/build_targets.py` or adding a new script
- For every signed set and every 2026 hour, compute the target from that set's stations: sustained = set P75, gust = nearest-rank P90 with the 2-station floor and the 50%-gust-reporting rule. Same rules as training, **whatever the observed gradient sign**.
- Needed because the forecast sign chooses which model runs (Phase 4), so its target must exist even when the observed sign differs.
- Keep a column for the observed gradient sign so results can be split later.
- **DONE 2026-11-22:** `scripts/build_holdout_targets_signfree.py` (40 s). It reuses `build_targets.table_for` /
  `add_time`, so the rules are identical by construction.
  - Outputs: `derived/verification/holdout_targets_signfree_{sustained,gust}.parquet`, plus `_check.csv` and `_meta.json`.
  - Time grid: station1's matched METAR times, one per valid hour (`valid_utc` = obs time rounded up to the hour).
    Columns `gradient_hpa` (NaN if missing), `obs_sign`, `obs_sign_matches`.
  - Sustained: 183,742 qualifying set-hours, of which 81,654 have observed sign = set sign; 1,774 have no observed
    gradient and 2,596 have gradient 0.
  - Check: every same-sign row reproduces `targets/<var>_holdout.parquet` exactly (max diff 0.0). The only rows
    absent are 12 per variable at 2026-10-31 23:5x, which map to valid 2026-11-01 00Z, outside the window.
  - Known gap: KOTH METAR outage Apr-Jul 2026 (14-208 hours per month), so OTH-ACV- has 4,485 qualifying hours. No
    station1 time means no matched winds; this would need a re-match on an hourly grid if it matters.

### Phase 4: forecasts, `scripts/nwp_run_gbt.py --model gfs`
- For each valid hour and lead day, build all 22 gradients from corrected NWP pressure: the pair's own |grad| plus 19 signed other-pair gradients, hour sin/cos and day-of-year sin/cos. This gives the 24 inputs, in exactly the training column order. Missing values stay NaN.
- Run the frozen `gbt_all` models:
  - The **forecast sign** picks the + or − set model. Hours whose sign has no model are left out, as in training.
  - Run linear (speed) and logistic (event probability) for sustained and gust.
- Also run **GBT-obs**: the same models driven by the observed 2026 gradients, as the upper limit with perfect pressure.
- Also compute **raw NWP**: the set statistic (P75 sustained, P90 / k-th gust) of the NWP winds at the set's stations, using only stations that reported at that hour, the same as the obs target.
- Obs windows end at :51-:56. Match them to the next full hour (ceil), as in all earlier work.
- Before running, write SHA-256 checksums of the model files to `verification/model_checksums.txt`.
- **DONE 2026-11-23 (sustained):** `scripts/nwp_run_gbt.py --model gfs` (3 min) writes
  `derived/verification/gfs/forecasts_sustained.parquet`: 1,480,832 rows = 26 sets x hours x 8 lead days.
  - Columns: own-pair gradient obs / nwp_raw / nwp_corr; sel_* (does that sign pick this set); linear GBT
    (`gbt_*`) and logistic probabilities (`p15`/`p20`/`p30_*`) for each input; raw NWP P75 at reporting stations
    (`nwp_p75_rep`) and over all set stations (`nwp_p75_all`).
  - GBT runs at every row; score variant X on `sel_X & qualifies`.
  - SLP correction from `pressure_calibration/coefficients_2025.json` (shared design in `nwp_config.slp_correction_design`).
  - Checksums of the 95 sustained gbt_all models are in `verification/model_checksums.txt`.
  - **Feature check (`--check`):** the builder reproduces the training features for all 26 sets on 2025 exactly
    (0 feature diffs, 0 NaN mismatches, identical predictions). Valid hour V uses time features and other-pair
    gradients at V-1h, as in training.
  - Raw NWP uses exactly the target's reporting stations (n_nwp_rep == n_reporting on 100% of rows).
  - **First look, pooled, rows chosen by the corrected forecast sign (NOT the Phase 5 verdict):**
    - RMSE days 0-7, GBT-NWP corrected: 2.99 3.13 3.35 3.52 3.77 4.15 4.53 4.99 mph.
    - Raw NWP P75: 4.63 4.81 4.94 5.05 5.21 5.49 5.71 5.95.
    - GBT-NWP uncorrected: 4.09 ... 5.65. GBT-obs (perfect pressure): 2.71.
    - Raw NWP is biased low, -2.2 to -2.5 mph (instantaneous 10 m wind vs hourly-max obs; SAN-IPL- -10, LAX-DAG-
      -6.6). Its error SD is 3.89 at day 0 and 5.54 at day 7, still above GBT's 2.99 / 4.99.
    - Correlation with obs: GBT 0.86 -> 0.60, NWP 0.76 -> 0.49 (days 0 -> 7).
    - Forecast sign correct on 89% (day 0) to 77% (day 7) of rows.
    - Raw NWP beats GBT at day 0 in 5 of 26 sets: LAX-TPH+, RDD-SAC+/-, SAC-RBL+/-. At day 3 in 1 set.

### Phase 5: scoring, `scripts/nwp_verify.py --model gfs`
- **Speed:**
  - RMSE, MAE and bias by lead day (0-7), per set and pooled.
  - Skill score = 1 − MSE(GBT-NWP) / MSE(raw NWP).
  - Breakdown by observed-speed bin, to see strong winds.
- **Events:**
  - Brier score of GBT probabilities against raw-NWP yes/no exceedance.
  - Hit rate, false-alarm rate and CSI. The probability cut-off comes from the training out-of-fold results, not the holdout.
- **NWP pressure error (added 2026-11-22):** document the GFS SLP forecast error itself in 2026 by lead day, to
  separate NWP input error from GBT skill. Cover per-station bias, SD, RMSE and diurnal bias, and per-pair gradient
  RMSE, r and sign agreement (same metrics as `nwp_pressure_check_pilot.py`). Relate gradient error to the GBT-obs ->
  GBT-NWP loss by lead day. No pressure correction for now (user, 2026-11-22). Revisit only if needed.
- **Significance:** paired bootstrap by whole days, 1,000 resamples, giving 95% intervals on the RMSE difference.
- **Splits:** forecast sign right or wrong; season; set.
- **Figures:**
  - RMSE vs lead day (GBT-NWP, raw NWP, GBT-obs, climatology).
  - Skill by set.
  - Time series for the 3-5 largest 2026 wind events, in the style of `plot_event_timeseries.py`.

- **DONE 2026-11-23 (sustained, GFS):** `scripts/nwp_verify.py --model gfs` (35 s). Scores in
  `derived/verification/gfs/scores/`; figures in `images/verification/gfs/` (rmse_by_lead, skill_by_set,
  rmse_by_speed_bin_{all,offshore}, brier_by_lead, pressure_error_and_loss, events/<set>.png).
  - Main rows: qualifying set-hours chosen by the corrected forecast sign.
  - Extra baseline: per-set, per-lead bias-corrected raw NWP, with the bias taken from 2026 itself (an ORACLE,
    generous to NWP).
  - Offshore group (my labelling, to confirm with the user): LAX-BFL-, LAX-DAG-, LAX-PMD-, LAX-TPH-, LAX-WJF-,
    SAN-IPL-, SAN-LAS-, SBA-SMX-.
  - **VERDICT (agreed criterion: GBT-NWP RMSE significantly below raw NWP at every lead, 95% day-block bootstrap):
    MET**, for all sets and for the offshore group.
    - All sets, RMSE difference -1.64 (day 0) to -0.96 mph (day 7); MSE skill 0.58 -> 0.30.
    - Offshore: -2.51 to -1.12 mph; skill 0.68 -> 0.29.
    - Against the oracle bias-corrected NWP: significant at 8/8 leads for all sets (skill 0.24 -> 0.08) and 7/8
      offshore (day 7 CI -0.43..+0.20).
  - **Strong winds (lead 1, all sets), RMSE GBT / raw / bias-corrected:**
    - 20-30 mph: 4.75 / 8.79 / 5.94. >= 30 mph: 8.49 / 16.26 / 11.37.
    - Bias at >= 30 mph: GBT -7.1, raw NWP -14.6. Offshore >= 30 mph: GBT 6.59 RMSE (bias -5.7) vs raw 16.29 (bias -15.3).
    - Below 10 mph raw NWP is slightly better (2.34 vs 2.48): GBT is biased +1.2 in calm hours.
  - **Events:** GBT Brier is below raw and bias-corrected NWP yes/no at every lead for >= 15 and >= 20 mph.
    - CSI >= 20 at day 1: GBT 0.49, NWP 0.22, bias-corrected 0.35.
    - >= 30 mph is rare (base rate 0.8%). CSI day 1: GBT 0.19 vs NWP 0.04. The GBT Brier is about the same as
      climatology after day 4.
  - **By set:** raw NWP wins at short lead in RDD-SAC+/-, SAC-RBL+/- (Sacramento Valley) and LAX-TPH+ (day 0 only).
    SBA-SMX- skill is about 0 from day 2. The bias-corrected NWP also beats GBT in LAX-DAG+ and SAN-LAS+ at short lead.
  - **Pressure loss:** GBT-obs 2.74 -> GBT-corr 3.04 (day 0) .. 5.08 mph (day 7), on rows where both signs pick the
    set. Median corrected gradient RMSE 0.85 -> 2.08 hPa.
  - **Top-5 event days per set** (+/-24 h windows; rows chosen by sign):
    - All events RMSE GBT 4.30, raw 8.09, bias-corrected 5.62 (day 1). Offshore 4.75 / 10.63 / 7.12.
    - Offshore mean event peak: obs 32.8, GBT-obs 29.5, GBT day 1 27.0, day 3 26.7; raw NWP day 1 23.2, day 3 21.8.
    - **Caveat:** the SoCal offshore sets share about 6 event dates (2026-01-11, 10-16, 03-07, 02-04, 04-03, 01-17),
      so the event results rest on about 6 independent events.
    - Mar 7: GBT day 1 is worse than day 3 at LAX-PMD/TPH (22-24 vs 29-30). Possibly the March 2026 SLP-correction
      residual; check.

  - **D5/D7 event plots (2026-11-23):** `images/verification/gfs/events_d5d7/<set>.png`. Same event days as
    `events/` (days 1/3). `event_window_scores.csv` now holds leads 1, 3, 5 and 7.
    - Offshore event-window RMSE, GBT vs raw NWP: 4.6 vs 9.9 (day 1), 5.8 vs 10.9 (day 3), 6.4 vs 11.9 (day 5),
      8.7 vs 12.7 mph (day 7).
    - Raw NWP has the lower window RMSE in 0/40 offshore windows at day 5 and 1/40 at day 7; 19/90 and 17/90 in
      the other sets.
    - Mean offshore peak: GBT 26.1 (day 5) and 25.2 (day 7) vs obs 32.9; raw NWP 22.3 and 20.9.
    - The day-7 GBT lines show 6-hourly steps. previous_dayN switches source run every 6 h (lead 24N..24N+5), so
      run-to-run jumps appear as steps.

### Phase 6: write-up
- Verdict against the success criterion.
- The error split: GBT-obs → GBT-NWP shows the loss from pressure-forecast error.
- Caveats:
  - Instantaneous vs hourly-max sustained wind.
  - Open-Meteo grid-cell selection.
  - The Jan 2025 flow-aloft limitation.
  - CC BY 4.0 attribution for Open-Meteo.
- Optionally repeat for NBM or ECMWF using `--model`.

## Rules for keeping the holdout clean
- Every choice is made on training out-of-fold results or fixed in advance: probability cut-offs, lead bins, cell
  selection. The pressure correction uses 2026 pressure only (never wind obs), cross-fitted by month (Phase 2).
- 2026 is used once, in Phase 5, with the criterion above.
- If something has to change after looking at 2026, log it here as a post-hoc change.

## Pilot results (Phase 0, run 2026-11-22). Verdict: GO

User findings (2026-11-22): Previous Runs is free (no subscription); the free tier's non-commercial terms are fine
(pure research). Open-Meteo does not document its `pressure_msl` reduction; `surface_pressure` is not what we want.
Scope: **sustained wind first**; gusts later, with 80 m wind as a possible gust-potential proxy.

Pilot (keyless requests; raw responses and `pilot_log.csv` in `/mnt/d/NWP/open_meteo/pilot/`):
1. **Access:** HTTP 200 with no key. No key is needed. The user has a key in reserve (store it in a file, never in chat).
2. **Model ID: `gfs_global`.** It is itself a blend of two grids: `pressure_msl`, 80 m wind and gusts are identical to
   `ncep_gfs025` (0.25 deg); 10 m wind comes from `ncep_gfs013` (returned cell e.g. 33.915, -118.359 for KLAX).
   `ncep_gfs025` has no 10 m wind; `ncep_gfs013` has no pressure. `gfs_seamless` differs (HRRR blend), so don't use it.
3. **Variables under `_previous_dayN`:** pressure_msl, wind_speed_10m, wind_direction_10m, **wind_gusts_10m (it IS
   available, contrary to the variable list)**, wind_speed_80m. Jan 2026 at 4 stations: 0% missing, all 40 columns.
4. **Lead definition (checked against Single Runs, June 2026):** day N comes from the latest run initialised at or
   before valid - 24N h, so lead = 24N + (valid hour mod 6) h. Day 0 = 0-5 h (near-analysis), day 1 = 24-29 h, ..., day
   7 = 168-173 h. Cycles are 00/06/12/18Z.
5. **Cell selection and downscaling:** `cell_selection` land vs nearest and `elevation` default vs nan give
   **identical** pressure and wind values at KTPH and JULC1 (only the reported grid elevation changes). Use the defaults.
6. **Archive start:** `previous_day1` from valid 2024-01-19 12Z (day 7 from 2024-01-25 12Z). Phase 2 uses 2024-01-26
   onward. **Single Runs** for GFS exist only from partway through 2026 (2026-01-04 is not available; 2026-06-01 is),
   so it can't replace Previous Runs for the holdout.
7. **Cost (measured from the `X-RateLimit-Cost` header):** cost = max(1, n_vars/10 x n_days/14), not rounded up.
   Each lead day counts as a variable. Requests take 1-2 s.

**Call budget (GFS, measured formula):**

| Fetch | Stations | Calls |
|---|---|---|
| pressure, days 0-7, 2024-01-18 to 2026-10-31 | 22 gradient | ~1,300 |
| speed + dir, days 0-7, 2026 holdout | 425 (all set stations) | ~14,800 |
| + gust | 425 | +7,400 |
| + 80 m speed | 425 | +7,400 |

Limits: 10,000/day and 5,000/hour, so speed + dir + pressure is about 2 days of fetching.

## Scope revision (2026-11-22, user)
- Data from **2026-01-01** onward only. No 2024-2025 fetch.
- The pressure check uses 2026, cross-fitted by month (Phase 2).
- **Sustained wind only:** fetch `wind_speed_10m` + `pressure_msl`. No direction. Gust and 80 m wind come later.
- The Phase 4 and 5 gust items wait until the gust phase.

## Pressure check pilot, Jan 2025 (2026-11-22)
Script `scripts/nwp_pressure_check_pilot.py` (39 weighted calls; raw data in `/mnt/d/NWP/open_meteo/pilot/pressure_check/`).
Outputs are in `data/v2/derived/verification/gfs/pressure_check_pilot/` and `images/verification/gfs/pressure_check_pilot/`.
GFS `pressure_msl` minus METAR SLP, obs matched to the next full hour.
- **Low stations (< 160 m):** day-0 bias -1.1 to +0.6 hPa, SD 0.4-0.6, r >= 0.99. Diurnal bias range < 0.9 hPa.
- **High stations (> 580 m):** SD 1.1-2.2 hPa, diurnal bias range 1.3-3.1 hPa (LMT 3.1, TPH 2.8, RNO 2.4).
  NWP is too high around 09-12Z (night) and too low around 21-00Z (afternoon). This matches the expected signature of
  a reduction that uses the current temperature, whereas METAR SLP uses a 12-h mean.
- **Day-0 gradients:** median RMSE 1.14 hPa, r 0.94, skill vs the obs-gradient variance 0.84.
  - Worst pairs: LAX-WJF 2.3, LAX-TPH 2.3, SFO-WMC 2.0 (obs SD 3.4-5.1).
  - Weak-gradient pairs suffer from bias: SBA-SMX skill -0.18; MFR-LMT 0.02; SAC-RDD sign agreement 0.62.
- **By lead day (median RMSE):** 1.14, 1.54, 1.73, 1.69, 1.67, 1.99, 2.28, 2.38 hPa for days 0-7. The common low drift
  with lead (about -0.5 hPa by day 3) largely cancels in the gradients.
- **Correction potential** (fit on Jan 1-15 day 0, test on Jan 16-31, median gradient RMSE):
  - Day 0: 1.11 raw, 0.99 with a station offset, 0.84 with a station x hour offset.
  - Day 3: 1.65, 1.62, 1.54.
  - Days 5-7: no gain.
  - Not robust: SAN-LAS worsens from 0.82 to 1.66. The bias is regime-dependent, not a stable offset.
- **Reading:** the values are reasonable. Random NWP error, not the reduction method, dominates beyond day 1.
  Station x hour correction matters only at day 0-2 and on weak-gradient pairs. Only one winter month was checked.

## Fetch and data QC, GFS 2026 (2026-11-22/23)
- **Fetch:** 437/437 stations ok, 7,765 weighted calls, 1 h 56 min. Two SSL timeouts, both recovered on retry.
  The points table has 25.5 M rows (51 MB). QC script: `scripts/nwp_qc_points.py`; outputs in
  `derived/verification/gfs/qc/`.
- **Clean:**
  - 0 missing values.
  - Pressure 988-1043 hPa; wind 0-27 m/s.
  - No stuck values (longest identical run 8 h).
  - Lead days are distinct: no hour with day 0 = day 1 at 90% or more of stations. Wind climatology is flat across
    leads; pressure drifts about 1 hPa low by day 7.
  - Grid cells 0.2-13.9 km from the station. 2 with |dz| > 300 m: WAPC1F1, where the station metadata says 2,979 m
    but the DEM says 1,602 (a metadata error), and WMSC1.
  - 74 set stations have no 2026 obs (mostly inactive). They are harmless because raw NWP uses only reporting stations.
- **Day-0 wind vs matched max sustained:** median r 0.62 (5-95%: 0.22-0.81); NWP/obs mean ratio 0.94.
  15 stations have r < 0.2, mostly Oregon valley RAWS and KSXT (r -0.24).
- **FINDING: seasonal SLP-reduction bias at the high stations, much larger than the Jan 2025 pilot showed.**
  Day-0 NWP minus METAR bias:
  - TPH: -1.6 (Jan) to -6.6 hPa (Aug).
  - WMC: -0.7 to -5.7. RNO: +0.3 to -4.4. LMT: +0.3 to -3.2. LAS: -1.3 to -3.6.
  - LAX: -0.2 to -0.4 all year.
  This fits GFS reducing with the current (hot) temperature, whereas METAR SLP uses a 12-h mean (and the
  plateau correction).
  Resulting day-0 gradient bias, annual / JJA, with the JJA obs gradient SD in brackets:
  - LAX-TPH +3.9 / +5.8 [3.0]; SFO-WMC +3.1 / +5.1 [3.0]; SAC-RNO +1.8 / +3.7 [2.2]; MFR-LMT +1.4 / +2.9 [1.7].
  - JJA sign agreement: SAC-RNO 0.47, MFR-LMT 0.45.
  Every model takes all 22 gradients as inputs, so this reaches all sets. The no-correction decision needs
  revisiting with the user before Phase 4.

## SLP-convention calibration + station-pressure check (2026-11-23)
Script `scripts/nwp_pressure_calibration.py --model gfs --fit-year 2025` (172 weighted calls). It fetches day-0
pressure_msl, surface_pressure (elevation = station) and temperature_2m for the 22 gradient stations in 2025.
Raw data in `/mnt/d/NWP/open_meteo/raw/gfs_calib/`; outputs in `derived/verification/gfs/pressure_calibration/`;
figure `images/verification/gfs/pressure_calibration/high_station_bias_by_month.png`.
- **Open-Meteo `surface_pressure` is derived, not a model field.** It equals pressure_msl at elevation=0 and changes
  with the `elevation` parameter. It reproduces the hypsometric formula with the CURRENT 2 m temperature
  (lapse-rate downscaled) exactly: 998.5 hPa, h=1653.8 m, T=36.3 C -> 834.4 vs 834.5.
- **Station-pressure check (2025):** Open-Meteo pressure at station height vs observed station pressure
  (obs_hourly pressure_mean_hpa).
  - Bias -1.2 to +0.2 hPa; SD 0.3-1.1.
  - High stations: seasonal range 0.6-1.4 hPa, against 3.0-6.2 for SLP. Diurnal range 0.9-2.1 hPa, against 1.8-4.7.
  - **So at station level GFS is close to the observations; the SLP gap is mostly the reduction convention**
    (current temperature vs METAR's 12-h mean + plateau correction).
- **2025 SLP bias, high stations:** TPH -0.7 (Jan) to -6.0 (Aug); WMC +1.0 to -5.2; RNO +1.3 to -4.2; LMT +1.7 to -3.3.
  2026 shows the same pattern (TPH -6.6 in Aug).
- **Correction (form fixed in advance):** per station, err = pressure_msl - METAR SLP ~ 2 seasonal + 2 diurnal
  harmonics + first seasonal x first diurnal interaction (13 terms). Fitted on 2025; coefficients in
  `coefficients_2025.json`; applied to all lead days.
  - **2025 leave-one-month-out CV, day-0 gradient RMSE:** median 1.07 -> 0.87 hPa, 21/21 pairs improved.
    LAX-TPH 4.23 -> 1.85; SFO-WMC 3.65 -> 1.76; SAC-RNO 2.79 -> 1.57.
  - **Decision rule met: USE the correction.**
- **Verification period, 2026 pressure only (report, nothing tuned):**
  - Median gradient RMSE by lead day: raw 1.11 / 1.31 / 1.42 / 1.63 / 1.78 / 2.02 / 2.16 / 2.49; corrected
    0.84 / 1.02 / 1.18 / 1.35 / 1.49 / 1.67 / 1.87 / 2.04 (days 0-7). Median |bias| 0.64 -> 0.16 at day 0.
  - LAX-TPH day 0: bias +3.86 -> +0.41, RMSE 4.70 -> 1.99.
  - Residual: March 2026 is about -1.1 to -2.6 hPa at all high stations after correction (an anomalous month). A
    climatological correction can't follow year-to-year temperature anomalies. A temperature-based correction (fit
    on 2025, needs 2026 temperature_2m at all leads, ~380 calls) is a possible refinement.

### Free METAR-style reduction test (2026-11-23, no calls)
Script `scripts/nwp_slp_reduction_test.py`, using the cached 2025 day-0 data. Outputs `reduction_test_*_2025.csv`.
- **Round trip:** reducing Open-Meteo surface_pressure with the current temperature reproduces pressure_msl to
  0.03 hPa (median), 0.11 max. msl and sfc are exactly linked.
- **A simple METAR-style reduction does NOT fix it.** Using T = (T_now + T_12h)/2 cuts the diurnal error range at
  the high stations (median 1.96 -> 1.46 hPa) but leaves the seasonal range unchanged (3.46 -> 3.47) and the bias too
  (TPH -0.7 Jan .. -6.0 Aug, the same as raw). Median day-0 gradient RMSE: raw 1.07, manual 12-h 1.03.
- Adding a per-station constant offset (a stand-in for the plateau correction, leave-one-month-out) removes the
  bias (gradient RMSE 0.92) but not the seasonal cycle.
- The harmonic correction stays best: 0.87, with seasonal range 2.0 and diurnal range 0.3 at the high stations.
- **Reading:** the seasonal gap is not current vs 12-h-mean temperature. METAR SLP at these stations behaves as
  if reduced through a much cooler column in summer than the surface air (station-specific plateau-type
  correction). GFS / Open-Meteo reduce through the hot surface temperature.
- **Possible model-agnostic route (not done):** fit a "METAR-convention" emulator on OBS only (METAR SLP vs observed
  station pressure, per station; obs_hourly has both). Apply it to any model's station pressure
  (Open-Meteo surface_pressure at station elevation). Per model this needs only the station-pressure check, with no
  model-specific fit.

## Next questions (user, 2026-11-23) and probe results
1. Is the SLP bias specific to Open-Meteo, or would raw GFS GRIB from AWS avoid the correction?
2. Do other models have as severe a bias as GFS?
3. Do other models show a similar RMSE reduction with GBT post-processing?

**Open-Meteo Previous Runs probe** (KTPH, Jan 15 2025 and 2026; previous_day1/day7 for pressure_msl and wind_speed_10m):

| model id | pressure | wind | lead days | 2025 available | note |
|---|---|---|---|---|---|
| ecmwf_ifs025 | yes | yes | 0-7 | yes | full candidate |
| gem_global | yes | yes | 0-7 | yes | full candidate |
| jma_gsm | yes | yes | 0-7 | yes | full candidate (coarse) |
| icon_global | yes | yes | 0-6 | yes | marked discontinued on Open-Meteo 2026-11-30 |
| ukmo_global_deterministic_10km | yes | 2026 only | 0-6 | pressure only | |
| ncep_hrrr_conus / gfs_hrrr | yes | yes | 0-1 | yes | marked discontinued 2026-11-30 |
| ecmwf_aifs025_single | yes | yes | 0-7 | NO (2026 only) | no 2025 calibration year |
| ncep_nbm_conus | NO | yes | 0-7 | yes | raw-wind benchmark only |
| ncep_nam_conus | partial | partial | 0-1 | no | skip |
| meteofrance_arpege_world | yes | NO | 0-1 | yes | no raw-wind baseline; skip |
| gem_hrdps_continental | - | - | - | - | no data at this location |

**AWS (noaa-gfs-bdp-pds):** 0.25 deg pgrb2 with .idx files exist for 2024-2026. Each file has PRMSL (Shuell
reduction), MSLET (Eta/Mesinger reduction), PRES:surface, HGT:surface and TMP:2 m, so single fields can be
downloaded by byte range. There is no GRIB reader in analysis_env (cfgrib/eccodes/pygrib missing).

## Q1 result: AWS GFS GRIB vs Open-Meteo (2026-11-23)
Script `scripts/gfs_grib_slp_test.py`. Downloaded PRMSL, MSLET, PRES:sfc, HGT:sfc and TMP:2m by byte range for the
f000 analyses at 00/06/12/18Z, every day of Jan + Jul 2025 (248 files, 932 MB, about 4 min). Stored in
`/mnt/d/NWP/aws_gfs/grib/`. Outputs `derived/verification/gfs/aws_grib_test/`; figures
`images/verification/gfs/aws_grib_test/`.
- **Open-Meteo GFS `pressure_msl` = GFS MSLET** (Eta/Mesinger reduction). Median |diff| 0.03 hPa; 91% within 0.05.
  It is not PRMSL (median 0.22, max 8.5). Spot-checked independently at KTPH: |om - mslet| 0.02 vs
  |om - prmsl| 2.64 hPa.
- **PRMSL (NCEP standard / Shuell) largely removes the summer bias.** July bias vs METAR, MSLET -> PRMSL:
  TPH -5.98 -> -0.17; WMC -4.93 -> -0.45; RNO -3.56 -> +0.60; LAS -3.25 -> +0.14.
  - July day-0 gradient bias: LAX-TPH +5.72 -> -0.06; SFO-WMC +5.00 -> +0.50.
  - July RMSE on the 4 problem pairs: 3.4-6.0 -> 1.1-1.3 hPa.
- PRMSL is not perfect. January is similar for both (NID, WJF about -1.5). In July PRMSL runs high at DAG (+1.5)
  and RNO, so LAX-DAG and NID-RNO get about -1.7 hPa gradient bias. A small per-station fit may still help.
- **GRIB station pressure check:** surface pressure adjusted to station height vs barometers: bias -1.15..+0.46,
  with no seasonal swing at the high stations. GFS is right at station level; the gap is purely MSLET's reduction.
- **Answer to Q1:** the bias is in the field Open-Meteo serves (MSLET), not in GFS itself. AWS PRMSL would mostly
  remove the need for a correction.
  - Cost for the 2026 holdout: hourly f000-f197 for 4 cycles per day, about tens of GB even for PRMSL only.
    GFS is 3-hourly after f120.
  - Caveat: only 2 months of day-0 analyses were tested; lead-time behaviour is untested.

## Q2 result: ECMWF IFS and GEM SLP bias (2026-11-23)
`nwp_config.py` has new `ecmwf` (ecmwf_ifs025) and `gem` (gem_global) entries. `nwp_pressure_calibration.py` now
skips step 4 when no points file exists. New script `scripts/nwp_compare_models_slp.py`. About 365 weighted calls.
Outputs in `derived/verification/{ecmwf,gem}/pressure_calibration/` and `derived/verification/model_comparison/`;
figures in `images/verification/model_comparison/`.
- **Answer to Q2: neither has GFS's severe summer bias.**
  - Day-0 SLP bias at TPH, Jan / Aug: GFS -0.7 / -6.0; ECMWF +0.6 / +0.3; GEM +2.9 / -0.7.
  - Median high-station bias: GFS -1.71, ECMWF +0.10, GEM +0.29 hPa.
  - Seasonal swing of bias: GFS 3.45, ECMWF 1.28, GEM 2.41. Diurnal swing: GFS 1.96, ECMWF 1.42, GEM 2.98
    (GEM TPH 5.4).
- **GEM runs 2-3 hPa high at the high stations in winter** (Dec-Feb) and has a large diurnal swing. ECMWF is the
  cleanest. Its worst stations: KNID -1.6..-1.8 in summer; NID-RNO gradient bias -2.1.
- Open-Meteo `surface_pressure` is derived for all three models. For ECMWF and GEM the derived station pressure has
  about a 3 hPa seasonal swing at the high stations (GFS 1.3). Their msl is consistent with a cooler-column
  reduction (closer to METAR) that Open-Meteo's current-temperature formula doesn't invert.
- **Day-0 gradient RMSE, leave-one-month-out, raw -> corrected (median over pairs, excluding SAC-RDD):**
  GFS 1.10 -> 0.87, ECMWF 0.84 -> 0.61, GEM 0.92 -> 0.65. 21/21 pairs improve for each model, so the correction
  is used for both. Uncorrected ECMWF already beats corrected GFS.
  - LAX-TPH: GFS 4.23 -> 1.85, ECMWF 1.51 -> 1.33, GEM 2.54 -> 1.42.
  - Median |gradient bias| (annual): GFS 0.69, ECMWF 0.39, GEM 0.41.
- Config facts for Q3, to verify then: ECMWF runs 00/06/12/18Z, archive from about 2024-02-03. GEM runs only
  00/12Z (lead_h rule differs), archive from about 2024-02-01.

## AWS guidance inventory and download streamlining (2026-11-23 evening)
**Buckets checked (anonymous S3 listing, 2026-01-15 files inspected):**

| bucket | model | SLP field(s) | other | archive seen |
|---|---|---|---|---|
| noaa-gfs-bdp-pds | GFS 0.25/0.5/1.0 | PRMSL, MSLET, PRES:sfc | hourly to f120, then 3-hourly | 2021+ |
| noaa-gefs-pds | GEFS (0.5 deg members) | PRMSL, PRES:sfc | ensemble | 2017+ |
| noaa-hrrr-bdp-pds | HRRR 3 km CONUS | MSLMA (MAPS reduction), PRES:sfc | 0-48 h | 2014+ |
| hrrrzarr | HRRR in Zarr | (chunked spatial access) | small reads | exists |
| noaa-nam-pds | NAM | PRMSL, MSLET, PRES:sfc | 0-84 h | 2021+ |
| noaa-rap-pds | RAP | (not inspected) | | 2021+ |
| noaa-rrfs-pds | RRFS (rrfs_a) | (not inspected) | daily dirs exist in 2026; operational status unverified | |
| noaa-nbm-grib2-pds | NBM | none in core | WIND / GUST 10 m (raw-wind benchmark) | 2020+ |
| ecmwf-forecasts | ECMWF open data IFS (+AIFS) | msl, sp | 10u/10v; 0.25 from 2024-02 (0.4 before) | 2023+ |
| noaa-urma-pds / noaa-rtma-pds | analyses | | gridded obs analyses, not forecasts | |

GEM is not on AWS. noaa-href-pds does not exist. ECMWF open data is probably Open-Meteo's own ECMWF source.

**Message sizes (GFS PRMSL, one field):** 1.00 MB at 0.25 deg, 0.30 MB at 0.5, 0.08 MB at 1.0. GRIB messages are
global, so a byte range can't subset space. Volume scales with the number of (cycle, forecast-hour) messages, not
with the number of stations.

**Gradient-threshold sampling (user idea; tested on 2026 obs gradients):**
- The union over pairs keeps almost every hour: |grad| >= 3: 99.1%; >= 4: 94.7%; >= 5: 84.4%; >= 6: 66.6%.
  SAN-IPL and SAN-LAS are >= 3 hPa about 70% of the time. Per-pair selection saves nothing, because each message
  covers all stations.
- SoCal offshore only (any LAX-*/SAN-* pair <= -3 hPa): 30.5% of hours, a 3x saving.
- Selecting on the OBSERVED gradient conditions on the outcome side. Own-pair |grad_obs| >= 3 keeps 35% of scoring
  rows, drops 18% of the >= 30 mph hours, and inflates RMSE (GBT 3.66 / NWP 6.30 vs all-rows 3.13 / 4.81 at day 1).
  "obs OR Open-Meteo forecast >= 3" reduces the drop (87% of >= 30 mph hours kept).

**Systematic time subsampling (unbiased) — GFS PRMSL 0.25, Jan-Oct 2026, 4 cycles/day:**

| valid times | messages | 0.25 deg | 0.5 deg |
|---|---|---|---|
| hourly (Open-Meteo-equivalent; days 5-7 need interpolation) | ~58,400 | ~58 GB | ~17 GB |
| 3-hourly | ~19,500 | ~19.5 GB | ~5.8 GB |
| 6-hourly (00/06/12/18Z), exact 24N-h leads, no interpolation | ~9,700 | ~9.7 GB | ~2.9 GB |
| 6-hourly + SoCal-offshore filter | ~3,000 | ~3 GB | ~0.9 GB |

The Q1 download speed was about 3.9 MB/s, so 6-hourly at 0.25 deg takes about 45 min. Open-Meteo previous_dayN at
00/06/12/18Z also has lead exactly 24N h, so the two are directly comparable.

**Resolution test (0.5 vs 0.25 deg PRMSL, nearest point, Jan+Jul 2025, 248 analyses, 70 MB):**
- Median |diff| 0.14 hPa, but RNO is +1.52 +/- 1.40 and PMD 0.91 SD.
- Gradient RMS difference: SAC-RNO 2.07, NID-RNO 1.66, LAX-PMD 0.98. PRMSL in complex terrain varies over short
  distances, so keep 0.25 deg.

## AWS PRMSL as GBT input, 6-hourly 2026 (2026-11-23 night)
`scripts/gfs_aws_prmsl_fetch.py`: 9,728 PRMSL messages (0.25 deg, valid 00/06/12/18Z, cycle V-24N, f=24N), 9.3 GB in
22 min at 6.9 MB/s, 0 failures. Raw data in `/mnt/d/NWP/aws_gfs/prmsl/`; points in
`derived/verification/gfs/aws_prmsl/points_20260101_20261031.parquet`.
`scripts/nwp_prmsl_test.py`: the feature builder reproduces the stored gbt_obs exactly. Outputs in
`derived/verification/gfs/aws_prmsl/`; figure `images/verification/gfs/aws_prmsl/prmsl_vs_openmeteo.png`.
- **Gradient RMSE vs METAR (median over pairs), days 0 / 1 / 3 / 7:**
  - Open-Meteo corrected 0.83 / 1.03 / 1.37 / 2.07
  - PRMSL raw 1.03 / 1.18 / 1.46 / 2.27
  - PRMSL + Jan/Jul-2025 constant offset 0.87 / 1.13 / 1.34 / 2.22
  - Open-Meteo raw 1.11 / 1.33 / 1.70 / 2.46
  PRMSL fixes LAX-TPH / SFO-WMC / SAC-RNO (day 1 LAX-TPH 1.68 vs 2.06 corrected), but adds LAX-DAG / NID-RNO errors.
- **Wind RMSE, all sets, days 0 / 3 / 7** (each on its own sign rows):
  - GBT-PRMSL 3.45 / 3.72 / 5.17; GBT-PRMSL+offset 3.11 / 3.62 / 5.11
  - GBT-Open-Meteo corrected 2.99 / 3.50 / 4.95; GBT-Open-Meteo raw 4.10 / 4.42 / 5.63
  - Raw GFS wind 4.57 / 5.00 / 5.89
- **Head-to-head on common rows** (positive = Open-Meteo corrected better):
  - PRMSL raw: +0.44 (day 0) to +0.18..0.25 mph (days 3-7), all CIs above 0.
  - PRMSL + offset: +0.12..0.18 mph, all CIs above 0.
  - Offshore: PRMSL raw +0.86 at day 0, not significant at days 2-5. PRMSL + offset +0.23..0.48, all significant.
- **Strong winds, lead 1, >= 30 mph:** RMSE corrected 7.47, PRMSL 7.79, PRMSL+offset 8.24. PRMSL raw has the
  smallest negative bias (-5.97 vs -6.39). In calm hours PRMSL runs more positive (+1.93 vs +1.28).
- **Reading:** AWS PRMSL removes the large MSLET summer bias, and raw PRMSL beats raw Open-Meteo. But raw PRMSL is
  NOT as good an input as Open-Meteo + the 2025 harmonic correction: PRMSL keeps smaller seasonal/diurnal station
  biases that the full correction removes and a constant offset doesn't. So the correction step still pays off;
  switching to AWS doesn't remove it.
  - Open question: would PRMSL + the same harmonic correction (fitted on 2025 6-hourly PRMSL, about 1,460
    messages / 1.5 GB) beat Open-Meteo + correction?

## Next session: start here (updated 2026-11-23 night)
- **Decision (user, 2026-11-23):** keep Open-Meteo + the per-model 2025 harmonic SLP correction as the pressure
  source. AWS PRMSL was tested and not adopted. `gfs_aws_prmsl_fetch.py` and `nwp_prmsl_test.py` stay uncommitted
  by the user's choice.
- **Q3 (next):** run the full pipeline for ECMWF IFS first (coefficients already fitted), then HRRR (Open-Meteo
  marks it discontinued 2026-11-30; days 0-1 only), then GEM (00/12Z cycles only, so the lead_h rule differs).
  NBM serves as a wind-only benchmark.
  - Per model: verify the lead definition (single runs), fetch 2026 winds and pressure (about 8k calls, one day of
    quota), run QC, `nwp_run_gbt.py --model X`, then `nwp_verify.py --model X`.
  - Then compare across models.
- The plan doc changes since commit 543b577 are uncommitted.

## Q3 step 1: ECMWF IFS lead definition (2026-11-23)
Scratch check (34 weighted calls): KLAX, `previous_day0-7` for valid 2026-06-08, against all 32 single runs
2026-06-01 00Z to 06-08 18Z.
- **Run horizons:** 00/12Z runs reach 239 h; 06/18Z runs stop at 146 h.
- **Days 0-5:** fed by the latest 00/06/12/18Z run at or before valid - 24N h, so lead = 24N + (hour mod 6), as for GFS.
- **Days 6-7:** fed only by 00/12Z runs, so lead = 24N + (hour mod 12): day 6 = 144-155 h, day 7 = 168-179 h.
- **Native vs interpolated hours:** at 3-hourly valid times (6-hourly beyond 144 h) previous-runs equals the single
  run exactly. Other hours are Open-Meteo interpolations of the stitched series. They differ from the single run by
  up to 0.6 hPa / 0.7 m/s at days 0-6, and 2.1 hPa / 0.7 m/s at day 7. Neither within-run nor across-run linear
  interpolation reproduces them exactly. This is a data property, not a lead error.
  (GFS is natively hourly to 120 h, so it doesn't have this.)
- `nwp_config.py`: `ecmwf.cycle_hours_by_day = {6: 12, 7: 12}` plus a `cycle_hours_for()` helper, used by the
  fetcher for `lead_h` (metadata only; no model input).

## Q3 step 2: ECMWF test fetch + native-hours scoring decision (2026-11-23)
- **Test fetch (KACV, both variables, 2026-01-01..10-31):** cost 34.7 (set-only stations ~17.4), so the full fetch is
  about 7.8k weighted calls, as for GFS. Archive gap: `pressure_msl_previous_day5` and `_day6` are missing 2026-04-17
  15Z to 04-22 23Z (129 h each). Wind is complete.
- **Decision (user):** score only the models' **native output times**, not Open-Meteo's hourly interpolations, which
  a reviewer would question. The GFS/ECMWF difference in time resolution is discussed explicitly in the write-up.
  - `nwp_config.py`: `native_steps` per model plus `is_native(cfg, lead_h)`.
    GFS: hourly to 120 h, 3-hourly beyond (days 0-4 all hours; days 5-7 at 00/03/../21Z).
    ECMWF: 3-hourly to 144 h, 6-hourly beyond (days 0-5 at 00/03/../21Z; days 6-7 at 00/06/12/18Z).
  - `nwp_verify.py --valid-hours native` (default) filters the forecast rows and the pressure-error section, writing
    to `scores_native/` and `images/verification/<model>/native/`. `--valid-hours all` reproduces the earlier
    behaviour in `scores/`.
  - The existing GFS results (`scores/`, all hours) include interpolated hours at days 5-7. Rerun GFS with the
    native default (no calls) before the cross-model comparison. The like-for-like comparison uses the common times,
    i.e. ECMWF's native set, which is a subset of GFS's.

## Q3 step 3: ECMWF fetch and data QC, 2026 (2026-11-24)
- **Fetch:** 437/437 stations ok, 7,765 weighted calls, 1 h 47 min, 1 SSL timeout (recovered on retry). Points table
  25.5 M rows (50 MB). QC outputs in `derived/verification/ecmwf/qc/`.
- **Completeness:** wind is complete. Pressure is missing only for days 5-6, 2026-04-17 15Z to 04-22, at all 22
  gradient stations (5,676 values, an archive gap).
- **Values:** pressure 983.8-1046.3 hPa; wind 0-21.8 m/s; longest identical run 10 h. Leads are distinct (0 hours
  with day 0 = day 1 at >= 90% of stations). Wind climatology is flat across leads; pressure drifts about -0.7 hPa by
  day 7 (GFS about -1).
- **Grid cells:** 0.8-28.1 km from the station (median 10.2; 54 beyond 15 km, against a GFS max of 13.9). This is the
  0.25 deg grid with `cell_selection=land`. |dz| > 300 m only at WAPC1F1 (a metadata error, as for GFS) and WMSC1.
- **Day-0 pressure vs METAR SLP:** no seasonal bias. The pooled monthly bias is -0.12..+0.13 hPa (GFS TPH alone
  -1.6..-6.6). SD 0.25-0.9 at low stations and 1.2-1.4 at TPH / WMC / LMT / RNO. The 2025 harmonic correction is
  still applied (Q2 CV: gradient RMSE 0.84 -> 0.61).
- **Day-0 wind vs matched max sustained:** median r 0.65 (GFS 0.62); 5-95% 0.29-0.82; 8 stations r < 0.2
  (KSXT again -0.24, Oregon valley RAWS).
  **NWP/obs mean ratio 0.75 (GFS 0.94): ECMWF 10 m wind runs low.** That is a bias the raw-NWP baseline carries;
  the oracle bias-corrected baseline (`nwp_bc`) removes it.

## Q3 step 4: ECMWF GBT + verification, native times (2026-11-24)
`nwp_run_gbt.py --model ecmwf` (1.48 M forecast rows); `nwp_verify.py --model ecmwf` (native: 432k rows kept).
Outputs in `derived/verification/ecmwf/scores_native/`; figures in `images/verification/ecmwf/native/`.
- **RMSE (mph), all sets, days 0 / 1 / 3 / 5 / 7:**
  - GBT-corrected 2.75 / 2.83 / 3.12 / 3.74 / 4.65
  - oracle bias-corrected ECMWF 3.08 / 3.19 / 3.45 / 3.95 / 4.62
  - raw ECMWF 5.76 / 5.70 / 5.80 / 6.13 / 6.40 (bias about -4.1 mph: ECMWF 10 m wind runs low)
  - gbt_obs (perfect pressure) 2.72
- **Verdict:** GBT is significantly better than raw ECMWF at all 8 leads, all sets and offshore. Against the oracle
  bias-corrected ECMWF it is significantly better at days 0-5 (all sets) and days 0-6 (offshore). Day 6 all
  (-0.08, CI -0.22..+0.05) and day 7 (+0.03 all, -0.07 offshore) are ties.
  GFS (native) beat its oracle at all 8 leads (all sets) and days 0-6 (offshore), so the GBT margin over the
  NWP itself is smaller for ECMWF, mostly because bias-corrected ECMWF wind is better than bias-corrected GFS.
- **>= 30 mph, lead 1:** GBT bias -5.4, RMSE 7.10; oracle bias-corrected -10.6 / 11.5; raw -18.4 / 19.0.

### GFS vs ECMWF head-to-head (common rows)
`scripts/nwp_compare_gfs_ecmwf.py` -> `derived/verification/model_comparison/gfs_vs_ecmwf_common_native.csv`.
Rows: ECMWF native times (days 0-5 3-hourly, days 6-7 6-hourly), qualifying, selected by BOTH models' corrected sign
(22-26k rows per lead). The oracle bias correction was refitted on these rows. Paired day-block bootstrap (1,000).
- **GBT-ECMWF minus GBT-GFS RMSE, all sets:** -0.24 (day 0), -0.30, -0.38, -0.41, -0.38, -0.45, -0.30, -0.46 (day 7);
  every CI is below 0. Offshore: -0.10 (day 0, n.s.), -0.38..-0.67 at days 1-5, -0.33 (day 6, n.s.), -0.76 (day 7).
- **Raw wind ranks the other way:** raw GFS 4.76-6.22 beats raw ECMWF 5.88-6.80 (ECMWF low bias). Bias-corrected,
  ECMWF is better (3.09-4.65 vs 3.46-5.32).
- **>= 30 mph (about 145 rows per lead, no CI):** GBT-ECMWF 7.1-10.9 vs GBT-GFS 7.9-13.1.
- **Reading:** ECMWF pressure makes the better GBT input at every lead, consistent with its smaller gradient errors
  (Q2). The added value of the GBT over the NWP's own wind is clear for both models through day 5; at days 6-7 it is
  a tie against bias-corrected ECMWF.

## Q3 HRRR: lead definition, calibration, fetch (2026-11-24)
- **Lead check** (15 weighted calls; KLAX, valid 2026-06-08, `ncep_hrrr_conus`):
  - `previous_day2` is empty.
  - **Day 1** = the 00/06/12/18Z 48-h runs, lead 24 + (hour mod 6). It matches the single run exactly at every hour,
    so output is natively hourly.
  - **Day 0** = lead 0 of each hourly run (an analysis). GFS/ECMWF day 0 is 0-5 h, so day 1 is the HRRR comparison
    that counts.
  - The single-runs archive keeps only every 3rd run (00/03/../21Z).
  - `nwp_config.py` entry `hrrr`: `cycle_hours_by_day={0: 1}`, `max_lead_day=1`, no `native_steps` (all hours native).
- **2025 SLP calibration** (172 calls): the correction is USED (21/21 pairs improve in LOMO CV). Median day-0
  gradient RMSE 1.23 -> 1.03, against GFS 0.87 and ECMWF 0.61 after correction.
  - HRRR `pressure_msl` (MSLMA) is noisy at the high stations: SLP SD 3.4-3.7 hPa raw, 2.5-2.7 corrected at TPH / WMC /
    RNO / LMT.
  - Corrected gradient RMSE: LAX-TPH 2.36, SFO-WMC 2.26, SAC-RNO 2.32, against GFS 1.85 / 1.76 / 1.57.
  - Expect HRRR-driven GBT to suffer on the pairs with high stations.
- **Fetch:** 437/437 ok, 1,941 weighted calls, 35 min, no rate limit hit (about 7.1k calls on 2026-11-24 UTC).
  Points 6.4 M rows. Archive gap: `pressure_msl_previous_day1` is missing 2026-04-04 19Z to 04-10 17Z (143 h) at
  all 22 gradient stations.
- **QC** (`nwp_qc_points.py` now handles fewer than 8 leads):
  - Grid cells 0-5.6 km from the station (median 1.3).
  - Day-0 wind vs obs: median r 0.73 (GFS 0.62, ECMWF 0.65); NWP/obs ratio 1.11. Day-1 wind is about 9% windier than
    day 0.
  - Day-0 SLP vs METAR: SD 2.4-2.7 hPa at TPH / WMC / RNO / LMT; bias -1.4..-2.8 at the high stations.
- **Verification** (`nwp_verify.py` now adapts the event-plot leads: days 0/1 for HRRR, no d5d7). All hours are native.
  - RMSE, all sets, days 0 / 1:
    - GBT-HRRR-corrected 3.09 / 3.33
    - raw HRRR 2.74 / 3.16
    - bias-corrected HRRR 2.35 / 2.87
    - gbt_obs 2.71
  - **Raw HRRR beats the GBT driven by HRRR pressure at both days** (significant for all sets; ties offshore).
    HRRR pressure is the worst GBT input of the three models.
- **Raw HRRR vs the GBTs, common rows** (`scripts/nwp_compare_hrrr.py` ->
  `model_comparison/hrrr_vs_gbt_common_native.csv`; 3-hourly times, selected by all three corrected signs,
  days 0-1). GBT-ECMWF minus HRRR RMSE:
  - Day 1, all sets: vs raw -0.32 [-0.42, -0.23]; vs bias-corrected -0.04 [-0.13, +0.04] (tie).
  - Day 1, offshore: vs raw -0.52 [-0.74, -0.29]; vs bias-corrected -0.31 [-0.49, -0.11].
  - Day 0 (HRRR = analysis), all sets: vs raw -0.01 (tie); vs bias-corrected +0.40 (HRRR better).
    Offshore: vs raw -0.34 (GBT better); vs bias-corrected +0.13 (n.s.).
  - **>= 30 mph obs (about 145 rows):** raw HRRR 6.4 vs GBT-ECMWF 7.1-7.3 RMSE; bias -3.5 vs -5.4.
    20-30 mph, day 1: GBT-ECMWF 3.89 vs raw HRRR 4.04.
  - **Reading:** at day 1 the ECMWF-driven GBT beats raw HRRR and ties bias-corrected HRRR overall, and beats both
    offshore. HRRR's 3 km wind handles the strongest winds better (smaller low bias), which the gradient-only GBT
    can't match. HRRR day 0 is an analysis, so it isn't a fair forecast comparison.
- **HRRR figures** in `images/verification/hrrr/native/`: the standard set plus `events/` at days 0/1.
  `nwp_verify.py` now forces integer lead-day ticks on its line plots (2-lead axes showed 0.2 steps). GFS and ECMWF
  were rerun too; their numbers are unchanged.
- **Side-by-side event plots, GBT-ECMWF vs raw HRRR** (`scripts/nwp_compare_hrrr_events.py` ->
  `images/verification/model_comparison/hrrr_vs_gbt_ecmwf_events/`, window table
  `model_comparison/hrrr_vs_gbt_ecmwf_event_windows.csv`): the same top-5 event days per set, days 0/1, all hours
  (no sign selection).
  - Median event-window RMSE on common 3-hourly times: day 1 GBT-ECMWF 3.19 vs raw HRRR 3.41 (GBT better in
    79/130 windows); day 0 3.04 vs 2.97 (65/130).
  - Raw HRRR catches some short peaks that the GBT smooths over (e.g. SFO-WMC+ Feb 24, May 3).

## Q3 GEM: lead definition and native step (2026-11-24)
Scratch check, about 75 weighted calls. KLAX; previous_day0-7 compared against single runs.
- **Single-runs archive:** GEM runs for 2026-05-31..06-08 and 2026-08-01 return all-null values, and 2026-03-01 is
  rejected. Runs from 2026-10-20 12Z on are complete, so the check uses valid 2026-10-28 (17 runs, `forecast_days=11`;
  the default response is cut at 168 h).
- **Runs:** only 00/12Z runs exist; 06/18Z return "run not available". The horizon is 240 h.
- **Lead rule:** every day 0-7 is fed by the latest 00/12Z run at or before valid - 24N h, so lead = 24N + (hour mod 12).
  Previous-runs equals that single run at every hour: pressure exactly, wind within 0.05 m/s (rounding).
  - Caveat for the write-up: at the same lead day GEM is up to 6 h older than GFS/ECMWF days 0-5 (24N + 0-11 h vs
    24N + 0-5 h), as ECMWF is at days 6-7. Day 0 is a 0-11 h forecast.
- **Native step:** hourly to 84 h, 3-hourly beyond. Beyond 84 h the off-3-hourly hours equal a Catmull-Rom Hermite
  interpolation of the 3-hourly values to within 0.05 (rounding). Up to 83 h they differ by up to 0.7 hPa / 2.2 m/s,
  so they are real model output.
  - `nwp_config.py` gem: `cycle_hours=12`, `native_steps=((84, 1), (None, 3))`. Days 0-3 (lead <= 83 h) are native
    at all hours, days 4-7 at 00/03/../21Z. ECMWF's native set is a subset, so the common-row comparisons are
    unchanged.
- **Previous-runs completeness, KLAX 2026** (this doubled as the test fetch; cost 34.7, the same as ECMWF):
  - Days 0-1 complete.
  - Gaps are in June (16% missing at days 3-7; wind days 2 and 5 37%) and April (pressure day 5, 5%).
  - Overall missing per column is 0-3.6%.
- **Fetch (2026-11-24/25):** 437/437 stations ok, 7,765 weighted calls, 2 h 3 min, 1 read timeout (recovered on
  retry). Points table 25.5 M rows (36 MB).
- **QC** (`derived/verification/gem/qc/`):
  - Missing pressure is 13.7k values (0.6%), in the June/April gaps.
  - Grid cells are 0-23 km from the station (median 6.3).
  - Day-0 SLP vs METAR: median |bias| 0.29 hPa, median SD 0.54. For comparison, ECMWF is 0.27 / 0.43 and GFS
    0.77 / 0.84. SD is 1.8-2.5 hPa at RNO / WMC / TPH.
  - Day-0 wind vs obs: median r 0.62 (GFS 0.62, ECMWF 0.65, HRRR 0.73). Median NWP/obs ratio 0.65, the lowest of
    the four models (ECMWF 0.75, GFS 0.94, HRRR 1.11), so raw GEM 10 m wind is strongly low-biased.
- **GBT + verification, native times** (`nwp_run_gbt.py` / `nwp_verify.py --model gem`). 987,588 of 1,480,832 rows
  are native. The verdict passes vs raw at 8/8 leads; vs bias-corrected it is significant at 5/8 leads (all sets)
  and 6/8 (offshore).
  - RMSE, all sets, days 0 / 1 / 3 / 5 / 7:
    - GBT-GEM-corrected 2.80 / 3.00 / 3.42 / 4.14 / 4.93
    - bias-corrected GEM 3.44 / 3.55 / 3.80 / 4.35 / 4.90
    - raw GEM 6.72 / 6.73 / 6.84 / 7.12 / 7.27 (bias about -5 mph)
    - gbt_obs 2.71
  - Offshore: GBT-GEM bias rises from 0.0 at day 0 to +1.0..+1.4 mph at days 3-7. ECMWF's is +0.2.
- **GEM vs GFS vs ECMWF, common rows** (`scripts/nwp_compare_gem.py` ->
  `model_comparison/gem_vs_gfs_ecmwf_common_native.csv`; native to all three, selected by all three corrected signs,
  which leaves ECMWF's native set). RMSE, all sets, days 0 / 1 / 3 / 5 / 7:
  - GBT-ECMWF 2.80 / 2.88 / 3.20 / 3.79 / 4.74
  - GBT-GEM 2.84 / 3.06 / 3.51 / 4.24 / 4.95
  - GBT-GFS 3.04 / 3.19 / 3.64 / 4.29 / 5.11
  - **GBT-GEM minus GBT-ECMWF:** +0.05 [+0.01, +0.08] at day 0, growing to +0.45 [+0.30, +0.62] at day 5 and
    +0.21 [+0.00, +0.45] at day 7, so ECMWF is better at every lead.
  - **GBT-GEM minus GBT-GFS:** -0.20 [-0.27, -0.14] at day 0, -0.12 at day 1, -0.15 at day 2 (all significant). Days
    3-7 are ties.
  - **GBT-GEM minus bias-corrected GEM:** -0.61 at day 0 to -0.16 at day 5 (significant through day 4). Days 5-7
    are ties. Offshore: -1.50 at day 0 to -0.53 at day 5 (significant), days 6-7 ties.
  - **>= 30 mph obs:** GBT-GEM 6.9-7.6 RMSE at days 0-3 (GBT-ECMWF 7.0-7.5, GBT-GFS 7.9-9.2). Raw GEM is 21+
    (bias -21 mph).
  - **Reading:** the GBT ranking by driving model is ECMWF > GEM > GFS at days 0-2, and ECMWF > GEM ≈ GFS from day 3.
    GBT post-processing gives GEM the largest gain over its raw wind of any model (-4.2 mph at day 0), because raw
    GEM wind is the most low-biased. Against the oracle bias-corrected GEM, the gain is similar to ECMWF's: it holds
    through day 4 and ties at days 6-7.
- **Next:** NBM as a wind-only benchmark (no pressure; compare its raw wind against the GBTs on common rows).
- **GEM figures (2026-11-25):**
  - `images/verification/gem/native/`: the standard set from `nwp_verify.py`, plus `events/` (days 0/1) and
    `events_d5d7/`.
  - `scripts/nwp_compare_gem_plots.py` -> `images/verification/model_comparison/gem_vs_gfs_ecmwf_by_lead.png`.
    It shows RMSE by lead for each model's GBT and bias-corrected raw wind on common rows, and GBT-GEM minus
    GBT-ECMWF / GBT-GFS / bias-corrected GEM with CIs.
  - Side-by-side event plots, GBT-GEM vs GBT-ECMWF, days 1 and 5, each model at its native times:
    `model_comparison/gem_vs_gbt_ecmwf_events/<set>.png`. Window table:
    `model_comparison/gem_vs_gbt_ecmwf_event_windows.csv`.
  - Median event-window RMSE on common ECMWF-native times:
    - Day 1: GBT-ECMWF 3.19 vs GBT-GEM 3.16. GEM is better in 58/130 windows.
    - Day 5: 4.72 vs 4.33. GEM is better in 70/130 windows.
  - In the top-5 event windows, GEM-driven GBT matches or beats ECMWF-driven GBT. Its aggregate deficit comes from
    ordinary hours.

## Q3 NBM plan (user decisions 2026-11-25)
- NBM (`ncep_nbm_conus`) has no pressure, so there is no NBM-driven GBT. NBM serves as a raw-wind benchmark.
- **Sustained wind only** for now; no gusts.
- **Compare NBM against all guidance:** raw, bias-corrected and GBT for GFS, ECMWF and GEM, plus HRRR at days 0-1.
- **Bias-correct NBM too** (same per-set, per-lead oracle). NBM is calibrated to URMA 10 m wind, not to our set-P75
  hourly-max target, so a definitional offset is expected to remain.
- Steps:
  1. Lead check: runs feeding each day, native step, horizon.
  2. Wind-only fetch at set stations.
  3. QC.
  4. `nwp_raw_wind.py` builds set-level `nwp_p75_rep` (same logic as `nwp_run_gbt.py`).
  5. Common-row comparison plus figures.

## Q3 NBM: lead check, code, fetch status (2026-11-25), resume here
**Lead check** (about 200 weighted calls; KLAX, valid 2026-06-08, `ncep_nbm_conus`, wind only):
- **Single-runs archive:** keeps only 00/03/../21Z cycles (01Z, 02Z etc. return "run not available"). Each run
  reaches about 262 h and starts at +1 h (no f000). Archive runs exist from at least 2026-06-01, unlike GEM.
- **Lead rule:** days 1-7 come from the hourly run exactly 24N h before valid (lead = 24N), and day 0 is lead 1 h.
  - The values match exactly at the 8 hours per day where that run is archived.
  - At the other hours, the 3/6/12-hourly rules fail (up to 1.2 m/s off). Those hours are also not Hermite
    interpolations of the stitched previous-runs series (up to 0.57 m/s off), so they come from the unarchived
    hourly runs.
- **Native step within a run:** hourly to 36 h, then 3-hourly to about 192 h. Off-3-hourly hours equal Open-Meteo's
  Catmull-Rom Hermite fill within 0.005 m/s. Every NBM lead (1, 24, ..., 168 h) is native, so all hours are scored.
- **Fairness caveat:** at the same lead day NBM is fresher than GFS/ECMWF (24N + 0-5 h) and GEM (24N + 0-11 h). At
  00/12Z valid times every model's lead is exactly 24N, so the comparison reports an equal-lead subset.
- **Method:** a scratch script, not kept. Fetch `previous_day0-7` for one day and the single runs (`run=...`,
  `forecast_days=11`, every 3 h back 10 days). For each candidate rule (run spacing s = 1/3/6/12, minimum lead
  0/1 h), compute max |previous - single run|. For the native step, compare off-grid hours with Catmull-Rom
  Hermite interpolation of the 3-/6-hourly values, binned by lead.

**Code** (uncommitted as of 2026-11-25; tested on partial data):
- `nwp_config.py`:
  - `nbm` entry: `variables=("wind_speed_10m",)`, `cycle_hours=1`, `lead_h_add={0: 1}`,
    `native_steps=((36,1),(192,3),(None,6))`.
  - New `lead_hours_for(cfg, hour, day)` helper (adds `lead_h_add`).
- `nwp_fetch_openmeteo.py`: `station_table(only_vars)` keeps 425 set stations for wind-only models; `build()` uses
  `lead_hours_for`.
- `nwp_qc_points.py`: `HAS_P`; the pressure checks are skipped for wind-only models.
- `nwp_run_gbt.py`: `WIND_ONLY` path. It writes the obs columns, `nwp_p75_rep`, `nwp_p75_all` and `n_nwp_rep`, with
  no NWP-driven GBT or gradient columns. Regression check: the GEM output is identical after the edit.
- `nwp_compare_nbm.py` (new):
  - Common rows: native to NBM, GFS, ECMWF and GEM, selected by all three GBTs' corrected signs.
  - Forecasts: raw/bc NBM vs GBT/bc/raw for each of GFS, ECMWF and GEM.
  - Groups: all / offshore / >=30 mph obs / 00+12Z (equal lead).
  - HRRR table: days 0-1 on rows that also have HRRR.
  - Day-block bootstrap differences.
  - Outputs: `model_comparison/nbm_vs_guidance_{scores,diffs,hrrr_scores}.csv`.
- `nwp_compare_nbm_plots.py` (new):
  - `images/verification/model_comparison/nbm_vs_guidance_by_lead.png` (3x2).
  - `nbm_vs_gbt_ecmwf_events/<set>.png`: GBT-ECMWF vs raw NBM, days 1 and 5.
  - `model_comparison/nbm_vs_gbt_ecmwf_event_windows.csv`.

**Fetch status:**
- Started 2026-11-25 02:35Z; cost 17.37 per station.
- Hit the free-tier daily cap at 03:38Z after 275/425 stations (4,777 calls; the GEM fetch had used about 5.2k of
  the same UTC day).
- The process sleeps until about 00:01Z 2026-11-26, then resumes. It will probably not survive the session ending.
- Log: `/mnt/d/NWP/open_meteo/logs/fetch_nbm_2026.log`. Manifest: `/mnt/d/NWP/open_meteo/manifest_nbm.csv`.

**STALE partial outputs, overwrite them:** the NBM points table, `verification/nbm/qc/`,
`verification/nbm/forecasts_sustained.parquet`, the `nbm_vs_*` CSVs and figures. All were built from 19-32 stations
for testing. Set P75 over partial stations is NOT comparable, so do not quote any of those numbers.

**Resume steps** (from `scripts/`, with `~/.venv/analysis_env/bin/python`):
1. Check `tail /mnt/d/NWP/open_meteo/logs/fetch_nbm_2026.log` and `ps aux | grep nwp_fetch`. If the fetch isn't
   finished and isn't running, rerun `python nwp_fetch_openmeteo.py --model nbm --start 2026-01-01 --end 2026-10-31`.
   It skips stations marked ok in the manifest (about 150 left, 2.6k calls, 35 min) and then builds the points table.
   Confirm 425/425 ok in the manifest.
2. `python nwp_qc_points.py --model nbm`. Compare day-0 wind r and obs ratio with the other models: GFS 0.62 / 0.94,
   ECMWF 0.65 / 0.75, GEM 0.62 / 0.65, HRRR 0.73 / 1.11. The first 19 stations gave NBM 0.76 / 0.79. Also check the
   high day-0 frac_zero (up to 0.16) and the 41 h stuck runs at day 0.
3. `python nwp_run_gbt.py --model nbm` (wind-only path; about 2 min).
4. `python nwp_compare_nbm.py`, then `python nwp_compare_nbm_plots.py`. Look at the figures.
5. Log the results here, update the memory note, and commit. Uncommitted at this point: the NBM code above, this doc
   section, and the plan section above. The AWS PRMSL scripts stay uncommitted by choice.
6. Optional, if wanted for consistency: `nwp_verify.py` does not support wind-only models (it needs GBT-NWP
   columns), so NBM has no per-model `images/verification/nbm/native/` suite. Ask the user before adapting it.

## Q3 NBM results (2026-11-27)
**Fetch:** resumed 2026-11-27 06:39Z and finished in 40 min (2,606 weighted calls). The manifest shows 425/425 ok
and 0 missing. Points table: 24.8M rows. All the earlier STALE partial outputs are overwritten.

**QC (`nwp_qc_points.py --model nbm`):**
- Day-0 wind against observations: median r 0.69, median NWP/obs ratio 1.00 (351 matched stations).
- That is the best r after HRRR (0.73 / 1.11) and the best ratio of any model.
- Stuck values are station-specific. MSAC1 has a 218 h identical run and 46% exact zeros, and KEDU has 36% zeros.
  These are probably calm sites where NBM's blending of observations shows through. They are not a fetch fault.

**Comparison (`nwp_compare_nbm.py`):** common rows, RMSE in mph, day-block bootstrap 95% CI.

| lead day | raw NBM | bc NBM | GBT-ECMWF | bc ECMWF | GBT-GEM | GBT-GFS |
|---|---|---|---|---|---|---|
| 0 | 3.67 | 2.82 | 2.80 | 3.10 | 2.84 | 3.04 |
| 1 | 3.78 | 2.95 | 2.88 | 3.22 | 3.06 | 3.19 |
| 3 | 4.29 | 3.48 | 3.20 | 3.50 | 3.51 | 3.64 |
| 5 | 4.99 | 4.14 | 3.82 | 4.03 | 4.26 | 4.33 |
| 7 | 5.37 | 4.51 | 4.74 | 4.64 | 4.95 | 5.11 |

- **Raw NBM** has a bias of about -2 mph. GBT-ECMWF beats it at every lead by 0.6-1.2 mph (all CIs exclude 0).
  The offshore margin is 0.7-1.5 mph.
- **Bias-corrected NBM** is the strongest benchmark.
  - Against GBT-ECMWF: a tie at days 0-1, then GBT-ECMWF is better at days 2-5 (-0.14 to -0.33, significant).
    Days 6-7 favour bc NBM (day 7 +0.23, significant).
  - GBT-GEM and GBT-GFS are tied with it or worse in the all-rows group.
  - bc NBM beats bc ECMWF through day 2.
- **Offshore:** GBT-ECMWF beats bc NBM at days 0-5 (-0.33 to -0.66, all significant). Days 6-7 are ties.
- **>= 30 mph observed** (n of about 65-146): raw NBM 11.6-17.4 and bc NBM 7.8-12.3. GBT-ECMWF and GBT-GEM give
  7.0-10.7, which is better than bc NBM from day 2 on.
- **Equal-lead subset (00/12Z):** gives the same picture. GBT-ECMWF ties bc NBM at days 0-2 and beats it at days 3-5.
- **HRRR rows (days 0-1):**
  - Day 0: bc HRRR is best (2.41), then GBT-ECMWF (2.81) and raw HRRR (2.82), then bc NBM (2.83).
  - Day 1: bc HRRR 2.93, GBT-ECMWF 2.90, bc NBM 2.97, raw HRRR 3.21.

**Figures:**
- `images/verification/model_comparison/nbm_vs_guidance_by_lead.png`.
- Event plots in `nbm_vs_gbt_ecmwf_events/` (26 sets). Median event-window RMSE, GBT-ECMWF vs raw NBM: 3.19 vs 3.62
  at day 1 and 4.72 vs 6.43 at day 5. Raw NBM runs persistently low through offshore peaks (see LAX-TPH-).

**Takeaway:** the GBT driven by ECMWF SLP matches the best operational blended guidance (NBM) at short range. It
beats NBM in the mid-range and in offshore and strong-wind cases, even after NBM receives an oracle bias correction.
NBM keeps a small edge at days 6-7.
