# Gradient wind models: design decisions

Running record of agreed definitions for the statistical (logistic + linear) wind models driven by
surface pressure gradients. Update this file whenever a definition changes, with the date.

## Data and periods
- Training: 2015-2025. Holdout: all of 2026 (later verified against archived NWP forecasts at the
  correlated stations, perfect-prog style).
- Station selection uses correlations computed on the training period only
  (`gradient_pipeline.py correlate --before 2026-01-01` -> `correlations_before2026-01-01.parquet`).
- Excluded: WWG networks, portable stations, wind-QC exclusions; wind-QC masks applied at match time
  (see `data/v2/derived/gradients/match_qc.json`, `docs/wind_qc_method.md`).

## Station sets (step 1, done 2026-11-13)
Per signed gradient and variable: all stations with rho >= 0.5 and n >= 50; if fewer than 10, fill to 10
from rho >= 0.4; if still fewer than 10, skip. `scripts/build_station_sets.py` ->
`data/v2/derived/models/station_sets.csv` (26 sustained, 30 gust signed gradients).

## Event definitions (logistic models)
An hour is used only if >= 50% of the set's stations report the variable being modelled. For gusts
this means >= 50% of stations have a reported gust (decided 2026-11-16).

**Sustained** (unchanged, 2026-11-13): event when the set's 75th percentile (numpy linear) >= 15 / 20 / 30 mph.

**Gust** (decided 2026-11-16; replaces "any station >= T"):
- Statistic: **nearest-rank 90th percentile with a two-station floor**. With the hour's reporting gusts
  sorted high to low, the statistic is the k-th highest value, where
  `k = max(2, ceil(0.1 * n_reporting))`.
- Equivalently: an event at threshold T means **at least max(2, ceil(0.1 * n_reporting)) stations
  have gust >= T**. A single station can never make an event.
- **No gap filling (revised 2026-11-16):** a missing gust is NOT replaced with the sustained wind. Only
  stations with a reported gust in that hour count, and `n_reporting` is the number of stations with a
  gust value. (The comparison numbers below were computed with gaps filled; they will shift slightly.)
- **Calm hours accepted as dropped (2026-11-16):** ASOS reports gusts only when they occur, so the 50%
  gust-reporting rule drops many calm hours and the gust sample leans windy. The user accepted this for
  now; revisit if it biases event rates or model skill.
- Thresholds: **30, 40, 50, 60 mph**. 30 mph is kept because it matters operationally, even though it
  occurs in 35-50% of qualifying hours for the large sets. 70 mph dropped (0-23 event hours in 11 years
  for most sets).
- A (signed gradient, threshold) model is fitted only if it has >= ~100 event hours in 2015-2025.

Why not the alternatives (analysis 2026-11-16, 2015-2025):
- *Any station / peak gust*: dominated by a few windy sites; >= 30 mph in 52-77% of hours; about half
  of >= 50 mph hours had a single station. Lowest correlation with |gradient| of all statistics.
- *Linear-interpolated 90th percentile*: with <= ~10 stations reporting it lies between the top two
  values, so one extreme station can carry it over T (e.g. LAX-PMD+ 94%, LAX-BFL- 55% of >= 50 mph
  events had only one station >= T). Large sets (>= 20) were unaffected.
- The chosen statistic matches the linear P90 within ~1 point of event rate for large sets, keeps its
  better correlation with |gradient| (e.g. SFO-ACV+ 0.79 vs 0.74 for the max), and for small sets is
  exactly "at least two stations".

## Linear models
- **Sustained:** target is the set's 75th percentile (numpy linear), as for the sustained events.
- **Gust (revised 2026-11-16):** target is the SAME statistic as the gust events, so the linear and
  logistic gust models describe one quantity: the nearest-rank 90th percentile with a two-station floor
  (the k-th highest reported gust, `k = max(2, ceil(0.1 * n_reporting))`), no gap filling, same
  >= 50% gust-reporting rule. Replaces the earlier plan of a gap-filled 75th percentile.

## Wind direction (collected 2026-11-16, not yet a predictor)
- `gradient_pipeline.py match` stores, per station and gradient time, the direction reported with the
  window's max sustained speed (`wdir_at_sustained_max_deg`). Directions outside 0-360 or whose speed
  was QC-masked are dropped. Re-running match left all speed/gust values unchanged.
- Target tables carry, per hour, the circular (unit-vector) mean direction over the set's stations with
  sustained >= 1.5 m/s (~3 kt, ASOS calm threshold): `wdir_mean_deg`, mean resultant length `wdir_r`
  (1 = all agree, 0 = no agreement) and `n_wdir`.

## Predictors and validation
- Baseline |gradient|; then hour of day and day of year, both encoded with sin/cos.
- Leave-one-year-out cross-validation within 2015-2025.
- Direction add-on (step 4): SLP-derived pressure-gradient direction, kept only if CV skill improves.

## Steps
1. Station sets - done 2026-11-13.
2. Event and target tables - done 2026-11-16. `scripts/build_targets.py` ->
   `data/v2/derived/models/targets/{sustained,gust}_{train,holdout}.parquet`, `event_counts.csv`
   (modelled = >= 100 training event hours), `targets_meta.json`.
3. Baseline models with CV - done 2026-11-16. `scripts/fit_baseline_models.py` ->
   `data/v2/derived/models/baseline/` (cv_logistic.csv, cv_linear.csv, coefficients.csv,
   oof_predictions.parquet). Plain numpy (no sklearn in the env): damped-Newton IRLS for logistic,
   least squares for linear. Undamped IRLS diverged on rare events (coefficients ~1e9); step halving
   fixed it. Median pooled CV skill, |grad| -> +hour -> +hour+doy:
   - Sustained logistic BSS: 15 mph 0.42/0.46/0.49, 20 mph 0.36/0.40/0.42, 30 mph 0.13/0.14/0.14.
   - Gust logistic BSS: 30 mph 0.36/0.39/0.42, 40 mph 0.30/0.30/0.33, 50 mph 0.15/0.16/0.18,
     60 mph 0.08/0.08/0.07. AUC 0.84-0.95 throughout.
   - Linear R2: sustained 0.55/0.59/0.62, gust 0.49/0.50/0.54.
   Rare thresholds (gust 60, sustained 30) discriminate well (AUC ~0.9) but have little Brier skill.
   Per-year BSS is unstable for rare events (years with ~0 events), so judge on pooled metrics.
4a. Over-forecast diagnostic - done 2026-11-16. `scripts/diagnose_overforecast.py` ->
   `data/v2/derived/models/diagnostics/`, `images/models/diagnostics/`. In each logistic model's top 2%
   of out-of-fold forecasts (>= 100 h), observed rate / forecast < 0.85 flags over-forecasting: 25 of
   162 models (median ratio 0.97 overall), mostly gust 50/60 mph in the LA offshore sets (LAX-WJF-,
   LAX-TPH-, LAX-PMD-, LAX-BFL-), plus SAC-RNO+, SFO-WMC+/-, NID-RNO+.
   Hits vs FALSE ALARMS within those hours (forecast high, no event; true misses not examined) (AUC):
   - Another pair's gradient is the strongest separator in 17 of 25 models and beyond noise in all 25
     (|AUC-0.5| 0.10-0.38; note: max over ~19 pairs, so somewhat optimistic). For the LA offshore sets,
     SAN-LAS and LAX-DAG are more strongly negative in hits: the broad Great Basin offshore pattern, not
     just the local pair. For LAX-BFL-, RDD-SAC (north-south valley gradient). => the orientation of the
     regional gradient matters: supports gradient direction / crossing pairs in step 4b.
   - Stations reporting and station wind-direction agreement are higher in hits (top separator in 6).
     Both are known only after the fact (not predictors); they reflect the gust-reporting rule and
     organised flow.
   - Own |gradient| (curve shape), day of year, hour: weaker, in a few models.
4b. Other-gradient add-on - first run 2026-11-16. `scripts/fit_gradient_addon.py` ->
   `data/v2/derived/models/gradient_addon/`. Base = |grad| + hour + doy; add up to 3 other pairs'
   SIGNED gradients by forward BIC selection inside each training fold (final model: selection on all
   2015-2025). Same rows as step 3; missing other gradients filled with the training-fold mean. The
   stale SAC-RDD series in gradient_timeseries.parquet (RDD-SAC with flipped sign, not in
   gradients.json) is excluded.
   Results (median, base -> add-on): logistic BSS sustained 15/20/30 0.49/0.42/0.14 -> 0.57/0.56/0.27,
   gust 30/40/50/60 0.42/0.33/0.18/0.07 -> 0.50/0.40/0.29/0.12; AUC up everywhere (gust 60 0.92 -> 0.97);
   linear R2 sustained 0.62 -> 0.70, gust 0.54 -> 0.64 (every linear model improved, +0.02 to +0.24).
   BSS improved in 88-100% of models per threshold; a few rare-event models got slightly worse.
   Most-chosen pairs: LAX-DAG, SAN-LAS, NID-RNO. Selection stable (median 92% of folds contain each
   final pair). The 3-pair cap bound in 99% of fits, so more pairs may help further.
   Top-end over-forecasting only partly fixed: flagged models' median ratio 0.76 -> 0.82, 18 of 25 still
   < 0.85. Better ranking (AUC) but still over-confident at the top: likely a model-form issue (additive
   linear logit), not only missing predictors.
   Figures: `images/models/gradient_addon/` (`scripts/plot_gradient_addon.py`).
   Decision 2026-11-16: keep the 3-pair cap for now; the user expects limited further gains from more
   gradients.
5. Gradient boosted trees - first run 2026-11-17. `scripts/fit_gbt_models.py` (LightGBM 4.7.0, installed
   2026-11-17 with scipy) -> `data/v2/derived/models/gbt/` (separate folder; reads but never writes the
   regression outputs). Same rows (asserted identical times and obs vs the add-on run), targets and
   leave-one-year-out folds. Variants: gbt_sel = base + the add-on's own per-fold BIC selections (same
   information as the add-on, so it isolates model form); gbt_all = base + all other pairs' signed
   gradients. Rounds chosen by early stopping on inner validation years from the training folds only
   (nearest years, >= 30 events), then refit on all training years. Fixed params (lr 0.03, 15 leaves,
   min 100 h per leaf, L2 1), |grad| monotone increasing, missing gradients NaN (native). Final models
   saved in gbt/models/ for the 2026 holdout. Log: models/run_gbt_2026-11-17.log.
   Results (median addon / gbt_sel / gbt_all):
   - Logistic BSS sustained 15/20/30: 0.57/0.59/0.61, 0.56/0.56/0.57, 0.27/0.22/0.24;
     gust 30/40/50/60: 0.50/0.51/0.54, 0.40/0.41/0.45, 0.29/0.28/0.30, 0.12/0.08/0.10.
     gbt_all beats the add-on in 85-100% of models at common thresholds, 53-77% at rare ones.
   - Linear R2 sustained 0.70/0.75/0.78, gust 0.64/0.68/0.72; gbt_all better in every model.
   - Top-end over-forecast largely fixed: of the 25 step-4a flagged models, top ratio < 0.85 in
     18 (addon) / 9 (gbt_sel) / 2 (gbt_all); median ratio 0.82 / 0.90 / 0.96. Supports the model-form
     diagnosis (interactions / saturation).
   - Weak spot: rare thresholds (gust 60, sustained 30). Trees UNDER-forecast at the top (gbt_all ratio
     > 1.15 in 7 of 13 gust-60 models) and BSS is lower than the add-on. Few events per inner
     validation set -> early stopping keeps the trees shallow and shrunk toward the base rate.
   - Gain importance (gbt_all): |grad| 0.25, SAN-LAS 0.14, LAX-DAG 0.10, NID-RNO 0.06, then many
     small contributions, so the trees use more than the 3-pair cap.
5b. Tree-shape tuning for rare thresholds - 2026-11-17. `scripts/fit_gbt_tuned.py` ->
   `data/v2/derived/models/gbt_tuned/` (separate folder; gbt/ and regression outputs untouched; rows
   asserted identical). gbt_all features; scope gust 50/60 and sustained 30 (53 models). Inside each
   outer fold: 3 interleaved year-group inner CV over num_leaves {4,7,15} x min_data_in_leaf {20,50,100},
   early stopping on mean inner log loss; best shape + rounds refit on all training years.
   Choices: mostly small trees (4 leaves in 368 of 583 fold fits), median 177 rounds (vs ~60-100 before).
   Results (median addon / gbt_all / gbt_tuned):
   - gust 50: BSS 0.29/0.30/0.30, top ratio 0.90/0.96/0.94, < 0.85: 9/1/2, > 1.15: 0/2/0.
   - gust 60: BSS 0.12/0.10/0.10, top ratio 0.83/1.28/0.94, < 0.85: 9/0/4, > 1.15: 0/7/1.
   - sustained 30: BSS 0.27/0.24/0.26, top ratio 0.91/1.04/1.00, < 0.85: 4/1/1, > 1.15: 0/4/1.
   - Tuned beats the add-on in 74% / 69% / 59% of models (gust 50 / 60, sustained 30).
   => Tuning fixes the top-end calibration of the rare-event trees (the gust-60 under-forecast is gone)
   but adds little Brier skill; at these event counts the trees and the add-on are close. AUC ~0.96-0.98
   for all three.
   Figures 2026-11-17: `scripts/plot_gbt_compare.py` -> `images/models/gbt_compare/` (overview of all
   three model families by threshold; tuned-GBT reliability pooled and per gradient set).
   Decision 2026-11-17: the user is satisfied with these results (the high-end winds are the
   high-impact ones) and takes the tuned GBT as the chosen model family.
6. Speed (point-forecast) models - tuning DONE 2026-11-18; next step awaiting the user's choice. Targets
   unchanged (sustained = set P75, gust = nearest-rank P90 with 2-station floor; identical to the
   event-model values). Out-of-fold RMSE (2015-2025, leave-one-year-out, identical hours), linear add-on /
   GBT / tuned GBT: sustained 3.29 / 2.77 / 2.76 mph, gust 6.08 / 5.11 / 5.09 mph.
   - `scripts/fit_gbt_tuned.py --speed` tunes the GBT speed regressors (inner year-group CV over
     num_leaves {7,15,31,63} x min_data_in_leaf {20,50,100}) -> `derived/models/gbt_tuned_speed/`.
     Full run: 56 models, ~3 h 7 min at 14 workers (the earlier ~50 min estimate was wrong). Mostly picks
     31-63 leaves. Gain vs fixed GBT is noise: per set mean -0.016 mph RMSE (range -0.09 to +0.07).
     Not overfitting (held-out-year scores are not worse), but tree shape is not the limit.
   - All models under-predict strong winds; tuning does not change this. Bias by observed bin
     (linear / GBT / tuned): gust >= 70 mph -22.1 / -20.0 / -19.7, gust 55-60 -10.1 / -8.1 / -8.0,
     sustained >= 40 mph -13.3 / -11.2 / -10.9. This is the squared-error loss shrinking rare extremes.
   - `scripts/plot_speed_compare.py` (run 2026-11-18): RMSE, bias and hours by 5 mph bin, binned by
     observed and by forecast speed -> `images/models/speed_compare/rmse_by_speed_{pooled,sustained,gust}.png`,
     numbers -> `derived/models/speed_compare/error_by_bin.csv`. Known cosmetic issue: x tick labels
     overlap in the gust panels of the pooled figure (15 bins); rotate if the figures are kept.
   - DECIDED 2026-11-21 (user): the fixed-shape GBT (gbt_all) is the speed model; speed tuning dropped.
     Strong-wind under-prediction is addressed next.
   - Options for the next step (user to decide): quantile mapping to fix strong-wind under-prediction;
     10th/50th/90th percentile range forecasts (linear quantile regression + LightGBM quantile); an improved
     linear model with a curve on |gradient|; scoring for strong-wind hours.
   - Strong-wind under-prediction investigation (2026-11-19): `scripts/plot_error_boxes.py` -> box plots of
     GBT (gbt_all) error by observed speed, |gradient| and observed wind direction (16 sectors), pooled ->
     `images/models/error_boxes/error_boxes_pooled.png`, stats -> `derived/models/error_boxes/error_box_stats.csv`.
     Finding: error grows strongly with OBSERVED speed but is flat (mean ~0) across |gradient| bins and
     wind-direction sectors; only the >= 12 hPa bin dips slightly (gust mean -0.6 mph). Under-prediction is
     conditional on the outcome (hours windier than their gradient implies), not a bias at strong gradients.
   - Strong-hour misses (2026-11-19): `scripts/plot_strong_misses.py` -> `images/models/strong_misses/`,
     tables `derived/models/strong_misses/` (strong_hours.parquet, by_set.csv, bss_by_bin.csv). Strong =
     sustained > 40 / gust > 50 mph observed; miss = |GBT error| > 5 mph.
     Sustained: only 288 strong h, 91% missed (all under), mean error -11.2 mph; too few to break down.
     Gust: 21,659 strong h, 63% missed (13,402 under, 324 over). Miss share falls steadily with |gradient|
     (~85-95% below 4 hPa, mean error -12.8; ~35-45% at >= 13 hPa, mean error -4.9 at >= 10). No hour-of-day
     signal. Jul-Sep ~90% missed but few hours. Worst sets: SFO-WMC+/-, SFO-SAC+, SAC-RBL+, SAC-RNO+/-
     (-12 to -15 mph); best SFO-ACV+, SAC-MFR+, MFR-RDD- (about -4).
     Event-model BSS by |gradient| is lowest at 2-5 hPa and rises with gradient; BSS by observed speed shows
     mean probabilities that rise too slowly past the threshold (gust 60: ~0.25 at 70 mph).
   - SFO-WMC+ station check (2026-11-19): `scripts/plot_sfo_wmc_stations.py` -> `images/case_studies/sfo_wmc/`
     (station_timeseries_full, station_timeseries_events, station_map). The gust set is effectively a West
     Reno / Verdi cluster (6 of 10 stations within ~10 km, 1,490-1,650 m) plus Tahoe-area peaks; it sits
     ~300 km from SFO and ~200 km from WMC. NV052 (South Verdi, from 2022-07) and NV013 (NW Reno) are the
     windiest (p99 57 / 53 mph vs 34-46) but track their neighbours (hourly r 0.74-0.86 with NV052) and in
     the 6 worst events most stations peak together: the misses are real, widespread Reno-area wind events,
     not errant stations. Several have SFO-WMC near 0 hPa. DNMOG reports exact-zero gusts ~4x as often as
     its neighbours (3.6% vs 0.9%). Next: ERA5 maps for the worst events (paused for this check).
   - ERA5 case maps (2026-11-19): `scripts/era5_fetch_events.py` (CDS; day files era5_sfc_/era5_ua_YYYYMMDD.nc in
     /mnt/d/ERA5, 30-50N 135-110W, sfc T2m/MSLP/10 m wind/gust, 850/700/500 z/T/u/v) and
     `scripts/plot_era5_event_maps.py` -> `images/case_studies/sfo_wmc/era5/` (6 events at -12/-6/0 h + composite).
     Common pattern: at every peak the 700 hPa flow over Reno is SW (205-252 deg) at 32-65 kt (mean ~233/50 kt),
     usually strengthening through the 12 h before, with 500 hPa 44-100 kt; trough/jet to the north-west. Consistent
     with Sierra downslope / mountain-wave wind, which the sea-level SFO-WMC gradient does not measure.
     Caveat: 6 events, no comparison group yet (how often is 700 hPa SW >= 40 kt without strong gusts?).
   - SAN-IPL station check (2026-11-19): `scripts/plot_set_stations.py <signed gradient>` (general version of the
     SFO-WMC check; any set size) -> `images/case_studies/san_ipl_{pos,neg}/`, tables
     `derived/models/case_studies/san_ipl_{pos,neg}_stations.csv`. "Setters" = stations ranked 1..k in an hour.
     SAN-IPL+ (34 stations) spans SoCal to the Bay Area and one WA site (PS004, 1,623 km): rho >= 0.5 lets in
     chance correlations. With ~19 reporting, k = 2, so the target is the 2nd-highest gust in the whole set.
     Setters in misses: WWAC1 Whitewater 80%, JWBC1 Jawbone (327 km) 50%, TNSC1 Mountain Springs 36%, LPDC1
     Lake Palmdale (252 km) 15%, AATC1 Altamont (723 km) 6%. The 2 worst misses (2023-01-08, 2024-02-05) were
     set by Altamont + Jepson Prairie (NorCal storms). One (2023-07-31) is a sudden summer spike (likely
     convective outflow).
     Ruled out as general explanations: (1) distant setters: a setter > 200 km in 68% of missed vs 73% of
     well-forecast strong hours, so not specific to misses; (2) network change: core setters report 2015-2026
     and misses spread over all years (bias drifts -4 to -8.5 mph, noisy). SAN-IPL- (16 stations, qualifies from
     2020): setters are LA/Ventura/Inland Empire Santa Ana sites (Wiley Ridge 54%, Highland Springs 51%, Chino
     Hills 33%); HRICH (WA, 105 h) never sets.
     Design issue to revisit: station-set selection needs a distance / region constraint (and a minimum n). DEFERRED 2026-11-21: the user wants
     to think about the approach; until then, displaced stations are ignored case by case.
   - Speed model DECIDED 2026-11-21: fixed-shape GBT (gbt_all); see step 6.
   - Reno 700 hPa series (2026-11-21): `scripts/era5_fetch_reno700.py` -> /mnt/d/ERA5/reno700/era5_ua700_YYYYMM.nc
     (hourly u/v 700 hPa, 3x3 0.25 deg box centred 39.50N 120.00W, 2015-2025, one CDS request per month).
   - Reno 700 hPa vs SFO-WMC+ gust error (2026-11-21): `scripts/plot_reno700_relationship.py` ->
     `images/case_studies/sfo_wmc/reno700/reno700_vs_error.png`, numbers -> `derived/models/case_studies/
     sfo_wmc_reno700_{bins,grad_x_w700}.csv`. 21,556 gust hours. Target > 50 mph in 0% of hours below 20 kt aloft,
     3.5% at 30-40, 10% at 40-50, 31% at 50-60, 51% at >= 60 kt; GBT forecasts > 50 mph in 0 / 0 / 1.7 / 10 / 14%.
     79% of strong hours and 78% of missed strong hours have >= 40 kt (vs 10% of all hours); nearly all are SW.
     Error grows with 700 hPa speed in every |gradient| bin (0-4 hPa: about -7.5 mph at 50-60 kt); corr(error,
     w700) = -0.28. SW >= 40 kt alone is not enough (only 20% of those hours exceed 50 mph).
     CONCLUSION / CLOSED (user, 2026-11-21): SFO-WMC+ gust misses are explained by Sierra downslope / mountain-wave
     wind driven by strong SW flow aloft, which the sea-level gradient does not measure. Informational only:
     700 hPa wind is NOT added to the model (needs NWP input; the model stays gradient-only).
   - SAN-IPL+ regimes (2026-11-21). Strong-hour breakdown (exploratory, not a saved script): 2,101 under-forecast
     strong-gust hours in 339 episodes (gap > 24 h), 58% of 3,626 strong hours. Miss share is high for every
     setter area (top station: San Gorgonio 55%, Mojave/Jawbone 56%, San Diego mtns 69%, Antelope Valley 73%) and
     season (MAM 53% .. DJF 66%); it depends mainly on gradient (91% < 3 hPa, 75% 3-6, 54% 6-9, 41% >= 9).
     Forecasts are compressed vs obs at every gradient (3-6 hPa: obs > 50 mph 3.0% of hours, forecast 0.9%).
     So the misses look like a missing predictor more than a few discrete regimes.
     ERA5 maps: `era5_fetch_events.py` and `plot_era5_event_maps.py` now take `--set SAN-IPL+ --dates ...`
     (worst miss hour on each date) -> `images/case_studies/san_ipl_pos/era5/`. Four cases, three types:
       (a) 2017-05-26 10Z, WWAC1 74 mph, obs gradient +8.1 (ERA5 +7.9): late-May onshore surge into a deep
           desert heat low (~996 hPa); weak flow aloft (700 hPa ~20 kt). Gradient captured, magnitude missed.
       (b) 2023-01-03 14Z (TNSC1 63, WWAC1 62, gradient +2.6) and 2025-12-03 09Z (WWAC1 65, TNSC1 59, +4.6):
           strong W-NW flow aloft over SoCal (700 hPa 33-41 kt at Whitewater, 500 hPa jet 80-100 kt; 2025 an
           inside-slider trough from the north) with a weak-moderate surface gradient. Momentum aloft / mountain
           wave, the same mechanism as the Reno (SFO-WMC+) misses.
       (c) 2023-07-31 11Z (FHCC1 Fish Creek Mtns 54, WWAC1 53, gradient +1.4): summer, flow aloft < 10 kt,
           ERA5 Whitewater gust 5 kt. Convective outflow (monsoon); not resolvable from these fields.
     (A systematic 700 hPa test for SoCal was considered but not pursued: the user keeps the model gradient-only.)
   - SAN-IPL- regimes (2026-11-21). Set qualifies from 2020-10: 422 strong-gust hours, 237 under-forecast (56%),
     42 episodes; miss share 67% at < 3 hPa vs 42% at 3-6 hPa; DJF/SON almost entirely. Top stations: Highland
     Springs, Wiley Ridge, Chino Hills, Miller Ranch (HRICH, WA, never sets). ERA5 maps (6 worst, distinct
     episodes) -> `images/case_studies/san_ipl_neg/era5/` (map extended to 43N for the Great Basin high).
       Santa Ana, 4 of 6: 2021-01-19 20Z (78 mph, obs gradient 3.4; closed low dropping down the coast),
           2025-01-08 16Z (66 mph, 0.3; Palisades/Eaton fire windstorm), 2025-01-21 03Z (65 mph, 2.3),
           2024-03-14 22Z (57 mph, 0.7; cut-off low overhead, northerly flow). Great Basin high 1032-1044 hPa,
           NE-E flow aloft (700 hPa 27-55 kt at Highland Springs in 3 of 4). SAN-IPL runs E-W along the border, south
           of the main Great Basin-to-coast pressure gradient, so it reads only 0.3-3.7 hPa in strong Santa Anas.
       Not offshore, 2 of 6: 2025-12-24 15Z (58 mph, Miller Ranch 68): pre-frontal S-SW wind ahead of a deep Pacific
           low (~988 hPa off NorCal; 700 hPa 200 deg / 48 kt); SAN-IPL negative from falling pressure to the west.
           2023-08-20 16Z (60 mph): Tropical Storm Hilary (SE 700 hPa 48 kt).
     Reading: the negative sign mixes Santa Anas with pre-frontal and tropical southerly events, and in Santa Anas
     the gradient is a weak proxy. Explains the error; no model change.
