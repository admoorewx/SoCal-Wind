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
