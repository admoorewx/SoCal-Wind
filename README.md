# SoCal-Wind

Statistical models of strong surface winds in California and the West Coast, driven by
operational surface **pressure gradients** (for example LAX–DAG or SAN–LAS). For each gradient,
the project finds the stations whose winds track it, defines regional wind events and targets, and
fits models that predict:

- **Event probability:** sustained wind ≥ 15/20/30 mph, or gust ≥ 30/40/50/60 mph, over the
  gradient's station set.
- **Wind speed:** the regional sustained (75th percentile) and gust (90th percentile) speed.

Observations come from the [Synoptic Data](https://synopticdata.com/) API: ASOS/AWOS and
mesonet stations in CA, NV, OR and WA, hourly, 2015 onward.

The repo holds code and documentation only. The data (~22 GB), figures and model files are
not included and can be regenerated with the scripts.

## Pipeline

Run the scripts from `scripts/` in this order. Paths are relative to the repo root.

| Step | Script | What it does | Output |
|---|---|---|---|
| 1. Download | `synoptic_driver.py` → `synoptic_bulk_download.py` | Hourly wind and pressure for every station in CA/NV/OR/WA, month by month | `data/v2/obs_hourly/`, `stations.parquet`, `raw/` |
| 2. Raw tables | `build_obs_raw.py` | Full-resolution parquet tables built from the saved raw responses | `data/v2/obs_raw/` |
| 3. Wind QC | `wind_qc.py fit / apply / evaluate` | Flags bad readings using only each station's own record (constant runs, spikes, noise, level shifts, range) and excludes stations with ≥ 5% masked | `data/v2/qc/wind/`, `station_exclusions.csv` |
| 4. Gradients | `gradient_pipeline.py timeseries / match / correlate / maps` | Gradient time series from `gradients.json`, QC-filtered station winds matched to gradient times, and Spearman correlation per signed gradient and station | `data/v2/derived/gradients/` |
| 5. Station sets | `build_station_sets.py` | Stations representing each signed gradient: ρ ≥ 0.5, topped up to 10 stations from ρ ≥ 0.4 | `derived/models/station_sets.csv` |
| 6. Targets | `build_targets.py` | Hourly event flags and regional targets; 2015–2025 for training, 2026 held out | `derived/models/targets/` |
| 7. Baseline | `fit_baseline_models.py` | Logistic and linear models on \|gradient\| + hour + day of year | `derived/models/baseline/` |
| 8. Add-on | `fit_gradient_addon.py` | Baseline plus up to 3 other gradients, chosen by BIC inside each fold | `derived/models/gradient_addon/` |
| 9. Trees | `fit_gbt_models.py` | LightGBM on the same rows and folds: the add-on's gradients (`gbt_sel`) or all gradients (`gbt_all`) | `derived/models/gbt/` |
| 10. Tuned trees | `fit_gbt_tuned.py` | Tree shape tuned by inner year-group cross-validation (rare thresholds by default; `--all` for every model) | `derived/models/gbt_tuned/` |

The diagnostic and figure scripts are `diagnose_overforecast.py`, `plot_baseline_models.py`,
`plot_gradient_addon.py`, `plot_gbt_compare.py`, `plot_station_set_scatter.py` and
`gradient_influence_area.py`. Scripts named `explore_*.py` and `verify_suspects.py` were used
for one-off data exploration and QC checks. Every script describes its inputs, outputs and usage
in its docstring.

## Validation design

- **Holdout:** all of 2026 is held out. Stations are selected with correlations from 2015–2025 only
  (`gradient_pipeline.py correlate --before 2026-01-01`).
- **Cross-validation:** leave-one-year-out over 2015–2025. Every model is scored on the same hours,
  and the tree scripts check this.
- **Metrics:** Brier skill score, ROC AUC and log loss for probabilities; R² and RMSE for speeds.
  Top-end calibration is measured by the **top-end ratio** (observed event rate ÷ mean forecast
  over the top 2% of forecasts; 1 means calibrated).
- **No leakage:** predictor selection, early stopping and tree-shape tuning use the training years
  of each fold only.

## Current results (cross-validation, 2015–2025)

| | Add-on regression | GBT (all gradients) | Tuned GBT |
|---|---|---|---|
| Linear R², sustained / gust | 0.70 / 0.64 | 0.78 / 0.72 | n/a |
| BSS, gust ≥ 40 mph | 0.40 | 0.45 | n/a |
| BSS, gust ≥ 60 mph | 0.12 | 0.10 | 0.10 |
| Models over-forecasting at the top (ratio < 0.85), of the 25 flagged in step 4a | 18 | 2 | n/a |
| Gust ≥ 60 models with top-end ratio outside 0.85–1.15 | 9 of 13 | 7 of 13 | 5 of 13 |

The trees improve speed prediction and event probabilities at the common thresholds, and remove
most of the regression model's top-end over-forecasting. At the rare thresholds, tuning the
tree shape gives the best top-end calibration with similar Brier skill. Details, decisions and
dates are in [`docs/wind_model_plan.md`](docs/wind_model_plan.md).

## Data decisions

- **Excluded networks:** WWG networks are excluded because their stations drop in and out in
  blocks and vanish after 2021. Portable stations are excluded because they have a single stored
  location.
- **Wind QC:** each station is judged only on its own record, so terrain-driven local winds aren't
  flagged as errors. See [`docs/wind_qc_method.md`](docs/wind_qc_method.md).
- **Missing gusts:** never filled with sustained speeds. Gust hours require ≥ 50% of the set's
  stations to report a gust.

## Setup

Python 3.14. Install the dependencies with:

```bash
python -m venv ~/.venv/analysis_env
~/.venv/analysis_env/bin/pip install -r requirements.txt
```

Downloading data needs a Synoptic API token. Put it in `scripts/synoptic_api.txt.txt`, or pass
`--token-file`. The file is gitignored; never commit it.

```bash
cd scripts
~/.venv/analysis_env/bin/python synoptic_driver.py        # download missing months
~/.venv/analysis_env/bin/python wind_qc.py fit && ~/.venv/analysis_env/bin/python wind_qc.py apply
~/.venv/analysis_env/bin/python gradient_pipeline.py all
# then steps 5-10 from the table above
```

## Layout

```
scripts/   pipeline, modelling, diagnostics and plotting code; gradients.json lists the pairs
docs/      method notes and the running record of model decisions
data/      (not in repo) raw and derived data
images/    (not in repo) figures
```
