# Wind observation QC: method, calibration and decisions

Status: in use from 2026-11-12. Code: `scripts/wind_qc.py` (CONFIG `version: 1`).
Scope: sustained wind (`wind_speed_mps`) and gust (`wind_gust_mps`) in the Synoptic
full-resolution observations (`data/v2/obs_raw/`), CA/NV/OR/WA, 2015-01 to 2026-10.

This document records what the QC does, every threshold and why it was chosen, what was tried
and rejected, the results, and the known limitations, so that the method can be reviewed and
re-run later. Numbers are from the run of 2026-11-12, unless a section says otherwise.

---

## 1. Summary

* Each observation is checked against **its own station's time series and statistics only**.
  Comparisons with neighbouring stations are computed and stored, but they are information
  only and never remove data (section 3).
* **Observation flags.** Six automatic rules plus one manual rule remove readings: CONST,
  SPIKE, NOISE, SHIFT, LEVEL, RANGE and MANUAL. Each reading carries a bitmask, so the rule
  that removed it is always visible.
* **Station exclusion.** If the automatic rules mask at least **5%** of a station's sustained
  readings, the whole station is dropped from analysis. 9 of 2,565 stations are excluded
  (stations with fewer than 1,000 sustained readings aren't assessed).
* **Manual exclusion.** KWSIX (Kirkwood Chair 6 Ridge) is masked from 2025-12-01 onward,
  after visual review.
* **Effect.**

  | | Raw | After QC |
  |---|---|---|
  | Sustained readings masked | — | 0.062% (322,224 of 517.9M) |
  | Sustained readings > 40 m/s | 69,072 | 3,442 |
  | Sustained readings > 25 m/s | 164,408 | 43,312 |
  | Gust readings > 40 m/s | 32,242 | 11,046 |

  These "after" counts exclude masked readings but still include excluded stations; dropping
  those stations as well leaves 3,381 readings > 40 m/s and 42,278 > 25 m/s
  (see section 8.2).
* **Validation.** Readings above 25 m/s were split by the stations' earlier classification
  (section 8.3):

  | Earlier classification | Share kept |
  |---|---|
  | Real, weather-like stations | 92.5% |
  | Spike-prone stations | 24.6% |
  | Stuck / broken stations | 5.8% |

---

## 2. Inputs and pre-processing (upstream of this QC)

| Item | Source / rule |
|---|---|
| Observations | `data/v2/obs_raw/YYYY/MM/<STATE>.parquet`, built by `scripts/build_obs_raw.py`. One row per station per report time; columns `stid, time_utc, wind_speed_mps, wind_direction_deg, wind_gust_mps, pressure_hpa, sea_level_pressure_hpa`. |
| Upstream cleaning already applied | Synoptic-QC-flagged values removed; primary sensor set preferred; duplicate (stid, time) collapsed; rows with all variables missing dropped (see `build_obs_raw.py`). |
| Station metadata | `data/v2/stations.parquet` (lat, lon, elevation_m, network_shortname). |
| Networks excluded entirely | Every network whose `network_shortname` starts with `WWG` (WWG-SONOMA, WWG-LLWC, WWG-MISC). Their stations appear and disappear in blocks of 50-150 (decided 2026-11-09, `scripts/explore_station_networks.py`). |
| Units | m/s throughout. |
| Volume | 517,935,419 sustained and 304,060,505 gust readings; 4,721 non-WWG stations in metadata, 2,601 of them with sustained wind data. |

Other network decisions in force: the utility networks PGN and PSE are kept (decided
2026-11-12). Gust correlations at ASOS stations use only hours with a reported gust, which is
accepted (decided 2026-11-12).

---

## 3. Design principles

1. **Self-based flagging.** Neighbour comparisons can't tell a sensor that is biased high
   from a station deliberately sited to sample a local terrain effect (ridge tops, passes,
   gap winds). Both read much higher than nearby stations, and this is common in mountains.
   The checks showed this directly: the neighbour-based rules flagged real exposed sites such
   as San Marcos Pass, Round Mountain and Eagle Peak (section 10.4). So the user decided
   (2026-11-12) that only rules using the station's own record may remove data.
2. **Two levels.** Suspect *observations* are flagged; stations with a *significant share*
   of flagged observations are omitted entirely.
3. **Flags, not deletion.** `obs_raw` is never modified. Flags are stored separately as
   bitmasks; the masking happens when the data is loaded (`load_month_qc`). Rules can
   therefore be audited, switched on or off (`mask_rules`) and recomputed.
4. **Frozen reference statistics.** Per-station statistics (spike percentiles, typical
   monthly median) are fitted once (`fit`) and frozen, so that new months are judged against
   the same references. They change only when `fit` is rerun on purpose.
5. **Calibrated against labelled examples.** Thresholds were chosen with:
   * known-bad stations, confirmed by eye: MDTCA, DDRIO, DDBUF, CLE, KEE, BEUO, CPPW, OWY,
     CSPW, MCUES, LALN2, OOSC1, PVPN2
   * known-good high-wind stations: SLIDE, HVRTM, HVRGM, BPKEN, WTFO3, BPKC1, FMRC1, BIJC1,
     JWBC1, SJS02, HP016, KMHV

   Neighbours were used only for this calibration and checking, never in the masking rules
   themselves.

---

## 4. Definitions used by all rules

| Term | Definition |
|---|---|
| Reading | One non-missing, non-negative value of a variable at a station. Sustained and gust are treated separately unless stated. |
| Consecutive reports | Two readings of the same station and variable, adjacent in time, with a gap of at most `max_gap_min` = **120 min**. A larger gap breaks every run and pair. |
| Day / month | UTC calendar day / calendar month. |
| Processing unit | One calendar month, all states together. Pairs and runs that cross a month boundary are split; the effect is negligible. |
| Masked | The reading carries at least one bit in `mask_rules`; `load_month_qc` sets it to NaN. |
| Automatic mask | `mask_rules` minus MANUAL; used for the station-exclusion share. |

---

## 5. Observation rules

### 5.0 Bit table

| Bit | Name | Masks? | Scope | Based on |
|---|---|---|---|---|
| 1 | CONST | yes | reading (run) | station's own series |
| 2 | SPIKE | yes | reading | station's own series |
| 64 | NOISE | yes | station-day, both variables | station's own series (sustained) |
| 32 | SHIFT | yes | station-month, both variables | station's own record (sustained) |
| 128 | LEVEL | yes | whole station, both variables | station's own record (sustained) |
| 8 | RANGE | yes | reading | physical limit |
| 256 | MANUAL | yes | station period, both variables | visual review |
| 4 | NEIGH | **no** (info) | station-day, both variables | neighbours |
| 16 | DECOR | **no** (info) | station-month, both variables | neighbours |

Flags are stored as `uint16`. Several bits can be set on one reading. For bits whose scope is
a day, month or station, the bit goes only on readings where that variable exists.

### 5.1 CONST: constant high value (bit 1)

**Definition.** A run is a maximal sequence of consecutive reports with identical value,
compared after rounding to 0.01 m/s. All readings in a run are flagged when:

* value >= `const_min_mps` = **20 m/s**, and
* the run has >= `const_min_reports` = **3** reports, and
* the run lasts >= `const_min_minutes` = **60 min**, measured from its first to its last
  report.

**Why these values.** `fit` tabulated every run of 2 or more identical reports at every
station, by value and duration (`const_calibration.png`, `const_runs_hist_*.npy`):

| Threshold (with >= 3 reports and >= 60 min) | Stations | Readings | Notes |
|---|---|---|---|
| 15 m/s | 192 | 35.8k | Many ordinary hourly stations repeating a whole-mph value (34 mph = 15.2 m/s) for 2-3 hours |
| **20 m/s** | **49** | **28.7k** | Dominated by known-bad stations: OWY, KEE, CLE, MCUES, MDTCA, BEUO |

* The minimum of 3 reports stops a pair of identical hourly reports (which spans 60 min) from
  counting as a run.
* Calm (zero) runs are not flagged; the rule targets constant *high* values.

**Result.** 28,742 sustained readings at 49 stations; 27,052 gust readings at 309 stations.

### 5.2 SPIKE: up-and-back jump (bit 2)

**Definition.** For reading *i* with consecutive reports on both sides:

* d_prev = v_i − v_(i−1)
* d_next = v_i − v_(i+1)

The reading is flagged when both differences exceed `thr` with the same sign: either both
are > thr (a spike up) or both are < −thr (a drop-out).

* `thr` = the station's `spike_pct` = **99.9th** percentile of |change between consecutive
  reports| for that variable, clipped to [`spike_floor_mps`, `spike_cap_mps`] =
  [**10, 15**] m/s.
* Percentiles are fitted by `fit` over all data (2015-01 to 2026-10) and frozen
  (`thresholds.parquet`).
* A station with fewer than 500 pairs uses its network's median percentile, then the global
  median.
* The reading after a spike is not flagged (`spike_mode = "two_sided"`).

**In practice the floor sets the threshold.** The median station p99.9 is 4.1 m/s for
sustained wind. 97.4% of stations sit at the 10 m/s floor and 0.5% at the 15 m/s cap.

**How the rule was chosen** (`spike_calibration.png`, `eval_spike_variants.csv`; 12 winter
months). The yardstick was corroboration: the share of readings for which some neighbour (up
to 5 within 75 km and ±400 m, hourly max) reached at least half the value in the same hour. A
useful spike rule should flag readings that are corroborated *less often* than unflagged
readings of the same speed.

| Variant | % of 15-25 m/s readings flagged | Corroboration of flagged vs all readings (10-20 m/s) | Verdict |
|---|---|---|---|
| Change from **prior** report > station **p95**, as first proposed | about 18-19% | same as all readings (73/59% vs 74/59%) | **Rejected.** Removes a fifth of strong-wind readings and has no skill: it flags real gustiness. |
| Two-sided, p95, no floor | about 5.7% | same as all readings | Rejected, no skill |
| Two-sided, p99.9, no floor | about 0.35-0.4% | slightly lower | Rejected, weak |
| **Two-sided, p99.9 clipped to [10, 15] m/s** | **0.15-0.34%** | **9-22% vs 59-74%** | **Adopted** |

* **Why a cap.** At stations that spike often, their own spikes inflate the percentile: LALN2's
  p99.9 was 42 m/s, so its 40-63 m/s spikes passed. The cap fixes this.
* **Result.** 6,589 sustained readings at 485 stations; 12,011 gust readings at 1,181
  stations.
* **Gusts.** Gusts use their own fitted percentiles but the same floor and cap; they were not
  calibrated separately (section 11).

### 5.3 NOISE: erratic day (bit 64)

**Definition.** For each station and UTC day, from sustained readings:

* jumps = consecutive pairs with |change| > the station's spike `thr` (same clipping as SPIKE)
* pairs = all consecutive pairs that day

The day is flagged when jumps >= `noise_min_jumps` = **5** and jumps >= `noise_min_share` =
**5%** of pairs. Every reading that day, sustained and gust, gets the bit.

**Why.** Some sensors produce bursts of noise lasting hours to days, such as MCUES jumping
34 → 62 → 53 → 79 m/s in 1-minute data. These are neither single up-and-back spikes nor
constant runs. Candidate thresholds, counted over all station-days with at least 1 jump:

| Rule | Days flagged | Stations | Days at known-good stations |
|---|---|---|---|
| ≥ 3 jumps, ≥ 2% | 1,276 | 156 | 66 |
| **≥ 5 jumps, ≥ 5%** | **387** | **59** | **9** |
| ≥ 10 jumps, ≥ 10% | 78 | 18 | 2 (misses most MDTCA/OOSC1 days) |

The stations most affected are PVPN2, CPPW, OOSC1, MDTCA and MCUES, all confirmed noisy
(`self_rules_calibration.png`, middle panel).

**Result.** 59,459 sustained readings at 59 stations; 18,057 gust readings at 37 stations.

### 5.4 SHIFT: level shift relative to the station's own normal (bit 32)

**Definition.**

* Typical monthly median = the median of the station's monthly medians of sustained wind,
  over months with >= `shift_min_reports` = **100** readings. Fitted by `fit` and frozen
  (`station_level.parquet`).
* A month (with >= 100 readings) is flagged when its median >
  `shift_ratio` = **4** × max(typical monthly median, 0.5 m/s) **and** >
  `shift_min_mps` = **10 m/s**.
* Every reading in the month, sustained and gust, gets the bit.

**Why.** Some sensors read far above their own normal for months: MDTCA Nov 2016 to Dec 2017,
and DDRIO/DDBUF Jul 2023 to Oct 2024. Single readings in these stretches look plausible, but
the month as a whole does not. Candidate thresholds:

| Rule | Months | Stations | Months at known-good stations |
|---|---|---|---|
| ratio > 3, median > 8 | 94 | 21 | 0 |
| **ratio > 4, median > 10** | **73** | **8** | **0** |
| ratio > 5, median > 15 | 54 | 5 | 0 |

Flagged months by station: CPPW 21, MDTCA 14, CSPW 13, DDRIO 12, DDBUF 10, SJS04 1, KNXF 1,
KWSIX 1. A month median above 10 m/s keeps calm stations, where a small typical level makes
the ratio large, from being flagged.

**Result.** 129,980 sustained readings at 8 stations; 23,720 gust readings at 5 stations.

### 5.5 LEVEL: implausible station (bit 128)

**Definition.** A station whose typical monthly median > `level_max_mps` = **15 m/s** has
every reading flagged.

**Why.** SHIFT needs a normal period to compare against. Stations that are wrong for their
*entire* record have none. The four that exceed 15 m/s are far from everything else:

| Station | Typical monthly median (m/s) | What the series looks like |
|---|---|---|
| CLE | 84.1 | 60-100 m/s |
| BEUO | 74.6 | staircase 35 → 46 → 53 → 100 |
| KEE | 45.6 | staircase 36-41 |
| OWY | 22.0 | smooth 21-23 band |
| *Next highest: WMSRS (White Mtn summit, 4,342 m)* | *11.2* | *plausible for the site* |

**Result.** All 40,709 sustained readings of BEUO, CLE, KEE and OWY.

### 5.6 RANGE: physical limit (bit 8)

**Definition.** Sustained >= `range_max_spd_mps` = **60 m/s**, or gust >=
`range_max_gust_mps` = **80 m/s**.

**Why.**
* It is the only rule that can catch a station with no comparable neighbours whose values are
  impossible from the start.
* It also gives a hard backstop for fill values, such as the recurring ~102.36 m/s at Las
  Vegas network sites.
* In the earlier same-hour analysis, no sustained reading above 40 m/s was ever matched by
  another station within 300 km and ±500 m: the highest regional value seen was 33 m/s.
* The limits were set well above that (60/80) so that summit extremes are not removed by this
  rule alone.

**Result.** 37,144 sustained readings at 67 stations; 569 gust readings at 14 stations.

### 5.7 MANUAL: reviewed station periods (bit 256)

**Definition.** Periods listed in `CONFIG["manual_exclusions"]`. Each entry has `stid`,
`start` (inclusive), `end` (exclusive; None = open) and a `reason`. All readings of both
variables in the period are flagged. MANUAL is **not** counted toward the station-exclusion
share.

**Current entries.**

| Station | Period | Reason |
|---|---|---|
| KWSIX (Kirkwood Chair 6 Ridge, 2,800 m) | 2025-12-01 onward | See below |

* **Before mid-2025 the station is real.** Compared with HVRTM (Heavenly ridge, 2,992 m,
  about 30 km), hourly means correlate with Spearman ρ = 0.63 and KWSIX reads a median 1.15×
  HVRTM's. Its 50th and 99th percentiles are 8.3/33.7 m/s against HVRTM's 6.8/21.0 m/s. That
  is consistent with a more exposed ridge, not a faulty sensor.
* **From December 2025 it is not.** The Dec 2025 monthly median was 36 m/s, while SLIDE
  (2,941 m) read 7.1 and XMPC1 (3,008 m) read 3.6. The 10-day review window shows 40-80 m/s
  sustained for 5 days.
* **The automatic rules catch only part of it.** SHIFT flags only Dec 2025. Jan-Apr 2026
  (monthly medians 21.1, 23.9, 17.5, 16.3 m/s, 2-3× its typical 8.2) pass the 4× ratio.
* **It may have recovered.** May-Oct 2026 medians are back to 6.3-11.6 m/s, so the station
  may have been repaired. The open-ended exclusion is the conservative choice. Review it
  (set an `end` date) if summer/fall 2026 KWSIX data is needed (section 11).
* Reviewed 2026-11-12 (`images/wind_qc/verify/KWSIX.png`).

**Result.** 89,396 sustained readings.

---

## 6. Station exclusion

**Definition.**
* share = sustained readings carrying any *automatic* masked bit ÷ the station's sustained
  readings.
* A station is excluded when share >= `station_exclude_share` = **5%** and it has >=
  `station_exclude_min_reports` = **1,000** sustained readings. The 36 stations with fewer
  readings are listed as `few_reports` and not assessed.
* Output: `station_exclusions.csv`. `load_month_qc(..., drop_excluded=True)` removes these
  stations.

**Distribution** (2,565 assessed stations; `station_exclusion.png`):

| Share masked | Stations |
|---|---|
| > 0 | 509 |
| ≥ 0.01% | 120 |
| ≥ 0.1% | 41 |
| ≥ 1% | 14 |
| **≥ 5% (excluded)** | **9** |
| ≥ 10% | 6 |
| ≥ 50% | 4 |

**Why 5%.** There is a natural gap: the lowest excluded station, MDTCA, is at 5.4%, and the
next station, PVPN2, is at 2.9%. Below the cut, every station's flags are short episodes
(noise days, spikes), not a failed sensor. In short, 2,056 stations have nothing masked, about
500 have a few suspect readings, and 9 are bad enough to drop.

**Excluded stations (9).**

| Station | Network | Share | Main rules | Visual verdict (`verify/<STID>.png`) |
|---|---|---|---|---|
| CLE | USBR HYDRO | 100% | LEVEL, RANGE, CONST | 60-100 m/s; not wind |
| OWY | USBR HYDRO | 100% | LEVEL, CONST | Smooth 21-23 band unrelated to nearby winds; not wind |
| KEE | USBR HYDRO | 100% | LEVEL, CONST | Staircase 36-41; not wind |
| BEUO | USBR HYDRO | 100% | LEVEL, RANGE, CONST | Staircase 35 → 100; not wind |
| DDRIO | DRI | 27.7% | SHIFT | Jul 2023 to Oct 2024: anti-correlated with nearest similar-elevation station DD2UR (ρ = −0.26), about 3.7× its level; pegs at 44.7 m/s (= 100 mph) |
| DDBUF | DRI | 23.7% | SHIFT | Same deployment period: ρ = −0.18 vs DD2UR, about 3.9× |
| CPPW | USBR HYDRO | 9.9% | SHIFT, NOISE | Random 0-100 m/s scatter while neighbours read 0-5 |
| CSPW | USBR HYDRO | 9.1% | SHIFT, CONST | Smooth daily cycle 11-14.4 that flattens at 14.4, consistent with battery voltage; not wind |
| MDTCA | MAP | 5.4% | SHIFT, RANGE, CONST | Smooth ramps up to a flat 99.93 ceiling, about 10× all neighbours, including a 1,440 m station 7 km away |

**The DDRIO/DDBUF unit check.** We tested whether these stations reported in mph but were
labelled m/s, which would make them real but mis-scaled. Real mis-scaled data would give a
ratio of about 2.24 and a positive correlation with the neighbour. The observed ratio was
about 3.8 with a negative correlation, so the idea was rejected. All of these stations'
values are whole mph; that is the normal DRI resolution, not evidence either way.

---

## 7. Neighbour diagnostics (information only)

These are computed in `apply`, written to the flag files and `station_month_all.parquet`,
and shown in `neigh_calibration.png`. They are **not in `mask_rules`**. Add their names to
`mask_rules` to make them remove data.

**Neighbours.**
* Up to `neigh_max` = 5 stations within `neigh_km` = 75 km and ±`neigh_dz_m` = 400 m
  elevation (`neighbours.parquet`). 4,681 of 4,721 stations have at least one.
* The station's hourly **mean** sustained wind is compared with the median of its neighbours'
  hourly means. Means are used because hourly *max* inflated 1-minute reporters (such as
  HPWREN) against hourly ones.

**NEIGH (bit 4), daily magnitude.**
* Definition: on a day with >= 6 joint hours, the station's 95th percentile is > 25 m/s
  and > 4× the neighbours' 95th percentile.
* Calibration: it flagged 66% of days at known-bad stations and 5% at known-good ridge
  stations. The earlier setting (> 20 m/s, > 3×) flagged every windy day at BPKC1, WTFO3,
  FMRC1, BIJC1 and JWBC1.
* Coverage: 61 stations, 158k sustained readings.
* It was the main working rule until the user chose self-based masking, and the self-based
  rules now cover its targets.

**DECOR (bit 16), monthly decorrelation.**
* Definition: in a month with >= 200 joint hours, Spearman ρ < 0.2 **and** the station's
  95th percentile > 3× the neighbours'.
* Of 13,939 station-months with ρ < 0.2, most are *calmer* than their neighbours. These are
  sheltered sites such as Cedar Grove, Cohasset and Brush Creek, and they are never flagged.
* Of 1,287 DECOR months, only 63 were at known-bad stations, and only 9 of those were not
  caught by another rule. The rest were real exposed sites with local wind regimes: Round
  Mountain, San Marcos Pass, Eagle Peak, Pike County Lookout (`inspect_decor_examples.png`).

---

## 8. Results

### 8.1 Readings per rule

| Rule | Sustained readings | Stations | Gust readings | Stations |
|---|---|---|---|---|
| CONST | 28,742 | 49 | 27,052 | 309 |
| SPIKE | 6,589 | 485 | 12,011 | 1,181 |
| NOISE | 59,459 | 59 | 18,057 | 37 |
| SHIFT | 129,980 | 8 | 23,720 | 5 |
| LEVEL | 40,709 | 4 | 1,971 | 4 |
| RANGE | 37,144 | 67 | 569 | 14 |
| MANUAL | 89,396 | 1 | 89,209 | 1 |
| **Any masked** | **322,224 (0.062%)** | | **167,144 (0.055%)** | |
| *NEIGH (info)* | *158,228* | *61* | *26,044* | *48* |
| *DECOR (info)* | *1,937,074* | *179* | *1,181,982* | *162* |

Rules overlap, so the column totals exceed "Any masked". Per-network effects are in
`summary_by_network.csv` and `before_after_spd.png` / `before_after_gust.png`. Networks
losing at least 0.1% of sustained readings to masking:

| Network | Masked |
|---|---|
| KIRKWOOD | 31.4% (MANUAL) |
| USBR HYDRO | 8.6% |
| MISC | 0.58% |
| REVWFO | 0.41% |
| LOXWFO | 0.27% |
| MAP | 0.24% |
| DRI | 0.14% |
| LKNWFO | 0.12% |

Every other network loses less than 0.1%. These figures count masked readings only; whole
excluded stations are dropped on top of this.

### 8.2 What remains above 40 m/s (sustained, masked readings and excluded stations removed)

3,381 readings at 34 stations:

| Station | Readings | Assessment |
|---|---|---|
| MCUES | 1,161 | Noise days only partly caught; station share 1.2% |
| KWSIX | 939 | Before Dec 2025; ridge winds up to 60 m/s |
| WMSRS | 926 | 4,342 m summit; no comparable station; likely real |
| OOSC1 | 118 | Erratic bursts not reaching the NOISE thresholds |
| SLIDE | 72 | Real (Slide Mtn, 2,941 m; corroborated by neighbours) |
| Others | ≤ 42 each | — |

Above 25 m/s, 42,278 readings remain at 181 stations. The largest are WMSRS, SLIDE, KWSIX
(before Dec 2025), MCUES, HRMTN, BPKEN and PVPN2, mostly real high-elevation winds.

### 8.3 Validation against the earlier station classification

The earlier classification is from `scripts/explore_suspect_winds.py` check 2
(`derived/qc/suspect/suspect_summary.csv`). Share of readings > 25 m/s kept after masking and
station exclusion:

| Earlier behaviour class | Readings > 25 m/s | Kept |
|---|---|---|
| Episodic (weather-like) | 18,019 | 92.5% |
| Elevated regime (months) | 34,635 | 54.0%, mainly KWSIX after Dec 2025 removed and WMSRS kept |
| Isolated spikes (noise) | 907 | 24.6% |
| Stuck (identical run) | 110,383 | 5.8% |

---

## 9. Visual verification (2026-11-12)

The 9 excluded stations, plus KWSIX and OOSC1, were reviewed with `scripts/verify_suspects.py`
(`images/wind_qc/verify/<STID>.png`). Each figure has:

* **left**: a terrain map with elevation contours (AWS Terrarium tiles, zoom 10, cached in
  `data/v2/derived/terrain/`), showing the 6 nearest stations of any elevation with their
  elevation and distance
* **right**: the 10-day window with the most masked readings, showing the station's readings
  (passed or masked, by rule) against the hourly means of the 4 nearest stations that
  reported then

Conclusions:
* **Excluded stations.** All nine are faulty or are not reporting wind (section 6 table).
* **KWSIX.** Real until mid-2025, degraded afterwards; handled by MANUAL (section 5.7).
* **OOSC1 (Sandstone Peak, 948 m).** Every nearby station is about 900 m lower, so no
  neighbour comparison is possible. The masked readings jump 5 → 90 → 10 m/s between
  consecutive reports, which is non-physical for sustained wind at any site. The station is
  kept (1.4% masked). A similar burst on 2019-02-28 (up to 60 m/s) passes (section 11).

---

## 10. Alternatives considered and rejected (decision log)

| # | Alternative | Outcome | Evidence |
|---|---|---|---|
| 10.1 | Spike = change from the prior report > station p95 (originally proposed) | Rejected: flags about 4% of all readings and about 19% of 15-25 m/s readings, no skill | §5.2, `spike_calibration.png` |
| 10.2 | Spike without a floor, at p95 or p99.9 | Rejected: same lack of skill | §5.2 |
| 10.3 | Spike without a cap | Rejected: spike-prone stations inflate their own threshold (LALN2 42 m/s) | §5.2 |
| 10.4 | Neighbour decorrelation alone (monthly ρ < 0.2) | Rejected: 6.7% of station-months flagged, 10% of RAWS; flags sheltered valley sites that read sensibly; misses MDTCA (garbage still rises and falls with storms, ρ 0.3-0.5) | `neigh_calibration.png`, `inspect_neigh_heavy_flagged.png` (made with the early rule set; shown for this point only) |
| 10.5 | Neighbour magnitude ratio alone (station p95 > 3× neighbours) | Rejected: the top hits are real exposed ridge RAWS stations (BPKC1, TRKW1, WTFO3) and coastal KCEC | diagnostics, 2026-11-12 |
| 10.6 | Monthly magnitude trigger | Rejected: too coarse for noise bursts lasting hours to days (MCUES); replaced by the daily NEIGH, now informational | — |
| 10.7 | Hourly max for neighbour comparisons | Rejected: inflates 1-minute reporters | §7 |
| 10.8 | NEIGH/DECOR as masking rules | Demoted to information by user decision (self-based principle) | §3, §7 |
| 10.9 | CONST at 15 m/s | Rejected: catches ordinary hourly whole-mph repeats | §5.1 |
| 10.10 | DDRIO/DDBUF as real data mis-scaled from mph | Rejected: ratio 3.8 with negative correlation (expected about 2.24 with positive) | §6 |
| 10.11 | Excluding KWSIX entirely | Rejected: about 6 years of good ridge data; MANUAL from Dec 2025 used instead | §5.7 |
| 10.12 | SHIFT ratio 2.5× to catch KWSIX Jan-Mar 2026 | Not adopted: effect on other stations untested; MANUAL used instead | §5.7 |

---

## 11. Known limitations and open items

1. **Gust thresholds are not calibrated separately.** Gusts use the sustained floor, cap and
   physical limits. Gust readings > 40 m/s fall only from 32,242 to 11,046. ASOS gusts are
   sparse (reported only when gusty), so consecutive-report changes mean something different
   for them.
2. **SPIKE is effectively a fixed 10 m/s up-and-back test** for 97% of stations, because
   station percentiles sit below the floor. The percentile only matters for the gustiest
   stations.
3. **NOISE misses some bursts.** For example OOSC1 on 2019-02-28, and parts of MCUES (1,161
   readings > 40 m/s remain).
4. **Untestable extremes.** WMSRS (4,342 m) has no station at a comparable elevation. Its
   25-60 m/s winter winds are kept.
5. **Frozen references.** Stations that first appear after the last `fit` have no
   SHIFT/LEVEL reference and use the network-median spike threshold until `fit` is rerun.
   Re-fitting changes thresholds for all stations, so it should be a deliberate, versioned
   step.
6. **SHIFT works on whole calendar months** and so can't catch a shorter shift. Equally, a
   SHIFT month removes every reading in it, including any plausible ones.
7. **Station exclusion uses sustained readings only.** Gust-only problems don't count toward
   the 5%.
8. **Upstream dependence.** The QC assumes `obs_raw` already has Synoptic-QC-flagged values
   removed. Any change to `build_obs_raw.py` needs `fit` and `apply` rerun.
9. **Calm/stuck-at-zero sensors are not flagged.** For example SGXWFO, where 41% of sustained
   readings are exactly 0, and HNXWFO/PQRWFO at 53-60%. They affect correlations, not the
   high-wind tail; not reviewed.
10. **KWSIX May-Oct 2026** may be good data (monthly medians 6.3-11.6 m/s) but is masked
    by the open-ended MANUAL entry.
11. **Map figures overlap labels** where stations are co-located (cosmetic).

---

## 12. Outputs and file formats

| Path | Contents |
|---|---|
| `data/v2/qc/wind/YYYY/MM.parquet` | Flagged readings only: `stid`, `time_utc`, `qc_spd`, `qc_gust` (uint16 bitmasks, section 5.0). Readings that aren't listed have no flags. |
| `data/v2/derived/qc/wind/config.json` | CONFIG used by the last `apply`. |
| `.../fit_info.json` | Months covered by `fit` and the CONFIG at fit time. |
| `.../thresholds.parquet` | Per station and variable: `n_pairs`, `modal_gap_min`, p90/p95/p99/p99.5/p99.9 of \|consecutive change\|. |
| `.../station_level.parquet` | Per station: `typical_month_median`, `months`, `n_spd`. |
| `.../station_month_medians.parquet` | Per station-month sustained median and count, from `fit`. |
| `.../neighbours.parquet` | Neighbour table (stid, neighbour, dist_km, dz_m). |
| `.../station_month/YYYY-MM.parquet`, `station_month_all.parquet` | Per station-month: month_median, typical_median, shift, level, noise_days, max_day_jumps, rho, joint_hours, p95_station, p95_neigh, neigh_mag(_days), neigh_decor. |
| `.../jump_days/YYYY-MM.parquet` | Per station-day with at least 1 jump: jumps, pairs, noise flag. |
| `.../hist/YYYY-MM.npz` | Per network, raw and after-QC value histograms, and per-rule histograms (0.5 m/s bins). |
| `.../station_summary.csv` | Per station: n_spd, masked (automatic), manual, per-rule counts, share, few_reports, excluded. |
| `.../station_exclusions.csv` | Excluded stations. |
| `.../summary_by_network.csv` | Per network and variable: % flagged per rule, counts > 25 / > 40 m/s before and after. |
| `.../eval_spike_variants.csv` | Spike-variant evaluation (section 5.2). |
| `.../const_runs_hist_*.npy`, `const_long_runs_*.parquet`, `dchange_hist_*.npy` | CONST and SPIKE calibration tables. |

(`...` = `data/v2/derived/qc/wind`)

---

## 13. How to run

All commands are run from `scripts/` with `~/.venv/analysis_env/bin/python`.

```bash
python wind_qc.py fit                 # once, or deliberately when references should change (~1 min)
python wind_qc.py apply               # flags every month not yet done (~2 min for all 142 months)
python wind_qc.py apply --months 2026-11 2026-12    # new data only, frozen thresholds
python wind_qc.py apply --overwrite   # after changing CONFIG
python wind_qc.py evaluate            # summaries, station exclusion, calibration figures
python wind_qc.py gallery             # images/wind_qc/inspect_gallery.png
python wind_qc.py inspect STID [STID ...] --start 2020-01 --end 2020-07 [--name tag]
python verify_suspects.py [STID ...] [--days 10]    # terrain map and neighbour time series
```

**After changing a threshold:** edit `CONFIG` in `wind_qc.py`, then run
`apply --overwrite` and `evaluate`. `config.json` records what was used.

**Adding a manual exclusion:** append to `CONFIG["manual_exclusions"]` with a reason and a
reference figure, then run `apply --overwrite` and `evaluate`.

**Downstream use:**

```python
import wind_qc
df = wind_qc.load_month_qc(2024, 1)          # masked values -> NaN, excluded stations dropped
df = wind_qc.load_month_qc(2024, 1, drop_excluded=False)   # keep all stations, still masked
# bitmasks available as df.qc_spd / df.qc_gust; wind_qc.RULES maps names to bits
```

As of 2026-11-12 `gradient_pipeline.py` does **not** use the QC yet.

---

## 14. Figures

All in `images/`.

| Figure | Shows |
|---|---|
| `distributions/wind_overview.png`, `wind_by_network.png` | Raw distributions by network (motivation) |
| `suspect_winds/check1_*.png`, `check2_*.png`, `check3_*.png` | First suspect-station analysis: elevation, same-hour regional context, time-series behaviour, neighbour comparison |
| `wind_qc/const_calibration.png` | Run value × duration; stations most affected by CONST |
| `wind_qc/spike_calibration.png` | Station thresholds; % flagged and corroboration for each spike variant |
| `wind_qc/self_rules_calibration.png` | SHIFT, NOISE and LEVEL thresholds |
| `wind_qc/neigh_calibration.png` | NEIGH/DECOR diagnostics (information only) |
| `wind_qc/station_exclusion.png` | Distribution of masked share; 30 most-flagged stations by rule |
| `wind_qc/before_after_spd.png`, `before_after_gust.png` | Per-network distributions before and after QC |
| `wind_qc/inspect_gallery.png` | 10 reference stations with flags by rule |
| `wind_qc/inspect_decor_examples.png`, `inspect_neigh_heavy_flagged.png` | Why neighbour rules don't remove data (made under earlier rule sets) |
| `wind_qc/verify/<STID>.png` | Terrain map and neighbour time series for the excluded stations, KWSIX and OOSC1 |

---

## 15. History

| Date | Change |
|---|---|
| 2026-11-09 | WWG networks excluded from all analyses |
| 2026-11-12 | Bulk distribution check; suspect-station analysis; first QC rules: CONST, SPIKE (prior, p95), neighbour decorrelation |
| 2026-11-12 | SPIKE changed to two-sided with floor and cap; neighbour rule reworked (hourly means, daily magnitude); RANGE added |
| 2026-11-12 | **Self-based masking adopted** (user decision); NOISE, SHIFT and LEVEL added; NEIGH/DECOR made information only; station exclusion at ≥ 5% |
| 2026-11-12 | Visual verification with terrain maps; KWSIX masked from 2025-12-01 (MANUAL); this document |
