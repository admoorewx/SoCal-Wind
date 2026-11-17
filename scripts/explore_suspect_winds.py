#!/usr/bin/env python3
"""Are the very high wind readings real (terrain) or bad data?

Steps (run in order, or `all`):
  scan     One pass over obs_raw: per-station sustained-wind histograms (0.1 m/s) and every
           reading with sustained > HIGH or gust > GUST_HIGH.
           -> derived/qc/suspect/station_wind_stats.parquet, high_readings.parquet
  extract  Defines suspects (>= MIN_HIGH sustained readings > HIGH) and their neighbours, then a
           second pass collecting
             - raw readings of every suspect station                    (check 2)
             - hourly max sustained/gust for suspects + neighbours      (check 3)
             - hourly max sustained for ALL stations at sampled hours   (check 1)
  checks   Figures -> images/suspect_winds/, tables -> derived/qc/suspect/

  Check 1  Same-time context. For a suspect reading at hour t, what do other (non-suspect)
           stations in the region (<= REGION_KM) at similar elevation report at t?
           Also: station p50/p99 vs elevation coloured by network.
  Check 2  Behaviour of each suspect's high readings: repeated exact value (stuck / fill value),
           isolated spikes (noise), or months-long elevated regime.
  Check 3  Nearest neighbours (<= NEIGH_KM, |dz| <= NEIGH_DZ): hour-by-hour comparison and
           Spearman agreement, in the suspect's normal vs high periods.

Hourly values are the max within the clock hour, which favours stations that report often;
that makes the neighbour comparison generous to the suspect.
"""
import argparse
import glob
import os
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data", "v2"))
OUT = os.path.join(DATA, "derived", "qc", "suspect")
IMAGES = os.path.normpath(os.path.join(HERE, "..", "images", "suspect_winds"))
EXCLUDE = ("WWG",)

HIGH, GUST_HIGH = 25.0, 40.0     # m/s
MIN_HIGH = 10                    # sustained readings > HIGH to be a suspect
BIN, NBIN = 0.1, 1600
REGION_KM, REGION_DZ = 300, 500  # check 1
NEIGH_KM, NEIGH_DZ, NEIGH_MAX = 75, 400, 5  # check 3
EVENT_HOURS_PER_STATION = 150
BASELINE_HOURS = 1500
SEED = 0

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
OTHER, INK, MUTED, GRID, BAD = "#8a8a85", "#1f1f1e", "#6b6a64", "#e6e5e0", "#e34948"

_ctx = {}


def _init(ctx):
    _ctx.update(ctx)


def stations():
    s = pd.read_parquet(os.path.join(DATA, "stations.parquet")).drop_duplicates("stid")
    s["network_shortname"] = s.network_shortname.fillna("UNKNOWN")
    return s[~s.network_shortname.str.startswith(EXCLUDE)].reset_index(drop=True)


def files():
    return sorted(glob.glob(os.path.join(DATA, "obs_raw", "*", "*", "*.parquet")))


def read(path):
    t = pq.read_table(path, columns=["stid", "time_utc", "wind_speed_mps", "wind_gust_mps"]).to_pandas()
    t["code"] = t.stid.map(_ctx["code"])
    return t[t.code.notna()]


# ----------------------------------------------------------------------------- scan

def scan_file(path):
    t = read(path)
    x = t.wind_speed_mps.to_numpy(np.float64)
    ok = np.isfinite(x) & (x >= 0)
    key = t.code.to_numpy()[ok].astype(np.int64) * NBIN + np.minimum((x[ok] / BIN + 1e-6).astype(np.int64), NBIN - 1)
    k, c = np.unique(key, return_counts=True)
    hi = t[(t.wind_speed_mps > HIGH) | (t.wind_gust_mps > GUST_HIGH)].drop(columns="code")
    return k, c, hi


def step_scan(workers):
    st = stations()
    code = dict(zip(st.stid, st.index))
    H = np.zeros(len(st) * NBIN, np.int64)
    hi = []
    with Pool(workers, initializer=_init, initargs=({"code": code},)) as p:
        for i, (k, c, h) in enumerate(p.imap_unordered(scan_file, files(), chunksize=2)):
            np.add.at(H, k, c)
            hi.append(h)
            if (i + 1) % 100 == 0:
                print(f"  scan {i + 1}")
    H = H.reshape(len(st), NBIN)
    centers = (np.arange(NBIN) + 0.5) * BIN
    n = H.sum(1)
    c = np.cumsum(H, 1) / np.maximum(n, 1)[:, None]
    q = {f"p{p:g}": centers[np.minimum((c < p / 100).sum(1), NBIN - 1)] for p in (50, 99, 99.9)}
    stats = st[["stid", "name", "state", "network_shortname", "lat", "lon", "elevation_m"]].assign(
        n=n, n_gt25=H[:, int(25 / BIN):].sum(1), n_gt40=H[:, int(40 / BIN):].sum(1), **q)
    stats.loc[stats.n == 0, list(q)] = np.nan
    os.makedirs(OUT, exist_ok=True)
    stats.to_parquet(os.path.join(OUT, "station_wind_stats.parquet"))
    pd.concat(hi).to_parquet(os.path.join(OUT, "high_readings.parquet"))
    print(f"{len(stats)} stations, {int((stats.n_gt25 >= MIN_HIGH).sum())} with >= {MIN_HIGH} readings > {HIGH} m/s")


# ----------------------------------------------------------------------------- extract

def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def suspects_and_neighbours():
    s = pd.read_parquet(os.path.join(OUT, "station_wind_stats.parquet"))
    sus = s[s.n_gt25 >= MIN_HIGH].copy()
    clean = s[(s.n_gt25 < MIN_HIGH) & (s.n > 0) & s.elevation_m.notna()]
    rows = []
    for _, r in sus.iterrows():
        d = haversine_km(r.lat, r.lon, clean.lat.to_numpy(), clean.lon.to_numpy())
        dz = (clean.elevation_m - r.elevation_m).to_numpy() if pd.notna(r.elevation_m) else np.zeros(len(clean))
        m = (d <= NEIGH_KM) & (np.abs(dz) <= NEIGH_DZ)
        for j in np.argsort(np.where(m, d, np.inf))[:min(NEIGH_MAX, m.sum())]:
            rows.append({"suspect": r.stid, "neighbour": clean.stid.iloc[j], "dist_km": round(d[j], 1),
                         "dz_m": round(dz[j]), "neighbour_network": clean.network_shortname.iloc[j]})
    return s, sus, pd.DataFrame(rows)


def extract_file(path):
    t = read(path)
    t["hour"] = t.time_utc.dt.floor("h")
    raw = t[t.stid.isin(_ctx["suspects"])][["stid", "time_utc", "wind_speed_mps", "wind_gust_mps"]]
    a = t[t.stid.isin(_ctx["series"])]
    series = a.groupby(["stid", "hour"], observed=True)[["wind_speed_mps", "wind_gust_mps"]].max().reset_index()
    b = t[t.hour.isin(_ctx["hours"])]
    snap = b.groupby(["stid", "hour"], observed=True).wind_speed_mps.max().dropna().reset_index()
    return raw, series, snap


def step_extract(workers):
    st = stations()
    s, sus, nb = suspects_and_neighbours()
    nb.to_csv(os.path.join(OUT, "neighbours.csv"), index=False)
    print(f"{len(sus)} suspects; {nb.suspect.nunique()} have neighbours ({len(nb)} pairs)")
    hi = pd.read_parquet(os.path.join(OUT, "high_readings.parquet"))
    hi = hi[hi.stid.isin(sus.stid) & (hi.wind_speed_mps > HIGH)]
    hi["hour"] = hi.time_utc.dt.floor("h")
    rng = np.random.default_rng(SEED)
    ev = (hi.drop_duplicates(["stid", "hour"]).sample(frac=1, random_state=SEED)
          .groupby("stid").head(EVENT_HOURS_PER_STATION))
    all_hours = pd.date_range("2015-01-01", "2026-10-31 23:00", freq="h", tz="UTC")
    base = pd.DatetimeIndex(rng.choice(all_hours, BASELINE_HOURS, replace=False))
    hours = set(ev.hour) | set(base)
    ev[["stid", "hour", "wind_speed_mps"]].to_parquet(os.path.join(OUT, "event_hours.parquet"))
    pd.DataFrame({"hour": base}).to_parquet(os.path.join(OUT, "baseline_hours.parquet"))
    ctx = {"code": dict(zip(st.stid, st.index)), "suspects": set(sus.stid),
           "series": set(sus.stid) | set(nb.neighbour), "hours": hours}
    R, S, P = [], [], []
    with Pool(workers, initializer=_init, initargs=(ctx,)) as p:
        for i, (r, se, sn) in enumerate(p.imap_unordered(extract_file, files(), chunksize=2)):
            R.append(r), S.append(se), P.append(sn)
            if (i + 1) % 100 == 0:
                print(f"  extract {i + 1}")
    pd.concat(R).sort_values(["stid", "time_utc"]).to_parquet(os.path.join(OUT, "suspect_raw.parquet"))
    pd.concat(S).to_parquet(os.path.join(OUT, "hourly_series.parquet"))
    pd.concat(P).to_parquet(os.path.join(OUT, "snapshots.parquet"))


# ----------------------------------------------------------------------------- checks

def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def check1(s, sus):
    """Station climatology vs elevation, and same-hour regional context."""
    s = s[(s.n > 1000) & s.elevation_m.notna()]
    top = s.groupby("network_shortname").n.sum().sort_values(ascending=False).index[:7].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    for ax, col, title in zip(axes, ["p50", "p99"], ["Median", "99th percentile"]):
        style(ax)
        o = s[~s.network_shortname.isin(top)]
        ax.scatter(o.elevation_m, o[col], s=9, color=OTHER, alpha=.5, lw=0, label="Other networks")
        for c, net in zip(SERIES, top):
            d = s[s.network_shortname == net]
            ax.scatter(d.elevation_m, d[col], s=9, color=c, alpha=.7, lw=0, label=net)
        d = s[s.stid.isin(sus.stid)]
        ax.scatter(d.elevation_m, d[col], s=40, facecolor="none", edgecolor=INK, lw=1.1, label="Suspect station")
        ax.set_yscale("log")
        ax.set_title(f"{title} sustained wind per station", loc="left", color=INK, fontsize=11)
        ax.set_xlabel("Station elevation (m)", color=MUTED, fontsize=9)
        ax.set_ylabel("m/s (log)", color=MUTED, fontsize=9)
    for _, r in s[s.stid.isin(sus.stid)].nlargest(12, "p99").iterrows():
        axes[1].annotate(r.stid, (r.elevation_m, r.p99), fontsize=7, color=INK, xytext=(4, 2), textcoords="offset points")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Check 1a. Station wind climatology vs elevation (stations with > 1000 readings)",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES, "check1_climatology_vs_elevation.png"), dpi=120)
    plt.close(fig)

    # same-hour regional context
    snap = pd.read_parquet(os.path.join(OUT, "snapshots.parquet"))
    ev = pd.read_parquet(os.path.join(OUT, "event_hours.parquet"))
    base = pd.read_parquet(os.path.join(OUT, "baseline_hours.parquet"))
    meta = s.set_index("stid")[["lat", "lon", "elevation_m", "network_shortname"]]
    snap = snap.join(meta, on="stid", how="inner")
    snap = snap[~snap.stid.isin(sus.stid)]
    by_hour = dict(tuple(snap.groupby("hour")))
    rows = []
    for _, e in ev.iterrows():
        if e.stid not in meta.index:
            continue
        m = meta.loc[e.stid]
        for kind, hours in (("event", [e.hour]), ("baseline", base.hour.sample(3, random_state=hash(e.stid) % 2**31))):
            for h in hours:
                d = by_hour.get(h)
                if d is None:
                    continue
                dist = haversine_km(m.lat, m.lon, d.lat.to_numpy(), d.lon.to_numpy())
                k = (dist <= REGION_KM) & (np.abs(d.elevation_m.to_numpy() - m.elevation_m) <= REGION_DZ)
                if k.sum() < 3:
                    continue
                w = d.wind_speed_mps.to_numpy()[k]
                rows.append({"stid": e.stid, "kind": kind, "hour": h, "suspect_mps": e.wind_speed_mps,
                             "region_n": int(k.sum()), "region_p90": np.percentile(w, 90), "region_max": w.max()})
    ctx = pd.DataFrame(rows)
    ctx.to_csv(os.path.join(OUT, "check1_event_context.csv"), index=False)
    summ = (ctx.groupby(["stid", "kind"]).agg(hours=("hour", "size"), suspect_med=("suspect_mps", "median"),
                                             region_p90_med=("region_p90", "median"),
                                             region_max_med=("region_max", "median"))
            .unstack("kind"))
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.drop(columns=["suspect_med_baseline"], errors="ignore")

    evc = ctx[ctx.kind == "event"]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    style(ax)
    ax.scatter(evc.region_max, evc.suspect_mps, s=7, color=SERIES[0], alpha=.35, lw=0)
    lim = [0, max(110, evc.suspect_mps.max() + 2)]
    ax.plot(lim, lim, color=MUTED, lw=1, ls="--")
    ax.text(lim[1] * .55, lim[1] * .58, "suspect = regional max", rotation=40, color=MUTED, fontsize=8)
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel(f"Max hourly wind at other stations within {REGION_KM} km and ±{REGION_DZ} m, same hour (m/s)",
                  color=MUTED, fontsize=9)
    ax.set_ylabel("Suspect station reading (m/s)", color=MUTED, fontsize=9)
    ax.set_title(f"Check 1b. Suspect readings > {HIGH:g} m/s vs the same hour in the region",
                 loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES, "check1_same_hour_context.png"), dpi=120)
    plt.close(fig)
    return summ


def runs(v, thresh):
    """Longest run of identical consecutive readings (in time order) whose value is > thresh."""
    if len(v) == 0:
        return 0
    idx = np.flatnonzero(np.r_[True, v[1:] != v[:-1]])
    length = np.diff(np.r_[idx, len(v)])
    keep = v[idx] > thresh
    return int(length[keep].max()) if keep.any() else 0


def check2(sus):
    raw = pd.read_parquet(os.path.join(OUT, "suspect_raw.parquet"))
    rows = []
    for stid, d in raw.groupby("stid"):
        d = d.dropna(subset=["wind_speed_mps"]).sort_values("time_utc")
        v = d.wind_speed_mps.to_numpy(np.float64)
        t = d.time_utc.to_numpy()
        hi = v > HIGH
        if hi.sum() == 0:
            continue
        hv = np.round(v[hi], 2)
        vals, cnt = np.unique(hv, return_counts=True)
        gap_p = np.r_[np.inf, np.diff(t).astype("timedelta64[m]").astype(float)]
        gap_n = np.r_[gap_p[1:], np.inf]
        prev = np.r_[np.nan, v[:-1]]
        nxt = np.r_[v[1:], np.nan]
        iso = hi & (gap_p <= 120) & (gap_n <= 120) & (prev < v / 2) & (nxt < v / 2)
        has_nb = hi & (gap_p <= 120) & (gap_n <= 120)
        mon = d.assign(m=d.time_utc.dt.strftime("%Y-%m")).groupby("m").wind_speed_mps.median()
        hi_months = mon.index[mon > 15].tolist()
        step = np.abs(np.diff(v))[(gap_p[1:] <= 120)]
        rows.append({
            "stid": stid, "n": len(v), "n_gt25": int(hi.sum()),
            "top_value": vals[cnt.argmax()], "top_value_share": round(cnt.max() / hi.sum(), 3),
            "distinct_gt25": len(vals),
            "longest_identical_run_gt25": runs(np.round(v, 2), HIGH),
            "n_ge95": int((v >= 95).sum()),
            "isolated_spike_share": round(iso.sum() / max(has_nb.sum(), 1), 3),
            "median_abs_step_mps": round(float(np.median(step)), 2) if len(step) else np.nan,
            "months_median_gt15": len(hi_months),
            "high_months": f"{hi_months[0]}..{hi_months[-1]}" if hi_months else "",
            "first_gt25": d.time_utc[hi].min().date(), "last_gt25": d.time_utc[hi].max().date(),
        })
    tab = pd.DataFrame(rows)

    def label(r):
        if r.longest_identical_run_gt25 >= 6:
            return "stuck (identical run)"
        if r.top_value >= 90 and r.top_value_share >= 0.2:
            return "fill value"
        if r.months_median_gt15 >= 2:
            return "elevated regime (months)"
        if r.isolated_spike_share >= 0.5:
            return "isolated spikes (noise)"
        return "episodic (weather-like)"
    tab["behaviour"] = tab.apply(label, axis=1)

    # time series figure, top suspects by count
    order = sus.sort_values("n_gt25", ascending=False).stid.tolist()
    show = [x for x in order if x in set(tab.stid)][:30]
    ncol = 5
    nrow = int(np.ceil(len(show) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.2 * nrow))
    meta = sus.set_index("stid")
    for ax, stid in zip(axes.flat, show):
        style(ax)
        d = raw[raw.stid == stid]
        h = d.set_index("time_utc").wind_speed_mps.resample("6h").max()
        ax.plot(h.index, h.values, color=SERIES[0], lw=0.4)
        k = d.wind_speed_mps > HIGH
        ax.scatter(d.time_utc[k], d.wind_speed_mps[k], s=2, color=BAD, lw=0, zorder=3)
        ax.axhline(HIGH, color=MUTED, lw=0.6, ls="--")
        b = tab.loc[tab.stid == stid, "behaviour"].iloc[0]
        ax.set_title(f"{stid} ({meta.loc[stid, 'network_shortname']}, {meta.loc[stid, 'elevation_m']:.0f} m)\n{b}",
                     fontsize=7.5, color=INK, loc="left")
        ax.tick_params(axis="x", labelsize=6.5)
    for ax in axes.flat[len(show):]:
        ax.set_visible(False)
    fig.suptitle(f"Check 2. Suspect stations: 6-hourly max sustained wind (m/s); red = readings > {HIGH:g} m/s",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES, "check2_suspect_timeseries.png"), dpi=110)
    plt.close(fig)
    return tab


def spearman(x, y):
    if len(x) < 30:
        return np.nan
    return pd.Series(x).rank().corr(pd.Series(y).rank())


def check3(sus):
    nb = pd.read_csv(os.path.join(OUT, "neighbours.csv"))
    ser = pd.read_parquet(os.path.join(OUT, "hourly_series.parquet"))
    w = ser.pivot_table(index="hour", columns="stid", values="wind_speed_mps", aggfunc="max")
    rows, plots = [], []
    for stid, g in nb.groupby("suspect"):
        if stid not in w:
            continue
        cols = [c for c in g.neighbour if c in w]
        if not cols:
            continue
        a = w[stid]
        nmax = w[cols].max(1)
        nmed = w[cols].median(1)
        d = pd.DataFrame({"s": a, "nmax": nmax, "nmed": nmed}).dropna()
        if len(d) < 100:
            continue
        hi = d.s > HIGH
        # "high months" = months where the suspect's monthly median is > 15 m/s
        mm = d.s.groupby(d.index.strftime("%Y-%m")).median()
        bad_m = set(mm.index[mm > 15])
        in_bad = d.index.strftime("%Y-%m").isin(bad_m)
        rows.append({
            "stid": stid, "neighbours": ",".join(cols), "n_hours": len(d), "hours_gt25": int(hi.sum()),
            "neigh_max_at_gt25_median": round(d.nmax[hi].median(), 1) if hi.any() else np.nan,
            "share_gt25_with_neigh_gt15": round((d.nmax[hi] > 15).mean(), 3) if hi.any() else np.nan,
            "ratio_suspect_to_neighmed_normal": round((d.s[~in_bad & ~hi] / d.nmed[~in_bad & ~hi].clip(lower=0.5)).median(), 2),
            "rho_normal": round(spearman(d.s[~in_bad].to_numpy(), d.nmed[~in_bad].to_numpy()), 2),
            "rho_high_months": round(spearman(d.s[in_bad].to_numpy(), d.nmed[in_bad].to_numpy()), 2),
            "hours_in_high_months": int(in_bad.sum()),
        })
        plots.append((stid, d, cols))
    tab = pd.DataFrame(rows)

    plots.sort(key=lambda p: -(p[1].s > HIGH).sum())
    plots = plots[:25]
    ncol = 5
    nrow = int(np.ceil(len(plots) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.2 * nrow))
    meta = sus.set_index("stid")
    for ax, (stid, d, cols) in zip(axes.flat, plots):
        style(ax)
        hi = d.s > HIGH
        ax.scatter(d.nmax[~hi], d.s[~hi], s=2, color=SERIES[0], alpha=.15, lw=0)
        ax.scatter(d.nmax[hi], d.s[hi], s=4, color=BAD, alpha=.6, lw=0)
        lim = [0, max(d.s.max(), d.nmax.max()) * 1.05]
        ax.plot(lim, lim, color=MUTED, lw=.8, ls="--")
        ax.set_xlim(lim), ax.set_ylim(lim)
        r = tab.set_index("stid").loc[stid]
        ax.set_title(f"{stid} ({meta.loc[stid, 'network_shortname']}) vs {len(cols)} nbr\n"
                     f"rho normal {r.rho_normal:.2f}, high-months {r.rho_high_months if pd.notna(r.rho_high_months) else '–'}",
                     fontsize=7.5, color=INK, loc="left")
    for ax in axes.flat[len(plots):]:
        ax.set_visible(False)
    fig.supxlabel(f"Neighbour max, same hour (m/s)  [neighbours ≤ {NEIGH_KM} km, ±{NEIGH_DZ} m]", color=MUTED, fontsize=9)
    fig.supylabel("Suspect hourly max (m/s)", color=MUTED, fontsize=9)
    fig.suptitle(f"Check 3. Suspect vs nearby similar-elevation stations; red = suspect > {HIGH:g} m/s",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES, "check3_neighbour_comparison.png"), dpi=110)
    plt.close(fig)
    return tab


def step_checks():
    os.makedirs(IMAGES, exist_ok=True)
    s, sus, _ = suspects_and_neighbours()
    c1 = check1(s, sus)
    c2 = check2(sus)
    c3 = check3(sus)
    base = sus[["stid", "network_shortname", "state", "elevation_m", "n", "n_gt25", "n_gt40", "p50", "p99"]]
    out = base.merge(c2.drop(columns=["n", "n_gt25"]), on="stid", how="left") \
              .merge(c1.reset_index(), on="stid", how="left").merge(c3, on="stid", how="left") \
              .sort_values("n_gt25", ascending=False)
    out.to_csv(os.path.join(OUT, "suspect_summary.csv"), index=False)
    pd.set_option("display.width", 300), pd.set_option("display.max_columns", 50)
    print(out.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["scan", "extract", "checks", "all"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.step in ("scan", "all"):
        step_scan(a.workers)
    if a.step in ("extract", "all"):
        step_extract(a.workers)
    if a.step in ("checks", "all"):
        step_checks()


if __name__ == "__main__":
    main()
