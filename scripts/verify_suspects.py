#!/usr/bin/env python3
"""Visual check of QC-suspect stations against their nearest neighbours and terrain.

For each station: the 10-day window with the most masked sustained readings, then
  left   terrain map (elevation contours from AWS Terrarium tiles) with the station and its
         nearest neighbours (any elevation) labelled with elevation and distance
  right  the station's sustained readings (passed / masked by wind_qc) and the hourly mean
         sustained wind of the nearest neighbours that reported in that window

Terrain tiles: https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png
(public; elevation_m = R*256 + G + B/256 - 32768), cached in data/v2/derived/terrain/.

Usage:
  python verify_suspects.py                       # excluded stations + KWSIX, OOSC1
  python verify_suspects.py KWSIX SLIDE --days 14
"""
import argparse
import glob
import io
import math
import os
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

import wind_qc as q

TERRAIN = os.path.join(q.DATA, "derived", "terrain", "terrarium")
OUT = os.path.join(q.IMAGES, "verify")
TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
ZOOM = 10
N_MAP = 6          # neighbours on the map
N_SERIES = 4       # neighbours in the time series (nearest that reported in the window)
DEFAULT = ["KWSIX", "OOSC1"]
NB_COLORS = ["#2a78d6", "#1baf7a", "#6250d6", "#eda100"]
FLAG = "#e34948"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


# ----------------------------------------------------------------------------- terrain

def tile_xy(lon, lat, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def tile_lonlat(x, y, z):
    n = 2 ** z
    lon = x / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def get_tile(z, x, y):
    path = os.path.join(TERRAIN, str(z), str(x), f"{y}.png")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        req = urllib.request.Request(TILE_URL.format(z=z, x=x, y=y), headers={"User-Agent": "socal-wind-qc"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        with open(path, "wb") as f:
            f.write(data)
    img = mpimg.imread(path)[..., :3] * 255.0          # float RGB 0..255
    return img[..., 0] * 256 + img[..., 1] + img[..., 2] / 256 - 32768


def dem(lon0, lon1, lat0, lat1, z=ZOOM):
    """Elevation mosaic and pixel-centre lon/lat vectors covering the box."""
    x0, y0 = tile_xy(lon0, lat1, z)
    x1, y1 = tile_xy(lon1, lat0, z)
    xs, ys = range(int(x0), int(x1) + 1), range(int(y0), int(y1) + 1)
    E = np.vstack([np.hstack([get_tile(z, x, y) for x in xs]) for y in ys])
    px = (np.arange(E.shape[1]) + 0.5) / 256 + xs[0]
    py = (np.arange(E.shape[0]) + 0.5) / 256 + ys[0]
    lons = np.array([tile_lonlat(x, ys[0], z)[0] for x in px])
    lats = np.array([tile_lonlat(xs[0], y, z)[1] for y in py])
    return E, lons, lats


# ----------------------------------------------------------------------------- data

def nearest(st, sid, k):
    r = st.set_index("stid").loc[sid]
    p = np.pi / 180
    a = (np.sin((st.lat - r.lat) * p / 2) ** 2
         + np.cos(r.lat * p) * np.cos(st.lat * p) * np.sin((st.lon - r.lon) * p / 2) ** 2)
    d = st.assign(dist_km=12742 * np.arcsin(np.sqrt(a)), dz_m=st.elevation_m - r.elevation_m)
    return d[d.stid != sid].nsmallest(k, "dist_km")


def flag_window(sid, days):
    """Start/end of the `days`-day window with the most masked sustained readings."""
    fl = pd.concat([pd.read_parquet(f, filters=[("stid", "==", sid)])
                    for f in glob.glob(os.path.join(q.FLAGS, "*", "*.parquet"))])
    fl = fl[(fl.qc_spd.astype(int) & q.mask_bits()) > 0]
    if fl.empty:
        return None
    per_day = fl.time_utc.dt.floor("D").value_counts().sort_index()
    roll = per_day.reindex(pd.date_range(per_day.index.min(), per_day.index.max(), freq="D"), fill_value=0) \
                  .rolling(days, min_periods=1).sum()
    end = roll.idxmax() + pd.Timedelta(days=1)
    return end - pd.Timedelta(days=days), end


def read_obs(stids, t0, t1):
    d = ds.dataset(os.path.join(q.DATA, "obs_raw"), format="parquet")
    f = (pc.field("stid").isin(list(stids)) & (pc.field("time_utc") >= pa_ts(t0)) & (pc.field("time_utc") < pa_ts(t1)))
    obs = d.to_table(columns=["stid", "time_utc", "wind_speed_mps"], filter=f).to_pandas()
    months = sorted({f"{t:%Y/%m}" for t in pd.date_range(t0, t1, freq="D")})
    fl = [pd.read_parquet(os.path.join(q.FLAGS, f"{m}.parquet"), filters=[("stid", "in", list(stids))])
          for m in months if os.path.exists(os.path.join(q.FLAGS, f"{m}.parquet"))]
    if fl:
        obs = obs.merge(pd.concat(fl)[["stid", "time_utc", "qc_spd"]], on=["stid", "time_utc"], how="left")
    else:
        obs["qc_spd"] = 0
    obs["qc_spd"] = obs.qc_spd.fillna(0).astype(int)
    return obs.dropna(subset=["wind_speed_mps"])


def pa_ts(t):
    import pyarrow as pa
    return pa.scalar(pd.Timestamp(t).to_pydatetime(), type=pa.timestamp("ms", tz="UTC"))


# ----------------------------------------------------------------------------- plot

def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def verify(sid, st, days, excluded):
    win = flag_window(sid, days)
    if win is None:
        print(f"{sid}: nothing masked, skipped")
        return
    t0, t1 = win
    cand = nearest(st, sid, 25)
    obs = read_obs([sid] + cand.stid.tolist(), t0, t1)
    reported = set(obs.stid)
    nb_map = cand.head(N_MAP)
    nb_ser = cand[cand.stid.isin(reported)].head(N_SERIES)
    color = dict(zip(nb_ser.stid, NB_COLORS))
    me = st.set_index("stid").loc[sid]

    fig = plt.figure(figsize=(17, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.9])
    # ---- map
    ax = fig.add_subplot(gs[0])
    pts = pd.concat([nb_map, nb_ser]).drop_duplicates("stid")
    lon0, lon1 = min(pts.lon.min(), me.lon), max(pts.lon.max(), me.lon)
    lat0, lat1 = min(pts.lat.min(), me.lat), max(pts.lat.max(), me.lat)
    pad = max(0.08, 0.15 * max(lon1 - lon0, lat1 - lat0))
    lon0, lon1, lat0, lat1 = lon0 - pad, lon1 + pad, lat0 - pad, lat1 + pad
    E, lons, lats = dem(lon0, lon1, lat0, lat1)
    E = np.where(E < -100, np.nan, E)
    ax.imshow(E, extent=[lons[0], lons[-1], lats[-1], lats[0]], cmap="Greys", alpha=0.45, aspect="auto",
              vmin=np.nanmin(E) - 0.3 * (np.nanmax(E) - np.nanmin(E)), vmax=np.nanmax(E))
    step = 250 if (np.nanmax(E) - np.nanmin(E)) > 1500 else 100
    lv = np.arange(np.floor(np.nanmin(E) / step) * step, np.nanmax(E) + step, step)
    cs = ax.contour(lons, lats, E, levels=lv, colors=MUTED, linewidths=0.4)
    major = [l for l in lv if l % (step * 4) == 0]
    ax.clabel(cs, levels=major, fontsize=6.5, fmt="%d m")
    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_aspect(1 / np.cos(np.radians(me.lat)))
    for _, r in nb_map.iterrows():
        c = color.get(r.stid, MUTED)
        ax.plot(r.lon, r.lat, "o", ms=7, mfc=c, mec="white", mew=1.2, zorder=4)
        ax.annotate(f"{r.stid}  {r.elevation_m:.0f} m, {r.dist_km:.0f} km" + ("" if r.stid in reported else " (no data)"),
                    (r.lon, r.lat), fontsize=7.5, color=INK, xytext=(5, 3), textcoords="offset points", zorder=5)
    ax.plot(me.lon, me.lat, "*", ms=16, mfc=FLAG, mec="white", mew=1.2, zorder=6)
    ax.annotate(f"{sid}  {me.elevation_m:.0f} m", (me.lon, me.lat), fontsize=9, fontweight="bold", color=INK,
                xytext=(7, -12), textcoords="offset points", zorder=6)
    ax.set_title(f"{sid} ({me.network_shortname}) — {me['name']}\nnearest stations, contours every {step} m",
                 loc="left", fontsize=10, color=INK)
    ax.tick_params(colors=MUTED, labelsize=7)

    # ---- time series
    ax = fig.add_subplot(gs[1])
    style(ax)
    ax.grid(True, color=GRID, lw=0.6)
    for _, r in nb_ser.iterrows():
        o = obs[obs.stid == r.stid]
        h = o.set_index("time_utc").wind_speed_mps.resample("1h").mean()
        ax.plot(h.index, h.values, color=color[r.stid], lw=1.3,
                label=f"{r.stid} ({r.network_shortname}) {r.elevation_m:.0f} m, {r.dist_km:.0f} km, dz {r.dz_m:+.0f} m")
    d = obs[obs.stid == sid]
    m = (d.qc_spd & q.mask_bits()) > 0
    ax.plot(d.time_utc[~m], d.wind_speed_mps[~m], ".", ms=2.5, color=INK, label=f"{sid} passed ({(~m).sum():,})", zorder=3)
    rules = sorted({r for r, b in q.RULES.items() if r in q.CONFIG["mask_rules"] and (d.qc_spd[m] & b).any()})
    ax.plot(d.time_utc[m], d.wind_speed_mps[m], "x", ms=3.5, mew=0.8, color=FLAG,
            label=f"{sid} masked ({m.sum():,}: {', '.join(rules)})", zorder=3)
    ax.set_ylabel("sustained wind (m/s); neighbours = hourly mean", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", markerscale=2)
    tag = "STATION EXCLUDED" if sid in excluded else "station kept"
    ax.set_title(f"{t0:%Y-%m-%d} to {t1:%Y-%m-%d} (window with the most masked readings) — {tag}",
                 loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{sid}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stids", nargs="*")
    ap.add_argument("--days", type=int, default=10)
    a = ap.parse_args()
    st = q.stations().dropna(subset=["lat", "lon", "elevation_m"])
    excluded = q.excluded_stations()
    stids = a.stids or sorted(excluded) + DEFAULT
    for sid in stids:
        verify(sid, st, a.days, excluded)


if __name__ == "__main__":
    main()
