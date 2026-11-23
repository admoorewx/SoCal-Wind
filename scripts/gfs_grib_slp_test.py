#!/usr/bin/env python3
"""Q1: is the GFS SLP bias specific to Open-Meteo, or does raw GFS GRIB (AWS) avoid it?

download  byte-range fetch of PRMSL, MSLET, PRES:surface, HGT:surface, TMP:2 m from the 0.25 deg f000 files
          (noaa-gfs-bdp-pds) for every 00/06/12/18Z cycle of Jan and Jul 2025, one small GRIB per cycle in
          /mnt/d/NWP/aws_gfs/grib/ (resumable; .part then rename).
extract   nearest 0.25 deg grid point at the 22 gradient stations -> derived/verification/gfs/aws_grib_test/points.parquet
analyze   (a) which GRIB field Open-Meteo pressure_msl is; (b) PRMSL / MSLET / Open-Meteo vs METAR SLP;
          (c) GRIB surface pressure adjusted to station height vs observed station pressure;
          (d) implied day-0 gradient bias per pair. Tables + figures. 2025 only.

Usage: python gfs_grib_slp_test.py download|extract|analyze|all [--workers 3]
"""
import argparse, gzip, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = "/mnt/d/socal"
DER = f"{ROOT}/data/v2/derived"
BASE = "/mnt/d/NWP/aws_gfs"
GRIB = f"{BASE}/grib"
LOG = f"{BASE}/logs/download.log"
OUT = f"{DER}/verification/gfs/aws_grib_test"
IMG = f"{ROOT}/images/verification/gfs/aws_grib_test"
URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{d}/{h}/atmos/gfs.t{h}z.pgrb2.0p25.f000"
FIELDS = [":PRMSL:mean sea level:", ":MSLET:mean sea level:", ":PRES:surface:", ":HGT:surface:",
          ":TMP:2 m above ground:"]
MONTHS = ["2025-01", "2025-07"]
G, RD, GAMMA = 9.80665, 287.05, 0.0065
HIGH = ["KDAG", "KLAS", "KNID", "KWJF", "KPMD", "KLMT", "KWMC", "KRNO", "KTPH"]
KEY_PAIRS = ["LAX-TPH", "SFO-WMC", "SAC-RNO", "MFR-LMT"]


def log(msg):
    line = f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def cycles():
    out = []
    for m in MONTHS:
        for d in pd.date_range(m + "-01", periods=pd.Period(m).days_in_month, freq="D"):
            for h in ("00", "06", "12", "18"):
                out.append((d.strftime("%Y%m%d"), h))
    return out


def get(url, rng=None, tries=5):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def fetch_cycle(dh):
    d, h = dh
    fn = f"{GRIB}/gfs_{d}{h}_f000.grib2"
    if os.path.exists(fn):
        return 0
    url = URL.format(d=d, h=h)
    lines = get(url + ".idx").decode().strip().split("\n")
    offs = [int(l.split(":")[1]) for l in lines]
    parts = []
    for pat in FIELDS:
        hit = [i for i, l in enumerate(lines) if pat in l]
        if len(hit) != 1:
            raise RuntimeError(f"{d}{h}: {pat} matched {len(hit)} lines")
        i = hit[0]
        end = offs[i + 1] - 1 if i + 1 < len(offs) else ""
        parts.append(get(url, (offs[i], end)))
    body = b"".join(parts)
    with open(fn + ".part", "wb") as f:
        f.write(body)
    os.replace(fn + ".part", fn)
    return len(body)


def download(workers):
    os.makedirs(GRIB, exist_ok=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    todo = [c for c in cycles() if not os.path.exists(f"{GRIB}/gfs_{c[0]}{c[1]}_f000.grib2")]
    log(f"{len(cycles())} cycles, {len(todo)} to download, {workers} workers")
    t0, total, done = time.time(), 0, 0
    with ThreadPoolExecutor(workers) as ex:
        for c, n in zip(todo, ex.map(fetch_cycle, todo)):
            total += n
            done += 1
            if done in (3, 10) or done % 20 == 0 or done == len(todo):
                el = time.time() - t0
                log(f"{done}/{len(todo)} {c[0]}{c[1]}  {total / 1e6:.0f} MB in {el / 60:.1f} min; "
                    f"ETA {el / done * (len(todo) - done) / 60:.1f} min")
    log(f"download done: {total / 1e6:.0f} MB in {(time.time() - t0) / 60:.1f} min")


def stations():
    import gradient_pipeline as gp
    pairs = gp.load_pairs(gp.DEFAULT_GRADIENTS)
    st = sorted({s for p in pairs.values() for s in p})
    meta = pd.read_parquet(f"{ROOT}/data/v2/stations.parquet").drop_duplicates("stid").set_index("stid").loc[st]
    return pairs, meta


def extract():
    import eccodes
    _, meta = stations()
    lat_i = np.round((90 - meta.lat.to_numpy()) / 0.25).astype(int)
    lon_i = np.round((meta.lon.to_numpy() % 360) / 0.25).astype(int) % 1440
    names = {"prmsl": "prmsl_hpa", "mslet": "mslet_hpa", "sp": "psfc_hpa", "orog": "hgt_m", "2t": "t2m_k"}
    rows = []
    for d, h in cycles():
        fn = f"{GRIB}/gfs_{d}{h}_f000.grib2"
        rec = {"valid_utc": pd.to_datetime(f"{d}{h}", format="%Y%m%d%H").tz_localize("UTC")}
        vals = {}
        with open(fn, "rb") as f:
            while True:
                gid = eccodes.codes_grib_new_from_file(f)
                if gid is None:
                    break
                sn = eccodes.codes_get(gid, "shortName")
                lvl = eccodes.codes_get(gid, "typeOfLevel")
                if eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees") != 90 or eccodes.codes_get(gid, "Ni") != 1440:
                    raise RuntimeError(f"unexpected grid in {fn}")
                key = {"prmsl": "prmsl", "mslet": "mslet", "sp": "sp", "orog": "orog", "2t": "2t"}.get(sn)
                if sn == "gh" and lvl == "surface":
                    key = "orog"
                if key is None:
                    raise RuntimeError(f"unexpected field {sn}/{lvl} in {fn}")
                a = eccodes.codes_get_values(gid).reshape(721, 1440)
                vals[key] = a[lat_i, lon_i]
                eccodes.codes_release(gid)
        for k in names:
            if k not in vals:
                raise RuntimeError(f"{fn}: missing {k}, got {list(vals)}")
        for j, s in enumerate(meta.index):
            rows.append(dict(valid_utc=rec["valid_utc"], stid=s, prmsl_hpa=vals["prmsl"][j] / 100,
                             mslet_hpa=vals["mslet"][j] / 100, psfc_hpa=vals["sp"][j] / 100, hgt_m=vals["orog"][j],
                             t2m_k=vals["2t"][j], grid_lat=90 - lat_i[j] * 0.25,
                             grid_lon=((lon_i[j] * 0.25 + 180) % 360) - 180))
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_parquet(f"{OUT}/points.parquet", index=False)
    print(f"wrote {OUT}/points.parquet: {len(df):,} rows; KLAX cell:",
          df[df.stid == "KLAX"][["grid_lat", "grid_lon"]].iloc[0].tolist())


def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pd.set_option("display.width", 220)
    pairs, meta = stations()
    A = pd.read_parquet(f"{OUT}/points.parquet")
    A["valid_utc"] = A.valid_utc.astype("datetime64[us, UTC]")

    # Open-Meteo cached day-0 (2025)
    om = []
    for s in meta.index:
        d = json.load(gzip.open(f"/mnt/d/NWP/open_meteo/raw/gfs_calib/{s}_2025_day0.json.gz", "rt"))
        h = pd.DataFrame(d["hourly"])
        h["valid_utc"] = pd.to_datetime(h.time).dt.tz_localize("UTC").astype("datetime64[us, UTC]")
        h["stid"] = s
        om.append(h[["valid_utc", "stid", "pressure_msl", "surface_pressure", "temperature_2m"]])
    A = A.merge(pd.concat(om).rename(columns={"pressure_msl": "om_msl_hpa"}), on=["valid_utc", "stid"], how="left")
    # reduction of GRIB PRES:surface with current T2m (Open-Meteo-style formula, from model terrain height)
    A["psfc_reduced_hpa"] = A.psfc_hpa * np.exp(G * A.hgt_m / (RD * (A.t2m_k + GAMMA * A.hgt_m / 2)))

    # (a) identity
    ida = []
    for c in ("prmsl_hpa", "mslet_hpa", "psfc_reduced_hpa"):
        d = (A.om_msl_hpa - A[c]).abs()
        ida.append(dict(field=c, median_abs=d.median(), p99_abs=d.quantile(.99), max_abs=d.max(),
                        frac_le_0p05=(d <= 0.05).mean(), n=int(d.notna().sum())))
    ida = pd.DataFrame(ida)
    ida.round(3).to_csv(f"{OUT}/identity.csv", index=False)
    print("== (a) Open-Meteo pressure_msl vs GRIB fields (|diff| hPa) ==")
    print(ida.round(3).to_string(index=False))

    # METAR SLP (valid = obs time ceil to hour)
    g = pd.read_parquet(f"{DER}/gradients/gradient_timeseries.parquet")
    g = g[((g.time_utc >= "2024-12-31 23:00") & (g.time_utc < "2025-02-01")) |
          ((g.time_utc >= "2025-06-30 23:00") & (g.time_utc < "2025-08-01"))].copy()
    g["valid_utc"] = g.time_utc.dt.ceil("h").astype("datetime64[us, UTC]")
    obs = pd.concat([g[["station1", "valid_utc", "slp1_hpa"]].set_axis(["stid", "valid_utc", "slp_obs"], axis=1),
                     g[["station2", "valid_utc", "slp2_hpa"]].set_axis(["stid", "valid_utc", "slp_obs"], axis=1)])
    obs = obs.dropna().groupby(["stid", "valid_utc"]).slp_obs.first().reset_index()
    M = A.merge(obs, on=["stid", "valid_utc"])
    M["month"] = M.valid_utc.dt.month.map({1: "Jan", 7: "Jul"})
    M["hour"] = M.valid_utc.dt.hour
    SRC = {"prmsl_hpa": "PRMSL", "mslet_hpa": "MSLET", "om_msl_hpa": "Open-Meteo"}
    rows = []
    for (s, mo), x in M.groupby(["stid", "month"]):
        for c, lab in SRC.items():
            e = x[c] - x.slp_obs
            r = dict(stid=s, elev_m=meta.elevation_m[s], month=mo, source=lab, n=len(e), bias=e.mean(), sd=e.std())
            for hh, eh in e.groupby(x.hour):
                r[f"bias_{hh:02d}z"] = eh.mean()
            rows.append(r)
    sb = pd.DataFrame(rows)
    sb.round(3).to_csv(f"{OUT}/slp_bias_vs_metar.csv", index=False)
    print("\n== (b) SLP bias vs METAR (hPa), high stations: bias [Jan | Jul] ==")
    t = sb[sb.stid.isin(HIGH)].pivot_table(index=["stid", "elev_m"], columns=["source", "month"], values="bias")
    print(t.round(2).sort_index(level="elev_m").to_string())
    print("\nSD of error, high stations:")
    print(sb[sb.stid.isin(HIGH)].pivot_table(index="stid", columns=["source", "month"], values="sd").round(2).to_string())
    print("\nhour-of-day bias, July, high stations (00/06/12/18Z):")
    hcols = ["bias_00z", "bias_06z", "bias_12z", "bias_18z"]
    print(sb[sb.stid.isin(HIGH) & (sb.month == "Jul")].set_index(["stid", "source"])[hcols].round(2).unstack("source").to_string())
    low = sb[~sb.stid.isin(HIGH)]
    print("\nlow stations, median |bias| by source and month:")
    print(low.assign(ab=low.bias.abs()).pivot_table(index="source", columns="month", values="ab", aggfunc="median").round(2).to_string())

    # (c) station pressure check
    sp = []
    for m in ("01", "07"):
        f = f"{ROOT}/data/v2/obs_hourly/2025/{m}"
        for st_file in os.listdir(f):
            sp.append(pd.read_parquet(f"{f}/{st_file}", columns=["stid", "time_utc", "pressure_mean_hpa"],
                                      filters=[("stid", "in", list(meta.index))]))
    SP = pd.concat(sp).dropna()
    SP["stid"] = SP.stid.astype(str)
    SP["time_utc"] = SP.time_utc.astype("datetime64[us, UTC]")
    SPw = SP.set_index(["stid", "time_utc"]).pressure_mean_hpa
    prev = SPw.rename("p_prev").reset_index()
    prev["time_utc"] = prev.time_utc + pd.Timedelta(hours=1)          # bin H-1 keyed on H
    pm = SP.rename(columns={"pressure_mean_hpa": "p_cur"}).merge(prev, on=["stid", "time_utc"])
    pm["p_obs"] = (pm.p_cur + pm.p_prev) / 2                          # centred on H:00
    pm = pm.rename(columns={"time_utc": "valid_utc"})[["stid", "valid_utc", "p_obs"]]
    A["stn_elev"] = A.stid.map(meta.elevation_m)
    dz = A.stn_elev - A.hgt_m
    Tm = A.t2m_k - GAMMA * dz / 2
    A["pstn_grib_hpa"] = A.psfc_hpa * np.exp(-G * dz / (RD * Tm))
    C = A.merge(pm, on=["stid", "valid_utc"])
    C["month"] = C.valid_utc.dt.month.map({1: "Jan", 7: "Jul"})
    C["e"] = C.pstn_grib_hpa - C.p_obs
    cs = C.groupby(["stid", "month"]).agg(n=("e", "size"), bias=("e", "mean"), sd=("e", "std"),
                                         dz_m=("stn_elev", "first")).reset_index()
    cs["dz_m"] = cs.stid.map(meta.elevation_m) - cs.stid.map(A.groupby("stid").hgt_m.first())
    cs.round(3).to_csv(f"{OUT}/station_pressure_check.csv", index=False)
    print("\n== (c) GRIB station pressure (PRES:surface adjusted to station elev) vs observed, hPa ==")
    ct = cs.pivot_table(index="stid", columns="month", values=["bias", "sd"])
    ct.columns = [f"{v}_{m}" for v, m in ct.columns]
    print(ct.join(cs.groupby("stid").dz_m.first()).round(2)
          .loc[sorted(meta.index, key=lambda s: meta.elevation_m[s])].to_string())

    # (d) gradient bias per pair
    W = M.pivot_table(index="valid_utc", columns="stid", values=list(SRC) + ["slp_obs"])
    gr = []
    for pn, (s1, s2) in pairs.items():
        go = W["slp_obs"][s1] - W["slp_obs"][s2]
        for c, lab in SRC.items():
            e = (W[c][s1] - W[c][s2] - go).dropna()
            for mo, em in e.groupby(e.index.month):
                gr.append(dict(gradient=pn, source=lab, month={1: "Jan", 7: "Jul"}[mo], n=len(em), bias=em.mean(),
                               rmse=np.sqrt((em ** 2).mean())))
    gr = pd.DataFrame(gr)
    gr.round(3).to_csv(f"{OUT}/gradient_bias.csv", index=False)
    print("\n== (d) day-0 gradient bias vs METAR (hPa) ==")
    gt = gr.pivot_table(index="gradient", columns=["source", "month"], values="bias")
    print(gt.round(2).loc[KEY_PAIRS + [p for p in gt.index if p not in KEY_PAIRS]].to_string())

    # ---- figures
    INK, MUTED, GRID = "#0b0b0b", "#8a8986", "#e5e4e0"
    COL = {"PRMSL": "#2a78d6", "MSLET": "#eb6834", "Open-Meteo": "#1baf7a"}
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.6, "axes.edgecolor": MUTED, "font.size": 9, "legend.frameon": False})
    os.makedirs(IMG, exist_ok=True)
    order = sorted(HIGH, key=lambda s: meta.elevation_m[s])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    w = 0.26
    for ax, mo in zip(axes, ("Jan", "Jul")):
        x = np.arange(len(order))
        for i, lab in enumerate(SRC.values()):
            v = sb[(sb.month == mo) & (sb.source == lab)].set_index("stid").reindex(order).bias
            ax.bar(x + (i - 1) * w, v, width=w - 0.03, color=COL[lab], label=lab)
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(x, [f"{s[1:]}\n{meta.elevation_m[s]:.0f} m" for s in order])
        ax.set_title(f"{mo} 2025: SLP minus METAR SLP, high stations (00/06/12/18Z analyses)", loc="left")
    axes[0].set_ylabel("bias (hPa)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(f"{IMG}/high_station_slp_bias_prmsl_mslet_om.png", dpi=130)
    plt.close(fig)

    allst = sorted(meta.index, key=lambda s: meta.elevation_m[s])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, mo in zip(axes, ("Jan", "Jul")):
        x = np.arange(len(allst))
        v_slp = sb[(sb.month == mo) & (sb.source == "PRMSL")].set_index("stid").reindex(allst).bias
        v_stn = cs[cs.month == mo].set_index("stid").reindex(allst).bias
        ax.bar(x - 0.2, v_slp, width=0.37, color=COL["PRMSL"], label="PRMSL - METAR SLP")
        ax.bar(x + 0.2, v_stn, width=0.37, color=MUTED, label="GRIB station pressure - observed station pressure")
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(x, [s[1:] for s in allst], rotation=90)
        ax.set_title(f"{mo} 2025 (stations ordered by elevation)", loc="left")
    axes[0].set_ylabel("bias (hPa)")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{IMG}/station_pressure_vs_slp_bias.png", dpi=130)
    plt.close(fig)
    print(f"\nwrote {OUT}/ and {IMG}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["download", "extract", "analyze", "all"])
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if a.step in ("download", "all"):
        download(a.workers)
    if a.step in ("extract", "all"):
        extract()
    if a.step in ("analyze", "all"):
        analyze()
