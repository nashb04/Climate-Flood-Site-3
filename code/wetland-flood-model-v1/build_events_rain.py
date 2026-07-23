#!/usr/bin/env python
"""
Mark's Model — Step 2: Event rainfall (P_e, V_e, API, intensity) from radar QPE.

Source: IEM IEMRE web service (https://mesonet.agron.iastate.edu/iemre/), which serves
NCEP Stage IV radar-gauge precipitation on a 0.125 deg (~12 km) analysis grid, 1997+.
Field `daily_precip_in` is the Stage IV-based calendar-day analysis (primary); we also
keep `mrms_precip_in` (1 km MRMS, 2014+) for QA / later upgrade.

Basin-average P_e: lay uniform sample points in each catchment, snap to the 0.125 deg
lattice, pull each unique node's daily series once (cached), then area-weight by the
sample-point count per node.

Per event (from Step 1) we add:
  P_e_mm            basin-mean rainfall summed over the event window [start..end]
  P_topeak_mm       rainfall summed [start..peak]
  V_e_m3            P_e * catchment area
  runoff_coeff      quick_vol_m3 / V_e_m3            (event runoff coefficient)
  storm_resp_ratio  Qp_cfs / P_e_mm
  peakstage_resp    peak_stage_ft / P_e_mm          (NaN where no stage)
  api_30_mm         antecedent precip index at start (k=0.9, 30-day)
  max_daily_mm      peak 1-day intensity within the window
Outputs: data/precip_daily_{site}.parquet, data/events_rain_{site}.parquet,
         outputs/events_rain_all.csv, outputs/step2_rain_qa.png
"""
from __future__ import annotations
import os, time, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data"); OUT = os.path.join(HERE, "outputs")
NODES = os.path.join(DATA, "iemre_nodes"); os.makedirs(NODES, exist_ok=True)
SENS = "/Users/jared/Wetland/sensors"

SITES = ["04087000", "04086500", "04086600", "04087120", "04087030",
         "04087050", "04087070", "04087088", "04087119"]
START, END = "2001-01-01", "2026-12-31"
GRID = 0.125                # IEMRE lattice spacing (deg)
SAMPLE = 0.02              # sample-point spacing inside catchments (deg)
API_K, API_N = 0.9, 30     # antecedent precip index
IN_TO_MM = 25.4
TZ = "America/Chicago"     # IEMRE daily = local calendar day

# ---------------------------------------------------------------- IEMRE pull
def snap(v):  # snap a coordinate to the 0.125 deg lattice
    return round(round(v / GRID) * GRID, 4)

def pull_node(lat, lon):
    """Daily Stage IV (+MRMS) precip [mm] for one lattice node, cached, chunked."""
    cache = os.path.join(NODES, f"node_{lat:.4f}_{lon:.4f}.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    frames = []
    for s, e in [("2001-01-01", "2008-12-31"), ("2009-01-01", "2016-12-31"),
                 ("2017-01-01", END)]:
        u = f"https://mesonet.agron.iastate.edu/iemre/multiday/{s}/{e}/{lat}/{lon}/json"
        for attempt in range(3):
            try:
                d = requests.get(u, timeout=120).json().get("data", [])
                break
            except Exception:
                time.sleep(2); d = []
        if d:
            df = pd.DataFrame(d)[["date", "daily_precip_in", "mrms_precip_in"]]
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["date", "stage4_mm", "mrms_mm"]).set_index("date")
    df = pd.concat(frames).drop_duplicates("date")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    out = pd.DataFrame(index=df.index)
    out["stage4_mm"] = pd.to_numeric(df["daily_precip_in"], errors="coerce") * IN_TO_MM
    out["mrms_mm"] = pd.to_numeric(df["mrms_precip_in"], errors="coerce") * IN_TO_MM
    out.to_parquet(cache)
    return out

# ---------------------------------------------------------------- basin average
def basin_nodes(site):
    """Return {(lat,lon): weight} of lattice nodes covering the catchment."""
    g = gpd.read_file(os.path.join(SENS, f"catchment_{site}.gpkg"))
    area_m2 = float(g.to_crs(32616).geometry.union_all().area)
    poly = g.to_crs(4326).geometry.union_all()
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(minx, maxx + SAMPLE, SAMPLE)
    ys = np.arange(miny, maxy + SAMPLE, SAMPLE)
    from shapely.geometry import Point
    counts = {}
    for x in xs:
        for y in ys:
            if poly.contains(Point(x, y)):
                key = (snap(y), snap(x))
                counts[key] = counts.get(key, 0) + 1
    if not counts:                                   # tiny basin: use centroid node
        c = poly.centroid
        counts[(snap(c.y), snap(c.x))] = 1
    tot = sum(counts.values())
    weights = {k: v / tot for k, v in counts.items()}
    return weights, area_m2

def basin_daily(site):
    cache = os.path.join(DATA, f"precip_daily_{site}.parquet")
    weights, area_m2 = basin_nodes(site)
    if os.path.exists(cache):
        return pd.read_parquet(cache), area_m2, len(weights)
    acc = None
    for (lat, lon), w in weights.items():
        nd = pull_node(lat, lon)
        if len(nd) == 0:
            continue
        contrib = nd[["stage4_mm", "mrms_mm"]].fillna(0.0) * w
        acc = contrib if acc is None else acc.add(contrib, fill_value=0.0)
    acc = acc.sort_index()
    # prefer MRMS where it exists & is nonzero-capable (2014+), else Stage IV
    acc["precip_mm"] = acc["stage4_mm"]
    acc.to_parquet(cache)
    return acc, area_m2, len(weights)

def api_series(daily):
    """Antecedent precip index: API(t)=sum_{n=1..N} k^n * P(t-n)."""
    p = daily["precip_mm"].fillna(0.0).to_numpy()
    api = np.zeros_like(p)
    for n in range(1, API_N + 1):
        api[n:] += (API_K ** n) * p[:-n]
    return pd.Series(api, index=daily.index)

# ---------------------------------------------------------------- event pairing
def pair_events(site, daily, area_m2):
    ev = pd.read_parquet(os.path.join(DATA, f"events_{site}.parquet"))
    api = api_series(daily)
    p = daily["precip_mm"].fillna(0.0)
    # local-day index for matching UTC event windows
    def local_dates(ts):
        return pd.Timestamp(ts).tz_convert(TZ).normalize().tz_localize(None)
    rows = []
    for _, r in ev.iterrows():
        d0, dpk, d1 = local_dates(r.t_start), local_dates(r.t_peak), local_dates(r.t_end)
        win = p.loc[d0:d1]
        topeak = p.loc[d0:dpk]
        P_e = float(win.sum())
        V_e = P_e / 1000.0 * area_m2                      # m3
        rc = (r.quick_vol_m3 / V_e) if V_e > 0 else np.nan
        prev = d0 - pd.Timedelta(days=1)
        api_v = float(api.loc[:prev].iloc[-1]) if (api.index <= prev).any() else np.nan
        row = r.to_dict()
        row.update(dict(
            P_e_mm=P_e, P_topeak_mm=float(topeak.sum()), V_e_m3=V_e,
            max_daily_mm=float(win.max()) if len(win) else np.nan,
            runoff_coeff=rc,
            storm_resp_ratio=(r.Qp_cfs / P_e) if P_e > 0 else np.nan,
            peakstage_resp=(r.peak_stage_ft / P_e) if (P_e > 0 and np.isfinite(r.peak_stage_ft)) else np.nan,
            api_30_mm=api_v,
            catch_area_km2=area_m2 / 1e6,
        ))
        rows.append(row)
    return pd.DataFrame(rows)

# ---------------------------------------------------------------- QA
def qa(all_ev):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sites = all_ev.site_no.unique()
    n = len(sites); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3 * nrow))
    for ax, s in zip(axes.ravel(), sites):
        d = all_ev[all_ev.site_no == s]
        ax.scatter(d.P_e_mm, d.Qp_cfs, s=8, alpha=0.4)
        ax.set_title(s, fontsize=9); ax.set_xlabel("P_e (mm)"); ax.set_ylabel("Qp (cfs)")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Step 2 QA: event peak discharge vs basin rainfall P_e", y=1.01)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "step2_rain_qa.png"), dpi=120,
                                    bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- main
def run():
    all_ev, summ = [], []
    for site in SITES:
        t0 = time.time()
        daily, area_m2, nnodes = basin_daily(site)
        evp = pair_events(site, daily, area_m2)
        evp.to_parquet(os.path.join(DATA, f"events_rain_{site}.parquet"))
        all_ev.append(evp)
        rc = evp.runoff_coeff.replace([np.inf, -np.inf], np.nan)
        summ.append(dict(site_no=site, area_km2=round(area_m2 / 1e6, 1), nodes=nnodes,
                         n_events=len(evp),
                         med_Pe_mm=round(evp.P_e_mm.median(), 1),
                         med_runoff_coeff=round(rc.median(), 3),
                         rc_gt1_pct=round(100 * (rc > 1).mean(), 1),
                         med_api_mm=round(evp.api_30_mm.median(), 1)))
        print(f"[{site}] area={area_m2/1e6:.0f}km2 nodes={nnodes} events={len(evp)} "
              f"medP={evp.P_e_mm.median():.1f}mm medRC={rc.median():.2f} "
              f"({round(time.time()-t0,1)}s)", flush=True)
    allev = pd.concat(all_ev, ignore_index=True)
    allev.to_csv(os.path.join(OUT, "events_rain_all.csv"), index=False)
    sdf = pd.DataFrame(summ); sdf.to_csv(os.path.join(OUT, "step2_summary.csv"), index=False)
    qa(allev)
    print("\n=== STEP 2 SUMMARY ===")
    print(sdf.to_string(index=False))
    print(f"\nTotal paired events: {len(allev)}  | with finite runoff_coeff: "
          f"{allev.runoff_coeff.replace([np.inf,-np.inf],np.nan).notna().sum()}")

if __name__ == "__main__":
    run()
