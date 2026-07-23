#!/usr/bin/env python
"""
Mark_model_v2 -- Step 2: Event rainfall from 800 m PRISM (upgraded from IEMRE 12 km).

Only the rainfall SOURCE changes vs the delivered Step 2: basin-mean daily rainfall is
now the simple mean of the 800 m PRISM grid cells whose centres fall inside each catchment
(native ~800 m, no convective smearing over small basins), instead of the 0.125 deg IEMRE
Stage IV analysis. The event-pairing and antecedent-wetness logic are preserved verbatim:

  P_e_mm         basin-mean rainfall summed over the event window [start..end]
  P_topeak_mm    rainfall summed [start..peak]
  V_e_m3         P_e * catchment area
  runoff_coeff   quick_vol_m3 / V_e_m3
  storm_resp_ratio, peakstage_resp
  P_eff_mm       effective cumulative rainfall through event start,
                 P_eff = sum_{n=0..30} k^n P(t-n), k=0.9   (Kohler & Linsley 1951 API,
                 off-by-one fixed; day-0 included). api_30_mm is written as an ALIAS of
                 P_eff_mm so the unchanged downstream fit_models control list still resolves.
  max_daily_mm   peak 1-day intensity within the window

Reads read-only: PRISM 800 m yearly csv.gz, catchment polygons, and this run's RREDI events.
Writes only into Mark_model_v2/.
"""
from __future__ import annotations
import os, time, glob, warnings
import numpy as np, pandas as pd, geopandas as gpd
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data"); OUT = os.path.join(HERE, "outputs")
os.makedirs(DATA, exist_ok=True); os.makedirs(OUT, exist_ok=True)
SENS = "/Users/jared/Wetland/sensors"
PRISM = "/Users/jared/Wetland/Mark's Model/step1_new_model_handoff/800m_data"

SITES = ["04087000", "04086500", "04086600", "04087120", "04087030",
         "04087050", "04087070", "04087088", "04087119"]
API_K, API_N = 0.9, 30                 # Kohler & Linsley (1951); k=0.90, 30-day
TZ = "America/Chicago"                  # PRISM daily = local calendar day
YR0, YR1 = 2000, 2023                   # incl. 2000 for antecedent of early-2001 events

# ---------------------------------------------------------------- PRISM basin average
def cell_membership():
    """{site: [cell_id,...]} for PRISM cells whose centre falls in each catchment."""
    from shapely.geometry import Point
    ref = pd.read_csv(glob.glob(os.path.join(PRISM, "*_2018.csv.gz"))[0],
                      usecols=["cell_id", "lon", "lat"]).drop_duplicates("cell_id")
    pts = gpd.GeoDataFrame(ref, geometry=[Point(x, y) for x, y in zip(ref.lon, ref.lat)],
                           crs=4326)
    cats = []
    for s in SITES:
        g = gpd.read_file(os.path.join(SENS, f"catchment_{s}.gpkg")).to_crs(4326)
        gg = g.dissolve(); gg["site_no"] = s; cats.append(gg[["site_no", "geometry"]])
    cats = pd.concat(cats, ignore_index=True)
    j = gpd.sjoin(pts, gpd.GeoDataFrame(cats, crs=4326), predicate="within", how="inner")
    return {s: sorted(j.loc[j.site_no == s, "cell_id"].unique()) for s in SITES}

def build_all_daily():
    """One pass over PRISM years -> basin-mean daily precip per site, cached."""
    caches = {s: os.path.join(DATA, f"precip_daily_{s}.parquet") for s in SITES}
    if all(os.path.exists(c) for c in caches.values()):
        return {s: pd.read_parquet(caches[s]) for s in SITES}
    mem = cell_membership()
    ncell = {s: len(v) for s, v in mem.items()}
    frames = {s: [] for s in SITES}
    for yr in range(YR0, YR1 + 1):
        fp = glob.glob(os.path.join(PRISM, f"*_{yr}.csv.gz"))
        if not fp:
            continue
        df = pd.read_csv(fp[0], usecols=["date", "cell_id", "precip_mm"], dtype={"cell_id": str})
        df["precip_mm"] = pd.to_numeric(df["precip_mm"], errors="coerce")
        for s in SITES:
            sub = df[df.cell_id.isin(mem[s])]
            daily = sub.groupby("date")["precip_mm"].mean()
            frames[s].append(daily)
        print(f"  PRISM {yr} read ({len(df):,} rows)", flush=True)
    out = {}
    for s in SITES:
        d = pd.concat(frames[s]).sort_index()
        d.index = pd.to_datetime(d.index)
        dd = pd.DataFrame({"precip_mm": d})
        dd.to_parquet(caches[s]); out[s] = dd
    return out

# ---------------------------------------------------------------- antecedent (P_eff)
def api_series(daily):
    """Effective rainfall P_eff(t) = sum_{n=0..N} k^n P(t-n); day t included (k^0=1)."""
    p = daily["precip_mm"].fillna(0.0).to_numpy()
    api = np.zeros_like(p)
    for n in range(0, API_N + 1):
        if n == 0:
            api += p
        else:
            api[n:] += (API_K ** n) * p[:-n]
    return pd.Series(api, index=daily.index)

# ---------------------------------------------------------------- event pairing (verbatim + alias)
def pair_events(site, daily, area_m2):
    ev = pd.read_parquet(os.path.join(DATA, f"events_{site}.parquet"))
    api = api_series(daily)
    p = daily["precip_mm"].fillna(0.0)
    def local_dates(ts):
        return pd.Timestamp(ts).tz_convert(TZ).normalize().tz_localize(None)
    rows = []
    for _, r in ev.iterrows():
        d0, dpk, d1 = local_dates(r.t_start), local_dates(r.t_peak), local_dates(r.t_end)
        win = p.loc[d0:d1]; topeak = p.loc[d0:dpk]
        P_e = float(win.sum()); V_e = P_e / 1000.0 * area_m2
        rc = (r.quick_vol_m3 / V_e) if V_e > 0 else np.nan
        api_v = float(api.loc[:d0].iloc[-1]) if (api.index <= d0).any() else np.nan
        row = r.to_dict()
        row.update(dict(
            P_e_mm=P_e, P_topeak_mm=float(topeak.sum()), V_e_m3=V_e,
            max_daily_mm=float(win.max()) if len(win) else np.nan,
            runoff_coeff=rc,
            storm_resp_ratio=(r.Qp_cfs / P_e) if P_e > 0 else np.nan,
            peakstage_resp=(r.peak_stage_ft / P_e) if (P_e > 0 and np.isfinite(r.peak_stage_ft)) else np.nan,
            P_eff_mm=api_v,
            api_30_mm=api_v,                         # ALIAS: unchanged fit_models control name
            catch_area_km2=area_m2 / 1e6,
        ))
        rows.append(row)
    return pd.DataFrame(rows)

def area_of(site):
    return float(gpd.read_file(os.path.join(SENS, f"catchment_{site}.gpkg"))
                 .to_crs(32616).geometry.union_all().area)

# ---------------------------------------------------------------- QA + main
def qa(all_ev):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sites = all_ev.site_no.unique(); n = len(sites); nrow = int(np.ceil(n / 3))
    fig, axes = plt.subplots(nrow, 3, figsize=(13, 3 * nrow))
    for ax, s in zip(axes.ravel(), sites):
        d = all_ev[all_ev.site_no == s]
        ax.scatter(d.P_e_mm, d.Qp_cfs, s=8, alpha=0.4)
        ax.set_title(s, fontsize=9); ax.set_xlabel("P_e (mm)"); ax.set_ylabel("Qp (cfs)")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Step 2 QA (800 m PRISM): event peak vs basin rainfall P_e", y=1.01)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "step2_rain_qa.png"), dpi=120,
                                    bbox_inches="tight"); plt.close(fig)

def run():
    print("building PRISM basin-mean daily series ...", flush=True)
    daily_all = build_all_daily()
    all_ev, summ = [], []
    for site in SITES:
        t0 = time.time(); area = area_of(site)
        daily = daily_all[site]
        evp = pair_events(site, daily, area)
        evp.to_parquet(os.path.join(DATA, f"events_rain_{site}.parquet"))
        all_ev.append(evp)
        rc = evp.runoff_coeff.replace([np.inf, -np.inf], np.nan)
        ann = daily["precip_mm"].resample("YE").sum().mean()
        summ.append(dict(site_no=site, area_km2=round(area/1e6, 1), n_events=len(evp),
                         ann_precip_mm=round(ann, 0), med_Pe_mm=round(evp.P_e_mm.median(), 1),
                         med_runoff_coeff=round(rc.median(), 3),
                         rc_gt1_pct=round(100*(rc > 1).mean(), 1),
                         med_Peff_mm=round(evp.P_eff_mm.median(), 1)))
        print(f"[{site}] area={area/1e6:.0f}km2 events={len(evp)} ann={ann:.0f}mm "
              f"medP={evp.P_e_mm.median():.1f} medRC={rc.median():.2f} ({round(time.time()-t0,1)}s)", flush=True)
    allev = pd.concat(all_ev, ignore_index=True)
    allev.to_csv(os.path.join(OUT, "events_rain_all.csv"), index=False)
    sdf = pd.DataFrame(summ); sdf.to_csv(os.path.join(OUT, "step2_summary.csv"), index=False)
    qa(allev)
    print("\n=== STEP 2 SUMMARY (800 m PRISM) ===")
    print(sdf.to_string(index=False))
    print(f"\nTotal paired events: {len(allev)} | finite runoff_coeff: "
          f"{allev.runoff_coeff.replace([np.inf,-np.inf],np.nan).notna().sum()}")

if __name__ == "__main__":
    run()
