#!/usr/bin/env python
"""
Wetland V2 -- Steps 2-4: new cumulative effectiveness W following the advisor's form
    W_j = sum_i  A_i * S_i * K(T_ij)          (NO connectivity term C -- redundant with K)
with
    S_i = M_type,i * d_i        d_i = per-wetland storage depth (Volume-Area)
    => A_i * S_i = M_type,i * V_i   (type-weighted wetland storage volume)
    K(T) = exp(-T/tau)          T from the improved celerity+roughness field T_v2
So W_j = sum_i M_type,i * V_i * K(T_ij) = connectivity-delivered, type-weighted wetland storage.

Literature choices:
  M_type  -- Cowardin/HGM attenuation weights (Acreman & Holden 2013): emergent/floodplain high,
             open-water/lacustrine low.
  V_i     -- glaciated Volume-Area power law V = 0.01725 * A^1.30086 (Wu & Lane 2016), replacing
             the placeholder 0.05*A^1.2; S_w = sum V_i.
  tau     -- catchment-median T_v2 (adaptive, now in physical celerity units); a fixed-tau variant
             and an event-lag CALIBRATION (SCS lag ~ 0.6 Tc) are also reported.

Reads existing caches read-only: ../nwi/nwi_{site}.gpkg, ../data/events_{site}.parquet,
../outputs/panel_W_Sw.csv (old W for comparison), data/T_v2_{site}.tif, sensors/catchment_*.gpkg.
Writes only inside wetland_v2/.
"""
from __future__ import annotations
import os, warnings
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.features import rasterize
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data"); OUT = os.path.join(HERE, "outputs")
MM = "/Users/jared/Wetland/Mark's Model"       # source caches: NWI polygons + old panel (read-only)
NWI = os.path.join(MM, "nwi"); EV = os.path.join(HERE, "data")   # events from THIS (RREDI) run
SENS = "/Users/jared/Wetland/sensors"
SITES = ["04087000", "04086500", "04086600", "04087120", "04087030",
         "04087050", "04087070", "04087088", "04087119"]
VA_C, VA_BETA = 0.01725, 1.30086              # Wu & Lane (2016) glaciated Volume-Area

def M_type(attr: str) -> float:
    """Cowardin -> attenuation weight (Acreman & Holden 2013 ordering)."""
    if not isinstance(attr, str) or not attr:
        return 0.7
    a = attr.upper(); sysL = a[0]
    if sysL == "L":  return 0.30                      # lacustrine (lakes)
    if sysL == "R":  return 0.60                      # riverine
    if "EM" in a:    return 1.00                       # palustrine emergent (marsh/floodplain)
    if "FO" in a or "SS" in a: return 0.85            # forested / scrub-shrub
    if "UB" in a or "AB" in a or "OW" in a: return 0.50  # open water / aquatic bed
    return 0.70

def rc_from_xy(x, y, tr, H, W):
    col = ((x - tr.c) / tr.a).astype(int); row = ((y - tr.f) / tr.e).astype(int)
    return np.clip(row, 0, H - 1), np.clip(col, 0, W - 1)

def run():
    old = pd.read_csv(os.path.join(MM, "outputs", "panel_W_Sw.csv"), dtype={"site_no": str})
    old["site_no"] = old["site_no"].str.zfill(8)
    rows, calib = [], []
    for site in SITES:
        with rasterio.open(os.path.join(DATA, f"T_v2_{site}.tif")) as r:
            Tg = r.read(1); tr = r.transform; H, W = r.height, r.width
        catch = gpd.read_file(os.path.join(SENS, f"catchment_{site}.gpkg")).to_crs(32616)
        area_m2 = float(catch.geometry.union_all().area)
        cmask = rasterize([(catch.geometry.union_all(), 1)], out_shape=(H, W),
                          transform=tr, dtype="uint8").astype(bool)
        Tc = Tg[cmask & np.isfinite(Tg) & (Tg > 0)]
        tau = float(np.median(Tc)) if Tc.size else 24.0

        gdf = gpd.read_file(os.path.join(NWI, f"nwi_{site}.gpkg")).to_crs(32616)
        cent = gdf.geometry.centroid
        A = gdf.geometry.area.to_numpy()
        row, col = rc_from_xy(cent.x.to_numpy(), cent.y.to_numpy(), tr, H, W)
        T_ij = Tg[row, col].astype(float)
        T_ij = np.where(np.isfinite(T_ij) & (T_ij > 0), T_ij, tau)
        M = np.array([M_type(x) for x in gdf.get("ATTRIBUTE", pd.Series([None]*len(gdf))).fillna("")])
        V = VA_C * np.power(A, VA_BETA)                 # per-wetland storage volume (m3)
        d = V / A                                       # storage depth (m)
        S_i = M * d                                     # advisor's S_i = type * storage-depth
        K = np.exp(-T_ij / tau)

        AS = A * S_i                                    # = M * V
        W_v2        = float((AS * K).sum())             # PRIMARY: sum A*S*K = sum M*V*K
        W_typeonly  = float((A * M * K).sum())          # S_i = M only
        W_storonly  = float((V * K).sum())              # S_i = d only  (=> A*d*K = V*K)
        W_area      = float((A * K).sum())              # plain area, travel-time weighted
        # normalized fraction (comparable to old W): weight share in wetlands
        denomK = float(np.exp(-Tg[cmask & np.isfinite(Tg)] / tau).sum()) * 100.0  # cell area cancels via count
        W_frac = W_area / denomK if denomK > 0 else np.nan
        Sw_v2 = float(V.sum())

        rec = dict(site_no=site, n_wet=int(len(gdf)),
                   wet_km2=round(A.sum()/1e6, 3), tau_v2_hr=round(tau, 1),
                   meanT_wet_hr=round(float(np.average(T_ij, weights=A)), 1),
                   W_v2=round(W_v2, 1), W_v2_typeonly=round(W_typeonly, 1),
                   W_v2_storonly=round(W_storonly, 1), W_v2_area=round(W_area, 1),
                   W_v2_frac=round(W_frac, 5),
                   W_v2_perkm2=round(W_v2/(area_m2/1e6), 2),
                   Sw_v2_m3=round(Sw_v2, 1), Sw_v2_depth_mm=round(Sw_v2/area_m2*1000, 2))
        rows.append(rec)
        # calibration: observed hydrograph lag vs modeled travel time
        ev = pd.read_parquet(os.path.join(EV, f"events_{site}.parquet"))
        calib.append(dict(site_no=site, obs_ttp_med_hr=round(float(ev.time_to_peak_hr.median()), 1),
                          model_medT_v2_hr=round(tau, 1)))
        print(f"[{site}] W_v2={W_v2:.3e}  Sw_v2={Sw_v2:.3e} (depth {rec['Sw_v2_depth_mm']}mm)  "
              f"tau={tau:.0f}h  n_wet={len(gdf)}", flush=True)

    panel = pd.DataFrame(rows).merge(
        old[["site_no", "W_exp", "wet_frac", "near_frac", "Sw_va_m3", "tau_hr"]]
        .rename(columns={"W_exp": "W_old", "Sw_va_m3": "Sw_old", "tau_hr": "tau_old"}),
        on="site_no", how="left")
    panel.to_csv(os.path.join(OUT, "panel_W_v2.csv"), index=False)

    cal = pd.DataFrame(calib)
    # SCS: basin lag ~ alpha * Tc ; fit through origin across 9 gauges
    x = cal.model_medT_v2_hr.to_numpy(); y = cal.obs_ttp_med_hr.to_numpy()
    alpha = float((x @ y) / (x @ x))
    yhat = alpha * x; ss = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    cal["pred_lag_hr"] = np.round(alpha * x, 1)
    cal.to_csv(os.path.join(OUT, "calibration.csv"), index=False)

    print("\n=== W_v2 PANEL (new vs old) ===")
    show = ["site_no", "wet_km2", "tau_v2_hr", "W_v2", "W_v2_perkm2", "Sw_v2_depth_mm",
            "W_old", "wet_frac"]
    print(panel[show].to_string(index=False))
    print("\n=== CALIBRATION: observed hydrograph lag vs modeled travel time (9 gauges) ===")
    print(cal.to_string(index=False))
    print(f"\n  observed_lag ~= {alpha:.3f} * modeled_T_v2   (R2={ss:.2f})")
    print(f"  SCS expectation is lag ~ 0.6*Tc; alpha={alpha:.2f} "
          f"=> modeled travel time is {'physically reasonable' if 0.2<alpha<1.2 else 'off-scale (needs tuning)'}.")

if __name__ == "__main__":
    run()
