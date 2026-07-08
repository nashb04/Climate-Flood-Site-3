#!/usr/bin/env python
"""
Wetland V2 -- Step 1: improved travel-time field T_v2 (celerity + wetland roughness).

Rebuilds the per-cell velocity/travel-time on the SAME cached master grid used by the
original Weighted Wetland Model, changing two physics assumptions (both literature-based):

  (T1) CELERITY, not water velocity, for the flood wave: channel speed = beta * v_manning,
       beta = 5/3 for a Manning wide channel (kinematic wave, c = dQ/dA).  De-inflates T.
  (T3) WETLAND ROUGHNESS: overland flow through wetland cells is slowed (Kadlec 1990 --
       emergent-marsh Manning n ~ 0.05-0.15 >> upland), so T mechanistically reflects
       wetlands delaying the water that reaches the gauge.

Reads (never writes) the existing caches:
  ../../data/aoi_dem_10m_cond.tif   conditioned DEM (WhiteBox breach)   -> routing
  ../../data/aoi_acc.npy            flow accumulation                   -> channel mask
  ../../data/aoi_nlcd_wetland_2021.tif  NLCD wetland (==1)              -> roughness
  ../../data/T_{site}.tif           original travel time                -> outlet cell (argmin)

Writes  data/T_v2_{site}.tif  and  outputs/T_v2_compare.csv (old vs new median T).
Everything stays inside Mark's Model/wetland_v2/.
"""
from __future__ import annotations
import os, time, warnings
import numpy as np, pandas as pd, rasterio
warnings.filterwarnings("ignore")
if not hasattr(np, "in1d"):
    np.in1d = np.isin
from pysheds.grid import Grid
from pysheds.sview import Raster as PyRaster

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data"); OUT = os.path.join(HERE, "outputs")
SRC = "/Users/jared/Wetland/data"
SITES = ["04087000", "04086500", "04086600", "04087120", "04087030",
         "04087050", "04087070", "04087088", "04087119"]

# --- physics parameters (celerity + roughness); calibratable in Step 4 ---
ACC_THRESH = 5000            # channel = accumulation > 0.5 km2 (matches original)
SLOPE_MIN  = 1e-3
N_CHANNEL  = 0.04; R_CHANNEL = 0.7
BETA       = 5.0 / 3.0       # T1: kinematic celerity multiplier for channels
K_OVERLAND = 4.92            # TR-55 shallow-concentrated (unpaved) upland
K_WETLAND  = 1.50            # T3: wetlands slow overland flow (~3x, Kadlec high-n)


def build():
    t0 = time.time()
    with rasterio.open(os.path.join(SRC, "T_04087000.tif")) as r:
        transform, (H, W) = r.transform, (r.height, r.width)
        prof = r.profile

    print("routing conditioned DEM ...", flush=True)
    grid = Grid.from_raster(os.path.join(SRC, "aoi_dem_10m_cond.tif"))
    demc = grid.read_raster(os.path.join(SRC, "aoi_dem_10m_cond.tif"))
    infl = grid.resolve_flats(demc)
    fdir = grid.flowdir(infl)
    L = grid.cell_distances(fdir); L_arr = np.array(L, float)
    S = np.array(grid.cell_slopes(infl, fdir), float)
    S = np.where(np.isfinite(S) & (S > 0), S, SLOPE_MIN)

    acc = np.load(os.path.join(SRC, "aoi_acc.npy"))
    stream = acc > ACC_THRESH
    with rasterio.open(os.path.join(SRC, "aoi_nlcd_wetland_2021.tif")) as r:
        wet = (r.read(1) == 1)

    # speed grid: channel celerity vs land-cover overland velocity
    v_channel = (1.0 / N_CHANNEL) * (R_CHANNEL ** (2.0 / 3.0)) * np.sqrt(S)   # Manning v
    c_channel = BETA * v_channel                                             # T1 celerity
    K = np.where(wet, K_WETLAND, K_OVERLAND)                                 # T3 roughness
    v_overland = K * np.sqrt(S)
    speed = np.where(stream, c_channel, v_overland)                          # m/s
    tpc = L_arr / speed                                                      # seconds/cell
    tpc = np.where(np.isfinite(tpc) & (tpc > 0), tpc, np.nan)
    tpc_rast = PyRaster(tpc, viewfinder=fdir.viewfinder)

    rows = []
    for site in SITES:
        with rasterio.open(os.path.join(SRC, f"T_{site}.tif")) as r:
            Told = r.read(1)
        # outlet = the cell where the original travel time is 0 (the gauge)
        flat = np.where(np.isfinite(Told), Told, np.inf)
        rr, cc = np.unravel_index(np.argmin(flat), flat.shape)
        T = grid.distance_to_outlet(int(cc), int(rr), fdir, weights=tpc_rast,
                                    nodata_out=np.nan, routing="d8", xytype="index")
        T_hr = np.array(T, float) / 3600.0
        pr = prof.copy(); pr.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
        with rasterio.open(os.path.join(DATA, f"T_v2_{site}.tif"), "w", **pr) as dst:
            dst.write(T_hr.astype("float32")[None])
        fin = np.isfinite(T_hr) & (T_hr > 0)
        old_fin = np.isfinite(Told) & (Told > 0)
        rows.append(dict(site_no=site,
                         med_T_old_hr=round(float(np.median(Told[old_fin])), 1),
                         med_T_v2_hr=round(float(np.median(T_hr[fin])), 1),
                         max_T_v2_hr=round(float(np.nanmax(T_hr[fin])), 1),
                         deinflate_x=round(float(np.median(Told[old_fin]) /
                                                 max(np.median(T_hr[fin]), 1e-6)), 2)))
        print(f"[{site}] med T: old {rows[-1]['med_T_old_hr']}h -> v2 "
              f"{rows[-1]['med_T_v2_hr']}h  (÷{rows[-1]['deinflate_x']})", flush=True)

    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "T_v2_compare.csv"), index=False)
    print(f"\n=== T_v2 built ({round(time.time()-t0)}s) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    build()
