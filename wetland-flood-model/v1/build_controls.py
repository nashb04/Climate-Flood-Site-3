#!/usr/bin/env python
"""
Mark's Model — Step 4: Control variables (to separate the wetland signal from
urbanization, soils, terrain, size).

Per gauge / catchment:
  DA_km2         drainage area                         (catchment polygon)
  imp_{yr}       NLCD impervious basin-mean %          (2001, 2011, 2021 -> interp by event yr)
  ksat_log       POLARIS ksat_5 basin-mean (log10 cm/hr, ~geometric-mean Ksat)  infiltration
  aws_mm         gNATSGO available water storage 0-100 cm, basin-mean (mm)
  sand_pct       POLARIS sand_5 basin-mean (%)         texture / infiltration proxy
  basin_slope    mean terrain slope over catchment (m/m)      from 3DEP DEM
  chan_slope     mean slope over channel cells (acc>5000)     from 3DEP DEM

Outputs: outputs/panel_controls.csv (static, one row/gauge)
         outputs/imp_by_year.csv    (site x {2001,2011,2021} impervious % for Step-5 interp)
API (antecedent precip) is per-event and already produced in Step 2.
"""
from __future__ import annotations
import os, time, warnings
import numpy as np, pandas as pd
import geopandas as gpd, rasterio
from rasterio.features import rasterize
import pygeohydro as gh
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "outputs")
ROOT = "/Users/jared/Wetland"; DATA = os.path.join(ROOT, "data"); SENS = os.path.join(ROOT, "sensors")
SITES = ["04087000", "04086500", "04086600", "04087120", "04087030",
         "04087050", "04087070", "04087088", "04087119"]
IMP_YEARS = [2001, 2011, 2021]
ACC_THRESH = 5000

# ---------------------------------------------------------------- DEM slope (once)
_slope = {}
def aoi_slope():
    if _slope:
        return _slope
    with rasterio.open(os.path.join(DATA, "T_04087000.tif")) as r:
        transform, shape = r.transform, (r.height, r.width)
    with rasterio.open(os.path.join(DATA, "aoi_dem_10m.tif")) as r:
        dem = r.read(1).astype("float32")
    gy, gx = np.gradient(dem, 10.0)              # 10 m cells
    slope = np.sqrt(gx * gx + gy * gy).astype("float32")   # m/m
    acc = np.load(os.path.join(DATA, "aoi_acc.npy"))
    stream = acc > ACC_THRESH
    _slope.update(transform=transform, shape=shape, slope=slope, stream=stream)
    return _slope

def slopes_for(catch32616, lay):
    cmask = rasterize([(catch32616.geometry.union_all(), 1)], out_shape=lay["shape"],
                      transform=lay["transform"], dtype="uint8").astype(bool)
    s = lay["slope"]
    basin = float(np.nanmean(s[cmask & np.isfinite(s)]))
    ch = cmask & lay["stream"] & np.isfinite(s)
    chan = float(np.nanmean(s[ch])) if ch.any() else np.nan
    return basin, chan

# ---------------------------------------------------------------- soils / impervious
def basin_mean_impervious(g, year):
    d = gh.nlcd_bygeom(g, years={"impervious": [year]}, resolution=30)
    ds = d[list(d)[0]] if isinstance(d, dict) else d
    a = ds[f"impervious_{year}"].values.astype(float)
    a = np.where(a > 100, np.nan, a)
    return float(np.nanmean(a))

def _pick(ds, key):
    v = [d for d in ds.data_vars if key in d.lower()]
    return ds[v[0]].values.astype(float) if v else np.array([np.nan])

def basin_mean_soil(geom):
    out = {}
    try:
        pol = gh.soil_polaris(["ksat_5", "sand_5"], geom, 4326)  # returns *_0_5cm_mean vars
        out["ksat_log"] = round(float(np.nanmean(_pick(pol, "ksat"))), 3)
        out["sand_pct"] = round(float(np.nanmean(_pick(pol, "sand"))), 1)
    except Exception as e:
        print("   POLARIS fail:", str(e)[:100]); out["ksat_log"] = np.nan; out["sand_pct"] = np.nan
    try:
        gn = gh.soil_gnatsgo("aws0_100", geom, 4326)
        out["aws_mm"] = round(float(np.nanmean(gn["aws0_100"].values.astype(float))), 1)
    except Exception as e:
        print("   gNATSGO fail:", str(e)[:100]); out["aws_mm"] = np.nan
    return out

# ---------------------------------------------------------------- main
def run():
    lay = aoi_slope()
    rows, imp_rows = [], []
    for site in SITES:
        t0 = time.time()
        g = gpd.read_file(os.path.join(SENS, f"catchment_{site}.gpkg"))
        g4326 = g.to_crs(4326); geom = g4326.geometry.union_all()
        DA = float(g.to_crs(32616).geometry.union_all().area) / 1e6
        basin_slope, chan_slope = slopes_for(g.to_crs(32616), lay)
        soil = basin_mean_soil(geom)
        imp = {}
        for yr in IMP_YEARS:
            try:
                imp[yr] = round(basin_mean_impervious(g4326, yr), 2)
            except Exception as e:
                print(f"   IMP {yr} fail:", str(e)[:90]); imp[yr] = np.nan
        rec = dict(site_no=site, DA_km2=round(DA, 1),
                   basin_slope=round(basin_slope, 4), chan_slope=round(chan_slope, 4),
                   imp_2001=imp[2001], imp_2011=imp[2011], imp_2021=imp[2021], **soil)
        rows.append(rec)
        imp_rows.append(dict(site_no=site, **{str(y): imp[y] for y in IMP_YEARS}))
        print(f"[{site}] DA={DA:.0f} imp01/11/21={imp[2001]}/{imp[2011]}/{imp[2021]}% "
              f"ksat_log={soil.get('ksat_log')} aws={soil.get('aws_mm')} sand={soil.get('sand_pct')} "
              f"slope b/c={basin_slope:.3f}/{chan_slope:.3f} ({round(time.time()-t0,1)}s)", flush=True)
    panel = pd.DataFrame(rows); panel.to_csv(os.path.join(OUT, "panel_controls.csv"), index=False)
    pd.DataFrame(imp_rows).to_csv(os.path.join(OUT, "imp_by_year.csv"), index=False)
    print("\n=== STEP 4 SUMMARY (panel_controls.csv) ===")
    print(panel.to_string(index=False))

if __name__ == "__main__":
    run()
