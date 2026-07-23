"""Yearly land cover (LCMAP, USGS) per sensor catchment, 1985–2023.

LCMAP LCPRI primary land-cover classes:
  1 Developed | 2 Cropland | 3 Grass/Shrub | 4 Tree | 5 Water
  6 WETLAND   | 7 Ice/Snow | 8 Barren
We extract, per catchment per year:
  wetland_km2, wetland_frac  (class 6)   — the explanatory variable of interest
  developed_frac             (class 1)   — urbanisation confounder

Source: Microsoft Planetary Computer STAC (collection usgs-lcmap-conus-v13),
COGs in EPSG:5070 (Albers), 30 m. Per-year AOI mosaics are cached to data/lcmap/.
"""
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
import rioxarray as rxr
from rioxarray.merge import merge_arrays
from rasterio.features import rasterize
import planetary_computer, pystac_client
warnings.filterwarnings("ignore")

ROOT=Path("/Users/jared/Wetland"); DATA=ROOT/"data"; SENS=ROOT/"sensors"; OUT=ROOT/"outputs"/"sensors"
LC=DATA/"lcmap"; LC.mkdir(parents=True, exist_ok=True)
ALBERS="EPSG:5070"
SENSOR_IDS=["04087000","04086600","04086500","04087120","04087030",
            "04087050","04087070","04087088","04087119"]
_PANEL_SF = Path("/Users/jared/Wetland/sensors/panel_sites.txt")
if _PANEL_SF.exists(): SENSOR_IDS = _PANEL_SF.read_text().split()
YEARS=[int(y) for y in (sys.argv[1:] or range(1985,2024))]
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# AOI + catchments in Albers
aoi=gpd.read_file(SENS/"aoi_watershed.gpkg").to_crs(4326)
bbox4326=list(aoi.total_bounds)
aoi_alb=aoi.to_crs(ALBERS)
w,s,e,n=aoi_alb.total_bounds; pad=300
bbox_alb=(w-pad,s-pad,e+pad,n+pad)
catch={sid: gpd.read_file(SENS/f"catchment_{sid}.gpkg").to_crs(ALBERS).geometry.iloc[0]
       for sid in SENSOR_IDS if (SENS/f"catchment_{sid}.gpkg").exists()}
log(f"catchments: {len(catch)}")

cat=pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                              modifier=planetary_computer.sign_inplace)

def aoi_lcpri(year):
    f=LC/f"lcpri_{year}.tif"
    if f.exists():
        return rxr.open_rasterio(f, masked=False).squeeze()
    items=list(cat.search(collections=["usgs-lcmap-conus-v13"], bbox=bbox4326,
                          datetime=f"{year}-01-01/{year}-12-31").items())
    if not items: raise RuntimeError(f"no LCMAP items for {year}")
    arrs=[]
    for it in items:
        da=rxr.open_rasterio(it.assets["lcpri"].href, masked=False).squeeze()
        arrs.append(da.rio.clip_box(*bbox_alb))
    mos=merge_arrays(arrs) if len(arrs)>1 else arrs[0]
    mos=mos.rio.clip_box(*bbox_alb)
    mos.rio.to_raster(f, compress="lzw")
    return mos

# rasterize catchment masks once, on the first year's grid
ref=aoi_lcpri(YEARS[0])
tr=ref.rio.transform(); shp=ref.shape[-2:]
masks={sid: rasterize([(g,1)], out_shape=shp, transform=tr, dtype="uint8").astype(bool)
       for sid,g in catch.items()}
for sid in masks: log(f"  {sid}: {masks[sid].sum():,} LCMAP cells = {masks[sid].sum()*900/1e6:.1f} km²")

rows=[]
for yr in YEARS:
    try:
        arr=np.array(aoi_lcpri(yr)).squeeze()
    except Exception as ex:
        log(f"{yr}: FAILED {ex}"); continue
    for sid,m in masks.items():
        v=arr[m]; tot=int((v>0).sum())
        if tot==0: continue
        wet=int((v==6).sum()); dev=int((v==1).sum())
        rows.append(dict(site_no=sid, year=yr,
                         wetland_km2=round(wet*900/1e6,3),
                         wetland_frac=round(wet/tot,5),
                         developed_frac=round(dev/tot,5),
                         catch_cells=tot))
    log(f"{yr} done")

df=pd.DataFrame(rows).sort_values(["site_no","year"])
df.to_csv(OUT/"panel_landcover_lcmap.csv", index=False)
log(f"Wrote {OUT/'panel_landcover_lcmap.csv'}  ({len(df)} rows)")
# quick wetland-trend peek
if len(df):
    piv=df.pivot_table(index="year",columns="site_no",values="wetland_km2")
    print(piv.iloc[[0,-1]].round(2).to_string())
