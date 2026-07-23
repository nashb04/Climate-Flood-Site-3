"""Add the Kinnickinnic / Wilson Park cluster (7 gauges just S of the AOI) on a
small dedicated grid, with the SAME methodology/params as the main pipeline, and
append their W + land-cover rows to the panels. Leaves the existing 50-gauge
results untouched (KK is a separate drainage to Lake Michigan).
"""
import os, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
import rioxarray as rxr, rasterio, rasterio.enums
from rioxarray.merge import merge_arrays
from rasterio.features import rasterize, shapes
from rasterio.transform import Affine
from shapely.geometry import Point, box, shape
from shapely.ops import unary_union
if not hasattr(np,"in1d"): np.in1d=np.isin
from pysheds.grid import Grid
from pysheds.sview import Raster as PyRaster
import py3dep, whitebox, planetary_computer, pystac_client
warnings.filterwarnings("ignore")
ROOT=Path("/Users/jared/Wetland"); DATA=ROOT/"data"; SENS=ROOT/"sensors"; OUT=ROOT/"outputs"/"sensors"
PROJ="EPSG:32616"; ALBERS="EPSG:5070"; DEM_RES=10; ACC_THRESH=5000; SLOPE_MIN=1e-3
K_OVERLAND=4.92; N_CHANNEL=0.04; R_CHANNEL=0.7
LCK=DATA/"lcmap_kk"; LCK.mkdir(exist_ok=True)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
CELL_KM2=DEM_RES**2/1e6

# ── KK gauges (from user's runoff metadata) ──────────────────────────────────
y=pd.read_csv("/Users/jared/Downloads/04_annual_runoff_metrics_1985_2025.csv",dtype={"site_no":str})
y["site_no"]=y["site_no"].str.zfill(8)
KK=["040871472","040871473","040871475","040871476","040871478","040871488","04087159"]
meta=y[y.site_no.isin(KK)].groupby("site_no").agg(
    station_nm=("station_nm","first"),lat=("lat","first"),lon=("lon","first"),
    da=("drainage_area_sqmi","first")).reset_index()
for c in ["lat","lon","da"]: meta[c]=pd.to_numeric(meta[c])
meta["drain_area_km2"]=(meta.da*2.58999).round(1)
gmeta=gpd.GeoDataFrame(meta,geometry=[Point(x,yy) for x,yy in zip(meta.lon,meta.lat)],crs=4326)

# register in sensors_dv_gauges.gpkg + panel_sites.txt
sens=gpd.read_file(SENS/"sensors_dv_gauges.gpkg")
add=gmeta[~gmeta.site_no.isin(sens.site_no)][["site_no","station_nm","drain_area_km2","geometry"]].copy()
if len(add):
    sens=pd.concat([sens,add],ignore_index=True)
    gpd.GeoDataFrame(sens,crs=4326).to_file(SENS/"sensors_dv_gauges.gpkg",driver="GPKG")
sites=[s.strip() for s in (SENS/"panel_sites.txt").read_text().split()]
for s in KK:
    if s not in sites: sites.append(s)
(SENS/"panel_sites.txt").write_text("\n".join(sites)+"\n")
log(f"registered {len(KK)} KK gauges")

# ── small KK DEM + conditioning + routing ────────────────────────────────────
bb=box(-88.03,42.92,-87.85,43.03)                 # covers KK basin (~61 km²) + buffer
dem_p=DATA/"kk_dem_10m.tif"; cond_p=DATA/"kk_dem_10m_cond.tif"
if not dem_p.exists():
    log("fetching KK DEM…")
    py3dep.get_dem(bb,resolution=DEM_RES).rio.reproject(PROJ,resolution=DEM_RES,
        resampling=rasterio.enums.Resampling.bilinear).squeeze().rio.to_raster(dem_p)
if not cond_p.exists():
    wbt=whitebox.WhiteboxTools(); wbt.verbose=False
    wbt.breach_depressions_least_cost(os.path.abspath(str(dem_p)),os.path.abspath(str(cond_p)),dist=2000,fill=True)
grid=Grid.from_raster(str(cond_p)); demc=grid.read_raster(str(cond_p)); aff=grid.affine
infl=grid.resolve_flats(demc); fdir=grid.flowdir(infl)
acc_arr=np.array(grid.accumulation(fdir),float); stream=acc_arr>ACC_THRESH; H,W=stream.shape
L=grid.cell_distances(fdir); L_arr=np.array(L,float)
S=np.array(grid.cell_slopes(infl,fdir),float); S=np.where(np.isfinite(S)&(S>0),S,SLOPE_MIN)
V=np.where(stream,(1/N_CHANNEL)*(R_CHANNEL**(2/3))*np.sqrt(S),K_OVERLAND*np.sqrt(S))
tau=L_arr/V; tau=np.where(np.isfinite(tau)&(tau>0),tau,np.nan)
tau_rast=PyRaster(tau,viewfinder=L.viewfinder)
log(f"KK grid {H}x{W}, max acc {acc_arr.max()*CELL_KM2:.0f} km²")

def snap_area(gx,gy,exp):
    cc0,rr0=grid.nearest_cell(gx,gy); best=None
    for R in (30,60,120,220):
        r0,r1=max(0,rr0-R),min(H,rr0+R+1); c0,c1=max(0,cc0-R),min(W,cc0+R+1)
        rs,cs=np.where(stream[r0:r1,c0:c1])
        if len(rs)==0: continue
        gr,gc=r0+rs,c0+cs; a=acc_arr[gr,gc]*CELL_KM2
        if exp and np.isfinite(exp):
            rel=np.abs(a-exp)/exp; k=int(np.argmin(rel))
            if best is None or rel[k]<best[0]: best=(rel[k],int(gr[k]),int(gc[k]))
            if rel[k]<0.1: break
        else: k=int(np.argmax(a)); best=(0,int(gr[k]),int(gc[k])); break
    return (best[1],best[2]) if best else (rr0,cc0)

# ── KK LCMAP per year (small bbox) ───────────────────────────────────────────
cat=pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                              modifier=planetary_computer.sign_inplace)
bb_alb=gpd.GeoSeries([bb],crs=4326).to_crs(ALBERS).total_bounds
def kk_lcpri(yr):
    f=LCK/f"lcpri_{yr}.tif"
    if f.exists(): return rxr.open_rasterio(f,masked=False).squeeze()
    items=list(cat.search(collections=["usgs-lcmap-conus-v13"],bbox=[-88.03,42.92,-87.85,43.03],
               datetime=f"{yr}-01-01/{yr}-12-31").items())
    arrs=[rxr.open_rasterio(it.assets["lcpri"].href,masked=False).squeeze().rio.clip_box(*bb_alb) for it in items]
    mos=(merge_arrays(arrs) if len(arrs)>1 else arrs[0]).rio.clip_box(*bb_alb)
    mos.rio.to_raster(f,compress="lzw"); return mos
YEARS=sorted(int(p.stem.split("_")[1]) for p in (DATA/"lcmap").glob("lcpri_*.tif"))
ref_lc=kk_lcpri(YEARS[-1]); lc_tr=ref_lc.rio.transform(); lc_shape=ref_lc.shape[-2:]
log(f"KK LCMAP ready, {len(YEARS)} years")

Wrows=[]; LCrows=[]
for _,m in meta.iterrows():
    sid=m.site_no; g=gpd.GeoSeries([Point(m.lon,m.lat)],crs=4326).to_crs(PROJ).iloc[0]
    rr,cc=snap_area(g.x,g.y,m.drain_area_km2)
    catch=np.array(grid.catchment(cc,rr,fdir,xytype="index",dirmap=(64,128,1,2,4,8,16,32)),bool)
    if catch.sum()<50: log(f"  {sid}: tiny catchment — skip"); continue
    # export catchment polygon
    ys,xs=np.where(catch); r0,r1,c0,c1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
    wtr=aff*Affine.translation(c0,r0); cu8=catch[r0:r1,c0:c1].astype(np.uint8)
    polys=[shape(gj) for gj,v in shapes(cu8,mask=cu8.astype(bool),transform=wtr) if v==1]
    gpd.GeoDataFrame({"site_no":[sid]},geometry=[unary_union(polys)],crs=PROJ).to_file(
        SENS/f"catchment_{sid}.gpkg",driver="GPKG")
    # travel time on KK grid, reproject to KK LCMAP grid
    T=np.array(grid.distance_to_outlet(cc,rr,fdir,weights=tau_rast,nodata_out=np.nan,
               routing="d8",xytype="index"),float)/3600.0
    Tp=DATA/f"T_{sid}.tif"
    with rasterio.open(Tp,"w",driver="GTiff",dtype="float32",width=W,height=H,count=1,
        crs=rasterio.crs.CRS.from_string(PROJ),transform=aff,nodata=np.nan,compress="lzw") as d:
        d.write(T.astype("float32")[None])
    T30=np.array(rxr.open_rasterio(Tp,masked=True).squeeze().rio.reproject_match(
        ref_lc,resampling=rasterio.enums.Resampling.bilinear),float)
    cpoly=gpd.read_file(SENS/f"catchment_{sid}.gpkg").to_crs(ALBERS).geometry.iloc[0]
    cm=rasterize([(cpoly,1)],out_shape=lc_shape,transform=lc_tr,dtype="uint8").astype(bool)
    valid=cm&np.isfinite(T30)&(T30>0); Tc=T30[valid]
    if valid.sum()<5: log(f"  {sid}: no LCMAP coverage — skip"); continue
    tau_m=float(np.median(Tc)); t33=float(np.percentile(Tc,33))
    wexp=float(np.exp(-Tc/tau_m).sum()); winv=float((1/np.maximum(Tc,1e-3)).sum())
    log(f"  {sid}: catch {catch.sum()*CELL_KM2:.1f} km² (exp {m.drain_area_km2})")
    for yr in YEARS:
        lc=np.array(kk_lcpri(yr)); wet=(lc==6)&valid; dev=int(((lc==1)&valid).sum()); nc=int(wet.sum())
        th=T30[wet]
        Wrows.append(dict(site_no=sid,year=yr,wet_km2=round(nc*900/1e6,3),
            wet_frac=round(nc/valid.sum(),5),
            Wfrac_exp=round(float(np.exp(-th/tau_m).sum()/wexp) if wexp>0 and nc else 0.0,5),
            Wfrac_inv=round(float((1/np.maximum(th,1e-3)).sum()/winv) if winv>0 and nc else 0.0,5),
            near_frac=round(float((th<=t33).mean()) if nc else 0.0,4),
            meanT_wet_hr=round(float(th.mean()),1) if nc else None, tau_hr=round(tau_m,1)))
        LCrows.append(dict(site_no=sid,year=yr,wetland_km2=round(nc*900/1e6,3),
            wetland_frac=round(nc/valid.sum(),5),developed_frac=round(dev/valid.sum(),5),
            catch_cells=int(valid.sum())))

# ── append to panels (replace any existing KK rows) ──────────────────────────
def append(csv, rows, keys=["site_no","year"]):
    new=pd.DataFrame(rows); old=pd.read_csv(csv,dtype={"site_no":str}); old["site_no"]=old["site_no"].str.zfill(8)
    old=old[~old["site_no"].isin(new["site_no"].unique())]
    pd.concat([old,new],ignore_index=True).sort_values(keys).to_csv(csv,index=False)
append(OUT/"panel_W.csv",Wrows); append(OUT/"panel_landcover_lcmap.csv",LCrows)
log(f"appended {len(set(r['site_no'] for r in Wrows))} KK gauges to panel_W + panel_landcover")
