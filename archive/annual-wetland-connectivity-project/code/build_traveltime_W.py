"""THE PREVIOUS WORKFLOW, restored: travel-time-weighted wetland metric W.

For each sensor we compute the travel time T from every upstream cell to that
gauge (velocity grid: Manning channel + TR-55 overland; pysheds distance_to_outlet
weighted by per-cell time), then weight each wetland cell by a kernel f(T):
  W_invtime = Σ area / T        (near-stream wetlands count most)
  W_exp     = Σ area · e^(-T/τ) (τ = 24 h)
vs the unweighted wetland area. This is the connectivity-weighted metric the whole
travel-time pipeline was built to produce.

Wetland source = LCMAP (per-year, 30 m Albers, cached). T is computed on the DEM
grid then reprojected to the LCMAP grid so W is consistent with the wetland data.
"""
import os, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd
import rioxarray as rxr, rasterio
from rasterio.features import rasterize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
if not hasattr(np,"in1d"): np.in1d=np.isin
from pysheds.grid import Grid
from pysheds.sview import Raster as PyRaster
warnings.filterwarnings("ignore")

ROOT=Path("/Users/jared/Wetland"); DATA=ROOT/"data"; SENS=ROOT/"sensors"; OUT=ROOT/"outputs"/"sensors"
LC=DATA/"lcmap"; ALBERS="EPSG:5070"; PROJ="EPSG:32616"
DEM_RES=10; ACC_THRESH=5000; SLOPE_MIN=1e-3
K_OVERLAND=4.92; N_CHANNEL=0.04; R_CHANNEL=0.7
TAU_HR=24.0
SENSOR_IDS=["04087000","04086600","04086500","04087120","04087030",
            "04087050","04087070","04087088","04087119"]
_PANEL_SF = Path("/Users/jared/Wetland/sensors/panel_sites.txt")
if _PANEL_SF.exists(): SENSOR_IDS = _PANEL_SF.read_text().split()
LIMIT=int(os.environ.get("LIMIT","0")) or None
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# ── master grid + velocity/tau (AOI-wide, once) ──────────────────────────────
grid=Grid.from_raster(str(DATA/"aoi_dem_10m_cond.tif"))
demc=grid.read_raster(str(DATA/"aoi_dem_10m_cond.tif")); aff=grid.affine
log("resolve_flats + flowdir…"); infl=grid.resolve_flats(demc); fdir=grid.flowdir(infl)
acc_arr=np.load(DATA/"aoi_acc.npy").astype(float); stream=acc_arr>ACC_THRESH; H,W=stream.shape
stream_rast=PyRaster(stream,viewfinder=fdir.viewfinder)
L=grid.cell_distances(fdir); L_arr=np.array(L,float)
S=np.array(grid.cell_slopes(infl,fdir),float); S=np.where(np.isfinite(S)&(S>0),S,SLOPE_MIN)
V=np.where(stream,(1/N_CHANNEL)*(R_CHANNEL**(2/3))*np.sqrt(S),K_OVERLAND*np.sqrt(S))
tau=L_arr/V; tau=np.where(np.isfinite(tau)&(tau>0),tau,np.nan)
tau_rast=PyRaster(tau,viewfinder=L.viewfinder)
log(f"tau ready: {np.nanmin(tau):.1f}-{np.nanmax(tau):.0f} s/cell")
CELL_KM2=DEM_RES**2/1e6

def snap_area(gx,gy,exp_km2):
    cc0,rr0=grid.nearest_cell(gx,gy); best=None
    for R in (30,60,120,220):
        r0,r1=max(0,rr0-R),min(H,rr0+R+1); c0,c1=max(0,cc0-R),min(W,cc0+R+1)
        rs,cs=np.where(stream[r0:r1,c0:c1])
        if len(rs)==0: continue
        gr,gc=r0+rs,c0+cs; a=acc_arr[gr,gc]*CELL_KM2
        if exp_km2 and np.isfinite(exp_km2):
            rel=np.abs(a-exp_km2)/exp_km2; k=int(np.argmin(rel))
            if best is None or rel[k]<best[0]: best=(rel[k],int(gr[k]),int(gc[k]))
            if rel[k]<0.1: break
        else:
            k=int(np.argmax(a)); best=(0,int(gr[k]),int(gc[k])); break
    return best[1],best[2]

sens=gpd.read_file(SENS/"sensors_dv_gauges.gpkg").set_index("site_no")
ids=SENSOR_IDS[:LIMIT] if LIMIT else SENSOR_IDS

# reference LCMAP grid (for reprojecting T)
ref_lc=rxr.open_rasterio(sorted(LC.glob("lcpri_*.tif"))[0], masked=False).squeeze()
lc_tr=ref_lc.rio.transform(); lc_shape=ref_lc.shape[-2:]
YEARS=sorted(int(p.stem.split("_")[1]) for p in LC.glob("lcpri_*.tif"))

rows=[]
for sid in ids:
    srow=sens.loc[sid]; g=gpd.GeoSeries([srow.geometry],crs=4326).to_crs(PROJ).iloc[0]
    gx,gy=g.x,g.y; exp=srow.get("drain_area_km2",np.nan)
    rr,cc=snap_area(gx,gy,exp)   # rr=row, cc=col
    # travel time from every cell to this outlet (seconds) along D8.
    # Use xytype='index' (col,row) — coordinate round-trip mis-rounds by one cell.
    T=grid.distance_to_outlet(cc,rr,fdir,weights=tau_rast,nodata_out=np.nan,
                              routing="d8",xytype="index")
    T_arr=np.array(T,float); T_hr=T_arr/3600.0
    nfin=int(np.isfinite(T_hr).sum())
    log(f"{sid}: T finite {nfin:,} cells = {nfin*CELL_KM2:.0f} km² (exp {exp:.0f})")
    # write T (hr) to a UTM raster, reproject to LCMAP 30m grid
    Tpath=DATA/f"T_{sid}.tif"
    prof=dict(driver="GTiff",dtype="float32",width=W,height=H,count=1,
              crs=rasterio.crs.CRS.from_string(PROJ),transform=aff,nodata=np.nan,compress="lzw")
    with rasterio.open(Tpath,"w",**prof) as dst: dst.write(T_hr.astype("float32")[None])
    T30=rxr.open_rasterio(Tpath,masked=True).squeeze().rio.reproject_match(
            ref_lc, resampling=rasterio.enums.Resampling.bilinear)
    T30=np.array(T30,float)
    # catchment mask on LCMAP grid
    catch_poly=gpd.read_file(SENS/f"catchment_{sid}.gpkg").to_crs(ALBERS).geometry.iloc[0]
    cm=rasterize([(catch_poly,1)],out_shape=lc_shape,transform=lc_tr,dtype="uint8").astype(bool)
    valid=cm&np.isfinite(T30)&(T30>0)
    Tc=T30[valid]                                   # all catchment cells' travel time (hr)
    # ADAPTIVE kernel: τ = catchment median travel time (absolute T is uncalibrated,
    # so a fixed τ=24h was degenerate). Connectivity-weighted wetland FRACTION is
    # dimensionless & cross-basin comparable; reduces to wetland_frac when f≡const.
    tau=float(np.median(Tc)); t33=float(np.percentile(Tc,33))
    w_all_exp=float(np.exp(-Tc/tau).sum()); w_all_inv=float((1/np.maximum(Tc,1e-3)).sum())
    Tc_for_fig=Tc
    for yr in YEARS:
        lc=np.array(rxr.open_rasterio(LC/f"lcpri_{yr}.tif",masked=False).squeeze())
        wet=(lc==6)&valid
        th=T30[wet]; ncell=int(wet.sum())
        rec=dict(site_no=sid,year=yr, wet_km2=round(ncell*900/1e6,3),
                 wet_frac=round(ncell/valid.sum(),5),
                 Wfrac_exp=round(float(np.exp(-th/tau).sum()/w_all_exp) if w_all_exp>0 and ncell else 0.0,5),
                 Wfrac_inv=round(float((1/np.maximum(th,1e-3)).sum()/w_all_inv) if w_all_inv>0 and ncell else 0.0,5),
                 near_frac=round(float((th<=t33).mean()) if ncell else 0.0,4),
                 meanT_wet_hr=round(float(th.mean()),1) if ncell else None,
                 tau_hr=round(tau,1))
        rows.append(rec)
    # ── signature figure: travel-time field to the gauge + wetlands (latest yr) ──
    yr=YEARS[-1]; lc=np.array(rxr.open_rasterio(LC/f"lcpri_{yr}.tif",masked=False).squeeze())
    rr_i,cc_i=np.where(valid); r0,r1,c0,c1=rr_i.min(),rr_i.max()+1,cc_i.min(),cc_i.max()+1
    Tsub=np.where(valid,T30,np.nan)[r0:r1,c0:c1]
    wetsub=((lc==6)&valid)[r0:r1,c0:c1]
    fig,ax=plt.subplots(figsize=(7,7))
    im=ax.imshow(Tsub,cmap="viridis",origin="upper",
                 vmax=np.nanpercentile(Tsub,98))
    plt.colorbar(im,ax=ax,label="travel time to gauge (hr)",fraction=0.046)
    ax.imshow(np.where(wetsub,1,np.nan),cmap="autumn",origin="upper",alpha=0.9)
    ax.set_title(f"{sid}: travel-time field + wetlands (orange)\n"
                 f"wet {rec['wet_km2']:.0f} km² · W_exp_frac {rec['Wfrac_exp']:.3f} vs area_frac {rec['wet_frac']:.3f}",
                 fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(OUT/f"W_{sid}.png",dpi=130,bbox_inches="tight"); plt.close(fig)
df=pd.DataFrame(rows).sort_values(["site_no","year"])
df.to_csv(OUT/"panel_W.csv",index=False)
log(f"Wrote {OUT/'panel_W.csv'} ({len(df)} rows)")
# cross-sectional snapshot (latest common year)
yr=df.year.max(); snap=df[df.year==yr]
print(f"\nCross-sectional connectivity-weighted wetland ({yr}):")
print("  wet_frac = plain wetland fraction;  Wfrac_exp/inv = travel-time-weighted;")
print("  near_frac = share of wetland in nearest travel-time tertile")
print(snap[["site_no","wet_km2","wet_frac","Wfrac_exp","Wfrac_inv","near_frac","meanT_wet_hr","tau_hr"]].to_string(index=False))
