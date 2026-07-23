"""Per-sensor wetland→stream pipeline over the globalwatershed AOI.

Master grid (whole AOI polygon) is computed ONCE and cached:
  DEM → WhiteBox breach conditioning → D8 flowdir/accumulation → stream network
  → HAND index (receiving stream cell j for every cell) → NLCD wetland mask.
Then for each sensor: snap to network, delineate upstream catchment, clip streams
and wetlands, and visualise wetland→stream flow direction.

Baseline NLCD year is configurable (BASE_YEAR); the design loops trivially over
1985–2024 later (routing is fixed; only the wetland mask changes per year).
"""
import os, sys, warnings, time
from pathlib import Path
import numpy as np
if not hasattr(np, "in1d"): np.in1d = np.isin
import pandas as pd
import geopandas as gpd
import rasterio, rasterio.enums, rasterio.crs
import rioxarray as rxr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.patches as mpatches
from matplotlib.colors import LightSource, ListedColormap
CMAP_STREAM = ListedColormap(["#1565C0"])   # solid blue for streams
CMAP_WET    = ListedColormap(["#2E7D32"])   # solid green for wetlands
from shapely.geometry import Point, shape
from shapely.ops import unary_union
import rasterio.features
from rasterio.transform import Affine
from pysheds.grid import Grid
from pysheds.sview import Raster as PyRaster
import pygeohydro as gh
from pynhd import NLDI
import py3dep, whitebox
warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────────
ROOT       = Path("/Users/jared/Wetland")
AOI_SHP    = "/Users/jared/Downloads/globalwatershed_zipped_shapefile/globalwatershed.shp"
DATA       = ROOT/"data";  OUT = ROOT/"outputs"/"sensors";  SENS = ROOT/"sensors"
DATA.mkdir(exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
PROJ_CRS   = "EPSG:32616"   # UTM 16N
DEM_RES    = 10
ACC_THRESH = 5000           # stream definition (cells)
BASE_YEAR  = 2021           # baseline NLCD year (any; 1985–2024 loop comes later)
LIMIT      = None           # set to an int to process only the first N sensors (None = all)
SKIP_FIG   = bool(int(os.environ.get("SKIP_FIG","0")))  # batch: skip per-gauge figs + NLDI
# 9 long-record gauges covering ~1985–2024 (user-selected scope)
SENSOR_IDS = ["04087120","04087030","04087050","04087070","04087088","04087119",
              "04086600","04086500","04087000"]
_PANEL_SF = Path("/Users/jared/Wetland/sensors/panel_sites.txt")
if _PANEL_SF.exists(): SENSOR_IDS = _PANEL_SF.read_text().split()

def hillshade(z, res=10):
    z = np.array(z, float); zf = np.where(np.isfinite(z), z, np.nanmin(z))
    return LightSource(315, 45).hillshade(zf, vert_exag=2.0, dx=res, dy=res)

# D8 (pysheds default) → (drow, dcol) in image space
D8 = {64:(-1,0),128:(-1,1),1:(0,1),2:(1,1),4:(1,0),8:(1,-1),16:(0,-1),32:(-1,-1)}

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# ── 1. AOI polygon ───────────────────────────────────────────────────────────
aoi = gpd.read_file(AOI_SHP).to_crs(4326)
aoi_geom = aoi.geometry.union_all()
aoi_proj = aoi.to_crs(PROJ_CRS)
log(f"AOI area {aoi_proj.area.sum()/1e6:.0f} km²")

# ── 2. Master DEM (whole AOI) ────────────────────────────────────────────────
dem_path = DATA/"aoi_dem_10m.tif"
if not dem_path.exists():
    log("Fetching 3DEP DEM for AOI polygon…")
    dem_raw = py3dep.get_dem(aoi_geom, resolution=DEM_RES)
    dem_raw.rio.reproject(PROJ_CRS, resolution=DEM_RES,
        resampling=rasterio.enums.Resampling.bilinear).squeeze().rio.to_raster(dem_path)
    log("  cached AOI DEM.")
dem_da = rxr.open_rasterio(dem_path, masked=True).squeeze()
dem_arr0 = np.array(dem_da, float)
log(f"DEM shape {dem_arr0.shape}")

# ── 3. WhiteBox conditioning ─────────────────────────────────────────────────
cond_path = DATA/"aoi_dem_10m_cond.tif"
if not cond_path.exists():
    log("WhiteBox BreachDepressionsLeastCost…")
    wbt = whitebox.WhiteboxTools(); wbt.verbose = False
    wbt.breach_depressions_least_cost(os.path.abspath(str(dem_path)),
        os.path.abspath(str(cond_path)), dist=2000, fill=True)
    log("  cached conditioned DEM.")

# ── 4. Flow routing (cache the arrays) ───────────────────────────────────────
grid = Grid.from_raster(str(cond_path))
demc = grid.read_raster(str(cond_path))
aff  = grid.affine
# fdir is always recomputed (kept as a proper pysheds Raster for snap/catchment);
# the expensive accumulation + HAND index are cached.
log("resolve_flats + D8 flow direction…")
infl = grid.resolve_flats(demc)
fdir = grid.flowdir(infl)
if (DATA/"aoi_acc.npy").exists() and (DATA/"aoi_hand_idx.npy").exists():
    acc_arr = np.load(DATA/"aoi_acc.npy").astype(float)
    hand_idx_arr = np.load(DATA/"aoi_hand_idx.npy")
    log("loaded acc + HAND index from cache.")
else:
    log("accumulation + HAND index (first run, ~minutes)…")
    acc = grid.accumulation(fdir); acc_arr = np.array(acc, float)
    hand_idx = grid.compute_hand(fdir, infl, acc > ACC_THRESH, return_index=True)
    hand_idx_arr = np.array(hand_idx, dtype=np.int64)
    np.save(DATA/"aoi_acc.npy", acc_arr); np.save(DATA/"aoi_hand_idx.npy", hand_idx_arr)
    log(f"  max acc {acc_arr.max()*DEM_RES**2/1e6:.0f} km²")
stream_arr = acc_arr > ACC_THRESH
H, W = stream_arr.shape
stream_raster = PyRaster(stream_arr, viewfinder=fdir.viewfinder)  # for snap_to_mask
CELL_KM2 = DEM_RES**2/1e6
log(f"stream cells {int(stream_arr.sum()):,}")

def snap_area(gx, gy, expected_km2):
    """Snap a gauge to the stream cell whose accumulation best matches the NWIS
    reported drainage area (robust against grabbing a nearby small tributary or
    the wrong main stem). Falls back to max-accumulation if area is unknown."""
    cc0, rr0 = grid.nearest_cell(gx, gy)
    best = None  # (relerr, r, c, acc_km2)
    for R in (30, 60, 120, 220):
        r0,r1 = max(0,rr0-R),min(H,rr0+R+1); c0,c1 = max(0,cc0-R),min(W,cc0+R+1)
        rs,cs = np.where(stream_arr[r0:r1, c0:c1])
        if len(rs)==0: continue
        gr,gc = r0+rs, c0+cs
        a = acc_arr[gr,gc]*CELL_KM2
        if expected_km2 and np.isfinite(expected_km2):
            rel = np.abs(a-expected_km2)/expected_km2
            k = int(np.argmin(rel))
            if best is None or rel[k] < best[0]:
                best = (rel[k], int(gr[k]), int(gc[k]), float(a[k]))
            if rel[k] < 0.10:    # excellent match — stop expanding
                break
        else:
            k = int(np.argmax(a))
            best = (0.0, int(gr[k]), int(gc[k]), float(a[k])); break
    if best is None:
        return rr0, cc0, float(acc_arr[rr0,cc0]*CELL_KM2), 0.0, np.nan
    relerr, r, c, a = best
    snapdist = float(np.hypot(r-rr0, c-cc0)*DEM_RES)
    return r, c, a, snapdist, relerr

# ── 5. NLCD wetland mask (baseline year) ─────────────────────────────────────
wet_path = DATA/f"aoi_nlcd_wetland_{BASE_YEAR}.tif"
if not wet_path.exists():
    log(f"Fetching NLCD {BASE_YEAR}…")
    nlcd = gh.nlcd_bygeom(aoi, resolution=30, years={"cover":[BASE_YEAR]})
    cov = nlcd[list(nlcd.keys())[0]][f"cover_{BASE_YEAR}"]
    arr = np.array(cov).squeeze().astype(float)
    wet = np.where(np.isin(arr,[90,95]),1.0,0.0)
    ref = rxr.open_rasterio(dem_path, masked=True)
    cov.copy(data=wet.reshape(cov.shape)).rio.reproject_match(
        ref, resampling=rasterio.enums.Resampling.nearest).squeeze().rio.to_raster(wet_path)
    log("  cached wetland mask.")
# NOTE: read as float and compare ==1.0 — do NOT cast straight to bool, because
# NaN nodata (outside the AOI) would cast to True and flood the mask.
_wraw = np.array(rxr.open_rasterio(wet_path, masked=True).squeeze(), float)
wetland_arr = np.nan_to_num(_wraw, nan=0.0) == 1.0
log(f"AOI wetland cells {int(wetland_arr.sum()):,} = {wetland_arr.sum()*DEM_RES**2/1e6:.1f} km²")

# ── 6. Sensors ───────────────────────────────────────────────────────────────
sensors = gpd.read_file(SENS/"sensors_dv_gauges.gpkg")
sensors = sensors[sensors["site_no"].isin(SENSOR_IDS)].copy()
# order by the SENSOR_IDS list (smallest/example first)
sensors["__o"] = sensors["site_no"].map({s:i for i,s in enumerate(SENSOR_IDS)})
sensors = sensors.sort_values("__o").to_crs(PROJ_CRS).reset_index(drop=True)
if LIMIT: sensors = sensors.iloc[:LIMIT]
log(f"Processing {len(sensors)} sensors")

def rc_to_xy(r, c):
    return aff.c + (c+0.5)*aff.a, aff.f + (r+0.5)*aff.e

rows = []; overview = []
for _, srow in sensors.iterrows():
    sid = srow["site_no"]; nm = srow["station_nm"]
    gx, gy = srow.geometry.x, srow.geometry.y
    exp_km2 = srow.get("drain_area_km2", np.nan)
    # snap to stream by matching NWIS drainage area
    rr, cc, acc_km2, snapdist, relerr = snap_area(gx, gy, exp_km2)  # rr=row, cc=col
    if relerr is not None and np.isfinite(relerr) and relerr > 0.3:
        log(f"  {sid}: WARN snap area off by {relerr*100:.0f}% "
            f"(got {acc_km2:.0f}, expected {exp_km2:.0f} km²)")
    # upstream catchment — use xytype='index' with (col,row); coordinate round-trip
    # mis-rounds by one cell and can trace from an off-channel cell (→ empty catchment).
    catch = grid.catchment(cc, rr, fdir, xytype="index", dirmap=(64,128,1,2,4,8,16,32))
    catch_arr = np.array(catch, bool)
    n_catch = int(catch_arr.sum())
    if n_catch < 50:
        log(f"  {sid}: tiny catchment ({n_catch} cells) — skipping"); continue
    # clip streams & wetlands to catchment
    cat_stream = stream_arr & catch_arr
    cat_wet    = wetland_arr & catch_arr
    n_wet = int(cat_wet.sum())
    # bbox of catchment for plotting
    ys_idx, xs_idx = np.where(catch_arr)
    r0,r1 = ys_idx.min(), ys_idx.max()+1; c0,c1 = xs_idx.min(), xs_idx.max()+1
    pad = max(5,(r1-r0)//40)
    r0,r1 = max(0,r0-pad),min(H,r1+pad); c0,c1 = max(0,c0-pad),min(W,c1+pad)

    # NLDI upstream basin polygon ("upstream shapefile")
    basin_gpkg = SENS/f"basin_{sid}.gpkg"
    basin_proj = None
    try:
        if basin_gpkg.exists():
            basin_proj = gpd.read_file(basin_gpkg).to_crs(PROJ_CRS)
        elif not SKIP_FIG:
            b = NLDI().get_basins(sid, fsource="nwissite")
            b.to_file(basin_gpkg, driver="GPKG"); basin_proj = b.to_crs(PROJ_CRS)
    except Exception as e:
        log(f"  {sid}: NLDI basin failed ({e})")

    area_km2 = n_catch*DEM_RES**2/1e6

    # export the upstream catchment as a polygon (GIS-usable "upstream shapefile")
    win_tr = aff * Affine.translation(c0, r0)
    csub_u8 = catch_arr[r0:r1, c0:c1].astype(np.uint8)
    polys = [shape(g) for g,v in rasterio.features.shapes(csub_u8, mask=csub_u8.astype(bool),
                                                          transform=win_tr) if v==1]
    catch_poly = unary_union(polys) if polys else None
    if catch_poly is not None:
        gpd.GeoDataFrame({"site_no":[sid],"station_nm":[nm],"catch_km2":[round(area_km2,1)],
                          "wetland_km2":[round(n_wet*DEM_RES**2/1e6,2)]},
                         geometry=[catch_poly], crs=PROJ_CRS).to_file(
                         SENS/f"catchment_{sid}.gpkg", driver="GPKG")
    overview.append((sid, catch_poly, rc_to_xy(rr,cc)))

    rows.append(dict(site_no=sid, station_nm=nm, catch_km2=round(area_km2,1),
                     nwis_area_km2=round(float(exp_km2),1) if np.isfinite(exp_km2) else None,
                     area_err_pct=round(100*(area_km2-exp_km2)/exp_km2,1) if np.isfinite(exp_km2) else None,
                     snap_dist_m=round(snapdist),
                     wetland_km2=round(n_wet*DEM_RES**2/1e6,2),
                     wetland_pct=round(100*n_wet/n_catch,2)))
    log(f"  {sid} {nm[:38]:38s} catch {area_km2:7.1f} km²  wet {100*n_wet/n_catch:4.1f}%")

    if SKIP_FIG: continue        # batch mode: catchment + summary only
    # ── Figure: 3 panels (overview + 2 zoomed) ───────────────────────────────
    sub  = lambda a: np.array(a)[r0:r1, c0:c1]
    hs   = hillshade(sub(dem_arr0), DEM_RES)
    extent=[c0,c1,r1,r0]
    Yc, Xc = np.mgrid[r0:r1, c0:c1]   # global pixel coords for contour

    # densest wetland window (~42% of catchment bbox) for the zoom panels
    wr_all, wc_all = np.where(cat_wet)
    if len(wr_all):
        nb=6; rb=np.linspace(r0,r1,nb+1); cb=np.linspace(c0,c1,nb+1)
        Hh,_,_=np.histogram2d(wr_all,wc_all,bins=[rb,cb])
        bi,bj=np.unravel_index(np.argmax(Hh),Hh.shape)
        rc=int((rb[bi]+rb[bi+1])/2); ccz=int((cb[bj]+cb[bj+1])/2)
    else:
        rc=(r0+r1)//2; ccz=(c0+c1)//2
    zh=max(60,int((r1-r0)*0.42)); zw=max(60,int((c1-c0)*0.42))
    zr0=max(r0,rc-zh//2); zr1=min(r1,zr0+zh); zr0=max(r0,zr1-zh)
    zc0=max(c0,ccz-zw//2); zc1=min(c1,zc0+zw); zc0=max(c0,zc1-zw)
    subz=lambda a: np.array(a)[zr0:zr1, zc0:zc1]
    hsz=hillshade(subz(dem_arr0),DEM_RES); extz=[zc0,zc1,zr1,zr0]

    fig, ax = plt.subplots(1, 3, figsize=(21, 7.6))

    # Panel A — upstream watershed overview: boundary + streams + wetlands + outlet
    ax[0].imshow(hs, cmap="gray", extent=extent, origin="upper")
    ax[0].imshow(np.where(sub(catch_arr),0,np.nan), cmap=ListedColormap(["#4F83CC"]),
                 alpha=0.18, extent=extent, origin="upper", vmin=0, vmax=1)
    ax[0].contour(Xc, Yc, sub(catch_arr).astype(float), levels=[0.5],
                  colors="navy", linewidths=1.8)
    ax[0].imshow(np.where(sub(cat_stream),0,np.nan), cmap=CMAP_STREAM, extent=extent, origin="upper", vmin=0, vmax=1)
    ax[0].imshow(np.where(sub(cat_wet),0,np.nan), cmap=CMAP_WET, extent=extent, origin="upper", vmin=0, vmax=1)
    ax[0].plot(cc, rr, "r^", ms=15, mec="k", zorder=6)
    ax[0].add_patch(mpatches.Rectangle((zc0,zr0), zc1-zc0, zr1-zr0, fill=False,
                    ec="red", lw=1.4, ls="--", zorder=7))
    ax[0].set_title(f"{sid} — {nm[:40]}\nupstream watershed {area_km2:.0f} km² · "
                    f"wetland {n_wet*DEM_RES**2/1e6:.1f} km² ({100*n_wet/n_catch:.1f}%)", fontsize=10)
    ax[0].set_xticks([]); ax[0].set_yticks([]); ax[0].set_xlim(c0,c1); ax[0].set_ylim(r1,r0)

    # Panel B — D8 flow field (zoom): arrows over streams + wetlands
    ax[1].imshow(hsz, cmap="gray", extent=extz, origin="upper")
    ax[1].imshow(np.where(subz(cat_stream),0,np.nan), cmap=CMAP_STREAM, extent=extz, origin="upper", vmin=0, vmax=1)
    ax[1].imshow(np.where(subz(cat_wet),0,np.nan), cmap=CMAP_WET, extent=extz, origin="upper", vmin=0, vmax=1)
    step=max(2,(zr1-zr0)//38)
    fz=subz(np.array(fdir)); cz=subz(catch_arr)
    U=np.full(fz.shape,np.nan); V=np.full(fz.shape,np.nan)
    for code,(dr,dc) in D8.items():
        m=(fz==code)&cz; U[m]=dc; V[m]=dr        # y-axis = row (inverted via ylim) → V=dr
    Ys,Xs=np.mgrid[zr0:zr1, zc0:zc1]
    ax[1].quiver(Xs[::step,::step],Ys[::step,::step],U[::step,::step],V[::step,::step],
                 color="#E65100", angles="xy", scale_units="xy", scale=1.0/(step*0.8),
                 width=0.004, headwidth=4, headlength=5, zorder=5)
    ax[1].set_title("D8 flow field (orange) · streams (blue) · wetlands (green)\n[zoom on wetland-rich area]", fontsize=10)
    ax[1].set_xticks([]); ax[1].set_yticks([]); ax[1].set_xlim(zc0,zc1); ax[1].set_ylim(zr1,zr0)

    # Panel C — wetland→stream connectivity (zoom): connectors to receiving cell j
    ax[2].imshow(hsz, cmap="gray", extent=extz, origin="upper")
    ax[2].imshow(np.where(subz(cat_stream),0,np.nan), cmap=CMAP_STREAM, extent=extz, origin="upper", vmin=0, vmax=1)
    wrz, wcz = np.where(subz(cat_wet))
    wrz_g, wcz_g = wrz+zr0, wcz+zc0
    if len(wrz_g):
        k=max(1,len(wrz_g)//1200)
        wrz_g, wcz_g = wrz_g[::k], wcz_g[::k]
        flat_j=hand_idx_arr[wrz_g,wcz_g]; ok=flat_j>=0
        jr,jc=np.unravel_index(flat_j[ok],(H,W))
        wrr,wcc=wrz_g[ok],wcz_g[ok]
        seglen=np.hypot(jr-wrr,jc-wcc)*DEM_RES/1000.0
        segs=[[(wcc[i],wrr[i]),(jc[i],jr[i])] for i in range(len(wrr))]
        lcoll=mc.LineCollection(segs,cmap="autumn_r",linewidths=0.6,alpha=0.7)
        lcoll.set_array(seglen); ax[2].add_collection(lcoll)
        ax[2].scatter(wcc,wrr,s=3,c="#2E7D32",alpha=0.6,zorder=4)
        cbar=plt.colorbar(lcoll,ax=ax[2],fraction=0.046); cbar.set_label("wetland→stream distance (km)")
    ax[2].set_title(f"Wetland → stream connectivity\n[{n_wet:,} wetland cells in catchment]", fontsize=10)
    ax[2].set_xticks([]); ax[2].set_yticks([]); ax[2].set_xlim(zc0,zc1); ax[2].set_ylim(zr1,zr0)

    fig.tight_layout()
    fig.savefig(OUT/f"sensor_{sid}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

# ── AOI overview figure: all sensors + their nested catchments ───────────────
fig, ax = plt.subplots(figsize=(11, 13))
aoi_proj.boundary.plot(ax=ax, color="black", lw=2.2, zorder=2, label="AOI (globalwatershed)")
# light major-stream backdrop, clipped to the AOI polygon (drops nodata-corner
# D8 artefacts outside the basin)
aoi_mask = rasterio.features.rasterize(
    [(g,1) for g in aoi_proj.geometry], out_shape=(H,W), transform=aff, dtype=np.uint8).astype(bool)
ys_s, xs_s = np.where(stream_arr & (acc_arr > ACC_THRESH*4) & aoi_mask)
ax.scatter(aff.c+(xs_s+0.5)*aff.a, aff.f+(ys_s+0.5)*aff.e, s=0.05, c="#90caf9", zorder=1)
cmap = plt.cm.tab10
for i,(sid,poly,(sx,sy)) in enumerate(overview):
    if poly is not None:
        gpd.GeoSeries([poly], crs=PROJ_CRS).boundary.plot(ax=ax, color=cmap(i%10), lw=1.4, zorder=3)
    ax.plot(sx, sy, "^", color=cmap(i%10), ms=11, mec="k", zorder=5)
    ax.annotate(sid, (sx,sy), fontsize=7, xytext=(4,4), textcoords="offset points", zorder=6)
ax.set_title(f"Sensors and upstream watersheds within AOI\n{len(overview)} long-record gauges · "
             f"NLCD {BASE_YEAR} wetlands", fontsize=12)
ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)"); ax.set_aspect("equal")
fig.tight_layout(); fig.savefig(OUT/"00_overview_all_sensors.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Summary table ────────────────────────────────────────────────────────────
summ = pd.DataFrame(rows)
summ.to_csv(OUT/"sensor_summary.csv", index=False)
log("Summary:")
print(summ.to_string(index=False))
log(f"Done. Figures + summary + catchment polygons in {OUT} / {SENS}")
