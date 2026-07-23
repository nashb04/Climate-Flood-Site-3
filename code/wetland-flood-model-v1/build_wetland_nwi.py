#!/usr/bin/env python
"""
Mark's Model — Step 3: NWI wetland layer -> cumulative effectiveness W_j and storage S_w,j.

For each of the 9 catchments:
  - fetch USFWS National Wetlands Inventory polygons (Cowardin ATTRIBUTE) by bbox from
    the FWS ArcGIS service, clip to the catchment (cached sensors/nwi_{site}.gpkg);
  - per wetland i:  A_i (m2)            polygon area
                    T_ij (hr)           sampled from data/T_{site}.tif (travel time to gauge)
                    dist_chan (m)        distance to channel (acc>5000) -> C_i
                    M_i                  Cowardin type modifier (attenuation potential)
                    V_i (m3)             Volume-Area power law  V = c * A^beta
  - assemble  W_j = sum_i A_i * C_i * M_i * f(T_ij),  f = exp(-T/tau) and harmonic 1/(1+T/tau)
    with tau = catchment-median travel time (absolute T is uncalibrated; see Step-1 notes);
  - storage  S_w,j = sum_i V_i   (Volume-Area), cross-checked with DEM depression volume
    sum of max(cond_dem - raw_dem, 0) over wetland footprints.

Outputs: sensors/nwi_{site}.gpkg, outputs/panel_W_Sw.csv, outputs/step3_W_qa.png
"""
from __future__ import annotations
import os, sys, time, warnings
import numpy as np, pandas as pd
import geopandas as gpd
import requests, rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_M = os.path.join(HERE, "data"); OUT = os.path.join(HERE, "outputs")
ROOT = "/Users/jared/Wetland"; DATA = os.path.join(ROOT, "data"); SENS = os.path.join(ROOT, "sensors")
NWI_DIR = os.path.join(HERE, "nwi"); os.makedirs(NWI_DIR, exist_ok=True)

SITES = ["04087000", "04086500", "04086600", "04087120", "04087030",
         "04087050", "04087070", "04087088", "04087119"]
NWI_URL = ("https://fwspublicservices.wim.usgs.gov/wetlandsmapservice/rest/services/"
           "Wetlands/MapServer/0/query")
ACC_THRESH = 5000          # channel = flow accumulation > 0.5 km2 (matches routing)
NEAR_M = 100.0             # connectivity buffer (workflow default <100 m)
SW_C, SW_BETA = 0.05, 1.2  # Volume-Area: V[m3] = c * A[m2]^beta (literature range b=1.1-1.3)
PROJ = "EPSG:32616"

# Cowardin-type attenuation modifier M_i (floodplain/emergent attenuate peaks more than
# open-water or lacustrine; documented + tunable; set all=1 to disable).
def modifier(attribute: str) -> float:
    if not isinstance(attribute, str) or not attribute:
        return 0.7
    a = attribute.upper()
    sysL = a[0]                                   # P/R/L/E/M
    cls = a[2:4] if len(a) >= 4 else a[1:3]
    if sysL == "L":      return 0.3               # lacustrine (lakes)
    if sysL == "R":      return 0.5               # riverine channel
    if "EM" in a:        return 1.0               # emergent marsh — high attenuation
    if "FO" in a or "SS" in a: return 0.8         # forested / scrub-shrub
    if "UB" in a or "AB" in a or "OW" in a: return 0.5  # open water / aquatic bed
    return 0.7

# ---------------------------------------------------------------- NWI fetch
def _query_tile(env, off):
    params = dict(geometry=env, geometryType="esriGeometryEnvelope", inSR=4326,
                  spatialRel="esriSpatialRelIntersects", where="1=1",
                  outFields="Wetlands.OBJECTID,Wetlands.ATTRIBUTE,Wetlands.WETLAND_TYPE,Wetlands.ACRES",
                  returnGeometry="true", outSR=32616, f="geojson",
                  resultOffset=off, resultRecordCount=2000)
    for attempt in range(3):
        try:
            return requests.get(NWI_URL, params=params, timeout=90).json().get("features", [])
        except Exception:
            time.sleep(2)
    return []

def fetch_nwi(site: str, tile_deg=0.08) -> gpd.GeoDataFrame:
    cache = os.path.join(NWI_DIR, f"nwi_{site}.gpkg")
    if os.path.exists(cache):
        return gpd.read_file(cache)
    catch4326 = gpd.read_file(os.path.join(SENS, f"catchment_{site}.gpkg")).to_crs(4326)
    poly = catch4326.geometry.union_all()
    minx, miny, maxx, maxy = catch4326.total_bounds
    from shapely.geometry import box
    xs = np.arange(minx, maxx, tile_deg); ys = np.arange(miny, maxy, tile_deg)
    feats, ntiles = [], 0
    for x in xs:
        for y in ys:
            tile = box(x, y, min(x + tile_deg, maxx), min(y + tile_deg, maxy))
            if not tile.intersects(poly):          # skip tiles outside the catchment
                continue
            ntiles += 1
            env = f"{tile.bounds[0]},{tile.bounds[1]},{tile.bounds[2]},{tile.bounds[3]}"
            off = 0
            while True:
                fs = _query_tile(env, off)
                feats.extend(fs)
                if len(fs) < 2000:
                    break
                off += 2000
    print(f"    [{site}] fetched {len(feats)} features over {ntiles} tiles", flush=True)
    if not feats:
        gdf = gpd.GeoDataFrame({"ATTRIBUTE": [], "WETLAND_TYPE": [], "ACRES": []},
                               geometry=[], crs=32616)
    else:
        gdf = gpd.GeoDataFrame.from_features(feats, crs=32616)
        gdf = gdf.rename(columns={c: c.split(".")[-1] for c in gdf.columns if c != "geometry"})
        if "OBJECTID" in gdf.columns:                # drop dups from overlapping tiles
            gdf = gdf.drop_duplicates(subset="OBJECTID")
        # clip to the actual catchment polygon
        cp = gpd.read_file(os.path.join(SENS, f"catchment_{site}.gpkg")).to_crs(32616)
        gdf = gpd.clip(gdf, cp)
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    # normalise column names (service returns qualified "Wetlands.ATTRIBUTE")
    gdf = gdf.rename(columns={c: c.split(".")[-1] for c in gdf.columns if c != "geometry"})
    for need in ("ATTRIBUTE", "WETLAND_TYPE"):
        if need not in gdf.columns:
            gdf[need] = None
    if len(gdf):
        gdf.to_file(cache, driver="GPKG")
    return gdf

# ---------------------------------------------------------------- raster helpers
_dist_cache = {}
def aoi_layers():
    """Load master-grid arrays once: transform, distance-to-channel (m), DEM fill depth (m)."""
    if _dist_cache:
        return _dist_cache
    with rasterio.open(os.path.join(DATA, "T_04087000.tif")) as r:
        transform, shape = r.transform, (r.height, r.width)
    acc = np.load(os.path.join(DATA, "aoi_acc.npy"))
    assert acc.shape == shape, f"acc {acc.shape} != T grid {shape}"
    stream = acc > ACC_THRESH
    dist = distance_transform_edt(~stream).astype("float32") * 10.0      # metres
    with rasterio.open(os.path.join(DATA, "aoi_dem_10m.tif")) as r:
        raw = r.read(1).astype("float32")
    with rasterio.open(os.path.join(DATA, "aoi_dem_10m_cond.tif")) as r:
        cond = r.read(1).astype("float32")
    fill_depth = np.clip(cond - raw, 0, None)                            # filled depression depth (m)
    _dist_cache.update(transform=transform, shape=shape, dist=dist, fill=fill_depth)
    return _dist_cache

def rc_from_xy(x, y, transform):
    a, _, c, _, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    col = ((x - c) / a).astype(int)
    row = ((y - f) / e).astype(int)
    return row, col

# ---------------------------------------------------------------- per-site W / Sw
def process(site: str, lay):
    gdf = fetch_nwi(site)
    catch = gpd.read_file(os.path.join(SENS, f"catchment_{site}.gpkg")).to_crs(32616)
    catch_area = float(catch.geometry.union_all().area)
    with rasterio.open(os.path.join(DATA, f"T_{site}.tif")) as r:
        Tg = r.read(1)
    transform, shape = lay["transform"], lay["shape"]; dist, fill = lay["dist"], lay["fill"]
    H, W = shape

    # catchment-median travel time -> tau
    cmask = rasterize([(catch.geometry.union_all(), 1)], out_shape=shape,
                      transform=transform, dtype="uint8").astype(bool)
    Tcatch = Tg[cmask & np.isfinite(Tg) & (Tg > 0)]
    tau = float(np.median(Tcatch)) if Tcatch.size else 24.0

    if len(gdf) == 0:
        return dict(site_no=site, n_wet=0), None

    # per-wetland sampling at centroids
    cent = gdf.geometry.centroid
    A = gdf.geometry.area.to_numpy()
    row, col = rc_from_xy(cent.x.to_numpy(), cent.y.to_numpy(), transform)
    inb = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    row, col = np.clip(row, 0, H - 1), np.clip(col, 0, W - 1)
    T_ij = Tg[row, col].astype(float)
    d_ch = dist[row, col].astype(float)
    # fall back to median tau where T is nan/non-contributing
    T_ij = np.where(np.isfinite(T_ij) & (T_ij > 0), T_ij, tau)
    M = np.array([modifier(x) for x in gdf["ATTRIBUTE"].fillna("")])
    C_bin = (d_ch < NEAR_M).astype(float)
    C_grad = np.exp(-d_ch / NEAR_M)
    f_exp = np.exp(-T_ij / tau)
    f_harm = 1.0 / (1.0 + T_ij / tau)
    V_va = SW_C * np.power(A, SW_BETA)

    gdf = gdf.assign(area_m2=A, T_hr=T_ij, dist_chan_m=d_ch, M=M, C_bin=C_bin,
                     f_exp=f_exp, V_va_m3=V_va)

    # DEM depression-volume cross-check: sum filled depth over wetland footprints
    wmask = rasterize([(g, 1) for g in gdf.geometry], out_shape=shape,
                      transform=transform, dtype="uint8").astype(bool)
    Sw_dem = float(np.nansum(fill[wmask]) * 100.0)     # m3 (10x10 m cells)

    wet_area = float(A.sum())
    rec = dict(
        site_no=site, n_wet=int(len(gdf)),
        catch_km2=round(catch_area / 1e6, 1),
        wet_km2=round(wet_area / 1e6, 3),
        wet_frac=round(wet_area / catch_area, 5),
        near_frac=round(float((C_bin * A).sum() / wet_area), 4) if wet_area else 0.0,
        meanT_wet_hr=round(float(np.average(T_ij, weights=A)), 1),
        tau_hr=round(tau, 1),
        # W variants (m2)
        W_area=round(wet_area, 1),
        W_exp=round(float((A * C_bin * M * f_exp).sum()), 1),
        W_harm=round(float((A * C_bin * M * f_harm).sum()), 1),
        W_exp_noC=round(float((A * M * f_exp).sum()), 1),
        W_exp_noM=round(float((A * C_bin * f_exp).sum()), 1),
        W_exp_grad=round(float((A * C_grad * M * f_exp).sum()), 1),
        # storage (m3)
        Sw_va_m3=round(float(V_va.sum()), 1),
        Sw_dem_m3=round(Sw_dem, 1),
        Sw_va_depth_mm=round(float(V_va.sum()) / catch_area * 1000, 2),
    )
    return rec, gdf

# ---------------------------------------------------------------- QA
def qa(panel, gdfs):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(13, 13))
    for ax, site in zip(axes.ravel(), SITES):
        g = gdfs.get(site)
        catch = gpd.read_file(os.path.join(SENS, f"catchment_{site}.gpkg")).to_crs(32616)
        catch.boundary.plot(ax=ax, color="k", lw=0.6)
        if g is not None and len(g):
            g.plot(ax=ax, column="f_exp", cmap="viridis", vmin=0, vmax=1,
                   markersize=2, legend=False)
        r = panel[panel.site_no == site].iloc[0]
        ax.set_title(f"{site}: {int(r.n_wet)} wetlands, wet {r.wet_km2:.0f}km²\n"
                     f"W_exp/area={r.W_exp/r.W_area:.2f}  near_frac={r.near_frac:.2f}",
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Step 3 QA: NWI wetlands shaded by travel-time decay f(T)=exp(-T/τ)", y=1.0)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "step3_W_qa.png"), dpi=110,
                                    bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- main
def run(sites):
    lay = aoi_layers()
    rows, gdfs = [], {}
    for site in sites:
        t0 = time.time()
        rec, gdf = process(site, lay)
        rows.append(rec); gdfs[site] = gdf
        if rec.get("n_wet"):
            print(f"[{site}] n_wet={rec['n_wet']} wet_km2={rec['wet_km2']} "
                  f"wet_frac={rec['wet_frac']} near_frac={rec['near_frac']} "
                  f"W_exp/area={rec['W_exp']/rec['W_area']:.2f} "
                  f"Sw_va={rec['Sw_va_m3']:.3e} Sw_dem={rec['Sw_dem_m3']:.3e} "
                  f"({round(time.time()-t0,1)}s)", flush=True)
        else:
            print(f"[{site}] n_wet=0 (no NWI features returned) ({round(time.time()-t0,1)}s)", flush=True)
    panel = pd.DataFrame(rows)
    panel.to_csv(os.path.join(OUT, "panel_W_Sw.csv"), index=False)
    if len(sites) == len(SITES):
        qa(panel, gdfs)
    print("\n=== STEP 3 SUMMARY (panel_W_Sw.csv) ===")
    show = ["site_no", "n_wet", "wet_km2", "wet_frac", "near_frac", "meanT_wet_hr",
            "tau_hr", "W_exp", "Sw_va_m3", "Sw_dem_m3", "Sw_va_depth_mm"]
    print(panel[show].to_string(index=False))

if __name__ == "__main__":
    run(sys.argv[1:] if len(sys.argv) > 1 else SITES)
