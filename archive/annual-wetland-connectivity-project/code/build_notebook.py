"""Generate wetland_travel_time_v3.ipynb — enhanced visualisations per step.

Computation logic is kept identical to v2.0 (already debugged); this version only
- repoints the data cache to ./data  (download once, reuse forever)
- adds a shared plotting helper (hillshade + figure saver)
- enriches every section with extra diagnostic plots.
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
md = lambda s: cells.append(new_markdown_cell(s))
code = lambda s: cells.append(new_code_cell(s))

# ── CELL 1: Overview ──────────────────────────────────────────────────────────
md("""\
# Wetland → Stream Travel-Time Pipeline  *(v3 — visual edition)*

**Goal:** For a USGS-gauged watershed, compute the total travel time `t` for water
from every upstream grid cell `i` to the sensor `s`.  Each cell gets:
- a boolean flag `is_wetland(i)` from NLCD
- overland travel time `t1` (hillslope → stream)
- channel travel time `t2` (stream → sensor)
- total `t_total = t1 + t2`

These are aggregated into a travel-time-weighted wetland metric `W`
to study how wetland change affects downstream river stage (Section 13 stub).

> **What's new in v3:** every processing step now produces extra diagnostic
> visualisations (hillshade context, histograms, before/after conditioning,
> overlays). All downloads are cached **once** into `./data/` — re-runs are instant.
> **No API keys are required** — NLDI, 3DEP, NLCD and USGS waterservices are all public.

---

## Notation

| Symbol | Name | Meaning | Units |
|---|---|---|---|
| `s` | Sensor / outlet | USGS gauge with observed stage | — |
| `i` | Pixel | Grid cell in the contributing watershed | — |
| `j` | Inflow point | Channel cell where pixel `i`'s flowpath enters the stream | — |
| `d1`,`t1`,`s1` | Overland | distance / time / slope `i → j` | m, s, m/m |
| `d2`,`t2`,`s2` | Channel | distance / time / slope `j → s` | m, s, m/m |
| `t_total` | Total | `t1 + t2` | s (or hr) |
| `W` | Weighted metric | Aggregate wetland metric (Section 13) | — |

## Key Assumptions
1. **Surface-only routing** from DEM-derived D8 flow directions (no groundwater).
2. **Kinematic approximation** — steady per-cell velocities (TR-55 style).
3. **Constant channel hydraulics** — uniform Manning `n` and hydraulic radius `R`.
4. **NLCD wetland classes 90 + 95** at 30 m (small/narrow wetlands missed).
""")

# ── CELL 2: Imports ───────────────────────────────────────────────────────────
md("## Section 2 — Imports, Environment Check & Plot Helpers")
code("""\
import os, sys, warnings
from pathlib import Path
from io import StringIO

import numpy as np
# FIX: pysheds 0.5 calls np.in1d, which NumPy 2.0 removed. Shim it.
if not hasattr(np, "in1d"):
    np.in1d = np.isin
import pandas as pd
import geopandas as gpd
import rasterio, rasterio.enums, rasterio.crs
import rioxarray as rxr
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LightSource
import scipy.ndimage
from shapely.geometry import Point

import pysheds
from pysheds.grid import Grid
from pysheds.sview import Raster as PyShedsRaster, ViewFinder

import pygeohydro as gh
from pygeohydro import NWIS
import py3dep
import pynhd
from pynhd import NLDI
import whitebox

warnings.filterwarnings("ignore")
%matplotlib inline
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": False})

print("Package versions:")
for name, mod in [("numpy",np),("pandas",pd),("geopandas",gpd),("rasterio",rasterio),
                  ("xarray",xr),("pysheds",pysheds),("pygeohydro",gh),
                  ("py3dep",py3dep),("pynhd",pynhd),("whitebox",whitebox)]:
    print(f"  {name:12s} {getattr(mod,'__version__','?')}")
print(f"  {'Python':12s} {sys.version.split()[0]}")


# ── Shared plot helpers (used throughout) ────────────────────────────────────
def hillshade(z, res=10, azdeg=315, altdeg=45, vert_exag=2.0):
    \"\"\"Matplotlib LightSource hillshade of a 2D elevation array (NaN-safe).\"\"\"
    z = np.array(z, dtype=float)
    zf = np.where(np.isfinite(z), z, np.nanmin(z))
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    return ls.hillshade(zf, vert_exag=vert_exag, dx=res, dy=res)

def finish(fig, name, outdir):
    \"\"\"Tidy + save a figure to outdir and show it.\"\"\"
    fig.tight_layout()
    fig.savefig(Path(outdir) / name, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  saved → {name}")
""")

# ── CELL 3: Config ────────────────────────────────────────────────────────────
md("""\
## Section 3 — Configuration

All tuneable parameters live here. Change `STATION_ID` to swap gauge.
`DATA_DIR` is the single download cache — fetched once, reused forever.
""")
code("""\
# ── CHANGE THESE TO SWITCH STATION OR SETTINGS ──────────────────────────────
STATION_ID    = "04087000"   # USGS — Milwaukee River at Milwaukee
PROJECTED_CRS = "EPSG:32616" # UTM 16N: distances/slopes come out in metres
DEM_RES       = 10           # 3DEP DEM resolution in metres
NLCD_YEARS    = [2021]       # NLCD epochs; extend to [2001, 2021] for ΔW

ACC_THRESHOLD = 5000         # accumulation cells → stream definition; TUNE per basin
SLOPE_MIN     = 1e-3         # floor slope so V ≠ 0 on flat cells

K_OVERLAND    = 4.92         # TR-55 overland  V = K·√S   [m/s @ 1 m/m]
N_CHANNEL     = 0.04         # Manning n (natural channel)
R_CHANNEL     = 0.7          # hydraulic radius [m]

# ─────────────────────────────────────────────────────────────────────────────
# Anchor data/ and outputs/ at the PROJECT ROOT so paths are identical whether
# the kernel's CWD is the project root (interactive) or ./notebook (nbconvert).
_cwd = Path.cwd()
PROJECT_ROOT = _cwd.parent if _cwd.name == "notebook" else _cwd
DATA_DIR = PROJECT_ROOT / "data"      # << every download lands here, once
OUT_DIR  = PROJECT_ROOT / "outputs"   # << every figure / table lands here
CACHE_DIR = DATA_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Data cache: {DATA_DIR.resolve()}")
print(f"Outputs:    {OUT_DIR.resolve()}")
print(f"Station:    {STATION_ID}   CRS: {PROJECTED_CRS}")
""")

# ── CELL 4: Basin + stage ─────────────────────────────────────────────────────
md("""\
## Section 4 — Delineate Basin & Fetch Stage Data

NLDI (`pynhd`) gives the upstream contributing basin polygon for the gauge — the
programmatic equivalent of a StreamStats delineation. Stage (gage height,
parameter `00065`) comes from the USGS waterservices daily-value REST endpoint.

**Viz:** basin + gauge map, basin-in-context locator, and an annotated stage
hydrograph with min/median/max bands.
""")
code("""\
import requests

basin_path = DATA_DIR / f"basin_{STATION_ID}.gpkg"
gauge_path = DATA_DIR / f"gauge_{STATION_ID}.gpkg"
stage_path = DATA_DIR / f"stage_{STATION_ID}.parquet"

if basin_path.exists() and gauge_path.exists():
    basin = gpd.read_file(basin_path); gauge = gpd.read_file(gauge_path)
    print("Loaded basin & gauge from cache.")
else:
    print("Fetching basin from NLDI (~10 s)…")
    nldi  = NLDI()
    basin = nldi.get_basins(STATION_ID, fsource="nwissite")
    # Gauge point: use the USGS site service for the lat/lon. (NLDI's
    # getfeature_byid hits a pygeoutils/geopandas incompatibility on this
    # stack; the site service is public, key-free, and far simpler.)
    site_url = f"https://waterservices.usgs.gov/nwis/site/?sites={STATION_ID}&format=rdb"
    sresp = requests.get(site_url, timeout=30); sresp.raise_for_status()
    slines = [l for l in sresp.text.splitlines() if not l.startswith("#")]
    site_df = pd.read_csv(StringIO("\\n".join(slines)), sep="\\t", skiprows=[1])
    site_df.columns = site_df.columns.str.strip()
    lat = float(site_df["dec_lat_va"].iloc[0]); lon = float(site_df["dec_long_va"].iloc[0])
    gauge = gpd.GeoDataFrame(
        {"site_no": [STATION_ID], "station_nm": [site_df["station_nm"].iloc[0]]},
        geometry=[Point(lon, lat)], crs="EPSG:4326")
    basin.to_file(basin_path, driver="GPKG"); gauge.to_file(gauge_path, driver="GPKG")
    print(f"  → cached {len(basin)} basin polygon(s) + gauge ({lat:.4f}, {lon:.4f}).")

basin_proj = basin.to_crs(PROJECTED_CRS)
gauge_proj = gauge.to_crs(PROJECTED_CRS)
basin_area_km2 = basin_proj.geometry.area.sum() / 1e6
print(f"Basin area: {basin_area_km2:.1f} km²")
assert basin_area_km2 > 0, "Basin area is zero — check STATION_ID"

if stage_path.exists():
    stage_df = pd.read_parquet(stage_path)
    print(f"Loaded {len(stage_df)} stage records from cache.")
else:
    print("Fetching stage from USGS waterservices…")
    url = ("https://waterservices.usgs.gov/nwis/dv/"
           f"?sites={STATION_ID}&parameterCd=00065&period=P730D&format=rdb")
    resp = requests.get(url, timeout=30); resp.raise_for_status()
    lines = [l for l in resp.text.splitlines() if not l.startswith("#")]
    stage_df = pd.read_csv(StringIO("\\n".join(lines)), sep="\\t", skiprows=[1])
    stage_df.columns = stage_df.columns.str.strip()
    val_cols = [c for c in stage_df.columns if "00065" in c and not c.endswith("_cd")]
    if val_cols:
        stage_df["stage_ft"] = pd.to_numeric(stage_df[val_cols[0]], errors="coerce")
    stage_df["datetime"] = pd.to_datetime(stage_df.get("datetime", None), errors="coerce")
    stage_df = stage_df.dropna(subset=["datetime"])
    stage_df.to_parquet(stage_path)
    print(f"  → cached {len(stage_df)} daily records.")

# ── Viz ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

basin_proj.plot(ax=axes[0], facecolor="#bcd6ec", edgecolor="steelblue", lw=1.5)
basin_proj.boundary.plot(ax=axes[0], color="steelblue", lw=1.5)
gauge_proj.plot(ax=axes[0], color="red", markersize=120, marker="^", zorder=5)
axes[0].set_title(f"Upstream watershed — USGS {STATION_ID}")
axes[0].set_xlabel("Easting (m)"); axes[0].set_ylabel("Northing (m)")
axes[0].annotate(f"{basin_area_km2:.0f} km²", xy=(0.05, 0.93),
                 xycoords="axes fraction", fontsize=11, fontweight="bold")

# locator (basin centroid in geographic coords, simple context)
b4326 = basin.to_crs(4326)
minx, miny, maxx, maxy = b4326.total_bounds
b4326.plot(ax=axes[1], facecolor="#7fb069", edgecolor="darkgreen")
axes[1].set_xlim(minx-4, maxx+4); axes[1].set_ylim(miny-3, maxy+3)
axes[1].set_title("Basin location (lon/lat context)")
axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude")
axes[1].grid(True, ls=":", alpha=0.5)

if "stage_ft" in stage_df.columns and stage_df["stage_ft"].notna().any():
    s = stage_df.dropna(subset=["stage_ft"])
    axes[2].plot(s["datetime"], s["stage_ft"], lw=0.9, color="steelblue")
    med = s["stage_ft"].median()
    axes[2].axhline(med, color="red", ls="--", lw=1, label=f"median {med:.2f} ft")
    axes[2].fill_between(s["datetime"], s["stage_ft"].min(), s["stage_ft"],
                         color="steelblue", alpha=0.12)
    axes[2].set_ylabel("Gage height (ft)"); axes[2].legend(fontsize=8)
else:
    axes[2].text(0.5, 0.5, "Stage data not available", transform=axes[2].transAxes, ha="center")
axes[2].set_title(f"Daily Stage — USGS {STATION_ID}"); axes[2].set_xlabel("Date")
axes[2].tick_params(axis="x", rotation=30)

finish(fig, "01_basin_stage.png", OUT_DIR)
print("Section 4 complete ✓")
""")

# ── CELL 5: DEM ───────────────────────────────────────────────────────────────
md("""\
## Section 5 — Fetch & Reproject the 3DEP DEM

3DEP DEM at `DEM_RES` m via `py3dep.get_dem()`, immediately reprojected to UTM so
distances/slopes are in metres.

**Viz:** elevation map, hillshade relief, and an elevation histogram.
""")
code("""\
dem_path = DATA_DIR / f"dem_{STATION_ID}_{DEM_RES}m.tif"

if dem_path.exists():
    dem_da = rxr.open_rasterio(dem_path, masked=True).squeeze()
    print(f"Loaded DEM from cache — shape {dem_da.shape}, CRS {dem_da.rio.crs}")
else:
    print(f"Fetching 3DEP DEM at {DEM_RES} m (30–60 s)…")
    dem_raw = py3dep.get_dem(basin.geometry.iloc[0], resolution=DEM_RES)
    dem_proj = dem_raw.rio.reproject(PROJECTED_CRS, resolution=DEM_RES,
                                     resampling=rasterio.enums.Resampling.bilinear)
    dem_da = dem_proj.squeeze(); dem_da.rio.to_raster(dem_path)
    print(f"  → cached DEM shape {dem_da.shape}")

dem_arr0 = np.array(dem_da, dtype=float)
dem_valid = dem_arr0[np.isfinite(dem_arr0) & (dem_arr0 > -9000)]
elev_min, elev_max = float(dem_valid.min()), float(dem_valid.max())
print(f"Elevation range: {elev_min:.1f} – {elev_max:.1f} m  (relief {elev_max-elev_min:.0f} m)")
assert dem_da.shape[0] > 10 and dem_da.shape[1] > 10
assert elev_max > elev_min

# ── Viz ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
im = axes[0].imshow(dem_arr0, cmap="terrain", origin="upper")
plt.colorbar(im, ax=axes[0], label="Elevation (m)", fraction=0.046)
basin_proj.boundary.plot(ax=axes[0], color="black", lw=1.0, transform=axes[0].transData) if False else None
axes[0].set_title(f"3DEP DEM — {DEM_RES} m"); axes[0].set_xticks([]); axes[0].set_yticks([])

axes[1].imshow(hillshade(dem_arr0, res=DEM_RES), cmap="gray", origin="upper")
axes[1].imshow(dem_arr0, cmap="terrain", origin="upper", alpha=0.45)
axes[1].set_title("Hillshade relief"); axes[1].set_xticks([]); axes[1].set_yticks([])

axes[2].hist(dem_valid, bins=60, color="sienna", edgecolor="white", lw=0.2)
axes[2].axvline(np.median(dem_valid), color="navy", ls="--",
                label=f"median {np.median(dem_valid):.0f} m")
axes[2].set_xlabel("Elevation (m)"); axes[2].set_ylabel("Cell count")
axes[2].set_title("Elevation distribution"); axes[2].legend(fontsize=8)

finish(fig, "02_dem.png", OUT_DIR)
print("Section 5 complete ✓")
""")

# ── CELL 6: Conditioning + flow grids ─────────────────────────────────────────
md("""\
## Section 6 — Condition DEM (WhiteBox) & Compute Flow Direction / Accumulation

**WhiteBox least-cost breaching** instead of pysheds `fill_depressions`. On this
low-relief glacial basin pysheds' fill left the drainage *fragmented* — flow
accumulation maxed out at ~108 km² for an 1809 km² basin, so the gauge captured
only a tiny sub-catchment. WhiteBox `BreachDepressionsLeastCost` carves through
the spurious dams, integrating the network so max accumulation ≈ basin area.
We then `resolve_flats` and compute D8 flow direction + accumulation in pysheds.

**Viz:** where breaching carved the DEM, the D8 direction field, and the
log-accumulation river network.
""")
code("""\
# ── WhiteBox hydrological conditioning ───────────────────────────────────────
# WHY: pysheds' fill_pits→fill_depressions→resolve_flats did NOT integrate this
# low-relief glacial basin — flow accumulation stalled at ~108 km² for a 1809 km²
# basin (the drainage fragmented into disconnected pieces). WhiteBox least-cost
# breaching CARVES through the spurious dams so the network becomes fully
# connected; verified max accumulation then ≈ basin area.
dem_cond_path = DATA_DIR / f"dem_{STATION_ID}_{DEM_RES}m_cond.tif"
if not dem_cond_path.exists():
    print("Conditioning DEM with WhiteBox BreachDepressionsLeastCost…")
    wbt = whitebox.WhiteboxTools(); wbt.verbose = False
    wbt.breach_depressions_least_cost(os.path.abspath(str(dem_path)),
                                      os.path.abspath(str(dem_cond_path)),
                                      dist=2000, fill=True)
    print("  → cached conditioned DEM.")
else:
    print("Loaded conditioned DEM from cache.")

grid     = Grid.from_raster(str(dem_cond_path))
dem_cond = grid.read_raster(str(dem_cond_path))
dem_raw_pys = grid.read_raster(str(dem_path))     # raw elevations, for carve viz
print(f"grid shape {grid.shape}  nodata {dem_cond.nodata}")

# resolve any residual flats, then D8 flow direction + accumulation
inflated = grid.resolve_flats(dem_cond)
fdir = grid.flowdir(inflated)
acc  = grid.accumulation(fdir)
acc_arr = np.array(acc, dtype=float); acc_arr[acc_arr <= 0] = np.nan
print(f"Accumulation max: {np.nanmax(acc_arr):.0f} cells "
      f"(= {np.nanmax(acc_arr)*DEM_RES**2/1e6:.1f} km²)  [NLDI basin ≈ {basin_area_km2:.0f} km²]")
np.save(DATA_DIR / f"acc_{STATION_ID}.npy", np.array(acc))

# carve depth: how much WhiteBox lowered each cell (breached channels/dams)
carve = np.array(dem_raw_pys, dtype=float) - np.array(dem_cond, dtype=float)
carve[~np.isfinite(carve)] = 0.0
n_carved = int((carve > 0.01).sum())
print(f"Cells lowered by breaching: {n_carved} "
      f"({100*n_carved/carve.size:.2f}%), max carve {carve.max():.2f} m")

# ── Viz ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

cb = np.where(carve > 0.01, carve, np.nan)
axes[0].imshow(hillshade(dem_arr0, res=DEM_RES), cmap="gray", origin="upper")
im0 = axes[0].imshow(cb, cmap="magma_r", origin="upper",
                     vmax=np.nanpercentile(carve[carve > 0.01], 98) if n_carved else 1)
plt.colorbar(im0, ax=axes[0], label="Breach carve depth (m)", fraction=0.046)
axes[0].set_title(f"WhiteBox breaching\\n({n_carved} cells carved)")
axes[0].set_xticks([]); axes[0].set_yticks([])

fdir_arr = np.array(fdir, dtype=float); fdir_arr[fdir_arr == fdir.nodata] = np.nan
im1 = axes[1].imshow(fdir_arr, cmap="hsv", origin="upper")
plt.colorbar(im1, ax=axes[1], label="D8 code (1..128)", fraction=0.046)
axes[1].set_title("D8 Flow Direction"); axes[1].set_xticks([]); axes[1].set_yticks([])

im2 = axes[2].imshow(np.log10(acc_arr), cmap="cubehelix_r", origin="upper")
plt.colorbar(im2, ax=axes[2], label="log₁₀(accumulation cells)", fraction=0.046)
axes[2].set_title("Flow Accumulation (log)"); axes[2].set_xticks([]); axes[2].set_yticks([])

finish(fig, "03_flow_grids.png", OUT_DIR)
print("Section 6 complete ✓")
""")

# ── CELL 7: Streams + outlet + HAND ───────────────────────────────────────────
md("""\
## Section 7 — Stream Network, Outlet Snapping & HAND

Streams = cells with `acc > ACC_THRESHOLD`. Gauge snapped to nearest stream cell.
HAND = height above nearest drainage; `hand_idx` (return_index) gives the flat
index of the receiving channel cell `j` for every cell — the key to splitting
`d1`/`d2` later.

**Viz:** stream network over hillshade with the snapped outlet, a zoom-in on the
outlet, and the HAND surface.
""")
code("""\
stream_mask = acc > ACC_THRESHOLD
stream_arr_bool = np.array(stream_mask, dtype=bool)
n_stream = int(stream_arr_bool.sum())
print(f"Stream cells: {n_stream} ({100*n_stream/stream_arr_bool.size:.2f}% of grid)")
assert n_stream > 0, f"No stream cells — reduce ACC_THRESHOLD ({ACC_THRESHOLD})"

gx = float(gauge_proj.geometry.x.iloc[0]); gy = float(gauge_proj.geometry.y.iloc[0])
snapped  = grid.snap_to_mask(stream_mask, np.array([[gx, gy]]))
x_s, y_s = float(snapped[0, 0]), float(snapped[0, 1])
snap_dist = float(np.hypot(gx - x_s, gy - y_s))
c_s, r_s  = grid.nearest_cell(x_s, y_s)
print(f"Snapped outlet ({x_s:.1f}, {y_s:.1f}) — snap distance {snap_dist:.1f} m")
assert snap_dist < 2000, "Snap distance > 2 km — check CRS or ACC_THRESHOLD"
np.save(DATA_DIR / f"outlet_{STATION_ID}.npy", np.array([x_s, y_s, r_s, c_s]))

hand = grid.compute_hand(fdir, inflated, stream_mask, return_index=False)
hand_arr = np.array(hand, dtype=float); hand_arr[hand_arr < 0] = 0.0
hand_idx = grid.compute_hand(fdir, inflated, stream_mask, return_index=True)
hand_idx_arr = np.array(hand_idx, dtype=np.int64)
print(f"HAND range: {np.nanmin(hand_arr):.2f} – {np.nanmax(hand_arr):.2f} m")
np.save(DATA_DIR / f"stream_mask_{STATION_ID}.npy", stream_arr_bool)
np.save(DATA_DIR / f"hand_{STATION_ID}.npy", hand_arr)
np.save(DATA_DIR / f"hand_idx_{STATION_ID}.npy", hand_idx_arr)

# ── Viz ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

axes[0].imshow(hillshade(dem_arr0, res=DEM_RES), cmap="gray", origin="upper")
axes[0].imshow(np.where(stream_arr_bool, 1.0, np.nan), cmap="winter", origin="upper")
axes[0].plot(c_s, r_s, "r^", markersize=14, label=f"outlet (snap {snap_dist:.0f} m)")
axes[0].set_title(f"Stream network (acc > {ACC_THRESHOLD})")
axes[0].legend(fontsize=8, loc="lower right"); axes[0].set_xticks([]); axes[0].set_yticks([])

pad = 40
r0,r1 = max(0,r_s-pad), min(grid.shape[0],r_s+pad)
c0,c1 = max(0,c_s-pad), min(grid.shape[1],c_s+pad)
axes[1].imshow(np.log10(acc_arr)[r0:r1, c0:c1], cmap="cubehelix_r", origin="upper")
axes[1].imshow(np.where(stream_arr_bool,1.0,np.nan)[r0:r1,c0:c1], cmap="autumn", origin="upper", alpha=0.6)
axes[1].plot(c_s-c0, r_s-r0, "b^", markersize=16)
axes[1].set_title("Outlet snapping (zoom)"); axes[1].set_xticks([]); axes[1].set_yticks([])

im = axes[2].imshow(hand_arr, cmap="terrain_r", origin="upper",
                    vmax=np.nanpercentile(hand_arr[hand_arr > 0], 95))
plt.colorbar(im, ax=axes[2], label="HAND (m)", fraction=0.046)
axes[2].set_title("HAND — Height Above Nearest Drainage")
axes[2].set_xticks([]); axes[2].set_yticks([])

finish(fig, "04_streams_hand.png", OUT_DIR)
print("Section 7 complete ✓")
""")

# ── CELL 8: velocity + tau ────────────────────────────────────────────────────
md("""\
## Section 8 — Per-Cell Distances, Slopes & Velocity Grid

`L` = D8 step length, `S` = step slope (floored at `SLOPE_MIN`).
Channel cells use Manning `V=(1/n)R^{2/3}S^{1/2}`; overland uses TR-55 `V=K·√S`.
Cell travel time `τ = L / V`.

**Viz:** slope (log), velocity grid, per-cell τ, plus distributions of slope and
velocity split by channel vs overland.
""")
code("""\
L = grid.cell_distances(fdir); L_arr = np.array(L, dtype=float)
S_raw = grid.cell_slopes(inflated, fdir); S_arr = np.array(S_raw, dtype=float)
S_arr = np.where(np.isfinite(S_arr) & (S_arr > 0), S_arr, SLOPE_MIN)
n_floor = int((S_arr == SLOPE_MIN).sum())
print(f"Step distance {np.nanmin(L_arr):.1f}–{np.nanmax(L_arr):.1f} m | "
      f"cells at slope floor: {n_floor} ({100*n_floor/S_arr.size:.1f}%)")

V_channel  = (1.0/N_CHANNEL) * (R_CHANNEL**(2.0/3.0)) * np.sqrt(S_arr)
V_overland = K_OVERLAND * np.sqrt(S_arr)
V = np.where(stream_arr_bool, V_channel, V_overland)
tau_arr = L_arr / V
tau_arr = np.where(np.isfinite(tau_arr) & (tau_arr > 0), tau_arr, np.nan)
print(f"Velocity channel {np.nanmin(V[stream_arr_bool]):.3f}–{np.nanmax(V[stream_arr_bool]):.2f} m/s | "
      f"overland {np.nanmin(V[~stream_arr_bool]):.4f}–{np.nanmax(V[~stream_arr_bool]):.2f} m/s")
print(f"Cell τ {np.nanmin(tau_arr):.2f}–{np.nanmax(tau_arr):.0f} s")
for nm, a in [("L",L_arr),("S",S_arr),("V",V),("tau",tau_arr)]:
    np.save(DATA_DIR / f"{nm}_{STATION_ID}.npy", a)

# ── Viz ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
im0 = axes[0,0].imshow(np.log10(S_arr), cmap="plasma", origin="upper")
plt.colorbar(im0, ax=axes[0,0], label="log₁₀ slope (m/m)", fraction=0.046)
axes[0,0].set_title("Slope (log)"); axes[0,0].set_xticks([]); axes[0,0].set_yticks([])

im1 = axes[0,1].imshow(V, cmap="viridis", origin="upper",
                       vmax=np.nanpercentile(V, 99))
plt.colorbar(im1, ax=axes[0,1], label="Velocity (m/s)", fraction=0.046)
axes[0,1].set_title("Velocity (Manning channel / TR-55 overland)")
axes[0,1].set_xticks([]); axes[0,1].set_yticks([])

tau_hr = tau_arr / 3600.0
im2 = axes[1,0].imshow(tau_hr, cmap="hot_r", origin="upper",
                       vmax=np.nanpercentile(tau_hr[np.isfinite(tau_hr)], 98))
plt.colorbar(im2, ax=axes[1,0], label="τ (hr)", fraction=0.046)
axes[1,0].set_title("Per-cell travel time τ = L/V")
axes[1,0].set_xticks([]); axes[1,0].set_yticks([])

axes[1,1].hist(V[~stream_arr_bool & np.isfinite(V)], bins=60, alpha=0.6,
               color="seagreen", label="overland", density=True)
axes[1,1].hist(V[stream_arr_bool & np.isfinite(V)], bins=60, alpha=0.6,
               color="steelblue", label="channel", density=True)
axes[1,1].set_xlabel("Velocity (m/s)"); axes[1,1].set_ylabel("Density")
axes[1,1].set_title("Velocity distribution by cell type"); axes[1,1].legend(fontsize=9)

finish(fig, "05_velocity_tau.png", OUT_DIR)
print("Section 8 complete ✓")
""")

# ── CELL 9: accumulate to outlet ──────────────────────────────────────────────
md("""\
## Section 9 — Whole-Grid Travel Time to Outlet

`distance_to_outlet` accumulates weights along the D8 flowpath to the outlet:
with weights `L` → distance `D`; with weights `τ` → travel time `T`.

**Viz:** flow-path distance surface, travel-time surface with iso-time contours,
and the basin-wide travel-time histogram (concentration-time curve).
""")
code("""\
tau_raster = PyShedsRaster(tau_arr, viewfinder=L.viewfinder)
print("Computing D (flow-path distance)…")
D_grid = grid.distance_to_outlet(x_s, y_s, fdir, weights=L, nodata_out=np.nan,
                                 routing="d8", method="shortest", xytype="coordinate")
print("Computing T (travel time)…")
T_grid = grid.distance_to_outlet(x_s, y_s, fdir, weights=tau_raster, nodata_out=np.nan,
                                 routing="d8", method="shortest", xytype="coordinate")
D_arr = np.array(D_grid, dtype=float); T_arr = np.array(T_grid, dtype=float)
T_hr = T_arr / 3600.0
finite_mask = np.isfinite(D_arr) & np.isfinite(T_arr)
print(f"D {np.nanmin(D_arr):.0f}–{np.nanmax(D_arr):.0f} m | "
      f"T {np.nanmin(T_hr):.2f}–{np.nanmax(T_hr):.2f} hr | "
      f"finite cells {finite_mask.sum()}/{finite_mask.size}")
assert finite_mask.sum() > 0
np.save(DATA_DIR / f"D_{STATION_ID}.npy", D_arr)
np.save(DATA_DIR / f"T_{STATION_ID}.npy", T_arr)

# ── Viz ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
im0 = axes[0].imshow(D_arr/1000, cmap="YlOrRd", origin="upper",
                     vmax=np.nanpercentile(D_arr[finite_mask]/1000, 98))
plt.colorbar(im0, ax=axes[0], label="Flow-path distance (km)", fraction=0.046)
axes[0].plot(c_s, r_s, "b^", markersize=12); axes[0].set_title("Flow-path distance D")
axes[0].set_xticks([]); axes[0].set_yticks([])

im1 = axes[1].imshow(T_hr, cmap="YlOrRd", origin="upper",
                     vmax=np.nanpercentile(T_hr[finite_mask], 98))
try:
    cs = axes[1].contour(T_hr, levels=np.nanpercentile(T_hr[finite_mask],[20,40,60,80]),
                         colors="k", linewidths=0.5, alpha=0.5)
except Exception:
    pass
plt.colorbar(im1, ax=axes[1], label="Travel time (hr)", fraction=0.046)
axes[1].plot(c_s, r_s, "b^", markersize=12); axes[1].set_title("Travel time T (iso-time contours)")
axes[1].set_xticks([]); axes[1].set_yticks([])

tvals = T_hr[finite_mask]
axes[2].hist(tvals, bins=60, color="darkorange", edgecolor="white", lw=0.2)
axes[2].axvline(np.median(tvals), color="navy", ls="--",
                label=f"median {np.median(tvals):.1f} hr")
axes[2].set_xlabel("Travel time to outlet (hr)"); axes[2].set_ylabel("Cell count")
axes[2].set_title("Basin time-to-outlet distribution"); axes[2].legend(fontsize=8)

finish(fig, "06_travel_time_surface.png", OUT_DIR)
print("Section 9 complete ✓")
""")

# ── CELL 10: NLCD wetland ─────────────────────────────────────────────────────
md("""\
## Section 10 — NLCD Wetland Mask

NLCD land cover via `pygeohydro.nlcd_bygeom()`, reclassified to wetland classes
**90** (woody) + **95** (emergent herbaceous), resampled to the DEM grid.

**Viz:** full land-cover map, the wetland mask over hillshade, and a wetland-area
readout.
""")
code("""\
nlcd_wet_path  = DATA_DIR / f"nlcd_wetland_{STATION_ID}_{NLCD_YEARS[0]}.tif"
nlcd_full_path = DATA_DIR / f"nlcd_cover_{STATION_ID}_{NLCD_YEARS[0]}.tif"

cover_grid = None
if nlcd_wet_path.exists():
    wet_da = rxr.open_rasterio(nlcd_wet_path, masked=True).squeeze()
    if nlcd_full_path.exists():
        cover_grid = np.array(rxr.open_rasterio(nlcd_full_path, masked=True).squeeze(), dtype=float)
    print(f"Loaded NLCD wetland mask from cache — shape {wet_da.shape}")
else:
    print(f"Fetching NLCD {NLCD_YEARS[0]} (30–90 s)…")
    nlcd_raw = gh.nlcd_bygeom(basin, resolution=30, years={"cover": NLCD_YEARS})
    nlcd_ds  = nlcd_raw[list(nlcd_raw.keys())[0]]
    cover_da = nlcd_ds[f"cover_{NLCD_YEARS[0]}"]
    cover_arr = np.array(cover_da).squeeze().astype(float)
    wet_arr   = np.where(np.isin(cover_arr, [90, 95]), 1.0, 0.0)
    dem_ref   = rxr.open_rasterio(dem_path, masked=True)
    wet_da = cover_da.copy(data=wet_arr.reshape(cover_da.shape)).rio.reproject_match(
        dem_ref, resampling=rasterio.enums.Resampling.nearest).squeeze()
    wet_da.rio.to_raster(nlcd_wet_path)
    # also keep the full cover raster (resampled) for the land-cover map
    cover_full = cover_da.rio.reproject_match(
        dem_ref, resampling=rasterio.enums.Resampling.nearest).squeeze()
    cover_full.rio.to_raster(nlcd_full_path)
    cover_grid = np.array(cover_full, dtype=float)
    print("  → cached wetland mask + full land-cover raster.")

wetland_arr = np.array(wet_da.squeeze(), dtype=bool)
n_wet = int(wetland_arr.sum())
print(f"Wetland cells: {n_wet}/{wetland_arr.size} ({100*n_wet/wetland_arr.size:.1f}%) "
      f"= {n_wet*DEM_RES**2/1e6:.2f} km²")
assert n_wet > 0, "No wetland cells found"
assert wetland_arr.shape == grid.shape, "Wetland mask not aligned to DEM grid"

# ── Viz ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
if cover_grid is not None:
    im = axes[0].imshow(cover_grid, cmap="tab20", origin="upper")
    plt.colorbar(im, ax=axes[0], label="NLCD class code", fraction=0.046)
    axes[0].set_title(f"NLCD land cover — {NLCD_YEARS[0]}")
else:
    axes[0].imshow(hillshade(dem_arr0, res=DEM_RES), cmap="gray", origin="upper")
    axes[0].set_title("Hillshade (full cover raster not cached)")
axes[0].set_xticks([]); axes[0].set_yticks([])

axes[1].imshow(hillshade(dem_arr0, res=DEM_RES), cmap="gray", origin="upper")
axes[1].imshow(np.where(wetland_arr, 1.0, np.nan), cmap="summer", origin="upper")
axes[1].set_title(f"NLCD Wetlands (90+95) — {n_wet*DEM_RES**2/1e6:.2f} km²")
axes[1].set_xticks([]); axes[1].set_yticks([])

finish(fig, "07_wetland_mask.png", OUT_DIR)
print("Section 10 complete ✓")
""")

# ── CELL 11: split t1/t2 ──────────────────────────────────────────────────────
md("""\
## Section 11 — Split Overland vs Channel at Inflow Point `j`

`hand_idx_arr[i]` is the flat index of the receiving channel cell `j`. Then
`d2=D[j]`, `t2=T[j]`, `d1=D[i]-D[j]`, `t1=T[i]-T[j]`. We verify every located `j`
is genuinely a stream cell.

**Viz:** maps of `t1` (overland) and `t2` (channel) at wetland cells, the overland
fraction histogram, and a `t1` vs `t2` scatter.
""")
code("""\
has_j  = hand_idx_arr >= 0
j_rows = np.full(grid.shape, -1, dtype=np.int64)
j_cols = np.full(grid.shape, -1, dtype=np.int64)
jr, jc = np.unravel_index(hand_idx_arr[has_j], grid.shape)
j_rows[has_j] = jr; j_cols[has_j] = jc

d2 = np.full(grid.shape, np.nan); t2 = np.full(grid.shape, np.nan)
d2[has_j] = D_arr[jr, jc]; t2[has_j] = T_arr[jr, jc]
with np.errstate(invalid="ignore"):
    d1 = D_arr - d2; t1 = T_arr - t2
d1 = np.where(d1 < 0, 0.0, d1); t1 = np.where(t1 < 0, 0.0, t1)
d1 = np.where(stream_arr_bool, 0.0, d1); t1 = np.where(stream_arr_bool, 0.0, t1)
print(f"d1 {np.nanmin(d1):.0f}–{np.nanmax(d1):.0f} m | t1 {np.nanmin(t1)/3600:.2f}–{np.nanmax(t1)/3600:.2f} hr")
print(f"d2 {np.nanmin(d2):.0f}–{np.nanmax(d2):.0f} m | t2 {np.nanmin(t2)/3600:.2f}–{np.nanmax(t2)/3600:.2f} hr")

dem_arr = np.array(dem_da, dtype=float)
z_i = dem_arr; z_j = dem_arr[j_rows, j_cols]; z_s = dem_arr[r_s, c_s]
with np.errstate(divide="ignore", invalid="ignore"):
    s1 = np.where(d1 > 0, np.maximum((z_i - z_j)/d1, SLOPE_MIN), SLOPE_MIN)
    s2 = np.where(d2 > 0, np.maximum((z_j - z_s)/d2, SLOPE_MIN), SLOPE_MIN)

valid_mask = (wetland_arr & np.isfinite(T_arr) & (T_arr > 0)
              & np.isfinite(d1) & np.isfinite(d2))
# verify every located j is a real stream cell (the meaningful check)
assert stream_arr_bool[j_rows[valid_mask], j_cols[valid_mask]].all(), \\
    "Some receiving cells j are not stream cells — check HAND index / stream mask"
print("All receiving points j lie on the stream network ✓")

# ── Viz ──────────────────────────────────────────────────────────────────────
t1_wet = np.where(valid_mask, t1/3600.0, np.nan)
t2_wet = np.where(valid_mask, t2/3600.0, np.nan)
wet_t1 = t1[valid_mask]/3600.0; wet_t2 = t2[valid_mask]/3600.0
frac_overland = wet_t1 / (wet_t1 + wet_t2 + 1e-12)

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
vmax1 = np.nanpercentile(t1_wet[np.isfinite(t1_wet)], 98) if np.isfinite(t1_wet).any() else 1
im0 = axes[0,0].imshow(hillshade(dem_arr0, res=DEM_RES), cmap="gray", origin="upper")
sc0 = axes[0,0].imshow(t1_wet, cmap="cool", origin="upper", vmax=vmax1)
plt.colorbar(sc0, ax=axes[0,0], label="t₁ overland (hr)", fraction=0.046)
axes[0,0].set_title("Overland time t₁ at wetlands"); axes[0,0].set_xticks([]); axes[0,0].set_yticks([])

vmax2 = np.nanpercentile(t2_wet[np.isfinite(t2_wet)], 98) if np.isfinite(t2_wet).any() else 1
axes[0,1].imshow(hillshade(dem_arr0, res=DEM_RES), cmap="gray", origin="upper")
sc1 = axes[0,1].imshow(t2_wet, cmap="autumn_r", origin="upper", vmax=vmax2)
plt.colorbar(sc1, ax=axes[0,1], label="t₂ channel (hr)", fraction=0.046)
axes[0,1].set_title("Channel time t₂ at wetlands"); axes[0,1].set_xticks([]); axes[0,1].set_yticks([])

axes[1,0].hist(frac_overland, bins=40, color="steelblue", edgecolor="white", lw=0.3)
axes[1,0].set_xlabel("t₁ / (t₁ + t₂)"); axes[1,0].set_ylabel("Count")
axes[1,0].set_title("Overland fraction of travel time (wetland cells)")

axes[1,1].scatter(wet_t1[::5], wet_t2[::5], alpha=0.12, s=4, color="steelblue")
axes[1,1].set_xlabel("t₁ overland (hr)"); axes[1,1].set_ylabel("t₂ channel (hr)")
axes[1,1].set_title("Overland vs channel travel time")

finish(fig, "08a_t1_t2_decomp.png", OUT_DIR)
print("Section 11 complete ✓")
""")

# ── CELL 12: assemble table ───────────────────────────────────────────────────
md("""\
## Section 12 — Assemble the Per-Wetland-Pixel Table

One row per wetland cell with full geometry + travel-time attributes. Saved as
GeoParquet, CSV, and a `t_total` GeoTIFF (all into `./outputs/`).

**Viz:** distribution panel for d₁/t₁/d₂/t₂ plus the wetland `t_total` map.
""")
code("""\
final_mask = (valid_mask & (T_arr > 0))
wet_rows, wet_cols = np.where(final_mask)
aff = grid.affine
xs = aff.c + (wet_cols + 0.5)*aff.a
ys = aff.f + (wet_rows + 0.5)*aff.e

df = pd.DataFrame({
    "row": wet_rows, "col": wet_cols, "x_proj": xs, "y_proj": ys,
    "d1_m": d1[wet_rows, wet_cols], "s1": s1[wet_rows, wet_cols],
    "t1_s": t1[wet_rows, wet_cols], "t1_hr": t1[wet_rows, wet_cols]/3600.0,
    "d2_m": d2[wet_rows, wet_cols], "s2": s2[wet_rows, wet_cols],
    "t2_s": t2[wet_rows, wet_cols], "t2_hr": t2[wet_rows, wet_cols]/3600.0,
    "t_total_s": T_arr[wet_rows, wet_cols], "t_total_hr": T_arr[wet_rows, wet_cols]/3600.0,
})
gdf = gpd.GeoDataFrame(df, geometry=[Point(x, y) for x, y in zip(df.x_proj, df.y_proj)],
                       crs=PROJECTED_CRS)
print(f"Wetland pixel table: {len(df):,} rows")
print(df[["d1_m","t1_hr","d2_m","t2_hr","t_total_hr"]].describe().round(3))

yr = NLCD_YEARS[0]; out_stem = OUT_DIR / f"wetland_pixels_{STATION_ID}_{yr}"
gdf.to_parquet(str(out_stem)+".parquet")
gdf.drop(columns="geometry").to_csv(str(out_stem)+".csv", index=False)

t_total_raster = np.full(dem_arr.shape, np.nan, dtype=np.float32)
t_total_raster[wet_rows, wet_cols] = (T_arr[wet_rows, wet_cols]/3600.0).astype(np.float32)
with rasterio.open(OUT_DIR / f"t_total_{STATION_ID}_{yr}.tif", "w", driver="GTiff",
                   dtype="float32", width=grid.shape[1], height=grid.shape[0], count=1,
                   crs=rasterio.crs.CRS.from_string(PROJECTED_CRS), transform=grid.affine,
                   nodata=np.nan, compress="lzw") as dst:
    dst.write(t_total_raster[np.newaxis])
print(f"Saved: {out_stem}.parquet + .csv + t_total raster")

# ── Viz ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 9))
gs = fig.add_gridspec(2, 3)
specs = [("d1_m","Overland distance d₁ (m)"),("t1_hr","Overland time t₁ (hr)"),
         ("d2_m","Channel distance d₂ (m)"),("t2_hr","Channel time t₂ (hr)")]
pos = [(0,0),(0,1),(1,0),(1,1)]
for (col,label),(r,c) in zip(specs, pos):
    ax = fig.add_subplot(gs[r,c]); vals = df[col].dropna()
    ax.hist(vals, bins=50, color="steelblue", edgecolor="white", lw=0.2)
    ax.axvline(vals.median(), color="red", ls="--", lw=1.3, label=f"median {vals.median():.1f}")
    ax.set_xlabel(label); ax.set_ylabel("Count"); ax.legend(fontsize=8)

axm = fig.add_subplot(gs[:, 2])
axm.imshow(hillshade(dem_arr0, res=DEM_RES), cmap="gray", origin="upper")
tt = np.where(final_mask, T_arr/3600.0, np.nan)
imm = axm.imshow(tt, cmap="turbo", origin="upper",
                 vmax=np.nanpercentile(tt[np.isfinite(tt)], 98))
plt.colorbar(imm, ax=axm, label="t_total (hr)", fraction=0.046)
axm.plot(c_s, r_s, "w^", markersize=12, markeredgecolor="k")
axm.set_title("Wetland t_total to outlet"); axm.set_xticks([]); axm.set_yticks([])

fig.suptitle(f"Wetland travel-time — USGS {STATION_ID}, {yr}", fontsize=13)
finish(fig, "08b_histograms.png", OUT_DIR)
print("Section 12 complete ✓")
""")

# ── CELL 13: weighting W ──────────────────────────────────────────────────────
md("""\
## Section 13 — Weighting Kernel `W` (stub + comparison plot)

How to aggregate per-pixel travel times into a scalar `W` is a scientific choice.
Three candidate kernels are sketched; the researcher calibrates `τ`.

**Viz:** kernel weight-vs-time curves and the resulting `W` totals side by side.
""")
code("""\
def wetland_weight(df, kernel="inverse_time", tau_hr=None):
    area = df.get("area_m2", DEM_RES**2)
    t = df["t_total_hr"].values.astype(float)
    if kernel == "unweighted":   w = np.ones_like(t)
    elif kernel == "inverse_time": w = 1.0/np.maximum(t, 1e-6)
    elif kernel == "exponential":
        if tau_hr is None: raise ValueError("tau_hr required for exponential kernel")
        w = np.exp(-t/tau_hr)
    else: raise ValueError(f"Unknown kernel: {kernel!r}")
    return float(np.sum(area*w))

W_flat = wetland_weight(df, "unweighted")
W_inv  = wetland_weight(df, "inverse_time")
W_exp  = wetland_weight(df, "exponential", tau_hr=24.0)
print(f"Unweighted area:      W = {W_flat:.3e} m²")
print(f"Inverse-time:         W = {W_inv:.3e}")
print(f"Exponential (τ=24hr): W = {W_exp:.3e}")

# ── Viz ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
tgrid = np.linspace(0.01, max(df["t_total_hr"].max(), 1), 300)
axes[0].plot(tgrid, np.ones_like(tgrid), label="unweighted")
axes[0].plot(tgrid, 1.0/np.maximum(tgrid,1e-6), label="inverse-time")
for tau in (6, 24, 72):
    axes[0].plot(tgrid, np.exp(-tgrid/tau), label=f"exp τ={tau}h")
axes[0].set_xlabel("t_total (hr)"); axes[0].set_ylabel("weight f(t)")
axes[0].set_yscale("log"); axes[0].set_title("Kernel shapes"); axes[0].legend(fontsize=8)

names = ["unweighted","inverse_time","exp τ=6h","exp τ=24h","exp τ=72h"]
Ws = [wetland_weight(df,"unweighted"), wetland_weight(df,"inverse_time"),
      wetland_weight(df,"exponential",6.0), wetland_weight(df,"exponential",24.0),
      wetland_weight(df,"exponential",72.0)]
axes[1].bar(names, Ws, color="steelblue"); axes[1].set_yscale("log")
axes[1].set_ylabel("W (log scale)"); axes[1].set_title("W under each kernel")
axes[1].tick_params(axis="x", rotation=30)

finish(fig, "09_weight_kernels.png", OUT_DIR)
print("Section 13 (stub) complete ✓")
""")

# ── CELL 14: sanity check ─────────────────────────────────────────────────────
md("""\
## Section 14 (Optional) — Sanity Check: Euclidean vs Flow-Path Distance

`d1` (D8 overland flow-path) should be ≥ straight-line distance to the nearest
stream. The ratio `d1/EDT` measures overland-path sinuosity.

**Viz:** scatter vs the 1:1 line and the sinuosity-ratio histogram.
""")
code("""\
edt_m = scipy.ndimage.distance_transform_edt(~stream_arr_bool) * DEM_RES
compare_mask = (~stream_arr_bool & (d1 > 0) & np.isfinite(d1) & (edt_m > 0))
edt_s = edt_m[compare_mask]; d1_s = d1[compare_mask]; ratio = d1_s/edt_s
print(f"Comparison cells: {compare_mask.sum():,}")
print(f"d1/EDT ratio: median {np.median(ratio):.2f}  p95 {np.percentile(ratio,95):.2f}")
assert (ratio >= 0.99).mean() > 0.9, "Most d1 < Euclidean — check CRS/DEM res"

rng = np.random.default_rng(42)
idx = rng.choice(len(edt_s), size=min(30000, len(edt_s)), replace=False)
p99 = np.percentile(d1_s, 99)/1000

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
axes[0].scatter(edt_s[idx]/1000, d1_s[idx]/1000, alpha=0.05, s=3,
                color="steelblue", rasterized=True)
axes[0].plot([0,p99],[0,p99], "r--", lw=1.5, label="1:1")
axes[0].set_xlim(0,p99); axes[0].set_ylim(0,p99)
axes[0].set_xlabel("Euclidean distance to stream (km)")
axes[0].set_ylabel("Overland flow-path d₁ (km)")
axes[0].set_title("Euclidean vs D8 flow-path"); axes[0].legend()

axes[1].hist(np.clip(ratio[idx], 1.0, 4.0), bins=60, color="steelblue",
             edgecolor="white", lw=0.2)
axes[1].axvline(np.median(ratio), color="orange", lw=1.5,
                label=f"median {np.median(ratio):.2f}")
axes[1].axvline(1.0, color="red", ls="--", lw=1.5, label="1:1")
axes[1].set_xlabel("d₁ / Euclidean (clipped 4×)"); axes[1].set_ylabel("Count")
axes[1].set_title("Flow-path sinuosity"); axes[1].legend(fontsize=8)

finish(fig, "10_sanity_check.png", OUT_DIR)
print("Section 14 complete ✓")
print("\\nAll sections complete. Outputs in:", OUT_DIR.resolve())
""")

# ── write ─────────────────────────────────────────────────────────────────────
nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3 (wetland)", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.12"}
out_path = Path(__file__).parent / "notebook" / "wetland_travel_time_v3.ipynb"
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print(f"Wrote {out_path}  ({len(cells)} cells)")
