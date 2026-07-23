# Wetland → Stream Travel-Time Pipeline (v3, visual edition)

Computes per-pixel travel times from every upstream grid cell to a USGS stream
gauge, flags wetland cells (NLCD), and decomposes each into overland (`t1`) and
channel (`t2`) components — with rich visualisations at every step.

## Folder layout

```
Wetland/
├── build_notebook.py                  # regenerates the notebook from source
├── notebook/
│   └── wetland_travel_time_v3.ipynb   # the executed, enhanced notebook
├── data/                              # ← ALL downloads cached here, ONCE
│   ├── basin_*.gpkg, gauge_*.gpkg     #   NLDI basin + gauge
│   ├── stage_*.parquet                #   USGS gage height (param 00065)
│   ├── dem_*.tif                      #   3DEP DEM (10 m, UTM)
│   ├── nlcd_*.tif                     #   NLCD land cover + wetland mask
│   └── *.npy                          #   cached intermediate grids
├── outputs/                           # ← figures + result tables
│   ├── 01_basin_stage.png … 10_sanity_check.png
│   ├── wetland_pixels_*.parquet / .csv
│   └── t_total_*.tif
└── run.sh                             # one-command rebuild + execute
```

The `data/` folder is the single download cache. After the first run every web
fetch is skipped — re-running is instant and fully offline.

## DEM conditioning note

Section 6 uses **WhiteBox `BreachDepressionsLeastCost`** rather than pysheds
`fill_depressions`. On the low-relief Milwaukee basin pysheds left the drainage
fragmented (max accumulation ~108 km² for an 1809 km² basin, so the gauge
captured only ~31 km²). Breaching integrates the network — contributing area to
the gauge is now ~1738 km², covering the whole basin.

> Travel times then span the full basin (median ~500 hr) — these are *relative*
> kinematic estimates with floored slopes (`SLOPE_MIN`) and uniform channel
> hydraulics (`N_CHANNEL`, `R_CHANNEL`); tune those for absolute magnitudes. The
> spatial pattern is what feeds the wetland weighting `W`.

## No API keys required

All data sources are public and anonymous:
- **NLDI** (basin delineation) — `pynhd`
- **3DEP** (DEM) — `py3dep`
- **NLCD** (land cover) — `pygeohydro`
- **USGS waterservices** (gage height) — REST

## Environment

A dedicated conda env `wetland` was created (does not touch your `research` /
`soil-downloader` envs):

```bash
mamba create -n wetland -c conda-forge python=3.12 numpy pandas geopandas \
  rasterio rioxarray xarray matplotlib scipy shapely pysheds pygeohydro \
  py3dep pynhd whitebox pyarrow notebook nbconvert nbformat ipykernel requests
```

A Jupyter kernel "Python 3 (wetland)" is registered.

## Run it

```bash
./run.sh                 # rebuild notebook from build_notebook.py + execute
# or open interactively:
conda activate wetland
jupyter notebook notebook/wetland_travel_time_v3.ipynb
```

## Switch gauge / settings

Edit the **Section 3 — Configuration** cell:

```python
STATION_ID    = "04087000"   # any USGS station with stage data
PROJECTED_CRS = "EPSG:32616" # match the basin's UTM zone
DEM_RES       = 10           # metres (30 = faster for big basins)
NLCD_YEARS    = [2021]       # [2001, 2021] for change analysis (ΔW)
ACC_THRESHOLD = 5000         # tune until the stream network looks right
```

After changing `STATION_ID`, delete the matching files in `data/` to force a
fresh download for the new basin.

## What each figure shows

| File | Step |
|---|---|
| `01_basin_stage.png` | Basin polygon, location context, stage hydrograph |
| `02_dem.png` | DEM, hillshade relief, elevation histogram |
| `03_flow_grids.png` | WhiteBox breach-carve depth, D8 direction, log-accumulation |
| `04_streams_hand.png` | Stream network + snapped outlet (with zoom), HAND |
| `05_velocity_tau.png` | Slope, velocity, per-cell τ, velocity distributions |
| `06_travel_time_surface.png` | Flow-path distance, travel time + iso-contours, histogram |
| `07_wetland_mask.png` | NLCD land cover + wetland mask over hillshade |
| `08a_t1_t2_decomp.png` | Overland/channel time maps, fraction + scatter |
| `08b_histograms.png` | d₁/t₁/d₂/t₂ distributions + wetland t_total map |
| `09_weight_kernels.png` | Candidate W kernels + resulting W totals |
| `10_sanity_check.png` | Euclidean vs flow-path distance, sinuosity |

---

## Multi-sensor pipeline (`build_sensors.py`)

Processes every long-record USGS gauge inside an AOI watershed polygon
(`globalwatershed.shp`); for each it maps the upstream watershed + river network
+ NLCD wetlands + wetland→stream flow direction.

**Architecture:** the AOI DEM is conditioned (WhiteBox) and routed **once**; each
sensor is snapped to the network by *matching its NWIS reported drainage area*
(robust against grabbing a nearby tributary), the catchment is delineated with
`grid.catchment(..., xytype="index")`, and streams/wetlands are clipped to it.
Re-runs reuse the cached master grid; only the NLCD year changes for the planned
1985–2024 loop (`BASE_YEAR` in the config block). `python build_sensors.py`.

**Sensor enumeration** (`sensors/`):
- `sensors_all_stream_sites.{gpkg,csv}` — all 218 stream sites in the AOI
- `sensors_dv_gauges.{gpkg,csv}` — 50 daily-value gauges + period of record + drainage area
- `aoi_watershed.gpkg` — the AOI polygon
- `catchment_{site}.gpkg` — per-sensor upstream watershed polygon (DEM-delineated)
- `basin_{site}.gpkg` — per-sensor NLDI basin (for comparison)

**Outputs** (`outputs/sensors/`):
- `00_overview_all_sensors.png` — all gauges + nested catchments on the AOI
- `sensor_{site}.png` — 3-panel: watershed overview · D8 flow field (zoom) · wetland→stream connectivity (zoom)
- `sensor_summary.csv` — catchment km², NWIS area, % error, snap distance, wetland km²/%

Catchment areas validate to within ~5% of NWIS reported drainage areas.

### Toward 1985–2024 yearly wetlands
Classic NLCD (this `pygeohydro`) serves epoch years via MRLC WMS. For **annual**
1985–2023 land cover, use the **Annual NLCD** product (MRLC, 2024 release). The
routing/catchments are computed once and stay fixed; only the wetland mask is
re-fetched per year, then `wetland_km2` / `W` recomputed per sensor per year.

## Wetland-change → river-response study (scripts)

| Script | Produces |
|---|---|
| `build_confounders.py` | Dams (NID) per catchment; `confounders_dams_*.csv`, `00_confounders_dams.png` |
| `build_lcmap_yearly.py` | LCMAP yearly wetland/developed per catchment 1985-2021; `panel_landcover_lcmap.csv` |
| `build_precip.py` | Daymet annual precip per catchment (single-pixel REST); `panel_precip_daymet.csv` |
| `build_flow_metrics.py` | Flow-TIMING outcomes (peak, R-B flashiness, baseflow Q10) from daily Q; `panel_flowmetrics.csv` |
| `build_traveltime_W.py` | **Travel-time field T per gauge + connectivity-weighted wetland W** (adaptive kernel); `panel_W.csv`, `W_{site}.png`, `data/T_{site}.tif` |
| `build_clean_causal.py` | Clean causal workflow: descriptive OLS vs two-way FE, lagged FE, first differences, and event-study readiness; `panel_causal_ready.csv`, `clean_causal_report.txt` |
| `assemble_panel.py` | Panel + PanelOLS regression; `panel_master.csv`, `00_panel_timeseries.png` |
| `analyze_method.py` | W-vs-area comparison & confound partials; `method_summary.csv`, `00_method_W_vs_area.png`, `00_confound_diagnosis.png` |

**Design notes (important):**
- Outcome must be flow **timing** (flashiness / specific peak / baseflow), NOT annual mean discharge — wetlands change timing, not volume.
- The wetland metric is the **travel-time-weighted W** (near-stream wetlands weighted higher), not raw area. Absolute travel times are uncalibrated (SLOPE_MIN-inflated) → kernel τ is set adaptively to the catchment median travel time; calibrate velocities before trusting absolute W.
- The 9 gauges here are **nested** (2× area overcount, ~2 independent systems) and wetland is **collinear with urbanisation** (r=-0.93) → this basin is a METHOD DEMO, not a valid hypothesis test. A real test needs a regional sample of independent gauged catchments spanning a wetland gradient + within-gauge ΔW (Annual NLCD).
- Treat pooled OLS as **descriptive only**. The causal estimand should come from
  within-gauge variation: two-way FE, lagged FE, first differences, or an
  event-study/DiD around real wetland-change events. With the current LCMAP W,
  `build_clean_causal.py` correctly reports that W is not causally identified.
