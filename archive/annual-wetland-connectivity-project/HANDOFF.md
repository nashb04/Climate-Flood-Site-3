# Wetland → Streamflow Project — Handoff

**Working dir:** `/Users/jared/Wetland`  ·  **conda env:** `wetland` (`/Users/jared/miniforge3/envs/wetland/bin/python`)

---

## 1. Research question
Does **wetland connectivity** (a travel-time–weighted wetland metric `W`, where
near-stream wetlands count more) reduce **downstream flood response** (annual
flood-peak runoff depth), and is that effect **causal**?

`Y_it` (outcome) = annual runoff-depth metrics supplied by the user
(`/Users/jared/Downloads/04_annual_runoff_metrics_1985_2025.csv`: median / mean /
total / **max (flood peak)** / p95, mm).
`W_it` (treatment) = travel-time–weighted wetland fraction we compute.
Controls: precipitation, urbanization (developed fraction), basin area, dam dummy.

## 2. TL;DR status
- Full geospatial + panel pipeline works end-to-end. Current panel:
  **438 gauge-years, 26 gauges (SE Wisconsin / Milwaukee basin), 1985–2021, unbalanced.**
- **Cross-sectional OLS:** log flood-peak ~ `W` gives **W = −7.89, p<0.01**
  (direction = wetlands lower flood peaks). BUT it is **not causal** (see §3).
- **Two-way FE (entity+year):** `W` is **unidentified** (coef ≈ −541, SE 3749, p=0.89).

## 3. THE CORE OPEN PROBLEM (what to solve)
The wetland series has **no within-gauge time variation**, so a clean causal
(within / fixed-effects / first-difference) estimate is impossible:

- Our annual wetland source is **LCMAP** (USGS, Planetary Computer). LCMAP land
  cover is **temporally near-constant** by design (CCDC time-series fitting).
  Within-gauge SD of `W` ≈ **8e-6** → essentially zero.
- Diagnostics (`outputs/sensors/diagnostics_report.txt`) show the W coefficient is
  identified **100% cross-sectionally**, is **collinear with urbanization**
  (corr(W, urban) = **−0.96**, VIF[W]=19.5), and is **spec-fragile** (W swings
  −2.8 (ns) to −10.6 depending on controls).

**=> The single thing needed for a clean causal estimate: a wetland series that
actually changes year-to-year, i.e. ANNUAL NLCD (1985–2023), classes 90 (woody)
+ 95 (emergent herbaceous).** Then re-run the two-way FE / an event-study.

### Annual NLCD access — the blocker (tried, failed)
- `pygeohydro.nlcd_bygeom` only serves **epoch** years (2001…2021) via MRLC WMS.
- MRLC WMS exposes only Annual-NLCD **change-summary** layers, not per-year cover.
- MRLC public S3 COGs: `s3-us-west-2.amazonaws.com/mrlc/Annual_NLCD_LndCov_{yr}_CU_C1V0.tif`
  → **403**; virtual-host `mrlc.s3.us-west-2.amazonaws.com/...` returns a **42-byte stub**
  (not the real COG). `/vsis3` anonymous → AccessDenied. ScienceBase API → timeout.
- **Recommended routes for Codex to try:** (a) **AppEEARS API** (needs a free NASA
  **Earthdata** login — user will provide) to subset Annual NLCD by geometry+year;
  (b) **Google Earth Engine** (`USGS/NLCD_RELEASES/...` or the Annual NLCD asset);
  (c) find the correct MRLC/USGS Annual NLCD COG endpoint (the data is public; the
  URL pattern we used is wrong/stubbed).
- Output needed: per catchment per year (1985–2023), the wetland fraction; feed it
  into the W computation in place of LCMAP.

## 4. How to run
```bash
PY=/Users/jared/miniforge3/envs/wetland/bin/python
cd /Users/jared/Wetland
# the gauge list every script reads:
cat sensors/panel_sites.txt          # 57 USGS site_no (50 AOI DV gauges + 7 KK)
# pipeline order (master grid is cached in data/aoi_*):
$PY build_sensors.py                  # SKIP_FIG=1 for batch (catchments only)
$PY build_traveltime_W.py             # -> outputs/sensors/panel_W.csv  (W metrics)
$PY build_lcmap_yearly.py             # -> panel_landcover_lcmap.csv (wetland+developed)
$PY build_precip.py                   # -> panel_precip_daymet.csv
$PY build_dam_dummy.py                # -> sensors/dam_by_sensor.csv
$PY build_ols_output.py              # match user's Y, 6 descriptive OLS -> regression_output.txt + panel_matched_depth.csv
$PY build_twoway_fe.py                # two-way FE -> regression_twoway_fe.txt
$PY build_clean_causal.py             # main causal diagnostic/workflow -> clean_causal_report.txt + panel_causal_ready.csv
$PY build_diagnostics.py              # diagnostics_report.txt + 00_diagnostics.png
```

## 5. Data sources (and which official wrappers are BROKEN here)
| What | Source | Note |
|---|---|---|
| DEM (10 m) | `py3dep.get_dem` (3DEP) | OK |
| Basin polygons | `pynhd.NLDI.get_basins` | OK. **`getfeature_byid` BROKEN** (pygeoutils `.crs` error) → gauge coords from USGS site service |
| Land cover (epoch) | `pygeohydro.nlcd_bygeom` | OK but epoch-only |
| Land cover (annual) | **LCMAP** via `planetary-computer`+`pystac-client` (`usgs-lcmap-conus-v13`) | OK but **temporally flat** (the problem). Annual NLCD = unsolved |
| Precip | **Daymet single-pixel REST** (`daymet.ornl.gov/single-pixel/api`) | `pydaymet` THREDDS endpoint BROKEN → use the REST single-pixel call |
| Dams | **NID national CSV** (`nid.sec.usace.army.mil/api/nation/csv`) | `pygeohydro.NID` BROKEN (schema mismatch) → parse CSV directly |
| Discharge / runoff | USGS NWIS `waterservices` + user's runoff CSV | OK |

## 6. Pipeline scripts (I/O)
- `sensors/panel_sites.txt` — central gauge list; **every script reads it**.
- `build_sensors.py` — master AOI grid (DEM→WhiteBox breach→D8 routing→HAND); per-gauge
  area-matched snap + catchment → `sensors/catchment_{sid}.gpkg`, `sensor_summary.csv`.
  `SKIP_FIG=1` env = batch (no per-gauge figures/NLDI).
- `build_traveltime_W.py` — velocity grid (Manning channel / TR-55 overland) → per-cell
  travel time τ → `distance_to_outlet` (per gauge) → travel-time field `T` → kernel-weight
  wetland (`Wfrac_exp` exp kernel, `Wfrac_inv`, `near_frac`) → `outputs/sensors/panel_W.csv`,
  `data/T_{sid}.tif`, `W_{sid}.png`. **Wetland comes from LCMAP here.**
- `build_lcmap_yearly.py` — LCMAP per-year mosaics (`data/lcmap/lcpri_{yr}.tif`); zonal per
  catchment → `panel_landcover_lcmap.csv` (wetland_frac, **developed_frac**).
- `build_precip.py` — Daymet annual precip per catchment centroid → `panel_precip_daymet.csv`.
- `build_flow_metrics.py` — USGS daily Q → peak/flashiness/Q10 → `panel_flowmetrics.csv`
  (NOTE: the matched regression uses the USER's runoff CSV as Y, not this).
- `build_confounders.py` / `build_dam_dummy.py` — NID dams in AOI → `dams_aoi.gpkg`,
  `dam_by_sensor.csv` (Dam dummy = catchment contains a dam, earliest_dam_year).
- `build_kk_extra.py` — Kinnickinnic/Wilson-Park cluster (7 gauges S of AOI) on a **separate
  small grid** (`data/kk_*`, `data/lcmap_kk/`); appends to panel_W + panel_landcover.
- `build_ols_output.py` — joins user runoff Y + W + precip + developed + dam → 6 OLS
  (3 outcomes × raw/log Y), Stata-style → `regression_output.txt`, `panel_matched_depth.csv`.
  **Descriptive only; not causal.**
- `build_twoway_fe.py` — entity+year FE → `regression_twoway_fe.txt`.
- `build_clean_causal.py` — the main causal workflow: descriptive OLS benchmark,
  contemporaneous TWFE, lagged TWFE, first differences, and event-study readiness
  checks → `clean_causal_report.txt`, `panel_causal_ready.csv`.
- `build_diagnostics.py` — within/between, VIF, influence, leave-one-out, spec-sensitivity,
  between regression → `diagnostics_report.txt`, `00_diagnostics.png`.

## 7. Methodology details & GOTCHAS already fixed (don't re-break)
- **pysheds + NumPy 2:** add `if not hasattr(np,"in1d"): np.in1d=np.isin` before importing pysheds.
- **DEM conditioning:** plain pysheds depression-fill leaves this flat glacial basin's drainage
  FRAGMENTED (max accumulation 108 km² for an 1809 km² basin). **Use WhiteBox
  `breach_depressions_least_cost(dist=2000, fill=True)`** → integrates the network (max acc ≈ basin).
- **Catchment / outlet routing:** pysheds `grid.catchment(...)` and `distance_to_outlet(...)`
  MUST use **`xytype="index"` with (col,row)** — the coordinate round-trip mis-rounds by one
  cell and returns an empty/off-channel catchment.
- **Gauge snapping:** snap to the stream cell whose flow **accumulation matches the NWIS
  drainage area** (`snap_area`), not nearest-stream (Cedar Creek gauge sits 2.6 km off its
  mapped main stem). Falls back to max-accumulation when area unknown.
- **Wetland mask NaN bug:** read the wetland raster as float and compare `== 1.0`; do NOT cast
  to bool (NaN→True floods the mask).
- **`snap_to_mask` needs a pysheds `Raster`**, not an ndarray.
- **Travel-time `W`:** absolute T is uncalibrated (flat terrain + `SLOPE_MIN` floor inflate it
  to 100s of hours); kernel τ is set to each catchment's **median T** (adaptive). `Wfrac_exp`
  is a connectivity-weighted wetland FRACTION (0–1), comparable across basins.

## 8. Current results (context)
- Cross-sectional OLS, log flood-peak: `W = −7.89***`, precip `+`, urban `+`, area `−`.
- Two-way FE: `W` unidentified (within-SD ≈ 0); precip IS identified within-gauge (`+`, p<0.01)
  → the FE machinery works, only `W` lacks within variation.
- Diagnostics: corr(W,urban) = −0.96; VIF[W]=19.5; W coef fragile (−2.8 ns … −10.6);
  pure-between regression (n=26) W = −6.87 (p=0.002) ⇒ identification is cross-sectional.

## 9. NEXT TASKS (prioritized, for Codex)
1. **[BLOCKER] Get Annual NLCD 1985–2023 wetland fraction per catchment** (see §3 routes).
   Swap it into `build_traveltime_W.py` / `build_lcmap_yearly.py` (replace the LCMAP wetland
   read) so `W` and wetland_frac vary year-to-year. Recompute `panel_W.csv`, `panel_landcover`.
2. **Re-run `build_clean_causal.py`** — with real ΔW, the TWFE / lagged TWFE / first-difference
   sections should now identify the wetland effect (entity FE remove the W–urban cross-sectional
   collinearity). `build_twoway_fe.py` remains as a compact legacy FE table.
3. **Cleanest causal:** event-study / DiD around discrete wetland-change events (basins that
   lost/gained wetland in a given year vs controls).
4. **Break the W–urban collinearity:** expand to a REGIONAL sample of independent, mostly-rural
   gauged catchments spanning a wetland gradient at low urbanization (current sample: wetland-poor
   = urban). The FD/FE design uses any gauge with ≥2 consecutive years.
5. (Optional) calibrate the travel-time velocity model so absolute T / kernel τ are physical;
   small-cluster inference → wild-cluster bootstrap.

## 10. File map
```
sensors/        panel_sites.txt, sensors_dv_gauges.{gpkg,csv}, catchment_{sid}.gpkg,
                dam_by_sensor.csv, dams_aoi.gpkg, aoi_watershed.gpkg
data/           aoi_dem_10m[_cond].tif, aoi_acc.npy, aoi_hand_idx.npy, lcmap/, T_{sid}.tif,
                kk_*, lcmap_kk/, nid_full.csv, dailyQ_{sid}.parquet
outputs/sensors/ panel_W.csv, panel_landcover_lcmap.csv, panel_precip_daymet.csv,
                panel_matched_depth.csv  (<-- the analysis panel: Y + W + controls),
                regression_output.txt, regression_twoway_fe.txt,
                diagnostics_report.txt, 00_diagnostics.png, W_{sid}.png, sensor_{sid}.png
build_*.py      pipeline (see §6);  ppt/Wetland_Workflow.pptx (slide deck)
README.md       general project README;  notebook/wetland_travel_time_v3.ipynb (single-gauge demo)
```

**Analysis-ready file:** `outputs/sensors/panel_matched_depth.csv`
(columns: site_no, year, area, peak/base/total + ln_*, Wfrac_exp, wet_frac, near_frac,
precip, developed_frac, Dam). Replace `Wfrac_exp` with an Annual-NLCD-based W and re-run.
