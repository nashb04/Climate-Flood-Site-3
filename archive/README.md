# Archive

Scripts and data that predate, or were never wired into, the current
`code/wetland-flood-model-v3/` pipeline. Kept for reference in case any of it gets reused.

## ⭐ annual-wetland-connectivity-project/
A **separate research thread** with a different unit of analysis than the main model:
instead of (gauge × storm-event), it's an **annual gauge-year panel** across 26+ SE
Wisconsin gauges, testing whether a travel-time-weighted wetland connectivity metric (`W`)
reduces annual flood-peak runoff depth, causally.

**Status per its own handoff notes: not yet causally identified, and currently blocked.**
The annual wetland series (LCMAP) barely changes year-to-year within a gauge, so two-way
fixed-effects can't isolate `W`'s effect from cross-sectional confounding with urbanization
(corr(W, urban) = −0.96). Cross-sectional OLS finds W = −7.89 (p<0.01, wetlands lower flood
peaks) but that's not a causal estimate. The blocker: getting Annual NLCD (1985–2023)
wetland fraction per catchment to get real within-gauge variation — see `HANDOFF.md` §3 for
the attempted (and failed) data-access routes, and §9 for next steps if anyone picks it
back up.

Contents here are the pipeline scripts and status docs only (`README.md`, `HANDOFF.md`,
`code/`). The full data/outputs (~900 MB: DEM/NLCD/LCMAP rasters, catchment geometries,
cached API pulls, regression outputs) are regenerable and live only in the team's Box
folder (`Data+ Climate Resilience/Wetland/Wetland/`) — not committed here.

## code/
- `American_Community_Survey_income_demographic.py` — pulls ACS income/demographic data.
- `NHD_Milwauke.py` — NHD watershed/river mapping.
- `Soil Data Download & Visualization.py`
- `Terrain Data Download & Visualization.py`
- `Precipitation/` — 4km and 800m precipitation downloader/visualizer scripts. (The
  current model fetches its own 800m PRISM data directly — see
  `code/wetland-flood-model-v2/s2_build_events_rain.py`.)

## data/
- `CensusData/` — ACS tract-level income/demographic CSVs produced by the archived
  Census script.
- `NHD_Milwauke_output/` — watershed map PNG produced by the archived NHD script.
- `Soil_Data_visualization/` — the actual output of `Soil Data Download & Visualization.py`:
  5 basin maps (drainage class, hydrologic group, slope gradient, water table depth,
  available water storage), 5 summary charts, and 3 small CSVs backing them.
- `Terrain_Data_visualization/` — the actual output of `Terrain Data Download &
  Visualization.py`: elevation/slope/aspect/hillshade maps, an overview panel, 2 histogram
  charts, and 2 small CSVs backing them.
- `Precipitation_visualization/` — animated GIFs (`milwaukee_precip_1981_visualized_4km.gif`,
  `_800m.gif`) produced by the archived `Precipitation/precipitation_visualizer_*.py` scripts.
