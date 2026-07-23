# Archive

Scripts and data that predate, or were never wired into, the current
`code/wetland-flood-model-v2/` pipeline. Kept for reference in case any of it gets reused.

## code/
- `American_Community_Survey_income_demographic.py` — pulls ACS income/demographic data.
- `NHD_Milwauke.py` — NHD watershed/river mapping.
- `Soil Data Download & Visualization.py`
- `Terrain Data Download & Visualization.py`
- `Precipitation/` — 4km and 800m precipitation downloader/visualizer scripts. (The
  current model, v2, fetches its own 800m PRISM data directly — see
  `code/wetland-flood-model-v2/s2_build_events_rain.py`.)

## data/
- `CensusData/` — ACS tract-level income/demographic CSVs produced by the archived
  Census script.
- `NHD_Milwauke_output/` — watershed map PNG produced by the archived NHD script.
