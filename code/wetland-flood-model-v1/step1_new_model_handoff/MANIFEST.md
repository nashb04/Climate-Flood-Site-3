# Handoff Manifest

## Primary deliverables

- `README.md` - workflow summary, merge notes, run commands, validation notes.
- `build_events_flow.py` - modified Step 1 script with optional `--method rredi`.
- `requirements.txt` - minimal Step 1 Python dependencies.

## Required data for integrated RREDI Step 1

- `usgs_discharge_15min_raw/milwaukee_usgs_discharge_sites.csv`
- `usgs_discharge_15min_raw/download_log.csv`
- `usgs_discharge_15min_raw/yearly/usgs_discharge_15min_raw_1983.csv.gz`
- `usgs_discharge_15min_raw/yearly/usgs_discharge_15min_raw_1984.csv.gz`
- ...
- `usgs_discharge_15min_raw/yearly/usgs_discharge_15min_raw_2025.csv.gz`

Count: 43 yearly raw discharge files.

## Included validation outputs from integrated script

- `data/events_04086500.csv.gz`
- `data/events_04086600.csv.gz`
- `data/events_04087000.csv.gz`
- `data/events_04087030.csv.gz`
- `data/events_04087050.csv.gz`
- `data/events_04087070.csv.gz`
- `data/events_04087088.csv.gz`
- `data/events_04087119.csv.gz`
- `data/events_04087120.csv.gz`
- `outputs/events_all.csv`
- `outputs/step1_summary.csv`
- `outputs/step1_events_qa_{site}.png` for all 9 sites.

These are CSV fallback outputs because the local validation environment did not have `pyarrow` or `fastparquet`. With `pyarrow`, the same script writes `data/events_{site}.parquet`.

## New Model reference data

- `800m_data/milwaukee_precip_800m_1981.csv.gz`
- ...
- `800m_data/milwaukee_precip_800m_2023.csv.gz`
- `800m_data/precipitation_summary.csv`

Count: 43 precipitation yearly files.

## New Model reference outputs

- `reference_new_model/mark_model_pipeline.py`
- `reference_new_model/outputs/events/rredi_like_hydrograph_events_1981_2023.csv`
- `reference_new_model/outputs/events/rredi_like_hydrograph_events_1981_2023.csv.gz`
- `reference_new_model/outputs/plots/event_counts_by_year.png`
- `reference_new_model/outputs/plots/example_detected_hydrographs.png`
- `reference_new_model/outputs/plots/q_peak_vs_event_precip.png`
- `reference_new_model/outputs/templates/gage_covariates_template.csv`

## Existing workflow reference

- `workflow_reference/build_events_rain.py`
- `workflow_reference/build_wetland_nwi.py`
- `workflow_reference/build_controls.py`
- `workflow_reference/build_panel.py`
- `workflow_reference/fit_models.py`
- `workflow_reference/wild_bootstrap.py`
- `workflow_reference/report.tex`
- `workflow_reference/current_outputs/*.csv`

These files are included for review context only. The actual merge target is the project-level `build_events_flow.py`.
