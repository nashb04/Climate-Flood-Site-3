# Step 1 New Model Handoff

这个文件夹是一个可交接的 Step 1 integration package。目标是把 `New Model/mark_model_pipeline.py` 里的 RREDI-style hydrograph event extraction 融合进现有 `build_events_flow.py`，同时不改变原来的 workflow contract。

## What changed

- `build_events_flow.py` 仍然默认运行原来的 Step 1：
  `python build_events_flow.py`
- 新增一个可选模式：
  `python build_events_flow.py --method rredi`
- RREDI/new-model 模式从 `usgs_discharge_15min_raw/yearly/*.csv.gz` 读取 New Model 已缓存的 USGS discharge 数据，不需要重新联网下载。
- RREDI/new-model 模式仍然写回原 workflow 需要的 Step 1 输出名和字段：
  `data/events_{site}.parquet`、`outputs/events_all.csv`、`outputs/step1_summary.csv`、`outputs/step1_events_qa_{site}.png`
- 当前机器缺 `pyarrow/fastparquet`，所以验证运行时写出了 `data/events_{site}.csv.gz` fallback。合并到正式环境时请安装 `pyarrow`，即可生成原 Step 2 直接读取的 parquet。

## Original workflow summary

1. `build_events_flow.py`
   Pulls 15-minute USGS instantaneous values for 9 MMSD gauges, aggregates to hourly, applies Eckhardt baseflow separation, detects quickflow peaks, and outputs one event table per gauge.

2. `build_events_rain.py`
   Reads `data/events_{site}.parquet`, computes basin-average Stage IV/IEMRE precipitation, and adds event rainfall fields such as `P_e_mm`, `V_e_m3`, `runoff_coeff`, and `api_30_mm`.

3. `build_wetland_nwi.py`
   Builds gauge-level wetland effectiveness and storage from NWI polygons, travel-time surfaces, channel proximity, Cowardin modifiers, and volume-area storage assumptions.

4. `build_controls.py`
   Builds gauge-level controls: drainage area, imperviousness, soils, available water storage, and slope variables.

5. `build_panel.py`
   Joins Step 1+2 event/rainfall metrics with Step 3 wetland metrics and Step 4 controls into `outputs/events_panel.csv`.

6. `fit_models.py`
   Fits the nested event-level wetland/flood response models and writes coefficient tables and plots.

7. `wild_bootstrap.py`
   Runs exact wild cluster bootstrap inference for the few-cluster gauge setting.

The important downstream contract is Step 2: it expects `data/events_{site}.parquet` with fields such as `site_no`, `t_start`, `t_peak`, `t_end`, `dur_hr`, `time_to_peak_hr`, `hydro_width_hr`, `Qp_cfs`, `quick_vol_m3`, `total_vol_m3`, `rb_flashiness`, `peak_month`, and `peak_year`.

## New Model step 1 logic

The integrated RREDI-style method comes from `reference_new_model/mark_model_pipeline.py`.

Instead of Eckhardt quickflow peak prominence, it:

- reads raw instantaneous discharge by year,
- computes a centered rolling median baseline,
- starts an event when `Q / baseline >= start_ratio`,
- ends after the ratio stays below `end_ratio` for `end_hold_hours`,
- merges events separated by less than `min_separation_hours`,
- keeps events whose peak ratio is at least `min_peak_ratio`,
- writes original Step 1 fields plus diagnostic columns prefixed with `rredi_`.

Default RREDI settings in the handoff script:

```text
baseline_window       = 7D
start_ratio           = 1.20
end_ratio             = 1.05
min_peak_ratio        = 1.30
min_duration_hours    = 6.0
end_hold_hours        = 6.0
min_separation_hours  = 12.0
max_gap_hours         = 2.0
min_data_coverage     = 0.80
```

## How to run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the original workflow, unchanged:

```bash
python build_events_flow.py
```

Run the New Model / RREDI-style Step 1:

```bash
python build_events_flow.py --method rredi
```

Run a specific site:

```bash
python build_events_flow.py --method rredi 04087000
```

Run the full New Model historical range:

```bash
python build_events_flow.py --method rredi --start-year 1983 --end-year 2023
```

Use only QC-passing RREDI events:

```bash
python build_events_flow.py --method rredi --rredi-qc-pass-only
```

If the raw discharge folder is somewhere else:

```bash
python build_events_flow.py --method rredi --rredi-discharge-dir /path/to/usgs_discharge_15min_raw
```

## Handoff folder contents

- `build_events_flow.py`
  Modified handoff script. Default behavior remains original. New behavior is behind `--method rredi`.

- `requirements.txt`
  Minimal packages for Step 1. `pyarrow` is included because downstream Step 2 expects parquet.

- `usgs_discharge_15min_raw/`
  New Model raw USGS discharge cache. Contains site metadata, download log, and yearly raw discharge files for 1983-2025.

- `800m_data/`
  New Model 800m precipitation data and summary. Not required by the modified Step 1 script, but included for traceability because it was part of the New Model package.

- `data/`
  Validation output from running `python build_events_flow.py --method rredi` for 2001-2023 in this environment. Because this environment lacks `pyarrow`, these are `.csv.gz` fallback files. In a pyarrow environment they will be `.parquet`.

- `outputs/`
  Validation `events_all.csv`, `step1_summary.csv`, and QA hydrograph plots from the integrated RREDI run.

- `reference_new_model/`
  Original New Model pipeline script and its outputs for traceability.

- `workflow_reference/`
  Copies of the existing downstream scripts and current baseline summary outputs, so reviewers can see the original Step 2-6 contract.

## Validation run included here

Command run from this folder:

```bash
python build_events_flow.py --method rredi
```

Result:

```text
Total events across 9 gauges: 7355
QC-pass events: 6858
Years loaded: 2001-2023 raw files, ending at 2024-01-01 local/UTC boundary
```

Per-gauge summary is in `outputs/step1_summary.csv`.

Important note: RREDI-style event counts are intentionally higher than the original Eckhardt/quickflow Step 1. The integration is designed so the detection method changes, but the downstream file names and core columns stay compatible.

## Merge notes for colleague

1. Replace or patch the project-level `build_events_flow.py` with the handoff version.
2. Keep default `--method original` behavior unless explicitly testing the new detector.
3. Put `usgs_discharge_15min_raw/` beside `build_events_flow.py`, or pass `--rredi-discharge-dir`.
4. Install `pyarrow` before running for downstream Step 2 compatibility.
5. Run:

```bash
python build_events_flow.py --method rredi
python build_events_rain.py
python build_wetland_nwi.py
python build_controls.py
python build_panel.py
python fit_models.py
python wild_bootstrap.py
```

6. If reading fallback CSV files manually, use `dtype={"site_no": str}` to preserve leading zeros.

## Compatibility caveats

- New Model raw discharge has discharge only, not gage height, so `peak_stage_ft` is `NaN` in RREDI mode.
- This integration keeps the original 9 MMSD gauges by default, even though New Model discovered 41 HUC gauges.
- RREDI timestamps are localized from USGS Central time and converted to UTC, matching the existing Step 2 expectation that event times are timezone-aware.
- Current validation output is CSV fallback only because this local Python environment does not have a parquet engine. The script itself still writes parquet first when `pyarrow` or `fastparquet` is available.
