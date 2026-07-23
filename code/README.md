# Wetland–Flood Effectiveness Event Model (MMSD)

Event-based phenomenological model of how upstream wetland configuration relates to flood
response at nine long-record USGS gauges in the Milwaukee/Menomonee (MMSD) watersheds.
Unit of analysis is the **(gauge × storm event)**; every predictor is anchored to a published
method and public dataset.

Three versions are included:

- **`wetland-flood-model-v1/`** — original pipeline: Eckhardt/peak-prominence event
  detection, IEMRE 12 km Stage IV rainfall, connectivity-weighted wetland fraction `W`.
- **`wetland-flood-model-v2/`** — upgraded Steps 1–3 (Steps 4–6 unchanged): RREDI
  rolling-median/ratio event detector, **800 m PRISM** rainfall + corrected `P_eff`
  antecedent index, and a physical wetland effectiveness `W = Σ Aᵢ·Sᵢ·K(Tᵢⱼ)`
  (kinematic-celerity travel times, wetland roughness, connectivity term dropped,
  type-weighted glaciated storage).
- **`wetland-flood-model-v3/`** — **current model.** Builds directly on v2's panel and:
  (1) fixes a bug where v2's antecedent-rainfall control (`api_30_mm`) double-counted the
  event's own rainfall instead of measuring pre-event wetness only; (2) expands from one
  headline outcome to **10 pre-registered mechanism outcomes** — attenuation (peak,
  shape), delay/timing, storage/volume, cumulative and extreme severity, flashiness,
  relative peak, recession — each fit through nested Models 1–4 (basic wetland effect →
  peak-shaving interaction `W×P` → saturation `S` → nonlinear saturation `W×S`) with
  wild-cluster-bootstrap inference; (3) adds a falsification suite (VIF, placebo
  permutation across gauges, leave-one-gauge-out).

The v2 event detector implements Mark's RREDI method. The original standalone script he
wrote is preserved at
`wetland-flood-model-v1/step1_new_model_handoff/reference_new_model/mark_model_pipeline.py`,
along with `step1_new_model_handoff/README.md`, which documents exactly how that method was
folded into the pipeline that became v2. `wetland-flood-model-v1/y_variables_v2/` holds the
literature-grounded proposal (`PROPOSAL.md`) that justified the outcome set v3 later adopted.
`wetland-flood-model-v1/v2_prototype/` is where v2's wetland-effectiveness formula
(celerity travel time + wetland roughness) was first developed and calibrated, reusing v1's
cached DEM/routing grid, before it graduated into v2's own `s3a_build_T_v2.py`/
`s3b_build_W_v2.py`. Its outputs (`calibration.csv`, `T_v2_compare.csv`,
`models_v2_compare.csv`) are the old-vs-new comparison that justified building v2 — not v2's
actual results (those are in `data/wetland-flood-model-v2/`).

## Pipeline
1. `build_events_flow` (v1) / `s1_build_events_flow` (v2) — 15-min USGS discharge →
   hourly → event table (Y: peak, time-to-peak, hydrograph width, recession,
   R–B flashiness, …).
2. `build_events_rain` / `s2_build_events_rain` — basin-mean event rainfall → `P_e`,
   `V_e`, runoff coefficient, `P_eff`.
3. wetland `W` + storage `S_w` from NWI polygons + travel-time surfaces
   (`s3a_build_T_v2`, `s3b_build_W_v2`, `s3c_wetland_adapter` in v2).
4. `build_controls` / `s4_build_controls` — drainage area, slope, impervious, soils.
5. `build_panel` / `s5_build_panel` — join to one (gauge×event) panel + collinearity
   diagnostic.
6. `fit_models` / `s6_fit_models` — nested Models 1–4 (mixed effects + gauge-clustered
   SE; placebo, LOGO). `wild_bootstrap` / `s6_wild_bootstrap` — exact wild cluster
   bootstrap (few-cluster inference).
7. **v3 only:** `build_v3_panel` — rebuilds `api_30_mm` as a strictly-prior antecedent
   index and adds the Q99-excess-volume outcome from the 15-min hydrograph.
   `fit_models_v3` — nested M1–M4 across all 10 outcomes with wild-cluster bootstrap.
   `falsification_v3` — VIF / placebo / leave-one-gauge-out. `full_regression_tables` —
   assembles the publication-style tables.

## Precipitation vs. discharge resolution
**Event rainfall (`P_e`) and antecedent wetness are daily** — PRISM 800 m precipitation
aggregated to a daily total per catchment per day, summed/decayed over the event window
and prior days. **Discharge is 15-minute** USGS instantaneous data — used for RREDI event
detection in all versions, and directly (as a 15-min hydrograph) for v3's Q99-excess
severity outcome. Nothing in the pipeline uses sub-daily precipitation.

## Headline result
Wetlands shave flood **peaks** more strongly in larger storms — the peak-shaving
interaction `W×P` on log peak discharge survives placebo, leave-one-gauge-out, and wild
cluster bootstrap: **v1 β = −0.155 (p = 0.029); v2 β = −0.233 (p = 0.010)**. v3 confirms
this is not a fluke of one outcome variable: across the 10 pre-registered outcomes, `W×P`
matches its predicted sign in **9 of 10**, with wild-cluster-bootstrap p < 0.05 on peak
discharge, runoff coefficient, Q99-excess depth, R–B flashiness, and relative peak. The
wetland *level* effect alone is still not identifiable across nine urbanization-confounded
gauges — it's the *interaction with storm size* that's robust.

## Reports
See [../literature/](../literature/) for the full write-ups: `v1_report.pdf` →
`v2_report.pdf` → `v3_report.pdf` (current), `v3_full_regression_tables.pdf`, and
`v3_poster.pdf`/`.pptx` (the presented poster). `v1_variables.pdf`, `v1_wetland_variable.pdf`,
`v1_rainfall_variable_summary.pdf` cover the original pipeline's variable definitions.

## Result data
See [../data/](../data/) for the committed CSV/PNG outputs each version produced
(event tables, panels, coefficient tables, QA plots).

## Regenerating raw inputs (not committed)
Large inputs/caches are **regenerable** and excluded via `.gitignore`: 15-min discharge,
800 m PRISM, DEM/NLCD rasters (`*.tif`), NWI polygons (`*.gpkg`), and all `*.parquet`/`*.npy`
caches. Run the numbered scripts (env: `conda env wetland`) to rebuild them.
Sources: USGS NWIS, PRISM, USFWS NWI, NLCD, POLARIS, gNATSGO, USGS 3DEP, IEM.
