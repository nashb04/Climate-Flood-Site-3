# Wetland–Flood Effectiveness Event Model (MMSD)

Event-based phenomenological model of how upstream wetland configuration relates to flood
response at nine long-record USGS gauges in the Milwaukee/Menomonee (MMSD) watersheds.
Unit of analysis is the **(gauge × storm event)**; every predictor is anchored to a published
method and public dataset.

Two versions are included:

- **`v1/`** — original pipeline: Eckhardt/peak-prominence event detection, IEMRE 12 km Stage IV
  rainfall, connectivity-weighted wetland fraction `W`.
- **`v2/`** — upgraded Steps 1–3 (Steps 4–6 unchanged): RREDI rolling-median/ratio event
  detector, **800 m PRISM** rainfall + corrected `P_eff` antecedent index, and a physical
  wetland effectiveness `W = Σ Aᵢ·Sᵢ·K(Tᵢⱼ)` (kinematic-celerity travel times, wetland
  roughness, connectivity term dropped, type-weighted glaciated storage).

## Pipeline (both versions)
1. `build_events_flow` — 15-min USGS discharge → hourly → event table (Y: peak, time-to-peak,
   hydrograph width, recession, R–B flashiness, …).
2. `build_events_rain` — basin-mean event rainfall → `P_e`, `V_e`, runoff coefficient, `P_eff`.
3. wetland `W` + storage `S_w` from NWI polygons + travel-time surfaces.
4. `build_controls` — drainage area, slope, impervious, soils.
5. `build_panel` — join to one (gauge×event) panel + collinearity diagnostic.
6. `fit_models` — nested Models 1–4 (mixed effects + gauge-clustered SE; placebo, LOGO).
7. `wild_bootstrap` — exact wild cluster bootstrap (few-cluster inference).

## Headline result
Wetlands shave flood **peaks** more strongly in larger storms — the peak-shaving interaction
`W×P` on log peak discharge survives placebo, leave-one-gauge-out, and wild cluster bootstrap:
**v1 β = −0.155 (p = 0.029); v2 β = −0.233 (p = 0.010)** — the finding is reproduced and
strengthened by the upgraded variables. The wetland *level* effect is not identifiable across
nine urbanization-confounded gauges.

## Reports
- `v2/report_v2.pdf` — full v2 study (methods, per-step tables, models, bootstrap).
- `v1/report.pdf`, `v1/variables.pdf`, `v1/wetland_v2/wetland_variable.pdf`,
  `v1/build_events_rain/rainfall_variable_summary.pdf` — variable-construction + literature notes.

## Data (not committed)
Large inputs/caches are **regenerable** and excluded via `.gitignore`: 15-min discharge,
800 m PRISM, DEM/NLCD rasters (`*.tif`), NWI polygons (`*.gpkg`), and all `*.parquet`/`*.npy`
caches. Run the numbered scripts (env: `conda env wetland`) to rebuild them.
Sources: USGS NWIS, PRISM, USFWS NWI, NLCD, POLARIS, gNATSGO, USGS 3DEP, IEM.
