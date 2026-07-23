# Climate-Flood-Site

We study how investments in flood protection, such as levees, stormwater systems, and
nature-based solutions, affect not only flood risk, but also local property values, tax
revenues, and community financial resilience. We combine climate, economic, and public
finance data to explore how reducing flood risk can reshape local economies.

## Table of contents

- [`literature/`](#literature) — reports, the presented poster, and the external papers
  the methodology cites
- [`code/`](#code) — the three model versions (v1 → v2 → v3; v3 is current)
- [`data/`](#data) — committed results, mirrored to match `code/`
- [`archive/`](#archive) — superseded/unused scripts and data, plus a separate blocked
  research thread

## literature/
The actual write-ups: `v1_report.pdf` → `v2_report.pdf` → `v3_report.pdf` (full study
reports, v3 current), `v3_full_regression_tables.pdf`, `v3_poster.pdf`/`.pptx` (the
presented poster), plus the v1 variable-construction notes (`v1_variables.pdf`,
`v1_wetland_variable.pdf`, `v1_rainfall_variable_summary.pdf`). `.tex` source is included
alongside each PDF. `references/` holds the external papers the project's methodology is
built on (Gourevitch/Gold/Garcia and Taylor & Druckenmiller are the core cites) — see
[literature/references/README.md](literature/references/README.md).

## code/
The wetland–flood effectiveness event model, in three versions — see
[code/README.md](code/README.md) for what each version does and how they relate:

- **`wetland-flood-model-v1/`** — original pipeline (Eckhardt/peak-prominence event
  detection, IEMRE 12 km Stage IV rainfall). Includes `step1_new_model_handoff/`, the
  reference package documenting how Mark's RREDI event-detection method (see
  `reference_new_model/mark_model_pipeline.py`) was integrated forward into v2, and
  `y_variables_v2/`, the proposal that justified v3's expanded outcome set.
- **`wetland-flood-model-v2/`** — upgraded Steps 1–3: RREDI rolling-median/ratio event
  detector (Mark's method), 800 m PRISM rainfall, and a physical wetland-effectiveness
  variable `W`.
- **`wetland-flood-model-v3/`** — **current model.** Builds on v2's panel, fixes an
  antecedent-rainfall/event-rainfall double-counting bug, and expands from one headline
  outcome to 10 pre-registered mechanism outcomes (peak, shape, timing, volume, severity,
  flashiness, recession) across nested Models 1–4 with wild-cluster-bootstrap inference
  and a falsification suite (VIF, placebo, leave-one-gauge-out). The wetland peak-shaving
  interaction (`W×P`) matches its pre-registered sign in 9 of 10 outcomes.

Both **precipitation** and **discharge** inputs matter here at different resolutions:
event rainfall (`P_e`, antecedent wetness) comes from **PRISM 800 m rainfall aggregated
to daily totals**; **discharge is 15-minute** USGS instantaneous data, used both for
RREDI event detection and for v3's sub-daily severity outcome (Q99-excess volume).

All three pipelines' own raw inputs/caches (discharge, PRISM, rasters, NWI polygons) are
regenerable and gitignored — run the numbered scripts to rebuild them. See
`code/.gitignore`.

**Why v1 and v2 are still here, not archived:** v3 is the current model, but it doesn't
rebuild its panel from raw data on its own — `build_v3_panel.py` reads v2's already-built
`events_panel.csv` (committed, in `data/wetland-flood-model-v2/outputs/`) plus v2's cached
daily-PRISM files and v1's ("Mark's Model") cached 15-minute discharge files (neither
committed — regenerable caches, currently only in the team's Box folder). v1's and v2's
*code* is the only record of how those intermediate files were produced in the first place
(USGS discharge → v1's `build_events_flow.py`; PRISM grid → v2's `s2_build_events_rain.py`;
wetland `W` → v2's `s3a_build_T_v2.py`/`s3b_build_W_v2.py`). Re-running v3's actual
regressions (`fit_models_v3.py`, `falsification_v3.py`, `full_regression_tables.py`)
needs none of this — they only read v3's own committed panel. But a full from-scratch
rebuild of that panel needs v1 and v2's scripts to know how each upstream step was made,
so both stay.

## data/
Committed result outputs (event/panel CSVs, coefficient tables, QA plots) that each
pipeline version produced, kept separate from the scripts so it's obvious what's a
script vs. what's a result: `data/wetland-flood-model-v1/`, `-v2/`, `-v3/`.

## archive/
Two kinds of things, neither wired into the current (v3) model:
- Early data-pulling scripts for Census demographics/income, NHD watershed mapping,
  soil, terrain, and precipitation (4km/800m) — plus the GIS maps/charts/animations they
  actually produced.
- `annual-wetland-connectivity-project/` — a **separate, currently-blocked** research
  thread (different unit of analysis: annual gauge-year panel, travel-time-weighted
  wetland connectivity vs. causal flood response). Per its own handoff notes, the wetland
  effect isn't causally identified yet (blocked on getting annual land-cover data). Code +
  status docs only — the full data/outputs (~900 MB, regenerable) live in the team's Box
  folder, not in git.

See [archive/README.md](archive/README.md) for details.
