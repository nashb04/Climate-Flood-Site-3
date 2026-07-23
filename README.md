# Climate-Flood-Site

We study how investments in flood protection, such as levees, stormwater systems, and
nature-based solutions, affect not only flood risk, but also local property values, tax
revenues, and community financial resilience. We combine climate, economic, and public
finance data to explore how reducing flood risk can reshape local economies.

## Repo layout

```
literature/   Reports and variable-construction write-ups (PDF + LaTeX source)
code/         Model pipeline scripts
data/         Committed result outputs the pipelines produced
archive/      Older/unused scripts and data, kept for reference only
```

### literature/
The actual write-ups: `v1_report.pdf` / `v2_report.pdf` (full study reports), plus the
variable-construction notes (`v1_variables.pdf`, `v1_wetland_variable.pdf`,
`v1_rainfall_variable_summary.pdf`). `.tex` source is included alongside each PDF.

### code/
The wetland–flood effectiveness event model, in two versions — see
[code/README.md](code/README.md) for what each version does and how they relate:

- **`wetland-flood-model-v1/`** — original pipeline (Eckhardt/peak-prominence event
  detection, IEMRE 12 km Stage IV rainfall). Includes `step1_new_model_handoff/`, the
  reference package documenting how Mark's RREDI event-detection method (see
  `reference_new_model/mark_model_pipeline.py`) was integrated forward into v2.
- **`wetland-flood-model-v2/`** — **current model.** Upgraded Steps 1–3: RREDI
  rolling-median/ratio event detector (Mark's method), 800 m PRISM rainfall, and a
  physical wetland-effectiveness variable `W`. This is the pipeline behind the strongest
  headline result (β = −0.233, p = 0.010).

Both pipelines' own raw inputs/caches (discharge, PRISM, rasters, NWI polygons) are
regenerable and gitignored — run the numbered scripts to rebuild them. See
`code/.gitignore`.

### data/
Committed result outputs (event/panel CSVs, coefficient tables, QA plots) that each
pipeline version produced, kept separate from the scripts so it's obvious what's a
script vs. what's a result: `data/wetland-flood-model-v1/`, `data/wetland-flood-model-v2/`.

### archive/
Scripts and data not wired into the current (v2) model — early data-pulling scripts for
Census demographics/income, NHD watershed mapping, soil, terrain, and precipitation
(4km/800m). Kept for reference in case any of it gets reused later, but none of it feeds
the current pipeline. See [archive/README.md](archive/README.md).
