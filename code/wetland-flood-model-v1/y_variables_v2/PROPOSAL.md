# Outcome (Y) expansion — proposal (NOT yet run)

Isolated workspace. Reads the existing v2 panel read-only; writes only inside `y_variables_v2/`.
Nothing in the current workflow is modified.

**Panel to use:** `Mark_model_v2/outputs/events_panel.csv` — the v2 RREDI events (7-day rolling-median
detector), **5,907 usable gauge-events**, 2001–2023, with wetland `W`, storage `S_w`, and controls
already joined. This is the panel the README documents.

---

## 1. What we already have (by mechanism)

Most of the discussion-note outcome list is **already computed** in the v2 panel:

| Mechanism (discussion note) | Canonical signature | Column present? |
|---|---|---|
| Attenuation — peak | peak discharge | `log_Qp`, `Qp_quick_cfs`, `rredi_peak_ratio` ✅ |
| Attenuation — shape | flashiness / width | `rb_flashiness`, `hydro_width_hr` ✅ |
| Delay — timing | time to peak / lag | `time_to_peak_hr`, `recession_time_hr` ✅ |
| Storage — volume | event runoff ratio / event volume | `runoff_coeff`, `quick_vol_m3`, `total_vol_m3` ✅ |
| Release | recession rate | `recession_k_per_hr` ✅ |
| Extreme persistence | time above threshold | `dur_above_q99_hr` ✅ |

The RREDI source CSV also carries the **integrated volumes** (`total_volume_cfs_seconds`,
`total_excess_volume_cfs_seconds` → convert ×0.0283 to m³), so the volume Riemann-sums exist too.

**Genuinely missing:** only the **Q99-excess *volume*** (magnitude × time *above* the extreme
threshold) — we have `dur_above_q99_hr` (time only) but not the integral.

## 2. What the literature says (so we don't over-invent)

- McMillan (2020) *Linking hydrologic signatures to hydrologic processes* and McMillan et al. (2022,
  WRR) confirm the **canonical event signatures are runoff ratio, recession rate, and flashiness** —
  i.e. the ones we already have. The field notes **few clean event-scale signatures exist**; there is
  no exotic "better" signature to chase.
- McMillan et al. (2022) *When good signatures go bad* (HP) warns that **flashiness is
  sampling-interval and event-boundary sensitive** — matching the README's own caveat. ⇒ we should
  (a) demote flashiness to a caveated robustness outcome, and (b) prefer signatures that are less
  interval-fragile.
- Blume et al. (2007) establish the **event runoff coefficient** as the storage/infiltration signature.

**Two literature-motivated *improvements* over what we have:**
1. **Cross-basin normalization.** Our 9 gauges span 21–1751 km². Absolute Y (peak, volume) is
   dominated by basin size. Use **area-normalized** versions — specific peak (Qp/DA) and runoff
   **depth (mm)** — for cross-gauge comparability (standard in USGS regional flood regression).
2. **A more robust shape signature than flashiness: "peakedness"** = Qp / (event volume) (or
   Qp·dur/V). It directly measures how concentrated the hydrograph is (attenuation flattens it),
   and is far less interval-sensitive than flashiness.

## 3. Proposed canonical Y set (mechanism-complete, most already present)

Each outcome = one distinct mechanism, with a **pre-committed** predicted `W×P` sign (set before
looking at results). Six outcomes, not fifteen.

| # | Mechanism | Outcome `y` | Status | Predicted `W×P` |
|---|---|---|---|---|
| 1 | Attenuation (peak) | `log_Qp` *(control DA)* | ✅ present | − |
| 2 | Attenuation (shape) | **`log_peakedness` = log(Qp/quick_vol)** | ★ new (cheap) | − |
| 3 | Delay (timing) | `time_to_peak_hr` | ✅ present *(internal-lag caveat)* | + |
| 4 | Storage (volume) | `runoff_coeff` **and** `log_quick_depth_mm` | ✅ present + transform | ~0 / weak − |
| 5 | Cumulative severity | `log_quickflow_riemann_m3` (or depth) | ✅ transform of `quick_vol_m3` | − |
| 6 | **Extreme severity** | **`log1p_q99_excess_depth`** | ★ new integral | − |

Robustness / demoted: `rb_flashiness` (interval caveat), `rredi_peak_ratio`, `recession_k_per_hr`,
`dur_above_q99_hr` — reported but not headline.

**Explicitly dropped** (redundant, would be p-hacking): the 5 other peak synonyms and the 7 volume
variants — one canonical per mechanism only.

## 4. Genuinely new computation (only two things)

1. **`q99_excess` Riemann-sum volume/depth** — integrate `max(Q − Q99, 0)` over each event window
   using the cached 15-min discharge; convert to m³ and to depth (÷ DA). `log1p` for the many zeros.
2. **`peakedness`** = `Qp_cfs / quick_vol_m3` (units 1/s·... → treat as index; log). Trivial.

Everything else is an add/transform of existing columns.

## 5. Anti-p-hacking discipline (write into Methods)
1. Mechanism → sign map above is **pre-registered** (fixed before estimation).
2. **Report all six** (including nulls / wrong-sign) — the "weak volume effect" null *supports* the
   discussion-note hypothesis and is a feature, not a failure.
3. Each `y` gets the **same** Models 1–4 + placebo + LOGO + wild cluster bootstrap.
4. No α-correction: these are pre-specified *distinct mechanisms* with directional hypotheses, not
   one hypothesis tested many ways. State this explicitly.

## 6. Deliverable when we run (later, on your go)
- `build_y_expanded.py` — compute the 2 new Y + transforms, merge into a local copy of the v2 panel.
- `fit_y_matrix.py` — Models 1–4 + WCB per outcome → one **mechanism matrix** table
  (Y × predicted sign × β(W×P) × WCB p × LOGO range × n) + a forest figure.
- Full report. All inside `y_variables_v2/`.

**Status: proposal only. Not run.** Awaiting sign-off on the six-outcome set and the panel choice.
