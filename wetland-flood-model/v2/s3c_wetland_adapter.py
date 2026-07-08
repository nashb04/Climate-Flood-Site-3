#!/usr/bin/env python
"""
Mark_model_v2 -- Step 3 adapter: map the v2 wetland panel to the unchanged downstream
contract file panel_W_Sw.csv (so build_panel.py / fit_models.py run verbatim).

The treatment W_exp is set to the new W_v2 (celerity T, C dropped, S_i = type x storage,
glaciated V-A); fit_models standardizes W, so the scale change is harmless. Static geometric
columns (wet_frac, near_frac) are carried through from build_W_v2's merge. Unused variants
(W_harm, W_exp_no*) are filled with v2 variants for schema completeness.
"""
import os, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "outputs")

v2 = pd.read_csv(os.path.join(OUT, "panel_W_v2.csv"), dtype={"site_no": str})
v2["site_no"] = v2["site_no"].str.zfill(8)
out = pd.DataFrame({
    "site_no":        v2["site_no"],
    "wet_km2":        v2["wet_km2"],
    "wet_frac":       v2["wet_frac"],
    "near_frac":      v2["near_frac"],
    "meanT_wet_hr":   v2["meanT_wet_hr"],
    "tau_hr":         v2["tau_v2_hr"],
    "W_area":         v2["W_v2_area"],
    "W_exp":          v2["W_v2"],            # <-- NEW primary treatment
    "W_harm":         v2["W_v2"],
    "W_exp_noC":      v2["W_v2_typeonly"],
    "W_exp_noM":      v2["W_v2_storonly"],
    "W_exp_grad":     v2["W_v2"],
    "Sw_va_m3":       v2["Sw_v2_m3"],        # <-- NEW storage
    "Sw_va_depth_mm": v2["Sw_v2_depth_mm"],
})
out.to_csv(os.path.join(OUT, "panel_W_Sw.csv"), index=False)
print(f"wrote panel_W_Sw.csv (v2 wetland in contract schema), {len(out)} gauges")
print(out[["site_no", "wet_km2", "W_exp", "Sw_va_m3", "tau_hr"]].to_string(index=False))
