#!/usr/bin/env python
"""
Mark's Model — Step 5: Assemble the (gauge x storm-event) modeling panel.

Joins:
  Step 1+2  events_rain_{site}.parquet  -> per-event Y metrics + rainfall (P_e,V_e,API,...)
  Step 3    panel_W_Sw.csv              -> gauge-level W variants + storage S_w (+ wet_frac,...)
  Step 4    panel_controls.csv          -> gauge-level DA, slopes, Ksat, AWS, sand
            imp_by_year.csv             -> impervious %, interpolated to each event's year

Adds derived modelling terms (Ve/Sw saturation ratio), season / snow flags, event id.
Output: outputs/events_panel.csv  (one row per gauge-event; raw, pre-standardisation)
        outputs/step5_collinearity.png  (gauge-level predictor correlation heatmap)
"""
from __future__ import annotations
import os, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data"); OUT = os.path.join(HERE, "outputs")

IMP_YEARS = [2001, 2011, 2021]

def run():
    # 1+2 — stack per-event files
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(DATA, "events_rain_*.parquet")))]
    ev = pd.concat(frames, ignore_index=True)
    ev["site_no"] = ev["site_no"].astype(str).str.zfill(8)

    # 3 — gauge-level wetland effectiveness + storage
    w = pd.read_csv(os.path.join(OUT, "panel_W_Sw.csv"), dtype={"site_no": str})
    w["site_no"] = w["site_no"].str.zfill(8)
    wcols = ["site_no", "wet_km2", "wet_frac", "near_frac", "meanT_wet_hr", "tau_hr",
             "W_area", "W_exp", "W_harm", "W_exp_noC", "W_exp_noM", "W_exp_grad",
             "Sw_va_m3", "Sw_va_depth_mm"]
    ev = ev.merge(w[wcols], on="site_no", how="left")

    # 4 — controls (static) + impervious interpolated to event year
    c = pd.read_csv(os.path.join(OUT, "panel_controls.csv"), dtype={"site_no": str})
    c["site_no"] = c["site_no"].str.zfill(8)
    ccols = ["site_no", "DA_km2", "basin_slope", "chan_slope", "ksat_log", "sand_pct", "aws_mm"]
    ev = ev.merge(c[ccols], on="site_no", how="left")
    imp = pd.read_csv(os.path.join(OUT, "imp_by_year.csv"), dtype={"site_no": str})
    imp["site_no"] = imp["site_no"].str.zfill(8)
    imp_map = {r.site_no: [r["2001"], r["2011"], r["2021"]] for _, r in imp.iterrows()}
    ev["imp_pct"] = [float(np.interp(y, IMP_YEARS, imp_map[s]))
                     for s, y in zip(ev["site_no"], ev["peak_year"])]

    # derived modelling terms
    ev["Ve_over_Sw"] = ev["V_e_m3"] / ev["Sw_va_m3"]
    ev["winter_flag"] = ev["peak_month"].isin([12, 1, 2, 3]).astype(int)
    ev["rc_gt1_flag"] = (ev["runoff_coeff"] > 1).astype(int)
    ev["log_Qp"] = np.log(ev["Qp_cfs"].clip(lower=1e-3))
    ev["log_Pe"] = np.log(ev["P_e_mm"].clip(lower=1e-3))
    ev["log_Ve"] = np.log(ev["V_e_m3"].clip(lower=1))
    ev["log_DA"] = np.log(ev["DA_km2"])
    ev["log_Sw"] = np.log(ev["Sw_va_m3"].clip(lower=1))
    ev["log_Wexp"] = np.log(ev["W_exp"].clip(lower=1))
    ev = ev.sort_values(["site_no", "t_peak"]).reset_index(drop=True)
    ev.insert(0, "event_id", ev["site_no"] + "_" + ev.groupby("site_no").cumcount().astype(str).str.zfill(4))

    # usable subset: real storm rainfall present
    ev["usable"] = ((ev["P_e_mm"] > 1) & np.isfinite(ev["runoff_coeff"])).astype(int)

    ev.to_csv(os.path.join(OUT, "events_panel.csv"), index=False)

    # ---- report ----
    print("=== STEP 5 PANEL ===")
    print(f"rows (gauge-events): {len(ev)} | usable (P_e>1 & finite RC): {int(ev.usable.sum())}")
    print("\nper-gauge event counts:")
    g = ev.groupby("site_no").agg(n=("event_id", "size"), usable=("usable", "sum"),
                                  wet_frac=("wet_frac", "first"), imp=("imp_pct", "mean"),
                                  DA=("DA_km2", "first"), medQp=("Qp_cfs", "median"),
                                  medPe=("P_e_mm", "median")).round(2)
    print(g.to_string())

    # gauge-level collinearity (the identification crux)
    gl = ev.groupby("site_no").agg(wet_frac=("wet_frac", "first"), W_exp=("W_exp", "first"),
                                   Sw=("Sw_va_m3", "first"), imp=("imp_pct", "mean"),
                                   log_DA=("log_DA", "first"), ksat=("ksat_log", "first"),
                                   sand=("sand_pct", "first"), chan_slope=("chan_slope", "first"))
    cor = gl.corr()
    print("\ngauge-level predictor correlations (n=9 gauges):")
    print(cor.round(2).to_string())
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cor, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cor))); ax.set_xticklabels(cor.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(cor))); ax.set_yticklabels(cor.columns, fontsize=8)
        for i in range(len(cor)):
            for j in range(len(cor)):
                ax.text(j, i, f"{cor.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7)
        plt.colorbar(im, fraction=0.046)
        ax.set_title("Step 5: gauge-level predictor collinearity (n=9)", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "step5_collinearity.png"), dpi=120)
        plt.close(fig)
    except Exception as e:
        print("plot skip:", e)

if __name__ == "__main__":
    run()
