"""Match user's annual runoff-DEPTH (Y_it) to our covariates, log-transform Y,
run 6 panel regressions (3 outcomes x {Pooled OLS, Two-way FE}), and write a
Stata-style output file + the merged panel CSV.
"""
import warnings
from pathlib import Path
import numpy as np, pandas as pd
from linearmodels.panel import PooledOLS, PanelOLS, compare
warnings.filterwarnings("ignore")
ROOT=Path("/Users/jared/Wetland"); OUT=ROOT/"outputs"/"sensors"; SENS=ROOT/"sensors"
YCSV="/Users/jared/Downloads/04_annual_runoff_metrics_1985_2025.csv"

def rd(p,**k):
    d=pd.read_csv(p,dtype={"site_no":str},**k); d["site_no"]=d["site_no"].str.zfill(8); return d
y=rd(YCSV); y["year"]=y["year"].astype(int)
for c in ["annual_max_runoff_depth_mm_day","annual_median_runoff_depth_mm_day",
          "annual_total_runoff_depth_mm","drainage_area_sqmi"]:
    y[c]=pd.to_numeric(y[c],errors="coerce")
y=y[y["complete_90pct"].astype(str).str.lower().isin(["true","1"])]          # >=90% complete

W = rd(OUT/"panel_W.csv")[["site_no","year","wet_frac","Wfrac_exp","near_frac"]]
pr= rd(OUT/"panel_precip_daymet.csv"); lc=rd(OUT/"panel_landcover_lcmap.csv")[["site_no","year","developed_frac"]]
dam=rd(SENS/"dam_by_sensor.csv")[["site_no","earliest_dam_year"]]
P=(y.merge(W,on=["site_no","year"]).merge(pr,on=["site_no","year"])
     .merge(lc,on=["site_no","year"]).merge(dam,on="site_no",how="left"))
P["Dam"]=((P["earliest_dam_year"].notna())&(P["year"]>=P["earliest_dam_year"])).astype(int)
# ── log-transform the outcomes (Y) ───────────────────────────────────────────
P["ln_flood_peak"]=np.log(P["annual_max_runoff_depth_mm_day"].clip(lower=1e-3))
P["ln_baseflow"]  =np.log(P["annual_median_runoff_depth_mm_day"].clip(lower=1e-3))
P["ln_annual_total"]=np.log(P["annual_total_runoff_depth_mm"].clip(lower=1e-1))
# regressors
P["W"]=P["Wfrac_exp"]; P["urban"]=P["developed_frac"]
P["ln_precip"]=np.log(P["precip_mm"].clip(lower=1))
P["ln_area"]=np.log(P["drainage_area_sqmi"].clip(lower=0.05))
P=P.dropna(subset=["ln_flood_peak","W","ln_precip","urban","ln_area"]).sort_values(["site_no","year"])

# ── save merged panel CSV ────────────────────────────────────────────────────
cols=["site_no","station_nm","year","drainage_area_sqmi",
      "annual_max_runoff_depth_mm_day","annual_median_runoff_depth_mm_day","annual_total_runoff_depth_mm",
      "ln_flood_peak","ln_baseflow","ln_annual_total",
      "Wfrac_exp","wet_frac","near_frac","precip_mm","ln_precip","developed_frac","ln_area","Dam"]
P[cols].to_csv(OUT/"panel_matched_depth.csv",index=False)

# ── 6 regressions: 3 outcomes x {Pooled, Two-way FE} ─────────────────────────
pp=P.set_index(["site_no","year"])
outcomes=[("ln_flood_peak","Flood-peak depth"),
          ("ln_baseflow","Baseflow depth"),
          ("ln_annual_total","Annual total")]
lines=[]
lines.append("="*78)
lines.append("PANEL REGRESSION — annual runoff DEPTH (log) on connectivity-weighted wetland")
lines.append(f"Unbalanced panel: {len(P)} gauge-years, {P.site_no.nunique()} gauges, "
             f"{int(P.year.min())}-{int(P.year.max())}.  SE clustered by gauge.")
lines.append("Outcomes are log(runoff depth). W = travel-time-weighted wetland fraction.")
lines.append("="*78)
for yv,lab in outcomes:
    pooled=PooledOLS.from_formula(f"{yv} ~ 1 + W + ln_precip + urban + Dam + ln_area", pp)\
        .fit(cov_type="clustered", cluster_entity=True)
    fe=PanelOLS.from_formula(f"{yv} ~ 1 + W + ln_precip + urban + EntityEffects + TimeEffects", pp)\
        .fit(cov_type="clustered", cluster_entity=True)
    cmp=compare({"Pooled OLS":pooled,"Two-way FE":fe}, stars=True, precision="std_errors")
    lines.append("")
    lines.append(f"OUTCOME: {lab}   [ {yv} ]")
    lines.append(str(cmp))
    lines.append("")
txt="\n".join(lines)
(OUT/"regression_output.txt").write_text(txt)
print(txt[:1500])
print("..."); print("\nWrote:")
print("  ", OUT/"regression_output.txt")
print("  ", OUT/"panel_matched_depth.csv", f"({len(P)} rows)")
