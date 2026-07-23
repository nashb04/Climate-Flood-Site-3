"""Match the user's annual runoff-DEPTH metrics (their Y_it) to our covariate
panel (W, precip, developed/urban, dam) by site_no + year, write the merged
panel, and run the panel regression.

Outcomes (runoff depth, mm — area-normalised, so comparable across basins):
  annual_max  = flood-peak depth (wetlands expected to attenuate -> W<0)
  annual_p95  = high-flow depth
  annual_median = low-flow / baseflow depth (wetlands sustain -> W>0)
  annual_total = annual water yield (mostly precipitation)
"""
import warnings
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
warnings.filterwarnings("ignore")
ROOT=Path("/Users/jared/Wetland"); OUT=ROOT/"outputs"/"sensors"; SENS=ROOT/"sensors"
YCSV="/Users/jared/Downloads/04_annual_runoff_metrics_1985_2025.csv"

def rd(p,**k):
    d=pd.read_csv(p,dtype={"site_no":str},**k); d["site_no"]=d["site_no"].str.zfill(8); return d
# ── their depth data (keep only adequately complete years) ───────────────────
y=rd(YCSV); y["year"]=y["year"].astype(int)
ycol=["annual_median_runoff_depth_mm_day","annual_mean_runoff_depth_mm_day",
      "annual_total_runoff_depth_mm","annual_max_runoff_depth_mm_day",
      "annual_p95_runoff_depth_mm_day"]
for c in ycol+["drainage_area_sqmi","data_completeness"]: y[c]=pd.to_numeric(y[c],errors="coerce")
y=y[y["complete_90pct"].astype(str).str.lower().isin(["true","1"])]      # >=90% days

# ── our covariates ───────────────────────────────────────────────────────────
W  = rd(OUT/"panel_W.csv")[["site_no","year","wet_frac","Wfrac_exp","near_frac"]]
pr = rd(OUT/"panel_precip_daymet.csv")
lc = rd(OUT/"panel_landcover_lcmap.csv")[["site_no","year","developed_frac"]]
dam= rd(SENS/"dam_by_sensor.csv")[["site_no","earliest_dam_year"]]

P=(y.merge(W,on=["site_no","year"]).merge(pr,on=["site_no","year"])
     .merge(lc,on=["site_no","year"]).merge(dam,on="site_no",how="left"))
P["Dam"]=((P["earliest_dam_year"].notna())&(P["year"]>=P["earliest_dam_year"])).astype(int)
# transforms
P["y_max"]=np.log(P["annual_max_runoff_depth_mm_day"].clip(lower=1e-3))
P["y_p95"]=np.log(P["annual_p95_runoff_depth_mm_day"].clip(lower=1e-3))
P["y_med"]=np.log(P["annual_median_runoff_depth_mm_day"].clip(lower=1e-3))
P["y_tot"]=np.log(P["annual_total_runoff_depth_mm"].clip(lower=1e-1))
P["W"]=P["Wfrac_exp"]; P["urban"]=P["developed_frac"]
P["lprecip"]=np.log(P["precip_mm"].clip(lower=1))
P["larea"]=np.log(P["drainage_area_sqmi"].clip(lower=0.05))
P=P.dropna(subset=["y_max","W","lprecip","urban","larea"]).sort_values(["site_no","year"])

# ── save merged panel ────────────────────────────────────────────────────────
keep=["site_no","station_nm","year","drainage_area_sqmi",
      "annual_max_runoff_depth_mm_day","annual_p95_runoff_depth_mm_day",
      "annual_median_runoff_depth_mm_day","annual_mean_runoff_depth_mm_day",
      "annual_total_runoff_depth_mm","data_completeness",
      "wet_frac","Wfrac_exp","near_frac","precip_mm","developed_frac","Dam"]
P[keep].to_csv(OUT/"panel_matched_depth.csv",index=False)
print(f"MATCHED PANEL: {len(P)} gauge-years, {P.site_no.nunique()} gauges, "
      f"years {int(P.year.min())}-{int(P.year.max())}, "
      f"{P.groupby('site_no').size().min()}-{P.groupby('site_no').size().max()} yrs/gauge")
print(f"Saved -> {OUT/'panel_matched_depth.csv'}")

# Mundlak between/within terms
for v in ["W","lprecip","urban"]:
    P[v+"_bar"]=P.groupby("site_no")[v].transform("mean"); P[v+"_dev"]=P[v]-P[v+"_bar"]
pp=P.set_index(["site_no","year"])
def show(m,keys,lab):
    print(f"\n----- {lab} -----")
    for k in keys:
        if k in m.params.index:
            st="*" if m.pvalues[k]<0.05 else " "
            print(f"   {k:10s} {m.params[k]:+9.4f}  (p={m.pvalues[k]:.3f}){st}")
    r2=getattr(m,"rsquared_within",getattr(m,"rsquared",np.nan)); print(f"   R2={float(r2):.3f} N={int(m.nobs)}")

for yv,lab in [("y_max","FLOOD-PEAK runoff depth (log mm/day)"),
               ("y_med","LOW-FLOW / baseflow runoff depth (log)"),
               ("y_tot","ANNUAL TOTAL runoff depth (log mm)")]:
    print(f"\n================= OUTCOME: {lab} =================")
    mA=smf.ols(f"{yv} ~ W + lprecip + urban + Dam + larea", P).fit(
        cov_type="cluster",cov_kwds={"groups":P["site_no"]})
    show(mA,["W","lprecip","urban","Dam","larea"],"(A) Pooled OLS (cross+time)")
    try:
        mB=PanelOLS.from_formula(f"{yv} ~ W + lprecip + urban + EntityEffects + TimeEffects",pp)\
            .fit(cov_type="clustered",cluster_entity=True)
        show(mB,["W","lprecip","urban"],"(B) Two-way FE (within only)")
    except Exception as e: print("   (B) failed:",str(e)[:60])
    mC=smf.ols(f"{yv} ~ W_bar + W_dev + lprecip + urban + Dam + larea + C(year)", P).fit(
        cov_type="cluster",cov_kwds={"groups":P["site_no"]})
    show(mC,["W_bar","W_dev","lprecip","urban","Dam","larea"],
         "(C) Mundlak: W_bar=between(cross-sec), W_dev=within(temporal)")
print(f"\nClusters = {P.site_no.nunique()} gauges (nested) -> wild bootstrap for final inference.")
