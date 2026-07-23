"""Six DESCRIPTIVE OLS regressions of annual runoff DEPTH on connectivity-weighted wetland.
3 outcomes (flood-peak / baseflow / annual-total) x {raw Y, log Y}.
Controls in RAW units (no log on precip/area). SE clustered by gauge.
Writes a Stata-style comparison table + the merged panel CSV.

Important: these pooled OLS models are cross-sectional/descriptive. They should
not be interpreted as causal wetland effects because the current W series has
almost no within-gauge time variation and is highly collinear with urbanization.
Use build_clean_causal.py for the causal workflow and identification diagnostics.
"""
import warnings
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
warnings.filterwarnings("ignore")
ROOT=Path("/Users/jared/Wetland"); OUT=ROOT/"outputs"/"sensors"; SENS=ROOT/"sensors"
YCSV="/Users/jared/Downloads/04_annual_runoff_metrics_1985_2025.csv"

def rd(p,**k):
    d=pd.read_csv(p,dtype={"site_no":str},**k); d["site_no"]=d["site_no"].str.zfill(8); return d
y=rd(YCSV); y["year"]=y["year"].astype(int)
for c in ["annual_max_runoff_depth_mm_day","annual_median_runoff_depth_mm_day",
          "annual_total_runoff_depth_mm","drainage_area_sqmi"]:
    y[c]=pd.to_numeric(y[c],errors="coerce")
y=y[y["complete_90pct"].astype(str).str.lower().isin(["true","1"])]

W = rd(OUT/"panel_W.csv")[["site_no","year","wet_frac","Wfrac_exp","near_frac"]]
pr= rd(OUT/"panel_precip_daymet.csv"); lc=rd(OUT/"panel_landcover_lcmap.csv")[["site_no","year","developed_frac"]]
dam=rd(SENS/"dam_by_sensor.csv")[["site_no","earliest_dam_year"]]
P=(y.merge(W,on=["site_no","year"]).merge(pr,on=["site_no","year"])
     .merge(lc,on=["site_no","year"]).merge(dam,on="site_no",how="left"))
P["Dam"]=((P["earliest_dam_year"].notna())&(P["year"]>=P["earliest_dam_year"])).astype(int)

# outcomes: raw + log
P["peak"]=P["annual_max_runoff_depth_mm_day"]
P["base"]=P["annual_median_runoff_depth_mm_day"]
P["total"]=P["annual_total_runoff_depth_mm"]
P["ln_peak"]=np.log(P["peak"].clip(lower=1e-3))
P["ln_base"]=np.log(P["base"].clip(lower=1e-3))
P["ln_total"]=np.log(P["total"].clip(lower=1e-1))
# regressors — RAW units
P["W"]=P["Wfrac_exp"]; P["urban"]=P["developed_frac"]
P["precip"]=P["precip_mm"]; P["area"]=P["drainage_area_sqmi"]
P=P.dropna(subset=["peak","W","precip","urban","area"]).sort_values(["site_no","year"])

cols=["site_no","station_nm","year","area",
      "peak","base","total","ln_peak","ln_base","ln_total",
      "Wfrac_exp","wet_frac","near_frac","precip","developed_frac","Dam"]
P[cols].to_csv(OUT/"panel_matched_depth.csv",index=False)

RHS="W + precip + urban + Dam + area"
def ols(yv): return smf.ols(f"{yv} ~ {RHS}", P).fit(cov_type="cluster", cov_kwds={"groups":P["site_no"]})
specs=[("peak","Peak (raw)"),("ln_peak","Peak (log)"),
       ("base","Base (raw)"),("ln_base","Base (log)"),
       ("total","Total (raw)"),("ln_total","Total (log)")]
models=[ols(v) for v,_ in specs]; names=[n for _,n in specs]
tab=summary_col(models, stars=True, model_names=names, float_format="%0.4f",
    info_dict={"N":lambda m:f"{int(m.nobs)}","R2":lambda m:f"{m.rsquared:0.3f}"},
    regressor_order=["W","precip","urban","Dam","area","Intercept"])
hdr=("="*92+"\nSIX DESCRIPTIVE OLS REGRESSIONS — annual runoff depth ~ wetland(W) + precip + urban + Dam + area\n"
     f"Unbalanced panel: {len(P)} gauge-years, {P.site_no.nunique()} gauges, {int(P.year.min())}-{int(P.year.max())}.\n"
     "Columns = each outcome with RAW Y and LOG Y. SE clustered by gauge (in parentheses).\n"
     "W = travel-time-weighted wetland fraction; precip in mm; area in sq mi.\n"+"="*92+"\n")
out=hdr+tab.as_text()
(OUT/"regression_output.txt").write_text(out)
print(out)
print("\nWrote:", OUT/"regression_output.txt", "and", OUT/"panel_matched_depth.csv")
