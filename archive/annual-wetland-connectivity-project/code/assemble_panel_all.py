"""Assemble the FULL unbalanced panel (all AOI gauges, all available years) and
run panel regressions per the user's design:
  Y(i,t) ~ W(i,t) + precip + urban(developed) + Dam(i,t)   [+ area / FE / year]
Outcome Y = discharge-based baseline (swap in real depth later by replacing the
y-columns). Three estimators shown so identification is transparent:
  (A) Pooled OLS  — uses cross-section + time (size-controlled)
  (B) Two-way FE  — entity + year (within only)
  (C) Mundlak     — between (gauge-mean W) vs within (deviation) decomposition
"""
import warnings
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
warnings.filterwarnings("ignore")
OUT=Path("/Users/jared/Wetland/outputs/sensors"); SENS=Path("/Users/jared/Wetland/sensors")

def rd(p, **k):
    d=pd.read_csv(p, dtype={"site_no":str}, **k); d["site_no"]=d["site_no"].str.zfill(8); return d
W   = rd(OUT/"panel_W.csv")
fm  = rd(OUT/"panel_flowmetrics.csv")
pr  = rd(OUT/"panel_precip_daymet.csv")
lc  = rd(OUT/"panel_landcover_lcmap.csv")
dam = rd(SENS/"dam_by_sensor.csv")
area= rd(OUT/"sensor_summary.csv")[["site_no","catch_km2"]]

p=(fm.merge(W[["site_no","year","wet_frac","Wfrac_exp","near_frac"]],on=["site_no","year"])
     .merge(pr,on=["site_no","year"])
     .merge(lc[["site_no","year","developed_frac"]],on=["site_no","year"])
     .merge(area,on="site_no",how="left")
     .merge(dam[["site_no","earliest_dam_year"]],on="site_no",how="left"))
# Dam(i,t) = 1 if a dam existed in the catchment that year
p["Dam"]=((p["earliest_dam_year"].notna()) & (p["year"]>=p["earliest_dam_year"])).astype(int)
# transforms
p["lpeak"]=np.log(p["peak_cfs"].clip(lower=1e-3))
p["lq10"] =np.log(p["q10_cfs"].clip(lower=1e-2))
p["lqmean"]=np.log(p["q_mean_cfs"].clip(lower=1e-3))
p["lprecip"]=np.log(p["precip_mm"].clip(lower=1))
p["larea"]=np.log(p["catch_km2"].clip(lower=0.1))
p["W"]=p["Wfrac_exp"]; p["urban"]=p["developed_frac"]
p=p.dropna(subset=["lpeak","W","lprecip","urban","larea"]).sort_values(["site_no","year"])
p.to_csv(OUT/"panel_master_all.csv",index=False)
print(f"PANEL: {len(p)} gauge-years, {p.site_no.nunique()} gauges, years {int(p.year.min())}-{int(p.year.max())}")
print(f"  gauges with a dam: {p.groupby('site_no').Dam.max().sum()}   "
      f"unbalanced: {p.groupby('site_no').size().min()}-{p.groupby('site_no').size().max()} yrs/gauge")
# gauge means for Mundlak
for v in ["W","lprecip","urban"]:
    p[v+"_bar"]=p.groupby("site_no")[v].transform("mean"); p[v+"_dev"]=p[v]-p[v+"_bar"]

pp=p.set_index(["site_no","year"])
def show(m, keys, label):
    print(f"\n----- {label} -----")
    for k in keys:
        if k in m.params.index:
            star="*" if m.pvalues[k]<0.05 else " "
            print(f"   {k:12s} {m.params[k]:+10.4f}  (p={m.pvalues[k]:.3f}){star}")
    r2=getattr(m,"rsquared_within",getattr(m,"rsquared",np.nan))
    print(f"   R2={float(r2):.3f}  N={int(m.nobs)}")

for y,lab in [("lpeak","PEAK FLOW (log)"),("rb_flashiness","FLASHINESS"),
              ("lq10","BASEFLOW Q10 (log)")]:
    print(f"\n========================= OUTCOME: {lab} =========================")
    # (A) pooled OLS
    mA=smf.ols(f"{y} ~ W + lprecip + urban + Dam + larea", p).fit(
        cov_type="cluster", cov_kwds={"groups":p["site_no"]})
    show(mA,["W","lprecip","urban","Dam","larea"],"(A) Pooled OLS (cross+time, size-controlled)")
    # (B) two-way FE
    try:
        mB=PanelOLS.from_formula(f"{y} ~ W + lprecip + urban + EntityEffects + TimeEffects", pp)\
            .fit(cov_type="clustered", cluster_entity=True)
        show(mB,["W","lprecip","urban"],"(B) Two-way FE (entity+year, WITHIN only)")
    except Exception as e: print("   (B) failed:",str(e)[:70])
    # (C) Mundlak between/within
    mC=smf.ols(f"{y} ~ W_bar + W_dev + lprecip + urban + Dam + larea + C(year)", p).fit(
        cov_type="cluster", cov_kwds={"groups":p["site_no"]})
    show(mC,["W_bar","W_dev","lprecip","urban","Dam","larea"],
         "(C) Mundlak: W_bar=between(cross-sec), W_dev=within(temporal)")

print("\nNOTE: clusters =", p.site_no.nunique(), "gauges (nested) -> wild-cluster bootstrap "
      "recommended for final inference. Replace y-cols with real depth to upgrade from baseline.")
print("Wrote", OUT/"panel_master_all.csv")
