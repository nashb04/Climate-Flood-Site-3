"""Annual first-difference regression: d_Y ~ d_W + d_Precip + d_Urban + DamRemoval + YearFE.

First differences (consecutive years only, within gauge) cancel every time-invariant
basin characteristic — directly defusing the cross-sectional / nesting confounding.
Outcomes are flow-response metrics (USGS serves no stage for these gauges):
  peak flow, Richards-Baker flashiness, baseflow Q10.
Key regressor d_W = change in travel-time-weighted wetland metric (+ raw-area horse race).
SE clustered by gauge (few clusters -> caveat noted).
"""
import warnings
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
OUT=Path("/Users/jared/Wetland/outputs/sensors")

def rd(n):
    d=pd.read_csv(OUT/n, dtype={"site_no":str}); d["site_no"]=d["site_no"].str.zfill(8); return d
W=rd("panel_W.csv"); fm=rd("panel_flowmetrics.csv")
pr=rd("panel_precip_daymet.csv"); lc=rd("panel_landcover_lcmap.csv")
sens=pd.read_csv("/Users/jared/Wetland/sensors/sensors_dv_gauges.csv",dtype={"site_no":str})
sens["site_no"]=sens["site_no"].str.zfill(8)
area=sens.set_index("site_no")["drain_area_km2"]

# ── merge to a levels panel (gauge x year) ───────────────────────────────────
p=(fm.merge(W[["site_no","year","wet_frac","Wfrac_exp","near_frac"]],on=["site_no","year"],how="inner")
     .merge(pr,on=["site_no","year"],how="inner")
     .merge(lc[["site_no","year","developed_frac"]],on=["site_no","year"],how="inner"))
# wetland measures (already 0-1 fractions): wet_frac=raw, Wfrac_exp=travel-time
# weighted (exp kernel), near_frac=near-stream wetland fraction. levels for outcomes:
p["lpeak"]=np.log(p["peak_cfs"].clip(lower=1e-3))
p["lq10"]=np.log(p["q10_cfs"].clip(lower=1e-2))
p["lprecip"]=np.log(p["precip_mm"].clip(lower=1))
p=p.sort_values(["site_no","year"]).reset_index(drop=True)

# ── first differences within gauge, consecutive years only ───────────────────
def fd(df, cols):
    out=[]
    for sid,g in df.groupby("site_no"):
        g=g.sort_values("year")
        d=g[cols].diff()
        d["dyear"]=g["year"].diff()
        d["site_no"]=sid; d["year"]=g["year"]
        out.append(d[g["year"].notna()])
    d=pd.concat(out)
    return d[d["dyear"]==1].copy()          # keep only consecutive-year pairs

cols=["lpeak","rb_flashiness","lq10","Wfrac_exp","wet_frac","near_frac","lprecip","developed_frac"]
D=fd(p,cols).rename(columns={
    "lpeak":"d_lpeak","rb_flashiness":"d_flash","lq10":"d_lq10",
    "Wfrac_exp":"d_W","wet_frac":"d_area","near_frac":"d_near",
    "lprecip":"d_lprecip","developed_frac":"d_urban"})
# dam-removal pulse (FD of a step) — North Avenue 1997, Estabrook 2018 on 04087000
D["DamRemoval"]=(((D.site_no=="04087000")&(D.year.isin([1997,2018])))).astype(int)
D=D.dropna(subset=["d_lpeak","d_W","d_lprecip","d_urban"])
D.to_csv(OUT/"fd_panel.csv",index=False)

print(f"FIRST-DIFFERENCE PANEL: {len(D)} gauge-year changes, {D.site_no.nunique()} gauges, "
      f"years {int(D.year.min())}-{int(D.year.max())}")
print("\nIdentification check — SD of each differenced regressor:")
for c,lab in [("d_W","d_W  (weighted wetland)"),("d_area","d_area (raw wetland)"),
              ("d_lprecip","d_lprecip"),("d_urban","d_urban")]:
    print(f"   {lab:26s} SD = {D[c].std():.6f}")
print("   -> if d_W SD ~ 0, its coefficient is not identified (LCMAP wetland is flat).")

# ── run the model for each outcome, with raw-area horse race ─────────────────
def run(y, key, label):
    f=f"{y} ~ {key} + d_lprecip + d_urban + DamRemoval + C(year)"
    m=smf.ols(f, D).fit(cov_type="cluster", cov_kwds={"groups":D["site_no"]})
    keep=[key,"d_lprecip","d_urban","DamRemoval"]
    print(f"\n===== {label}   [outcome {y}] =====")
    for k in keep:
        if k in m.params.index:
            print(f"   {k:12s} coef={m.params[k]:+9.4f}  SE={m.bse[k]:7.4f}  p={m.pvalues[k]:.3f}")
    print(f"   R2={m.rsquared:.3f}  N={int(m.nobs)}  (year FE absorbed; SE clustered by gauge)")

for y,lab in [("d_lpeak","PEAK FLOW (log)"),("d_flash","FLASHINESS"),("d_lq10","BASEFLOW Q10 (log)")]:
    run(y,"d_W",  f"{lab}  — weighted W")
    run(y,"d_area",f"{lab}  — raw area (horse race)")

print("\nNOTE: only 9 gauge-clusters -> cluster-robust SE are approximate; "
      "wild-cluster bootstrap recommended for final inference.")
print("Wrote", OUT/"fd_panel.csv")
