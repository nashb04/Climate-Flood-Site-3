"""Two-way fixed-effects (entity + year) panel regression of log runoff depth on
the travel-time-weighted wetland metric. Within estimator (clean-causal form):
uses only within-gauge year-to-year variation; entity FE absorb all time-invariant
basin characteristics (size, soils, the wetland-urban cross-sectional collinearity),
year FE absorb common annual shocks. SE clustered by gauge.

For the full causal workflow, including lagged W, first differences, and
automatic identification warnings, run build_clean_causal.py.
"""
import warnings
from pathlib import Path
import numpy as np, pandas as pd
from linearmodels.panel import PanelOLS, compare
warnings.filterwarnings("ignore")
OUT=Path("/Users/jared/Wetland/outputs/sensors")
P=pd.read_csv(OUT/"panel_matched_depth.csv",dtype={"site_no":str}); P["site_no"]=P["site_no"].str.zfill(8)
P=P.rename(columns={"Wfrac_exp":"W","developed_frac":"urban"})
P=P.dropna(subset=["ln_peak","ln_base","ln_total","W","precip","urban"]).sort_values(["site_no","year"])
pp=P.set_index(["site_no","year"])

# within-gauge SD of W (context for identification)
wW=P.groupby("site_no")["W"].transform(lambda x:x-x.mean()).std()
wU=P.groupby("site_no")["urban"].transform(lambda x:x-x.mean()).std()

outs=[("ln_peak","Flood-peak"),("ln_base","Baseflow"),("ln_total","Annual-total")]
models={}
for yv,lab in outs:
    models[lab]=PanelOLS.from_formula(
        f"{yv} ~ 1 + W + precip + urban + EntityEffects + TimeEffects", pp
    ).fit(cov_type="clustered", cluster_entity=True)

tab=compare(models, stars=True, precision="std_errors")
hdr=("="*84+"\nTWO-WAY FIXED EFFECTS (entity + year)  —  log runoff depth ~ W + precip + urban\n"
     f"Panel: {len(P)} gauge-years, {P.site_no.nunique()} gauges, {int(P.year.min())}-{int(P.year.max())}; "
     "SE clustered by gauge.\n"
     "Entity FE absorb size/soils/baseline land cover; Time FE absorb common annual shocks.\n"
     f"Within-gauge SD:  W = {wW:.6f}   urban = {wU:.5f}   (~0 => W barely moves within a gauge)\n"
     +"="*84+"\n")
out=hdr+str(tab)
(OUT/"regression_twoway_fe.txt").write_text(out)
print(out)
# explicit per-model W line
print("\nW (within) coefficient by outcome:")
for lab,m in models.items():
    print(f"   {lab:12s} W = {m.params['W']:+12.2f}  (SE {m.std_errors['W']:.1f}, p={m.pvalues['W']:.3f}) "
          f" within-R2={m.rsquared_within:.3f}  N={int(m.nobs)}  entities={m.entity_info.total}  periods={m.time_info.total}")
print("\nWrote", OUT/"regression_twoway_fe.txt")
