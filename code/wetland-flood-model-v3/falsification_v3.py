#!/usr/bin/env python
"""v3 falsification suite for the primary outcome log_Qp (M2): VIF, placebo (permute wetlands
across the 9 gauges), leave-one-gauge-out, for W (level) and W*P (interaction). Reads v3 panel."""
from __future__ import annotations
import os, itertools, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
CTRL = ["z_log_DA","z_imp_pct","z_ksat_log","z_chan_slope","z_api_30_mm","winter_flag"]
def z(s):
    s=pd.to_numeric(s,errors="coerce");sd=s.std(ddof=0);return (s-s.mean())/sd if sd>0 else s*0
def load():
    df=pd.read_csv(os.path.join(OUT,"events_panel_v3.csv"),dtype={"site_no":str})
    df["site_no"]=df.site_no.str.zfill(8);df=df[df.usable==1].copy()
    df["Wn"]=df.W_exp/(df.DA_km2*1e6);df["P"]=z(df.log_Pe);df["W"]=z(df.Wn);df["WP"]=df.W*df.P
    for c in ["log_DA","imp_pct","ksat_log","chan_slope","api_30_mm"]: df["z_"+c]=z(df[c])
    df["winter_flag"]=df.winter_flag.astype(float);return df
def ols_cl(df,terms,y="log_Qp"):
    return smf.ols(f"{y} ~ "+" + ".join(terms),df).fit(cov_type="cluster",cov_kwds={"groups":df.site_no})
def run():
    df=load();terms=["P","W","WP"]+CTRL
    X=sm.add_constant(df[terms].astype(float))
    vif={t:round(variance_inflation_factor(X.values,i+1),1) for i,t in enumerate(terms)}
    print("VIF:",{k:vif[k] for k in ["P","W","WP","z_imp_pct","z_chan_slope"]})
    obs=ols_cl(df,terms);bW,bWP=obs.params["W"],obs.params["WP"]
    # placebo: permute Wn across gauges
    gmap=df.groupby("site_no").Wn.first();rng=np.random.default_rng(0);pW=[];pWP=[]
    for _ in range(500):
        perm=pd.Series(rng.permutation(gmap.values),index=gmap.index);d=df.copy()
        d["W"]=z(d.site_no.map(perm));d["WP"]=d.W*d.P
        r=smf.ols("log_Qp ~ "+" + ".join(terms),d).fit()
        pW.append(r.params["W"]);pWP.append(r.params["WP"])
    plaW=float(np.mean(np.abs(pW)>=abs(bW)));plaWP=float(np.mean(np.abs(pWP)>=abs(bWP)))
    # LOGO
    loW=[];loWP=[]
    for g in sorted(df.site_no.unique()):
        r=ols_cl(df[df.site_no!=g],terms);loW.append(r.params["W"]);loWP.append(r.params["WP"])
    print(f"\nW (level):       obs={bW:+.3f}  placebo_p={plaW:.3f}  LOGO=[{min(loW):+.3f},{max(loW):+.3f}]")
    print(f"W×P (interaction): obs={bWP:+.3f}  placebo_p={plaWP:.3f}  LOGO=[{min(loWP):+.3f},{max(loWP):+.3f}]")
    pd.DataFrame([
        dict(term="W (level)",coef=round(bW,3),placebo_p=round(plaW,3),logo_lo=round(min(loW),3),logo_hi=round(max(loW),3),vif=vif["W"]),
        dict(term="WxP (interaction)",coef=round(bWP,3),placebo_p=round(plaWP,3),logo_lo=round(min(loWP),3),logo_hi=round(max(loWP),3),vif=vif["WP"]),
    ]).to_csv(os.path.join(OUT,"falsification_v3.csv"),index=False)
    print("\nwrote falsification_v3.csv")
if __name__=="__main__": run()
