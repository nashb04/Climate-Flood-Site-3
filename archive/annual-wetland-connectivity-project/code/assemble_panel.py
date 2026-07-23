"""Assemble the 1985–2023 sensor×year panel and run the depth~wetland regression
with confounder controls.

Outcome: annual mean discharge (depth proxy — USGS serves no stage for these
gauges). Key regressor: LCMAP wetland_frac. Controls: precip (Daymet),
developed_frac (urbanisation), dam-removal events. Entity (sensor) fixed effects
absorb time-invariant basin differences; SE clustered by sensor.
"""
import warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
OUT=Path("/Users/jared/Wetland/outputs/sensors")

def rd(name, **kw): return pd.read_csv(OUT/name, dtype={"site_no":str}, **kw)
lc=rd("panel_landcover_lcmap.csv"); pr=rd("panel_precip_daymet.csv"); fs=rd("panel_flow_stage.csv")
for d in (lc,pr,fs): d["site_no"]=d["site_no"].str.zfill(8)

p=(fs.merge(lc,on=["site_no","year"],how="inner")
      .merge(pr,on=["site_no","year"],how="inner"))
p=p.dropna(subset=["q_mean_cfs"]).copy()

# dam covariates
dams=rd("confounders_dams_by_sensor.csv"); dams["site_no"]=dams["site_no"].str.zfill(8)
p=p.merge(dams[["site_no","n_dams"]],on="site_no",how="left")
# Milwaukee main-stem dam REMOVALS (not in NID): North Avenue 1997, Estabrook 2018.
# These sit in 04087000's catchment (downstream of Cedarburg), so flag that gauge.
p["dam_removal"]=(((p.site_no=="04087000")&(p.year>=1997))).astype(int)
p["dam_removal_estabrook"]=(((p.site_no=="04087000")&(p.year>=2018))).astype(int)

# transforms
p["log_q"]=np.log(p["q_mean_cfs"].clip(lower=1e-3))
p["precip_m"]=p["precip_mm"]/1000.0
p=p.sort_values(["site_no","year"]).reset_index(drop=True)
p.to_csv(OUT/"panel_master.csv",index=False)
print(f"PANEL: {len(p)} rows, {p.site_no.nunique()} sensors, years {p.year.min()}-{p.year.max()}")

# within-sensor variance of the key regressor — is wetland change identifiable?
wv=p.groupby("site_no")["wetland_frac"].std().mean()
dv=p.groupby("site_no")["developed_frac"].std().mean()
print(f"\nMean WITHIN-sensor SD:  wetland_frac={wv:.5f}   developed_frac={dv:.5f}   "
      f"(near-zero ⇒ effect unidentifiable with entity FE)")

# ── Regressions ──────────────────────────────────────────────────────────────
from linearmodels.panel import PanelOLS
pp=p.set_index(["site_no","year"])
def run(formula, label):
    try:
        m=PanelOLS.from_formula(formula, pp).fit(cov_type="clustered", cluster_entity=True)
        print(f"\n===== {label} =====\n{m.params.round(4).to_string()}")
        print("  (p-values)\n"+m.pvalues.round(4).to_string())
        print(f"  within R²={m.rsquared_within:.3f}  N={m.nobs}")
        return m
    except Exception as e:
        print(f"\n{label}: FAILED {e}")

# (1) Entity FE (within) — the correct spec for "change → change"
run("log_q ~ wetland_frac + developed_frac + precip_m + dam_removal + EntityEffects",
    "FE: log(Q) ~ wetland + developed + precip + dam_removal  (entity FE)")
# (2) Pooled (between+within) — cross-sensor wetland association (confounded)
import statsmodels.formula.api as smf
ols=smf.ols("log_q ~ wetland_frac + developed_frac + precip_m", p).fit(
    cov_type="cluster", cov_kwds={"groups":p["site_no"]})
print("\n===== POOLED OLS (no FE) — cross-sectional, confounded =====")
print(ols.params.round(4).to_string()); print("(p)\n"+ols.pvalues.round(4).to_string())
print(f"  R²={ols.rsquared:.3f}  N={int(ols.nobs)}")

# ── Visual: per-sensor time series ───────────────────────────────────────────
fig,axes=plt.subplots(2,2,figsize=(15,9))
for sid,g in p.groupby("site_no"):
    axes[0,0].plot(g.year,g.q_mean_cfs,lw=0.9,label=sid)
    axes[0,1].plot(g.year,g.wetland_frac,lw=0.9)
    axes[1,0].plot(g.year,g.developed_frac,lw=0.9)
    axes[1,1].plot(g.year,g.precip_mm,lw=0.9)
axes[0,0].set_title("Annual mean discharge (depth proxy)"); axes[0,0].set_ylabel("cfs"); axes[0,0].set_yscale("log")
axes[0,1].set_title("LCMAP wetland fraction (≈ FLAT — the problem)"); axes[0,1].set_ylabel("frac")
axes[1,0].set_title("LCMAP developed fraction (urbanisation ↑)"); axes[1,0].set_ylabel("frac")
axes[1,1].set_title("Daymet annual precip"); axes[1,1].set_ylabel("mm")
axes[0,0].legend(fontsize=6,ncol=3)
for ax in axes.ravel(): ax.set_xlabel("year")
fig.tight_layout(); fig.savefig(OUT/"00_panel_timeseries.png",dpi=150,bbox_inches="tight"); plt.close(fig)
print(f"\nWrote panel_master.csv + 00_panel_timeseries.png")
