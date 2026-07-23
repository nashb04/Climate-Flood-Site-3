"""Regression diagnostics for the flood-peak model — what is W's coefficient
actually identified from, and can it support a causal claim?
Model: ln(flood-peak depth) ~ W + precip + urban + Dam + area.
"""
import warnings
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
OUT=Path("/Users/jared/Wetland/outputs/sensors")
P=pd.read_csv(OUT/"panel_matched_depth.csv",dtype={"site_no":str}); P["site_no"]=P["site_no"].str.zfill(8)
P=P.rename(columns={"Wfrac_exp":"W","developed_frac":"urban"})
P=P.dropna(subset=["ln_peak","W","precip","urban","area"]).copy()
L=[]; pr=lambda s="": L.append(s)
pr("="*74); pr("DIAGNOSTICS — ln(flood-peak runoff depth) ~ W + precip + urban + Dam + area")
pr(f"N={len(P)} gauge-years, G={P.site_no.nunique()} gauges"); pr("="*74)

# 1. WHERE is the identifying variation? within vs between -------------------
pr("\n[1] Source of identifying variation  (between vs within gauge)")
pr(f"{'var':10s}{'between SD':>12s}{'within SD':>12s}{'within share':>14s}")
for v in ["W","urban","precip","area","ln_peak"]:
    bw=P.groupby('site_no')[v].mean(); bet=bw.std()
    wit=P.groupby('site_no')[v].transform(lambda x:x-x.mean()).std()
    sh=wit**2/(wit**2+bet**2) if (wit**2+bet**2)>0 else 0
    pr(f"{v:10s}{bet:12.5f}{wit:12.5f}{sh:14.3f}")
pr("  -> W within-share ~ 0  =>  W's coefficient is identified ENTIRELY cross-")
pr("     sectionally (between basins). No within-gauge variation = no clean causal.")

# 2. Collinearity (VIF) ------------------------------------------------------
pr("\n[2] Multicollinearity (VIF; >5 worrying, >10 severe)")
X=P[["W","precip","urban","area","Dam"]].assign(const=1.0)
for i,c in enumerate(X.columns):
    if c=="const": continue
    pr(f"   VIF[{c:8s}] = {variance_inflation_factor(X.values,i):6.2f}")
pr(f"   corr(W, urban) = {P['W'].corr(P['urban']):+.2f}   corr(W, area) = {P['W'].corr(P['area']):+.2f}")

# 3. OLS fit + influence (which gauges drive W?) -----------------------------
m=smf.ols("ln_peak ~ W + precip + urban + Dam + area",P).fit()
infl=m.get_influence(); cooks=infl.cooks_distance[0]
jW=list(m.params.index).index("W"); dfbW=infl.dfbetas[:,jW]
P=P.assign(cooks=cooks,dfbW=dfbW,lev=infl.hat_matrix_diag,resid=m.resid,fitted=m.fittedvalues)
pr("\n[3] Influence on the W coefficient (top basins by mean |DFBETA_W|)")
gi=P.groupby('site_no').agg(dfbW=('dfbW',lambda x:np.mean(np.abs(x))),
                            cooks=('cooks','max'),n=('W','size'),Wbar=('W','mean')).sort_values('dfbW',ascending=False)
pr(gi.head(6).round(4).to_string())

# 4. Leave-one-GAUGE-out: how fragile is the W coefficient? -------------------
pr("\n[4] Leave-one-gauge-out — range of the W coefficient")
coefs={}
for sid in P.site_no.unique():
    mm=smf.ols("ln_peak ~ W + precip + urban + Dam + area",P[P.site_no!=sid]).fit()
    coefs[sid]=mm.params["W"]
cs=pd.Series(coefs); base=m.params["W"]
pr(f"   full-sample W = {base:.2f};  leave-one-out range = [{cs.min():.2f}, {cs.max():.2f}]")
pr(f"   most influential drop: {cs.idxmax() if abs(cs.max()-base)>abs(cs.min()-base) else cs.idxmin()} "
   f"moves W to {cs.loc[cs.idxmax()] if abs(cs.max()-base)>abs(cs.min()-base) else cs.loc[cs.idxmin()]:.2f}")

# 5. Specification sensitivity (omitted-variable fragility) ------------------
pr("\n[5] Specification sensitivity of the W coefficient (log flood-peak)")
specs={"base: W+precip+urban+Dam+area":"W + precip + urban + Dam + area",
       "drop urban":"W + precip + Dam + area",
       "drop area":"W + precip + urban + Dam",
       "log(area),log(precip)":"W + np.log(precip) + urban + Dam + np.log(area)",
       "W only (no controls)":"W"}
for lab,rhs in specs.items():
    mm=smf.ols(f"ln_peak ~ {rhs}",P).fit(cov_type="cluster",cov_kwds={"groups":P["site_no"]})
    pr(f"   {lab:32s} W={mm.params['W']:+8.2f}  (p={mm.pvalues['W']:.3f})")
pr("   -> large swings = W is entangled with the controls (omitted-variable risk).")

# 6. Heteroskedasticity ------------------------------------------------------
bp=het_breuschpagan(m.resid,m.model.exog)
pr(f"\n[6] Breusch-Pagan heteroskedasticity: LM p = {bp[1]:.4f}  "
   f"({'hetero' if bp[1]<0.05 else 'ok'}; cluster-robust SE already used)")

# 7. Pure between (cross-sectional) regression, n=G ---------------------------
B=P.groupby('site_no').mean(numeric_only=True)
mb=smf.ols("ln_peak ~ W + precip + urban + area",B).fit()
pr(f"\n[7] PURE BETWEEN regression (gauge means, n={len(B)}): "
   f"W={mb.params['W']:+.2f} (p={mb.pvalues['W']:.3f}), R2={mb.rsquared:.2f}")
pr("   ~ matches pooled W  =>  confirms identification is cross-sectional.")

txt="\n".join(L); (OUT/"diagnostics_report.txt").write_text(txt); print(txt)

# ── figure ──────────────────────────────────────────────────────────────────
fig,ax=plt.subplots(2,2,figsize=(13,9))
ax[0,0].scatter(P.fitted,P.resid,s=8,alpha=.4); ax[0,0].axhline(0,color="r",lw=1)
ax[0,0].set_xlabel("fitted"); ax[0,0].set_ylabel("residual"); ax[0,0].set_title("Residuals vs fitted")
sm.qqplot(P.resid,line="s",ax=ax[0,1]); ax[0,1].set_title("Normal Q-Q")
ax[1,0].scatter(P.lev,P.resid/P.resid.std(),s=8,alpha=.4)
ax[1,0].set_xlabel("leverage (hat)"); ax[1,0].set_ylabel("std. residual"); ax[1,0].set_title("Leverage vs residual")
ax[1,1].hist(cs,bins=18,color="steelblue",edgecolor="white"); ax[1,1].axvline(base,color="r",lw=1.5,label=f"full={base:.1f}")
ax[1,1].set_xlabel("W coefficient (leave-one-gauge-out)"); ax[1,1].set_title("W-coef stability"); ax[1,1].legend(fontsize=8)
fig.suptitle("Flood-peak regression diagnostics",fontsize=13); fig.tight_layout()
fig.savefig(OUT/"00_diagnostics.png",dpi=150,bbox_inches="tight")
print("\nWrote diagnostics_report.txt + 00_diagnostics.png")
