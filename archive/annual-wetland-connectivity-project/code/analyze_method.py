"""Method demo (9 gauges, DESCRIPTIVE — nested, n small): does the travel-time-
weighted wetland metric explain flow-timing better than raw wetland fraction,
after removing the size + urbanisation confounds?
"""
import warnings; from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
OUT=Path("/Users/jared/Wetland/outputs/sensors"); SENS=Path("/Users/jared/Wetland/sensors")
def rd(p,**k):
    d=pd.read_csv(p,dtype={"site_no":str},**k); d["site_no"]=d["site_no"].str.zfill(8); return d
W=rd(OUT/"panel_W.csv"); fm=rd(OUT/"panel_flowmetrics.csv"); lc=rd(OUT/"panel_landcover_lcmap.csv")
sens=rd(SENS/"sensors_dv_gauges.csv")

g=(sens[["site_no","station_nm","drain_area_km2"]]
   .merge(W.groupby("site_no")[["wet_frac","Wfrac_exp","Wfrac_inv","near_frac","meanT_wet_hr"]].mean().reset_index(),on="site_no")
   .merge(fm.groupby("site_no")[["rb_flashiness","peak_cfs","q10_cfs","q_mean_cfs"]].mean().reset_index(),on="site_no")
   .merge(lc.groupby("site_no")["developed_frac"].mean().reset_index(),on="site_no"))
g["logA"]=np.log(g.drain_area_km2)
g["peak_per_km2"]=g.peak_cfs/g.drain_area_km2
g["q10_per_km2"]=g.q10_cfs/g.drain_area_km2
g.to_csv(OUT/"method_summary.csv",index=False)
pd.set_option("display.width",240)
print("=== Per-gauge method summary ===")
print(g[["site_no","drain_area_km2","developed_frac","wet_frac","Wfrac_exp","near_frac",
         "rb_flashiness","peak_per_km2"]].round(3).to_string(index=False))

def pcorr(y,x,ctrls):
    Z=np.c_[np.ones(len(g)),g[ctrls].values]
    ry=g[y].values-Z@np.linalg.lstsq(Z,g[y].values,rcond=None)[0]
    rx=g[x].values-Z@np.linalg.lstsq(Z,g[x].values,rcond=None)[0]
    return np.corrcoef(rx,ry)[0,1]

print("\n=== Correlation with flow-timing (n=9, NESTED → descriptive) ===")
print(f"{'predictor':12s} {'raw r(flash)':>12s} {'partial|A,urb':>14s} {'raw r(peak)':>12s} {'partial|A,urb':>14s}")
for x in ["wet_frac","Wfrac_exp","near_frac"]:
    rf=np.corrcoef(g[x],g.rb_flashiness)[0,1]; pf=pcorr("rb_flashiness",x,["logA","developed_frac"])
    rp=np.corrcoef(g[x],g.peak_per_km2)[0,1]; pp=pcorr("peak_per_km2",x,["logA","developed_frac"])
    print(f"{x:12s} {rf:>12.2f} {pf:>14.2f} {rp:>12.2f} {pp:>14.2f}")
print("\nNote: wetland is collinear with urbanisation (r=-0.93) & area (r=0.75);")
print("partial correlations strip those out. n=9 nested ⇒ illustrative, not inferential.")

# ── comparison figure ────────────────────────────────────────────────────────
fig,ax=plt.subplots(1,2,figsize=(13,5.5))
for a,xcol,xl in [(ax[0],"wet_frac","raw wetland fraction"),(ax[1],"near_frac","near-stream wetland fraction (travel-time)")]:
    sc=a.scatter(g[xcol],g.rb_flashiness,s=40+g.drain_area_km2/4,c=g.developed_frac,cmap="autumn_r",ec="k")
    a.set_xlabel(xl); a.set_ylabel("flashiness (R-B index)")
    for _,r in g.iterrows(): a.annotate(r.site_no[-5:],(r[xcol],r.rb_flashiness),fontsize=6)
    pf=pcorr("rb_flashiness",xcol,["logA","developed_frac"])
    a.set_title(f"flashiness vs {xl}\nraw r={np.corrcoef(g[xcol],g.rb_flashiness)[0,1]:+.2f}, partial|area,urban={pf:+.2f}")
plt.colorbar(sc,ax=ax[1],label="developed fraction")
fig.tight_layout(); fig.savefig(OUT/"00_method_W_vs_area.png",dpi=150,bbox_inches="tight"); plt.close(fig)
print(f"\nWrote method_summary.csv + 00_method_W_vs_area.png")
