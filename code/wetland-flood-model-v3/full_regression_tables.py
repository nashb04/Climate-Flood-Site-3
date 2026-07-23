#!/usr/bin/env python
"""
Mark_model_v3 -- full regression tables for Models 1-4 (complete coefficient output).
One table per model; columns = canonical outcomes; rows = ALL coefficients with gauge-clustered
SE and stars. WCB p-values on the key added term are appended. Writes a LaTeX file + CSVs.
"""
from __future__ import annotations
import os, itertools, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrices
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "outputs")
CTRL = ["z_log_DA", "z_imp_pct", "z_ksat_log", "z_chan_slope", "z_api_30_mm", "winter_flag"]

OUTS = [("log_Qp", "log Qp"), ("log_peakedness", "log peaked."), ("log1p_ttp", "log TTP"),
        ("runoff_coeff_w", "runoff coef"), ("log_quick_depth_mm", "log qdepth"),
        ("log1p_q99_excess_depth", "log q99exc")]
MODELS = {"M1": (["P","W"]+CTRL, None), "M2": (["P","W","WP"]+CTRL, "WP"),
          "M3": (["P","W","WP","S"]+CTRL, "S"), "M4": (["P","W","WP","S","WS"]+CTRL, "WS")}
ROW_ORDER = ["Intercept","P","W","WP","S","WS","z_log_DA","z_imp_pct","z_ksat_log",
             "z_chan_slope","z_api_30_mm","winter_flag"]
LAB = {"Intercept":"Intercept","P":"P (rain)","W":"W (wetland)","WP":"W$\\times$P",
       "S":"S=log(Ve/Sw)","WS":"W$\\times$S","z_log_DA":"log DA","z_imp_pct":"impervious",
       "z_ksat_log":"Ksat","z_chan_slope":"chan.\\ slope","z_api_30_mm":"antecedent",
       "winter_flag":"winter"}

def z(s):
    s = pd.to_numeric(s, errors="coerce"); sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0

def load():
    df = pd.read_csv(os.path.join(OUT, "events_panel_v3.csv"), dtype={"site_no": str})
    df["site_no"] = df["site_no"].str.zfill(8); df = df[df.usable == 1].copy()
    df["Wn"] = df["W_exp"]/(df["DA_km2"]*1e6)
    df["S"] = z(np.log(df["Ve_over_Sw"].clip(lower=1e-6)))
    df["P"] = z(df["log_Pe"]); df["W"] = z(df["Wn"]); df["WP"] = df.W*df.P; df["WS"] = df.W*df.S
    for c in ["log_DA","imp_pct","ksat_log","chan_slope","api_30_mm"]:
        df["z_"+c] = z(df[c])
    df["winter_flag"] = df["winter_flag"].astype(float)
    return df

def cl_se(X,u,groups,XtXi):
    G=len(groups);n,k=X.shape;m=np.zeros((k,k))
    for idx in groups:
        s=X[idx].T@u[idx]; m+=np.outer(s,s)
    return np.sqrt(np.diag((G/(G-1))*((n-1)/(n-k))*XtXi@m@XtXi))

def wcb(df,y,rhs,term):
    d=df[list({*rhs,"site_no",y})].replace([np.inf,-np.inf],np.nan).dropna()
    gids=d.site_no.to_numpy();groups=[np.where(gids==g)[0] for g in sorted(np.unique(gids))]
    yv,X=dmatrices(f"{y} ~ "+" + ".join(rhs),d,return_type="dataframe")
    ti=list(X.columns).index(term);X=X.to_numpy();yv=yv.to_numpy().ravel();XtXi=np.linalg.inv(X.T@X)
    b=XtXi@(X.T@yv);t=b[ti]/cl_se(X,yv-X@b,groups,XtXi)[ti]
    Xr=np.delete(X,ti,1);fr=Xr@(np.linalg.inv(Xr.T@Xr)@(Xr.T@yv));rr=yv-fr
    ts=[]
    for sg in itertools.product([1.,-1.],repeat=len(groups)):
        w=np.empty(len(yv))
        for gi,idx in enumerate(groups): w[idx]=sg[gi]
        ys=fr+w*rr;bs=XtXi@(X.T@ys);ts.append(bs[ti]/cl_se(X,ys-X@bs,groups,XtXi)[ti])
    return float(np.mean(np.abs(ts)>=abs(t)))

def star(p): return "***" if p<.01 else "**" if p<.05 else "*" if p<.1 else ""

def fit(df,y,rhs):
    d=df[list({*rhs,"site_no",y})].replace([np.inf,-np.inf],np.nan).dropna()
    m=smf.ols(f"{y} ~ "+" + ".join(rhs),d).fit(cov_type="cluster",cov_kwds={"groups":d.site_no})
    return m

def run():
    df=load()
    tex=[r"\documentclass[10pt]{article}",r"\usepackage[margin=0.6in]{geometry}",
         r"\usepackage{booktabs,pdflscape,amsmath,array}",
         r"\renewcommand{\arraystretch}{1.05}",r"\begin{document}",
         r"\begin{center}\Large\textbf{Mark Model v3 --- Full Regression Output, Models 1--4}\\",
         r"\normalsize event-scale panel, 5{,}907 gauge-events, 9 gauges; OLS with gauge-clustered SE\end{center}"]
    for mname,(rhs,key) in MODELS.items():
        # collect coefficients per outcome
        cell={}; nobs={}; r2={}; wcbp={}
        for yc,_ in OUTS:
            m=fit(df,yc,rhs); nobs[yc]=int(m.nobs); r2[yc]=m.rsquared
            for v in m.params.index:
                cell[(v,yc)]=(m.params[v],m.bse[v],m.pvalues[v])
            if key: wcbp[yc]=wcb(df,yc,rhs,key)
        rows=[v for v in ROW_ORDER if any((v,yc) in cell for yc,_ in OUTS)]
        # LaTeX table
        tex.append(r"\begin{landscape}\begin{table}[htbp]\centering")
        cap={"M1":"basic wetland effect","M2":"+ peak-shaving W$\\times$P",
             "M3":"+ saturation S","M4":"+ nonlinear saturation W$\\times$S"}[mname]
        tex.append(rf"\caption{{\textbf{{Model {mname[1]}}} ({cap}). Cell: coefficient (clustered SE); "
                   r"stars from clustered $p$: *** $<$.01, ** $<$.05, * $<$.1.}")
        tex.append(r"\scriptsize\begin{tabular}{l"+ "r"*len(OUTS) +"}\toprule")
        tex.append(" & "+" & ".join(f"\\textbf{{{lab}}}" for _,lab in OUTS)+r" \\ \midrule")
        for v in rows:
            line=LAB.get(v,v)
            for yc,_ in OUTS:
                if (v,yc) in cell:
                    b,se,p=cell[(v,yc)]
                    line+=f" & {b:.3f}{star(p)} ({se:.3f})"
                else: line+=" & "
            tex.append(line+r" \\")
        tex.append(r"\midrule")
        if key:
            wl="WCB $p$ ("+LAB.get(key,key)+")"
            tex.append(wl+" & "+" & ".join(f"{wcbp[yc]:.3f}" for yc,_ in OUTS)+r" \\")
        tex.append("$N$ & "+" & ".join(str(nobs[yc]) for yc,_ in OUTS)+r" \\")
        tex.append("$R^2$ & "+" & ".join(f"{r2[yc]:.3f}" for yc,_ in OUTS)+r" \\")
        tex.append(r"\bottomrule\end{tabular}")
        tex.append(r"\end{table}\end{landscape}")
        # CSV
        recs=[]
        for v in rows:
            row={"term":v}
            for yc,_ in OUTS:
                if (v,yc) in cell:
                    b,se,p=cell[(v,yc)]; row[yc]=f"{b:.4f}{star(p)}"; row[yc+"_se"]=round(se,4)
            recs.append(row)
        pd.DataFrame(recs).to_csv(os.path.join(OUT,f"regtable_{mname}.csv"),index=False)
        print(f"[{mname}] fitted {len(OUTS)} outcomes; key={key}")
    tex.append(r"\end{document}")
    with open(os.path.join(OUT,"full_regression_tables.tex"),"w") as f:
        f.write("\n".join(tex))
    print("wrote full_regression_tables.tex + regtable_M1..M4.csv")

if __name__=="__main__":
    run()
