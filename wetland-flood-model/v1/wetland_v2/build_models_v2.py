#!/usr/bin/env python
"""
Wetland V2 -- Step 5: does the improved W change the event-model result?
Re-fits the peak-shaving models with W_v2 (celerity T, no C, type*storage S_i, glaciated Sw)
vs the original W, on the SAME event panel, and wild-cluster-bootstraps the W*P term.

Reads read-only: ../outputs/events_panel.csv, outputs/panel_W_v2.csv. Writes only here.
"""
from __future__ import annotations
import os, itertools, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrices
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "outputs")
MM = os.path.dirname(HERE)
OUTCOMES = ["log_Qp", "hydro_width_hr", "rb_flashiness"]
CTRL = ["z_log_DA", "z_imp_pct", "z_ksat_log", "z_chan_slope", "z_api_30_mm", "winter_flag"]

def z(s):
    s = pd.to_numeric(s, errors="coerce"); sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0

def load():
    df = pd.read_csv(os.path.join(MM, "outputs", "events_panel.csv"), dtype={"site_no": str})
    df["site_no"] = df["site_no"].str.zfill(8)
    df = df[df["usable"] == 1].copy()
    v2 = pd.read_csv(os.path.join(OUT, "panel_W_v2.csv"), dtype={"site_no": str})
    v2["site_no"] = v2["site_no"].str.zfill(8)
    df = df.merge(v2[["site_no", "W_v2", "Sw_v2_m3"]], on="site_no", how="left")
    df["Wn_old"] = df["W_exp"] / (df["DA_km2"] * 1e6)
    df["Wn_v2"] = df["W_v2"] / (df["DA_km2"] * 1e6)
    df["hydro_width_hr"] = np.log(df["hydro_width_hr"].clip(lower=0.5))
    df["P"] = z(df["log_Pe"])
    for c in ["log_DA", "imp_pct", "ksat_log", "chan_slope", "api_30_mm"]:
        df["z_" + c] = z(df[c])
    df["winter_flag"] = df["winter_flag"].astype(float)
    return df

# ---- wild cluster bootstrap (restricted, 512 exact) on term WP ----
def cluster_se(X, u, groups, XtXi):
    G = len(groups); n, k = X.shape; meat = np.zeros((k, k))
    for idx in groups:
        s = X[idx].T @ u[idx]; meat += np.outer(s, s)
    c = (G/(G-1))*((n-1)/(n-k)); V = c * XtXi @ meat @ XtXi
    return np.sqrt(np.diag(V))

def wcb_p(df, y, terms, term):
    gids = df["site_no"].to_numpy()
    groups = [np.where(gids == g)[0] for g in sorted(np.unique(gids))]
    yv, X = dmatrices(f"{y} ~ " + " + ".join(terms), df, return_type="dataframe")
    cols = list(X.columns); ti = cols.index(term); X = X.to_numpy(); yv = yv.to_numpy().ravel()
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ (X.T @ yv); u = yv - X @ b; t_obs = b[ti] / cluster_se(X, u, groups, XtXi)[ti]
    Xr = np.delete(X, ti, 1); br = np.linalg.inv(Xr.T @ Xr) @ (Xr.T @ yv)
    fr = Xr @ br; rr = yv - fr
    signs = np.array(list(itertools.product([1., -1.], repeat=len(groups))))
    tst = np.empty(len(signs))
    for s in range(len(signs)):
        w = np.empty(len(yv))
        for gi, idx in enumerate(groups):
            w[idx] = signs[s, gi]
        ys = fr + w * rr; bs = XtXi @ (X.T @ ys); us = ys - X @ bs
        tst[s] = bs[ti] / cluster_se(X, us, groups, XtXi)[ti]
    return b[ti], t_obs, float(np.mean(np.abs(tst) >= abs(t_obs)))

def run():
    df = load()
    print(f"usable events: {len(df)}  gauges: {df.site_no.nunique()}\n")
    rows = []
    for tag, wn in [("W_old", "Wn_old"), ("W_v2", "Wn_v2")]:
        d = df.copy(); d["W"] = z(d[wn]); d["WP"] = d["W"] * d["P"]
        terms = ["P", "W", "WP"] + CTRL
        for y in OUTCOMES:
            mm = smf.mixedlm(f"{y} ~ " + " + ".join(terms), d, groups=d["site_no"]).fit(reml=False)
            bWP = mm.params["WP"]; pWP = mm.pvalues["WP"]; bW = mm.params["W"]; pW = mm.pvalues["W"]
            b, t, pw = wcb_p(d, y, terms, "WP")
            rows.append(dict(W=tag, outcome=y, WP_b=round(bWP, 4), WP_p_asym=round(pWP, 4),
                             WP_p_wcb=round(pw, 4), W_b=round(bW, 4), W_p_asym=round(pW, 3)))
            print(f"[{tag:5} {y:14}] W×P b={bWP:+.3f} (asym p={pWP:.3f}, WCB p={pw:.3f})  "
                  f"| W_main b={bW:+.3f} p={pW:.2f}", flush=True)
    res = pd.DataFrame(rows); res.to_csv(os.path.join(OUT, "models_v2_compare.csv"), index=False)
    print("\n=== OLD vs NEW W: peak-shaving W×P (MixedLM + wild cluster bootstrap) ===")
    print(res.to_string(index=False))

if __name__ == "__main__":
    run()
