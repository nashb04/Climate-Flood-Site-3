#!/usr/bin/env python
"""
Y-expansion Step 2: the mechanism matrix. For each outcome we fit the peak-shaving model
(M2: Y ~ P + W + W*P + controls, gauge random effect) and test the W*P interaction with the
same discipline as the main analysis: asymptotic p, wild cluster bootstrap (512 exact
Rademacher), and leave-one-gauge-out range. Predicted signs are PRE-REGISTERED below.
Reads panel_y_expanded.csv read-only; writes only in y_variables_v2/.
"""
from __future__ import annotations
import os, itertools, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrices
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "outputs")
CTRL = ["z_log_DA", "z_imp_pct", "z_ksat_log", "z_chan_slope", "z_api_30_mm", "winter_flag"]

# (mechanism, outcome column/expr, predicted W*P sign, tier)
OUTCOMES = [
    ("attenuation-peak",   "log_Qp",                   "-", "canonical"),
    ("attenuation-shape",  "log_peakedness",           "-", "canonical"),
    ("delay-timing",       "log1p_ttp",                "+", "canonical"),
    ("storage-volume",     "runoff_coeff_w",           "-", "canonical"),
    ("cumulative-severity","log_quick_depth_mm",       "-", "canonical"),
    ("extreme-severity",   "log1p_q99_excess_depth",   "-", "canonical"),
    ("shape (robust)",     "rb_flashiness",            "-", "robustness"),
    ("shape/delay (rob.)", "log_hydro_width",          "+", "robustness"),
    ("rel-peak (robust)",  "log_peak_ratio",           "-", "robustness"),
    ("release (robust)",   "recession_k_per_hr",       "-", "robustness"),
]

def z(s):
    s = pd.to_numeric(s, errors="coerce"); sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0

def load():
    df = pd.read_csv(os.path.join(HERE, "data", "panel_y_expanded.csv"), dtype={"site_no": str})
    df["site_no"] = df["site_no"].str.zfill(8)
    df = df[df["usable"] == 1].copy()
    # derived outcome columns
    df["log1p_ttp"] = np.log1p(df["time_to_peak_hr"].clip(lower=0))
    df["runoff_coeff_w"] = df["runoff_coeff"].clip(0, 3)                 # winsorise
    df["log_peak_ratio"] = np.log(df["rredi_peak_ratio"].clip(lower=1e-3))
    # design
    df["Wn"] = df["W_exp"] / (df["DA_km2"] * 1e6)
    df["P"] = z(df["log_Pe"]); df["W"] = z(df["Wn"]); df["WP"] = df["W"] * df["P"]
    for c in ["log_DA", "imp_pct", "ksat_log", "chan_slope", "api_30_mm"]:
        df["z_" + c] = z(df[c])
    df["winter_flag"] = df["winter_flag"].astype(float)
    return df

def cl_se(X, u, groups, XtXi):
    G = len(groups); n, k = X.shape; meat = np.zeros((k, k))
    for idx in groups:
        s = X[idx].T @ u[idx]; meat += np.outer(s, s)
    return np.sqrt(np.diag((G/(G-1))*((n-1)/(n-k)) * XtXi @ meat @ XtXi))

def analyse(df, y, terms=("P", "W", "WP")):
    d = df[[*{*terms, "WP", "W", "P", "site_no"}, *CTRL, y]].replace([np.inf, -np.inf], np.nan).dropna()
    rhs = ["P", "W", "WP"] + CTRL
    gids = d["site_no"].to_numpy(); groups = [np.where(gids == g)[0] for g in sorted(np.unique(gids))]
    yv, X = dmatrices(f"{y} ~ " + " + ".join(rhs), d, return_type="dataframe")
    cols = list(X.columns); ti = cols.index("WP"); X = X.to_numpy(); yv = yv.to_numpy().ravel()
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ (X.T @ yv); u = yv - X @ b; t_obs = b[ti] / cl_se(X, u, groups, XtXi)[ti]
    # asymptotic mixed p
    mm = smf.mixedlm(f"{y} ~ " + " + ".join(rhs), d, groups=d["site_no"]).fit(reml=False)
    p_asym = mm.pvalues["WP"]
    # WCB restricted (512) + WCU CI
    Xr = np.delete(X, ti, 1); br = np.linalg.inv(Xr.T @ Xr) @ (Xr.T @ yv); fr = Xr @ br; rr = yv - fr
    signs = np.array(list(itertools.product([1., -1.], repeat=len(groups))))
    tst = np.empty(len(signs)); bst = np.empty(len(signs))
    for s in range(len(signs)):
        w = np.empty(len(yv))
        for gi, idx in enumerate(groups):
            w[idx] = signs[s, gi]
        ys = fr + w * rr; bs = XtXi @ (X.T @ ys); us = ys - X @ bs
        tst[s] = bs[ti] / cl_se(X, us, groups, XtXi)[ti]
        yu = X @ b + w * u; bst[s] = (XtXi @ (X.T @ yu))[ti]
    p_wcb = float(np.mean(np.abs(tst) >= abs(t_obs))); ci = np.percentile(bst, [2.5, 97.5])
    # LOGO
    lo = []
    for g in sorted(np.unique(gids)):
        dd = d[d.site_no != g]
        yv2, X2 = dmatrices(f"{y} ~ " + " + ".join(rhs), dd, return_type="dataframe")
        b2 = np.linalg.inv(X2.values.T @ X2.values) @ (X2.values.T @ yv2.values.ravel())
        lo.append(b2[list(X2.columns).index("WP")])
    return dict(n=len(d), WP_b=round(b[ti], 4), p_asym=round(p_asym, 4), p_wcb=round(p_wcb, 4),
                ci_lo=round(ci[0], 4), ci_hi=round(ci[1], 4),
                logo_lo=round(min(lo), 4), logo_hi=round(max(lo), 4))

def run():
    df = load()
    rows = []
    for mech, y, sign, tier in OUTCOMES:
        r = analyse(df, y)
        obs = "+" if r["WP_b"] > 0 else "-"
        r.update(mechanism=mech, y=y, pred=sign, obs_sign=obs, tier=tier,
                 sign_ok=(obs == sign), robust=(r["p_wcb"] < 0.05 and obs == sign))
        rows.append(r)
        print(f"[{tier:10}] {y:24} pred {sign}  b={r['WP_b']:+.3f}  "
              f"WCB p={r['p_wcb']:.3f}  LOGO[{r['logo_lo']:+.3f},{r['logo_hi']:+.3f}]  "
              f"{'ROBUST' if r['robust'] else ''}", flush=True)
    res = pd.DataFrame(rows)[["tier","mechanism","y","n","pred","obs_sign","WP_b",
                              "p_asym","p_wcb","ci_lo","ci_hi","logo_lo","logo_hi",
                              "sign_ok","robust"]]
    res.to_csv(os.path.join(OUT, "mechanism_matrix.csv"), index=False)
    print("\n=== MECHANISM MATRIX (W×P interaction per outcome) ===")
    print(res.to_string(index=False))
    _forest(res)

def _forest(res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    r = res.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, row in r.iterrows():
        col = "crimson" if row.robust else ("#1f77b4" if row.sign_ok else "gray")
        ax.plot([row.ci_lo, row.ci_hi], [i, i], color=col, lw=1.5, zorder=2)
        ax.scatter(row.WP_b, i, color=col, s=45, zorder=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(r)))
    ax.set_yticklabels([f"{row.y}  (pred {row.pred})" for _, row in r.iterrows()], fontsize=8)
    ax.set_xlabel("W×P standardized coefficient (95% wild-bootstrap CI)")
    ax.set_title("Mechanism matrix: wetland × rainfall interaction by outcome\n"
                 "red = robust (WCB p<.05 & predicted sign); blue = predicted sign; gray = off", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "mechanism_matrix.png"), dpi=130); plt.close(fig)

if __name__ == "__main__":
    run()
