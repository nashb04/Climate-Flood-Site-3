#!/usr/bin/env python
"""
Mark_model_v3 -- fit the nested Models 1-4 for every outcome, with wild-cluster-bootstrap
inference (9 gauges) on each model's key added term.

  M1 basic wetland      Y ~ P + W + controls              key: W
  M2 peak-shaving       + W*P                              key: WP
  M3 saturation         + S=log(Ve/Sw)                    key: S
  M4 nonlinear sat.     + W*S                              key: WS

Controls include the CORRECTED strictly-prior antecedent api_30. Predicted W*P signs are
pre-registered per outcome. Reads events_panel_v3.csv read-only; writes only in Mark_model_v3/.
"""
from __future__ import annotations
import os, itertools, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrices
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "outputs")
CTRL = ["z_log_DA", "z_imp_pct", "z_ksat_log", "z_chan_slope", "z_api_30_mm", "winter_flag"]

OUTCOMES = [   # (tier, mechanism, y, predicted W*P sign)
    ("canonical", "attenuation-peak",    "log_Qp",                 "-"),
    ("canonical", "attenuation-shape",   "log_peakedness",         "-"),
    ("canonical", "delay-timing",        "log1p_ttp",              "+"),
    ("canonical", "storage-volume",      "runoff_coeff_w",         "-"),
    ("canonical", "cumulative-severity", "log_quick_depth_mm",     "-"),
    ("canonical", "extreme-severity",    "log1p_q99_excess_depth", "-"),
    ("robust",    "shape",               "rb_flashiness",          "-"),
    ("robust",    "shape/delay",         "log_hydro_width",        "+"),
    ("robust",    "rel-peak",            "log_peak_ratio",         "-"),
    ("robust",    "release",             "recession_k_per_hr",     "-"),
]
MODELS = {
    "M1_basic":      (["P", "W"] + CTRL, "W"),
    "M2_peakshave":  (["P", "W", "WP"] + CTRL, "WP"),
    "M3_saturation": (["P", "W", "WP", "S"] + CTRL, "S"),
    "M4_nonlinsat":  (["P", "W", "WP", "S", "WS"] + CTRL, "WS"),
}

def z(s):
    s = pd.to_numeric(s, errors="coerce"); sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0

def load():
    df = pd.read_csv(os.path.join(OUT, "events_panel_v3.csv"), dtype={"site_no": str})
    df["site_no"] = df["site_no"].str.zfill(8); df = df[df.usable == 1].copy()
    df["Wn"] = df["W_exp"] / (df["DA_km2"] * 1e6)
    df["S"] = z(np.log(df["Ve_over_Sw"].clip(lower=1e-6)))
    df["P"] = z(df["log_Pe"]); df["W"] = z(df["Wn"])
    df["WP"] = df["W"] * df["P"]; df["WS"] = df["W"] * df["S"]
    for c in ["log_DA", "imp_pct", "ksat_log", "chan_slope", "api_30_mm"]:
        df["z_" + c] = z(df[c])
    df["winter_flag"] = df["winter_flag"].astype(float)
    return df

def cl_se(X, u, groups, XtXi):
    G = len(groups); n, k = X.shape; meat = np.zeros((k, k))
    for idx in groups:
        s = X[idx].T @ u[idx]; meat += np.outer(s, s)
    return np.sqrt(np.diag((G/(G-1))*((n-1)/(n-k)) * XtXi @ meat @ XtXi))

def fit_term(df, y, rhs, term):
    d = df[list({*rhs, "site_no", y})].replace([np.inf, -np.inf], np.nan).dropna()
    gids = d["site_no"].to_numpy(); groups = [np.where(gids == g)[0] for g in sorted(np.unique(gids))]
    yv, X = dmatrices(f"{y} ~ " + " + ".join(rhs), d, return_type="dataframe")
    cols = list(X.columns); ti = cols.index(term); X = X.to_numpy(); yv = yv.to_numpy().ravel()
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ (X.T @ yv); u = yv - X @ b; t_obs = b[ti] / cl_se(X, u, groups, XtXi)[ti]
    mm = smf.mixedlm(f"{y} ~ " + " + ".join(rhs), d, groups=d["site_no"]).fit(reml=False)
    p_asym = mm.pvalues.get(term, np.nan)
    Xr = np.delete(X, ti, 1); br = np.linalg.inv(Xr.T @ Xr) @ (Xr.T @ yv); fr = Xr @ br; rr = yv - fr
    signs = np.array(list(itertools.product([1., -1.], repeat=len(groups))))
    tst = np.empty(len(signs)); bst = np.empty(len(signs))
    for s in range(len(signs)):
        w = np.empty(len(yv))
        for gi, idx in enumerate(groups):
            w[idx] = signs[s, gi]
        ys = fr + w * rr; bs = XtXi @ (X.T @ ys); us = ys - X @ bs
        tst[s] = bs[ti] / cl_se(X, us, groups, XtXi)[ti]
        bst[s] = (XtXi @ (X.T @ (X @ b + w * u)))[ti]
    ci = np.percentile(bst, [2.5, 97.5])
    return dict(n=len(d), coef=round(b[ti], 4), p_asym=round(float(p_asym), 4),
                p_wcb=round(float(np.mean(np.abs(tst) >= abs(t_obs))), 4),
                ci_lo=round(ci[0], 4), ci_hi=round(ci[1], 4))

def run():
    df = load(); print(f"usable {len(df)}  gauges {df.site_no.nunique()}\n")
    rows = []
    for tier, mech, y, sign in OUTCOMES:
        line = []
        for mname, (rhs, term) in MODELS.items():
            r = fit_term(df, y, rhs, term)
            rows.append(dict(tier=tier, mechanism=mech, outcome=y, model=mname, key=term,
                             pred_WP=sign, **r))
            line.append(f"{term}={r['coef']:+.3f}(wcb {r['p_wcb']:.3f})")
        print(f"[{tier:9}] {y:24} | " + " | ".join(line), flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "models_v3_all.csv"), index=False)

    # wide table: W*P interaction (the headline test) per outcome
    wp = res[res.key == "WP"].copy()
    wp["match"] = np.where(np.sign(wp.coef) == np.where(wp.pred_WP == "+", 1, -1), "Y", "N")
    wp["robust"] = ((wp.p_wcb < 0.05) & (wp["match"] == "Y"))
    wide = wp[["tier","mechanism","outcome","n","pred_WP","coef","p_asym","p_wcb",
               "ci_lo","ci_hi","match","robust"]]
    wide.to_csv(os.path.join(OUT, "models_v3_WxP.csv"), index=False)
    print("\n=== V3 W×P (peak-shaving, M2) BY OUTCOME ===")
    print(wide.to_string(index=False))
    _forest(wp)

def _forest(wp):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    r = wp.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, row in r.iterrows():
        col = "crimson" if row.robust else ("#1f77b4" if row["match"] == "Y" else "gray")
        ax.plot([row.ci_lo, row.ci_hi], [i, i], color=col, lw=1.5)
        ax.scatter(row.coef, i, color=col, s=45, zorder=3)
    ax.axvline(0, color="k", lw=0.8); ax.set_yticks(range(len(r)))
    ax.set_yticklabels([f"{x.outcome} (pred {x.pred_WP})" for _, x in r.iterrows()], fontsize=8)
    ax.set_xlabel("W×P standardized coefficient (95% wild-bootstrap CI)")
    ax.set_title("v3 mechanism matrix: wetland × rainfall interaction (M2)\n"
                 "red = robust (WCB p<.05 & predicted sign)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "models_v3_forest.png"), dpi=130); plt.close(fig)

if __name__ == "__main__":
    run()
