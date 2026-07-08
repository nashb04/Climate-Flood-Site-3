#!/usr/bin/env python
"""
Mark's Model — wild cluster bootstrap (Cameron-Gelbach-Miller) for valid inference
with only G=9 gauges (clusters). Cluster-robust asymptotics are unreliable at 9 clusters;
the wild cluster bootstrap is the standard few-cluster fix.

For each model's newly-added term (M1:W, M2:WP, M3:S, M4:WS) and each outcome we report:
  - the cluster-robust t (asymptotic, for reference)
  - WCR p-value: restricted (impose-null) wild bootstrap-t, Rademacher weights, ALL 2^9=512
    sign vectors enumerated exactly (deterministic, no simulation error)
  - WCU 95% CI: unrestricted wild bootstrap percentile interval of the coefficient

Caveat: for cluster-INVARIANT regressors (W main effect, constant within gauge) few-cluster
inference is intrinsically weak regardless of method; the bootstrap is trustworthy for the
within-cluster-varying interaction terms (WP, S, WS).
"""
from __future__ import annotations
import os, itertools
import numpy as np, pandas as pd
from patsy import dmatrices
import fit_models as fm

OUT = fm.OUT
OUTCOMES = fm.OUTCOMES
# (model name, newly-added term tested)
TESTS = [("M1_basic", "W"), ("M2_peakshave", "WP"), ("M3_saturation", "S"), ("M4_nonlinsat", "WS")]


def cluster_vcov(X, u, groups_idx, XtX_inv):
    n, k = X.shape; G = len(groups_idx)
    meat = np.zeros((k, k))
    for idx in groups_idx:
        Xg = X[idx]; ug = u[idx]
        sg = Xg.T @ ug
        meat += np.outer(sg, sg)
    c = (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    return c * XtX_inv @ meat @ XtX_inv


def fit(X, y, groups_idx, XtX_inv):
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    V = cluster_vcov(X, u, groups_idx, XtX_inv)
    return beta, u, np.sqrt(np.diag(V))


def run():
    df = fm.load()
    df = df.reset_index(drop=True)
    gids = df["site_no"].to_numpy()
    groups = [np.where(gids == g)[0] for g in sorted(np.unique(gids))]
    G = len(groups)
    signs = np.array(list(itertools.product([1.0, -1.0], repeat=G)))   # 512 x 9

    rows = []
    for y_name in OUTCOMES:
        for model, term in TESTS:
            terms = fm.MODELS[model]
            formula = f"{y_name} ~ " + " + ".join(terms)
            yv, X = dmatrices(formula, df, return_type="dataframe")
            cols = list(X.columns); ti = cols.index(term)
            X = X.to_numpy(); y = yv.to_numpy().ravel()
            XtX_inv = np.linalg.inv(X.T @ X)
            beta_o, u_o, se_o = fit(X, y, groups, XtX_inv)
            t_obs = beta_o[ti] / se_o[ti]

            # restricted design (impose H0: term = 0) -> restricted fitted + resid
            Xr = np.delete(X, ti, axis=1)
            XrtXr_inv = np.linalg.inv(Xr.T @ Xr)
            beta_r = XrtXr_inv @ (Xr.T @ y)
            fitted_r = Xr @ beta_r; resid_r = y - fitted_r

            # enumerate all 512 sign vectors
            t_star = np.empty(len(signs)); b_star = np.empty(len(signs))
            # per-obs weight built from cluster signs
            for s in range(len(signs)):
                w = np.empty(len(y))
                for gi, idx in enumerate(groups):
                    w[idx] = signs[s, gi]
                # WCR: bootstrap under the null for the p-value
                y_wcr = fitted_r + w * resid_r
                b, u, se = fit(X, y_wcr, groups, XtX_inv)
                t_star[s] = b[ti] / se[ti]
                # WCU: bootstrap around the point estimate for the CI
                y_wcu = (X @ beta_o) + w * u_o
                b2 = XtX_inv @ (X.T @ y_wcu)
                b_star[s] = b2[ti]
            p_wcr = float(np.mean(np.abs(t_star) >= abs(t_obs)))
            ci = np.percentile(b_star, [2.5, 97.5])
            rows.append(dict(outcome=y_name, model=model, term=term,
                             beta=round(beta_o[ti], 4), t_cluster=round(t_obs, 2),
                             p_wcr=round(p_wcr, 4),
                             ci95_lo=round(ci[0], 4), ci95_hi=round(ci[1], 4)))
            print(f"[{y_name:>14} {model:>13} {term:>3}] beta={beta_o[ti]:+.3f} "
                  f"t={t_obs:+.2f}  WCR p={p_wcr:.4f}  95%CI[{ci[0]:+.3f},{ci[1]:+.3f}]", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "wild_bootstrap.csv"), index=False)
    print("\n=== WILD CLUSTER BOOTSTRAP SUMMARY (512 exact Rademacher, G=9) ===")
    print(res.to_string(index=False))
    print("\np_wcr: fraction of |bootstrap t| >= |observed t| under the null "
          "(min resolvable ~1/512=0.002). CI = unrestricted wild-bootstrap percentile.")


if __name__ == "__main__":
    run()
