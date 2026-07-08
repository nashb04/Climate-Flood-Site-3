#!/usr/bin/env python
"""
Mark's Model — Step 6: fit the phenomenological Models 1-4 and falsification diagnostics.

  Y_{j,e} = b0 + b1 P + b2 W + b3 (W*P) + b4 (Ve/Sw) + b5 (W * Ve/Sw) + X gamma + u_j + e

  M1 basic wetland effect     Y ~ P + W + controls          (b2: do wetlands reduce peaks?)
  M2 peak-shaving             + W:P                          (b3<0: matter more in big storms?)
  M3 saturation               + sat=log(Ve/Sw)              (b4>0: response up past capacity?)
  M4 nonlinear saturation     + W:sat                        (b5: wetland benefit decays past cap)

Treatment W = Wn = W_exp / catchment area (effectiveness-weighted wetland fraction; intensive,
less DA-collinear than extensive W_exp). Predictors standardised. Estimated with (a) gauge
random-intercept MixedLM and (b) OLS with gauge-clustered SE. Falsification: gauge-level
wetland permutation placebo + leave-one-gauge-out. Outcomes: log peak, R-B flashiness,
hydrograph width.  (depth->damage intentionally excluded.)
"""
from __future__ import annotations
import os, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "outputs")
OUTCOMES = ["log_Qp", "rb_flashiness", "hydro_width_hr"]
CONTROLS = ["log_DA", "imp_pct", "ksat_log", "chan_slope", "api_30_mm", "winter_flag"]
PRIMARY = "log_Qp"

def z(s):
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0

def load():
    df = pd.read_csv(os.path.join(OUT, "events_panel.csv"), dtype={"site_no": str})
    df["site_no"] = df["site_no"].str.zfill(8)
    df = df[df["usable"] == 1].copy()
    # intensive treatment: effectiveness-weighted wetland fraction
    df["Wn"] = df["W_exp"] / (df["DA_km2"] * 1e6)
    df["sat"] = np.log(np.clip(df["Ve_over_Sw"], 1e-6, None))
    df["hydro_width_hr"] = np.log(df["hydro_width_hr"].clip(lower=0.5))   # log-width (skew)
    # standardised design columns
    df["P"] = z(df["log_Pe"])
    df["W"] = z(df["Wn"])
    df["S"] = z(df["sat"])
    for c in CONTROLS:
        df["z_" + c] = z(df[c]) if c != "winter_flag" else df[c].astype(float)
    df["WP"] = df["W"] * df["P"]
    df["WS"] = df["W"] * df["S"]
    return df

Zc = ["z_" + c for c in CONTROLS]

MODELS = {
    "M1_basic":      ["P", "W"] + Zc,
    "M2_peakshave":  ["P", "W", "WP"] + Zc,
    "M3_saturation": ["P", "W", "WP", "S"] + Zc,
    "M4_nonlinsat":  ["P", "W", "WP", "S", "WS"] + Zc,
}

def fit_mixed(df, y, terms):
    f = f"{y} ~ " + " + ".join(terms)
    try:
        m = smf.mixedlm(f, df, groups=df["site_no"]).fit(reml=False, method="lbfgs")
        return m
    except Exception as e:
        print("   mixed fail:", str(e)[:80]); return None

def fit_ols_cluster(df, y, terms):
    f = f"{y} ~ " + " + ".join(terms)
    return smf.ols(f, df).fit(cov_type="cluster", cov_kwds={"groups": df["site_no"]})

def vif_report(df, terms):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = sm.add_constant(df[terms].astype(float).dropna())
    return {terms[i]: round(variance_inflation_factor(X.values, i + 1), 1) for i in range(len(terms))}

def key_row(res, term):
    if res is None or term not in res.params.index:
        return (np.nan, np.nan)
    return (round(res.params[term], 4), round(res.pvalues[term], 4))

def run():
    df = load()
    print(f"usable events: {len(df)} across {df.site_no.nunique()} gauges\n")

    # ---- VIF on the fullest design (M4) ----
    print("=== VIF (M4 design; between-gauge controls will be high — expected, 9 gauges) ===")
    v = vif_report(df, MODELS["M4_nonlinsat"])
    print("  " + "  ".join(f"{k}={val}" for k, val in v.items()))

    # ---- Models 1-4 for each outcome, both estimators ----
    rows = []
    for y in OUTCOMES:
        for name, terms in MODELS.items():
            mm = fit_mixed(df, y, terms)
            oc = fit_ols_cluster(df, y, terms)
            rec = dict(outcome=y, model=name, n=int(oc.nobs))
            for t in ["P", "W", "WP", "S", "WS"]:
                bm, pm = key_row(mm, t); bo, po = key_row(oc, t)
                rec[f"{t}_mix_b"], rec[f"{t}_mix_p"] = bm, pm
                rec[f"{t}_ols_b"], rec[f"{t}_ols_p"] = bo, po
            rec["mix_aic"] = round(mm.aic, 1) if mm is not None else np.nan
            rows.append(rec)
    coef = pd.DataFrame(rows)
    coef.to_csv(os.path.join(OUT, "step6_coefficients.csv"), index=False)

    def show(y):
        s = coef[coef.outcome == y]
        cols = ["model", "P_mix_b", "W_mix_b", "W_mix_p", "WP_mix_b", "WP_mix_p",
                "S_mix_b", "S_mix_p", "WS_mix_b", "WS_mix_p", "mix_aic"]
        print(f"\n=== {y} — MixedLM (gauge RE); b=std coef, p ===")
        print(s[cols].to_string(index=False))
    for y in OUTCOMES:
        show(y)

    # ---- placebo: permute wetland among the 9 gauges, refit M2 on primary ----
    print(f"\n=== PLACEBO (permute Wn across gauges), outcome={PRIMARY}, M2 ===")
    gmap = df.groupby("site_no")["Wn"].first()
    obs = fit_ols_cluster(df, PRIMARY, MODELS["M2_peakshave"])
    b_obs_W = obs.params["W"]; b_obs_WP = obs.params["WP"]
    rng = np.random.default_rng(0); permsW, permsWP = [], []
    for _ in range(500):
        perm = pd.Series(rng.permutation(gmap.values), index=gmap.index)
        d2 = df.copy()
        d2["Wn_p"] = d2["site_no"].map(perm)
        d2["W"] = z(d2["Wn_p"]); d2["WP"] = d2["W"] * d2["P"]
        r = smf.ols(f"{PRIMARY} ~ " + " + ".join(MODELS["M2_peakshave"]), d2).fit()
        permsW.append(r.params["W"]); permsWP.append(r.params["WP"])
    pW = float(np.mean(np.abs(permsW) >= abs(b_obs_W)))
    pWP = float(np.mean(np.abs(permsWP) >= abs(b_obs_WP)))
    print(f"  W:  observed b={b_obs_W:.3f}  permutation p={pW:.3f}")
    print(f"  WP: observed b={b_obs_WP:.3f}  permutation p={pWP:.3f}")
    print("  (small p = effect stronger than random gauge relabeling; large p = confounded/uninformative)")

    # ---- leave-one-gauge-out stability, M2 primary ----
    print(f"\n=== LEAVE-ONE-GAUGE-OUT (M2, {PRIMARY}): W and WP coef range ===")
    bs = []
    for g in sorted(df.site_no.unique()):
        r = fit_ols_cluster(df[df.site_no != g], PRIMARY, MODELS["M2_peakshave"])
        bs.append((g, round(r.params["W"], 3), round(r.params["WP"], 3)))
    lo = pd.DataFrame(bs, columns=["dropped_gauge", "W", "WP"])
    print(lo.to_string(index=False))
    print(f"  W  range [{lo.W.min()}, {lo.W.max()}]   WP range [{lo.WP.min()}, {lo.WP.max()}]")

    # ---- figure: coefficient forest (MixedLM) across models, key terms ----
    _forest(coef)
    print("\nWrote step6_coefficients.csv and step6_forest.png")

def _forest(coef):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    terms = ["W", "WP", "S", "WS"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, y in zip(axes, OUTCOMES):
        s = coef[coef.outcome == y]
        ypos = 0; labels = []
        for _, r in s.iterrows():
            for t in terms:
                b = r[f"{t}_mix_b"]
                if pd.notna(b):
                    sig = r[f"{t}_mix_p"] < 0.05
                    ax.scatter(b, ypos, color=("crimson" if sig else "gray"),
                               s=35, zorder=3)
                    labels.append(f"{r.model.split('_')[0]}:{t}")
                    ypos += 1
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(y, fontsize=10); ax.set_xlabel("std coefficient")
    fig.suptitle("Step 6: Models 1-4 key coefficients (red = p<0.05, MixedLM gauge RE)", y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "step6_forest.png"), dpi=120,
                                    bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    run()
