#!/usr/bin/env python
"""
Mark_model_v3 -- panel build on top of v2, applying:

 (FIX) Antecedent control decoupled from the forcing. In v2, api_30_mm was P_eff evaluated
       AT the event-start day WITH day-0 included, so the "antecedent wetness" control shared
       the event-start-day rainfall with P_e (the cumulative event-window forcing) -> the two
       overlapped (corr 0.35). v3 defines a STRICTLY-PRIOR antecedent:
           api_30_mm = sum_{n=1..30} k^n * P(d0 - n),   k = 0.9,   day-0 EXCLUDED
       so it captures only pre-event wetness and no longer double-counts P_e.
       P_e stays cumulative rainfall over the event window [start..end] (unchanged, correct).

 (Y-EXPANSION) Adds the mechanism-complete outcome set from the y_variables work:
       log_peakedness, log_quick_depth_mm, log1p_ttp, log_hydro_width, log_peak_ratio,
       and the new Q99-excess integral (log1p_q99_excess_depth) from the 15-min hydrographs.

Improved wetland W (celerity T + roughness + type-weighted glaciated storage, C dropped),
RREDI events, PRISM 800 m rainfall, and controls are INHERITED from v2 unchanged.
Reads read-only; writes only in Mark_model_v3/.
"""
from __future__ import annotations
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data"); OUT = os.path.join(HERE, "outputs")
V2 = "/Users/jared/Wetland/Mark_model_v2"
IVDIR = "/Users/jared/Wetland/Mark's Model/data"
CFS_CMS = 0.0283168466
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
API_K, API_N, TZ = 0.9, 30, "America/Chicago"
SITES = ["04087000","04086500","04086600","04087120","04087030",
         "04087050","04087070","04087088","04087119"]

def antecedent_prior(daily):
    """STRICTLY-PRIOR antecedent: api_prev(t) = sum_{n=1..N} k^n P(t-n)  (day-0 excluded)."""
    p = daily["precip_mm"].fillna(0.0).to_numpy()
    api = np.zeros_like(p)
    for n in range(1, API_N + 1):                     # n starts at 1 -> excludes day 0
        api[n:] += (API_K ** n) * p[:-n]
    return pd.Series(api, index=daily.index)

def q99_excess(iv, q99, t0, t1):
    w = iv.loc[t0:t1, "Q_cfs"].dropna()
    if len(w) < 2:
        return 0.0
    E = np.clip(w.to_numpy() - q99, 0, None)
    ns = w.index.asi8.astype("float64"); secs = (ns - ns[0]) / 1e9
    return float(_trapz(E, secs) * CFS_CMS)

def run():
    p = pd.read_csv(os.path.join(V2, "outputs", "events_panel.csv"), dtype={"site_no": str})
    p["site_no"] = p["site_no"].str.zfill(8)
    p["t_start"] = pd.to_datetime(p["t_start"], utc=True)
    p["t_end"] = pd.to_datetime(p["t_end"], utc=True)
    p["api_30_v2"] = p["api_30_mm"]                    # keep old (day-0-included) for comparison
    area_m2 = p["DA_km2"] * 1e6

    api_fixed = np.full(len(p), np.nan); exc = np.full(len(p), np.nan)
    for site in SITES:
        daily = pd.read_parquet(os.path.join(V2, "data", f"precip_daily_{site}.parquet"))
        ap = antecedent_prior(daily)                  # date-indexed strictly-prior antecedent
        iv = pd.read_parquet(os.path.join(IVDIR, f"iv_{site}.parquet"))
        q99 = float(np.nanpercentile(iv["Q_cfs"], 99))
        idx = p.index[p.site_no == site]
        for i in idx:
            d0 = pd.Timestamp(p.at[i, "t_start"]).tz_convert(TZ).normalize().tz_localize(None)
            j = ap.index[ap.index <= d0]
            api_fixed[i] = float(ap.loc[j[-1]]) if len(j) else np.nan
            if p.at[i, "usable"] == 1:
                exc[i] = q99_excess(iv, q99, p.at[i, "t_start"], p.at[i, "t_end"])
        print(f"[{site}] Q99={q99:.0f} cfs  events={len(idx)}", flush=True)

    p["api_30_mm"] = api_fixed                         # <-- FIXED antecedent (decoupled from P_e)
    # expanded outcomes
    p["q99_excess_m3"] = exc
    p["log1p_q99_excess_depth"] = np.log1p((p["q99_excess_m3"] / area_m2 * 1000).clip(lower=0))
    p["log_peakedness"] = np.log((p["Qp_cfs"] * CFS_CMS) / p["quick_vol_m3"].clip(lower=1e-3))
    p["log_quick_depth_mm"] = np.log((p["quick_vol_m3"] / area_m2 * 1000).clip(lower=1e-3))
    p["log1p_ttp"] = np.log1p(p["time_to_peak_hr"].clip(lower=0))
    p["log_hydro_width"] = np.log(p["hydro_width_hr"].clip(lower=0.5))
    p["log_peak_ratio"] = np.log(p["rredi_peak_ratio"].clip(lower=1e-3))
    p["runoff_coeff_w"] = p["runoff_coeff"].clip(0, 3)

    p.to_csv(os.path.join(OUT, "events_panel_v3.csv"), index=False)
    u = p[p.usable == 1]
    print("\n=== V3 PANEL ===")
    print(f"rows {len(p)}  usable {int(p.usable.sum())}")
    print(f"corr(api_30, P_e):  v2(day-0 incl)={u['api_30_v2'].corr(u['P_e_mm']):.3f}  "
          f"-> v3(strictly prior)={u['api_30_mm'].corr(u['P_e_mm']):.3f}")
    print(f"api_30 median: v2={u['api_30_v2'].median():.1f}mm  v3={u['api_30_mm'].median():.1f}mm")

if __name__ == "__main__":
    run()
