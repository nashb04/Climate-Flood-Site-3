#!/usr/bin/env python
"""
Y-expansion Step 1: add the two genuinely-new outcomes + transforms to a LOCAL copy of the
v2 panel. Reads existing caches read-only; writes only in y_variables_v2/.

New:
  q99_excess_*  -- trapezoidal integral of max(Q - Q99, 0) over each v2 event window
                   (magnitude x time ABOVE the gauge's 99th-pct flow), from cached 15-min Q.
  log_peakedness -- log( Qp[m3/s] / quickflow_volume[m3] ), a robust hydrograph-concentration
                    (attenuation-of-shape) signature, less interval-fragile than flashiness.
Transforms of existing columns: quick runoff depth, log width.
"""
from __future__ import annotations
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
MM = os.path.dirname(HERE)
PANEL = os.path.join(os.path.dirname(MM), "Mark_model_v2", "outputs", "events_panel.csv")
IVDIR = os.path.join(MM, "data")
CFS_CMS = 0.0283168466
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
SITES = ["04087000","04086500","04086600","04087120","04087030",
         "04087050","04087070","04087088","04087119"]

def q99_excess(iv, q99, t0, t1):
    """trapezoidal integral of max(Q-Q99,0) dt over [t0,t1]; returns m3."""
    w = iv.loc[t0:t1, "Q_cfs"].dropna()
    if len(w) < 2:
        return 0.0
    E = np.clip(w.to_numpy() - q99, 0, None)          # excess cfs
    ns = w.index.asi8.astype("float64")               # ns since epoch
    secs = (ns - ns[0]) / 1e9
    return float(_trapz(E, secs) * CFS_CMS)           # cfs*s = ft3 -> *0.0283 = m3

def run():
    p = pd.read_csv(PANEL, dtype={"site_no": str}); p["site_no"] = p["site_no"].str.zfill(8)
    p["t_start"] = pd.to_datetime(p["t_start"], utc=True)
    p["t_end"] = pd.to_datetime(p["t_end"], utc=True)
    area_m2 = p["DA_km2"] * 1e6

    exc = np.full(len(p), np.nan)
    for site in SITES:
        iv = pd.read_parquet(os.path.join(IVDIR, f"iv_{site}.parquet"))
        q99 = float(np.nanpercentile(iv["Q_cfs"], 99))
        idx = p.index[(p.site_no == site) & p.usable.eq(1)]
        for i in idx:
            exc[i] = q99_excess(iv, q99, p.at[i, "t_start"], p.at[i, "t_end"])
        n = p.loc[idx, "Qp_cfs"].gt(q99).sum()
        print(f"[{site}] Q99={q99:6.0f} cfs  events={len(idx)}  events>Q99={n}", flush=True)

    p["q99_excess_m3"] = exc
    p["q99_excess_depth_mm"] = p["q99_excess_m3"] / area_m2 * 1000
    p["log1p_q99_excess_depth"] = np.log1p(p["q99_excess_depth_mm"].clip(lower=0))
    # peakedness (attenuation of shape): peak flow / event quick volume
    p["log_peakedness"] = np.log((p["Qp_cfs"] * CFS_CMS) / p["quick_vol_m3"].clip(lower=1e-3))
    # cumulative severity as area-normalised runoff depth
    p["quick_depth_mm"] = p["quick_vol_m3"] / area_m2 * 1000
    p["log_quick_depth_mm"] = np.log(p["quick_depth_mm"].clip(lower=1e-3))
    p["log_hydro_width"] = np.log(p["hydro_width_hr"].clip(lower=0.5))

    p.to_csv(os.path.join(DATA, "panel_y_expanded.csv"), index=False)
    u = p[p.usable == 1]
    print(f"\nwrote panel_y_expanded.csv ({len(p)} rows, {int(p.usable.sum())} usable)")
    print("\nnew-outcome sanity (usable):")
    for c in ["q99_excess_depth_mm", "log1p_q99_excess_depth", "log_peakedness",
              "log_quick_depth_mm"]:
        s = u[c].replace([np.inf, -np.inf], np.nan)
        print(f"  {c:26} median={s.median():8.3f}  p10={s.quantile(.1):8.3f}  "
              f"p90={s.quantile(.9):8.3f}  zero%={100*(u[c]==0).mean():.0f}" if c.startswith('q99') else
              f"  {c:26} median={s.median():8.3f}  p10={s.quantile(.1):8.3f}  p90={s.quantile(.9):8.3f}")

if __name__ == "__main__":
    run()
