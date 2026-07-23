#!/usr/bin/env python
"""
Mark's Model — Step 1: Event extraction from streamflow (Y_{j,e} response metrics).

For each of the 9 long-record MMSD gauges:
  1. Pull 15-min instantaneous values (00060 discharge, 00065 gage height) from NWIS,
     2001-10-01 -> present, in yearly chunks, cached to data/iv_{site}.parquet.
  2. Aggregate to hourly, separate baseflow with the Eckhardt recursive digital filter
     (workflow Listing 4), detect storm events on the quickflow series.
  3. Compute per-event response metrics Y (peak Q / stage, time-to-peak, hydrograph
     width, recession, R-B flashiness, quickflow volume, duration above threshold).

Outputs:
  data/iv_{site}.parquet      raw 15-min cache (Q_cfs, stage_ft) — also feeds depth->damage later
  data/events_{site}.parquet  one row per storm event
  outputs/events_all.csv      all gauges stacked
  outputs/step1_events_qa_{site}.png   QA hydrograph for one gauge

Rainfall-dependent fields (P_e, V_e, runoff coefficient, rainfall-based time-to-peak)
are added in Step 2. Here time_to_peak is hydrograph-internal (start-of-rise -> peak).
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import dataretrieval.nwis as nwis

# ---------------------------------------------------------------- config
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT  = os.path.join(HERE, "outputs")
os.makedirs(DATA, exist_ok=True); os.makedirs(OUT, exist_ok=True)

SITES = ["04087000", "04086500", "04086600", "04087120", "04087030",
         "04087050", "04087070", "04087088", "04087119"]

START = "2001-10-01"          # 1-yr pre-roll before Stage IV radar era (2002)
END   = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

# Eckhardt filter params (hourly). a_daily=0.98 -> a_hourly = 0.98^(1/24).
A_HOURLY   = 0.98 ** (1.0 / 24.0)     # ~0.99916
BFI_MAX    = 0.60                     # urban/flashy WI streams (sensitivity-tested later)

# event detection (on hourly quickflow)
MIN_SEP_H     = 24      # min hours between distinct event peaks
REL_BASE      = 0.90    # event start/end = where quick drops 90% of prominence
REL_FWHM      = 0.50    # hydrograph width = full width at half prominence
MIN_PEAK_QUICK_FRAC = 0.02   # ignore peaks below 2% of the gauge's max quickflow
MAX_GAP_FRAC  = 0.20    # drop events whose window is >20% missing (data gap)
CFS_TO_CMS    = 0.0283168

# ---------------------------------------------------------------- IV pull
def _parse_iv(df: pd.DataFrame) -> pd.DataFrame:
    """Robustly pull Q (00060) and stage (00065) out of dataretrieval's dynamic cols."""
    def pick(prefix):
        cands = [c for c in df.columns if c.startswith(prefix) and not c.endswith("_cd")]
        if not cands:
            return None
        # choose the series with the most real numbers
        counts = {c: pd.to_numeric(df[c], errors="coerce").notna().sum() for c in cands}
        return max(counts, key=counts.get)
    qcol, scol = pick("00060"), pick("00065")
    out = pd.DataFrame(index=df.index)
    out["Q_cfs"]   = pd.to_numeric(df[qcol], errors="coerce") if qcol else np.nan
    out["stage_ft"] = pd.to_numeric(df[scol], errors="coerce") if scol else np.nan
    out.loc[out["Q_cfs"] < 0, "Q_cfs"] = np.nan      # ice/missing flags
    return out


def pull_iv(site: str) -> pd.DataFrame:
    cache = os.path.join(DATA, f"iv_{site}.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    frames = []
    y0, y1 = int(START[:4]), int(END[:4])
    for y in range(y0, y1 + 1):
        s = f"{y}-01-01" if y > y0 else START
        e = f"{y}-12-31"
        try:
            df, _ = nwis.get_iv(sites=site, start=s, end=e,
                                parameterCd=["00060", "00065"])
        except Exception as ex:
            print(f"    {site} {y}: {ex}")
            continue
        if df is None or len(df) == 0:
            continue
        frames.append(_parse_iv(df))
    if not frames:
        return pd.DataFrame(columns=["Q_cfs", "stage_ft"])
    iv = pd.concat(frames).sort_index()
    iv = iv[~iv.index.duplicated(keep="first")]
    iv.index.name = "datetime"
    iv.to_parquet(cache)
    return iv

# ---------------------------------------------------------------- hydrology
def eckhardt(Q: np.ndarray, a: float = A_HOURLY, bfi_max: float = BFI_MAX) -> np.ndarray:
    """Eckhardt (2005) recursive baseflow filter. Q must be gap-free (interpolated)."""
    b = np.zeros_like(Q, dtype=float)
    b[0] = Q[0] if np.isfinite(Q[0]) else 0.0
    denom = 1.0 - a * bfi_max
    for t in range(1, len(Q)):
        b[t] = ((1 - bfi_max) * a * b[t - 1] + (1 - a) * bfi_max * Q[t]) / denom
        if b[t] > Q[t]:
            b[t] = Q[t]
    return b


def to_hourly(iv: pd.DataFrame) -> pd.DataFrame:
    h = iv.resample("1h").mean()
    return h


def _recession_k(q: np.ndarray) -> float:
    """Slope of ln(Q) over the falling limb (per hour); NaN if not enough points."""
    q = q[np.isfinite(q) & (q > 0)]
    if len(q) < 4:
        return np.nan
    t = np.arange(len(q))
    try:
        k = np.polyfit(t, np.log(q), 1)[0]      # negative for a recession
        return -k
    except Exception:
        return np.nan


def extract_events(site: str, h: pd.DataFrame) -> pd.DataFrame:
    from scipy.signal import find_peaks, peak_widths
    Q = h["Q_cfs"].copy()
    valid = Q.notna()
    if valid.sum() < 100:
        return pd.DataFrame()
    # gap-free copy for the filter; remember where data was real
    Qi = Q.interpolate(limit_direction="both").to_numpy()
    base = eckhardt(Qi)
    quick = np.clip(Qi - base, 0, None)
    flood_thr = np.nanpercentile(Q.to_numpy(), 99)     # per-gauge "flood-relevant" level

    qmax = np.nanmax(quick)
    if not np.isfinite(qmax) or qmax <= 0:
        return pd.DataFrame()
    prom = max(qmax * MIN_PEAK_QUICK_FRAC, np.nanpercentile(quick, 95))
    peaks, props = find_peaks(quick, prominence=prom, distance=MIN_SEP_H)
    if len(peaks) == 0:
        return pd.DataFrame()

    # event base extent (rel_height=0.90) and FWHM (rel_height=0.50)
    w_base = peak_widths(quick, peaks, rel_height=REL_BASE)
    w_half = peak_widths(quick, peaks, rel_height=REL_FWHM)
    left_ips, right_ips = w_base[2], w_base[3]
    fwhm = w_half[0]                                   # hours

    idx = h.index
    real = valid.to_numpy()
    rows = []
    for k, pk in enumerate(peaks):
        i0 = int(np.floor(left_ips[k])); i1 = int(np.ceil(right_ips[k]))
        i0 = max(i0, 0); i1 = min(i1, len(Qi) - 1)
        if i1 <= i0:
            continue
        win = slice(i0, i1 + 1)
        # reject events sitting mostly on interpolated gaps
        if real[win].mean() < (1 - MAX_GAP_FRAC):
            continue
        t0, tp, t1 = idx[i0], idx[pk], idx[i1]
        Qwin = Q.iloc[win].to_numpy()                  # real (NaN-bearing) discharge
        stage_win = h["stage_ft"].iloc[win].to_numpy()
        quick_win = quick[win]
        peak_stage = np.nanmax(stage_win) if np.isfinite(stage_win).any() else np.nan
        # R-B flashiness over the event window
        dq = np.abs(np.diff(Qwin[np.isfinite(Qwin)]))
        sq = np.nansum(Qwin)
        rb = dq.sum() / sq if sq > 0 else np.nan
        rows.append(dict(
            site_no=site,
            t_start=t0, t_peak=tp, t_end=t1,
            dur_hr=(t1 - t0).total_seconds() / 3600.0,
            time_to_peak_hr=(tp - t0).total_seconds() / 3600.0,
            hydro_width_hr=float(fwhm[k]),
            Qp_cfs=float(np.nanmax(Qwin)),
            Qp_quick_cfs=float(np.nanmax(quick_win)),
            base_at_peak_cfs=float(base[pk]),
            peak_stage_ft=float(peak_stage) if np.isfinite(peak_stage) else np.nan,
            recession_k_per_hr=_recession_k(Qi[pk:i1 + 1]),
            recession_time_hr=(t1 - tp).total_seconds() / 3600.0,
            quick_vol_m3=float(np.nansum(quick_win) * 3600.0 * CFS_TO_CMS),
            total_vol_m3=float(np.nansum(Qwin) * 3600.0 * CFS_TO_CMS),
            rb_flashiness=float(rb) if np.isfinite(rb) else np.nan,
            dur_above_q99_hr=float(np.nansum(Qwin > flood_thr)),
            peak_month=int(tp.month), peak_year=int(tp.year),
        ))
    ev = pd.DataFrame(rows)
    return ev

# ---------------------------------------------------------------- QA figure
def qa_figure(site: str, h: pd.DataFrame, ev: pd.DataFrame, year=2018):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sub = h.loc[f"{year}"]
    if len(sub) == 0:
        return
    Qi = sub["Q_cfs"].interpolate(limit_direction="both").to_numpy()
    base = eckhardt(Qi)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(sub.index, sub["Q_cfs"], lw=0.6, color="#1f77b4", label="Q (hourly)")
    ax.plot(sub.index, base, lw=0.8, color="#d62728", label="baseflow (Eckhardt)")
    evy = ev[(ev.t_peak >= sub.index[0]) & (ev.t_peak <= sub.index[-1])]
    for _, r in evy.iterrows():
        ax.axvspan(r.t_start, r.t_end, color="orange", alpha=0.18)
        ax.plot(r.t_peak, r.Qp_cfs, "v", color="k", ms=5)
    ax.set_title(f"{site} — {year}: {len(evy)} events (of {len(ev)} total {ev.peak_year.min()}-{ev.peak_year.max()})")
    ax.set_ylabel("Q (cfs)"); ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"step1_events_qa_{site}.png"), dpi=120)
    plt.close(fig)

# ---------------------------------------------------------------- main
def run(sites):
    summary = []
    all_ev = []
    for site in sites:
        print(f"[{site}] pulling IV ...", flush=True)
        iv = pull_iv(site)
        if len(iv) == 0:
            print(f"  no IV data"); continue
        h = to_hourly(iv)
        ev = extract_events(site, h)
        n = len(ev)
        stage_n = int(ev["peak_stage_ft"].notna().sum()) if n else 0
        print(f"  IV rows={len(iv):,}  span={iv.index.min().date()}..{iv.index.max().date()}"
              f"  events={n}  (with stage={stage_n})", flush=True)
        if n:
            ev.to_parquet(os.path.join(DATA, f"events_{site}.parquet"))
            all_ev.append(ev)
            qa_figure(site, h, ev)
            summary.append(dict(site_no=site, iv_rows=len(iv),
                                span_start=str(iv.index.min().date()),
                                span_end=str(iv.index.max().date()),
                                n_events=n, n_with_stage=stage_n,
                                med_dur_hr=round(ev.dur_hr.median(), 1),
                                med_ttp_hr=round(ev.time_to_peak_hr.median(), 1),
                                med_Qp_cfs=round(ev.Qp_cfs.median(), 1)))
    if all_ev:
        allev = pd.concat(all_ev, ignore_index=True)
        allev.to_csv(os.path.join(OUT, "events_all.csv"), index=False)
        sdf = pd.DataFrame(summary)
        sdf.to_csv(os.path.join(OUT, "step1_summary.csv"), index=False)
        print("\n=== STEP 1 SUMMARY ===")
        print(sdf.to_string(index=False))
        print(f"\nTotal events across {len(sdf)} gauges: {len(allev)}")


if __name__ == "__main__":
    arg = sys.argv[1:] if len(sys.argv) > 1 else SITES
    run(arg)
