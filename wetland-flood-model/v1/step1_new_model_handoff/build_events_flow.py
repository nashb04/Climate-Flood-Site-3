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
import argparse
import os, sys, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

try:
    import dataretrieval.nwis as nwis
except ImportError:  # RREDI/new-model mode can run from cached CSVs without dataretrieval.
    nwis = None

# ---------------------------------------------------------------- config
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT  = os.path.join(HERE, "outputs")
os.makedirs(DATA, exist_ok=True); os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(OUT, "_mplconfig"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

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
FT3_TO_M3     = 0.0283168

# Optional New Model / RREDI-style Step 1 defaults. The original method remains the
# default CLI behavior; these settings are used only with --method rredi.
RREDI_START_YEAR = 2001
RREDI_END_YEAR   = 2023
RAW_DISCHARGE_DIRNAME = "usgs_discharge_15min_raw"


@dataclass
class RrediEventConfig:
    """Thresholds ported from New Model/mark_model_pipeline.py step 1."""
    baseline_window: str = "7D"
    baseline_min_periods: int = 10
    start_ratio: float = 1.20
    end_ratio: float = 1.05
    min_peak_ratio: float = 1.30
    min_duration_hours: float = 6.0
    end_hold_hours: float = 6.0
    min_separation_hours: float = 12.0
    max_gap_hours: float = 2.0
    min_data_coverage: float = 0.80

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
    if nwis is None:
        raise ImportError(
            "dataretrieval is required for the original NWIS-pull workflow. "
            "Install it or run with --method rredi using cached New Model discharge files."
        )
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


# ---------------------------------------------------------------- New Model / RREDI-style Step 1
def _default_rredi_discharge_dir() -> str:
    """Find the cached New Model raw discharge directory in a handoff or merged tree."""
    candidates = [
        os.path.join(HERE, RAW_DISCHARGE_DIRNAME),
        os.path.join(HERE, "New Model", RAW_DISCHARGE_DIRNAME),
        os.path.join(os.path.dirname(HERE), "New Model", RAW_DISCHARGE_DIRNAME),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]


def _yearly_dir(raw_dir: str) -> str:
    raw = Path(raw_dir).expanduser().resolve()
    return str(raw if raw.name == "yearly" else raw / "yearly")


def _to_utc_datetime(series: pd.Series) -> pd.Series:
    """Treat New Model raw timestamps as USGS local Central time, then convert to UTC."""
    dt = pd.to_datetime(series, errors="coerce")
    try:
        if dt.dt.tz is None:
            return dt.dt.tz_localize(
                "America/Chicago",
                ambiguous="NaT",
                nonexistent="shift_forward",
            ).dt.tz_convert("UTC")
        return dt.dt.tz_convert("UTC")
    except AttributeError:
        # Mixed aware/naive values are rare but can happen after CSV round-trips.
        dt = pd.to_datetime(series.astype(str), errors="coerce", utc=True)
        return dt


def load_rredi_discharge_yearly_files(
    raw_dir: str,
    start_year: int,
    end_year: int,
    sites: List[str],
) -> pd.DataFrame:
    """Load New Model yearly raw discharge CSVs and keep the original 9-site contract."""
    yearly = _yearly_dir(raw_dir)
    wanted = {str(s).zfill(8) for s in sites}
    frames = []
    missing = []
    for year in range(start_year, end_year + 1):
        path = os.path.join(yearly, f"usgs_discharge_15min_raw_{year}.csv.gz")
        if not os.path.exists(path):
            missing.append(year)
            continue
        df = pd.read_csv(path, dtype={"gage_id": str, "site_no": str}, low_memory=False)
        if "gage_id" not in df.columns and "site_no" in df.columns:
            df = df.rename(columns={"site_no": "gage_id"})
        if "discharge_cfs" not in df.columns and "Q_cfs" in df.columns:
            df = df.rename(columns={"Q_cfs": "discharge_cfs"})
        needed = {"gage_id", "datetime", "discharge_cfs"}
        if not needed.issubset(df.columns):
            raise ValueError(f"{path} is missing {sorted(needed - set(df.columns))}")
        df["gage_id"] = df["gage_id"].astype(str).str.strip().str.zfill(8)
        df = df[df["gage_id"].isin(wanted)].copy()
        if df.empty:
            continue
        df["datetime"] = _to_utc_datetime(df["datetime"])
        df["discharge_cfs"] = pd.to_numeric(df["discharge_cfs"], errors="coerce")
        df = df.dropna(subset=["gage_id", "datetime", "discharge_cfs"])
        frames.append(df[["gage_id", "datetime", "discharge_cfs"]])
    if missing:
        print(f"  missing yearly raw files: {missing[0]}..{missing[-1]}" if len(missing) > 2 else f"  missing yearly raw files: {missing}")
    if not frames:
        raise FileNotFoundError(f"No usable New Model discharge files found in {yearly}")
    out = pd.concat(frames, ignore_index=True)
    out = out.groupby(["gage_id", "datetime"], as_index=False)["discharge_cfs"].mean()
    out = out.sort_values(["gage_id", "datetime"]).reset_index(drop=True)
    return out


def trapezoid_area(y: np.ndarray, x: np.ndarray) -> float:
    if len(y) < 2:
        return np.nan
    try:
        return float(np.trapezoid(y, x))
    except AttributeError:
        return float(np.trapz(y, x))


def prepare_rredi_discharge(discharge_df: pd.DataFrame) -> pd.DataFrame:
    df = discharge_df.copy()
    df["gage_id"] = df["gage_id"].astype(str).str.strip().str.zfill(8)
    df["datetime"] = _to_utc_datetime(df["datetime"])
    df["discharge_cfs"] = pd.to_numeric(df["discharge_cfs"], errors="coerce")
    df = df.dropna(subset=["gage_id", "datetime", "discharge_cfs"])
    df = df.groupby(["gage_id", "datetime"], as_index=False)["discharge_cfs"].mean()
    return df.sort_values(["gage_id", "datetime"]).reset_index(drop=True)


def compute_rredi_baseline_for_gage(gage_df: pd.DataFrame, cfg: RrediEventConfig) -> pd.DataFrame:
    g = gage_df.copy().sort_values("datetime").set_index("datetime")
    g["baseline"] = g["discharge_cfs"].rolling(
        cfg.baseline_window,
        center=True,
        min_periods=cfg.baseline_min_periods,
    ).median()
    g["ratio"] = g["discharge_cfs"] / g["baseline"]
    g.loc[g["baseline"] <= 0, "ratio"] = np.nan
    g["excess_q"] = (g["discharge_cfs"] - g["baseline"]).clip(lower=0)
    return g


def detect_rredi_events_from_ratio(g: pd.DataFrame, cfg: RrediEventConfig) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    events: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    in_event = False
    event_start: Optional[pd.Timestamp] = None
    possible_end: Optional[pd.Timestamp] = None
    last_below_end: Optional[pd.Timestamp] = None

    for t, row in g.iterrows():
        ratio = row.get("ratio", np.nan)
        if pd.isna(ratio):
            continue
        if ratio <= cfg.end_ratio:
            last_below_end = t
        if not in_event and ratio >= cfg.start_ratio:
            in_event = True
            event_start = last_below_end if last_below_end is not None and last_below_end < t else t
            possible_end = None
        elif in_event:
            if ratio <= cfg.end_ratio:
                if possible_end is None:
                    possible_end = t
                else:
                    below_hours = (t - possible_end).total_seconds() / 3600.0
                    if below_hours >= cfg.end_hold_hours:
                        event_end = possible_end
                        if event_start is not None and event_end > event_start:
                            events.append((event_start, event_end))
                        in_event = False
                        event_start = None
                        possible_end = None
            else:
                possible_end = None
    return events


def merge_close_rredi_events(
    events: List[Tuple[pd.Timestamp, pd.Timestamp]],
    cfg: RrediEventConfig,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not events:
        return []
    events = sorted(events, key=lambda x: x[0])
    merged = [events[0]]
    for start, end in events[1:]:
        prev_start, prev_end = merged[-1]
        gap_hours = (start - prev_end).total_seconds() / 3600.0
        if gap_hours < cfg.min_separation_hours:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def summarize_rredi_event(
    g: pd.DataFrame,
    site: str,
    event_seq: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: RrediEventConfig,
    flood_thr: float,
) -> Optional[Dict]:
    seg = g.loc[start:end].copy()
    if len(seg) < 2:
        return None
    duration_hours = (end - start).total_seconds() / 3600.0
    if duration_hours < cfg.min_duration_hours:
        return None

    dt = seg.index.to_series().diff().dt.total_seconds()
    median_dt = dt.median()
    if pd.isna(median_dt) or median_dt <= 0:
        return None
    max_gap_hours = dt.max() / 3600.0
    expected_n = duration_hours * 3600.0 / median_dt + 1.0
    data_coverage = len(seg) / expected_n

    peak_time = seg["discharge_cfs"].idxmax()
    q_peak = float(seg["discharge_cfs"].max())
    q_start = float(seg["discharge_cfs"].iloc[0])
    q_end = float(seg["discharge_cfs"].iloc[-1])
    baseline_peak = float(seg.loc[peak_time, "baseline"])
    peak_ratio = q_peak / baseline_peak if baseline_peak > 0 else np.nan
    if pd.isna(peak_ratio) or peak_ratio < cfg.min_peak_ratio:
        return None

    time_to_peak_hours = (peak_time - start).total_seconds() / 3600.0
    recession_time_hours = (end - peak_time).total_seconds() / 3600.0
    x_seconds = (seg.index - seg.index[0]).total_seconds().to_numpy()
    q = seg["discharge_cfs"].to_numpy(dtype=float)
    baseline = seg["baseline"].to_numpy(dtype=float)
    excess_q = np.maximum(q - baseline, 0)
    total_volume = trapezoid_area(q, x_seconds)
    total_excess_volume = trapezoid_area(excess_q, x_seconds)
    peak_excess_q = float(np.nanmax(excess_q)) if len(excess_q) else np.nan

    if pd.notna(peak_excess_q) and peak_excess_q > 0:
        half_peak_threshold = 0.5 * peak_excess_q
        mask = excess_q >= half_peak_threshold
        if mask.any():
            half_times = seg.index[mask]
            width_half_peak_hours = (half_times[-1] - half_times[0]).total_seconds() / 3600.0
        else:
            width_half_peak_hours = np.nan
    else:
        width_half_peak_hours = np.nan

    finite_q = q[np.isfinite(q)]
    dq = np.abs(np.diff(finite_q))
    sq = np.nansum(finite_q)
    rb = dq.sum() / sq if sq > 0 else np.nan

    qc_flags: List[str] = []
    if max_gap_hours > cfg.max_gap_hours:
        qc_flags.append("large_streamflow_gap")
    if data_coverage < cfg.min_data_coverage:
        qc_flags.append("low_data_coverage")
    if recession_time_hours <= 0:
        qc_flags.append("no_recession")

    peak_pos = seg.index.get_loc(peak_time)
    return dict(
        site_no=site,
        t_start=start,
        t_peak=peak_time,
        t_end=end,
        dur_hr=duration_hours,
        time_to_peak_hr=time_to_peak_hours,
        hydro_width_hr=float(width_half_peak_hours) if pd.notna(width_half_peak_hours) else np.nan,
        Qp_cfs=q_peak,
        Qp_quick_cfs=peak_excess_q,
        base_at_peak_cfs=baseline_peak,
        peak_stage_ft=np.nan,
        recession_k_per_hr=_recession_k(q[peak_pos:]),
        recession_time_hr=recession_time_hours,
        quick_vol_m3=float(total_excess_volume * FT3_TO_M3) if np.isfinite(total_excess_volume) else np.nan,
        total_vol_m3=float(total_volume * FT3_TO_M3) if np.isfinite(total_volume) else np.nan,
        rb_flashiness=float(rb) if np.isfinite(rb) else np.nan,
        dur_above_q99_hr=float(np.nansum(q > flood_thr) * median_dt / 3600.0),
        peak_month=int(peak_time.month),
        peak_year=int(peak_time.year),
        rredi_event_seq=event_seq,
        rredi_q_start_cfs=q_start,
        rredi_q_end_cfs=q_end,
        rredi_peak_ratio=peak_ratio,
        rredi_n_obs=int(len(seg)),
        rredi_median_dt_seconds=float(median_dt),
        rredi_max_gap_hours=float(max_gap_hours),
        rredi_data_coverage=float(data_coverage),
        rredi_qc_pass=len(qc_flags) == 0,
        rredi_qc_flags=";".join(qc_flags),
    )


def extract_rredi_events_for_site(
    site: str,
    discharge_df: pd.DataFrame,
    cfg: RrediEventConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    discharge = discharge_df[discharge_df["gage_id"] == site].copy()
    if discharge.empty:
        return pd.DataFrame(), pd.DataFrame(), {"raw": 0, "merged": 0}
    g = compute_rredi_baseline_for_gage(discharge, cfg)
    raw_events = detect_rredi_events_from_ratio(g, cfg)
    merged_events = merge_close_rredi_events(raw_events, cfg)
    flood_thr = np.nanpercentile(g["discharge_cfs"].to_numpy(), 99)
    rows = []
    for event_seq, (start, end) in enumerate(merged_events, start=1):
        summary = summarize_rredi_event(g, site, event_seq, start, end, cfg, flood_thr)
        if summary is not None:
            rows.append(summary)
    return pd.DataFrame(rows), g, {"raw": len(raw_events), "merged": len(merged_events)}


def qa_rredi_figure(site: str, g: pd.DataFrame, ev: pd.DataFrame, year=2018):
    if g.empty or ev.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as ex:
        print(f"  QA plot skipped: {ex}")
        return
    try:
        sub = g.loc[f"{year}"]
    except KeyError:
        return
    if len(sub) == 0:
        return
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(sub.index, sub["discharge_cfs"], lw=0.6, color="#1f77b4", label="Q (raw)")
    ax.plot(sub.index, sub["baseline"], lw=0.8, color="#d62728", label="rolling median baseline")
    evy = ev[(ev.t_peak >= sub.index[0]) & (ev.t_peak <= sub.index[-1])]
    for _, r in evy.iterrows():
        ax.axvspan(r.t_start, r.t_end, color="orange", alpha=0.18)
        ax.plot(r.t_peak, r.Qp_cfs, "v", color="k", ms=5)
    ax.set_title(f"{site} - {year}: {len(evy)} RREDI-style events (of {len(ev)} total)")
    ax.set_ylabel("Q (cfs)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"step1_events_qa_{site}.png"), dpi=120)
    plt.close(fig)

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


def write_site_events(ev: pd.DataFrame, site: str) -> str:
    """Write the original Step 1 parquet contract, with CSV fallback for lean envs."""
    path = os.path.join(DATA, f"events_{site}.parquet")
    try:
        ev.to_parquet(path)
        return path
    except ImportError as ex:
        csv_path = os.path.join(DATA, f"events_{site}.csv.gz")
        ev.to_csv(csv_path, index=False, compression="gzip")
        print(
            f"  parquet engine missing; wrote {csv_path} instead. "
            "Install pyarrow or fastparquet to produce the exact Step 2 parquet input."
        )
        print(f"  parquet error: {str(ex).splitlines()[0]}")
        return csv_path

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
            write_site_events(ev, site)
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


def run_rredi(
    sites,
    raw_dir: str,
    start_year: int,
    end_year: int,
    cfg: RrediEventConfig,
    qc_pass_only: bool = False,
):
    sites = [str(s).zfill(8) for s in sites]
    print(f"[RREDI] loading cached New Model discharge {start_year}-{end_year} ...", flush=True)
    discharge = load_rredi_discharge_yearly_files(raw_dir, start_year, end_year, sites)
    discharge = prepare_rredi_discharge(discharge)
    print(
        f"  rows={len(discharge):,} sites={discharge.gage_id.nunique()} "
        f"span={discharge.datetime.min()}..{discharge.datetime.max()}",
        flush=True,
    )

    summary = []
    all_ev = []
    for site in sites:
        print(f"[{site}] extracting RREDI-style events ...", flush=True)
        ev, processed, counts = extract_rredi_events_for_site(site, discharge, cfg)
        if qc_pass_only and len(ev):
            ev = ev[ev["rredi_qc_pass"]].copy()
        n = len(ev)
        qc_n = int(ev["rredi_qc_pass"].sum()) if n else 0
        print(
            f"  raw={counts['raw']} merged={counts['merged']} kept={n} "
            f"qc_pass={qc_n}",
            flush=True,
        )
        if n:
            write_site_events(ev, site)
            all_ev.append(ev)
            qa_rredi_figure(site, processed, ev)
            summary.append(dict(
                site_no=site,
                iv_rows=int((discharge["gage_id"] == site).sum()),
                span_start=str(discharge.loc[discharge["gage_id"] == site, "datetime"].min().date()),
                span_end=str(discharge.loc[discharge["gage_id"] == site, "datetime"].max().date()),
                n_events=n,
                n_with_stage=0,
                rredi_qc_pass_events=qc_n,
                med_dur_hr=round(ev.dur_hr.median(), 1),
                med_ttp_hr=round(ev.time_to_peak_hr.median(), 1),
                med_Qp_cfs=round(ev.Qp_cfs.median(), 1),
            ))
    if all_ev:
        allev = pd.concat(all_ev, ignore_index=True)
        allev.to_csv(os.path.join(OUT, "events_all.csv"), index=False)
        sdf = pd.DataFrame(summary)
        sdf.to_csv(os.path.join(OUT, "step1_summary.csv"), index=False)
        print("\n=== STEP 1 SUMMARY (RREDI-style New Model mode) ===")
        print(sdf.to_string(index=False))
        print(f"\nTotal events across {len(sdf)} gauges: {len(allev)}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 1 event extraction. Default preserves the original workflow; "
                    "--method rredi enables the New Model step-1 detector."
    )
    parser.add_argument("sites", nargs="*", help="Optional USGS site numbers. Default: original 9 MMSD gauges.")
    parser.add_argument("--method", choices=["original", "rredi"], default="original")
    parser.add_argument("--rredi-discharge-dir", default=None,
                        help="New Model raw discharge directory containing yearly/*.csv.gz.")
    parser.add_argument("--start-year", type=int, default=RREDI_START_YEAR,
                        help="RREDI mode only. Default aligns with downstream Stage IV-era workflow.")
    parser.add_argument("--end-year", type=int, default=RREDI_END_YEAR,
                        help="RREDI mode only. Use 2023 for bundled New Model outputs.")
    parser.add_argument("--baseline-window", default="7D", help="RREDI rolling median baseline window.")
    parser.add_argument("--start-ratio", type=float, default=1.20, help="RREDI event start Q/baseline threshold.")
    parser.add_argument("--end-ratio", type=float, default=1.05, help="RREDI event end Q/baseline threshold.")
    parser.add_argument("--min-peak-ratio", type=float, default=1.30, help="RREDI minimum peak Q/baseline ratio.")
    parser.add_argument("--rredi-qc-pass-only", action="store_true",
                        help="Drop events with large streamflow gaps, low coverage, or no recession.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    selected_sites = args.sites if args.sites else SITES
    if args.method == "original":
        run(selected_sites)
    else:
        cfg = RrediEventConfig(
            baseline_window=args.baseline_window,
            start_ratio=args.start_ratio,
            end_ratio=args.end_ratio,
            min_peak_ratio=args.min_peak_ratio,
        )
        run_rredi(
            selected_sites,
            raw_dir=args.rredi_discharge_dir or _default_rredi_discharge_dir(),
            start_year=args.start_year,
            end_year=args.end_year,
            cfg=cfg,
            qc_pass_only=args.rredi_qc_pass_only,
        )
