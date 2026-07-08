#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mark model end-to-end pipeline for Milwaukee Basin.

This script is designed to be placed in:
    /Users/zhangjiaxuan/Desktop/Data+/New Model/

It can:
1. Discover USGS discharge gages in a HUC, default Milwaukee HUC 04040003.
2. Download raw instantaneous USGS discharge, parameter 00060, by year.
3. Load 800m precipitation files, one year per .csv/.csv.gz file.
4. Extract RREDI-style storm-event hydrographs from raw discharge.
5. Calculate event metrics: peak, log peak, duration, volume, time to peak, etc.
6. Merge precipitation to each event.
7. Merge wetland/runoff basin covariates if provided.
8. Run Mark's basic event-level models.

Main outputs:
    outputs/events/rredi_like_hydrograph_events_<start>_<end>.csv.gz
    outputs/models/mark_model_final_panel.csv.gz
    outputs/models/model1_basic_peak.txt
    outputs/models/model2_gageFE_interaction_peak.txt
    outputs/models/model_coefficients.csv
    outputs/templates/gage_covariates_template.csv

Required Python packages:
    pandas numpy requests matplotlib statsmodels

Notes:
- Discharge parameter 00060 = discharge in cubic feet per second.
- This keeps raw instantaneous observations. It does NOT daily-average discharge.
- If your precipitation file is grid-cell level and has no gage_id, the script first
  uses the Milwaukee-wide daily mean precipitation as a pilot input. Later you can
  replace this with basin/gage-specific precipitation.
- If no covariates file is provided, the script creates a template and stops before
  regression, because Mark's model needs wetland_effectiveness and runoff_propensity.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import requests
except ImportError as e:
    raise ImportError("Please install requests: pip install requests") from e

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import statsmodels.formula.api as smf
except ImportError:
    smf = None


# =============================================================================
# 0. CONFIG
# =============================================================================

@dataclass
class EventConfig:
    # Rolling baseline settings
    baseline_window: str = "7D"
    baseline_min_periods: int = 10

    # Event detection thresholds
    start_ratio: float = 1.20
    end_ratio: float = 1.05
    min_peak_ratio: float = 1.30

    # Event quality filters
    min_duration_hours: float = 6.0
    end_hold_hours: float = 6.0
    min_separation_hours: float = 12.0
    max_gap_hours: float = 2.0
    min_data_coverage: float = 0.80

    # Precip matching
    precip_pre_days: int = 1
    precip_post_days: int = 0
    antecedent_days_1: int = 7
    antecedent_days_2: int = 14


# USGS constants
USGS_SITE_SERVICE = "https://waterservices.usgs.gov/nwis/site/"
USGS_IV_SERVICE = "https://waterservices.usgs.gov/nwis/iv/"
USGS_OLD_UV_SERVICE = "https://nwis.waterdata.usgs.gov/usa/nwis/uv/"
PARAM_CD_DISCHARGE = "00060"


# =============================================================================
# 1. GENERAL HELPERS
# =============================================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def print_header(msg: str) -> None:
    print("\n" + "=" * 88)
    print(msg)
    print("=" * 88)


def infer_year_from_name(path: Path) -> Optional[int]:
    m = re.search(r"(19\d{2}|20\d{2})", path.name)
    return int(m.group(1)) if m else None


def safe_read_csv(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - s.mean(skipna=True)) / sd


# =============================================================================
# 2. USGS DOWNLOAD HELPERS
# =============================================================================

def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "DataPlus-MarkModel-Milwaukee/1.0",
        "Accept-Encoding": "gzip, deflate",
    })
    return session


def request_text(
    session: requests.Session,
    url: str,
    params: Dict,
    max_retries: int = 4,
    timeout: int = 120,
    sleep_seconds: float = 0.6,
) -> Tuple[Optional[str], int, str, Optional[str]]:
    last_error = None
    final_url = ""
    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(url, params=params, timeout=timeout)
            final_url = r.url
            if r.status_code == 404:
                return None, 404, final_url, "404_no_data"
            if r.status_code == 400:
                # Return the body because USGS sometimes explains the bad parameter.
                return r.text, 400, final_url, f"400_bad_request: {r.text[:400]}"
            r.raise_for_status()
            if not r.text or len(r.text.strip()) == 0:
                return None, r.status_code, final_url, "empty_response"
            return r.text, r.status_code, final_url, None
        except Exception as e:
            last_error = str(e)
            wait = sleep_seconds * attempt * 2
            print(f"    request failed attempt {attempt}/{max_retries}: {last_error}")
            time.sleep(wait)
    return None, -1, final_url, last_error


def read_usgs_rdb(text: Optional[str]) -> pd.DataFrame:
    if text is None:
        return pd.DataFrame()
    lines: List[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            continue
        lines.append(line)
    if len(lines) < 2:
        return pd.DataFrame()
    try:
        df = pd.read_csv(io.StringIO("\n".join(lines)), sep="\t", dtype=str)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    # Drop USGS RDB type row, usually values like 5s, 15s, 20d.
    first_col = df.columns[0]
    df = df[~df[first_col].astype(str).str.match(r"^\d+[a-zA-Z]$", na=False)]
    if "agency_cd" in df.columns:
        df = df[df["agency_cd"].astype(str) != "agency_cd"]
        df = df[df["agency_cd"].astype(str) != "5s"]
    if "site_no" in df.columns:
        df = df[df["site_no"].astype(str) != "site_no"]
        df = df[df["site_no"].astype(str) != "15s"]
    return df.reset_index(drop=True)


def discover_usgs_sites_for_huc(
    huc: str,
    output_path: Path,
    sleep_seconds: float = 0.6,
) -> pd.DataFrame:
    """
    Discover USGS stream sites in HUC with discharge parameter 00060.
    Uses query variants because USGS Site Service rejects some parameter combos.
    """
    session = make_session()

    query_variants = [
        {
            "format": "rdb",
            "huc": huc,
            "siteType": "ST",
            "hasDataTypeCd": "iv",
            "parameterCd": PARAM_CD_DISCHARGE,
            "siteStatus": "all",
            "siteOutput": "basic",
            "seriesCatalogOutput": "true",
        },
        {
            "format": "rdb",
            "huc": huc,
            "siteType": "ST",
            "hasDataTypeCd": "uv",
            "parameterCd": PARAM_CD_DISCHARGE,
            "siteStatus": "all",
            "siteOutput": "basic",
            "seriesCatalogOutput": "true",
        },
        {
            "format": "rdb",
            "huc": huc,
            "siteType": "ST",
            "parameterCd": PARAM_CD_DISCHARGE,
            "siteStatus": "all",
            "siteOutput": "basic",
            "seriesCatalogOutput": "true",
        },
        {
            "format": "rdb",
            "huc": huc,
            "siteType": "ST",
            "parameterCd": PARAM_CD_DISCHARGE,
            "siteStatus": "all",
            "siteOutput": "basic",
        },
    ]

    print_header(f"Discovering USGS discharge sites for HUC {huc}")
    raw = pd.DataFrame()
    last_url = None
    last_err = None

    for i, params in enumerate(query_variants, start=1):
        print(f"Trying site query variant {i}...")
        text, status, final_url, err = request_text(
            session,
            USGS_SITE_SERVICE,
            params,
            sleep_seconds=sleep_seconds,
        )
        last_url, last_err = final_url, err
        if err is not None:
            print(f"  Failed: {err}")
            print(f"  URL: {final_url}")
            continue
        temp = read_usgs_rdb(text)
        print(f"  Raw rows: {len(temp)}")
        print(f"  Columns: {temp.columns.tolist()}")
        if temp.empty:
            continue
        if "agency_cd" in temp.columns:
            temp = temp[temp["agency_cd"] == "USGS"].copy()
        if "parm_cd" in temp.columns:
            temp = temp[temp["parm_cd"] == PARAM_CD_DISCHARGE].copy()
        if "data_type_cd" in temp.columns:
            temp = temp[temp["data_type_cd"].isin(["iv", "uv", "rt", "id"])].copy()
        print(f"  Filtered rows: {len(temp)}")
        if not temp.empty:
            raw = temp.copy()
            break
        time.sleep(sleep_seconds)

    if raw.empty:
        raise RuntimeError(
            f"No USGS discharge sites found for HUC={huc}.\n"
            f"Last error: {last_err}\nLast URL: {last_url}"
        )

    meta_cols_preferred = [
        "agency_cd", "site_no", "station_nm", "site_tp_cd", "dec_lat_va", "dec_long_va",
        "coord_acy_cd", "dec_coord_datum_cd", "alt_va", "alt_datum_cd", "huc_cd",
        "drain_area_va",
    ]
    existing_meta_cols = [c for c in meta_cols_preferred if c in raw.columns]

    rows: List[Dict] = []
    for site_no, sub in raw.groupby("site_no"):
        row: Dict = {}
        for c in existing_meta_cols:
            vals = sub[c].dropna().astype(str).unique()
            row[c] = vals[0] if len(vals) > 0 else pd.NA
        if "begin_date" in sub.columns:
            row["begin_date"] = pd.to_datetime(sub["begin_date"], errors="coerce").min()
        else:
            row["begin_date"] = pd.NaT
        if "end_date" in sub.columns:
            end_dates = pd.to_datetime(sub["end_date"], errors="coerce")
            row["end_date"] = end_dates.max() if end_dates.notna().any() else pd.NaT
        else:
            row["end_date"] = pd.NaT
        if "data_type_cd" in sub.columns:
            row["data_type_cd_all"] = ",".join(sorted(sub["data_type_cd"].dropna().astype(str).unique()))
        else:
            row["data_type_cd_all"] = pd.NA
        if "parm_cd" in sub.columns:
            row["parm_cd_all"] = ",".join(sorted(sub["parm_cd"].dropna().astype(str).unique()))
        else:
            row["parm_cd_all"] = PARAM_CD_DISCHARGE
        row["n_series_rows"] = len(sub)
        rows.append(row)

    sites = pd.DataFrame(rows)
    if "site_no" not in sites.columns:
        raise RuntimeError("USGS site discovery did not return site_no.")
    sites = sites.rename(columns={"site_no": "gage_id"})
    sites["gage_id"] = sites["gage_id"].astype(str).str.strip()
    sites = sites.sort_values("gage_id").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sites.to_csv(output_path, index=False)

    print(f"Found {len(sites)} sites. Saved to {output_path}")
    cols = [c for c in ["gage_id", "station_nm", "begin_date", "end_date", "data_type_cd_all"] if c in sites.columns]
    print(sites[cols].head(50))
    return sites


def find_datetime_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["datetime", "dateTime", "DateTime", "date_time", "lev_dt"]:
        if c in df.columns:
            return c
    return None


def find_discharge_value_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        c_lower = c.lower()
        if c_lower.endswith("_cd"):
            continue
        if c in ["agency_cd", "site_no", "datetime", "dateTime", "tz_cd"]:
            continue
        if PARAM_CD_DISCHARGE in c:
            cols.append(c)
    return cols


def normalize_usgs_discharge(df: pd.DataFrame, source_service: str, year: int) -> pd.DataFrame:
    if df.empty or "site_no" not in df.columns:
        return pd.DataFrame()
    dt_col = find_datetime_column(df)
    if dt_col is None:
        return pd.DataFrame()
    q_cols = find_discharge_value_columns(df)
    if not q_cols:
        return pd.DataFrame()

    pieces: List[pd.DataFrame] = []
    for q_col in q_cols:
        cd_candidates = [f"{q_col}_cd", q_col.replace(PARAM_CD_DISCHARGE, f"{PARAM_CD_DISCHARGE}_cd")]
        cd_col = next((c for c in cd_candidates if c in df.columns), None)
        keep_cols = [c for c in ["agency_cd", "site_no", dt_col, "tz_cd"] if c in df.columns]
        keep_cols.append(q_col)
        if cd_col:
            keep_cols.append(cd_col)
        temp = df[keep_cols].copy()
        rename_map = {"site_no": "gage_id", dt_col: "datetime", q_col: "discharge_cfs"}
        if cd_col:
            rename_map[cd_col] = "discharge_cd"
        temp = temp.rename(columns=rename_map)
        if "agency_cd" not in temp.columns:
            temp["agency_cd"] = "USGS"
        if "tz_cd" not in temp.columns:
            temp["tz_cd"] = pd.NA
        if "discharge_cd" not in temp.columns:
            temp["discharge_cd"] = pd.NA
        temp["gage_id"] = temp["gage_id"].astype(str).str.strip()
        temp["datetime"] = pd.to_datetime(temp["datetime"], errors="coerce")
        temp["discharge_cfs"] = pd.to_numeric(temp["discharge_cfs"], errors="coerce")
        temp = temp.dropna(subset=["gage_id", "datetime", "discharge_cfs"])
        if temp.empty:
            continue
        temp["parameter_cd"] = PARAM_CD_DISCHARGE
        temp["ts_column"] = q_col
        temp["source_service"] = source_service
        temp["year"] = year
        pieces.append(temp)
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, ignore_index=True)
    out = out[[
        "agency_cd", "gage_id", "datetime", "tz_cd", "discharge_cfs", "discharge_cd",
        "parameter_cd", "ts_column", "source_service", "year",
    ]]
    out = out.sort_values(["gage_id", "datetime", "ts_column"])
    out = out.drop_duplicates(subset=["gage_id", "datetime", "ts_column"])
    return out.reset_index(drop=True)


def site_overlaps_year(site_row: pd.Series, year: int) -> bool:
    year_start = pd.Timestamp(f"{year}-01-01")
    year_end = pd.Timestamp(f"{year}-12-31 23:59:59")
    begin = pd.to_datetime(site_row.get("begin_date", pd.NaT), errors="coerce")
    end = pd.to_datetime(site_row.get("end_date", pd.NaT), errors="coerce")
    if pd.notna(begin) and year_end < begin:
        return False
    if pd.notna(end) and year_start > end:
        return False
    return True


def download_site_year_waterservices_iv(
    session: requests.Session,
    gage_id: str,
    year: int,
    sleep_seconds: float = 0.6,
) -> Tuple[pd.DataFrame, Dict]:
    params = {
        "format": "rdb",
        "sites": gage_id,
        "parameterCd": PARAM_CD_DISCHARGE,
        "startDT": f"{year}-01-01",
        "endDT": f"{year}-12-31",
        "siteStatus": "all",
    }
    text, status, final_url, err = request_text(session, USGS_IV_SERVICE, params, sleep_seconds=sleep_seconds)
    log = {
        "gage_id": gage_id, "year": year, "source_service": "waterservices_iv",
        "status_code": status, "error": err, "url": final_url,
    }
    if err is not None:
        log["n_rows"] = 0
        return pd.DataFrame(), log
    rdb = read_usgs_rdb(text)
    out = normalize_usgs_discharge(rdb, "waterservices_iv", year)
    log["n_rows"] = len(out)
    return out, log


def download_site_year_old_uv(
    session: requests.Session,
    gage_id: str,
    year: int,
    sleep_seconds: float = 0.6,
) -> Tuple[pd.DataFrame, Dict]:
    params = {
        "cb_00060": "on",
        "format": "rdb",
        "site_no": gage_id,
        "period": "",
        "begin_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
    }
    text, status, final_url, err = request_text(session, USGS_OLD_UV_SERVICE, params, sleep_seconds=sleep_seconds)
    log = {
        "gage_id": gage_id, "year": year, "source_service": "old_nwis_uv",
        "status_code": status, "error": err, "url": final_url,
    }
    if err is not None:
        log["n_rows"] = 0
        return pd.DataFrame(), log
    rdb = read_usgs_rdb(text)
    out = normalize_usgs_discharge(rdb, "old_nwis_uv", year)
    log["n_rows"] = len(out)
    return out, log


def download_site_year(
    session: requests.Session,
    gage_id: str,
    year: int,
    sleep_seconds: float = 0.6,
) -> Tuple[pd.DataFrame, List[Dict]]:
    logs: List[Dict] = []
    pieces: List[pd.DataFrame] = []
    if year >= 2008:
        df, log = download_site_year_waterservices_iv(session, gage_id, year, sleep_seconds=sleep_seconds)
        pieces.append(df)
        logs.append(log)
    elif year <= 2006:
        df, log = download_site_year_old_uv(session, gage_id, year, sleep_seconds=sleep_seconds)
        pieces.append(df)
        logs.append(log)
    else:
        # 2007 transition year: try both and deduplicate.
        df_old, log_old = download_site_year_old_uv(session, gage_id, year, sleep_seconds=sleep_seconds)
        pieces.append(df_old)
        logs.append(log_old)
        time.sleep(sleep_seconds)
        df_iv, log_iv = download_site_year_waterservices_iv(session, gage_id, year, sleep_seconds=sleep_seconds)
        pieces.append(df_iv)
        logs.append(log_iv)
    nonempty = [p for p in pieces if p is not None and len(p) > 0]
    if not nonempty:
        return pd.DataFrame(), logs
    out = pd.concat(nonempty, ignore_index=True)
    out = out.drop_duplicates(subset=["gage_id", "datetime", "ts_column"])
    out = out.sort_values(["gage_id", "datetime", "ts_column"]).reset_index(drop=True)
    return out, logs


def download_discharge_by_year(
    huc: str,
    start_year: int,
    end_year: int,
    output_dir: Path,
    overwrite: bool = False,
    sleep_seconds: float = 0.6,
) -> Path:
    print_header("USGS raw instantaneous discharge download")
    yearly_dir = ensure_dir(output_dir / "yearly")
    sites_path = output_dir / "milwaukee_usgs_discharge_sites.csv"
    log_path = output_dir / "download_log.csv"

    if sites_path.exists() and not overwrite:
        print(f"Using existing sites file: {sites_path}")
        sites = pd.read_csv(sites_path, dtype={"gage_id": str})
    else:
        sites = discover_usgs_sites_for_huc(huc=huc, output_path=sites_path, sleep_seconds=sleep_seconds)

    sites["gage_id"] = sites["gage_id"].astype(str).str.strip()
    print(f"Total sites: {len(sites)}")
    if "station_nm" in sites.columns:
        print(sites[["gage_id", "station_nm"]].head(30))
    else:
        print(sites[["gage_id"]].head(30))

    session = make_session()
    all_logs: List[Dict] = []

    for year in range(start_year, end_year + 1):
        year_out = yearly_dir / f"usgs_discharge_15min_raw_{year}.csv.gz"
        if year_out.exists() and not overwrite:
            print(f"Skipping {year}; file already exists: {year_out.name}")
            continue
        print_header(f"Downloading discharge year {year}")
        year_pieces: List[pd.DataFrame] = []
        for i, site_row in sites.iterrows():
            gage_id = str(site_row["gage_id"]).strip()
            if not site_overlaps_year(site_row, year):
                print(f"  [{i+1}/{len(sites)}] {gage_id}: skip, outside metadata period")
                all_logs.append({
                    "gage_id": gage_id, "year": year, "source_service": "metadata_skip",
                    "status_code": None, "error": "outside_metadata_period", "n_rows": 0, "url": None,
                })
                continue
            print(f"  [{i+1}/{len(sites)}] {gage_id}...")
            df_site_year, logs = download_site_year(session, gage_id, year, sleep_seconds=sleep_seconds)
            all_logs.extend(logs)
            if len(df_site_year) > 0:
                print(f"      rows: {len(df_site_year):,}")
                year_pieces.append(df_site_year)
            else:
                print("      no data")
            time.sleep(sleep_seconds)
        if year_pieces:
            year_df = pd.concat(year_pieces, ignore_index=True)
            year_df = year_df.sort_values(["gage_id", "datetime", "ts_column"])
            year_df = year_df.drop_duplicates(subset=["gage_id", "datetime", "ts_column"])
            year_df.to_csv(year_out, index=False, compression="gzip")
            print(f"Saved {len(year_df):,} rows to {year_out}")
        else:
            print(f"No data found for {year}; no yearly file written.")
        if all_logs:
            pd.DataFrame(all_logs).to_csv(log_path, index=False)
            print(f"Updated log: {log_path}")
    return yearly_dir


# =============================================================================
# 3. DISCHARGE / PRECIP LOADING
# =============================================================================

def load_discharge_yearly_files(
    yearly_dir: Path,
    start_year: int,
    end_year: int,
    sample_only: bool = False,
) -> pd.DataFrame:
    print_header("Loading raw discharge yearly files")
    pieces: List[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        path = yearly_dir / f"usgs_discharge_15min_raw_{year}.csv.gz"
        if not path.exists():
            print(f"Missing discharge file: {path.name}")
            continue
        print(f"Reading {path.name}")
        df = pd.read_csv(path, dtype={"gage_id": str}, low_memory=False)
        pieces.append(df)
        if sample_only and len(pieces) >= 2:
            break
    if not pieces:
        raise FileNotFoundError(f"No discharge files found in {yearly_dir} for {start_year}-{end_year}.")
    discharge = pd.concat(pieces, ignore_index=True)
    needed = ["gage_id", "datetime", "discharge_cfs"]
    missing = [c for c in needed if c not in discharge.columns]
    if missing:
        raise ValueError(f"Discharge files missing columns {missing}. Columns: {discharge.columns.tolist()}")
    discharge["gage_id"] = discharge["gage_id"].astype(str).str.strip()
    discharge["datetime"] = pd.to_datetime(discharge["datetime"], errors="coerce")
    discharge["discharge_cfs"] = pd.to_numeric(discharge["discharge_cfs"], errors="coerce")
    discharge = discharge.dropna(subset=["gage_id", "datetime", "discharge_cfs"])
    discharge = discharge.sort_values(["gage_id", "datetime"])
    print(f"Loaded discharge rows: {len(discharge):,}")
    print(f"Unique gages: {discharge['gage_id'].nunique()}")
    return discharge


def infer_date_column(cols: Sequence[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in cols}
    exact = ["date", "datetime", "time", "day", "system:time_start"]
    for x in exact:
        if x in lower_map:
            return lower_map[x]
    for c in cols:
        cl = c.lower()
        if "date" in cl or "time" in cl or cl in ["day"]:
            return c
    return None


def infer_precip_column(cols: Sequence[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in cols}
    exact = ["precip_mm", "precip", "prcp", "rain", "rainfall", "precipitation", "value", "mean"]
    for x in exact:
        if x in lower_map:
            return lower_map[x]
    for c in cols:
        cl = c.lower()
        if "precip" in cl or "prcp" in cl or "rain" in cl:
            return c
    # Last resort: numeric column not lat/lon/x/y.
    excluded = {"lat", "latitude", "lon", "longitude", "x", "y", "id", "gage_id", "site_no"}
    for c in cols:
        if c.lower() not in excluded:
            return c
    return None


def load_800m_precip(
    precip_dir: Path,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    print_header("Loading 800m precipitation files")
    if not precip_dir.exists():
        raise FileNotFoundError(f"Precip directory does not exist: {precip_dir}")

    all_files = sorted([p for p in precip_dir.rglob("*.csv*") if p.is_file()])
    selected: List[Path] = []
    for p in all_files:
        yr = infer_year_from_name(p)
        if yr is not None and start_year <= yr <= end_year:
            selected.append(p)
    if not selected:
        raise FileNotFoundError(f"No precip .csv/.csv.gz files found in {precip_dir} for {start_year}-{end_year}.")

    pieces: List[pd.DataFrame] = []
    for p in selected:
        print(f"Reading precip {p.name}")
        temp = safe_read_csv(p)
        if temp.empty:
            continue
        temp["source_file"] = p.name
        pieces.append(temp)
    if not pieces:
        raise RuntimeError("Precip files were found but all were empty.")
    raw = pd.concat(pieces, ignore_index=True)
    print(f"Raw precip rows: {len(raw):,}")
    print(f"Raw precip columns: {raw.columns.tolist()}")

    date_col = infer_date_column(raw.columns)
    precip_col = infer_precip_column(raw.columns)
    if date_col is None or precip_col is None:
        raise ValueError(
            "Could not infer precipitation date/value columns.\n"
            f"Columns: {raw.columns.tolist()}\n"
            "Rename columns to date and precip_mm or pass a preprocessed file."
        )

    # If date is encoded as YYYYMMDD or timestamp numeric, pd.to_datetime handles many cases.
    raw["date"] = pd.to_datetime(raw[date_col], errors="coerce").dt.floor("D")
    # Try special YYYYMMDD if direct parse failed badly.
    if raw["date"].isna().mean() > 0.8:
        raw["date"] = pd.to_datetime(raw[date_col].astype(str), format="%Y%m%d", errors="coerce").dt.floor("D")
    raw["precip_mm"] = pd.to_numeric(raw[precip_col], errors="coerce")
    raw = raw.dropna(subset=["date", "precip_mm"])

    # If precipitation has gage-specific values, preserve them; otherwise use daily Milwaukee-wide average.
    gage_candidates = ["gage_id", "site_no", "sensor_id", "station_id"]
    gage_col = next((c for c in gage_candidates if c in raw.columns), None)

    if gage_col:
        raw["gage_id"] = raw[gage_col].astype(str).str.strip()
        precip = raw.groupby(["gage_id", "date"], as_index=False)["precip_mm"].mean()
        print("Using gage-specific precipitation from column:", gage_col)
    else:
        precip = raw.groupby("date", as_index=False)["precip_mm"].mean()
        print("No gage_id in precipitation. Using daily mean across all 800m grid cells as pilot precipitation.")

    print(f"Prepared precip rows: {len(precip):,}")
    print(precip.head())
    return precip


# =============================================================================
# 4. RREDI-STYLE EVENT EXTRACTION
# =============================================================================

def trapezoid_area(y: np.ndarray, x: np.ndarray) -> float:
    if len(y) < 2:
        return np.nan
    try:
        return float(np.trapezoid(y, x))
    except AttributeError:
        return float(np.trapz(y, x))


def prepare_discharge_for_events(discharge_df: pd.DataFrame) -> pd.DataFrame:
    df = discharge_df.copy()
    df["gage_id"] = df["gage_id"].astype(str).str.strip()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["discharge_cfs"] = pd.to_numeric(df["discharge_cfs"], errors="coerce")
    df = df.dropna(subset=["gage_id", "datetime", "discharge_cfs"])
    df = df.groupby(["gage_id", "datetime"], as_index=False)["discharge_cfs"].mean()
    df = df.sort_values(["gage_id", "datetime"])
    return df


def compute_baseline_for_gage(gage_df: pd.DataFrame, cfg: EventConfig) -> pd.DataFrame:
    g = gage_df.copy().sort_values("datetime").set_index("datetime")
    g["baseline"] = g["discharge_cfs"].rolling(
        cfg.baseline_window,
        center=True,
        min_periods=cfg.baseline_min_periods,
    ).median()
    g["ratio"] = g["discharge_cfs"] / g["baseline"]
    g.loc[g["baseline"] <= 0, "ratio"] = np.nan
    g["excess_q"] = g["discharge_cfs"] - g["baseline"]
    g.loc[g["excess_q"] < 0, "excess_q"] = 0
    return g


def detect_events_from_ratio(g: pd.DataFrame, cfg: EventConfig) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
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
                    below_hours = (t - possible_end).total_seconds() / 3600
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


def merge_close_events(events: List[Tuple[pd.Timestamp, pd.Timestamp]], cfg: EventConfig) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not events:
        return []
    events = sorted(events, key=lambda x: x[0])
    merged = [events[0]]
    for start, end in events[1:]:
        prev_start, prev_end = merged[-1]
        gap_hours = (start - prev_end).total_seconds() / 3600
        if gap_hours < cfg.min_separation_hours:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def summarize_event(
    g: pd.DataFrame,
    gage_id: str,
    event_id: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: EventConfig,
) -> Optional[Dict]:
    seg = g.loc[start:end].copy()
    if len(seg) < 2:
        return None
    duration_hours = (end - start).total_seconds() / 3600
    if duration_hours < cfg.min_duration_hours:
        return None

    dt = seg.index.to_series().diff().dt.total_seconds()
    median_dt = dt.median()
    if pd.isna(median_dt) or median_dt <= 0:
        return None
    max_gap_hours = dt.max() / 3600
    expected_n = duration_hours * 3600 / median_dt + 1
    data_coverage = len(seg) / expected_n

    peak_time = seg["discharge_cfs"].idxmax()
    q_peak = float(seg["discharge_cfs"].max())
    q_start = float(seg["discharge_cfs"].iloc[0])
    q_end = float(seg["discharge_cfs"].iloc[-1])
    baseline_peak = float(seg.loc[peak_time, "baseline"])
    peak_ratio = q_peak / baseline_peak if baseline_peak > 0 else np.nan
    if pd.isna(peak_ratio) or peak_ratio < cfg.min_peak_ratio:
        return None

    time_to_peak_hours = (peak_time - start).total_seconds() / 3600
    recession_time_hours = (end - peak_time).total_seconds() / 3600
    x_seconds = (seg.index - seg.index[0]).total_seconds().to_numpy()
    q = seg["discharge_cfs"].to_numpy(dtype=float)
    baseline = seg["baseline"].to_numpy(dtype=float)
    excess_q = np.maximum(q - baseline, 0)
    total_volume = trapezoid_area(q, x_seconds)
    total_excess_volume = trapezoid_area(excess_q, x_seconds)
    peak_excess_q = float(np.nanmax(excess_q)) if len(excess_q) > 0 else np.nan

    if pd.notna(peak_excess_q) and peak_excess_q > 0:
        half_peak_threshold = 0.5 * peak_excess_q
        mask = excess_q >= half_peak_threshold
        if mask.any():
            half_times = seg.index[mask]
            width_half_peak_hours = (half_times[-1] - half_times[0]).total_seconds() / 3600
        else:
            width_half_peak_hours = np.nan
    else:
        width_half_peak_hours = np.nan

    if total_excess_volume > 0 and time_to_peak_hours > 0 and pd.notna(peak_excess_q):
        flashiness_ratio = (peak_excess_q * time_to_peak_hours * 3600) / total_excess_volume
    else:
        flashiness_ratio = np.nan

    qc_flags: List[str] = []
    if max_gap_hours > cfg.max_gap_hours:
        qc_flags.append("large_streamflow_gap")
    if data_coverage < cfg.min_data_coverage:
        qc_flags.append("low_data_coverage")
    if recession_time_hours <= 0:
        qc_flags.append("no_recession")

    return {
        "gage_id": str(gage_id),
        "event_id": event_id,
        "event_start": start,
        "event_end": end,
        "event_date": start.floor("D"),
        "peak_time": peak_time,
        "q_peak": q_peak,
        "log_q_peak": np.log(q_peak) if q_peak > 0 else np.nan,
        "q_start": q_start,
        "q_end": q_end,
        "baseline_at_peak": baseline_peak,
        "peak_ratio": peak_ratio,
        "duration_hours": duration_hours,
        "time_to_peak_hours": time_to_peak_hours,
        "recession_time_hours": recession_time_hours,
        "width_half_peak_hours": width_half_peak_hours,
        "total_volume_cfs_seconds": total_volume,
        "total_excess_volume_cfs_seconds": total_excess_volume,
        "flashiness_ratio": flashiness_ratio,
        "n_obs": len(seg),
        "median_dt_seconds": median_dt,
        "max_gap_hours": max_gap_hours,
        "data_coverage": data_coverage,
        "qc_pass": len(qc_flags) == 0,
        "qc_flags": ";".join(qc_flags),
    }


def attach_precip_to_events(events_df: pd.DataFrame, precip_df: Optional[pd.DataFrame], cfg: EventConfig) -> pd.DataFrame:
    if precip_df is None or len(precip_df) == 0:
        return events_df
    events = events_df.copy()
    p = precip_df.copy()
    p["date"] = pd.to_datetime(p["date"], errors="coerce").dt.floor("D")
    p["precip_mm"] = pd.to_numeric(p["precip_mm"], errors="coerce")
    p = p.dropna(subset=["date", "precip_mm"])
    has_gage = "gage_id" in p.columns
    if has_gage:
        p["gage_id"] = p["gage_id"].astype(str).str.strip()
        p = p.groupby(["gage_id", "date"], as_index=False)["precip_mm"].sum()
        cache = {gid: sub.set_index("date")["precip_mm"].sort_index() for gid, sub in p.groupby("gage_id")}
    else:
        p = p.groupby("date", as_index=False)["precip_mm"].mean()
        all_series = p.set_index("date")["precip_mm"].sort_index()
        cache = {"__all__": all_series}

    event_precip: List[float] = []
    max_1day: List[float] = []
    ant7: List[float] = []
    ant14: List[float] = []
    days_avail: List[int] = []

    for _, row in events.iterrows():
        if has_gage:
            s = cache.get(str(row["gage_id"]), pd.Series(dtype=float))
        else:
            s = cache["__all__"]
        event_start_day = pd.to_datetime(row["event_start"]).floor("D")
        event_end_day = pd.to_datetime(row["event_end"]).floor("D")
        p_start = event_start_day - pd.Timedelta(days=cfg.precip_pre_days)
        p_end = event_end_day + pd.Timedelta(days=cfg.precip_post_days)
        window = s[(s.index >= p_start) & (s.index <= p_end)]
        event_precip.append(float(window.sum()) if len(window) > 0 else np.nan)
        max_1day.append(float(window.max()) if len(window) > 0 else np.nan)
        days_avail.append(int(len(window)))
        a7_start = event_start_day - pd.Timedelta(days=cfg.antecedent_days_1)
        a7_end = event_start_day - pd.Timedelta(days=1)
        w7 = s[(s.index >= a7_start) & (s.index <= a7_end)]
        ant7.append(float(w7.sum()) if len(w7) > 0 else np.nan)
        a14_start = event_start_day - pd.Timedelta(days=cfg.antecedent_days_2)
        a14_end = event_start_day - pd.Timedelta(days=1)
        w14 = s[(s.index >= a14_start) & (s.index <= a14_end)]
        ant14.append(float(w14.sum()) if len(w14) > 0 else np.nan)

    events["event_precip_mm"] = event_precip
    events["max_1day_precip_mm"] = max_1day
    events["antecedent_7d_precip_mm"] = ant7
    events["antecedent_14d_precip_mm"] = ant14
    events["precip_days_available"] = days_avail
    return events


def extract_rredi_like_events(
    discharge_df: pd.DataFrame,
    precip_df: Optional[pd.DataFrame],
    cfg: EventConfig,
    output_processed_plots_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    print_header("RREDI-style hydrograph event extraction")
    discharge = prepare_discharge_for_events(discharge_df)
    all_events: List[Dict] = []
    processed_by_gage: Dict[str, pd.DataFrame] = {}

    for i, (gage_id, gage_df) in enumerate(discharge.groupby("gage_id"), start=1):
        print(f"Processing gage {i}/{discharge['gage_id'].nunique()}: {gage_id}")
        g = compute_baseline_for_gage(gage_df, cfg)
        processed_by_gage[gage_id] = g
        raw_events = detect_events_from_ratio(g, cfg)
        merged_events = merge_close_events(raw_events, cfg)
        kept = 0
        for event_id, (start, end) in enumerate(merged_events, start=1):
            summary = summarize_event(g, gage_id, event_id, start, end, cfg)
            if summary is not None:
                all_events.append(summary)
                kept += 1
        print(f"  raw events={len(raw_events)}, merged={len(merged_events)}, kept={kept}")

    events = pd.DataFrame(all_events)
    if events.empty:
        print("No events detected. Try lowering start_ratio/min_peak_ratio or increasing baseline_window.")
        return events, processed_by_gage
    events = attach_precip_to_events(events, precip_df, cfg)
    print(f"Total events kept: {len(events):,}")
    print(events.head())
    return events, processed_by_gage


# =============================================================================
# 5. COVARIATES AND MARK MODEL
# =============================================================================

def standardize_covariates(covariates: pd.DataFrame) -> pd.DataFrame:
    cov = covariates.copy()
    if "gage_id" not in cov.columns:
        # Try common names
        for c in ["site_no", "sensor_id", "station_id"]:
            if c in cov.columns:
                cov = cov.rename(columns={c: "gage_id"})
                break
    if "gage_id" not in cov.columns:
        raise ValueError("Covariates file must contain gage_id or site_no.")
    cov["gage_id"] = cov["gage_id"].astype(str).str.strip()

    # Compute simple wetland effectiveness if not already supplied.
    if "wetland_effectiveness" not in cov.columns:
        if {"wetland_area", "basin_area"}.issubset(cov.columns):
            cov["wetland_effectiveness"] = pd.to_numeric(cov["wetland_area"], errors="coerce") / pd.to_numeric(cov["basin_area"], errors="coerce")
        elif {"wetland_area_m2", "basin_area_m2"}.issubset(cov.columns):
            cov["wetland_effectiveness"] = pd.to_numeric(cov["wetland_area_m2"], errors="coerce") / pd.to_numeric(cov["basin_area_m2"], errors="coerce")

    # Compute runoff propensity R if not supplied.
    if "runoff_propensity" not in cov.columns:
        candidates = {
            "impervious": ["impervious", "impervious_share", "pct_impervious"],
            "slope": ["slope", "mean_slope"],
            "agriculture": ["agriculture", "agriculture_share", "ag_share", "crop_share"],
            "forest": ["forest", "forest_share"],
            "soil_infiltration": ["soil_infiltration", "infiltration", "ksat", "hydraulic_conductivity"],
        }
        found: Dict[str, str] = {}
        for key, names in candidates.items():
            for name in names:
                if name in cov.columns:
                    found[key] = name
                    break
        # R = +impervious +slope +agriculture -forest -soil_infiltration
        r = pd.Series(np.zeros(len(cov)), index=cov.index, dtype=float)
        used = []
        if "impervious" in found:
            r += zscore(cov[found["impervious"]]); used.append("+impervious")
        if "slope" in found:
            r += zscore(cov[found["slope"]]); used.append("+slope")
        if "agriculture" in found:
            r += zscore(cov[found["agriculture"]]); used.append("+agriculture")
        if "forest" in found:
            r -= zscore(cov[found["forest"]]); used.append("-forest")
        if "soil_infiltration" in found:
            r -= zscore(cov[found["soil_infiltration"]]); used.append("-soil_infiltration")
        if used:
            cov["runoff_propensity"] = r
            print("Computed runoff_propensity from:", ", ".join(used))

    return cov


def create_covariate_template(events: pd.DataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    gages = sorted(events["gage_id"].astype(str).unique())
    template = pd.DataFrame({
        "gage_id": gages,
        "wetland_effectiveness": np.nan,
        "runoff_propensity": np.nan,
        "wetland_storage_capacity": np.nan,
        "notes": "Fill wetland_effectiveness and runoff_propensity, then rerun with --covariates path/to/this.csv",
    })
    template.to_csv(output_path, index=False)
    print(f"Created covariates template: {output_path}")


def merge_covariates(events: pd.DataFrame, covariates_path: Optional[Path], template_path: Path) -> Optional[pd.DataFrame]:
    if covariates_path is None:
        print("No covariates file provided. Mark model needs wetland_effectiveness and runoff_propensity.")
        create_covariate_template(events, template_path)
        return None
    if not covariates_path.exists():
        print(f"Covariates file does not exist: {covariates_path}")
        create_covariate_template(events, template_path)
        return None
    print_header("Merging wetland/runoff covariates")
    cov = pd.read_csv(covariates_path, dtype={"gage_id": str}, low_memory=False)
    cov = standardize_covariates(cov)
    required = ["gage_id", "wetland_effectiveness", "runoff_propensity"]
    missing = [c for c in required if c not in cov.columns]
    if missing:
        print(f"Covariates file missing required model columns: {missing}")
        print("Columns available:", cov.columns.tolist())
        create_covariate_template(events, template_path)
        return None
    panel = events.copy()
    panel["gage_id"] = panel["gage_id"].astype(str).str.strip()
    cov = cov.drop_duplicates(subset=["gage_id"])
    panel = panel.merge(cov, on="gage_id", how="left")
    print(f"Panel rows: {len(panel):,}")
    print("Missing wetland_effectiveness:", panel["wetland_effectiveness"].isna().sum())
    print("Missing runoff_propensity:", panel["runoff_propensity"].isna().sum())
    return panel


def write_model_result(result, txt_path: Path, coef_rows: List[Dict], model_name: str) -> None:
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(str(result.summary()))
    params = result.params
    ses = result.bse
    pvals = result.pvalues
    conf = result.conf_int()
    for term in params.index:
        coef_rows.append({
            "model": model_name,
            "term": term,
            "coef": params[term],
            "std_err": ses[term],
            "p_value": pvals[term],
            "ci_low": conf.loc[term, 0],
            "ci_high": conf.loc[term, 1],
            "nobs": result.nobs,
            "rsquared": getattr(result, "rsquared", np.nan),
        })
    print(f"Saved model result: {txt_path}")


def run_mark_models(panel: pd.DataFrame, output_dir: Path) -> None:
    if smf is None:
        raise ImportError("statsmodels is required for model fitting. Install with: pip install statsmodels")
    print_header("Running Mark's event-level models")
    ensure_dir(output_dir)
    data = panel.copy()
    data["gage_id"] = data["gage_id"].astype(str)
    numeric_cols = [
        "log_q_peak", "q_peak", "event_precip_mm", "antecedent_7d_precip_mm",
        "wetland_effectiveness", "runoff_propensity", "duration_hours",
        "time_to_peak_hours", "total_excess_volume_cfs_seconds", "flashiness_ratio",
    ]
    for c in numeric_cols:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")

    # Keep events with usable core model variables.
    model_data = data.dropna(subset=["log_q_peak", "event_precip_mm", "wetland_effectiveness", "runoff_propensity"]).copy()
    if model_data.empty:
        raise ValueError("No complete rows for Mark model after dropping NA core variables.")
    if "antecedent_7d_precip_mm" not in model_data.columns:
        model_data["antecedent_7d_precip_mm"] = 0.0
    model_data["antecedent_7d_precip_mm"] = model_data["antecedent_7d_precip_mm"].fillna(0.0)

    panel_path = output_dir / "mark_model_final_panel.csv.gz"
    model_data.to_csv(panel_path, index=False, compression="gzip")
    print(f"Saved final model panel: {panel_path}")
    print(f"Model rows: {len(model_data):,}, gages: {model_data['gage_id'].nunique()}")

    coef_rows: List[Dict] = []

    # Model 1: Basic wetland effect.
    # Larger precipitation should raise peaks; wetland_effectiveness expected negative for peak.
    formula1 = "log_q_peak ~ event_precip_mm + antecedent_7d_precip_mm + wetland_effectiveness + runoff_propensity"
    m1 = smf.ols(formula1, data=model_data).fit(cov_type="HC3")
    write_model_result(m1, output_dir / "model1_basic_peak.txt", coef_rows, "model1_basic_peak")

    # Model 2: Gauge fixed effects + sensitivity interactions.
    # W main effect is absorbed by gage FE, so key term is event_precip_mm:wetland_effectiveness.
    if model_data["gage_id"].nunique() > 1:
        formula2 = "log_q_peak ~ event_precip_mm + antecedent_7d_precip_mm + event_precip_mm:wetland_effectiveness + event_precip_mm:runoff_propensity + C(gage_id)"
        m2 = smf.ols(formula2, data=model_data).fit(cov_type="HC3")
        write_model_result(m2, output_dir / "model2_gageFE_interaction_peak.txt", coef_rows, "model2_gageFE_interaction_peak")
    else:
        print("Skipping Model 2 because only one gage is available.")

    # Model 3: Full interaction, no gage FE. Useful when transferring/predicting to new basins.
    formula3 = "log_q_peak ~ event_precip_mm + antecedent_7d_precip_mm + wetland_effectiveness + runoff_propensity + event_precip_mm:wetland_effectiveness + event_precip_mm:runoff_propensity"
    m3 = smf.ols(formula3, data=model_data).fit(cov_type="HC3")
    write_model_result(m3, output_dir / "model3_full_interaction_peak.txt", coef_rows, "model3_full_interaction_peak")

    # Optional saturation model.
    if "wetland_storage_capacity" in model_data.columns:
        temp = model_data.copy()
        temp["wetland_storage_capacity"] = pd.to_numeric(temp["wetland_storage_capacity"], errors="coerce")
        # This is a rough placeholder saturation ratio unless you later compute rainfall volume.
        temp["saturation_ratio"] = temp["event_precip_mm"] / temp["wetland_storage_capacity"]
        temp = temp.replace([np.inf, -np.inf], np.nan).dropna(subset=["saturation_ratio"])
        if len(temp) > 20:
            formula4 = "log_q_peak ~ event_precip_mm + antecedent_7d_precip_mm + wetland_effectiveness + runoff_propensity + event_precip_mm:wetland_effectiveness + saturation_ratio + wetland_effectiveness:saturation_ratio"
            m4 = smf.ols(formula4, data=temp).fit(cov_type="HC3")
            write_model_result(m4, output_dir / "model4_saturation_peak.txt", coef_rows, "model4_saturation_peak")

    # Alternative outcomes using Mark's suggested metrics.
    alt_outcomes = [
        "duration_hours",
        "time_to_peak_hours",
        "total_excess_volume_cfs_seconds",
        "flashiness_ratio",
    ]
    for y in alt_outcomes:
        if y in model_data.columns:
            temp = model_data.dropna(subset=[y, "event_precip_mm", "wetland_effectiveness", "runoff_propensity"]).copy()
            if len(temp) >= 30:
                # Log volume is usually more stable.
                y_term = y
                if y == "total_excess_volume_cfs_seconds":
                    temp = temp[temp[y] > 0].copy()
                    temp["log_total_excess_volume"] = np.log(temp[y])
                    y_term = "log_total_excess_volume"
                formula_alt = f"{y_term} ~ event_precip_mm + antecedent_7d_precip_mm + wetland_effectiveness + runoff_propensity"
                res = smf.ols(formula_alt, data=temp).fit(cov_type="HC3")
                write_model_result(res, output_dir / f"model_alt_{y_term}.txt", coef_rows, f"model_alt_{y_term}")

    coef_df = pd.DataFrame(coef_rows)
    coef_path = output_dir / "model_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)
    print(f"Saved coefficient table: {coef_path}")

    # A small interpretation helper.
    interp_path = output_dir / "model_interpretation_notes.txt"
    with open(interp_path, "w", encoding="utf-8") as f:
        f.write("How to read the main Mark model terms:\n\n")
        f.write("1. wetland_effectiveness in Model 1: negative coefficient means more/effective wetlands are associated with lower peak discharge, controlling for precipitation and runoff propensity.\n")
        f.write("2. event_precip_mm:wetland_effectiveness in Model 2/3: negative coefficient means wetlands reduce the sensitivity of peak discharge to storm size, which is the peak-shaving mechanism.\n")
        f.write("3. runoff_propensity: positive coefficient means basins with more impervious/slope/agriculture and less forest/infiltration have larger peaks.\n")
        f.write("4. time_to_peak/duration outcomes: positive wetland_effectiveness coefficients are consistent with delay/attenuation.\n")
        f.write("5. Treat results as mechanistically informed associations unless the design is strengthened for causal inference.\n")
    print(f"Saved interpretation notes: {interp_path}")


# =============================================================================
# 6. PLOTS
# =============================================================================

def make_basic_plots(events: pd.DataFrame, output_dir: Path) -> None:
    if plt is None or events.empty:
        return
    ensure_dir(output_dir)
    events = events.copy()
    events["event_year"] = pd.to_datetime(events["event_start"]).dt.year

    # Event counts by year
    counts = events.groupby("event_year").size()
    fig = plt.figure(figsize=(12, 4))
    counts.plot(kind="bar")
    plt.title("Detected hydrograph events by year")
    plt.xlabel("Year")
    plt.ylabel("Number of events")
    plt.tight_layout()
    fig.savefig(output_dir / "event_counts_by_year.png", dpi=200)
    plt.close(fig)

    # q_peak vs precip if precip exists
    if "event_precip_mm" in events.columns:
        temp = events.dropna(subset=["event_precip_mm", "q_peak"])
        if len(temp) > 0:
            fig = plt.figure(figsize=(6, 5))
            plt.scatter(temp["event_precip_mm"], temp["q_peak"], s=10, alpha=0.4)
            plt.xlabel("Event precipitation (mm)")
            plt.ylabel("Peak discharge (cfs)")
            plt.title("Peak discharge vs event precipitation")
            plt.tight_layout()
            fig.savefig(output_dir / "q_peak_vs_event_precip.png", dpi=200)
            plt.close(fig)


def plot_one_gage_events(processed_by_gage: Dict[str, pd.DataFrame], events: pd.DataFrame, output_path: Path) -> None:
    if plt is None or events.empty:
        return
    gage_id = str(events["gage_id"].iloc[0])
    if gage_id not in processed_by_gage:
        return
    g = processed_by_gage[gage_id]
    ev = events[events["gage_id"].astype(str) == gage_id].copy()
    # Plot one year around first event to avoid huge plot.
    first_start = pd.to_datetime(ev["event_start"].iloc[0])
    plot_start = first_start - pd.Timedelta(days=30)
    plot_end = first_start + pd.Timedelta(days=120)
    g2 = g[(g.index >= plot_start) & (g.index <= plot_end)].copy()
    ev2 = ev[(pd.to_datetime(ev["event_end"]) >= plot_start) & (pd.to_datetime(ev["event_start"]) <= plot_end)].copy()
    fig = plt.figure(figsize=(14, 5))
    plt.plot(g2.index, g2["discharge_cfs"], label="Observed discharge")
    plt.plot(g2.index, g2["baseline"], label="Rolling baseline")
    for _, row in ev2.iterrows():
        plt.axvspan(pd.to_datetime(row["event_start"]), pd.to_datetime(row["event_end"]), alpha=0.2)
        plt.scatter(pd.to_datetime(row["peak_time"]), row["q_peak"], s=25)
    plt.title(f"Example extracted hydrograph events: gage {gage_id}")
    plt.xlabel("Time")
    plt.ylabel("Discharge (cfs)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# =============================================================================
# 7. MAIN CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end pipeline for Mark's wetland-discharge event model.")
    parser.add_argument("--start-year", type=int, default=1981)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--huc", type=str, default="04040003", help="Default Milwaukee Basin HUC-8.")
    parser.add_argument("--project-dir", type=str, default=None, help="Default is parent of this script's folder.")
    parser.add_argument("--precip-dir", type=str, default=None, help="Folder containing annual 800m precipitation csv.gz files.")
    parser.add_argument("--discharge-dir", type=str, default=None, help="Folder for downloaded/available USGS yearly discharge files.")
    parser.add_argument("--covariates", type=str, default=None, help="CSV with gage_id, wetland_effectiveness, runoff_propensity.")
    parser.add_argument("--skip-download", action="store_true", help="Use existing discharge yearly files; do not download.")
    parser.add_argument("--download-only", action="store_true", help="Only download raw discharge; do not extract events or run models.")
    parser.add_argument("--events-only", action="store_true", help="Extract events but do not run Mark regressions.")
    parser.add_argument("--overwrite-download", action="store_true", help="Overwrite existing downloaded discharge files.")
    parser.add_argument("--sample-only", action="store_true", help="Use only first two discharge years loaded; useful for testing.")
    parser.add_argument("--sleep", type=float, default=0.6, help="Sleep between USGS requests.")
    parser.add_argument("--baseline-window", type=str, default="7D")
    parser.add_argument("--start-ratio", type=float, default=1.20)
    parser.add_argument("--end-ratio", type=float, default=1.05)
    parser.add_argument("--min-peak-ratio", type=float, default=1.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_dir = Path(args.project_dir).expanduser().resolve() if args.project_dir else script_dir.parent
    new_model_dir = script_dir

    outputs_dir = ensure_dir(new_model_dir / "outputs")
    events_dir = ensure_dir(outputs_dir / "events")
    models_dir = ensure_dir(outputs_dir / "models")
    plots_dir = ensure_dir(outputs_dir / "plots")
    templates_dir = ensure_dir(outputs_dir / "templates")

    discharge_base_dir = Path(args.discharge_dir).expanduser().resolve() if args.discharge_dir else new_model_dir / "usgs_discharge_15min_raw"
    precip_dir = Path(args.precip_dir).expanduser().resolve() if args.precip_dir else (
        new_model_dir / "800m_data" if (new_model_dir / "800m_data").exists()
        else project_dir / "Data" / "Precipitation_Data" / "800m_data"
    )

    print_header("Paths")
    print("script_dir:", script_dir)
    print("project_dir:", project_dir)
    print("discharge_base_dir:", discharge_base_dir)
    print("precip_dir:", precip_dir)
    print("outputs_dir:", outputs_dir)

    cfg = EventConfig(
        baseline_window=args.baseline_window,
        start_ratio=args.start_ratio,
        end_ratio=args.end_ratio,
        min_peak_ratio=args.min_peak_ratio,
    )

    if not args.skip_download:
        yearly_dir = download_discharge_by_year(
            huc=args.huc,
            start_year=args.start_year,
            end_year=args.end_year,
            output_dir=discharge_base_dir,
            overwrite=args.overwrite_download,
            sleep_seconds=args.sleep,
        )
    else:
        yearly_dir = discharge_base_dir / "yearly"
        print(f"Skipping download. Using yearly discharge directory: {yearly_dir}")

    if args.download_only:
        print("download-only mode finished.")
        return

    discharge = load_discharge_yearly_files(
        yearly_dir=yearly_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        sample_only=args.sample_only,
    )
    precip = load_800m_precip(
        precip_dir=precip_dir,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    events, processed_by_gage = extract_rredi_like_events(discharge, precip, cfg)
    if events.empty:
        print("No event table to save/model.")
        return

    events_path = events_dir / f"rredi_like_hydrograph_events_{args.start_year}_{args.end_year}.csv.gz"
    events.to_csv(events_path, index=False, compression="gzip")
    print(f"Saved event table: {events_path}")

    make_basic_plots(events, plots_dir)
    plot_one_gage_events(processed_by_gage, events, plots_dir / "example_detected_hydrographs.png")

    if args.events_only:
        print("events-only mode finished. Mark regression not run.")
        return

    covariates_path = Path(args.covariates).expanduser().resolve() if args.covariates else None
    template_path = templates_dir / "gage_covariates_template.csv"
    panel = merge_covariates(events, covariates_path, template_path)
    if panel is None:
        print("Stopping before Mark regressions. Fill the covariates template and rerun with --covariates.")
        return

    run_mark_models(panel, models_dir)
    print_header("Pipeline complete")
    print("Event table:", events_path)
    print("Model outputs:", models_dir)
    print("Plots:", plots_dir)


if __name__ == "__main__":
    main()
