# Author: Nash Boykin
# Last Edited: 2026-05-22
# Description: Downloads daily PRISM 800m precipitation data for the Milwaukee basin.

import io
import os
import sys
import time
import zipfile
from datetime import datetime, timedelta
from multiprocessing import Pool

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.io import MemoryFile
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping


# ── Configuration ──────────────────────────────────────────────────────────────

SHAPEFILE  = r"C:\Users\cnboy\Downloads\download\layers\globalwatershed.shp"
PRISM_URL  = "https://services.nacse.org/prism/data/get/us/800m/ppt"
OUTPUT_DIR = r"C:\Users\cnboy\Box\Data+□ Climate Resilience\Data\Precipitation_Data\800m_data"          # all output files go here, separate from 4km
SLEEP      = 1.0             # seconds between requests
RETRIES    = 3
WORKERS    = 8               # parallel years at once
SKIP_YEARS = {}          # already downloaded 


# ── Setup ───────────────────────────────────────────────────────────────────────

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def output_path(year):
    return os.path.join(OUTPUT_DIR, f"milwaukee_precip_800m_{year}.csv.gz")

def failed_path(year):
    return os.path.join(OUTPUT_DIR, f"failed_{year}.csv")

def grid_path():
    return os.path.join(OUTPUT_DIR, "milwaukee_grid_cells_800m.csv")


# ── Basin ───────────────────────────────────────────────────────────────────────

def load_basin():
    """Load Milwaukee basin shapefile and return clipping geometries."""
    gdf = gpd.read_file(SHAPEFILE)
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return [mapping(geom) for geom in gdf.geometry]


def build_grid_reference(geometries):
    """
    Download one sample day, clip to basin, save cell coordinates to OUTPUT_DIR.
    Skips if grid reference already exists.
    """
    path = grid_path()
    if os.path.exists(path):
        ref = pd.read_csv(path)
        print(f"Grid reference: {len(ref)} cells (800m)")
        return ref

    print("Building 800m grid reference...")
    zip_bytes = fetch_day(datetime(2010, 6, 15))
    tif_bytes = unzip_tif(zip_bytes)
    df        = clip_and_extract(tif_bytes, geometries, datetime(2010, 6, 15))

    ref = df[["cell_id", "lon", "lat"]]
    ref.to_csv(path, index=False)
    print(f"  {len(ref)} cells saved to {path}")
    return ref


# ── PRISM Download ──────────────────────────────────────────────────────────────

def fetch_day(date):
    """
    Download PRISM 800m zip for one date.
    Returns raw bytes on success, raises on all failures.
    Retries up to RETRIES times with exponential backoff.
    """
    url = f"{PRISM_URL}/{date.strftime('%Y%m%d')}"

    for attempt in range(RETRIES):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                raise FileNotFoundError(f"No PRISM data for {date.date()}")
            time.sleep(2 ** attempt)
        except requests.exceptions.RequestException:
            if attempt == RETRIES - 1:
                raise
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Download failed after {RETRIES} attempts: {date.date()}")


def unzip_tif(zip_bytes):
    """Extract the .tif raster from a PRISM zip file, entirely in memory."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        tif = next(f for f in zf.namelist()
                   if f.endswith(".tif") and not f.endswith(".aux.xml"))
        return zf.read(tif)


# ── Raster Processing ───────────────────────────────────────────────────────────

def clip_and_extract(tif_bytes, geometries, date):
    """
    Open TIF in memory, clip to Milwaukee basin, return one row per grid cell.
    The full US raster is never written to disk.

    Returns DataFrame with columns: date, cell_id, lon, lat, precip_mm
    """
    with MemoryFile(tif_bytes) as memfile:
        with memfile.open() as src:
            clipped, transform = rio_mask(src, geometries, crop=True)
            nodata = src.nodata

    data  = clipped[0]
    valid = (data != nodata) if nodata is not None else np.ones(data.shape, bool)
    rows, cols = np.where(valid)

    if len(rows) == 0:
        return pd.DataFrame()

    lons, lats = rasterio.transform.xy(transform, rows, cols)
    values     = np.clip(data[rows, cols].astype(float), 0, None)

    return pd.DataFrame({
        "date":      date.strftime("%Y-%m-%d"),
        "cell_id":   [f"r{r}_c{c}" for r, c in zip(rows, cols)],
        "lon":       lons,
        "lat":       lats,
        "precip_mm": values,
    })


# ── Year Download ───────────────────────────────────────────────────────────────

def download_year(year):
    """
    Download all days for one year, save to 800m/milwaukee_precip_800m_YYYY.csv.gz.
    - Skips years in SKIP_YEARS
    - Skips days already in the output file (resume support)
    - Logs each failed date immediately and continues
    - Prints progress every 30 days with ETA
    """
    if year in SKIP_YEARS:
        print(f"  {year}: skipped (in SKIP_YEARS)")
        return

    out   = output_path(year)
    fails = failed_path(year)

    all_dates = [datetime(year, 1, 1) + timedelta(days=i)
                 for i in range(365 + int(year % 4 == 0))]

    # Resume: find which dates are already saved
    completed = set()
    if os.path.exists(out):
        completed = set(pd.read_csv(out, usecols=["date"])["date"])
        if len(completed) == len(all_dates):
            print(f"  {year}: complete — skipping")
            return
        print(f"  {year}: resuming ({len(completed)}/{len(all_dates)} done)")

    # Also skip dates already logged as failed so we don't retry endlessly
    previously_failed = set()
    if os.path.exists(fails):
        previously_failed = set(pd.read_csv(fails, header=None)[0])

    remaining  = [d for d in all_dates
                  if d.strftime("%Y-%m-%d") not in completed
                  and d.strftime("%Y-%m-%d") not in previously_failed]

    if not remaining:
        print(f"  {year}: nothing left to download")
        return

    geometries = load_basin()
    print(f"  {year}: {len(remaining)} days to download")

    rows_by_day = []
    failed      = []
    start       = time.time()

    for i, date in enumerate(remaining):
        date_str = date.strftime("%Y-%m-%d")
        try:
            zip_bytes = fetch_day(date)
            tif_bytes = unzip_tif(zip_bytes)
            day_df    = clip_and_extract(tif_bytes, geometries, date)
            if len(day_df):
                rows_by_day.append(day_df)

        except Exception as e:
            # Log failure immediately and keep going
            print(f"    FAILED {date_str}: {e}")
            failed.append(date_str)
            with open(fails, "a") as f:
                f.write(date_str + "\n")

        if (i + 1) % 30 == 0:
            elapsed = time.time() - start
            eta     = (len(remaining) - i - 1) / ((i + 1) / elapsed)
            print(f"    {date_str}  {i+1}/{len(remaining)} "
                  f"({100*(i+1)/len(remaining):.0f}%)  "
                  f"ETA {time.strftime('%Hh %Mm', time.gmtime(eta))}")

        time.sleep(SLEEP)

    # Merge new rows with any existing data and write compressed
    if rows_by_day:
        new_data = pd.concat(rows_by_day, ignore_index=True)
        if completed and os.path.exists(out):
            new_data = pd.concat([pd.read_csv(out), new_data], ignore_index=True)
        new_data.to_csv(out, index=False, compression="gzip")

    status = f"done → {out}"
    if failed:
        status += f"  ({len(failed)} failed → {fails})"
    print(f"  {year}: {status}")


# ── Parallel Runner ─────────────────────────────────────────────────────────────

def run_all(start_year=1982, end_year=2023):
    """
    Download all years in parallel, skipping years in SKIP_YEARS.
    Builds the grid reference first so workers don't race to create it.
    """
    ensure_output_dir()
    geometries = load_basin()
    build_grid_reference(geometries)

    years = [y for y in range(start_year, end_year + 1) if y not in SKIP_YEARS]
    print(f"\nDownloading {len(years)} years with {WORKERS} parallel workers")
    print(f"Skipping: {sorted(SKIP_YEARS)}\n")

    with Pool(processes=WORKERS) as pool:
        pool.map(download_year, years)

    print("\nAll years complete.")
    print(f"Check {OUTPUT_DIR}/ for failed_YYYY.csv files if any dates failed.")


# ── Entry Point ─────────────────────────────────────────────────────────────────

def main():
    ensure_output_dir()
    args = sys.argv[1:]

    if "--grid" in args:
        build_grid_reference(load_basin())

    elif "--all" in args:
        run_all()

    elif len(args) == 2:
        try:
            start, end = int(args[0]), int(args[1])
            if start == end:
                download_year(start)
            else:
                run_all(start, end)
        except ValueError:
            print("ERROR: years must be integers, e.g. 1995 2005")
            sys.exit(1)

    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()