# Author: Nash Boykin
# Last Edited: 2026-05-21
# Description: Downloads daily PRISM 4km precipitation data for the Milwaukee basin.

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


# ── Configuration ─────────────────────────────────────────────────────────────

SHAPEFILE  = r"C:\Users\cnboy\Downloads\download\layers\globalwatershed.shp"
PRISM_URL  = "https://services.nacse.org/prism/data/get/us/4km/ppt"
SLEEP      = 1.0   # seconds between requests
RETRIES    = 3
WORKERS    = 8     # parallel years at once


# ── Basin ──────────────────────────────────────────────────────────────────────

def load_basin():
    """Load Milwaukee basin shapefile and return clipping geometries."""
    gdf = gpd.read_file(SHAPEFILE)
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return [mapping(geom) for geom in gdf.geometry]


def build_grid_reference(geometries):
    """
    Download one sample day, clip to basin, save cell coordinates.
    Skips if milwaukee_grid_cells.csv already exists.
    """
    path = "milwaukee_grid_cells.csv"
    if os.path.exists(path):
        ref = pd.read_csv(path)
        print(f"Grid reference: {len(ref)} cells")
        return ref

    print("Building grid reference...")
    sample_day = datetime(2010, 6, 15)
    zip_bytes  = fetch_day(sample_day)
    tif_bytes  = unzip_tif(zip_bytes)
    df         = clip_and_extract(tif_bytes, geometries, sample_day)

    ref = df[["cell_id", "lon", "lat"]]
    ref.to_csv(path, index=False)
    print(f"  {len(ref)} cells saved to {path}")
    return ref


# ── PRISM Download ─────────────────────────────────────────────────────────────

def fetch_day(date):
    """
    Download PRISM zip for one date. Returns raw bytes or raises on failure.
    Retries up to RETRIES times with exponential backoff.
    """
    url = f"{PRISM_URL}/{date.strftime('%Y%m%d')}"

    for attempt in range(RETRIES):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                raise FileNotFoundError(f"PRISM has no data for {date.date()}")
            time.sleep(2 ** attempt)
        except requests.exceptions.RequestException as e:
            if attempt == RETRIES - 1:
                raise
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Failed to download {date.date()} after {RETRIES} attempts")


def unzip_tif(zip_bytes):
    """Extract the .tif raster from a PRISM zip file (in memory)."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        tif = next(f for f in zf.namelist()
                   if f.endswith(".tif") and not f.endswith(".aux.xml"))
        return zf.read(tif)


# ── Raster Processing ──────────────────────────────────────────────────────────

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

    data = clipped[0]
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


# ── Year Download ──────────────────────────────────────────────────────────────

def download_year(year):
    """
    Download all days for one year and save to milwaukee_precip_YYYY.csv.gz.
    Skips days already downloaded if the file exists (resume support).
    """
    output    = f"milwaukee_precip_{year}.csv.gz"
    all_dates = [datetime(year, 1, 1) + timedelta(days=i)
                 for i in range(365 + int(year % 4 == 0))]

    # Resume: find which dates are already saved
    completed = set()
    if os.path.exists(output):
        completed = set(pd.read_csv(output, usecols=["date"])["date"])
        if len(completed) == len(all_dates):
            print(f"  {year}: complete — skipping")
            return

    remaining  = [d for d in all_dates if d.strftime("%Y-%m-%d") not in completed]
    geometries = load_basin()

    print(f"  {year}: downloading {len(remaining)} days")

    rows_by_day = []
    failed      = []
    start       = time.time()

    for i, date in enumerate(remaining):
        try:
            zip_bytes = fetch_day(date)
            tif_bytes = unzip_tif(zip_bytes)
            day_df    = clip_and_extract(tif_bytes, geometries, date)
            if len(day_df):
                rows_by_day.append(day_df)
        except Exception as e:
            print(f"    FAILED {date.date()}: {e}")
            failed.append(date.strftime("%Y-%m-%d"))

        if (i + 1) % 30 == 0:
            elapsed = time.time() - start
            eta     = (len(remaining) - i - 1) / ((i + 1) / elapsed)
            print(f"    {date.date()}  {i+1}/{len(remaining)} "
                  f"({100*(i+1)/len(remaining):.0f}%)  "
                  f"ETA {time.strftime('%Hh %Mm', time.gmtime(eta))}")

        time.sleep(SLEEP)

    # Merge new data with any existing data and write
    if rows_by_day:
        new_data = pd.concat(rows_by_day, ignore_index=True)
        if completed and os.path.exists(output):
            new_data = pd.concat([pd.read_csv(output), new_data], ignore_index=True)
        new_data.to_csv(output, index=False, compression="gzip")

    print(f"  {year}: saved → {output}")

    if failed:
        fail_log = f"failed_{year}.csv"
        pd.Series(failed).to_csv(fail_log, index=False, header=False)
        print(f"  {year}: {len(failed)} failed dates → {fail_log}")


# ── Parallel Runner ────────────────────────────────────────────────────────────

def run_all(start_year=1981, end_year=2023):
    """
    Download all years in parallel using a pool of worker processes.
    Builds the grid reference first so workers don't race to create it.
    """
    geometries = load_basin()
    build_grid_reference(geometries)

    years = list(range(start_year, end_year + 1))
    print(f"\nDownloading {len(years)} years with {WORKERS} parallel workers\n")

    with Pool(processes=WORKERS) as pool:
        pool.map(download_year, years)

    print("\nAll years complete.")


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if "--grid" in args:
        build_grid_reference(load_basin())

    elif "--all" in args:
        run_all()

    elif len(args) == 2:
        try:
            start, end = int(args[0]), int(args[1])
            download_year(start) if start == end else run_all(start, end)
        except ValueError:
            print("ERROR: years must be integers, e.g. 1995 2005")
            sys.exit(1)

    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()