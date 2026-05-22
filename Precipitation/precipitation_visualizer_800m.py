# Author: Nash Boykin
# Last Edited: 2026-05-21
# Description: This script visualizes daily PRISM 800m precipitation data clipped to the Milwaukee basin.

"""
PRISM 800m Milwaukee Basin GIF for Selected 10-Day Wet Window
-------------------------------------------------------------
Downloads PRISM 800m daily precipitation for a selected list of dates,
clips each raster to the Milwaukee basin, saves PNG frames, and builds
an animated GIF.

Dependencies:
    pip install requests rasterio geopandas matplotlib numpy shapely pillow
"""

import io
import os
import time
import zipfile
from datetime import datetime

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.mask import mask as rio_mask
from rasterio.transform import array_bounds


# ============================================================
# CONFIGURATION
# ============================================================
SHAPEFILE_PATH = r"C:\Users\cnboy\Downloads\download\layers\globalwatershed.shp"
BASE_URL = "https://services.nacse.org/prism/data/get/us/800m/ppt"

DATES = [
    "1981-07-12",
    "1981-07-13",
    "1981-07-14",
    "1981-07-15",
    "1981-07-16",
    "1981-07-17",
    "1981-07-18",
    "1981-07-19",
    "1981-07-20",
    "1981-07-21",
]

OUTPUT_DIR = "prism_800m_selected_10day_output"
GIF_NAME = "milwaukee_800m_wettest_10day_window.gif"

SLEEP_SECONDS = 1.0
MAX_RETRIES = 3

GIF_DURATION_MS = 260
CMAP = "turbo"
VMAX = 12.0


# ============================================================
# DATA LOADING
# ============================================================
def load_basin():
    """Load the Milwaukee basin shapefile."""
    basin_gdf = gpd.read_file(SHAPEFILE_PATH)

    if basin_gdf.crs is None:
        raise ValueError("Shapefile has no CRS defined.")

    return basin_gdf


def download_day(date_obj):
    """
    Download one PRISM daily precipitation zip file.

    Returns
    -------
    bytes or None
        Zip file contents if successful, otherwise None.
    """
    date_code = date_obj.strftime("%Y%m%d")
    url = f"{BASE_URL}/{date_code}"

    print(f"Downloading {date_code} ...")

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=120)

            if response.status_code == 200:
                return response.content

            if response.status_code == 404:
                print(f"  {date_code}: 404 not found")
                return None

            print(f"  {date_code}: HTTP {response.status_code}, attempt {attempt + 1}")
            time.sleep(2 ** attempt)

        except requests.exceptions.RequestException as exc:
            print(f"  {date_code}: {exc}, attempt {attempt + 1}")
            time.sleep(2 ** attempt)

    print(f"  {date_code}: failed after {MAX_RETRIES} attempts")
    return None


def extract_tif_from_zip(zip_bytes):
    """
    Extract the GeoTIFF file from a PRISM zip archive held in memory.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zip_file:
        tif_files = [
            name for name in zip_file.namelist()
            if name.endswith(".tif") and not name.endswith(".aux.xml")
        ]

        if not tif_files:
            print("No tif found in zip archive.")
            return None

        return zip_file.read(tif_files[0])


# ============================================================
# RASTER PROCESSING
# ============================================================
def clip_raster_to_basin(tif_bytes, basin_gdf):
    """
    Clip an in-memory raster to the Milwaukee basin.

    Returns
    -------
    clipped : np.ndarray
        Clipped raster array
    transform : affine.Affine
        Transform for clipped raster
    meta : dict
        Updated raster metadata
    basin_projected : GeoDataFrame
        Basin reprojected to raster CRS
    nodata : float or int or None
        Raster nodata value
    """
    with MemoryFile(tif_bytes) as memory_file:
        with memory_file.open() as src:
            basin_projected = basin_gdf.to_crs(src.crs)
            geometries = [geom.__geo_interface__ for geom in basin_projected.geometry]

            clipped, transform = rio_mask(src, geometries, crop=True)

            meta = src.meta.copy()
            meta.update({
                "height": clipped.shape[1],
                "width": clipped.shape[2],
                "transform": transform,
            })

            return clipped, transform, meta, basin_projected, src.nodata


def summarize_precipitation(clipped, nodata):
    """
    Compute simple daily precipitation summary statistics.
    """
    data = clipped[0].astype(float)

    if nodata is not None:
        data[data == nodata] = np.nan

    data[data < 0] = 0
    valid = np.isfinite(data)

    if not valid.any():
        return 0.0, 0.0, 0

    values = data[valid]
    mean_mm = float(np.nanmean(values))
    max_mm = float(np.nanmax(values))
    wet_pixels = int(np.sum(values > 0.1))

    return mean_mm, max_mm, wet_pixels


# ============================================================
# PLOTTING
# ============================================================
def save_frame_png(clipped, transform, basin_gdf, nodata, output_path, title, subtitle):
    """
    Save one precipitation map frame as a PNG.
    """
    data = clipped[0].astype(float)

    if nodata is not None:
        data[data == nodata] = np.nan

    data[data < 0] = 0
    plot_data = np.clip(data, 0, VMAX)

    west, south, east, north = array_bounds(
        plot_data.shape[0],
        plot_data.shape[1],
        transform,
    )

    fig, ax = plt.subplots(figsize=(8, 6.5))

    image = ax.imshow(
        plot_data,
        extent=[west, east, south, north],
        origin="upper",
        cmap=CMAP,
        vmin=0,
        vmax=VMAX,
    )

    basin_gdf.boundary.plot(ax=ax, color="black", linewidth=1.0)

    ax.set_title(f"{title}\n{subtitle}", fontsize=12)
    ax.set_xlabel("Longitude / X")
    ax.set_ylabel("Latitude / Y")

    colorbar = plt.colorbar(image, ax=ax, shrink=0.88)
    colorbar.set_label(f"Precipitation (mm, capped at {VMAX})")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def build_gif(frame_paths, gif_path, duration_ms=130):
    """
    Combine PNG frames into an animated GIF.
    """
    images = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]

    if not images:
        print("No frames available for GIF.")
        return

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


# ============================================================
# MAIN WORKFLOW
# ============================================================
def main():
    """Run the full 10-day PRISM download and visualization workflow."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    basin_gdf = load_basin()
    frame_paths = []

    for day_number, date_text in enumerate(DATES, start=1):
        date_obj = datetime.strptime(date_text, "%Y-%m-%d")
        date_code = date_obj.strftime("%Y%m%d")

        zip_bytes = download_day(date_obj)
        if zip_bytes is None:
            continue

        tif_bytes = extract_tif_from_zip(zip_bytes)
        if tif_bytes is None:
            continue

        try:
            clipped, transform, meta, basin_projected, nodata = clip_raster_to_basin(
                tif_bytes,
                basin_gdf,
            )

            mean_mm, max_mm, wet_pixels = summarize_precipitation(clipped, nodata)

            output_path = os.path.join(OUTPUT_DIR, f"frame_{day_number:02d}_{date_code}.png")

            title = f"Milwaukee Basin PRISM 800m Daily Precipitation\n{date_text}"
            subtitle = (
                f"Day {day_number}/{len(DATES)}   |   "
                f"Mean: {mean_mm:.2f} mm   |   "
                f"Max: {max_mm:.2f} mm   |   "
                f"Wet pixels: {wet_pixels}"
            )

            save_frame_png(
                clipped=clipped,
                transform=transform,
                basin_gdf=basin_projected,
                nodata=nodata,
                output_path=output_path,
                title=title,
                subtitle=subtitle,
            )

            frame_paths.append(output_path)
            print(f"Saved frame: {output_path}")

        except Exception as exc:
            print(f"Error processing {date_text}: {exc}")

        time.sleep(SLEEP_SECONDS)

    gif_path = os.path.join(OUTPUT_DIR, GIF_NAME)
    build_gif(frame_paths, gif_path, duration_ms=GIF_DURATION_MS)

    print(f"\nSaved GIF: {gif_path}")


if __name__ == "__main__":
    main()