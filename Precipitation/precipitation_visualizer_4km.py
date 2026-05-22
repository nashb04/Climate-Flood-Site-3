# Author: Nash Boykin
# Last Edited: 2026-05-21
# Description: This script visualizes daily PRISM 4km precipitation data, clips to Milwaukee basin

"""
Visualize daily PRISM 4km precipitation for the Milwaukee basin from a local CSV.

This script:
1. Loads a yearly precipitation table exported from the Milwaukee basin workflow
2. Reconstructs the basin grid from lon/lat coordinates
3. Generates one PNG frame per day
4. Combines frames into an animated GIF

Input columns expected:
    date, lon, lat, precip_mm
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================
CSV_PATH = Path(
    r"C:\Users\cnboy\Box\Data+□ Climate Resilience\Data\Precipitation_Data\4km_data\milwaukee_precip_1981.csv\milwaukee_precip_1981.csv"
)

OUTPUT_DIR = Path("gif_from_1981_precip_visible")
GIF_NAME = "milwaukee_precip_1981_visible.gif"

GIF_DURATION_MS = 120
DPI = 140
VMAX = 8.0
CMAP = "turbo"


# ============================================================
# DATA LOADING
# ============================================================
def load_precipitation_data(csv_path: Path) -> pd.DataFrame:
    """Load daily precipitation cell data from CSV."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ============================================================
# GRID CONSTRUCTION
# ============================================================
def build_spatial_index(df: pd.DataFrame):
    """
    Build row/column lookup tables from unique lat/lon values.

    Returns
    -------
    unique_lons : np.ndarray
        Sorted longitude coordinates
    unique_lats : np.ndarray
        Sorted latitude coordinates in descending order for plotting
    lon_to_col : dict
        Maps longitude to grid column index
    lat_to_row : dict
        Maps latitude to grid row index
    """
    unique_lons = np.sort(df["lon"].unique())
    unique_lats = np.sort(df["lat"].unique())[::-1]

    lon_to_col = {lon: idx for idx, lon in enumerate(unique_lons)}
    lat_to_row = {lat: idx for idx, lat in enumerate(unique_lats)}

    print(f"Grid size: {len(unique_lats)} rows x {len(unique_lons)} columns")
    return unique_lons, unique_lats, lon_to_col, lat_to_row


def build_daily_grid(
    day_df: pd.DataFrame,
    unique_lons: np.ndarray,
    unique_lats: np.ndarray,
    lon_to_col: dict,
    lat_to_row: dict,
    vmax: float,
) -> np.ndarray:
    """
    Convert one day of point/cell data into a 2D grid for plotting.
    """
    grid = np.full((len(unique_lats), len(unique_lons)), np.nan, dtype=float)

    for _, row in day_df.iterrows():
        row_idx = lat_to_row[row["lat"]]
        col_idx = lon_to_col[row["lon"]]
        precip_value = max(row["precip_mm"], 0.0)
        grid[row_idx, col_idx] = min(precip_value, vmax)

    return grid


# ============================================================
# PLOTTING
# ============================================================
def save_daily_frame(
    grid: np.ndarray,
    date_label: str,
    mean_precip_mm: float,
    max_precip_mm: float,
    output_path: Path,
    vmax: float,
    cmap: str,
    dpi: int,
) -> None:
    """Save one daily precipitation frame as a PNG."""
    fig, ax = plt.subplots(figsize=(8, 6.5))

    image = ax.imshow(
        grid,
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_title(
        f"Milwaukee Basin Daily Precipitation\n"
        f"{date_label}   Mean={mean_precip_mm:.2f} mm   Max={max_precip_mm:.2f} mm",
        fontsize=13,
    )
    ax.set_xlabel("Grid Column")
    ax.set_ylabel("Grid Row")

    colorbar = plt.colorbar(image, ax=ax, shrink=0.88)
    colorbar.set_label(f"Precipitation (mm, capped at {vmax})")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


# ============================================================
# GIF CREATION
# ============================================================
def build_gif(frame_paths: list[Path], output_path: Path, duration_ms: int) -> None:
    """Combine PNG frames into an animated GIF."""
    print(f"Building GIF: {output_path}")

    images = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]

    if not images:
        print("No frames found. GIF was not created.")
        return

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


# ============================================================
# MAIN WORKFLOW
# ============================================================
def main() -> None:
    """Run the full visualization workflow."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_precipitation_data(CSV_PATH)

    unique_lons, unique_lats, lon_to_col, lat_to_row = build_spatial_index(df)

    daily_stats = (
        df.groupby("date")
        .agg(
            basin_mean_mm=("precip_mm", "mean"),
            basin_max_mm=("precip_mm", "max"),
        )
        .reset_index()
    )
    daily_stats["date_str"] = daily_stats["date"].dt.strftime("%Y-%m-%d")

    unique_dates = sorted(daily_stats["date_str"].unique())

    print(f"Total frames: {len(unique_dates)}")
    print(f"Display scale: 0 to {VMAX} mm")

    frame_paths = []

    for frame_number, date_str in enumerate(unique_dates, start=1):
        day_df = df[df["date"].dt.strftime("%Y-%m-%d") == date_str]
        stats_row = daily_stats.loc[daily_stats["date_str"] == date_str].iloc[0]

        grid = build_daily_grid(
            day_df=day_df,
            unique_lons=unique_lons,
            unique_lats=unique_lats,
            lon_to_col=lon_to_col,
            lat_to_row=lat_to_row,
            vmax=VMAX,
        )

        frame_path = OUTPUT_DIR / f"frame_{frame_number:03d}_{date_str}.png"

        save_daily_frame(
            grid=grid,
            date_label=date_str,
            mean_precip_mm=stats_row["basin_mean_mm"],
            max_precip_mm=stats_row["basin_max_mm"],
            output_path=frame_path,
            vmax=VMAX,
            cmap=CMAP,
            dpi=DPI,
        )

        frame_paths.append(frame_path)

        if frame_number % 25 == 0 or frame_number == len(unique_dates):
            print(f"Saved {frame_number}/{len(unique_dates)} frames")

    gif_path = OUTPUT_DIR / GIF_NAME
    build_gif(frame_paths, gif_path, GIF_DURATION_MS)

    print("Done.")


if __name__ == "__main__":
    main()