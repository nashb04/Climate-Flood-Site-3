# Author: Nash Boykin
# Last Edited: 2026-05-22
# Description: Summary statistics for PRISM 800m Milwaukee basin precipitation data.

import glob
import os
import pandas as pd
import numpy as np


# ── Configuration ──────────────────────────────────────────────────────────────

DATA_DIR   = r"C:\Users\cnboy\Box\Data+□ Climate Resilience\Data\Precipitation_Data\800m_data"
OUTPUT_CSV = os.path.join(DATA_DIR, "precipitation_summary.csv")
RAINY_DAY  = 1.0    # mm threshold to count as a rainy day
EXTREME_DAY = 25.0  # mm threshold to count as an extreme precipitation day


# ── Load and summarize ─────────────────────────────────────────────────────────

def summarize_year(path):
    """Load one year's gzipped CSV and return a dict of summary stats."""
    df   = pd.read_csv(path)
    year = int(df["date"].iloc[0][:4])

    # Daily totals — average across all cells for each day
    daily = df.groupby("date")["precip_mm"].mean()

    # Cell-level stats — average across all days for each cell
    cell_means = df.groupby("cell_id")["precip_mm"].mean()

    return {
        "year":             year,
        "days":             df["date"].nunique(),
        "cells":            df["cell_id"].nunique(),
        "mean_daily_mm":    round(daily.mean(), 3),
        "median_daily_mm":  round(daily.median(), 3),
        "max_daily_mm":     round(daily.max(), 3),
        "total_annual_mm":  round(daily.sum(), 1),
        "rainy_days":       int((daily > RAINY_DAY).sum()),
        "extreme_days":     int((daily > EXTREME_DAY).sum()),
        "cell_mean_min":    round(cell_means.min(), 3),
        "cell_mean_max":    round(cell_means.max(), 3),
        "cell_mean_std":    round(cell_means.std(), 3),
    }


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "milwaukee_precip_800m_*.csv.gz")))

    if not files:
        print(f"No files found in {DATA_DIR}")
        return

    print(f"Found {len(files)} years — loading...")

    rows = []
    for path in files:
        year = os.path.basename(path).split("_")[-1].replace(".csv.gz", "")
        print(f"  {year}...", end=" ", flush=True)
        try:
            row = summarize_year(path)
            rows.append(row)
            print("done")
        except Exception as e:
            print(f"ERROR: {e}")

    summary = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    summary.to_csv(OUTPUT_CSV, index=False)

    # Print clean table
    print("\n" + "=" * 95)
    print(f"{'Year':<6} {'Days':<6} {'Cells':<7} {'Mean':>8} {'Median':>8} "
          f"{'Max':>8} {'Annual':>9} {'Rainy':>7} {'Extreme':>8} {'CellStd':>8}")
    print(f"{'':6} {'':6} {'':7} {'mm/day':>8} {'mm/day':>8} "
          f"{'mm/day':>8} {'mm':>9} {'days':>7} {f'>{EXTREME_DAY}mm':>8} {'mm':>8}")
    print("-" * 95)

    for _, r in summary.iterrows():
        print(f"{int(r.year):<6} {int(r.days):<6} {int(r.cells):<7} "
              f"{r.mean_daily_mm:>8.2f} {r.median_daily_mm:>8.2f} "
              f"{r.max_daily_mm:>8.1f} {r.total_annual_mm:>9.0f} "
              f"{int(r.rainy_days):>7} {int(r.extreme_days):>8} "
              f"{r.cell_mean_std:>8.3f}")

    print("=" * 95)
    print(f"\nAll years summary:")
    print(f"  Mean annual precipitation:  {summary.total_annual_mm.mean():.0f} mm")
    print(f"  Wettest year:               {int(summary.loc[summary.total_annual_mm.idxmax(), 'year'])} "
          f"({summary.total_annual_mm.max():.0f} mm)")
    print(f"  Driest year:                {int(summary.loc[summary.total_annual_mm.idxmin(), 'year'])} "
          f"({summary.total_annual_mm.min():.0f} mm)")
    print(f"  Most extreme days:          {int(summary.loc[summary.extreme_days.idxmax(), 'year'])} "
          f"({summary.extreme_days.max()} days >{EXTREME_DAY}mm)")
    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()