#!/usr/bin/env python3
"""
Mangalavanam Adaptive Sentinel
Phase 2a: Raster → Dynamic Occupancy Matrix (CORRECTED)

Converts guild occupancy rasters into pixel-level time series
suitable for dynamic occupancy modelling.

Author: Senior Geospatial Data Scientist
Date: January 2026
"""

import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

RASTER_DIR = Path("data/processed/phase1/guild_occupancy_rasters")
OUTPUT_DIR = Path("data/processed/phase2")
OUTPUT_FILE = OUTPUT_DIR / "guild_pixel_timeseries.csv"

NODATA = 255

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# UTILITIES
# =============================================================================

def parse_raster_name(filename):
    """
    Extract guild, year, zone from raster filename.
    Accepts Core/Buffer in any case.
    Example: Forest_2019_Buffer.tif
    """
    match = re.match(r"(\w+)_(\d{4})_(core|buffer)", filename, re.IGNORECASE)
    if not match:
        raise ValueError(f"Unexpected raster name: {filename}")

    guild = match.group(1)
    year = int(match.group(2))
    zone = match.group(3).lower()  # normalize to lowercase

    return guild, year, zone


def pixel_id(row, col):
    """Stable pixel identifier."""
    return f"P_{row}_{col}"

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 70)
    print("PHASE 2a — BUILDING OCCUPANCY MATRIX")
    print("=" * 70)

    records = []

    rasters = sorted(RASTER_DIR.glob("*.tif"))
    
    if not rasters:
        print(f"\n❌ ERROR: No rasters found in {RASTER_DIR}")
        print("Please ensure Phase 1 has been completed.")
        return 1
    
    print(f"\nFound {len(rasters)} occupancy rasters")

    for raster_path in rasters:
        try:
            guild, year, zone = parse_raster_name(raster_path.name)
        except ValueError as e:
            print(f"⚠️  Skipping {raster_path.name}: {e}")
            continue

        print(f"  Processing: {raster_path.name}")

        with rasterio.open(raster_path) as src:
            data = src.read(1)

        # Find pixels with effort (not NoData)
        rows, cols = np.where(data != NODATA)

        for r, c in zip(rows, cols):
            value = data[r, c]

            records.append({
                "pixel_id": pixel_id(r, c),
                "row": int(r),
                "col": int(c),
                "guild": guild,
                "zone": zone,
                "year": int(year),
                "detected": int(value == 1),
                "effort": 1
            })

    if not records:
        print("\n❌ ERROR: No valid records extracted from rasters")
        return 1

    df = pd.DataFrame(records)
    
    print(f"\n✓ Extracted {len(df):,} observation records")

    # -----------------------------------------------------------------
    # ENSURE ZERO-EFFORT YEARS ARE EXPLICIT
    # -----------------------------------------------------------------
    print("\nExpanding to include zero-effort years...")

    # Get all unique combinations
    unique_pixels = df[["pixel_id", "row", "col", "guild", "zone"]].drop_duplicates()
    all_years = sorted(df["year"].unique())
    
    print(f"  Unique pixels: {len(unique_pixels):,}")
    print(f"  Years: {all_years}")

    # Create full index (all pixel × year combinations)
    full_index = unique_pixels.merge(
        pd.DataFrame({"year": all_years}),
        how="cross"
    )
    
    print(f"  Expected records: {len(full_index):,}")

    # Merge with observed data
    df = full_index.merge(
        df[["pixel_id", "guild", "zone", "year", "detected", "effort"]],
        on=["pixel_id", "guild", "zone", "year"],
        how="left"
    )

    # Fill missing values
    df["effort"] = df["effort"].fillna(0).astype(int)
    df["detected"] = df["detected"].where(df["effort"] == 1)  # NaN where no effort

    print(f"  Final records: {len(df):,}")

    # -----------------------------------------------------------------
    # SAVE
    # -----------------------------------------------------------------
    df.sort_values(["guild", "zone", "pixel_id", "year"], inplace=True)
    df.to_csv(OUTPUT_FILE, index=False)

    # -----------------------------------------------------------------
    # SUMMARY STATISTICS
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 2a COMPLETE")
    print("=" * 70)
    print(f"Output: {OUTPUT_FILE}")
    print(f"Total rows: {len(df):,}")
    print(f"\nBreakdown by guild:")
    for guild, count in df.groupby("guild").size().items():
        print(f"  {guild}: {count:,} records")
    print(f"\nBreakdown by zone:")
    for zone, count in df.groupby("zone").size().items():
        print(f"  {zone}: {count:,} records")
    print(f"\nYear range: {df['year'].min()} - {df['year'].max()}")
    print(f"Pixels with effort: {df[df['effort'] == 1]['pixel_id'].nunique():,}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())