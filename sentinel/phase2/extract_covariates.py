#!/usr/bin/env python3
"""
Mangalavanam Adaptive Sentinel
Phase 2b: Environmental Covariate Extraction (CORRECTED)

Extracts environmental covariates for each pixel × year combination.

IMPROVEMENTS:
- Keeps all pixels (not just effort=1) for complete spatial coverage
- Better error handling for missing rasters
- Validates covariate extraction

Author: Senior Geospatial Data Scientist
Date: January 2026
"""

import pandas as pd
import rasterio
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

TIMESERIES_CSV = Path("data/processed/phase2/guild_pixel_timeseries.csv")
RASTER_DIR = Path("data/raw/rasters")
OUTPUT_DIR = Path("data/processed/phase2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "guild_pixel_timeseries_with_covariates.csv"

# Static rasters (no year/zone variation)
STATIC_RASTERS = {
    "dist_edge": RASTER_DIR / "DIST_EDGE.tif",
    "dist_drainage": RASTER_DIR / "DIST_DRAINAGE.tif",
    "dist_builtup": RASTER_DIR / "DIST_BUILTUP.tif",
}

# Yearly rasters (vary by year + zone)
YEARLY_VARIABLES = [
    "NDVI",
    "NDWI",
    "VIIRS"
]

# =============================================================================
# UTILITIES
# =============================================================================

def read_pixel_value(raster_path, row, col):
    """
    Read value at specific pixel location.
    Returns NaN if out of bounds or NoData.
    """
    try:
        with rasterio.open(raster_path) as ds:
            arr = ds.read(1)
            
            # Check bounds
            if row < 0 or row >= arr.shape[0] or col < 0 or col >= arr.shape[1]:
                return np.nan
            
            val = arr[int(row), int(col)]
            
            # Handle NoData
            if ds.nodata is not None and val == ds.nodata:
                return np.nan
            
            return float(val)
    except Exception as e:
        print(f"    ⚠️ Error reading {raster_path.name} at ({row}, {col}): {e}")
        return np.nan


def check_raster_exists(raster_path):
    """Check if raster exists and is readable."""
    if not raster_path.exists():
        return False
    
    try:
        with rasterio.open(raster_path) as ds:
            _ = ds.read(1, window=((0, 1), (0, 1)))  # Try reading one pixel
        return True
    except Exception:
        return False

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 70)
    print("PHASE 2b — EXTRACTING ENVIRONMENTAL COVARIATES")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Load time series
    # -----------------------------------------------------------------
    if not TIMESERIES_CSV.exists():
        print(f"\n❌ ERROR: {TIMESERIES_CSV} not found")
        print("Please run Phase 2a first.")
        return 1

    print(f"\nLoading: {TIMESERIES_CSV}")
    df = pd.read_csv(TIMESERIES_CSV)
    print(f"  ✓ Loaded {len(df):,} records")

    # -----------------------------------------------------------------
    # Check which rasters are available
    # -----------------------------------------------------------------
    print("\nChecking raster availability...")
    
    available_static = {}
    for name, path in STATIC_RASTERS.items():
        if check_raster_exists(path):
            available_static[name] = path
            print(f"  ✓ {name}: {path.name}")
        else:
            print(f"  ⚠️ {name}: NOT FOUND ({path.name})")
    
    if not available_static:
        print("\n❌ ERROR: No static rasters found!")
        return 1

    # -----------------------------------------------------------------
    # Extract static covariates
    # -----------------------------------------------------------------
    print("\nExtracting static covariates...")

    for name, raster_path in available_static.items():
        print(f"  → {name}")
        df[name] = df.apply(
            lambda r: read_pixel_value(raster_path, r["row"], r["col"]),
            axis=1
        )
        
        # Report extraction success
        valid_count = df[name].notna().sum()
        print(f"    Valid values: {valid_count:,} / {len(df):,} ({valid_count/len(df)*100:.1f}%)")

    # -----------------------------------------------------------------
    # Extract yearly covariates
    # -----------------------------------------------------------------
    print("\nExtracting yearly covariates...")

    for var in YEARLY_VARIABLES:
        print(f"  → {var}")
        values = []
        missing_count = 0

        for _, r in df.iterrows():
            # Construct filename: VAR_YEAR_ZONE.tif
            # Zone should be capitalized to match Phase 1 output
            zone_formatted = r['zone'].capitalize()
            raster_path = RASTER_DIR / f"{var}_{int(r['year'])}_{zone_formatted}.tif"

            if not raster_path.exists():
                # Try lowercase zone
                raster_path = RASTER_DIR / f"{var}_{int(r['year'])}_{r['zone']}.tif"
            
            if not raster_path.exists():
                values.append(np.nan)
                missing_count += 1
                continue

            values.append(read_pixel_value(raster_path, r["row"], r["col"]))

        df[var.lower()] = values
        
        valid_count = pd.Series(values).notna().sum()
        print(f"    Valid values: {valid_count:,} / {len(df):,} ({valid_count/len(df)*100:.1f}%)")
        if missing_count > 0:
            print(f"    ⚠️ Missing rasters: {missing_count:,} cases")

    # -----------------------------------------------------------------
    # Data quality checks
    # -----------------------------------------------------------------
    print("\n" + "-" * 70)
    print("DATA QUALITY SUMMARY")
    print("-" * 70)
    
    # Count missing values per covariate
    covariate_cols = list(available_static.keys()) + [v.lower() for v in YEARLY_VARIABLES]
    
    missing_summary = []
    for col in covariate_cols:
        if col in df.columns:
            missing_pct = df[col].isna().sum() / len(df) * 100
            missing_summary.append({
                'Covariate': col,
                'Missing %': missing_pct,
                'Valid Count': df[col].notna().sum()
            })
    
    missing_df = pd.DataFrame(missing_summary).sort_values('Missing %', ascending=False)
    print(missing_df.to_string(index=False))
    
    # Overall completeness
    print(f"\nOverall completeness:")
    complete_rows = df[covariate_cols].notna().all(axis=1).sum()
    print(f"  Rows with all covariates: {complete_rows:,} / {len(df):,} ({complete_rows/len(df)*100:.1f}%)")

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("PHASE 2b COMPLETE")
    print("=" * 70)
    print(f"Output: {OUTPUT_FILE}")
    print(f"Total records: {len(df):,}")
    print(f"Covariates extracted: {len(covariate_cols)}")
    print("\n✓ Data ready for occupancy modeling")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())