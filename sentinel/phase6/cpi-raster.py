#!/usr/bin/env python3
"""
Create CPI Raster from site-level CSV
"""

import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path

# Configuration
BASE_DIR = Path("data")
PHASE6_DIR = BASE_DIR / "processed/phase6"
CPI_CSV = PHASE6_DIR / "site_level_CPI.csv"
OUTPUT_RASTER = PHASE6_DIR / "CPI_conservation_priority.tif"
OUTPUT_CLASSES = PHASE6_DIR / "priority_classes.tif"

BOUNDS = {
    'left': 76.2150,
    'bottom': 9.9800,
    'right': 76.2184,
    'top': 9.9866
}

GRID_ROWS = 7
GRID_COLS = 7

def create_cpi_rasters():
    """Create CPI rasters from site-level CSV"""
    
    print("🎯 Creating CPI rasters...")
    
    if not CPI_CSV.exists():
        print(f"❌ CPI CSV not found: {CPI_CSV}")
        print("   Run Phase 6 analysis first:")
        print("   python sentinel/phase6/conservation_priority.py")
        return
    
    # Load CPI data
    cpi_df = pd.read_csv(CPI_CSV)
    print(f"✓ Loaded {len(cpi_df)} sites")
    
    # Get CPI values
    cpi_values = cpi_df['CPI'].values
    
    # Reshape to grid
    n_pixels = GRID_ROWS * GRID_COLS
    if len(cpi_values) < n_pixels:
        cpi_padded = np.full(n_pixels, np.nan)
        cpi_padded[:len(cpi_values)] = cpi_values
        cpi_grid = cpi_padded.reshape((GRID_ROWS, GRID_COLS))
    else:
        cpi_grid = cpi_values[:n_pixels].reshape((GRID_ROWS, GRID_COLS))
    
    # Create transform
    transform = from_bounds(
        BOUNDS['left'], BOUNDS['bottom'],
        BOUNDS['right'], BOUNDS['top'],
        GRID_COLS, GRID_ROWS
    )
    
    # Save continuous CPI raster
    with rasterio.open(
        OUTPUT_RASTER,
        'w',
        driver='GTiff',
        height=GRID_ROWS,
        width=GRID_COLS,
        count=1,
        dtype=cpi_grid.dtype,
        crs='EPSG:4326',
        transform=transform,
        nodata=np.nan
    ) as dst:
        dst.write(cpi_grid, 1)
    
    print(f"✅ Saved continuous CPI: {OUTPUT_RASTER}")
    
    # Create priority classes (1-5)
    # Use quantiles from actual data
    quantiles = np.nanquantile(cpi_values, [0.2, 0.4, 0.6, 0.8])
    
    priority_classes = np.full_like(cpi_grid, np.nan)
    mask = ~np.isnan(cpi_grid)
    
    priority_classes[mask & (cpi_grid <= quantiles[0])] = 1  # Minimal
    priority_classes[mask & (cpi_grid > quantiles[0]) & (cpi_grid <= quantiles[1])] = 2  # Low
    priority_classes[mask & (cpi_grid > quantiles[1]) & (cpi_grid <= quantiles[2])] = 3  # Medium
    priority_classes[mask & (cpi_grid > quantiles[2]) & (cpi_grid <= quantiles[3])] = 4  # High
    priority_classes[mask & (cpi_grid > quantiles[3])] = 5  # Critical
    
    # Save priority classes
    with rasterio.open(
        OUTPUT_CLASSES,
        'w',
        driver='GTiff',
        height=GRID_ROWS,
        width=GRID_COLS,
        count=1,
        dtype='uint8',
        crs='EPSG:4326',
        transform=transform,
        nodata=0
    ) as dst:
        dst.write(priority_classes.astype('uint8'), 1)
    
    print(f"✅ Saved priority classes: {OUTPUT_CLASSES}")
    
    # Print distribution
    print("\n📊 Priority Distribution:")
    for level in range(1, 6):
        count = np.sum(priority_classes == level)
        pct = (count / np.sum(~np.isnan(priority_classes))) * 100
        print(f"   Level {level}: {count} sites ({pct:.1f}%)")

if __name__ == "__main__":
    create_cpi_rasters()