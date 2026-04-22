#!/usr/bin/env python3
"""
Mangalavanam Adaptive Sentinel
Phase 1c: Guild Occupancy Rasterization (ZONE-AWARE)

Purpose:
- Generate binary occupancy rasters for each guild × year × zone combination
- Rasters encode: 0 = surveyed but not detected, 1 = detected, 255 = NoData

Author: Senior Geospatial Data Scientist
Date: January 2026
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Point
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def load_master_grid(reference_raster):
    """
    Load spatial reference parameters from master grid.
    
    Args:
        reference_raster: Path to reference raster (e.g., NDVI)
        
    Returns:
        dict: Grid parameters (crs, transform, width, height)
    """
    with rasterio.open(reference_raster) as src:
        return {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
        }


def pixel_index(x, y, transform):
    """
    Convert geographic coordinates to pixel indices.
    
    Args:
        x: X coordinate (easting)
        y: Y coordinate (northing)
        transform: Rasterio affine transform
        
    Returns:
        tuple: (row, col) pixel indices
    """
    col = int((x - transform.c) / transform.a)
    row = int((y - transform.f) / transform.e)
    return row, col


def build_occupancy_raster(effort_pixels, detection_pixels, grid):
    """
    Create binary occupancy raster from pixel sets.
    
    Encoding:
    - 255: No survey effort (NoData)
    - 0: Surveyed, species not detected
    - 1: Surveyed, species detected
    
    Args:
        effort_pixels: Set of (row, col) tuples for all surveyed pixels
        detection_pixels: Set of (row, col) tuples where species detected
        grid: Grid parameters dict
        
    Returns:
        np.ndarray: Occupancy raster (uint8)
    """
    # Initialize with NoData
    raster = np.full(
        (grid["height"], grid["width"]),
        255,
        dtype=np.uint8
    )
    
    # Mark effort pixels (surveyed = 0)
    for r, c in effort_pixels:
        if 0 <= r < grid["height"] and 0 <= c < grid["width"]:
            raster[r, c] = 0
    
    # Mark detection pixels (detected = 1)
    for r, c in detection_pixels:
        if 0 <= r < grid["height"] and 0 <= c < grid["width"]:
            raster[r, c] = 1
    
    return raster


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def main():
    
    print("=" * 70)
    print("PHASE 1c — GUILD OCCUPANCY RASTERIZATION")
    print("=" * 70)
    
    # =================================================================
    # CONFIGURATION
    # =================================================================
    NDVI_GRID = "data/raw/rasters/NDVI_2025_core.tif"
    INPUT_FILE = "data/processed/phase1/ebird_with_guilds_and_zones.csv"
    
    OUTPUT_DIR = Path("data/processed/phase1/guild_occupancy_rasters")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    GUILDS = ["Wetland", "Forest", "Urban"]
    ZONES = ["Core", "Buffer"]
    YEARS = range(2019, 2026)
    
    # =================================================================
    # STEP 1: Load master grid
    # =================================================================
    print("\n[1/4] Loading master grid reference...")
    try:
        grid = load_master_grid(NDVI_GRID)
        print(f"  ✓ CRS: {grid['crs']}")
        print(f"  ✓ Dimensions: {grid['width']} × {grid['height']}")
        print(f"  ✓ Transform: {grid['transform']}")
    except Exception as e:
        print(f"  ✗ Error loading master grid: {e}")
        return 1
    
    # =================================================================
    # STEP 2: Load observation data
    # =================================================================
    print("\n[2/4] Loading observation data...")
    try:
        df = pd.read_csv(INPUT_FILE)
        
        # Verify required columns
        required_cols = [
            'SAMPLING EVENT IDENTIFIER',
            'SCIENTIFIC NAME',
            'LATITUDE',
            'LONGITUDE',
            'OBSERVATION DATE',
            'YEAR',
            'ZONE',
            'GUILD'
        ]
        
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.LONGITUDE, df.LATITUDE),
            crs="EPSG:4326"
        ).to_crs(grid["crs"])
        
        print(f"  ✓ Loaded {len(gdf):,} observations")
        print(f"  ✓ Unique checklists: {gdf['SAMPLING EVENT IDENTIFIER'].nunique():,}")
        print(f"  ✓ Unique species: {gdf['SCIENTIFIC NAME'].nunique():,}")
        
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        return 1
    
    # =================================================================
    # STEP 3: Create checklist-level metadata
    # =================================================================
    print("\n[3/4] Preparing checklist metadata...")
    
    # Get unique checklists with their zones and coordinates
    checklist_meta = gdf[[
        'SAMPLING EVENT IDENTIFIER',
        'YEAR',
        'ZONE',
        'geometry'
    ]].drop_duplicates(subset=['SAMPLING EVENT IDENTIFIER'])
    
    print(f"  ✓ Unique checklists prepared: {len(checklist_meta):,}")
    
    # =================================================================
    # STEP 4: Rasterization loop
    # =================================================================
    print("\n[4/4] Generating occupancy rasters...")
    
    raster_count = 0
    stats_records = []
    
    for guild in GUILDS:
        print(f"\n  Guild: {guild}")
        
        # Filter to current guild
        guild_obs = gdf[gdf["GUILD"] == guild]
        
        for year in YEARS:
            # Filter checklists for this year
            year_checklists = checklist_meta[checklist_meta["YEAR"] == year]
            
            for zone in ZONES:
                # Filter to current zone
                zone_checklists = year_checklists[year_checklists["ZONE"] == zone]
                
                if zone_checklists.empty:
                    print(f"    ⊘ {year} {zone}: No effort")
                    continue
                
                # Filter observations for this zone
                zone_obs = guild_obs[
                    (guild_obs["YEAR"] == year) &
                    (guild_obs["ZONE"] == zone)
                ]
                
                # Get checklist IDs where guild was detected
                detected_checklists = set(zone_obs["SAMPLING EVENT IDENTIFIER"].unique())
                
                # Convert to pixel coordinates
                effort_pixels = set()
                detection_pixels = set()
                
                for _, row in zone_checklists.iterrows():
                    px = pixel_index(
                        row.geometry.x,
                        row.geometry.y,
                        grid["transform"]
                    )
                    effort_pixels.add(px)
                    
                    # Mark as detection if guild observed in this checklist
                    if row["SAMPLING EVENT IDENTIFIER"] in detected_checklists:
                        detection_pixels.add(px)
                
                # Build raster
                occ_raster = build_occupancy_raster(
                    effort_pixels,
                    detection_pixels,
                    grid
                )
                
                # Save raster
                output_path = OUTPUT_DIR / f"{guild}_{year}_{zone}.tif"
                
                with rasterio.open(
                    output_path,
                    "w",
                    driver="GTiff",
                    height=grid["height"],
                    width=grid["width"],
                    count=1,
                    dtype="uint8",
                    crs=grid["crs"],
                    transform=grid["transform"],
                    nodata=255,
                    compress="lzw"
                ) as dst:
                    dst.write(occ_raster, 1)
                
                # Calculate statistics
                n_effort = len(effort_pixels)
                n_detected = len(detection_pixels)
                occupancy_rate = (n_detected / n_effort * 100) if n_effort > 0 else 0
                
                print(f"    ✓ {year} {zone}: {occupancy_rate:.1f}% "
                      f"({n_detected}/{n_effort} pixels)")
                
                # Record statistics
                stats_records.append({
                    'guild': guild,
                    'year': year,
                    'zone': zone,
                    'effort_pixels': n_effort,
                    'detection_pixels': n_detected,
                    'occupancy_rate': occupancy_rate,
                    'file': output_path.name
                })
                
                raster_count += 1
    
    # =================================================================
    # SAVE STATISTICS
    # =================================================================
    stats_df = pd.DataFrame(stats_records)
    stats_file = OUTPUT_DIR / "rasterization_statistics.csv"
    stats_df.to_csv(stats_file, index=False)
    
    # =================================================================
    # SUMMARY
    # =================================================================
    print("\n" + "=" * 70)
    print("PHASE 1c COMPLETE")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total rasters created: {raster_count}")
    print(f"\nRaster specifications:")
    print(f"  CRS: {grid['crs']}")
    print(f"  Dimensions: {grid['width']} × {grid['height']}")
    print(f"  Data type: uint8")
    print(f"  Encoding: 0=surveyed/absent, 1=surveyed/present, 255=NoData")
    print(f"\nStatistics saved to: {stats_file}")
    print(f"\nOccupancy rate summary by guild:")
    summary = stats_df.groupby('guild')['occupancy_rate'].agg(['mean', 'min', 'max'])
    print(summary)
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())