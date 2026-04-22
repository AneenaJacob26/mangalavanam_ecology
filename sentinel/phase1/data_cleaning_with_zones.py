#!/usr/bin/env python3
"""
Mangalavanam Adaptive Sentinel
Phase 1a: Data Cleaning & Spatial Zone Assignment

Purpose:
- Load and clean eBird occurrence and metadata
- Perform quality control filters
- Assign spatial zones (Core/Buffer/Outside) to each observation
- Prepare data for guild assignment in Phase 1b

Author: Senior Geospatial Data Scientist
Date: January 2026
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def normalize_columns(df):
    """Standardize column names: uppercase with spaces."""
    df.columns = df.columns.str.strip().str.upper().str.replace('_', ' ')
    return df


def load_and_clean_ebird_data(raw_dir):
    """
    Load and perform initial cleaning of eBird data.
    
    Args:
        raw_dir: Path to directory containing eBird CSVs
        
    Returns:
        tuple: (occurrence_df, metadata_df)
    """
    print("\n[1/5] Loading raw eBird data...")
    
    # Load files
    occ = pd.read_csv(
        raw_dir / "ebd_occurrence.csv",
        low_memory=False
    )
    
    meta = pd.read_csv(
        raw_dir / "ebd_metadata.csv",
        low_memory=False
    )
    
    # Normalize column names
    occ = normalize_columns(occ)
    meta = normalize_columns(meta)
    
    print(f"  ✓ Loaded {len(occ):,} occurrences")
    print(f"  ✓ Loaded {len(meta):,} checklists")
    
    return occ, meta


def filter_occurrence_data(occ):
    """
    Apply quality filters to occurrence records.
    
    Args:
        occ: Raw occurrence DataFrame
        
    Returns:
        pd.DataFrame: Filtered occurrences
    """
    print("\n[2/5] Filtering occurrence data...")
    
    initial_count = len(occ)
    
    # Keep only approved records
    occ = occ[occ["APPROVED"] == 1]
    
    # Keep essential columns
    essential_cols = [
        "SAMPLING EVENT IDENTIFIER",
        "SCIENTIFIC NAME",
        "COMMON NAME",
        "OBSERVATION COUNT"
    ]
    
    occ = occ[essential_cols]
    
    # Remove duplicates
    occ = occ.drop_duplicates()
    
    filtered_count = len(occ)
    print(f"  ✓ Kept {filtered_count:,} of {initial_count:,} records "
          f"({filtered_count/initial_count*100:.1f}%)")
    
    return occ


def filter_metadata(meta):
    """
    Apply quality filters to checklist metadata.
    
    Args:
        meta: Raw metadata DataFrame
        
    Returns:
        pd.DataFrame: Filtered metadata
    """
    print("\n[3/5] Filtering checklist metadata...")
    
    initial_count = len(meta)
    
    # Quality filters
    meta = meta[
        (meta["ALL SPECIES REPORTED"] == 1) &
        (meta["DURATION MINUTES"] > 0) &
        (meta["DURATION MINUTES"] <= 300) &  # Max 5 hours
        (meta["EFFORT DISTANCE KM"] >= 0) &
        (meta["EFFORT DISTANCE KM"] <= 5)    # Max 5 km
    ]
    
    # Keep essential columns
    essential_cols = [
        "SAMPLING EVENT IDENTIFIER",
        "LATITUDE",
        "LONGITUDE",
        "OBSERVATION DATE",
        "DURATION MINUTES",
        "EFFORT DISTANCE KM",
        "NUMBER OBSERVERS"
    ]
    
    meta = meta[essential_cols]
    
    filtered_count = len(meta)
    print(f"  ✓ Kept {filtered_count:,} of {initial_count:,} checklists "
          f"({filtered_count/initial_count*100:.1f}%)")
    
    return meta


def assign_spatial_zones(meta, core_file, buffer_file):
    """
    Assign spatial zone (Core/Buffer/Outside) to each checklist.
    
    Args:
        meta: Metadata DataFrame with LATITUDE/LONGITUDE
        core_file: Path to Core sanctuary GeoJSON
        buffer_file: Path to Buffer zone GeoJSON
        
    Returns:
        pd.DataFrame: Metadata with ZONE column added
    """
    print("\n[4/5] Assigning spatial zones...")
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        meta,
        geometry=[Point(xy) for xy in zip(meta['LONGITUDE'], meta['LATITUDE'])],
        crs="EPSG:4326"
    )
    
    # Load zone polygons
    core = gpd.read_file(core_file).to_crs(gdf.crs)
    buffer = gpd.read_file(buffer_file).to_crs(gdf.crs)
    
    print(f"  → Core area: {core.unary_union.area * 111320**2 / 10000:.2f} ha")
    print(f"  → Buffer area: {buffer.unary_union.area * 111320**2 / 10000:.2f} ha")
    
    # Initialize all as Outside
    gdf['ZONE'] = 'Outside'
    
    # Core has priority (most restrictive zone)
    core_mask = gdf.within(core.unary_union)
    gdf.loc[core_mask, 'ZONE'] = 'Core'
    
    # Buffer (excluding areas already in Core)
    buffer_mask = gdf.within(buffer.unary_union) & ~core_mask
    gdf.loc[buffer_mask, 'ZONE'] = 'Buffer'
    
    # Print zone distribution
    print(f"  ✓ Zone assignment:")
    for zone, count in gdf['ZONE'].value_counts().items():
        print(f"    - {zone}: {count:,} checklists")
    
    # Convert back to DataFrame (drop geometry)
    return gdf.drop(columns='geometry')


def merge_and_finalize(occ, meta):
    """
    Merge occurrence and metadata, add temporal features.
    
    Args:
        occ: Filtered occurrences
        meta: Filtered metadata with zones
        
    Returns:
        pd.DataFrame: Final cleaned dataset
    """
    print("\n[5/5] Merging and finalizing dataset...")
    
    # Inner join to keep only valid checklist-observation pairs
    df = occ.merge(
        meta,
        on="SAMPLING EVENT IDENTIFIER",
        how="inner"
    )
    
    print(f"  ✓ Merged dataset: {len(df):,} observations")
    
    # Add temporal features
    df["OBSERVATION DATE"] = pd.to_datetime(df["OBSERVATION DATE"])
    df["YEAR"] = df["OBSERVATION DATE"].dt.year
    df["MONTH"] = df["OBSERVATION DATE"].dt.month
    df["DAY OF YEAR"] = df["OBSERVATION DATE"].dt.dayofyear
    
    # Temporal filter (2019-2025)
    df = df[(df["YEAR"] >= 2019) & (df["YEAR"] <= 2025)]
    
    print(f"  ✓ Temporal filter (2019-2025): {len(df):,} observations")
    print(f"\n  Year distribution:")
    for year, count in sorted(df['YEAR'].value_counts().items()):
        print(f"    {year}: {count:,}")
    
    return df


def main():
    """Main execution pipeline."""
    
    print("=" * 70)
    print("PHASE 1a — DATA CLEANING & SPATIAL ZONE ASSIGNMENT")
    print("=" * 70)
    
    # =====================================================================
    # CONFIGURATION
    # =====================================================================
    RAW_EBIRD_DIR = Path("data/raw/occurrences/EBD")
    CORE_GEOJSON = Path("data/raw/shapes/Mangalavanam_Core.geojson")
    BUFFER_GEOJSON = Path("data/raw/shapes/Mangalavanam_Buffer.geojson")
    
    OUTPUT_FILE = Path("data/processed/phase1/ebird_cleaned_with_zones.csv")
    
    # =====================================================================
    # EXECUTION
    # =====================================================================
    
    try:
        # Step 1: Load raw data
        occ, meta = load_and_clean_ebird_data(RAW_EBIRD_DIR)
        
        # Step 2-3: Filter data
        occ = filter_occurrence_data(occ)
        meta = filter_metadata(meta)
        
        # Step 4: Assign spatial zones
        meta = assign_spatial_zones(meta, CORE_GEOJSON, BUFFER_GEOJSON)
        
        # Step 5: Merge and finalize
        df = merge_and_finalize(occ, meta)
        
        # Save output
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        
        # ================================================================
        # SUMMARY
        # ================================================================
        print("\n" + "=" * 70)
        print("PHASE 1a COMPLETE")
        print("=" * 70)
        print(f"Output: {OUTPUT_FILE}")
        print(f"Total observations: {len(df):,}")
        print(f"Unique species: {df['SCIENTIFIC NAME'].nunique():,}")
        print(f"Unique checklists: {df['SAMPLING EVENT IDENTIFIER'].nunique():,}")
        print(f"\nZone distribution:")
        for zone, count in df['ZONE'].value_counts().items():
            print(f"  {zone}: {count:,} observations")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Required file not found")
        print(f"   {e}")
        print(f"\nPlease ensure the following files exist:")
        print(f"  - {RAW_EBIRD_DIR}/ebd_occurrence.csv")
        print(f"  - {RAW_EBIRD_DIR}/ebd_metadata.csv")
        print(f"  - {CORE_GEOJSON}")
        print(f"  - {BUFFER_GEOJSON}")
        return 1
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())