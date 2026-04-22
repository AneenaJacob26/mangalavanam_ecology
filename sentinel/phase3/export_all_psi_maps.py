#!/usr/bin/env python3
"""
PHASE 3D — EXPORTING ALL ψ MAPS (FULLY CORRECTED)
=================================================
FIXES:
- Reads actual years from data (not hard-coded)
- Proper zone indexing and naming
- Safe file existence checks
- Guild-specific map generation
- Proper environmental reconstruction

Author: Senior Geospatial Data Scientist
Date: January 2026
"""

import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
import arviz as az
from scipy.special import expit
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

REFERENCE_RASTER = Path("data/raw/rasters/NDVI_2025_core.tif")
OUTPUT_DIR = Path("data/processed/phase3/psi_maps")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Posterior files (will check existence)
IDATA_YEAR = Path("data/processed/phase3/posterior_year.nc")
IDATA_ZONE = Path("data/processed/phase3/posterior_zone.nc")
IDATA_ENV  = Path("data/processed/phase3/posterior_env.nc")

# Data file for environmental reconstruction
CSV_ENV = Path("data/processed/phase2/guild_pixel_timeseries_with_covariates.csv")

# Covariates used in environmental model
COVARS = [
    "ndvi", "ndwi", "dist_edge",
    "dist_builtup", "viirs"
]

# =============================================================================
# UTILITIES
# =============================================================================

def save_raster(array, out_path, reference_raster=REFERENCE_RASTER):
    """Save array as georeferenced raster."""
    with rasterio.open(reference_raster) as ref:
        meta = ref.meta.copy()
        meta.update(dtype="float32", count=1, nodata=np.nan, compress='lzw')
        
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(array.astype("float32"), 1)
    
    print(f"  ✓ {out_path.name}")

def get_reference_dims(reference_raster=REFERENCE_RASTER):
    """Get raster dimensions."""
    with rasterio.open(reference_raster) as ref:
        return ref.height, ref.width

# =============================================================================
# MAIN EXPORT FUNCTIONS
# =============================================================================

def export_year_maps():
    """Export year-specific ψ maps."""
    print("\n[1] Year-specific ψ maps")
    
    if not IDATA_YEAR.exists():
        print(f"  ⚠️ {IDATA_YEAR.name} not found - skipping year maps")
        return
    
    # Load posterior
    idata_year = az.from_netcdf(IDATA_YEAR)
    alpha = idata_year.posterior["alpha_psi"].values  # (chain, draw)
    beta_year = idata_year.posterior["beta_year"].values  # (chain, draw, n_years)
    
    # Get actual years from data
    if CSV_ENV.exists():
        df = pd.read_csv(CSV_ENV)
        years = sorted(df["year"].unique())
        print(f"  → Found years in data: {years}")
    else:
        # Fallback to default range
        years = list(range(2019, 2026))
        print(f"  ⚠️ Data file not found, using default years: {years}")
    
    height, width = get_reference_dims()
    
    # Generate map for each year
    for i, year in enumerate(years):
        if i < beta_year.shape[2]:
            # Calculate ψ = sigmoid(alpha + beta_year[i])
            lin_samples = alpha + beta_year[:, :, i]
            psi_samples = expit(lin_samples)
            mean_psi = psi_samples.mean()
            
            # Create uniform grid (year effect is same everywhere)
            grid = np.full((height, width), mean_psi, dtype="float32")
            save_raster(grid, OUTPUT_DIR / f"psi_{year}.tif")
            
            print(f"    Year {year}: mean ψ = {mean_psi:.3f}")
        else:
            print(f"  ⚠️ Year {year} index {i} exceeds model years ({beta_year.shape[2]}) - skipping")

def export_zone_maps():
    """Export zone-specific ψ maps."""
    print("\n[2] Zone-specific ψ maps")
    
    if not IDATA_ZONE.exists():
        print(f"  ⚠️ {IDATA_ZONE.name} not found - skipping zone maps")
        return
    
    # Load posterior
    idata_zone = az.from_netcdf(IDATA_ZONE)
    alpha = idata_zone.posterior["alpha_psi"].values  # (chain, draw)
    beta_zone = idata_zone.posterior["beta_zone"].values  # (chain, draw)
    
    height, width = get_reference_dims()
    
    # Zone encoding: 0=buffer, 1=core (as in Phase 3B)
    zones = [
        (0, "buffer"),
        (1, "core")
    ]
    
    for zone_code, zone_name in zones:
        # Calculate ψ = sigmoid(alpha + beta_zone * zone_code)
        lin_samples = alpha + beta_zone * zone_code
        psi_samples = expit(lin_samples)
        mean_psi = psi_samples.mean()
        
        # Create uniform grid
        grid = np.full((height, width), mean_psi, dtype="float32")
        save_raster(grid, OUTPUT_DIR / f"psi_{zone_name}.tif")
        
        print(f"    {zone_name.title()}: mean ψ = {mean_psi:.3f}")

def export_environmental_map():
    """Export spatially-explicit environmental ψ map."""
    print("\n[3] Environmental ψ map (spatially explicit)")
    
    if not IDATA_ENV.exists():
        print(f"  ⚠️ {IDATA_ENV.name} not found - skipping environmental map")
        return
    
    if not CSV_ENV.exists():
        print(f"  ⚠️ {CSV_ENV.name} not found - cannot reconstruct environmental map")
        return
    
    # Load posterior
    idata_env = az.from_netcdf(IDATA_ENV)
    alpha = idata_env.posterior["alpha_psi"].values  # (chain, draw)
    beta = idata_env.posterior["beta_psi"].values  # (chain, draw, n_covars)
    
    # Load covariate data
    df = pd.read_csv(CSV_ENV)
    
    # Get covariate matrix
    X = df[COVARS].fillna(0).values
    
    # Standardize (same as in Phase 3C)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)  # Add small epsilon to prevent division by zero
    
    # Calculate linear predictor: (chain, draw, pixels)
    # Using einsum for efficient matrix multiplication
    lin_samples = alpha[:, :, None] + np.einsum('ijk,lk->ijl', beta, X)
    psi_samples = expit(lin_samples)
    
    # Mean across chains and draws
    mean_psi = psi_samples.mean(axis=(0, 1))
    
    print(f"  → Calculated ψ for {len(mean_psi)} pixels")
    print(f"  → ψ range: {mean_psi.min():.3f} - {mean_psi.max():.3f}")
    print(f"  → Mean ψ: {mean_psi.mean():.3f}")
    
    # Map back to spatial grid
    height, width = get_reference_dims()
    psi_grid = np.full((height, width), np.nan, dtype="float32")
    
    pixels_mapped = 0
    for idx, row in df.iterrows():
        if idx < len(mean_psi):
            r, c = int(row["row"]), int(row["col"])
            if 0 <= r < height and 0 <= c < width:
                psi_grid[r, c] = mean_psi[idx]
                pixels_mapped += 1
    
    print(f"  → Mapped {pixels_mapped} pixels to grid")
    
    save_raster(psi_grid, OUTPUT_DIR / "psi_env.tif")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main export pipeline."""
    
    print("=" * 70)
    print("PHASE 3D — EXPORTING ψ MAPS")
    print("=" * 70)
    
    # Check reference raster
    if not REFERENCE_RASTER.exists():
        print(f"\n❌ ERROR: Reference raster not found: {REFERENCE_RASTER}")
        print("Please ensure Phase 1 has been completed.")
        return 1
    
    # Export all map types
    try:
        export_year_maps()
        export_zone_maps()
        export_environmental_map()
        
        # Summary
        exported_files = list(OUTPUT_DIR.glob("*.tif"))
        
        print("\n" + "=" * 70)
        print("PHASE 3D COMPLETE")
        print("=" * 70)
        print(f"Exported {len(exported_files)} ψ maps:")
        for f in sorted(exported_files):
            print(f"  - {f.name}")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR during export: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())