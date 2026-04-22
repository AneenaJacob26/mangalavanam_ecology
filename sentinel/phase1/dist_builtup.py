#!/usr/bin/env python3
"""
Regenerate Missing Distance Rasters
Specifically creates DIST_BUILTUP.tif which is 76% missing

This script:
1. Creates urban/built-up mask from VIIRS night lights
2. Calculates distance from each pixel to nearest built-up area
3. Saves as DIST_BUILTUP.tif matching Phase 1 grid
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from scipy.ndimage import distance_transform_edt
from pathlib import Path
import glob

# =============================================================================
# CONFIGURATION
# =============================================================================

RAW_RASTERS_DIR = Path("data/raw/rasters")
PHASE1_REFERENCE = Path("data/raw/rasters/DIST_EDGE.tif")  # Use this as reference grid

OUTPUT_FILE = RAW_RASTERS_DIR / "DIST_BUILTUP.tif"

# =============================================================================
# FUNCTIONS
# =============================================================================

def load_reference_grid():
    """Load reference raster to match CRS, resolution, bounds"""
    print("📐 Loading reference grid...")
    
    if not PHASE1_REFERENCE.exists():
        print(f"❌ Reference file not found: {PHASE1_REFERENCE}")
        print("   Trying alternative references...")
        
        # Try NDVI as reference
        alt_ref = sorted(glob.glob(str(RAW_RASTERS_DIR / "NDVI_*_core.tif")))
        if len(alt_ref) > 0:
            ref_file = Path(alt_ref[0])
            print(f"   Using: {ref_file.name}")
        else:
            raise FileNotFoundError("No suitable reference raster found!")
    else:
        ref_file = PHASE1_REFERENCE
    
    with rasterio.open(ref_file) as src:
        ref_meta = src.meta.copy()
        ref_bounds = src.bounds
        ref_transform = src.transform
        ref_crs = src.crs
        ref_shape = (src.height, src.width)
    
    print(f"   ✓ Reference: {ref_file.name}")
    print(f"   Shape: {ref_shape}")
    print(f"   CRS: {ref_crs}")
    print(f"   Resolution: {ref_transform.a:.2f}m")
    
    return ref_meta, ref_bounds, ref_transform, ref_shape

def create_urban_mask_from_viirs():
    """Create urban/built-up mask from VIIRS night lights"""
    print("\n🌃 Creating urban mask from VIIRS night lights...")
    
    # Load VIIRS rasters
    viirs_files = sorted(glob.glob(str(RAW_RASTERS_DIR / "VIIRS_*_*.tif")))
    
    if len(viirs_files) == 0:
        print("❌ No VIIRS files found!")
        return None
    
    print(f"   Found {len(viirs_files)} VIIRS files")
    
    # Load reference grid
    ref_meta, ref_bounds, ref_transform, ref_shape = load_reference_grid()
    
    # Accumulate VIIRS values
    viirs_sum = np.zeros(ref_shape, dtype=np.float32)
    viirs_count = np.zeros(ref_shape, dtype=np.int16)
    
    print("   Loading VIIRS data...")
    for i, filepath in enumerate(viirs_files):
        try:
            with rasterio.open(filepath) as src:
                # Read and resample if needed
                if src.shape != ref_shape:
                    # Resample to reference grid
                    from rasterio.warp import reproject, Resampling
                    
                    data = np.zeros(ref_shape, dtype=np.float32)
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=data,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_meta['crs'],
                        resampling=Resampling.bilinear
                    )
                else:
                    data = src.read(1).astype(np.float32)
                
                # Handle nodata
                if src.nodata is not None:
                    data = np.where(data == src.nodata, np.nan, data)
                
                # Accumulate
                valid_mask = np.isfinite(data)
                viirs_sum[valid_mask] += data[valid_mask]
                viirs_count[valid_mask] += 1
        
        except Exception as e:
            print(f"   ⚠️  Error loading {Path(filepath).name}: {e}")
            continue
        
        if (i + 1) % 5 == 0:
            print(f"   Processed {i+1}/{len(viirs_files)} files...")
    
    # Calculate mean VIIRS
    viirs_mean = np.zeros(ref_shape, dtype=np.float32)
    valid_count = viirs_count > 0
    viirs_mean[valid_count] = viirs_sum[valid_count] / viirs_count[valid_count]
    
    print(f"   ✓ Mean VIIRS calculated")
    print(f"   Range: {np.nanmin(viirs_mean):.3f} to {np.nanmax(viirs_mean):.3f}")
    
    # Create urban mask using threshold
    # Pixels with VIIRS > 75th percentile = urban
    threshold = np.nanpercentile(viirs_mean[viirs_mean > 0], 75)
    print(f"   Urban threshold (75th percentile): {threshold:.3f}")
    
    urban_mask = viirs_mean > threshold
    urban_pixels = np.sum(urban_mask)
    urban_pct = (urban_pixels / urban_mask.size) * 100
    
    print(f"   ✓ Urban mask created: {urban_pixels} pixels ({urban_pct:.1f}%)")
    
    return urban_mask, ref_meta

def calculate_distance_transform(mask, ref_meta):
    """Calculate distance from each pixel to nearest True pixel in mask"""
    print("\n📏 Calculating distance transform...")
    
    # Invert mask (we want distance to urban, so urban = 0, non-urban = 1)
    inverted = ~mask
    
    # Calculate Euclidean distance
    # distance_transform_edt returns distance in pixels
    dist_pixels = distance_transform_edt(inverted)
    
    # Convert to meters (multiply by pixel resolution)
    pixel_size = abs(ref_meta['transform'].a)  # Pixel width in meters
    dist_meters = dist_pixels * pixel_size
    
    print(f"   ✓ Distance calculated")
    print(f"   Min distance: {dist_meters.min():.1f}m")
    print(f"   Max distance: {dist_meters.max():.1f}m")
    print(f"   Mean distance: {dist_meters.mean():.1f}m")
    
    return dist_meters

def save_distance_raster(distance_array, ref_meta, output_path):
    """Save distance array as GeoTIFF"""
    print(f"\n💾 Saving distance raster...")
    
    # Update metadata
    out_meta = ref_meta.copy()
    out_meta.update({
        'dtype': 'float32',
        'count': 1,
        'nodata': -9999.0
    })
    
    # Save
    with rasterio.open(output_path, 'w', **out_meta) as dst:
        dst.write(distance_array.astype(np.float32), 1)
    
    print(f"   ✓ Saved: {output_path}")
    
    # Verify
    with rasterio.open(output_path) as src:
        check_data = src.read(1)
        valid_pixels = np.sum(np.isfinite(check_data))
        total_pixels = check_data.size
        valid_pct = (valid_pixels / total_pixels) * 100
        
        print(f"   Verification: {valid_pixels}/{total_pixels} valid pixels ({valid_pct:.1f}%)")

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution"""
    
    print("="*80)
    print("🔧 REGENERATING DIST_BUILTUP.TIF")
    print("="*80)
    print()
    
    # Check if already exists
    if OUTPUT_FILE.exists():
        print(f"⚠️  Warning: {OUTPUT_FILE.name} already exists!")
        response = input("   Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("   Aborted.")
            return
        print()
    
    try:
        # Step 1: Create urban mask from VIIRS
        urban_mask, ref_meta = create_urban_mask_from_viirs()
        
        if urban_mask is None:
            print("\n❌ Failed to create urban mask!")
            print("   Alternative: Create DIST_BUILTUP from OSM data or manual digitization")
            return
        
        # Step 2: Calculate distance transform
        dist_builtup = calculate_distance_transform(urban_mask, ref_meta)
        
        # Step 3: Save
        save_distance_raster(dist_builtup, ref_meta, OUTPUT_FILE)
        
        print("\n" + "="*80)
        print("✅ DIST_BUILTUP.TIF SUCCESSFULLY REGENERATED!")
        print("="*80)
        print()
        print("Next steps:")
        print("1. Verify raster in QGIS")
        print("2. Re-run Phase 2: python sentinel/phase2_covariate_extraction.py")
        print("3. Validate: python sentinel/validate_data.py")
        print()
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("Troubleshooting:")
        print("1. Check if VIIRS files exist and have valid data")
        print("2. Check if reference raster (DIST_EDGE.tif) exists")
        print("3. Try manual creation in QGIS:")
        print("   - Create urban polygon from satellite imagery")
        print("   - Raster → Analysis → Proximity (distance from)")
        print("   - Save as DIST_BUILTUP.tif")

if __name__ == "__main__":
    main()
