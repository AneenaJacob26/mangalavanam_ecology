#!/usr/bin/env python3
"""
Mangalavanam Adaptive Sentinel
Phase 1e: Occupancy Raster Validation

Purpose:
- Verify spatial alignment of all occupancy rasters
- Check value encoding (0, 1, 255)
- Validate effort presence and detection rates
- Generate validation reports

Author: Senior Geospatial Data Scientist
Date: January 2026
"""

import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# =====================================================================
# CONFIGURATION
# =====================================================================

NDVI_MASTER = "data/raw/rasters/NDVI_2025_core.tif"
RASTER_DIR = Path("data/processed/phase1/guild_occupancy_rasters")
OUTPUT_DIR = Path("data/processed/phase1/validation")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# LOAD REFERENCE GRID
# =====================================================================

print("=" * 70)
print("PHASE 1e — OCCUPANCY RASTER VALIDATION")
print("=" * 70)

print("\n[1/4] Loading reference grid...")
try:
    with rasterio.open(NDVI_MASTER) as ref:
        REF_META = {
            "crs": ref.crs,
            "transform": ref.transform,
            "width": ref.width,
            "height": ref.height,
            "res": ref.res,
        }
    print(f"  ✓ Reference grid loaded")
    print(f"    CRS: {REF_META['crs']}")
    print(f"    Dimensions: {REF_META['width']} × {REF_META['height']}")
    print(f"    Resolution: {REF_META['res']}")
except Exception as e:
    print(f"  ✗ Error loading reference grid: {e}")
    exit(1)


# =====================================================================
# VALIDATION LOOP
# =====================================================================

print("\n[2/4] Validating occupancy rasters...")

raster_files = sorted(RASTER_DIR.glob("*.tif"))

if not raster_files:
    print(f"  ✗ No rasters found in {RASTER_DIR}")
    print(f"  ⚠ Please run Phase 1c first!")
    exit(1)

print(f"  → Found {len(raster_files)} rasters to validate")

validation_records = []
value_summary_records = []

for raster_path in raster_files:
    try:
        with rasterio.open(raster_path) as src:
            # Read data
            data = src.read(1)
            
            # Check alignment
            crs_match = src.crs == REF_META["crs"]
            transform_match = src.transform == REF_META["transform"]
            dimension_match = (
                src.width == REF_META["width"]
                and src.height == REF_META["height"]
            )
            
            # Check value encoding
            unique_values, counts = np.unique(data, return_counts=True)
            value_counts = dict(zip(unique_values, counts))
            
            valid_values_only = set(unique_values).issubset({0, 1, 255})
            
            # Calculate pixel statistics
            pixels_nodata = int(value_counts.get(255, 0))
            pixels_absent = int(value_counts.get(0, 0))  # Surveyed but not detected
            pixels_present = int(value_counts.get(1, 0))  # Surveyed and detected
            pixels_effort = pixels_absent + pixels_present
            
            # Occupancy rate
            occupancy_rate = (
                (pixels_present / pixels_effort * 100) if pixels_effort > 0 else 0
            )
            
            # Overall validation status
            all_checks_passed = (
                crs_match and 
                transform_match and 
                dimension_match and 
                valid_values_only and
                pixels_effort > 0
            )
            
            # Record validation results
            validation_records.append({
                "file": raster_path.name,
                "crs_match": crs_match,
                "transform_match": transform_match,
                "dimension_match": dimension_match,
                "valid_values_only": valid_values_only,
                "has_effort": pixels_effort > 0,
                "all_checks_passed": all_checks_passed
            })
            
            # Record value summary
            value_summary_records.append({
                "file": raster_path.name,
                "pixels_nodata": pixels_nodata,
                "pixels_absent": pixels_absent,
                "pixels_present": pixels_present,
                "pixels_effort": pixels_effort,
                "occupancy_rate": occupancy_rate
            })
            
    except Exception as e:
        print(f"  ✗ Error validating {raster_path.name}: {e}")
        validation_records.append({
            "file": raster_path.name,
            "crs_match": False,
            "transform_match": False,
            "dimension_match": False,
            "valid_values_only": False,
            "has_effort": False,
            "all_checks_passed": False
        })

print(f"  ✓ Validated {len(raster_files)} rasters")


# =====================================================================
# GENERATE REPORTS
# =====================================================================

print("\n[3/4] Generating validation reports...")

# Validation report
validation_df = pd.DataFrame(validation_records)
validation_file = OUTPUT_DIR / "raster_alignment_report.csv"
validation_df.to_csv(validation_file, index=False)
print(f"  ✓ Saved: {validation_file}")

# Value summary report
value_summary_df = pd.DataFrame(value_summary_records)
value_summary_file = OUTPUT_DIR / "raster_value_summary.csv"
value_summary_df.to_csv(value_summary_file, index=False)
print(f"  ✓ Saved: {value_summary_file}")

# Human-readable diagnostic
diagnostic_file = OUTPUT_DIR / "occupancy_diagnostics.txt"
with open(diagnostic_file, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("PHASE 1e — OCCUPANCY VALIDATION REPORT\n")
    f.write("=" * 70 + "\n\n")
    
    # Overall summary
    total_rasters = len(validation_df)
    passed_rasters = validation_df['all_checks_passed'].sum()
    
    f.write(f"SUMMARY\n")
    f.write("-" * 70 + "\n")
    f.write(f"Total rasters: {total_rasters}\n")
    f.write(f"Passed validation: {passed_rasters}\n")
    f.write(f"Failed validation: {total_rasters - passed_rasters}\n")
    f.write(f"Pass rate: {passed_rasters/total_rasters*100:.1f}%\n\n")
    
    # Detailed results
    f.write(f"DETAILED RESULTS\n")
    f.write("=" * 70 + "\n\n")
    
    for _, val_row in validation_df.iterrows():
        sum_row = value_summary_df[value_summary_df['file'] == val_row['file']].iloc[0]
        
        status = "✓ PASS" if val_row['all_checks_passed'] else "✗ FAIL"
        f.write(f"{status} — {val_row['file']}\n")
        f.write("-" * 70 + "\n")
        
        # Alignment checks
        f.write(f"Spatial Alignment:\n")
        f.write(f"  CRS match: {'✓' if val_row['crs_match'] else '✗'}\n")
        f.write(f"  Transform match: {'✓' if val_row['transform_match'] else '✗'}\n")
        f.write(f"  Dimension match: {'✓' if val_row['dimension_match'] else '✗'}\n")
        
        # Value checks
        f.write(f"\nValue Encoding:\n")
        f.write(f"  Valid values only (0,1,255): {'✓' if val_row['valid_values_only'] else '✗'}\n")
        f.write(f"  Has effort data: {'✓' if val_row['has_effort'] else '✗'}\n")
        
        # Statistics
        f.write(f"\nPixel Statistics:\n")
        f.write(f"  NoData pixels: {sum_row['pixels_nodata']:,}\n")
        f.write(f"  Absent pixels (0): {sum_row['pixels_absent']:,}\n")
        f.write(f"  Present pixels (1): {sum_row['pixels_present']:,}\n")
        f.write(f"  Total effort: {sum_row['pixels_effort']:,}\n")
        f.write(f"  Occupancy rate: {sum_row['occupancy_rate']:.1f}%\n")
        f.write("\n")

print(f"  ✓ Saved: {diagnostic_file}")


# =====================================================================
# SUMMARY STATISTICS
# =====================================================================

print("\n[4/4] Computing summary statistics...")

# Overall validation statistics
total = len(validation_df)
passed = validation_df['all_checks_passed'].sum()
failed = total - passed

print(f"\n  Validation Results:")
print(f"    Total rasters: {total}")
print(f"    Passed: {passed} ({passed/total*100:.1f}%)")
print(f"    Failed: {failed} ({failed/total*100:.1f}%)")

if failed > 0:
    print(f"\n  ⚠ Failed checks:")
    for check in ['crs_match', 'transform_match', 'dimension_match', 
                  'valid_values_only', 'has_effort']:
        failures = (~validation_df[check]).sum()
        if failures > 0:
            print(f"    {check}: {failures} rasters")

# Occupancy statistics
print(f"\n  Occupancy Statistics:")
print(f"    Mean occupancy rate: {value_summary_df['occupancy_rate'].mean():.1f}%")
print(f"    Min occupancy rate: {value_summary_df['occupancy_rate'].min():.1f}%")
print(f"    Max occupancy rate: {value_summary_df['occupancy_rate'].max():.1f}%")
print(f"    Median occupancy rate: {value_summary_df['occupancy_rate'].median():.1f}%")

# Parse guild/year/zone from filenames
value_summary_df['guild'] = value_summary_df['file'].str.split('_').str[0]
value_summary_df['year'] = value_summary_df['file'].str.split('_').str[1].astype(int)
value_summary_df['zone'] = value_summary_df['file'].str.split('_').str[2].str.replace('.tif', '')

print(f"\n  Occupancy by Guild:")
for guild, group in value_summary_df.groupby('guild'):
    mean_occ = group['occupancy_rate'].mean()
    print(f"    {guild}: {mean_occ:.1f}%")

print(f"\n  Occupancy by Zone:")
for zone, group in value_summary_df.groupby('zone'):
    mean_occ = group['occupancy_rate'].mean()
    print(f"    {zone}: {mean_occ:.1f}%")


# =====================================================================
# FINAL SUMMARY
# =====================================================================

print("\n" + "=" * 70)
print("PHASE 1e COMPLETE")
print("=" * 70)
print(f"Validation reports saved to: {OUTPUT_DIR}")
print(f"  - {validation_file.name}")
print(f"  - {value_summary_file.name}")
print(f"  - {diagnostic_file.name}")

if passed == total:
    print(f"\n✅ All rasters passed validation!")
else:
    print(f"\n⚠ {failed} rasters failed validation - see reports for details")

print("=" * 70)