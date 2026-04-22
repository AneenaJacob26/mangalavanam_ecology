#!/usr/bin/env python3
"""
Mangalavanam Adaptive Sentinel
Phase 6: Conservation Priority Index (PAPER-COMPLIANT VERSION)

CORRECTIONS TO MATCH PAPER:
- Uses REAL Shannon entropy from guild diversity
- Correct weights: 0.5 occupancy, 0.3 entropy, 0.2 threat
- Proper threat calculation with edge, light, vegetation
- Quantile-based priority classification

Formula from paper:
    CPI = 0.5×ψ̄ + 0.3×H + 0.2×Threat

Author: Conservation Data Scientist
Date: March 2026
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.enums import Resampling
import geopandas as gpd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

TEMPLATE_TIF = Path("data/raw/rasters/NDVI_2025_core.tif")

# Guild-specific ψ maps from Phase 4
PSI_MAPS_DIR = Path("data/processed/phase4")

PSI_WETLAND_TIF = PSI_MAPS_DIR / "psi_env_Wetland.tif"
PSI_FOREST_TIF = PSI_MAPS_DIR / "psi_env_Forest.tif"
PSI_URBAN_TIF = PSI_MAPS_DIR / "psi_env_Urban.tif"

# Environmental rasters for threat assessment
NDVI_TIF = Path("data/raw/rasters/NDVI_2025_core.tif")
NDWI_TIF = Path("data/raw/rasters/NDWI_2025_core.tif")
VIIRS_TIF = Path("data/raw/rasters/VIIRS_2025_core.tif")
DIST_EDGE_TIF = Path("data/raw/rasters/DIST_EDGE.tif")

# Boundary files
CORE_GEOJSON = Path("data/raw/shapes/Mangalavanam_Core.geojson")
BUFFER_GEOJSON = Path("data/raw/shapes/Mangalavanam_Buffer.geojson")

# Output
OUT_DIR = Path("data/processed/phase6")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CPI weights FROM PAPER (Equation 10)
WEIGHTS = {
    "occupancy": 0.50,  # Mean occupancy ψ̄
    "diversity": 0.30,  # Shannon entropy H
    "threat": 0.20      # Threat pressure
}

print("=" * 80)
print("PHASE 6 — CONSERVATION PRIORITY INDEX (PAPER-COMPLIANT)")
print("=" * 80)
print()

print("Formula: CPI = 0.5×ψ̄ + 0.3×H + 0.2×Threat")
print()

# =============================================================================
# UTILITIES
# =============================================================================

def read_raster_match(path, template_meta):
    """Read raster and match to template dimensions"""
    if not path.exists():
        print(f"  ⚠️ {path.name} not found")
        return None
    
    with rasterio.open(path) as src:
        arr = src.read(
            1,
            out_shape=(template_meta["height"], template_meta["width"]),
            resampling=Resampling.bilinear
        ).astype("float32")
        
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
    
    return arr

def normalize(arr):
    """Normalize array to 0-1 range"""
    arr = arr.astype("float32")
    mask = np.isfinite(arr)
    
    if mask.sum() == 0:
        return arr
    
    mn, mx = arr[mask].min(), arr[mask].max()
    
    if mn == mx:
        return np.where(mask, 0.5, np.nan)
    
    out = np.full_like(arr, np.nan, dtype="float32")
    out[mask] = (arr[mask] - mn) / (mx - mn)
    
    return out

def save_raster(arr, out_path, template_meta):
    """Save array as GeoTIFF"""
    with rasterio.open(out_path, "w", **template_meta) as dst:
        dst.write(arr.astype("float32"), 1)
    print(f"  ✓ {out_path.name}")

# =============================================================================
# PART 1: LOAD TEMPLATE
# =============================================================================

print("=" * 80)
print("PART 1: LOADING TEMPLATE")
print("=" * 80)
print()

if not TEMPLATE_TIF.exists():
    print(f"❌ ERROR: Template raster not found: {TEMPLATE_TIF}")
    exit(1)

with rasterio.open(TEMPLATE_TIF) as tmp:
    meta = tmp.meta.copy()
    transform = tmp.transform
    crs = tmp.crs
    shape = (tmp.height, tmp.width)

meta.update(dtype="float32", count=1, nodata=np.nan, compress='lzw')

print(f"✓ Template: {shape[0]} × {shape[1]} pixels")
print(f"✓ CRS: {crs}")
print()

# =============================================================================
# PART 2: LOAD GUILD OCCUPANCY MAPS
# =============================================================================

print("=" * 80)
print("PART 2: LOADING GUILD OCCUPANCY MAPS")
print("=" * 80)
print()

psi_wetland = read_raster_match(PSI_WETLAND_TIF, meta)
psi_forest = read_raster_match(PSI_FOREST_TIF, meta)
psi_urban = read_raster_match(PSI_URBAN_TIF, meta)

if any(x is None for x in [psi_wetland, psi_forest, psi_urban]):
    print("❌ ERROR: Missing guild occupancy maps")
    print("   Run Phase 4 first to generate guild-specific predictions")
    exit(1)

print("✓ All guild maps loaded")
print(f"  Wetland mean ψ: {np.nanmean(psi_wetland):.3f}")
print(f"  Forest mean ψ: {np.nanmean(psi_forest):.3f}")
print(f"  Urban mean ψ: {np.nanmean(psi_urban):.3f}")
print()

# =============================================================================
# PART 3: CALCULATE MEAN OCCUPANCY (Equation 7)
# =============================================================================

print("=" * 80)
print("PART 3: MEAN OCCUPANCY")
print("=" * 80)
print()

print("Formula: ψ̄ = (ψ_Wetland + ψ_Forest + ψ_Urban) / 3")
print()

# Calculate mean occupancy
psi_mean = (psi_wetland + psi_forest + psi_urban) / 3

print(f"✓ Mean occupancy calculated")
print(f"  Mean: {np.nanmean(psi_mean):.3f}")
print(f"  Std: {np.nanstd(psi_mean):.3f}")
print(f"  Range: [{np.nanmin(psi_mean):.3f}, {np.nanmax(psi_mean):.3f}]")
print()

# Save
save_raster(psi_mean, OUT_DIR / "mean_occupancy.tif", meta)
print()

# =============================================================================
# PART 4: CALCULATE SHANNON ENTROPY (Equation 8)
# =============================================================================

print("=" * 80)
print("PART 4: SHANNON ENTROPY")
print("=" * 80)
print()

print("Formula: H = −Σ (p_g · log(p_g))")
print("  where p_g = ψ_g / Σψ")
print()

def calculate_shannon_entropy(psi_wetland, psi_forest, psi_urban):
    """
    Calculate Shannon entropy from guild occupancies.
    
    H = -Σ (p_i * log(p_i))
    where p_i = ψ_i / Σψ (normalized proportions)
    """
    # Stack guild probabilities
    psi_stack = np.stack([psi_wetland, psi_forest, psi_urban], axis=0)
    
    # Calculate total occupancy
    total_occ = np.sum(psi_stack, axis=0)
    
    # Avoid division by zero
    total_occ_safe = total_occ.copy()
    total_occ_safe[total_occ_safe == 0] = 1e-10
    
    # Normalize to proportions
    proportions = psi_stack / total_occ_safe
    
    # Calculate Shannon entropy: H = -Σ(p_i * log(p_i))
    entropy = np.zeros_like(total_occ)
    
    for i in range(3):  # Three guilds
        p = proportions[i]
        mask = p > 0
        entropy[mask] -= p[mask] * np.log(p[mask])
    
    # Set entropy to 0 where no guilds present
    entropy[total_occ == 0] = 0
    
    return entropy

# Calculate Shannon entropy
shannon_H = calculate_shannon_entropy(psi_wetland, psi_forest, psi_urban)

print(f"✓ Shannon entropy calculated")
print(f"  Mean: {np.nanmean(shannon_H):.3f}")
print(f"  Std: {np.nanstd(shannon_H):.3f}")
print(f"  Range: [{np.nanmin(shannon_H):.3f}, {np.nanmax(shannon_H):.3f}]")
print()
print("  Interpretation:")
print("    H ≈ 0: One guild dominates (low diversity)")
print("    H ≈ 1.1: All guilds equal (maximum diversity for 3 guilds)")
print()

# Save
save_raster(shannon_H, OUT_DIR / "shannon_entropy.tif", meta)
print()

# =============================================================================
# PART 5: CALCULATE THREAT SCORE (Equation 9)
# =============================================================================

print("=" * 80)
print("PART 5: THREAT EXPOSURE")
print("=" * 80)
print()

print("Formula: Threat = 0.4×(1−EdgeDist) + 0.3×VIIRS + 0.3×(1−NDVI)")
print()

# Load environmental rasters
ndvi = read_raster_match(NDVI_TIF, meta)
viirs = read_raster_match(VIIRS_TIF, meta)
dist_edge = read_raster_match(DIST_EDGE_TIF, meta)

# Normalize each component to 0-1
if ndvi is not None:
    ndvi_norm = normalize(ndvi)
else:
    print("  ⚠️ NDVI not found, using zeros")
    ndvi_norm = np.zeros(shape, dtype=np.float32)

if viirs is not None:
    viirs_norm = normalize(viirs)
else:
    print("  ⚠️ VIIRS not found, using zeros")
    viirs_norm = np.zeros(shape, dtype=np.float32)

if dist_edge is not None:
    edge_norm = normalize(dist_edge)
else:
    print("  ⚠️ Edge distance not found, using zeros")
    edge_norm = np.zeros(shape, dtype=np.float32)

# Calculate threat components
# Higher threat = closer to edge, higher light, lower vegetation
edge_threat = 1 - edge_norm  # Invert: close to edge = high threat
light_threat = viirs_norm    # High light = high threat
veg_threat = 1 - ndvi_norm   # Low vegetation = high threat

# Combined threat (weights from paper)
threat = (
    0.4 * edge_threat +
    0.3 * light_threat +
    0.3 * veg_threat
)

print(f"✓ Threat score calculated")
print(f"  Mean: {np.nanmean(threat):.3f}")
print(f"  Std: {np.nanstd(threat):.3f}")
print(f"  Range: [{np.nanmin(threat):.3f}, {np.nanmax(threat):.3f}]")
print()

# Save components
save_raster(edge_threat, OUT_DIR / "threat_edge.tif", meta)
save_raster(light_threat, OUT_DIR / "threat_light.tif", meta)
save_raster(veg_threat, OUT_DIR / "threat_vegetation.tif", meta)
save_raster(threat, OUT_DIR / "threat_composite.tif", meta)
print()

# =============================================================================
# PART 6: CALCULATE CPI (Equation 10)
# =============================================================================

print("=" * 80)
print("PART 6: CONSERVATION PRIORITY INDEX")
print("=" * 80)
print()

print("Formula: CPI = 0.5×ψ̄ + 0.3×H + 0.2×Threat")
print()
print(f"Weights:")
print(f"  Mean Occupancy (ψ̄): {WEIGHTS['occupancy']:.2f}")
print(f"  Shannon Entropy (H): {WEIGHTS['diversity']:.2f}")
print(f"  Threat Pressure: {WEIGHTS['threat']:.2f}")
print()

# Normalize all components to 0-1 for fair weighting
psi_mean_norm = normalize(psi_mean)
shannon_H_norm = normalize(shannon_H)
threat_norm = normalize(threat)

# Calculate CPI
# NOTE: Threat is ADDED not subtracted because higher threat = higher priority
# (needs urgent conservation attention)
CPI = (
    WEIGHTS['occupancy'] * psi_mean_norm +
    WEIGHTS['diversity'] * shannon_H_norm +
    WEIGHTS['threat'] * threat_norm
)

# Normalize final CPI to 0-1
CPI_normalized = normalize(CPI)

print(f"✓ CPI calculated")
print(f"  Mean: {np.nanmean(CPI_normalized):.3f}")
print(f"  Median: {np.nanmedian(CPI_normalized):.3f}")
print(f"  Std: {np.nanstd(CPI_normalized):.3f}")
print(f"  Range: [{np.nanmin(CPI_normalized):.3f}, {np.nanmax(CPI_normalized):.3f}]")
print()

# Save CPI
save_raster(CPI_normalized, OUT_DIR / "CPI_conservation_priority.tif", meta)
print()

# =============================================================================
# PART 7: PRIORITY CLASSIFICATION (Quantile-based)
# =============================================================================

print("=" * 80)
print("PART 7: PRIORITY LEVEL CLASSIFICATION")
print("=" * 80)
print()

print("Using empirical quantiles:")
print("  Level 5 (Critical): CPI > 90th percentile")
print("  Level 4 (High): 75th - 90th percentile")
print("  Level 3 (Medium): 50th - 75th percentile")
print("  Level 2 (Low): 25th - 50th percentile")
print("  Level 1 (Minimal): < 25th percentile")
print()

# Calculate quantiles
valid_cpi = CPI_normalized[np.isfinite(CPI_normalized)]
q90 = np.percentile(valid_cpi, 90)
q75 = np.percentile(valid_cpi, 75)
q50 = np.percentile(valid_cpi, 50)
q25 = np.percentile(valid_cpi, 25)

print(f"Quantile thresholds:")
print(f"  90th percentile: {q90:.3f}")
print(f"  75th percentile: {q75:.3f}")
print(f"  50th percentile: {q50:.3f}")
print(f"  25th percentile: {q25:.3f}")
print()

# Classify into priority levels
priority = np.full_like(CPI_normalized, np.nan, dtype="float32")

priority[CPI_normalized >= q90] = 5  # Critical
priority[(CPI_normalized >= q75) & (CPI_normalized < q90)] = 4  # High
priority[(CPI_normalized >= q50) & (CPI_normalized < q75)] = 3  # Medium
priority[(CPI_normalized >= q25) & (CPI_normalized < q50)] = 2  # Low
priority[CPI_normalized < q25] = 1  # Minimal

# Calculate areas (assuming 10m pixels as per paper)
pixel_area_ha = 0.01  # 10m × 10m = 100 m² = 0.01 ha

print("Priority class distribution:")
for level in [5, 4, 3, 2, 1]:
    count = np.sum(priority == level)
    area_ha = count * pixel_area_ha
    percent = (count / valid_cpi.size) * 100
    
    level_name = {5: "Critical", 4: "High", 3: "Medium", 2: "Low", 1: "Minimal"}[level]
    print(f"  Level {level} ({level_name:8s}): {count:,} pixels ({area_ha:.2f} ha, {percent:.1f}%)")

# Calculate urgent priority area (Levels 4+5)
urgent_count = np.sum((priority == 5) | (priority == 4))
urgent_area = urgent_count * pixel_area_ha
urgent_percent = (urgent_count / valid_cpi.size) * 100

print()
print(f"✓ URGENT PRIORITY (Critical + High): {urgent_area:.2f} ha ({urgent_percent:.1f}%)")
print()

# Save priority classification
save_raster(priority, OUT_DIR / "priority_classes.tif", meta)
print()

# =============================================================================
# PART 8: SAVE METADATA
# =============================================================================

print("=" * 80)
print("PART 8: SAVING METADATA")
print("=" * 80)
print()

metadata = pd.DataFrame({
    'Parameter': [
        'Weight_Occupancy',
        'Weight_Diversity',
        'Weight_Threat',
        'Mean_CPI',
        'Median_CPI',
        'Q90_threshold',
        'Q75_threshold',
        'Q50_threshold',
        'Q25_threshold',
        'Pixels_Critical',
        'Pixels_High',
        'Pixels_Medium',
        'Pixels_Low',
        'Pixels_Minimal',
        'Area_Critical_ha',
        'Area_High_ha',
        'Area_Medium_ha',
        'Area_Urgent_ha',
        'Percent_Urgent'
    ],
    'Value': [
        WEIGHTS['occupancy'],
        WEIGHTS['diversity'],
        WEIGHTS['threat'],
        float(np.nanmean(CPI_normalized)),
        float(np.nanmedian(CPI_normalized)),
        float(q90),
        float(q75),
        float(q50),
        float(q25),
        int(np.sum(priority == 5)),
        int(np.sum(priority == 4)),
        int(np.sum(priority == 3)),
        int(np.sum(priority == 2)),
        int(np.sum(priority == 1)),
        float(np.sum(priority == 5) * pixel_area_ha),
        float(np.sum(priority == 4) * pixel_area_ha),
        float(np.sum(priority == 3) * pixel_area_ha),
        float(urgent_area),
        float(urgent_percent)
    ]
})

metadata_file = OUT_DIR / "CPI_metadata.csv"
metadata.to_csv(metadata_file, index=False)
print(f"✓ Saved: {metadata_file.name}")
print()

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 80)
print("PHASE 6 COMPLETE")
print("=" * 80)
print()

print("Generated files:")
print("  ✓ mean_occupancy.tif - Mean ψ across guilds")
print("  ✓ shannon_entropy.tif - Functional diversity (H)")
print("  ✓ threat_composite.tif - Combined threat score")
print("  ✓ CPI_conservation_priority.tif - Priority index (0-1)")
print("  ✓ priority_classes.tif - 5 discrete levels")
print("  ✓ CPI_metadata.csv - Summary statistics")
print()

print("Key Results:")
print(f"  Mean Occupancy: {np.nanmean(psi_mean):.3f}")
print(f"  Shannon Entropy: {np.nanmean(shannon_H):.3f}")
print(f"  Threat Score: {np.nanmean(threat):.3f}")
print(f"  Final CPI: {np.nanmean(CPI_normalized):.3f}")
print()
print(f"  URGENT CONSERVATION PRIORITY: {urgent_area:.2f} ha ({urgent_percent:.1f}%)")
print()

print("Formula Used (Paper-Compliant):")
print("  CPI = 0.5×ψ̄ + 0.3×H + 0.2×Threat")
print("  where:")
print("    ψ̄ = mean guild occupancy")
print("    H = Shannon entropy of guild diversity")
print("    Threat = 0.4×edge + 0.3×light + 0.3×vegetation_loss")
print()

print("=" * 80)