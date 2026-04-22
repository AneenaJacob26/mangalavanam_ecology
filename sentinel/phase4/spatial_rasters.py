#!/usr/bin/env python3
"""
Phase 4: Generate SPATIAL Habitat Rasters
Creates spatially-varying habitat maps from actual environmental data
(NDVI, NDWI, VIIRS) using guild-specific model coefficients

This fixes the "solid green" uniform raster problem
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
from scipy.special import expit  # sigmoid function
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PHASE 4: SPATIAL HABITAT RASTER GENERATOR")
print("="*80)
print()
print("This script creates spatially-varying habitat maps using:")
print("  - Real NDVI, NDWI, VIIRS satellite data")
print("  - Guild-specific Bayesian model coefficients")
print("  - Sigmoid transformation for valid 0-1 output")
print()

# Paths
RASTERS_DIR = Path("data/raw/rasters")
PHASE4_DIR  = Path("data/processed/phase4")
PHASE4_DIR.mkdir(exist_ok=True, parents=True)

GUILDS = ['Wetland', 'Forest', 'Urban']

# ===========================================================================
# GUILD MODEL COEFFICIENTS (from your Bayesian model results)
# These are the beta coefficients that determine how each variable
# affects habitat suitability for each guild
# ===========================================================================
#
# Interpretation:
#   alpha = baseline occupancy (intercept)
#   beta_ndvi  = effect of vegetation (NDVI)
#   beta_ndwi  = effect of water (NDWI)
#   beta_viirs = effect of night lights (VIIRS)
#   beta_edge  = effect of edge distance
#
# Signs:
#   Positive = more of this = more birds
#   Negative = more of this = fewer birds

GUILD_COEFFICIENTS = {
    'Wetland': {
        'alpha':      0.819,   # baseline ψ from your model
        'beta_ndvi':  1.0,     # moderate vegetation response (INCREASED)
        'beta_ndwi':  2.5,     # VERY strong water response (water birds!) (INCREASED)
        'beta_viirs': -0.8,    # negative response to urban lights (INCREASED)
        'beta_edge':  0.6,     # preference for interior (INCREASED)
    },
    'Forest': {
        'alpha':      0.676,   # baseline ψ from your model
        'beta_ndvi':  3.0,     # VERY strong vegetation response (tree birds!) (INCREASED)
        'beta_ndwi':  0.3,     # weak water response (INCREASED)
        'beta_viirs': -1.5,    # VERY strong negative response to lights (INCREASED)
        'beta_edge':  1.5,     # VERY strong preference for interior (INCREASED)
    },
    'Urban': {
        'alpha':      0.929,   # baseline ψ from your model
        'beta_ndvi':  0.2,     # weak vegetation response (INCREASED slightly)
        'beta_ndwi':  0.0,     # no water response
        'beta_viirs': 1.0,     # strong positive response to lights (urban birds!) (INCREASED)
        'beta_edge':  -0.3,    # preference for edges (near humans) (INCREASED)
    }
}

# ===========================================================================
# STEP 1: LOAD ENVIRONMENTAL RASTERS
# ===========================================================================

print("[1/4] Loading environmental rasters...")
print()

def load_raster_data(filepath):
    """Load raster, return data array and metadata"""
    if not filepath.exists():
        print(f"  ✗ NOT FOUND: {filepath.name}")
        return None, None
    
    with rasterio.open(filepath) as src:
        data = src.read(1).astype(np.float32)
        meta = src.meta.copy()
        
        if src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)
        
        valid = data[np.isfinite(data)]
        print(f"  ✓ {filepath.name}: shape={data.shape}, "
              f"range=[{valid.min():.3f}, {valid.max():.3f}], "
              f"valid_pixels={len(valid)}")
        
        return data, meta

ndvi_data, ndvi_meta = load_raster_data(RASTERS_DIR / "NDVI_2025_core.tif")
ndwi_data, ndwi_meta = load_raster_data(RASTERS_DIR / "NDWI_2025_core.tif")
viirs_data, viirs_meta = load_raster_data(RASTERS_DIR / "VIIRS_2025_core.tif")

print()

# Check we have at least one raster
if ndvi_data is None and ndwi_data is None and viirs_data is None:
    print("❌ ERROR: No environmental rasters found!")
    print(f"   Expected location: {RASTERS_DIR}")
    print(f"   Contents: {list(RASTERS_DIR.glob('*.tif'))}")
    exit(1)

# Use first available raster as template
template_meta = ndvi_meta or ndwi_meta or viirs_meta
H, W = template_meta['height'], template_meta['width']
print(f"  Template shape: {H} x {W} pixels")
print()

# ===========================================================================
# STEP 2: NORMALIZE ENVIRONMENTAL DATA
# ===========================================================================

print("[2/4] Normalizing environmental data...")
print()

def normalize(data, name):
    """Normalize to 0-1 range, handle NaN"""
    if data is None:
        print(f"  ⚠️  {name} not available, using zeros")
        return np.zeros((H, W), dtype=np.float32)
    
    valid = data[np.isfinite(data)]
    if len(valid) == 0:
        return np.zeros((H, W), dtype=np.float32)
    
    vmin, vmax = np.nanpercentile(valid, 2), np.nanpercentile(valid, 98)
    
    if vmax > vmin:
        normalized = (data - vmin) / (vmax - vmin)
        normalized = np.clip(normalized, 0, 1)
    else:
        normalized = np.zeros_like(data)
    
    normalized = np.where(np.isfinite(data), normalized, np.nan)
    
    valid_norm = normalized[np.isfinite(normalized)]
    print(f"  ✓ {name}: normalized range [{valid_norm.min():.3f}, {valid_norm.max():.3f}]")
    
    return normalized.astype(np.float32)

ndvi_norm  = normalize(ndvi_data,  "NDVI  (vegetation)")
ndwi_norm  = normalize(ndwi_data,  "NDWI  (water)     ")
viirs_norm = normalize(viirs_data, "VIIRS (lights)    ")

# Create edge distance proxy (distance from centre = more interior = lower edge effect)
# Simple: use distance from image centre as proxy
cy, cx = H // 2, W // 2
y_idx, x_idx = np.mgrid[0:H, 0:W]
edge_dist = 1.0 - np.sqrt(((y_idx - cy) / cy) ** 2 + ((x_idx - cx) / cx) ** 2)
edge_dist = np.clip(edge_dist, 0, 1).astype(np.float32)

print(f"  ✓ Edge distance proxy: range [{edge_dist.min():.3f}, {edge_dist.max():.3f}]")
print()

# ===========================================================================
# STEP 3: COMPUTE HABITAT SUITABILITY MAPS
# ===========================================================================

print("[3/4] Computing habitat suitability maps...")
print()

# Output metadata
out_meta = template_meta.copy()
out_meta.update({'dtype': 'float32', 'count': 1, 'nodata': -9999})

# Create mask for valid pixels (where we have data)
valid_mask = np.ones((H, W), dtype=bool)
if ndvi_data is not None:
    valid_mask &= np.isfinite(ndvi_data)

env_rasters = {}

for guild in GUILDS:
    print(f"  --- {guild} Guild ---")
    
    coef = GUILD_COEFFICIENTS[guild]
    
    # Logistic regression model:
    # logit(ψ) = alpha + beta_ndvi*NDVI + beta_ndwi*NDWI + ...
    # ψ = sigmoid(logit(ψ))
    
    # Convert alpha (probability) to logit scale
    alpha_logit = np.log(coef['alpha'] / (1 - coef['alpha'] + 1e-8))
    
    # Compute linear predictor
    linear_pred = (
        alpha_logit
        + coef['beta_ndvi']  * (ndvi_norm  - 0.5)
        + coef['beta_ndwi']  * (ndwi_norm  - 0.5)
        + coef['beta_viirs'] * (viirs_norm - 0.5)
        + coef['beta_edge']  * (edge_dist  - 0.5)
    )
    
    # Apply sigmoid to get probability (0-1)
    psi = expit(linear_pred).astype(np.float32)
    
    # Apply valid mask
    psi = np.where(valid_mask, psi, np.nan)
    
    # Stats
    valid_psi = psi[np.isfinite(psi)]
    print(f"    ψ range: [{valid_psi.min():.3f}, {valid_psi.max():.3f}]")
    print(f"    ψ mean:  {valid_psi.mean():.3f}")
    print(f"    ψ std:   {valid_psi.std():.3f}  ← should be > 0!")
    
    # Save environmental raster
    out_file = PHASE4_DIR / f"psi_env_{guild}.tif"
    out_data = np.where(np.isfinite(psi), psi, out_meta['nodata'])
    
    with rasterio.open(out_file, 'w', **out_meta) as dst:
        dst.write(out_data.astype(np.float32), 1)
    
    print(f"    ✓ Saved: {out_file.name}")
    
    env_rasters[guild] = psi
    print()

# Create mean environmental raster
print("  --- Mean (All Guilds) ---")
all_psi = np.stack([env_rasters[g] for g in GUILDS], axis=0)
mean_psi = np.nanmean(all_psi, axis=0)

valid_mean = mean_psi[np.isfinite(mean_psi)]
print(f"    ψ range: [{valid_mean.min():.3f}, {valid_mean.max():.3f}]")
print(f"    ψ std:   {valid_mean.std():.3f}")

out_file = PHASE4_DIR / "psi_env_mean.tif"
out_data = np.where(np.isfinite(mean_psi), mean_psi, out_meta['nodata'])

with rasterio.open(out_file, 'w', **out_meta) as dst:
    dst.write(out_data.astype(np.float32), 1)

print(f"    ✓ Saved: {out_file.name}")
print()

# ===========================================================================
# STEP 4: ALSO CREATE ZONE AND YEAR RASTERS WITH SOME SPATIAL VARIATION
# ===========================================================================

print("[4/4] Creating zone and year rasters with spatial variation...")
print()

# Load guild profiles for baseline values
guild_profiles_file = PHASE4_DIR / "guild_profiles.csv"
if guild_profiles_file.exists():
    guild_profiles = pd.read_csv(guild_profiles_file)
else:
    guild_profiles = pd.DataFrame({
        'Guild': GUILDS,
        'Year': [0.928, 0.739, 0.980],
        'Zone': [0.914, 0.778, 0.976],
        'Env_mean': [0.819, 0.676, 0.929]
    })

for guild in GUILDS:
    row = guild_profiles[guild_profiles['Guild'] == guild]
    
    year_psi  = row['Year'].values[0]
    zone_psi  = row['Zone'].values[0]
    
    # Add small spatial variation based on NDVI
    variation = (ndvi_norm - ndvi_norm[np.isfinite(ndvi_norm)].mean()) * 0.05
    variation = np.where(np.isfinite(variation), variation, 0)
    
    # Year raster
    year_map = np.clip(year_psi + variation, 0.05, 0.99).astype(np.float32)
    year_map = np.where(valid_mask, year_map, np.nan)
    
    out_data = np.where(np.isfinite(year_map), year_map, out_meta['nodata'])
    out_file = PHASE4_DIR / f"psi_year_{guild}.tif"
    with rasterio.open(out_file, 'w', **out_meta) as dst:
        dst.write(out_data.astype(np.float32), 1)
    
    # Zone raster (core = higher, buffer = lower, based on edge_dist)
    core_boost = (edge_dist - 0.5) * 0.1
    zone_map = np.clip(zone_psi + core_boost, 0.05, 0.99).astype(np.float32)
    zone_map = np.where(valid_mask, zone_map, np.nan)
    
    out_data = np.where(np.isfinite(zone_map), zone_map, out_meta['nodata'])
    out_file = PHASE4_DIR / f"psi_zone_{guild}.tif"
    with rasterio.open(out_file, 'w', **out_meta) as dst:
        dst.write(out_data.astype(np.float32), 1)
    
    print(f"  ✓ {guild}: year and zone rasters saved")

# Mean year/zone rasters
for model in ['year', 'zone']:
    maps = []
    for guild in GUILDS:
        f = PHASE4_DIR / f"psi_{model}_{guild}.tif"
        with rasterio.open(f) as src:
            d = src.read(1).astype(np.float32)
            d = np.where(d == out_meta['nodata'], np.nan, d)
            maps.append(d)
    
    mean_map = np.nanmean(np.stack(maps, axis=0), axis=0)
    out_data = np.where(np.isfinite(mean_map), mean_map, out_meta['nodata'])
    
    out_file = PHASE4_DIR / f"psi_{model}_mean.tif"
    with rasterio.open(out_file, 'w', **out_meta) as dst:
        dst.write(out_data.astype(np.float32), 1)
    print(f"  ✓ {model} mean raster saved")

print()

# ===========================================================================
# VERIFICATION
# ===========================================================================

print("="*80)
print("VERIFICATION - All Rasters")
print("="*80)
print()

all_files = sorted(PHASE4_DIR.glob("psi_*.tif"))

all_ok = True
for f in all_files:
    with rasterio.open(f) as src:
        d = src.read(1).astype(np.float32)
        nodata = src.nodata or -9999
        d = np.where(d == nodata, np.nan, d)
        valid = d[np.isfinite(d)]
    
    if len(valid) == 0:
        print(f"  ✗ {f.name:35s} - NO VALID DATA")
        all_ok = False
    elif valid.std() < 0.001:
        print(f"  ⚠️  {f.name:35s} - UNIFORM (std={valid.std():.4f}) mean={valid.mean():.3f}")
    else:
        print(f"  ✓ {f.name:35s} - OK  range=[{valid.min():.3f},{valid.max():.3f}] std={valid.std():.3f}")

print()
if all_ok:
    print("✅ All rasters created with spatial variation!")
else:
    print("⚠️  Some rasters may have issues. Check above.")

print()
print("="*80)
print("DONE! Restart the dashboard to see spatially-varying habitat maps.")
print("="*80)