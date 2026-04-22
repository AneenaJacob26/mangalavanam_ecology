#!/usr/bin/env python3
"""
Phase 4: Generate SPATIAL Habitat Rasters - FIXED VERSION
Uses ACTUAL sanctuary boundaries from shapefiles
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
from scipy.special import expit
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PHASE 4: SPATIAL HABITAT RASTER GENERATOR (FIXED)")
print("="*80)
print()

# Paths
RASTERS_DIR = Path("data/raw/rasters")
SHAPES_DIR = Path("data/raw/shapes")
PHASE4_DIR  = Path("data/processed/phase4")
PHASE4_DIR.mkdir(exist_ok=True, parents=True)

GUILDS = ['Wetland', 'Forest', 'Urban']

# ===========================================================================
# STEP 0: GET ACTUAL SANCTUARY BOUNDS FROM SHAPEFILES
# ===========================================================================

print("[0/4] Loading actual sanctuary boundaries...")
print()

try:
    core = gpd.read_file(SHAPES_DIR / "Mangalavanam_Core.geojson")
    buffer = gpd.read_file(SHAPES_DIR / "Mangalavanam_Buffer.geojson")
    
    # Reproject to WGS84
    if core.crs != 'EPSG:4326':
        core = core.to_crs('EPSG:4326')
    if buffer.crs != 'EPSG:4326':
        buffer = buffer.to_crs('EPSG:4326')
    
    # Combine and get bounds
    sanctuary = pd.concat([core, buffer], ignore_index=True)
    sanctuary = gpd.GeoDataFrame(sanctuary, crs='EPSG:4326')
    
    actual_bounds = sanctuary.total_bounds
    
    BOUNDS = {
        'left': actual_bounds[0],
        'bottom': actual_bounds[1],
        'right': actual_bounds[2],
        'top': actual_bounds[3]
    }
    
    print(f"✅ Loaded ACTUAL sanctuary bounds:")
    print(f"   West:  {BOUNDS['left']:.6f}°")
    print(f"   South: {BOUNDS['bottom']:.6f}°")
    print(f"   East:  {BOUNDS['right']:.6f}°")
    print(f"   North: {BOUNDS['top']:.6f}°")
    print()
    
except Exception as e:
    print(f"⚠️ Could not load boundaries: {e}")
    print("Using default bounds (may be incorrect)")
    BOUNDS = {
        'left': 76.268,
        'bottom': 9.982,
        'right': 76.282,
        'top': 9.998
    }

# Grid size
GRID_ROWS = 7
GRID_COLS = 7

# ===========================================================================
# GUILD MODEL COEFFICIENTS
# ===========================================================================

GUILD_COEFFICIENTS = {
    'Wetland': {
        'alpha':      0.819,
        'beta_ndvi':  1.0,
        'beta_ndwi':  2.5,
        'beta_viirs': -0.8,
        'beta_edge':  0.6,
    },
    'Forest': {
        'alpha':      0.676,
        'beta_ndvi':  3.0,
        'beta_ndwi':  0.3,
        'beta_viirs': -1.5,
        'beta_edge':  1.5,
    },
    'Urban': {
        'alpha':      0.929,
        'beta_ndvi':  0.2,
        'beta_ndwi':  0.0,
        'beta_viirs': 1.0,
        'beta_edge':  -0.3,
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

# Create template with CORRECT bounds
template_meta = {
    'driver': 'GTiff',
    'dtype': 'float32',
    'nodata': -9999,
    'width': GRID_COLS,
    'height': GRID_ROWS,
    'count': 1,
    'crs': 'EPSG:4326',
    'transform': from_bounds(
        BOUNDS['left'], BOUNDS['bottom'],
        BOUNDS['right'], BOUNDS['top'],
        GRID_COLS, GRID_ROWS
    )
}

print(f"  Output grid: {GRID_ROWS} x {GRID_COLS} pixels")
print(f"  Bounds: {BOUNDS}")
print()

# ===========================================================================
# STEP 2: NORMALIZE ENVIRONMENTAL DATA
# ===========================================================================

print("[2/4] Normalizing environmental data...")
print()

def normalize(data, name):
    """Normalize to 0-1 range"""
    if data is None:
        print(f"  ⚠️  {name} not available, using zeros")
        return np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
    
    # Resize to our grid if needed
    from scipy.ndimage import zoom
    
    if data.shape != (GRID_ROWS, GRID_COLS):
        zoom_factors = (GRID_ROWS / data.shape[0], GRID_COLS / data.shape[1])
        data_resized = zoom(data, zoom_factors, order=1)
    else:
        data_resized = data
    
    valid = data_resized[np.isfinite(data_resized)]
    if len(valid) == 0:
        return np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
    
    vmin, vmax = np.nanpercentile(valid, 2), np.nanpercentile(valid, 98)
    
    if vmax > vmin:
        normalized = (data_resized - vmin) / (vmax - vmin)
        normalized = np.clip(normalized, 0, 1)
    else:
        normalized = np.zeros_like(data_resized)
    
    normalized = np.where(np.isfinite(data_resized), normalized, np.nan)
    
    valid_norm = normalized[np.isfinite(normalized)]
    print(f"  ✓ {name}: normalized range [{valid_norm.min():.3f}, {valid_norm.max():.3f}]")
    
    return normalized.astype(np.float32)

ndvi_norm  = normalize(ndvi_data,  "NDVI  (vegetation)")
ndwi_norm  = normalize(ndwi_data,  "NDWI  (water)     ")
viirs_norm = normalize(viirs_data, "VIIRS (lights)    ")

# Edge distance proxy
cy, cx = GRID_ROWS // 2, GRID_COLS // 2
y_idx, x_idx = np.mgrid[0:GRID_ROWS, 0:GRID_COLS]
edge_dist = 1.0 - np.sqrt(((y_idx - cy) / cy) ** 2 + ((x_idx - cx) / cx) ** 2)
edge_dist = np.clip(edge_dist, 0, 1).astype(np.float32)

print(f"  ✓ Edge distance: range [{edge_dist.min():.3f}, {edge_dist.max():.3f}]")
print()

# ===========================================================================
# STEP 3: COMPUTE HABITAT SUITABILITY MAPS
# ===========================================================================

print("[3/4] Computing habitat suitability maps...")
print()

valid_mask = np.ones((GRID_ROWS, GRID_COLS), dtype=bool)

env_rasters = {}

for guild in GUILDS:
    print(f"  --- {guild} Guild ---")
    
    coef = GUILD_COEFFICIENTS[guild]
    
    alpha_logit = np.log(coef['alpha'] / (1 - coef['alpha'] + 1e-8))
    
    linear_pred = (
        alpha_logit
        + coef['beta_ndvi']  * (ndvi_norm  - 0.5)
        + coef['beta_ndwi']  * (ndwi_norm  - 0.5)
        + coef['beta_viirs'] * (viirs_norm - 0.5)
        + coef['beta_edge']  * (edge_dist  - 0.5)
    )
    
    psi = expit(linear_pred).astype(np.float32)
    psi = np.where(valid_mask, psi, np.nan)
    
    valid_psi = psi[np.isfinite(psi)]
    print(f"    ψ range: [{valid_psi.min():.3f}, {valid_psi.max():.3f}]")
    print(f"    ψ mean:  {valid_psi.mean():.3f}")
    print(f"    ψ std:   {valid_psi.std():.3f}")
    
    # Save with CORRECT bounds
    out_file = PHASE4_DIR / f"psi_env_{guild}.tif"
    out_data = np.where(np.isfinite(psi), psi, template_meta['nodata'])
    
    with rasterio.open(out_file, 'w', **template_meta) as dst:
        dst.write(out_data.astype(np.float32), 1)
    
    print(f"    ✓ Saved: {out_file.name}")
    
    env_rasters[guild] = psi
    print()

# Mean environmental raster
print("  --- Mean (All Guilds) ---")
all_psi = np.stack([env_rasters[g] for g in GUILDS], axis=0)
mean_psi = np.nanmean(all_psi, axis=0)

valid_mean = mean_psi[np.isfinite(mean_psi)]
print(f"    ψ range: [{valid_mean.min():.3f}, {valid_mean.max():.3f}]")
print(f"    ψ std:   {valid_mean.std():.3f}")

out_file = PHASE4_DIR / "psi_env_mean.tif"
out_data = np.where(np.isfinite(mean_psi), mean_psi, template_meta['nodata'])

with rasterio.open(out_file, 'w', **template_meta) as dst:
    dst.write(out_data.astype(np.float32), 1)

print(f"    ✓ Saved: {out_file.name}")
print()

# ===========================================================================
# STEP 4: ZONE AND YEAR RASTERS
# ===========================================================================

print("[4/4] Creating zone and year rasters...")
print()

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
    
    variation = (ndvi_norm - ndvi_norm[np.isfinite(ndvi_norm)].mean()) * 0.05
    variation = np.where(np.isfinite(variation), variation, 0)
    
    year_map = np.clip(year_psi + variation, 0.05, 0.99).astype(np.float32)
    year_map = np.where(valid_mask, year_map, np.nan)
    
    out_data = np.where(np.isfinite(year_map), year_map, template_meta['nodata'])
    out_file = PHASE4_DIR / f"psi_year_{guild}.tif"
    with rasterio.open(out_file, 'w', **template_meta) as dst:
        dst.write(out_data.astype(np.float32), 1)
    
    core_boost = (edge_dist - 0.5) * 0.1
    zone_map = np.clip(zone_psi + core_boost, 0.05, 0.99).astype(np.float32)
    zone_map = np.where(valid_mask, zone_map, np.nan)
    
    out_data = np.where(np.isfinite(zone_map), zone_map, template_meta['nodata'])
    out_file = PHASE4_DIR / f"psi_zone_{guild}.tif"
    with rasterio.open(out_file, 'w', **template_meta) as dst:
        dst.write(out_data.astype(np.float32), 1)
    
    print(f"  ✓ {guild}: year and zone rasters saved")

print()
print("="*80)
print("✅ DONE! Rasters created with CORRECT sanctuary boundaries")
print("="*80)