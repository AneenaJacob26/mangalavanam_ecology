#!/usr/bin/env python3
"""
Mangalavanam Adaptive Sentinel
Phase 1d: Spatial Proxies

Purpose:
- Generate time-invariant landscape structure layers
- DIST_EDGE: Distance to core sanctuary edge (fragmentation proxy)
- DIST_DRAINAGE: Distance to waterways (pollution/disturbance proxy)

Author: Senior Geospatial Data Scientist
Date: January 2026
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt, binary_erosion, binary_dilation
import osmnx as ox
from pathlib import Path
from shapely.geometry import box
import warnings
warnings.filterwarnings('ignore')


def load_master_grid(master_path):
    """
    Load spatial reference parameters from master grid.
    
    Args:
        master_path: Path to NDVI master grid raster
        
    Returns:
        dict: Grid parameters (crs, transform, width, height, bounds, resolution)
    """
    with rasterio.open(master_path) as src:
        return {
            'crs': src.crs,
            'transform': src.transform,
            'width': src.width,
            'height': src.height,
            'bounds': src.bounds,
            'resolution': src.transform.a  # Pixel size in meters
        }


def create_core_mask(core_geojson, grid_params):
    """
    Create binary raster mask from core sanctuary polygon.
    
    Args:
        core_geojson: Path to core sanctuary GeoJSON
        grid_params: Master grid parameters
        
    Returns:
        np.ndarray: Binary mask (1=core, 0=outside)
    """
    # Load core polygon
    core_gdf = gpd.read_file(core_geojson)
    
    # Reproject to master grid CRS
    core_gdf = core_gdf.to_crs(grid_params['crs'])
    
    # Rasterize polygon
    shapes = [(geom, 1) for geom in core_gdf.geometry]
    
    mask = rasterize(
        shapes,
        out_shape=(grid_params['height'], grid_params['width']),
        transform=grid_params['transform'],
        fill=0,
        dtype=np.uint8
    )
    
    return mask


def compute_distance_to_edge(core_mask, resolution):
    """
    Compute Euclidean distance to core sanctuary edge.
    
    Distance increases both inside and outside the core boundary,
    representing edge effects and fragmentation pressure.
    
    Args:
        core_mask: Binary mask (1=core, 0=outside)
        resolution: Pixel size in meters
        
    Returns:
        np.ndarray: Distance to edge in meters (Float32)
    """
    # Find edge by XOR of original and eroded/dilated versions
    eroded = binary_erosion(core_mask)
    dilated = binary_dilation(core_mask)
    
    # Edge pixels are where core differs from erosion OR dilation
    edge_mask = (core_mask != eroded) | (core_mask != dilated)
    
    # Invert edge mask for distance transform (0=edge, 1=calculate distance)
    edge_inverted = ~edge_mask
    
    # Compute Euclidean distance from every pixel to nearest edge
    distance_pixels = distance_transform_edt(edge_inverted)
    
    # Convert from pixels to meters
    distance_meters = distance_pixels * resolution
    
    return distance_meters.astype(np.float32)


def download_waterways(bounds, target_crs):
    """
    Download waterway features from OpenStreetMap.
    
    Args:
        bounds: rasterio.coords.BoundingBox
        target_crs: Target CRS for reprojection
        
    Returns:
        gpd.GeoDataFrame: Waterway features in target CRS
    """
    # Create bounding box polygon for query
    bbox_polygon = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    
    # Create GeoDataFrame with proper CRS (master grid CRS)
    bbox_gdf = gpd.GeoDataFrame(
        {'geometry': [bbox_polygon]}, 
        crs=target_crs
    )
    
    # Convert to WGS84 for OSM query
    bbox_wgs84 = bbox_gdf.to_crs('EPSG:4326')
    
    # Get bounds in WGS84
    minx, miny, maxx, maxy = bbox_wgs84.total_bounds
    
    print(f"  → Querying OSM within bounds: ({miny:.4f}, {minx:.4f}) to ({maxy:.4f}, {maxx:.4f})")
    
    # Download waterways using osmnx
    tags = {'waterway': ['canal', 'drain', 'river']}
    
    try:
        # Use geometries_from_bbox for waterway features
        waterways_gdf = ox.features_from_bbox(
            north=maxy,
            south=miny,
            east=maxx,
            west=minx,
            tags=tags
        )
        
        # Filter to LineString and MultiLineString geometries only
        waterways_gdf = waterways_gdf[
            waterways_gdf.geometry.type.isin(['LineString', 'MultiLineString'])
        ]
        
        # Reproject to target CRS
        waterways_gdf = waterways_gdf.to_crs(target_crs)
        
        return waterways_gdf
        
    except Exception as e:
        print(f"  ⚠ Warning: OSM download failed: {e}")
        print(f"  → Returning empty GeoDataFrame")
        # Return empty GeoDataFrame with correct CRS
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)


def rasterize_waterways(waterways_gdf, grid_params):
    """
    Rasterize waterway features to master grid.
    
    Args:
        waterways_gdf: GeoDataFrame of waterway features
        grid_params: Master grid parameters
        
    Returns:
        np.ndarray: Binary mask (1=waterway, 0=land)
    """
    if len(waterways_gdf) == 0:
        # Return empty mask if no waterways
        return np.zeros(
            (grid_params['height'], grid_params['width']), 
            dtype=np.uint8
        )
    
    # Create shapes for rasterization
    shapes = [(geom, 1) for geom in waterways_gdf.geometry]
    
    # Rasterize
    waterway_mask = rasterize(
        shapes,
        out_shape=(grid_params['height'], grid_params['width']),
        transform=grid_params['transform'],
        fill=0,
        dtype=np.uint8
    )
    
    return waterway_mask


def compute_distance_to_drainage(waterway_mask, resolution):
    """
    Compute Euclidean distance to nearest waterway.
    
    Args:
        waterway_mask: Binary mask (1=waterway, 0=land)
        resolution: Pixel size in meters
        
    Returns:
        np.ndarray: Distance to drainage in meters (Float32)
    """
    # Invert mask for distance transform (0=waterway, 1=calculate distance)
    waterway_inverted = ~waterway_mask.astype(bool)
    
    # Compute Euclidean distance
    distance_pixels = distance_transform_edt(waterway_inverted)
    
    # Convert from pixels to meters
    distance_meters = distance_pixels * resolution
    
    return distance_meters.astype(np.float32)


def save_raster(data, output_path, grid_params, nodata=None):
    """
    Save array as GeoTiff with master grid alignment.
    
    Args:
        data: Numpy array to save
        output_path: Output file path
        grid_params: Master grid parameters
        nodata: NoData value (optional)
    """
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Write raster
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=grid_params['height'],
        width=grid_params['width'],
        count=1,
        dtype=data.dtype,
        crs=grid_params['crs'],
        transform=grid_params['transform'],
        nodata=nodata,
        compress='lzw'
    ) as dst:
        dst.write(data, 1)
    
    # Calculate and print statistics
    if nodata is not None:
        valid_data = data[data != nodata]
    else:
        valid_data = data
    
    print(f"  ✓ Saved: {output_path}")
    print(f"    - Min: {valid_data.min():.2f} m")
    print(f"    - Max: {valid_data.max():.2f} m")
    print(f"    - Mean: {valid_data.mean():.2f} m")
    print(f"    - Std: {valid_data.std():.2f} m")


def main():
    """Main processing pipeline for Phase 1d."""
    
    print("=" * 70)
    print("PHASE 1d — SPATIAL PROXIES")
    print("Fragmentation & Pollution Pathways")
    print("=" * 70)
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    MASTER_GRID = 'data/raw/rasters/NDVI_2025_core.tif'
    CORE_GEOJSON = 'data/raw/shapes/Mangalavanam_Core.geojson'
    
    OUTPUT_DIST_EDGE = 'data/raw/rasters/DIST_EDGE.tif'
    OUTPUT_DIST_DRAINAGE = 'data/raw/rasters/DIST_DRAINAGE.tif'
    
    # ========================================================================
    # STEP 1: Load master grid parameters
    # ========================================================================
    print("\n[1/4] Loading master grid reference...")
    try:
        grid_params = load_master_grid(MASTER_GRID)
        print(f"  ✓ CRS: {grid_params['crs']}")
        print(f"  ✓ Dimensions: {grid_params['width']} × {grid_params['height']}")
        print(f"  ✓ Resolution: {grid_params['resolution']:.1f} m")
        print(f"  ✓ Bounds: {grid_params['bounds']}")
    except Exception as e:
        print(f"  ✗ Error loading master grid: {e}")
        return 1
    
    # ========================================================================
    # STEP 2: Generate Distance-to-Edge (Fragmentation Proxy)
    # ========================================================================
    print("\n[2/4] Generating Distance-to-Edge layer...")
    try:
        # Create core sanctuary mask
        print("  → Rasterizing core sanctuary boundary...")
        core_mask = create_core_mask(CORE_GEOJSON, grid_params)
        core_pixels = core_mask.sum()
        print(f"  ✓ Core sanctuary: {core_pixels:,} pixels "
              f"({core_pixels * grid_params['resolution']**2 / 10000:.2f} ha)")
        
        # Compute distance to edge
        print("  → Computing Euclidean distance to edge...")
        dist_edge = compute_distance_to_edge(core_mask, grid_params['resolution'])
        
        # Save raster
        print("  → Saving raster...")
        save_raster(dist_edge, OUTPUT_DIST_EDGE, grid_params)
        
    except Exception as e:
        print(f"  ✗ Error generating distance-to-edge: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ========================================================================
    # STEP 3: Generate Distance-to-Drainage (Pollution Proxy)
    # ========================================================================
    print("\n[3/4] Generating Distance-to-Drainage layer...")
    try:
        # Download waterways from OSM
        print("  → Downloading waterways from OpenStreetMap...")
        waterways_gdf = download_waterways(
            grid_params['bounds'], 
            grid_params['crs']
        )
        
        if len(waterways_gdf) > 0:
            print(f"  ✓ Downloaded {len(waterways_gdf)} waterway features")
            
            # Breakdown by type
            if 'waterway' in waterways_gdf.columns:
                for wtype, count in waterways_gdf['waterway'].value_counts().items():
                    print(f"    - {wtype}: {count}")
        else:
            print(f"  ⚠ No waterways found in study area")
        
        # Rasterize waterways
        print("  → Rasterizing waterway features...")
        waterway_mask = rasterize_waterways(waterways_gdf, grid_params)
        waterway_pixels = waterway_mask.sum()
        print(f"  ✓ Waterway pixels: {waterway_pixels:,}")
        
        # Compute distance to drainage
        print("  → Computing Euclidean distance to drainage...")
        dist_drainage = compute_distance_to_drainage(
            waterway_mask, 
            grid_params['resolution']
        )
        
        # Save raster
        print("  → Saving raster...")
        save_raster(dist_drainage, OUTPUT_DIST_DRAINAGE, grid_params)
        
    except Exception as e:
        print(f"  ✗ Error generating distance-to-drainage: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ========================================================================
    # STEP 4: Validation
    # ========================================================================
    print("\n[4/4] Validating outputs...")
    
    validation_passed = True
    
    for output_path in [OUTPUT_DIST_EDGE, OUTPUT_DIST_DRAINAGE]:
        try:
            with rasterio.open(output_path) as src:
                # Check CRS
                if src.crs != grid_params['crs']:
                    print(f"  ✗ CRS mismatch in {output_path}")
                    validation_passed = False
                
                # Check dimensions
                if src.width != grid_params['width'] or src.height != grid_params['height']:
                    print(f"  ✗ Dimension mismatch in {output_path}")
                    validation_passed = False
                
                # Check transform
                if src.transform != grid_params['transform']:
                    print(f"  ✗ Transform mismatch in {output_path}")
                    validation_passed = False
                
                # Check data type
                if src.dtypes[0] != 'float32':
                    print(f"  ✗ Data type should be float32 in {output_path}")
                    validation_passed = False
                    
        except Exception as e:
            print(f"  ✗ Error validating {output_path}: {e}")
            validation_passed = False
    
    if validation_passed:
        print("  ✓ All outputs perfectly aligned to master grid")
    else:
        print("  ✗ Validation failed - check errors above")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1d COMPLETE")
    print("=" * 70)
    print(f"Spatial proxies generated:")
    print(f"  1. {OUTPUT_DIST_EDGE}")
    print(f"     → Fragmentation proxy (distance to sanctuary edge)")
    print(f"  2. {OUTPUT_DIST_DRAINAGE}")
    print(f"     → Pollution pathway proxy (distance to waterways)")
    print(f"\nGrid specifications:")
    print(f"  CRS: {grid_params['crs']}")
    print(f"  Resolution: {grid_params['resolution']:.1f} m × {grid_params['resolution']:.1f} m")
    print(f"  Dimensions: {grid_params['width']} × {grid_params['height']}")
    print(f"  Data type: Float32")
    print(f"\nThese layers are time-invariant and represent:")
    print(f"  - Landscape structure (edge effects, core-periphery gradients)")
    print(f"  - Pollution/disturbance pathways (hydrological connectivity)")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())