#!/usr/bin/env python3
"""
Create Priority Map with Exact Coordinates
Shows problem areas with lat/lon for field teams
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Configuration
BASE_DIR = Path("data")
PHASE6_DIR = BASE_DIR / "processed/phase6"
PHASE4_DIR = BASE_DIR / "processed/phase4"

CPI_CSV = PHASE6_DIR / "site_level_CPI.csv"
COORD_REF = PHASE4_DIR / "psi_maps_masked/pixel_coordinates_reference.csv"

OUTPUT_MAP = PHASE6_DIR / "priority_map_with_coordinates.png"
OUTPUT_URGENT = PHASE6_DIR / "urgent_action_sites_coordinates.csv"

def create_priority_map_with_coords():
    """Create priority map showing exact coordinates of problem areas"""
    
    print("🎯 Creating priority map with coordinates...")
    
    # Load data
    cpi_df = pd.read_csv(CPI_CSV)
    
    if COORD_REF.exists():
        coords_df = pd.read_csv(COORD_REF)
        print(f"✓ Loaded coordinate reference ({len(coords_df)} pixels)")
    else:
        print("⚠️ Coordinate reference not found - run create_masked_rasters.py first")
        coords_df = None
    
    # Classify priority
    quantiles = cpi_df['CPI'].quantile([0.2, 0.4, 0.6, 0.8])
    
    def classify_priority(cpi):
        if cpi <= quantiles[0.2]:
            return 1, 'Minimal', '#e0e0e0'
        elif cpi <= quantiles[0.4]:
            return 2, 'Low', '#689f38'
        elif cpi <= quantiles[0.6]:
            return 3, 'Medium', '#fbc02d'
        elif cpi <= quantiles[0.8]:
            return 4, 'High', '#f57c00'
        else:
            return 5, 'Critical', '#d32f2f'
    
    cpi_df[['Priority_Level', 'Priority_Name', 'Color']] = cpi_df['CPI'].apply(
        lambda x: pd.Series(classify_priority(x))
    )
    
    # Merge with coordinates if available
    if coords_df is not None:
        # Add pixel IDs to CPI data (assuming same order)
        cpi_df['Pixel_Row'] = cpi_df.index // 7
        cpi_df['Pixel_Col'] = cpi_df.index % 7
        cpi_df['Pixel_ID'] = cpi_df.apply(lambda r: f"R{int(r['Pixel_Row'])}C{int(r['Pixel_Col'])}", axis=1)
        
        # Merge
        cpi_with_coords = cpi_df.merge(
            coords_df[['Pixel_ID', 'Latitude', 'Longitude', 'Inside_Sanctuary', 'Google_Maps_Link']],
            on='Pixel_ID',
            how='left'
        )
        
        # Filter to only inside sanctuary
        cpi_inside = cpi_with_coords[cpi_with_coords['Inside_Sanctuary'] == True].copy()
        
        print(f"✓ {len(cpi_inside)} pixels inside sanctuary")
    else:
        cpi_inside = cpi_df
    
    # Identify urgent action sites (Critical + High)
    urgent = cpi_inside[cpi_inside['Priority_Level'] >= 4].copy()
    urgent = urgent.sort_values('CPI', ascending=False)
    
    print(f"\n🚨 URGENT ACTION SITES: {len(urgent)}")
    print("="*80)
    
    for i, row in urgent.iterrows():
        if coords_df is not None:
            print(f"\n📍 Site {row['Pixel_ID']} - Priority {row['Priority_Name']}")
            print(f"   Location: {row['Latitude']:.6f}°N, {row['Longitude']:.6f}°E")
            print(f"   CPI Score: {row['CPI']:.3f}")
            print(f"   Google Maps: {row['Google_Maps_Link']}")
        else:
            print(f"\n📍 Site {i} - Priority {row['Priority_Name']}")
            print(f"   CPI Score: {row['CPI']:.3f}")
    
    # Save urgent sites to CSV
    if coords_df is not None:
        # Check what columns we actually have
        available_cols = urgent.columns.tolist()
        print(f"\n  Available columns: {available_cols[:10]}...")  # Debug

        # Export only columns that exist
        export_cols = ['Pixel_ID', 'Latitude', 'Longitude', 'CPI', 'Priority_Name', 'Google_Maps_Link']

        # Add optional columns if they exist
        for col in ['Mean_Occupancy', 'Shannon_Diversity', 'Threat_Score', 'mean_occupancy', 'shannon_diversity', 'threat_score']:
            if col in available_cols and col not in export_cols:
                export_cols.append(col)

        urgent_export = urgent[export_cols].copy()
        
        urgent_export.to_csv(OUTPUT_URGENT, index=False)
        print(f"\n✅ Saved urgent sites with coordinates: {OUTPUT_URGENT}")
    
    # Create visual map
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Reshape CPI to grid
    grid_size = 7
    cpi_grid = np.full((grid_size, grid_size), np.nan)
    color_grid = np.empty((grid_size, grid_size), dtype=object)
    
    for _, row in cpi_inside.iterrows():
        r = int(row['Pixel_Row'])
        c = int(row['Pixel_Col'])
        cpi_grid[r, c] = row['CPI']
        color_grid[r, c] = row['Color']
    
    # Plot grid
    for r in range(grid_size):
        for c in range(grid_size):
            if not np.isnan(cpi_grid[r, c]):
                rect = mpatches.Rectangle(
                    (c, grid_size - r - 1), 1, 1,
                    facecolor=color_grid[r, c],
                    edgecolor='black',
                    linewidth=1.5
                )
                ax.add_patch(rect)
                
                # Add pixel ID and CPI value
                pixel_id = f"R{r}C{c}"
                ax.text(
                    c + 0.5, grid_size - r - 0.3,
                    pixel_id,
                    ha='center', va='top',
                    fontsize=8, fontweight='bold'
                )
                ax.text(
                    c + 0.5, grid_size - r - 0.7,
                    f"{cpi_grid[r, c]:.2f}",
                    ha='center', va='bottom',
                    fontsize=7
                )
    
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    ax.set_title('Conservation Priority Index - Mangalavanam Bird Sanctuary\n(Showing only pixels inside sanctuary boundaries)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#d32f2f', label='Critical (Level 5) - Urgent action needed'),
        mpatches.Patch(facecolor='#f57c00', label='High (Level 4) - Priority intervention'),
        mpatches.Patch(facecolor='#fbc02d', label='Medium (Level 3) - Moderate concern'),
        mpatches.Patch(facecolor='#689f38', label='Low (Level 2) - Routine monitoring'),
        mpatches.Patch(facecolor='#e0e0e0', label='Minimal (Level 1) - Good condition')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
             fontsize=10, framealpha=0.9)
    
    # Add coordinate labels
    if coords_df is not None:
        # Add lat/lon labels on axes
        ax.set_xlabel('Longitude →', fontsize=11, fontweight='bold')
        ax.set_ylabel('Latitude →', fontsize=11, fontweight='bold')
    
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_MAP, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved priority map: {OUTPUT_MAP}")
    
    plt.show()

if __name__ == "__main__":
    create_priority_map_with_coords()