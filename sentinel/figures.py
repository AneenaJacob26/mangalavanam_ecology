#!/usr/bin/env python3
"""
Generate Figure 4: Conservation Priority Index Map
Creates publication-ready 300 DPI PNG
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
CPI_FILE = Path("data/processed/phase6/site_level_CPI.csv")
OUTPUT = Path("figures")
OUTPUT.mkdir(exist_ok=True)

# Load CPI data
cpi_df = pd.read_csv(CPI_FILE)

# Define priority classes based on your actual quantiles
# From your Phase 6 output: Critical >80th, High 60-80th, etc.
quantiles = cpi_df['CPI'].quantile([0.2, 0.4, 0.6, 0.8, 1.0])

priority_classes = [
    {'name': 'Minimal', 'color': '#f7f7f7', 'range': (0, quantiles[0.2])},
    {'name': 'Low', 'color': '#d9f0d3', 'range': (quantiles[0.2], quantiles[0.4])},
    {'name': 'Medium', 'color': '#ffffbf', 'range': (quantiles[0.4], quantiles[0.6])},
    {'name': 'High', 'color': '#fc8d59', 'range': (quantiles[0.6], quantiles[0.8])},
    {'name': 'Critical', 'color': '#d73027', 'range': (quantiles[0.8], 1.0)}
]

# Assign priority class to each site
def assign_priority(cpi_value):
    for i, pc in enumerate(priority_classes):
        if pc['range'][0] <= cpi_value <= pc['range'][1]:
            return i, pc['name'], pc['color']
    return 2, 'Medium', '#ffffbf'  # Default

cpi_df['priority_idx'], cpi_df['priority_name'], cpi_df['priority_color'] = \
    zip(*cpi_df['CPI'].map(assign_priority))

# Create figure
fig, ax = plt.subplots(figsize=(12, 10))

# Scatter plot - assuming you have x, y coordinates
# If not, create a simple grid
if 'x' in cpi_df.columns and 'y' in cpi_df.columns:
    x, y = cpi_df['x'], cpi_df['y']
else:
    # Create simple grid layout
    n_sites = len(cpi_df)
    cols = int(np.ceil(np.sqrt(n_sites)))
    cpi_df['x'] = cpi_df.index % cols
    cpi_df['y'] = cpi_df.index // cols
    x, y = cpi_df['x'], cpi_df['y']

# Plot each priority class
for pc in priority_classes:
    subset = cpi_df[cpi_df['priority_name'] == pc['name']]
    if len(subset) > 0:
        ax.scatter(subset['x'], subset['y'], 
                  c=pc['color'], 
                  s=500,  # Large marker size
                  edgecolors='black',
                  linewidths=1,
                  label=f"{pc['name']} ({len(subset)} sites, {len(subset)/len(cpi_df)*100:.1f}%)",
                  alpha=0.9)

# Formatting
ax.set_title('Conservation Priority Index (CPI) - Spatial Distribution', 
            fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Relative X Position', fontsize=12)
ax.set_ylabel('Relative Y Position', fontsize=12)
ax.legend(loc='best', fontsize=11, framealpha=0.95, title='Priority Class')
ax.grid(True, alpha=0.3, linestyle='--')

# Add statistics box
stats_text = f"Total Sites: {len(cpi_df)}\n"
stats_text += f"Mean CPI: {cpi_df['CPI'].mean():.3f}\n"
stats_text += f"Urgent Priority: {len(cpi_df[cpi_df['priority_idx'] >= 3])} sites "
stats_text += f"({len(cpi_df[cpi_df['priority_idx'] >= 3])/len(cpi_df)*100:.1f}%)"

ax.text(0.02, 0.98, stats_text, 
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()

# Save
output_file = OUTPUT / "Figure4_Conservation_Priority_Index.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Saved: {output_file}")
print(f"\nPriority Distribution:")
for pc in priority_classes:
    count = len(cpi_df[cpi_df['priority_name'] == pc['name']])
    print(f"  {pc['name']:10s}: {count:2d} sites ({count/len(cpi_df)*100:5.1f}%)")

plt.show()