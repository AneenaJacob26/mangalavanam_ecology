#!/usr/bin/env python3
"""
Mangalavanam Bird Sanctuary - Conservation Monitoring Dashboard
Interactive data visualization for adaptive habitat management

Author: MSc Data Science Project
Version: 2.0
Date: March 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import rasterio
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
import base64

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Adaptive Urban Biodiversity Monitoring System for Mangalavanam Bird Sanctuary",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM STYLING
# =============================================================================

st.markdown("""
<style>
    /* Clean, professional theme */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Card containers */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
        border-left: 4px solid #4caf50;
    }
    
    /* Headers */
    h1 { color: #1b5e20; font-weight: 600; }
    h2 { color: #2e7d32; font-weight: 500; }
    h3 { color: #388e3c; font-weight: 500; }
    
    /* Info boxes - minimal */
    .stAlert {
        padding: 1rem;
        border-radius: 6px;
    }
    
    /* Tabs - clean design */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #e8f5e9;
        border-radius: 6px 6px 0 0;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4caf50;
        color: white;
    }
    
    /* Remove excessive padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 1rem;
        color: #666;
        font-size: 0.9rem;
        border-top: 1px solid #e0e0e0;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# PATHS & CONFIGURATION
# =============================================================================

class Config:
    """Centralized file paths and constants"""
    
    BASE_DIR = Path("data")
    RAW_DIR = BASE_DIR / "raw"
    PROCESSED_DIR = BASE_DIR / "processed"
    
    # Phase outputs
    PHASE1_DIR = PROCESSED_DIR / "phase1"
    PHASE4_DIR = PROCESSED_DIR / "phase4"
    PHASE6_DIR = PROCESSED_DIR / "phase6"
    PHASE7_DIR = PROCESSED_DIR / "phase7_trends"
    
    # Key files - UPDATED PATHS
    EBIRD_DATA = PHASE1_DIR / "ebird_with_guilds_and_zones.csv"
    GUILD_PROFILES = PHASE4_DIR / "guild_profiles.csv"
    FUNCTIONAL_SUMMARY = PHASE4_DIR / "functional_indices_summary.csv"
    
    # Masked rasters (CORRECT LOCATION)
    MASKED_DIR = PHASE4_DIR / "psi_maps_masked"
    PSI_ENV_MEAN = MASKED_DIR / "psi_env_mean_masked.tif"
    COORD_REFERENCE = MASKED_DIR / "pixel_coordinates_reference.csv"
    SANCTUARY_BOUNDARY = MASKED_DIR / "sanctuary_boundary.geojson"
    
    # Conservation priority
    CPI_RASTER = PHASE6_DIR / "CPI_conservation_priority.tif"
    CPI_METADATA = PHASE6_DIR / "CPI_metadata.csv"
    PRIORITY_CLASSES = PHASE6_DIR / "priority_classes.tif"
    URGENT_SITES = PHASE6_DIR / "urgent_action_sites_coordinates.csv"
    
    # Trends
    TRENDS_DATA = PHASE7_DIR / "species_trends_analysis.csv"
    
    # Boundaries
    SHAPES_DIR = RAW_DIR / "shapes"
    CORE_BOUNDARY = SHAPES_DIR / "Mangalavanam_Core.geojson"
    BUFFER_BOUNDARY = SHAPES_DIR / "Mangalavanam_Buffer.geojson"
    
    # Rasters
    RASTERS_DIR = RAW_DIR / "rasters"
    
    # Constants
    GUILDS = ["Wetland", "Forest", "Urban"]
    
    # Colors
    GUILD_COLORS = {
        "Wetland": "#2196F3",
        "Forest": "#4CAF50",
        "Urban": "#FF9800"
    }
    
    PRIORITY_COLORS = {
        5: "#d32f2f",
        4: "#f57c00",
        3: "#fbc02d",
        2: "#689f38",
        1: "#e0e0e0"
    }

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_data
def load_csv(filepath):
    """Load CSV file with caching"""
    try:
        if filepath.exists():
            return pd.read_csv(filepath)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading {filepath.name}: {e}")
        return pd.DataFrame()

@st.cache_data
def load_boundaries():
    """Load sanctuary boundaries"""
    try:
        core = gpd.read_file(Config.CORE_BOUNDARY)
        buffer = gpd.read_file(Config.BUFFER_BOUNDARY)
        return core, buffer
    except Exception as e:
        st.warning(f"Could not load boundaries: {e}")
        return None, None

def load_rasters(filepath):
    """Load raster file as numpy array"""
    try:
        with rasterio.open(filepath) as src:
            data = src.read(1).astype(np.float32)
            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)
            return data, src.bounds, src.crs
    except Exception as e:
        st.error(f"Error loading {filepath.name}: {e}")
        return None, None, None
    
def load_raster(file_path):
    """Load a raster file and return as numpy array with metadata"""
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)
            # Replace nodata with NaN
            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)
            return data, src.bounds, src.crs, src.transform
    except Exception as e:
        st.error(f"Error loading raster {file_path.name}: {e}")
        return None, None, None, None


def get_user_mode():
    """Check if user wants simple or technical language"""
    return st.session_state.get('simple_mode', True)

def format_occupancy(value, simple=True):
    """Format occupancy values for display"""
    if simple:
        score = int(value * 100)
        if score >= 80:
            return f"⭐⭐⭐⭐⭐ {score}/100"
        elif score >= 60:
            return f"⭐⭐⭐⭐ {score}/100"
        elif score >= 40:
            return f"⭐⭐⭐ {score}/100"
        else:
            return f"⭐⭐ {score}/100"
    else:
        return f"ψ = {value:.3f}"

def quality_label(score):
    """Get quality label from score"""
    if score >= 0.8:
        return "Excellent", "🟢"
    elif score >= 0.6:
        return "Good", "🟡"
    elif score >= 0.4:
        return "Fair", "🟠"
    else:
        return "Poor", "🔴"
    
@st.cache_data
def load_cpi_metadata():
    """Load CPI metadata"""
    try:
        if Config.CPI_METADATA.exists():
            return pd.read_csv(Config.CPI_METADATA)
        else:
            return None
    except Exception as e:
        st.error(f"Error loading CPI metadata: {e}")
        return None


    
# =============================================================================
# REUSABLE COMPONENTS
# =============================================================================

def create_metric_card(title, value, help_text=None, delta=None):
    """Create a styled metric card"""
    st.metric(
        label=title,
        value=value,
        delta=delta,
        help=help_text
    )

def create_map_with_boundaries(center=[9.9833, 76.2167], zoom=16):
    """Create base map with sanctuary boundaries"""
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles='OpenStreetMap'
    )
    
    # Add satellite layer
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False
    ).add_to(m)
    
    # Add boundaries
    core, buffer = load_boundaries()
    
    if buffer is not None:
        folium.GeoJson(
            buffer,
            name="Buffer Zone",
            style_function=lambda x: {
                'fillColor': 'transparent',
                'color': '#FF9800',
                'weight': 3,
                'dashArray': '10, 5'
            }
        ).add_to(m)
    
    if core is not None:
        folium.GeoJson(
            core,
            name="Core Sanctuary",
            style_function=lambda x: {
                'fillColor': 'rgba(76,175,80,0.2)',
                'color': '#2E7D32',
                'weight': 4
            }
        ).add_to(m)
    
    folium.LayerControl().add_to(m)
    return m

def show_clean_header(title, subtitle=None):
    """Display clean page header"""
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    st.markdown("---")

# =============================================================================
# PAGE 1: OVERVIEW
# =============================================================================

def render_overview_page():
    """Clean, scannable overview"""
    
    show_clean_header(
        "Adaptive Urban Biodiversity Monitoring System for Mangalavanam Bird Sanctuary"
    )
    
    simple = get_user_mode()
    
    # Quick Stats Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card(
            "Bird Species",
            "194",
            "Total unique species recorded (2019-2025)"
        )
    
    with col2:
        create_metric_card(
            "Study Area",
            "7.86 ha"
        )
    
    with col3:
        create_metric_card(
            "Overall Quality" if simple else "Mean Occupancy",
            "68/100" if simple else "0.68",
            "Average habitat suitability across all bird groups"
        )
    
    with col4:
        create_metric_card(
            "Priority Sites",
            "14",
            "Locations needing urgent conservation action"
        )
    
    st.markdown("###")
    
    # Guild Performance
    guild_profiles = load_csv(Config.GUILD_PROFILES)
    
    if not guild_profiles.empty:
        st.subheader("📊 Bird Group Performance")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Clean bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=guild_profiles['Guild'],
                y=guild_profiles['Env_mean'],
                marker_color=[Config.GUILD_COLORS[g] for g in guild_profiles['Guild']],
                text=guild_profiles['Env_mean'].apply(lambda x: f"{int(x*100)}/100" if simple else f"{x:.2f}"),
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Score: %{y:.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                xaxis_title="Bird Group",
                yaxis_title="Habitat Quality Score" if simple else "Occupancy (ψ)",
                showlegend=False,
                height=350,
                yaxis=dict(range=[0, 1]),
                margin=dict(t=20, b=60, l=60, r=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("###")
            
            for _, row in guild_profiles.iterrows():
                guild = row['Guild']
                score = row['Env_mean']
                label, emoji = quality_label(score)
                
                st.markdown(f"""
                **{emoji} {guild} Birds**  
                {int(score*100)}/100 - {label}
                """)
    
    # Simple Map
    st.markdown("###")
    st.subheader("📍 Location")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        map_obj = create_map_with_boundaries()
        
        # Add sanctuary marker
        folium.Marker(
            [9.9833, 76.2167],
            popup="<b>Mangalavanam Bird Sanctuary</b>",
            icon=folium.Icon(color='green', icon='leaf', prefix='fa')
        ).add_to(map_obj)
        
        st_folium(map_obj, width=700, height=400)
    
    with col2:
        st.markdown("###")
        st.markdown("""
        **Setting**  
        Urban mangrove ecosystem  
        Kochi city center
        
        **Zones**  
        🟢 Core: 2.74 ha (protected)  
        🟡 Buffer: 5.12 ha (managed)
        
        **Significance**  
        Biodiversity hotspot  
        Wetland ecosystem  
        Climate resilience
        """)

# =============================================================================
# PAGE 2: HABITAT MAPS
# =============================================================================

def render_habitat_maps_page():
    """Visual habitat quality maps with coordinates"""
    
    show_clean_header(
        "🗺️ Habitat Suitability Maps",
        "Where different bird groups thrive"
    )
    
    simple = get_user_mode()
    
    # Guild selector
    guild_names = {
        "Water Birds (Herons, Kingfishers)" if simple else "Wetland Guild": "Wetland",
        "Forest Birds (Woodpeckers, Drongos)" if simple else "Forest Guild": "Forest",
        "City Birds (Crows, Mynas)" if simple else "Urban Guild": "Urban"
    }
    
    selected = st.selectbox(
        "Select Bird Group",
        list(guild_names.keys())
    )
    
    guild = guild_names[selected]
    
    # Load raster
    raster_file = Config.MASKED_DIR / f"psi_env_{guild}_masked.tif"
    
    if not raster_file.exists():
        st.error("⚠️ Map data not found. Run: `python sentinel/phase4/create_rasters.py`")
        return
    
    data, bounds, crs = load_rasters(raster_file)
    
    if data is None:
        return
    
    # Main visualization
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Clean heatmap
        fig = px.imshow(
            data,
            color_continuous_scale='RdYlGn',
            aspect='auto',
            zmin=0, zmax=1,
            labels={'color': 'Quality'}
        )
        
        fig.update_layout(
            height=500,
            coloraxis_colorbar=dict(
                title="Score" if simple else "ψ",
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'] if simple else None
            ),
            xaxis=dict(showticklabels=False, title=""),
            yaxis=dict(showticklabels=False, title="")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("###")
        
        valid_data = data[np.isfinite(data)]
        mean_score = valid_data.mean()
        label, emoji = quality_label(mean_score)
        
        # st.metric(
        #     "Average",
        #     f"{int(mean_score*100)}/100" if simple else f"{mean_score:.2f}",
        #     help="Overall habitat quality"
        # )
        
        # st.metric(
        #     "Best Area",
        #     f"{int(valid_data.max()*100)}/100" if simple else f"{valid_data.max():.2f}"
        # )
        
        # st.metric(
        #     "Worst Area",
        #     f"{int(valid_data.min()*100)}/100" if simple else f"{valid_data.min():.2f}"
        # )
    
    # Location coordinates
    if Config.COORD_REFERENCE.exists():
        
        with st.expander("View Pixel Coordinates"):
            coords_df = load_csv(Config.COORD_REFERENCE)
            inside = coords_df[coords_df['Inside_Sanctuary'] == True]
            
            display_cols = ['Pixel_ID', 'Latitude', 'Longitude', 'Google_Maps_Link']
            display_df = inside[display_cols].copy()
            
            display_df['Google_Maps_Link'] = display_df['Google_Maps_Link'].apply(
                lambda x: f'[Open Maps]({x})'
            )
            
            st.dataframe(
                display_df.head(20),
                use_container_width=True,
                hide_index=True
            )

# =============================================================================
# PAGE 3: CONSERVATION PRIORITY
# =============================================================================

def render_conservation_priority_page():
    """Render conservation priority analysis with action recommendations"""
    st.title("Conservation Priority Analysis")
    st.markdown("*Data-driven identification of critical management areas*")

    # Load data
    
    if Config.PRIORITY_CLASSES.exists():
        # Load priority raster
        priority_data, bounds, crs, transform = load_raster(Config.PRIORITY_CLASSES)
        
        if priority_data is not None:
            # Two-column layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### Priority Classification Map")
                
                # Create categorical color map
                fig = px.imshow(
                    priority_data,
                    color_continuous_scale=[
                        [0.0, '#e0e0e0'],  # Level 1 - Gray
                        [0.25, '#689f38'], # Level 2 - Green
                        [0.5, '#fbc02d'],  # Level 3 - Yellow
                        [0.75, '#f57c00'], # Level 4 - Orange
                        [1.0, '#d32f2f']   # Level 5 - Red
                    ],
                    aspect='auto',
                    title=None
                )
                
                fig.update_layout(
                    height=500,
                    coloraxis_colorbar=dict(
                        title="Priority<br>Level",
                        tickvals=[1, 2, 3, 4, 5],
                        ticktext=['1-Minimal', '2-Low', '3-Medium', '4-High', '5-Critical']
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            
            with col2:
                st.markdown("### Priority Distribution")
                
                # Calculate area statistics
                valid_data = priority_data[np.isfinite(priority_data)]
                pixel_area_ha = 0.09  # 30m × 30m = 900m² = 0.09 ha
                
                priority_stats = {}
                for level in range(1, 6):
                    count = np.sum(valid_data == level)
                    area_ha = count * pixel_area_ha
                    pct = (count / len(valid_data)) * 100
                    priority_stats[level] = {
                        'Pixels': int(count),
                        'Area (ha)': round(area_ha, 2),
                        'Percentage': round(pct, 1)
                    }
                
                # Display as metrics
                for level in [5, 4, 3, 2, 1]:
                    emoji = ['⚪', '🟢', '🟡', '🟠', '🔴'][level-1]
                    color = Config.PRIORITY_COLORS[level]
                    area = priority_stats[level]['Area (ha)']
                    pct = priority_stats[level]['Percentage']
                    
                    st.markdown(f"""
                    <div style="background: {color}15; padding: 10px; border-radius: 5px; margin: 5px 0; border-left: 4px solid {color};">
                        <strong>{emoji} Level {level}</strong><br>
                        {area} ha ({pct}% of area)
                    </div>
                    """, unsafe_allow_html=True)

# =============================================================================
# PAGE 4: SPECIES DECLINE TRACKER - FIXED
# =============================================================================

def render_species_decline_page():
    """Population trends with focus on declining species"""
    
    show_clean_header(
        "Species Population Trends",
        "Tracking changes from 2019-2025"
    )
    
    simple = get_user_mode()
    
    # Load trends
    if not Config.TRENDS_DATA.exists():
        st.warning("⚠️ Trends data not found.")
        return
    
    trends_df = load_csv(Config.TRENDS_DATA)
    
    if trends_df.empty:
        st.warning("⚠️ No trends data available")
        return
    
    # FLEXIBLE: Auto-detect trend column name
    trend_col = None
    for col in ['Trend_Category', 'Status', 'Trend', 'trend_category', 'status', 'Category']:
        if col in trends_df.columns:
            trend_col = col
            break
    
    if trend_col is None:
        st.error("⚠️ Trends data missing required column")
        with st.expander("🔍 Debug Info"):
            st.write("**Available columns:**", list(trends_df.columns))
            st.write("**Expected:** Trend_Category, Status, or Trend")
        return
    
    # Summary stats
    declining = trends_df[trends_df[trend_col].str.contains('Declin', case=False, na=False)]
    stable = trends_df[trends_df[trend_col].str.contains('Stable', case=False, na=False)]
    increasing = trends_df[trends_df[trend_col].str.contains('Increas', case=False, na=False)]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "🔴 Declining",
            len(declining),
            delta=f"{len(declining)/len(trends_df)*100:.0f}%",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "⚪ Stable",
            len(stable),
            delta=f"{len(stable)/len(trends_df)*100:.0f}%"
        )
    
    with col3:
        st.metric(
            "🟢 Increasing",
            len(increasing),
            delta=f"{len(increasing)/len(trends_df)*100:.0f}%"
        )
    
    st.markdown("###")
    
    # Declining species visualization
    if len(declining) > 0:
        st.subheader("🔴 Species in Decline")
        
        declining_sorted = declining.sort_values('Percent_Change').head(15)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=declining_sorted['Species'],
            x=declining_sorted['Percent_Change'],
            orientation='h',
            marker_color='#d32f2f',
            text=declining_sorted['Percent_Change'].apply(lambda x: f"{x:.0f}%"),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Change: %{x:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            xaxis_title="Population Change (%)",
            yaxis_title="",
            height=500,
            margin=dict(l=200)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed list
        st.markdown("###")
        
        with st.expander("📋 View Detailed Species Data"):
            # Auto-detect available columns
            available_cols = declining_sorted.columns.tolist()
            
            display_cols = ['Species', 'Percent_Change']
            if trend_col in available_cols:
                display_cols.append(trend_col)
            if 'P_Value' in available_cols:
                display_cols.append('P_Value')
            elif 'p_value' in available_cols:
                display_cols.append('p_value')
            
            if simple:
                display_df = declining_sorted[display_cols].copy()
                rename_dict = {
                    'Percent_Change': 'Change (%)',
                    trend_col: 'Trend'
                }
                
                if 'P_Value' in display_cols:
                    rename_dict['P_Value'] = 'Confidence'
                    display_df = display_df.rename(columns=rename_dict)
                    display_df['Confidence'] = display_df['Confidence'].apply(
                        lambda x: "High" if x < 0.01 else "Medium" if x < 0.05 else "Low"
                    )
                elif 'p_value' in display_cols:
                    rename_dict['p_value'] = 'Confidence'
                    display_df = display_df.rename(columns=rename_dict)
                    display_df['Confidence'] = display_df['Confidence'].apply(
                        lambda x: "High" if x < 0.01 else "Medium" if x < 0.05 else "Low"
                    )
                else:
                    display_df = display_df.rename(columns=rename_dict)
            else:
                display_df = declining_sorted[display_cols]
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
    else:
        st.success("🎉 No significantly declining species detected!")
    
    # Increasing species (brief)
    if len(increasing) > 0:
        st.markdown("###")
        
        with st.expander("🟢 Species Showing Improvement"):
            increasing_sorted = increasing.sort_values('Percent_Change', ascending=False).head(10)
            
            fig2 = go.Figure()
            
            fig2.add_trace(go.Bar(
                y=increasing_sorted['Species'],
                x=increasing_sorted['Percent_Change'],
                orientation='h',
                marker_color='#4caf50',
                text=increasing_sorted['Percent_Change'].apply(lambda x: f"+{x:.0f}%"),
                textposition='outside'
            ))
            
            fig2.update_layout(
                xaxis_title="Population Change (%)",
                height=350
            )
            
            st.plotly_chart(fig2, use_container_width=True)

# =============================================================================
# PAGE 5: TEMPORAL TRENDS
# =============================================================================

def render_temporal_trends_page():
    """Year-over-year observation patterns"""
    
    show_clean_header(
        "📈 Temporal Trends",
        "Bird activity patterns over time"
    )
    
    simple = get_user_mode()
    
    # Load eBird data
    ebird_df = load_csv(Config.EBIRD_DATA)
    
    if ebird_df.empty:
        st.warning("⚠️ eBird observation data not found")
        return
    
    # Parse dates
    if 'OBSERVATION DATE' in ebird_df.columns:
        ebird_df['date'] = pd.to_datetime(ebird_df['OBSERVATION DATE'], errors='coerce')
        ebird_df['year'] = ebird_df['date'].dt.year
        ebird_df['month'] = ebird_df['date'].dt.month
        ebird_df = ebird_df.dropna(subset=['year'])
    elif 'year' not in ebird_df.columns:
        st.error("No temporal data available")
        return
    
    # Detect guild column
    guild_col = None
    for col in ['guild', 'Guild', 'functional_guild']:
        if col in ebird_df.columns:
            guild_col = col
            break
    
    # Annual trends
    st.subheader(" Annual Observations")
    
    if guild_col:
        yearly_counts = ebird_df.groupby(['year', guild_col]).size().reset_index(name='count')
        yearly_counts = yearly_counts.rename(columns={guild_col: 'guild'})
        
        fig = px.line(
            yearly_counts,
            x='year',
            y='count',
            color='guild',
            markers=True,
            color_discrete_map=Config.GUILD_COLORS
        )
    else:
        yearly_counts = ebird_df.groupby('year').size().reset_index(name='count')
        
        fig = px.line(
            yearly_counts,
            x='year',
            y='count',
            markers=True
        )
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Observations",
        height=350,
        showlegend=True,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Stats
    col1, col2, col3 = st.columns(3)
    
    years = sorted(ebird_df['year'].unique())
    
    if len(years) >= 2:
        first_year, last_year = min(years), max(years)
        
        count_first = len(ebird_df[ebird_df['year'] == first_year])
        count_last = len(ebird_df[ebird_df['year'] == last_year])
        change_pct = ((count_last - count_first) / count_first * 100) if count_first > 0 else 0
        
        with col1:
            st.metric(
                f"{first_year} Observations",
                count_first
            )
        
        with col2:
            st.metric(
                f"{last_year} Observations",
                count_last,
                delta=f"{change_pct:+.0f}%"
            )
        
        with col3:
            # Species richness
            if 'COMMON NAME' in ebird_df.columns:
                sp_first = ebird_df[ebird_df['year'] == first_year]['COMMON NAME'].nunique()
                sp_last = ebird_df[ebird_df['year'] == last_year]['COMMON NAME'].nunique()
                
                st.metric(
                    "Species Richness",
                    sp_last,
                    delta=sp_last - sp_first
                )
    
    # Seasonal patterns
    if 'month' in ebird_df.columns:
        st.markdown("###")
        st.subheader(" Seasonal Patterns")
        
        monthly = ebird_df.groupby('month').size().reset_index(name='count')
        
        month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                      7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        
        monthly['month_name'] = monthly['month'].map(month_names)
        
        fig2 = px.bar(
            monthly,
            x='month_name',
            y='count',
            color='count',
            color_continuous_scale='Greens'
        )
        
        fig2.update_layout(
            xaxis_title="Month",
            yaxis_title="Observations",
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Peak months
        peak_month = monthly.loc[monthly['count'].idxmax(), 'month_name']
        st.caption(f"🔝 Peak activity: {peak_month}")

# =============================================================================
# PAGE 6: GUILD ANALYSIS
# =============================================================================

def render_guild_analysis_page():
    """Comparison of different bird functional groups"""
    
    show_clean_header(
        "Bird Functional Groups",
        "Understanding different ecological roles"
    )
    
    simple = get_user_mode()
    
    # Load data
    guild_profiles = load_csv(Config.GUILD_PROFILES)
    functional = load_csv(Config.FUNCTIONAL_SUMMARY)
    
    if guild_profiles.empty:
        st.warning("⚠️ Guild analysis not found. Run Phase 4.")
        return
    
    # Guild cards
    st.subheader("📊 Group Comparison")
    
    cols = st.columns(3)
    
    guild_info = {
        "Wetland": {
            "icon": "🦆",
            "examples": "Herons, Kingfishers, Egrets",
            "habitat": "Water channels, mudflats"
        },
        "Forest": {
            "icon": "🦜",
            "examples": "Woodpeckers, Drongos, Bulbuls",
            "habitat": "Dense trees, canopy"
        },
        "Urban": {
            "icon": "🐦",
            "examples": "Crows, Mynas, Sparrows",
            "habitat": "Buildings, parks, roads"
        }
    }
    
    for i, (guild, info) in enumerate(guild_info.items()):
        with cols[i]:
            row = guild_profiles[guild_profiles['Guild'] == guild]
            
            if len(row) > 0:
                score = row['Env_mean'].values[0]
                label, emoji = quality_label(score)
                
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {Config.GUILD_COLORS[guild]};">
                    <h3 style="text-align: center;">{info['icon']} {guild}</h3>
                    <h1 style="text-align: center; color: {Config.GUILD_COLORS[guild]};">
                        {int(score*100)}/100
                    </h1>
                    <p style="text-align: center; color: #666;">
                        {label} {emoji}
                    </p>
                    <hr>
                    <p><strong>Examples:</strong><br>{info['examples']}</p>
                    <p><strong>Preferred:</strong><br>{info['habitat']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("###")
    
    # Detailed comparison
    st.subheader("📈 Performance Metrics")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Comparison chart
        models = {
            'Year': 'Temporal',
            'Zone': 'Spatial',
            'Env_mean': 'Environmental'
        }
        
        fig = go.Figure()
        
        for metric, label in models.items():
            fig.add_trace(go.Bar(
                name=label,
                x=guild_profiles['Guild'],
                y=guild_profiles[metric],
                text=guild_profiles[metric].apply(lambda x: f"{int(x*100)}"),
                textposition='auto'
            ))
        
        fig.update_layout(
            xaxis_title="Guild",
            yaxis_title="Score" if simple else "Occupancy (ψ)",
            barmode='group',
            height=350,
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("###")
        
        st.dataframe(
            guild_profiles[['Guild', 'Env_mean', 'Env_std']].rename(columns={
                'Env_mean': 'Mean Score',
                'Env_std': 'Variability'
            }).style.format({
                'Mean Score': '{:.2f}',
                'Variability': '{:.3f}'
            }),
            hide_index=True
        )
    
    # Ecosystem diversity
    if not functional.empty:
        st.markdown("###")
        st.subheader("🌿 Ecosystem Diversity")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig2 = go.Figure()
            
            fig2.add_trace(go.Bar(
                x=functional['Index'],
                y=functional['Mean'],
                error_y=dict(type='data', array=functional['Std']),
                marker_color='#4caf50',
                text=functional['Mean'].apply(lambda x: f"{x:.2f}"),
                textposition='outside'
            ))
            
            fig2.update_layout(
                xaxis_title="Diversity Index",
                yaxis_title="Value",
                height=300
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            st.markdown("###")
            
            shannon = functional[functional['Index'] == 'Shannon Diversity']['Mean'].values
            
            if len(shannon) > 0:
                h_val = shannon[0]
                
                if h_val > 0.8:
                    st.success("🟢 High diversity - Healthy ecosystem")
                elif h_val > 0.6:
                    st.info("🟡 Moderate diversity - Good balance")
                else:
                    st.warning("🟠 Low diversity - Needs attention")
                
                st.metric("Shannon H", f"{h_val:.3f}")

# =============================================================================
# PAGE 7: GUILD-SPECIFIC HABITAT MAPS
# =============================================================================

def render_guild_habitat_details():
    """Detailed guild-specific habitat maps"""
    
    show_clean_header(
        "🗺️ Detailed Guild Habitats",
        "Guild-specific suitability with satellite comparison"
    )
    
    simple = get_user_mode()
    
    # Tab-based layout for each guild
    tab1, tab2, tab3 = st.tabs([
        "🦆 Wetland Birds",
        "🦜 Forest Birds",
        "🐦 Urban Birds"
    ])
    
    for tab, guild in zip([tab1, tab2, tab3], Config.GUILDS):
        with tab:
            # Load guild-specific raster
            raster_file = Config.MASKED_DIR / f"psi_env_{guild}_masked.tif"
            
            if not raster_file.exists():
                st.warning(f"Map data not found for {guild}")
                continue
            
            data, bounds, crs = load_rasters(raster_file)
            
            if data is None:
                continue
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Habitat map
                fig = px.imshow(
                    data,
                    color_continuous_scale='RdYlGn',
                    aspect='auto',
                    zmin=0, zmax=1
                )
                
                fig.update_layout(
                    title=f"{guild} Bird Habitat Quality",
                    height=400,
                    coloraxis_colorbar=dict(
                        title="Quality"
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("###")
                
                valid = data[np.isfinite(data)]
                mean_score = valid.mean()
                
                st.metric("Average", f"{int(mean_score*100)}/100")
                st.metric("Best", f"{int(valid.max()*100)}/100")
                st.metric("Worst", f"{int(valid.min()*100)}/100")
                
                # Quality distribution
                excellent = (valid > 0.8).sum() / len(valid) * 100
                good = ((valid > 0.6) & (valid <= 0.8)).sum() / len(valid) * 100
                poor = (valid <= 0.6).sum() / len(valid) * 100
                
                st.markdown("###")
                st.markdown("**Distribution:**")
                st.markdown(f"🟢 Excellent: {excellent:.0f}%")
                st.markdown(f"🟡 Good: {good:.0f}%")
                st.markdown(f"🔴 Needs work: {poor:.0f}%")
            
            # Environmental correlate
            st.markdown("###")
            
            if guild == "Wetland":
                env_file = Config.RASTERS_DIR / "NDWI_2025_core.tif"
                env_label = "Water Availability (NDWI)"
            elif guild == "Forest":
                env_file = Config.RASTERS_DIR / "NDVI_2025_core.tif"
                env_label = "Vegetation Density (NDVI)"
            else:
                env_file = Config.RASTERS_DIR / "VIIRS_2025_core.tif"
                env_label = "Human Activity (Night Lights)"
            
            if env_file.exists():
                with st.expander(f"📊 Compare with {env_label}"):
                    env_data, _, _ = load_rasters(env_file)
                    
                    if env_data is not None:
                        # Resize to match habitat grid if needed
                        from scipy.ndimage import zoom
                        
                        if env_data.shape != data.shape:
                            zoom_factors = (data.shape[0] / env_data.shape[0], 
                                          data.shape[1] / env_data.shape[1])
                            env_data = zoom(env_data, zoom_factors, order=1)
                        
                        fig_env = px.imshow(
                            env_data,
                            color_continuous_scale='Blues' if guild == 'Wetland' else 'Greens' if guild == 'Forest' else 'Hot',
                            aspect='auto'
                        )
                        
                        fig_env.update_layout(height=300)
                        st.plotly_chart(fig_env, use_container_width=True)

# =============================================================================
# MAIN APPLICATION & NAVIGATION
# =============================================================================

def main():
    """Main application entry point"""
    
    # Sidebar - Clean navigation
    with st.sidebar:
        st.title("Mangalavanam")
        
        st.markdown("###")
        
        # # Simple mode toggle
        # simple_mode = st.checkbox(
        #     "💬 Simple Language",
        #     value=True,
        #     help="Use easy-to-understand explanations"
        # )
        
        # st.session_state['simple_mode'] = simple_mode
        
        # st.markdown("---")
        
        # Navigation
        st.markdown("### Navigation")
        
        page = st.radio(
            "",
            [
                "Overview",
                "Habitat Maps",
                "Conservation Priority",
                "Species Decline",
                "Temporal Trends",
                "Guild Analysis",
                "Guild Details"
            ],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 📈 Quick Stats")
        
        guild_profiles = load_csv(Config.GUILD_PROFILES)
        
        if not guild_profiles.empty:
            mean_score = guild_profiles['Env_mean'].mean()
            
            st.metric(
                "Overall Quality",
                f"{int(mean_score*100)}/100"
            )
        
        if Config.URGENT_SITES.exists():
            urgent = load_csv(Config.URGENT_SITES)
            st.metric(
                "Priority Sites",
                len(urgent)
            )
        
        st.markdown("---")
        
        # Info
        st.markdown("""
        ### ℹ️ About
        
        **Study Period**  
        2019 - 2025
        
        **Coverage**  
        7.86 hectares
        
        **Species**  
        194 birds
        
        **Data Source**  
        eBird citizen science
        """)

    # Route to selected page
    if page == "Overview":
        render_overview_page()
    
    elif page == "Habitat Maps":
        render_habitat_maps_page()
    
    elif page == "Conservation Priority":
        render_conservation_priority_page()
    
    elif page == "Species Decline":
        render_species_decline_page()
    
    elif page == "Temporal Trends":
        render_temporal_trends_page()
    
    elif page == "Guild Analysis":
        render_guild_analysis_page()
    
    elif page == "Guild Details":
        render_guild_habitat_details()
    

# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    main()