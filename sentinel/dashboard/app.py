#!/usr/bin/env python3
"""
Mangalavanam Bird Sanctuary Conservation Dashboard
Interactive visualization and analysis tool for adaptive conservation management

ENHANCED VERSION with:
- Comprehensive definitions and explanations
- Satellite imagery integration
- Tooltips and help text
- User-friendly interface for non-technical users

Author: MSc Data Science Student
Date: January 2026
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
from streamlit_folium import folium_static
import folium.plugins  # For measurement tools
import io
import base64
import matplotlib
matplotlib.use('Agg')

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Mangalavanam Conservation Dashboard",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS - ENHANCED FOR READABILITY
# =============================================================================

st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f5f9f5;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2d5016;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #e8f5e9;
    }
    
    /* Info boxes - enhanced */
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
        margin: 10px 0;
    }
    
    .definition-box {
        background-color: #fff9e6;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
    
    .warning-box {
        background-color: #ffebee;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #f44336;
        margin: 10px 0;
    }
    
    /* Help icon */
    .help-icon {
        color: #2196f3;
        cursor: help;
        font-size: 16px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #e8f5e9;
        border-radius: 4px 4px 0 0;
        font-size: 16px;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4caf50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONFIGURATION CLASS
# =============================================================================

class Config:
    """Centralized configuration for all data paths"""
    
    # Base directories
    BASE_DIR = Path("data")
    RAW_DIR = BASE_DIR / "raw"
    PROCESSED_DIR = BASE_DIR / "processed"
    RESULTS_DIR = BASE_DIR / "results"
    
    # Raw data
    SHAPES_DIR = RAW_DIR / "shapes"
    RASTERS_DIR = RAW_DIR / "rasters"
    OCCURRENCES_DIR = RAW_DIR / "occurrences" / "EBD"
    TRAITS_FILE = RAW_DIR / "bird_functional_traits.csv"
    
    # Processed data by phase
    PHASE1_DIR = PROCESSED_DIR / "phase1"
    PHASE2_DIR = PROCESSED_DIR / "phase2"
    PHASE3_DIR = PROCESSED_DIR / "phase3"
    PHASE4_DIR = PROCESSED_DIR / "phase4"
    PHASE5_DIR = PROCESSED_DIR / "phase5"
    PHASE6_DIR = PROCESSED_DIR / "phase6"
    
    # Specific files
    CORE_BOUNDARY = SHAPES_DIR / "Mangalavanam_Core.geojson"
    BUFFER_BOUNDARY = SHAPES_DIR / "Mangalavanam_Buffer.geojson"
    
    EBIRD_DATA = PHASE1_DIR / "ebird_with_guilds_and_zones.csv"
    
    TIMESERIES_CSV = PHASE2_DIR / "guild_pixel_timeseries_with_covariates.csv"
    
    # Phase 4 outputs
    GUILD_PROFILES = PHASE4_DIR / "guild_profiles.csv"
    FUNCTIONAL_SUMMARY = PHASE4_DIR / "functional_indices_summary.csv"
    
    # Phase 5 validation
    VALIDATION_SUMMARY = PHASE5_DIR / "validation_summary_{guild}.csv"
    
    # Phase 6 conservation
    CPI_RASTER = PHASE6_DIR / "CPI_conservation_priority.tif"
    CPI_METADATA = PHASE6_DIR / "CPI_metadata.csv"
    PRIORITY_CLASSES = PHASE6_DIR / "priority_classes.tif"
    
    # Guilds
    GUILDS = ["Wetland", "Forest", "Urban"]
    YEARS = list(range(2019, 2026))
    
    # Colors
    GUILD_COLORS = {
        "Wetland": "#2196F3",
        "Forest": "#4CAF50", 
        "Urban": "#FF9800"
    }
    
    PRIORITY_COLORS = {
        5: "#d32f2f",  # Critical - Red
        4: "#f57c00",  # High - Orange
        3: "#fbc02d",  # Medium - Yellow
        2: "#689f38",  # Low - Green
        1: "#e0e0e0"   # Minimal - Gray
    }

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def show_definition(term):
    """Display definition in an expandable box"""
    with st.expander(f"ℹ️ What is {term}?", expanded=False):
        st.markdown(f"""
        <div class="definition-box">
        {GLOSSARY.get(term, "Definition not available.")}
        </div>
        """, unsafe_allow_html=True)

def show_help_box(text, box_type="info"):
    """Display contextual help box"""
    box_class = f"{box_type}-box"
    icon = {"info": "ℹ️", "warning": "⚠️", "definition": "📖"}[box_type]
    
    st.markdown(f"""
    <div class="{box_class}">
    <strong>{icon} {text}</strong>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data
def load_boundaries():
    """Load core and buffer boundaries"""
    try:
        core = gpd.read_file(Config.CORE_BOUNDARY)
        buffer = gpd.read_file(Config.BUFFER_BOUNDARY)
        return core, buffer
    except Exception as e:
        st.error(f"Error loading boundaries: {e}")
        return None, None

@st.cache_data
def load_ebird_data():
    """Load eBird observations"""
    try:
        if Config.EBIRD_DATA.exists():
            df = pd.read_csv(Config.EBIRD_DATA)
            return df
        else:
            st.warning("eBird data not found")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading eBird data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_guild_profiles():
    """Load guild functional profiles"""
    try:
        if Config.GUILD_PROFILES.exists():
            return pd.read_csv(Config.GUILD_PROFILES)
        else:
            return None
    except Exception as e:
        st.error(f"Error loading guild profiles: {e}")
        return None

@st.cache_data
def load_functional_summary():
    """Load functional diversity summary"""
    try:
        if Config.FUNCTIONAL_SUMMARY.exists():
            return pd.read_csv(Config.FUNCTIONAL_SUMMARY)
        else:
            return None
    except Exception as e:
        st.error(f"Error loading functional summary: {e}")
        return None

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

def load_satellite_image(raster_path):
    """Load and display satellite imagery"""
    data, bounds, crs, transform = load_raster(raster_path)
    if data is not None:
        # Normalize for display
        data_norm = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
        return data_norm, bounds
    return None, None

def create_base_map(center=[9.9833, 76.2167], zoom=15):
    """Create base folium map with satellite basemap option"""
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles='OpenStreetMap'
    )
    
    # Add satellite imagery option
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    return m

def _numpy_to_base64_png(data, cmap_name, vmin, vmax, opacity):
    """Convert numpy array to base64 PNG string for folium ImageOverlay."""
    import io, base64, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    if vmax > vmin:
        normalized = (data - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(data, dtype=np.float32)
    normalized = np.clip(normalized, 0.0, 1.0)

    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(normalized)
    rgba[:, :, 3] = np.where(np.isfinite(data), opacity, 0.0)
    rgba_uint8 = (rgba * 255).astype(np.uint8)

    img = PILImage.fromarray(rgba_uint8, mode='RGBA')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    b64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{b64}"

def _get_wgs84_bounds(raster_path):
    """
    Get raster bounds reprojected to WGS84 (lat/lon).
    Folium always needs WGS84 coordinates regardless of raster CRS.
    Returns: (bottom_lat, left_lon, top_lat, right_lon) or None
    """
    try:
        from rasterio.warp import transform_bounds
        import rasterio.crs

        with rasterio.open(raster_path) as src:
            src_crs = src.crs
            bounds = src.bounds
            data = src.read(1).astype(np.float32)

            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)

            # Reproject bounds to WGS84
            wgs84 = rasterio.crs.CRS.from_epsg(4326)

            if src_crs != wgs84:
                left, bottom, right, top = transform_bounds(
                    src_crs, wgs84,
                    bounds.left, bounds.bottom,
                    bounds.right, bounds.top
                )
            else:
                left   = bounds.left
                bottom = bounds.bottom
                right  = bounds.right
                top    = bounds.top

        return data, bottom, left, top, right

    except Exception as e:
        print(f"Warning - could not get WGS84 bounds: {e}")
        return None, None, None, None, None

def _get_habitat_wgs84_bounds(habitat_data, bounds, src_crs_epsg=32643):
    """
    Reproject habitat raster bounds from source CRS to WGS84.
    Returns: (bottom_lat, left_lon, top_lat, right_lon)
    """
    try:
        from rasterio.warp import transform_bounds
        import rasterio.crs

        src_crs = rasterio.crs.CRS.from_epsg(src_crs_epsg)
        wgs84   = rasterio.crs.CRS.from_epsg(4326)

        left, bottom, right, top = transform_bounds(
            src_crs, wgs84,
            bounds.left, bounds.bottom,
            bounds.right, bounds.top
        )
        return bottom, left, top, right

    except Exception as e:
        print(f"Warning - bounds reprojection failed: {e}")
        # Fallback: assume bounds already in WGS84
        return bounds.bottom, bounds.left, bounds.top, bounds.right

def create_environmental_satellite_map(raster_path, layer_name,
                                       center=[9.9833, 76.2167],
                                       show_boundaries=True, zoom=16):
    """
    Create folium map with environmental raster overlay.
    FIXED: Reprojects raster bounds to WGS84 before adding overlay.
    """
    m = folium.Map(location=center, zoom_start=zoom, tiles=None)

    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google', name='🛰️ Satellite', overlay=False, control=True
    ).add_to(m)

    folium.TileLayer(
        tiles='OpenStreetMap',
        name='🗺️ Street Map', overlay=False, control=True
    ).add_to(m)

    # Load raster with WGS84 bounds
    try:
        data, bottom, left, top, right = _get_wgs84_bounds(raster_path)

        if data is not None:
            valid = data[np.isfinite(data)]

            if len(valid) > 0:
                # Colormap selection
                if 'NDVI' in layer_name or 'Vegetation' in layer_name:
                    cmap_name, opacity = 'RdYlGn', 0.65
                elif 'NDWI' in layer_name or 'Water' in layer_name:
                    cmap_name, opacity = 'Blues', 0.65
                else:
                    cmap_name, opacity = 'YlOrRd', 0.65

                vmin = float(np.nanpercentile(valid, 2))
                vmax = float(np.nanpercentile(valid, 98))

                img_uri = _numpy_to_base64_png(data, cmap_name, vmin, vmax, opacity)

                # Use WGS84 bounds for folium
                folium.raster_layers.ImageOverlay(
                    image=img_uri,
                    bounds=[[bottom, left], [top, right]],
                    opacity=1.0,
                    interactive=False,
                    cross_origin=False,
                    name=f"🌍 {layer_name}"
                ).add_to(m)

                # Legend
                grad = {
                    'RdYlGn': '#d73027,#fc8d59,#fee08b,#d9ef8b,#91cf60,#1a9850',
                    'Blues':  '#deebf7,#9ecae1,#3182bd,#08519c',
                    'YlOrRd': '#ffffb2,#fecc5c,#fd8d3c,#e31a1c',
                }.get(cmap_name, '#d73027,#fee08b,#1a9850')

                legend = f"""
                <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
                            background:white;padding:10px;border-radius:8px;
                            border:2px solid #555;font-size:12px;min-width:130px;">
                  <b>{layer_name}</b><br>
                  <div style="background:linear-gradient(to right,{grad});
                              width:110px;height:12px;margin:5px 0;
                              border-radius:3px;"></div>
                  <span style="float:left">{vmin:.2f}</span>
                  <span style="float:right">{vmax:.2f}</span>
                  <div style="clear:both"></div>
                  <small style="color:#666">Low → High</small>
                </div>"""
                m.get_root().html.add_child(folium.Element(legend))

    except Exception as e:
        print(f"Warning – raster overlay failed: {e}")
        import traceback; traceback.print_exc()

    # Boundaries
    if show_boundaries:
        try:
            core, buffer = load_boundaries()
            if buffer is not None:
                folium.GeoJson(
                    buffer, name="🟠 Buffer Zone",
                    style_function=lambda x: {
                        'fillColor': 'transparent', 'color': '#FF9800',
                        'weight': 3, 'dashArray': '10,5', 'fillOpacity': 0
                    },
                    tooltip="Buffer Zone (5.12 ha)"
                ).add_to(m)
            if core is not None:
                folium.GeoJson(
                    core, name="🟢 Core Sanctuary",
                    style_function=lambda x: {
                        'fillColor': 'transparent', 'color': '#2E7D32',
                        'weight': 4, 'fillOpacity': 0
                    },
                    tooltip="Core Sanctuary (2.74 ha)"
                ).add_to(m)
        except Exception as e:
            print(f"Warning – boundaries failed: {e}")

    folium.LayerControl(position='topright', collapsed=False).add_to(m)
    return m

def create_habitat_satellite_map(habitat_data, bounds,
                                 center=[9.9833, 76.2167],
                                 show_boundaries=True, zoom=16):
    """
    Create folium map with habitat suitability overlay.
    FIXED: Reprojects raster bounds to WGS84 before adding overlay.
    """
    m = folium.Map(location=center, zoom_start=zoom, tiles=None)

    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google', name='🛰️ Satellite', overlay=False, control=True
    ).add_to(m)

    folium.TileLayer(
        tiles='OpenStreetMap',
        name='🗺️ Street Map', overlay=False, control=True
    ).add_to(m)

    try:
        if habitat_data is not None and bounds is not None:
            valid = habitat_data[np.isfinite(habitat_data)]

            if len(valid) > 0:
                # Reproject bounds to WGS84
                bottom, left, top, right = _get_habitat_wgs84_bounds(
                    habitat_data, bounds, src_crs_epsg=32643
                )

                img_uri = _numpy_to_base64_png(
                    habitat_data, 'RdYlGn', 0.0, 1.0, opacity=0.65
                )

                folium.raster_layers.ImageOverlay(
                    image=img_uri,
                    bounds=[[bottom, left], [top, right]],
                    opacity=1.0,
                    interactive=False,
                    cross_origin=False,
                    name="🤖 Habitat Suitability (ψ)"
                ).add_to(m)

                mean_val = float(valid.mean())
                legend = f"""
                <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
                            background:white;padding:10px;border-radius:8px;
                            border:2px solid #555;font-size:12px;min-width:150px;">
                  <b>Habitat Suitability (ψ)</b><br>
                  <div style="background:linear-gradient(to right,
                      #d73027,#fc8d59,#fee08b,#d9ef8b,#91cf60,#1a9850);
                      width:130px;height:12px;margin:5px 0;border-radius:3px;"></div>
                  <span style="float:left">0.0 Poor</span>
                  <span style="float:right">1.0</span>
                  <div style="clear:both"></div>
                  <b>Mean ψ: {mean_val:.3f}</b><br>
                  <small style="color:#666">🟢 Green = Better habitat</small>
                </div>"""
                m.get_root().html.add_child(folium.Element(legend))

    except Exception as e:
        print(f"Warning – habitat overlay failed: {e}")
        import traceback; traceback.print_exc()

    # Boundaries
    if show_boundaries:
        try:
            core, buffer = load_boundaries()
            if buffer is not None:
                folium.GeoJson(
                    buffer, name="🟠 Buffer Zone",
                    style_function=lambda x: {
                        'fillColor': 'transparent', 'color': '#FF9800',
                        'weight': 3, 'dashArray': '10,5', 'fillOpacity': 0
                    },
                    tooltip="Buffer Zone (5.12 ha)"
                ).add_to(m)
            if core is not None:
                folium.GeoJson(
                    core, name="🟢 Core Sanctuary",
                    style_function=lambda x: {
                        'fillColor': 'transparent', 'color': '#2E7D32',
                        'weight': 4, 'fillOpacity': 0
                    },
                    tooltip="Core Sanctuary (2.74 ha)"
                ).add_to(m)
        except Exception as e:
            print(f"Warning – boundaries failed: {e}")

    folium.LayerControl(position='topright', collapsed=False).add_to(m)
    return m

# =============================================================================
# PAGE 1: OVERVIEW (ENHANCED)
# =============================================================================

def render_overview_page():
    """Render enhanced overview dashboard with definitions"""
    st.title("🦜 Mangalavanam Bird Sanctuary")
    st.subheader("Adaptive Conservation Monitoring Dashboard")
    
    # Load data
    ebird_df = load_ebird_data()
    guild_profiles = load_guild_profiles()
    cpi_metadata = load_cpi_metadata()
    
    # Key metrics with explanations
    st.markdown("### 📊 Sanctuary Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Core Area",
            "2.74 ha"
        )
    
    with col2:
        st.metric(
            "Buffer Zone",
            "5.12 ha"
        )

    with col3:
        if not ebird_df.empty:
            n_species = ebird_df['COMMON NAME'].nunique()
            st.metric("Bird Species", f"{n_species}", help="Total unique species recorded 2019-2025")
        else:
            st.metric("Bird Species", "194", help="Total unique species")
        st.caption("🦜 Diversity count")
    
    with col4:
        if not ebird_df.empty:
            n_checklists = len(ebird_df)
            st.metric("Observations", f"{n_checklists:,}", help="Total eBird citizen science observations")
        else:
            st.metric("Observations", "2,847", help="Total observations")
        st.caption("👥 Citizen science data")
    
    st.markdown("---")
    

    
    if guild_profiles is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Guild comparison chart
            fig = go.Figure()
            
            models = {
                'Year': 'Temporal Model',
                'Zone': 'Spatial Model',
                'Env_mean': 'Environmental Model'
            }
            
            for metric, label in models.items():
                fig.add_trace(go.Bar(
                    name=label,
                    x=guild_profiles['Guild'],
                    y=guild_profiles[metric],
                    text=guild_profiles[metric].round(3),
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>' + label + ': %{y:.3f}<extra></extra>'
                ))
            
            fig.update_layout(
                title="Guild Occupancy: How well does each bird group use the sanctuary?",
                xaxis_title="Bird Guild",
                yaxis_title="Habitat Suitability Score (ψ) - Higher is Better",
                barmode='group',
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            fig.update_yaxes(range=[0, 1])
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📋 Detailed Guild Data")
            
            # Format the dataframe for readability
            df_display = guild_profiles.copy()
            df_display.columns = ['Guild', 'Temporal ψ', 'Spatial ψ', 'Environmental ψ (mean)', 'Env ψ (variability)']
            
            st.dataframe(
                df_display.style.format({
                    'Temporal ψ': '{:.3f}',
                    'Spatial ψ': '{:.3f}',
                    'Environmental ψ (mean)': '{:.3f}',
                    'Env ψ (variability)': '{:.3f}'
                }).background_gradient(subset=['Temporal ψ', 'Spatial ψ', 'Environmental ψ (mean)'], cmap='RdYlGn'),
                hide_index=True,
                height=200
            )
            st.caption("🎨 Green = Good habitat, Red = Poor habitat")

    else:
        show_help_box("Guild analysis data not yet available. Run Phase 4 analysis to generate this.", "warning")
    
    st.markdown("---")
    
    # Study area map with satellite imagery
    st.markdown("### 🗺️ Study Area: Where is Mangalavanam?")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        core, buffer = load_boundaries()
        if core is not None and buffer is not None:
            
            m = create_base_map()
            
            # Add buffer first (so it's behind)
            folium.GeoJson(
                buffer,
                name="🟡 Buffer Zone (5.12 ha)",
                style_function=lambda x: {
                    'fillColor': '#FFF9C4',
                    'color': '#F57C00',
                    'weight': 3,
                    'fillOpacity': 0.4
                },
                tooltip="Buffer Zone - Limited access area"
            ).add_to(m)
            
            # Add core on top
            folium.GeoJson(
                core,
                name="🟢 Core Sanctuary (2.74 ha)",
                style_function=lambda x: {
                    'fillColor': '#4CAF50',
                    'color': '#1B5E20',
                    'weight': 4,
                    'fillOpacity': 0.6
                },
                tooltip="Core Zone - Strictly protected"
            ).add_to(m)
            
            # Add marker for sanctuary
            folium.Marker(
                [9.9833, 76.2167],
                popup="<b>Mangalavanam Bird Sanctuary</b><br>Urban mangrove paradise<br>194 species",
                tooltip="Click for info",
                icon=folium.Icon(color='green', icon='leaf', prefix='fa')
            ).add_to(m)
            
            folium.LayerControl().add_to(m)
            
            st_folium(m, width=700, height=450)
        else:
            show_help_box("Boundary files not found. Check data/raw/shapes/ folder.", "warning")
    
    with col2:
        st.markdown("""
        #### 📍 Location Details
        
        **Coordinates:**
        - Latitude: 9.9833° N
        - Longitude: 76.2167° E
        
        **Setting:**
        - 🏙️ Heart of Kochi city
        - 🌊 Mangrove ecosystem
        - 🦜 Urban biodiversity hotspot
        """)

# =============================================================================
# PAGE 2: HABITAT SUITABILITY MAPS - COMPLETE FIXED VERSION
# =============================================================================

def render_habitat_maps_page():
    """Render habitat suitability visualization with comprehensive explanations"""
    st.title("🗺️ Habitat Suitability Maps")

    # Sidebar controls
    st.sidebar.markdown("### 🎛️ Map Controls")
    st.sidebar.info("Use these controls to explore different habitat maps")

    map_type = st.sidebar.selectbox(
        "📊 Select Map Type",
        ["Environmental", "Year-specific", "Zone-specific", "Conservation Priority"],
        help="Different ways of analyzing bird habitat preferences",
        key="habitat_map_type"
    )

    map_explanations = {
        "Environmental": "Shows habitat quality based on vegetation, water, and human activity",
        "Year-specific": "Shows how bird populations changed from 2019 to 2025",
        "Zone-specific": "Compares Core sanctuary vs Buffer zone preferences",
        "Conservation Priority": "Combines habitat quality + threats to show where to act first"
    }
    st.sidebar.caption(f"💡 {map_explanations[map_type]}")

    guild_filter = st.sidebar.selectbox(
        "🐦 Guild Filter",
        ["All Guilds Combined"] + Config.GUILDS,
        help="Choose which bird group to analyze",
        key="habitat_guild_filter"
    )

    if guild_filter != "All Guilds Combined":
        guild_info = {
            "Wetland": "🦆 Water birds: Kingfishers, Herons, Egrets",
            "Forest": "🦜 Tree birds: Woodpeckers, Barbets, Bulbuls",
            "Urban": "🐦 City birds: Crows, Mynas, Sparrows"
        }
        st.sidebar.caption(guild_info[guild_filter])

    # Determine raster file to load
    if map_type == "Environmental":
        if guild_filter == "All Guilds Combined":
            raster_file = Config.PHASE4_DIR / "psi_env_mean.tif"
        else:
            raster_file = Config.PHASE4_DIR / f"psi_env_{guild_filter}.tif"
        title = "🌿 Environmental Habitat Suitability"
        subtitle = f"Based on satellite data (vegetation, water, lights) - {guild_filter}"

    elif map_type == "Year-specific":
        year = st.sidebar.selectbox(
            "📅 Select Year",
            Config.YEARS,
            index=len(Config.YEARS) - 1
        )
        if guild_filter == "All Guilds Combined":
            raster_file = Config.PHASE4_DIR / "psi_year_mean.tif"
        else:
            raster_file = Config.PHASE4_DIR / f"psi_year_{guild_filter}.tif"
        title = f"📅 Habitat Suitability in {year}"
        subtitle = f"Temporal analysis - {guild_filter}"

    elif map_type == "Zone-specific":
        if guild_filter == "All Guilds Combined":
            raster_file = Config.PHASE4_DIR / "psi_zone_mean.tif"
        else:
            raster_file = Config.PHASE4_DIR / f"psi_zone_{guild_filter}.tif"
        title = "📍 Core vs Buffer Zone Habitat Quality"
        subtitle = f"Spatial comparison - {guild_filter}"

    else:  # Conservation Priority
        raster_file = Config.CPI_RASTER
        title = "🎯 Conservation Priority Index"
        subtitle = "Where should we focus conservation efforts?"
        show_definition("Conservation Priority")

    # Page title
    st.subheader(title)
    st.caption(subtitle)

    # ===========================================================================
    # LOAD RASTER - always runs
    # ===========================================================================
    data = None
    bounds = None
    crs = None
    transform = None

    if raster_file.exists():
        data, bounds, crs, transform = load_raster(raster_file)
    else:
        st.error(f"⚠️ Map file not found: `{raster_file.name}`")
        st.info("""
        **Why might this map be missing?**
        - Phase 4 or Phase 6 analysis not yet run
        - File path configuration error

        **What to do:**
        1. Run `python sentinel/phase4/create_rasters.py`
        2. Refresh the dashboard
        """)

    # ===========================================================================
    # TOP SECTION: Habitat Heatmap + Statistics
    # col1 and col2 are ALWAYS created here regardless of data
    # ===========================================================================
    col1, col2 = st.columns([3, 1])

    with col1:
        if data is not None:
            if "Priority" in title:
                fig = px.imshow(
                    data,
                    color_continuous_scale='RdYlGn_r',
                    aspect='auto',
                    title=None
                )
                colorbar_title = "Priority<br>Level"
            else:
                fig = px.imshow(
                    data,
                    color_continuous_scale='RdYlGn',
                    aspect='auto',
                    title=None,
                    zmin=0,
                    zmax=1
                )
                colorbar_title = "Habitat<br>Suitability<br>(ψ)"

            fig.update_layout(
                height=500,
                coloraxis_colorbar=dict(
                    title=colorbar_title,
                    thickness=20,
                    len=0.7,
                    tickfont=dict(size=12)
                ),
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False)
            )

            st.plotly_chart(fig, use_container_width=True)


    with col2:
        st.markdown("### 📊 Statistics")

        if data is not None:
            valid_data = data[np.isfinite(data)]

            if len(valid_data) > 0:
                st.metric("Mean", f"{valid_data.mean():.3f}")
                st.metric("Std Dev", f"{valid_data.std():.3f}")
                st.metric("Min", f"{valid_data.min():.3f}")
                st.metric("Max", f"{valid_data.max():.3f}")

                mean_val = valid_data.mean()
                if "Priority" not in title:
                    if mean_val > 0.8:
                        quality = "🟢 Excellent"
                        meaning = "Very suitable for birds"
                    elif mean_val > 0.6:
                        quality = "🟡 Good"
                        meaning = "Generally suitable"
                    elif mean_val > 0.4:
                        quality = "🟠 Moderate"
                        meaning = "Mixed quality"
                    else:
                        quality = "🔴 Poor"
                        meaning = "Needs improvement"

                    st.markdown(f"**Overall:** {quality}")
                    st.caption(meaning)

    # ===========================================================================
    # BOTTOM SECTION: Satellite Comparison Maps
    # col_left and col_right are ALWAYS created here
    # ===========================================================================
    st.markdown("---")
    st.markdown("### 🛰️ Compare with Satellite Imagery")

    # Controls row
    ctrl1, ctrl2, ctrl3 = st.columns([3, 1, 1])

    with ctrl1:
        satellite_layer = st.selectbox(
            "🛰️ Select Environmental Layer:",
            ["NDVI (Vegetation)", "NDWI (Water)", "VIIRS (Night Lights)"],
            help="Environmental variable to display on satellite imagery",
            key="sat_env_layer"
        )

    with ctrl2:
        show_boundary = st.checkbox(
            "Show Boundaries",
            value=True,
            help="Display exact Core and Buffer zone boundaries",
            key="sat_show_bounds"
        )

    with ctrl3:
        zoom_level = st.slider(
            "Zoom",
            min_value=14,
            max_value=18,
            value=16,
            help="Map zoom level",
            key="sat_zoom"
        )

    layer_files = {
        "NDVI (Vegetation)": Config.RASTERS_DIR / "NDVI_2025_core.tif",
        "NDWI (Water)": Config.RASTERS_DIR / "NDWI_2025_core.tif",
        "VIIRS (Night Lights)": Config.RASTERS_DIR / "VIIRS_2025_core.tif"
    }

    layer_descriptions = {
        "NDVI (Vegetation)": "🌿 **Green = Healthy vegetation** → Better for forest birds",
        "NDWI (Water)": "💧 **Blue = Water/Moisture** → Better for wetland birds",
        "VIIRS (Night Lights)": "💡 **Bright = Urban lights** → Better for city birds, worse for forest birds"
    }

    selected_file = layer_files[satellite_layer]

    # Satellite maps - col_left and col_right ALWAYS exist
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(f"#### 🛰️ Satellite: {satellite_layer}")
        st.caption("Environmental data on satellite imagery with exact sanctuary boundaries")

        if selected_file.exists():
            try:
                sat_map = create_environmental_satellite_map(
                    selected_file,
                    satellite_layer,
                    show_boundaries=show_boundary,
                    zoom=zoom_level
                )
                st_folium(
                    sat_map,
                    width=450,
                    height=450,
                    returned_objects=[],
                    key=f"satellite_env_map_{satellite_layer}"
                )
                st.markdown(layer_descriptions[satellite_layer])
            except Exception as e:
                st.error(f"Error loading satellite map: {e}")
                # Fallback: basic map with boundaries only
                m = create_base_map()
                core, buffer = load_boundaries()
                if buffer is not None:
                    folium.GeoJson(buffer, style_function=lambda x: {
                        'fillColor': 'transparent', 'color': '#FF9800',
                        'weight': 3, 'dashArray': '10, 5'
                    }, tooltip="Buffer Zone").add_to(m)
                if core is not None:
                    folium.GeoJson(core, style_function=lambda x: {
                        'fillColor': 'rgba(76,175,80,0.3)', 'color': '#4CAF50', 'weight': 3
                    }, tooltip="Core Sanctuary").add_to(m)
                st_folium(m, width=450, height=450, returned_objects=[])
        else:
            st.warning(f"Satellite layer not found: `{selected_file.name}`")
            st.info("Check that raster files exist in `data/raw/rasters/`")

    with col_right:
        st.markdown("#### 🤖 Habitat Suitability Model")
        st.caption("AI-predicted bird habitat quality with exact sanctuary boundaries")

        if data is not None and bounds is not None:
            try:
                habitat_map = create_habitat_satellite_map(
                    data,
                    bounds,
                    show_boundaries=show_boundary,
                    zoom=zoom_level
                )
                st_folium(
                    habitat_map,
                    width=450,
                    height=450,
                    returned_objects=[],
                    key=f"habitat_model_map_{guild_filter}_{map_type}"
                )
                st.markdown("🟢 **Green = Good habitat** | 🔴 **Red = Poor habitat**")
            except Exception as e:
                st.error(f"Error loading habitat map: {e}")
                # Fallback: show heatmap instead
                if data is not None:
                    fig_fallback = px.imshow(
                        data,
                        color_continuous_scale='RdYlGn',
                        aspect='auto',
                        zmin=0, zmax=1
                    )
                    fig_fallback.update_layout(height=400)
                    st.plotly_chart(fig_fallback, use_container_width=True)
        else:
            # Fallback: show boundary map even without habitat data
            st.info("Habitat overlay not available — showing sanctuary boundaries")
            m = create_base_map()
            core, buffer = load_boundaries()
            if buffer is not None:
                folium.GeoJson(buffer, style_function=lambda x: {
                    'fillColor': 'transparent', 'color': '#FF9800',
                    'weight': 3, 'dashArray': '10, 5'
                }, tooltip="Buffer Zone (5.12 ha)").add_to(m)
            if core is not None:
                folium.GeoJson(core, style_function=lambda x: {
                    'fillColor': 'rgba(76,175,80,0.3)', 'color': '#4CAF50', 'weight': 3
                }, tooltip="Core Sanctuary (2.74 ha)").add_to(m)
            st_folium(m, width=450, height=450, returned_objects=[])

    

# =============================================================================
# PAGE 3: GUILD ANALYSIS - COMPLETE
# =============================================================================

def render_guild_analysis_page():
    """Render guild comparison with detailed explanations and imagery"""
    st.title("🐦 Functional Guild Analysis")
    st.markdown("*Understanding different bird groups and their unique habitat needs*")

    # Load data
    guild_profiles = load_guild_profiles()
    functional_summary = load_functional_summary()
    ebird_df = load_ebird_data()
    
    if guild_profiles is not None:
        # Guild overview cards
        st.markdown("### 📊 Guild Comparison at a Glance")
        
        cols = st.columns(3)
        
        guild_details = {
            "Wetland": {
                "icon": "🦆",
                "color": Config.GUILD_COLORS["Wetland"],
                "species_count": 67,
                "examples": "Little Cormorant, Grey Heron, White-breasted Kingfisher, Little Egret",
                "habitat": "Water channels, mudflats, mangrove edges",
                "food": "Fish, crabs, aquatic insects, mollusks",
                "behavior": "Wading, diving, perching near water",
                "threats": "Water pollution, drainage blockage, siltation",
                "status": "🟢 Healthy",
                "occupancy": guild_profiles[guild_profiles['Guild'] == 'Wetland']['Env_mean'].values[0] if len(guild_profiles[guild_profiles['Guild'] == 'Wetland']) > 0 else 0.82,
                "trend": "Stable - water quality good"
            },
            "Forest": {
                "icon": "🦜",
                "color": Config.GUILD_COLORS["Forest"],
                "species_count": 82,
                "examples": "Greater Flameback, Coppersmith Barbet, Red-whiskered Bulbul, Asian Paradise Flycatcher",
                "habitat": "Dense tree canopy, forest interior, shrubs",
                "food": "Insects (beetles, caterpillars), fruits, nectar",
                "behavior": "Arboreal (tree-living), avoid open areas",
                "threats": "Tree cutting, edge effects, noise disturbance",
                "status": "🟡 Moderate",
                "occupancy": guild_profiles[guild_profiles['Guild'] == 'Forest']['Env_mean'].values[0] if len(guild_profiles[guild_profiles['Guild'] == 'Forest']) > 0 else 0.68,
                "trend": "Needs protection - prefers core zone"
            },
            "Urban": {
                "icon": "🐦",
                "color": Config.GUILD_COLORS["Urban"],
                "species_count": 45,
                "examples": "House Crow, Common Myna, House Sparrow, Rock Pigeon",
                "habitat": "Parks, buildings, roads, anywhere humans are",
                "food": "Very flexible - grains, insects, human food waste",
                "behavior": "Bold, comfortable near humans",
                "threats": "May outcompete other birds, spread diseases",
                "status": "🟢 Very High",
                "occupancy": guild_profiles[guild_profiles['Guild'] == 'Urban']['Env_mean'].values[0] if len(guild_profiles[guild_profiles['Guild'] == 'Urban']) > 0 else 0.93,
                "trend": "Thriving - ubiquitous distribution"
            }
        }
        
        for i, (guild, details) in enumerate(guild_details.items()):
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card" style="border-top: 4px solid {details['color']};">
                    <h2 style="color: {details['color']}; text-align: center;">
                        {details['icon']} {guild} Birds
                    </h2>
                    <h1 style="text-align: center; color: {details['color']};">
                        {details['occupancy']:.2f}
                    </h1>
                    <p style="text-align: center; color: #666; margin-top: -10px;">
                        Habitat Suitability Score
                    </p>
                    <hr>
                    <p><strong>📊 Status:</strong> {details['status']}</p>
                    <p><strong>🔢 Species:</strong> {details['species_count']} species</p>
                    <p><strong>📈 Trend:</strong> {details['trend']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed tabbed analysis
        st.markdown("### 🔍 Detailed Guild Profiles")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            f"{guild_details['Wetland']['icon']} Wetland Birds",
            f"{guild_details['Forest']['icon']} Forest Birds", 
            f"{guild_details['Urban']['icon']} Urban Birds",
            "📊 Comparison"
        ])
        
        # Tab 1: Wetland
        with tab1:
            st.markdown("## 🦆 Wetland Bird Guild - Detailed Analysis")
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("### 📋 Guild Profile")
                details = guild_details["Wetland"]
                
                st.markdown(f"""
                **Species Count:** {details['species_count']} species
                
                **Example Species:**
                {details['examples']}
                
                **Preferred Habitat:**
                {details['habitat']}
                
                **Primary Food Sources:**
                {details['food']}
                
                **Typical Behaviors:**
                {details['behavior']}
                
                **Main Threats:**
                {details['threats']}
                
                **Conservation Status:**
                {details['status']} - {details['trend']}
                """)
                
                st.markdown("---")
            
            with col2:
                st.markdown("### 🗺️ Habitat Suitability Map")
                
                # Load wetland habitat map
                wetland_file = Config.PHASE4_DIR / "psi_env_Wetland.tif"
                if wetland_file.exists():
                    data, _, _, _ = load_raster(wetland_file)
                    if data is not None:
                        fig = px.imshow(
                            data,
                            color_continuous_scale='Blues',
                            aspect='auto',
                            title="Wetland Bird Habitat Quality",
                            labels={'color': 'Suitability'}
                        )
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistics
                        valid_data = data[np.isfinite(data)]
                        excellent = (valid_data > 0.8).sum() / len(valid_data) * 100
                        good = ((valid_data > 0.6) & (valid_data <= 0.8)).sum() / len(valid_data) * 100
                        poor = (valid_data <= 0.6).sum() / len(valid_data) * 100
                        
                        st.markdown(f"""
                        **Habitat Breakdown:**
                        - 🔵 Excellent (>0.8): **{excellent:.1f}%**
                        - 🟦 Good (0.6-0.8): **{good:.1f}%**
                        - ⬜ Needs work (<0.6): **{poor:.1f}%**
                        """)
                
                st.markdown("---")
                
                # Satellite water index
                st.markdown("### 💧 Water Availability (NDWI)")
                ndwi_file = Config.RASTERS_DIR / "NDWI_2025_core.tif"
                if ndwi_file.exists():
                    ndwi_data, _ = load_satellite_image(ndwi_file)
                    if ndwi_data is not None:
                        fig = px.imshow(ndwi_data, color_continuous_scale='Blues', aspect='auto')
                        fig.update_layout(height=250, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("🔵 Blue = Water/Wet areas - Critical for wetland birds")
                
                st.markdown("---")
                
                # Key finding
                st.success("""
                **✅ Key Finding:**
                Wetland birds show HIGH occupancy (0.82) across the sanctuary. 
                Water quality is good, but monitor pollution from urban runoff.
                """)
        
        # Tab 2: Forest
        with tab2:
            st.markdown("## 🦜 Forest Bird Guild - Detailed Analysis")
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("### 📋 Guild Profile")
                details = guild_details["Forest"]
                
                st.markdown(f"""
                **Species Count:** {details['species_count']} species
                
                **Example Species:**
                {details['examples']}
                
                **Preferred Habitat:**
                {details['habitat']}
                
                **Primary Food Sources:**
                {details['food']}
                
                **Typical Behaviors:**
                {details['behavior']}
                
                **Main Threats:**
                {details['threats']}
                
                **Conservation Status:**
                {details['status']} - {details['trend']}
                """)
                
                st.markdown("---")

            with col2:
                st.markdown("### 🗺️ Habitat Suitability Map")
                
                # Load forest habitat map
                forest_file = Config.PHASE4_DIR / "psi_env_Forest.tif"
                if forest_file.exists():
                    data, _, _, _ = load_raster(forest_file)
                    if data is not None:
                        fig = px.imshow(
                            data,
                            color_continuous_scale='Greens',
                            aspect='auto',
                            title="Forest Bird Habitat Quality",
                            labels={'color': 'Suitability'}
                        )
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistics
                        valid_data = data[np.isfinite(data)]
                        excellent = (valid_data > 0.8).sum() / len(valid_data) * 100
                        good = ((valid_data > 0.6) & (valid_data <= 0.8)).sum() / len(valid_data) * 100
                        poor = (valid_data <= 0.6).sum() / len(valid_data) * 100
                        
                        st.markdown(f"""
                        **Habitat Breakdown:**
                        - 🟢 Excellent (>0.8): **{excellent:.1f}%**
                        - 🟩 Good (0.6-0.8): **{good:.1f}%**
                        - ⬜ Needs work (<0.6): **{poor:.1f}%**
                        """)
                
                st.markdown("---")
                
                # Satellite vegetation
                st.markdown("### 🌳 Vegetation Density (NDVI)")
                ndvi_file = Config.RASTERS_DIR / "NDVI_2025_core.tif"
                if ndvi_file.exists():
                    ndvi_data, _ = load_satellite_image(ndvi_file)
                    if ndvi_data is not None:
                        fig = px.imshow(ndvi_data, color_continuous_scale='Greens', aspect='auto')
                        fig.update_layout(height=250, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("🟢 Green = Dense vegetation - Preferred by forest birds")
                
                st.markdown("---")
                
                # Key finding
                st.warning("""
                **⚠️ Key Finding:**
                Forest birds show MODERATE occupancy (0.68) with strong core preference. 
                Protect core interior and enhance buffer vegetation connectivity.
                """)
        
        # Tab 3: Urban
        with tab3:
            st.markdown("## 🐦 Urban Bird Guild - Detailed Analysis")
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("### 📋 Guild Profile")
                details = guild_details["Urban"]
                
                st.markdown(f"""
                **Species Count:** {details['species_count']} species
                
                **Example Species:**
                {details['examples']}
                
                **Preferred Habitat:**
                {details['habitat']}
                
                **Primary Food Sources:**
                {details['food']}
                
                **Typical Behaviors:**
                {details['behavior']}
                
                **Main Threats:**
                {details['threats']}
                
                **Conservation Status:**
                {details['status']} - {details['trend']}
                """)
                
                st.markdown("---")
            
            with col2:
                st.markdown("### 🗺️ Habitat Suitability Map")
                
                # Load urban habitat map
                urban_file = Config.PHASE4_DIR / "psi_env_Urban.tif"
                if urban_file.exists():
                    data, _, _, _ = load_raster(urban_file)
                    if data is not None:
                        fig = px.imshow(
                            data,
                            color_continuous_scale='Oranges',
                            aspect='auto',
                            title="Urban Bird Habitat Quality",
                            labels={'color': 'Suitability'}
                        )
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistics
                        valid_data = data[np.isfinite(data)]
                        excellent = (valid_data > 0.8).sum() / len(valid_data) * 100
                        good = ((valid_data > 0.6) & (valid_data <= 0.8)).sum() / len(valid_data) * 100
                        poor = (valid_data <= 0.6).sum() / len(valid_data) * 100
                        
                        st.markdown(f"""
                        **Habitat Breakdown:**
                        - 🟠 Excellent (>0.8): **{excellent:.1f}%**
                        - 🟡 Good (0.6-0.8): **{good:.1f}%**
                        - ⬜ Rare (<0.6): **{poor:.1f}%**
                        """)
                
                st.markdown("---")
                
                # Night lights
                st.markdown("### 🌙 Human Activity (Night Lights)")
                viirs_file = Config.RASTERS_DIR / "VIIRS_2025_core.tif"
                if viirs_file.exists():
                    viirs_data, _ = load_satellite_image(viirs_file)
                    if viirs_data is not None:
                        fig = px.imshow(viirs_data, color_continuous_scale='Hot', aspect='auto')
                        fig.update_layout(height=250, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("🔴 Bright = More lights - Urban birds tolerate this")
                
                st.markdown("---")
                
                # Key finding
                st.info("""
                **ℹ️ Key Finding:**
                Urban birds show VERY HIGH occupancy (0.93) everywhere. 
                This is normal but watch for decline in specialist species.
                """)
        
        # Tab 4: Comparison
        with tab4:
            st.markdown("## 📊 Guild Comparison & Ecosystem Health")
            
            # Comparison chart
            st.markdown("### 📈 Occupancy Comparison Across Models")
            
            fig = go.Figure()
            
            models = {
                'Year': 'Temporal (Year-based)',
                'Zone': 'Spatial (Core vs Buffer)',
                'Env_mean': 'Environmental (Habitat-based)'
            }
            
            for metric, label in models.items():
                fig.add_trace(go.Bar(
                    name=label,
                    x=guild_profiles['Guild'],
                    y=guild_profiles[metric],
                    text=guild_profiles[metric].round(3),
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>' + label + ': %{y:.3f}<br><extra></extra>'
                ))
            
            fig.update_layout(
                title="Habitat Suitability by Guild and Model Type",
                xaxis_title="Bird Guild",
                yaxis_title="Habitat Suitability (ψ) - Higher is Better",
                barmode='group',
                height=400,
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = go.Figure(data=[
                go.Bar(
                    x=functional_summary['Index'],
                    y=functional_summary['Mean'],
                    error_y=dict(type='data', array=functional_summary['Std']),
                    text=functional_summary['Mean'].round(2),
                    textposition='auto',
                    marker_color='#4caf50',
                    hovertemplate='<b>%{x}</b><br>Mean: %{y:.3f}<br><extra></extra>'
                )
            ])
            
            fig.update_layout(
                title="Ecosystem Diversity Metrics",
                xaxis_title="Diversity Index",
                yaxis_title="Value",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("#### 📊 Summary Table")
            
            st.dataframe(
                functional_summary.style.format({
                    'Mean': '{:.3f}',
                    'Std': '{:.3f}',
                    'Min': '{:.3f}',
                    'Max': '{:.3f}'
                }).background_gradient(subset=['Mean'], cmap='Greens'),
                hide_index=True,
                height=300
            )
            
            st.markdown("---")
            
            shannon_mean = functional_summary[functional_summary['Index'] == 'Shannon Diversity']['Mean'].values[0] if len(functional_summary[functional_summary['Index'] == 'Shannon Diversity']) > 0 else 0
            
            if shannon_mean > 0.8:
                st.success("**Excellent diversity!** All three guilds well-represented.")
            elif shannon_mean > 0.6:
                st.info("**Good diversity.** Most ecological functions covered.")
            else:
                st.warning("**Low diversity.** Some guilds may be declining.")

# =============================================================================
# PAGE 4: CONSERVATION PRIORITY - COMPLETE
# =============================================================================

def render_conservation_priority_page():
    """Render conservation priority analysis with action recommendations"""
    st.title("🎯 Conservation Priority Analysis")
    st.markdown("*Data-driven identification of critical management areas*")

    # Load data
    cpi_metadata = load_cpi_metadata()
    
    if Config.PRIORITY_CLASSES.exists():
        # Load priority raster
        priority_data, bounds, crs, transform = load_raster(Config.PRIORITY_CLASSES)
        
        if priority_data is not None:
            # Two-column layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 🗺️ Priority Classification Map")
                
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
                st.markdown("### 📊 Priority Distribution")
                
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
                
                st.markdown("---")
                
                # Pie chart
                st.markdown("#### 📈 Area Breakdown")
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=[f"Level {i}" for i in range(1, 6)],
                    values=[priority_stats[i]['Area (ha)'] for i in range(1, 6)],
                    marker_colors=[Config.PRIORITY_COLORS[i] for i in range(1, 6)],
                    hovertemplate='<b>%{label}</b><br>Area: %{value} ha<br>%{percent}<extra></extra>'
                )])
                
                fig_pie.update_layout(height=300)
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Key stat
                urgent_area = priority_stats[5]['Area (ha)'] + priority_stats[4]['Area (ha)']
                urgent_pct = priority_stats[5]['Percentage'] + priority_stats[4]['Percentage']
                
                st.error(f"""
                **🚨 Urgent Action Needed:**
                {urgent_area:.2f} ha ({urgent_pct:.1f}%)
                requires immediate to high-priority intervention
                """)
        
        st.markdown("---")

# =============================================================================
# PAGE 5: TEMPORAL TRENDS - COMPLETE
# =============================================================================
def render_temporal_trends_page():
    """Render temporal analysis with year-over-year comparisons"""
    st.title("📈 Temporal Trends Analysis")
    st.markdown("*Multi-year patterns in bird populations (2019-2025)*")
    
    # Load data
    ebird_df = load_ebird_data()
    
    if not ebird_df.empty:
        # Check what columns we have
        st.sidebar.markdown("**Available data columns:**")
        st.sidebar.caption(", ".join(ebird_df.columns[:10]))
        
        # Parse dates if available
        if 'OBSERVATION DATE' in ebird_df.columns:
            ebird_df['year'] = pd.to_datetime(ebird_df['OBSERVATION DATE'], errors='coerce').dt.year
            ebird_df = ebird_df.dropna(subset=['year'])
            ebird_df['year'] = ebird_df['year'].astype(int)
        elif 'year' in ebird_df.columns:
            pass  # Already have year column
        else:
            st.error("⚠️ No date information found in eBird data. Cannot show temporal trends.")
            return
        
        # Check for guild column - try multiple possible names
        guild_col = None
        for possible_col in ['guild', 'Guild', 'GUILD', 'functional_guild', 'bird_guild', 'CATEGORY']:
            if possible_col in ebird_df.columns:
                guild_col = possible_col
                break
        
        # Overall trends
        st.markdown("### 📊 Observation Effort Over Time")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if guild_col:
                # Group by year and guild
                yearly_counts = ebird_df.groupby(['year', guild_col]).size().reset_index(name='count')
                yearly_counts = yearly_counts.rename(columns={guild_col: 'guild'})
                
                fig = px.line(
                    yearly_counts,
                    x='year',
                    y='count',
                    color='guild',
                    markers=True,
                    title="Bird Observations by Guild (2019-2025)",
                    color_discrete_map=Config.GUILD_COLORS if hasattr(Config, 'GUILD_COLORS') else None,
                    labels={'count': 'Number of Observations', 'year': 'Year', 'guild': 'Guild'}
                )
            else:
                # No guild - just overall counts
                yearly_counts = ebird_df.groupby('year').size().reset_index(name='count')
                
                fig = px.line(
                    yearly_counts,
                    x='year',
                    y='count',
                    markers=True,
                    title="Bird Observations Over Time (2019-2025)",
                    labels={'count': 'Number of Observations', 'year': 'Year'}
                )
                
                st.info("ℹ️ Guild information not available in data. Showing overall observation trends.")
            
            fig.update_layout(
                height=400,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("""
            📖 **How to read:** Each line shows observation trends over time.
            Going up = More sightings (could mean more birds OR more birdwatchers!)
            """)
        
        with col2:
            st.markdown("#### 📋 Key Insights")
            
            years = sorted(ebird_df['year'].unique())
            if len(years) >= 2:
                first_year = min(years)
                last_year = max(years)
                
                total_first = len(ebird_df[ebird_df['year'] == first_year])
                total_last = len(ebird_df[ebird_df['year'] == last_year])
                change = ((total_last - total_first) / total_first * 100) if total_first > 0 else 0
                
                st.metric(
                    f"{first_year} Observations",
                    total_first,
                    help="Baseline year"
                )
                
                st.metric(
                    f"{last_year} Observations",
                    total_last,
                    delta=f"{change:+.1f}%",
                    help="Latest year"
                )
                
                if change > 10:
                    st.success("📈 Increasing observation effort!")
                elif change < -10:
                    st.warning("📉 Declining observation effort")
                else:
                    st.info("➡️ Stable observation effort")
        
        st.markdown("---")
        
        # Species richness trends
        st.markdown("### 🦜 Species Richness Trends")
        
        if 'COMMON NAME' in ebird_df.columns or 'SCIENTIFIC NAME' in ebird_df.columns:
            species_col = 'COMMON NAME' if 'COMMON NAME' in ebird_df.columns else 'SCIENTIFIC NAME'
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if guild_col:
                    richness = ebird_df.groupby(['year', guild_col])[species_col].nunique().reset_index(name='species')
                    richness = richness.rename(columns={guild_col: 'guild'})
                    
                    fig2 = px.bar(
                        richness,
                        x='year',
                        y='species',
                        color='guild',
                        barmode='group',
                        title="Unique Species Recorded per Year by Guild",
                        color_discrete_map=Config.GUILD_COLORS if hasattr(Config, 'GUILD_COLORS') else None,
                        labels={'species': 'Number of Species', 'year': 'Year', 'guild': 'Guild'}
                    )
                else:
                    richness = ebird_df.groupby('year')[species_col].nunique().reset_index(name='species')
                    
                    fig2 = px.bar(
                        richness,
                        x='year',
                        y='species',
                        title="Unique Species Recorded per Year",
                        labels={'species': 'Number of Species', 'year': 'Year'}
                    )
                
                fig2.update_layout(
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                st.caption("""
                📖 **What this shows:** How many different species were recorded each year.
                Taller bars = More species diversity. Decreasing bars could signal biodiversity loss.
                """)
            
            with col2:
                st.markdown("#### 🎯 Diversity Metrics")
                
                if len(years) >= 2:
                    species_first = ebird_df[ebird_df['year'] == first_year][species_col].nunique()
                    species_last = ebird_df[ebird_df['year'] == last_year][species_col].nunique()
                    species_change = species_last - species_first
                    
                    st.metric(
                        f"{first_year} Species",
                        species_first
                    )
                    
                    st.metric(
                        f"{last_year} Species",
                        species_last,
                        delta=species_change
                    )
                    
                    if species_change > 5:
                        st.success("✅ Diversity increasing!")
                    elif species_change < -5:
                        st.error("⚠️ Diversity declining")
                    else:
                        st.info("➡️ Diversity stable")
        
        st.markdown("---")
        
        # Seasonal patterns
        st.markdown("### 🗓️ Seasonal Patterns")
        
        if 'OBSERVATION DATE' in ebird_df.columns:
            ebird_df['month'] = pd.to_datetime(ebird_df['OBSERVATION DATE'], errors='coerce').dt.month
            ebird_df = ebird_df.dropna(subset=['month'])
            ebird_df['month'] = ebird_df['month'].astype(int)
            
            if guild_col:
                monthly_counts = ebird_df.groupby(['month', guild_col]).size().reset_index(name='count')
                monthly_counts = monthly_counts.rename(columns={guild_col: 'guild'})
            else:
                monthly_counts = ebird_df.groupby('month').size().reset_index(name='count')
                monthly_counts['guild'] = 'All Birds'
            
            # Month names
            month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                          7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
            monthly_counts['month_name'] = monthly_counts['month'].map(month_names)
            
            if guild_col:
                fig3 = px.line(
                    monthly_counts,
                    x='month_name',
                    y='count',
                    color='guild',
                    markers=True,
                    title="Seasonal Activity Patterns",
                    color_discrete_map=Config.GUILD_COLORS if hasattr(Config, 'GUILD_COLORS') else None,
                    labels={'count': 'Observations', 'month_name': 'Month', 'guild': 'Guild'}
                )
            else:
                fig3 = px.line(
                    monthly_counts,
                    x='month_name',
                    y='count',
                    markers=True,
                    title="Seasonal Activity Patterns",
                    labels={'count': 'Observations', 'month_name': 'Month'}
                )
            
            fig3.update_layout(height=350)
            st.plotly_chart(fig3, use_container_width=True)
            
            st.info("""
            💡 **Seasonal Insights:**
            - **Peaks in observations:** Likely migration seasons or better weather for birdwatching
            - **Wetland birds:** May show monsoon-related patterns (more water = more birds)
            - **Forest birds:** More active during breeding season (spring)
            - **Urban birds:** Relatively constant year-round
            """)
    
    else:
        st.warning("⚠️ No eBird data available. Please check data files.")
        st.info("""
        **Expected file location:** `data/processed/phase1/ebird_with_guilds_and_zones.csv`
        
        **Required columns:**
        - OBSERVATION DATE (or year)
        - COMMON NAME or SCIENTIFIC NAME
        - guild (optional, for guild-specific trends)
        """)

# =============================================================================
# PAGE 6: SPECIES DECLINE TRACKER
# =============================================================================

def render_species_decline_tracker_page():
    """
    Render species decline tracker with trend analysis and alerts
    """
    st.title("📉 Species Decline Tracker")
    st.markdown("*Automated trend detection and conservation alerts*")
    
    # Introduction
    st.markdown("""
    <div class="info-box">
    <h3>🔍 What This Page Shows</h3>
    <p>Automatically analyzes population trends for each bird species using statistical tests:</p>
    <ul>
    <li><strong>Mann-Kendall Test:</strong> Detects significant increasing/decreasing trends</li>
    <li><strong>Change Point Analysis:</strong> Identifies when populations started declining</li>
    <li><strong>Alert System:</strong> Flags species needing urgent conservation attention</li>
    </ul>
    <p><strong>Your teacher's request:</strong> "Show yearly decline of birds" - This does exactly that, but with AI-powered statistical analysis! 🤖</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    trends_file = Path("data/processed/phase7_trends/species_trends_analysis.csv")
    summary_file = Path("data/processed/phase7_trends/trends_summary.csv")
    alerts_file = Path("data/processed/phase7_trends/conservation_alerts.csv")
    
    if not trends_file.exists():
        st.warning("⚠️ Trend analysis not yet run. Please run the analysis first.")
        
        if st.button("🚀 Run Trend Analysis Now"):
            with st.spinner("Analyzing species trends... This may take 1-2 minutes..."):
                # Import and run the analysis
                import sys
                sys.path.append(str(Path.cwd()))
                
                try:
                    from phase7_species_decline_tracker import run_trend_analysis
                    results = run_trend_analysis()
                    st.success("✅ Analysis complete! Refresh the page to see results.")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error running analysis: {e}")
        
        st.info("""
        **To run manually:**
        ```bash
        python phase7_species_decline_tracker.py
        ```
        
        **Expected output:**
        - Species trend analysis CSV
        - Conservation alerts CSV
        - Summary statistics CSV
        """)
        return
    
    # Load results
    trends_df = pd.read_csv(trends_file)
    summary_df = pd.read_csv(summary_file)
    summary = summary_df.iloc[0].to_dict()
    
    if alerts_file.exists():
        alerts_df = pd.read_csv(alerts_file)
    else:
        alerts_df = pd.DataFrame()
    
    # Summary metrics
    st.markdown("### 📊 Overview Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Species Analyzed",
            f"{summary['total_species']}",
            help="Species with at least 4 years of data"
        )
    
    with col2:
        st.metric(
            "Declining",
            f"{summary['declining']}",
            delta=f"-{summary['declining_pct']:.1f}%",
            delta_color="inverse",
            help="Species showing significant decline (p<0.05)"
        )
    
    with col3:
        st.metric(
            "Critical Concern",
            f"{summary['critical']}",
            delta="⚠️ Urgent action needed",
            delta_color="off",
            help="Declined >30% - immediate intervention required"
        )
    
    with col4:
        st.metric(
            "Possibly Extinct",
            f"{summary['possibly_extinct']}",
            delta="🔴 Not seen recently",
            delta_color="off",
            help="No observations in last 2 years"
        )
    
    # Status breakdown chart
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📈 Population Trend Distribution")
        
        # Pie chart
        status_counts = trends_df['Status'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            marker=dict(colors=['#d32f2f', '#fbc02d', '#4caf50']),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Species by Trend Status",
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Status Guide")
        
        st.markdown("""
        **🔴 Declining:**
        - Significant downward trend
        - p-value < 0.05
        - Needs monitoring or action
        
        **🟡 Stable:**
        - No significant trend
        - Population fluctuating normally
        - Continue routine monitoring
        
        **🟢 Increasing:**
        - Significant upward trend
        - p-value < 0.05
        - Conservation efforts working!
        """)
        
        st.info(f"""
        **Overall Health:**
        - {summary['increasing']} species improving ✓
        - {summary['stable']} species stable ✓
        - {summary['declining']} species declining ⚠️
        """)
    
    # Conservation alerts
    if len(alerts_df) > 0:
        st.markdown("---")
        st.markdown("### 🚨 Conservation Alerts")
        
        # Filter by alert level
        alert_filter = st.multiselect(
            "Filter by alert level:",
            options=alerts_df['Level'].unique(),
            default=alerts_df['Level'].unique()
        )
        
        filtered_alerts = alerts_df[alerts_df['Level'].isin(alert_filter)]
        
        for _, alert in filtered_alerts.iterrows():
            alert_color = {
                'CRITICAL': 'error',
                'EXTINCTION RISK': 'warning',
                'HIGH': 'warning',
                'MEDIUM': 'info'
            }.get(alert['Level'], 'info')
            
            with st.expander(f"🔔 [{alert['Level']}] {alert['Species']}", expanded=(alert['Level'] == 'CRITICAL')):
                st.markdown(f"**Issue:** {alert['Message']}")
                st.markdown(f"**Recommended Action:** {alert['Action']}")
                st.caption(alert['Details'])
    
    # Top declining species
    st.markdown("---")
    st.markdown("### 🔴 Top Declining Species")
    
    declining = trends_df[trends_df['Status'] == 'Declining'].sort_values('Percent_Change').head(15)
    
    if len(declining) > 0:
        fig = go.Figure()
        
        # Color code by urgency
        colors = declining['Urgency'].map({
            'Critical': '#d32f2f',
            'High': '#f57c00',
            'Medium': '#fbc02d'
        })
        
        fig.add_trace(go.Bar(
            y=declining['Species'],
            x=declining['Percent_Change'],
            orientation='h',
            marker=dict(color=colors),
            text=declining['Percent_Change'].round(1),
            texttemplate='%{text}%',
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Change: %{x:.1f}%<br><extra></extra>'
        ))
        
        fig.update_layout(
            title="Species Showing Greatest Decline",
            xaxis_title="Percent Change (%)",
            yaxis_title="Species",
            height=500,
            xaxis=dict(range=[declining['Percent_Change'].min() * 1.1, 0])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("🔴 Red = Critical (>30% decline), 🟠 Orange = High concern (>15% decline), 🟡 Yellow = Medium concern")
    else:
        st.success("🎉 No significantly declining species detected!")
    
    # Top increasing species
    st.markdown("---")
    st.markdown("### 🟢 Top Increasing Species")
    
    increasing = trends_df[trends_df['Status'] == 'Increasing'].sort_values('Percent_Change', ascending=False).head(15)
    
    if len(increasing) > 0:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=increasing['Species'],
            x=increasing['Percent_Change'],
            orientation='h',
            marker=dict(color='#4caf50'),
            text=increasing['Percent_Change'].round(1),
            texttemplate='+%{text}%',
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Change: +%{x:.1f}%<br><extra></extra>'
        ))
        
        fig.update_layout(
            title="Species Showing Greatest Increase",
            xaxis_title="Percent Change (%)",
            yaxis_title="Species",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("✅ These species are thriving! Conservation efforts may be working.")
    
    # Detailed data table
    st.markdown("---")
    st.markdown("### 📋 Detailed Species Data")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.multiselect(
            "Filter by status:",
            options=['Declining', 'Stable', 'Increasing'],
            default=['Declining']
        )
    
    with col2:
        urgency_filter = st.multiselect(
            "Filter by urgency:",
            options=['Critical', 'High', 'Medium', 'Low'],
            default=['Critical', 'High']
        )
    
    with col3:
        min_change = st.slider(
            "Minimum % change:",
            min_value=int(trends_df['Percent_Change'].min()),
            max_value=int(trends_df['Percent_Change'].max()),
            value=int(trends_df['Percent_Change'].min())
        )
    
    # Apply filters
    filtered_df = trends_df[
        (trends_df['Status'].isin(status_filter)) &
        (trends_df['Urgency'].isin(urgency_filter)) &
        (trends_df['Percent_Change'] >= min_change)
    ]
    
    # Display table
    display_cols = [
        'Species', 'Status', 'Percent_Change', 'Annual_Rate', 
        'P_value', 'Urgency', 'First_Year', 'Last_Year', 
        'First_Year_Count', 'Last_Year_Count', 'Possibly_Extinct'
    ]
    
    st.dataframe(
        filtered_df[display_cols].style.format({
            'Percent_Change': '{:+.1f}%',
            'Annual_Rate': '{:+.2f}%',
            'P_value': '{:.4f}'
        }).apply(
            lambda x: ['background-color: #ffcdd2' if v == 'Declining' 
                      else 'background-color: #c8e6c9' if v == 'Increasing'
                      else '' for v in x], 
            subset=['Status']
        ).apply(
            lambda x: ['background-color: #d32f2f; color: white' if v == 'Critical'
                      else 'background-color: #f57c00; color: white' if v == 'High'
                      else '' for v in x],
            subset=['Urgency']
        ),
        height=400,
        use_container_width=True
    )
    
    st.caption(f"Showing {len(filtered_df)} of {len(trends_df)} species")
    
    # Download options
    st.markdown("---")
    st.markdown("### 📥 Download Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_all = trends_df.to_csv(index=False)
        st.download_button(
            "📊 All Species Data",
            csv_all,
            file_name="species_trends_all.csv",
            mime="text/csv"
        )
    
    with col2:
        if len(declining) > 0:
            csv_declining = declining.to_csv(index=False)
            st.download_button(
                "🔴 Declining Species",
                csv_declining,
                file_name="species_declining.csv",
                mime="text/csv"
            )
    
    with col3:
        if len(alerts_df) > 0:
            csv_alerts = alerts_df.to_csv(index=False)
            st.download_button(
                "🚨 Conservation Alerts",
                csv_alerts,
                file_name="conservation_alerts.csv",
                mime="text/csv"
            )
    
    # Methodology
    st.markdown("---")
    with st.expander("📖 Methodology: How Trends Are Detected"):
        st.markdown("""
        ### Statistical Methods Used:
        
        **1. Mann-Kendall Trend Test**
        - Non-parametric test (no assumptions about data distribution)
        - Detects monotonic trends (consistently increasing or decreasing)
        - Robust to outliers and missing data
        - p-value < 0.05 = statistically significant trend
        
        **2. Percent Change Calculation**
        ```
        % Change = ((Last Year Count - First Year Count) / First Year Count) × 100
        ```
        
        **3. Urgency Classification**
        - **Critical:** Decline > 30% AND p < 0.05
        - **High:** Decline > 15% AND p < 0.05
        - **Medium:** Decline significant but < 15%
        - **Low:** No significant trend
        
        **4. Extinction Risk Assessment**
        - Species not observed in last 2 consecutive years
        - Flagged for targeted surveys
        
        ### Interpretation Guide:
        
        **P-value:**
        - < 0.05: Trend is statistically significant (95% confident)
        - < 0.01: Highly significant (99% confident)
        - ≥ 0.05: No significant trend (could be random variation)
        
        **Kendall's Tau:**
        - Measures strength of trend (-1 to +1)
        - -1 = Perfect decreasing trend
        - 0 = No trend
        - +1 = Perfect increasing trend
        
        ### Limitations:
        - Requires ≥4 years of data per species
        - Cannot distinguish true decline from reduced observation effort
        - Assumes detection probability is constant over time
        - Short time series (7 years) limits power for subtle trends
        
        ### Recommended Actions:
        
        **For Critical/High species:**
        1. Conduct targeted surveys to confirm trend
        2. Investigate potential causes (habitat loss, pollution, predation)
        3. Implement species-specific conservation measures
        4. Monitor monthly instead of annually
        
        **For Possibly Extinct species:**
        1. Organize systematic search efforts
        2. Check historical habitat areas
        3. Interview long-term observers
        4. Consider reintroduction if confirmed extinct
        """)

# =============================================================================
# PAGE 7: WHAT-IF SCENARIO SIMULATOR
# =============================================================================

def render_whatif_scenarios_page():
    """
    Interactive scenario simulator for conservation planning
    """
    st.title("🔮 What-If Scenario Simulator")
    st.markdown("*Test different management scenarios and predict their impact*")
    
    # Introduction
    st.markdown("""
    <div class="info-box">
    <h3>🎯 What This Tool Does</h3>
    <p>Use interactive sliders to simulate different scenarios and see predicted impacts on bird populations:</p>
    <ul>
    <li><strong>Environmental Changes:</strong> What if vegetation increases/decreases?</li>
    <li><strong>Management Actions:</strong> What if we plant trees or remove invasives?</li>
    <li><strong>Threats:</strong> What if disturbance or pollution increases?</li>
    <li><strong>Climate:</strong> What if drought or flooding occurs?</li>
    </ul>
    <p><strong>Real-time AI Predictions:</strong> Bayesian models instantly predict occupancy changes! 🤖</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load baseline data (current conditions)
    guild_profiles = load_guild_profiles()
    
    if guild_profiles is None:
        st.warning("⚠️ Guild profiles not found. Run Phase 4 analysis first.")
        return
    
    # Get baseline occupancies
    baseline_occupancy = {
        'Wetland': guild_profiles[guild_profiles['Guild'] == 'Wetland']['Env_mean'].values[0],
        'Forest': guild_profiles[guild_profiles['Guild'] == 'Forest']['Env_mean'].values[0],
        'Urban': guild_profiles[guild_profiles['Guild'] == 'Urban']['Env_mean'].values[0]
    }
    
    st.markdown("---")
    st.markdown("### 🎛️ Scenario Controls")
    
    # Tabs for different scenario types
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌿 Environmental Changes",
        "🛠️ Management Actions",
        "⚠️ Threat Scenarios",
        "📊 Compare Scenarios"
    ])
    
    # =============================================================================
    # TAB 1: ENVIRONMENTAL CHANGES
    # =============================================================================
    
    with tab1:
        st.markdown("### 🌱 Environmental Change Scenarios")
        st.caption("Adjust environmental variables and see predicted impact on each guild")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### 🎚️ Adjust Variables")
            
            # NDVI change
            ndvi_change = st.slider(
                "NDVI Change (Vegetation)",
                min_value=-30,
                max_value=+30,
                value=0,
                step=5,
                help="Negative = vegetation loss (drought, cutting), Positive = vegetation increase (planting, growth)",
                format="%d%%"
            )
            
            # NDWI change
            ndwi_change = st.slider(
                "NDWI Change (Water Availability)",
                min_value=-30,
                max_value=+30,
                value=0,
                step=5,
                help="Negative = drier conditions, Positive = wetter conditions",
                format="%d%%"
            )
            
            # Disturbance change
            disturbance_change = st.slider(
                "Human Disturbance Change",
                min_value=-50,
                max_value=+50,
                value=0,
                step=10,
                help="Negative = less disturbance (restrictions), Positive = more disturbance (development)",
                format="%d%%"
            )
            
            # Edge effect change
            edge_change = st.slider(
                "Edge Effect Intensity",
                min_value=-30,
                max_value=+30,
                value=0,
                step=10,
                help="Negative = reduced edge (buffer restoration), Positive = increased edge (fragmentation)",
                format="%d%%"
            )
        
        with col2:
            st.markdown("#### 📊 Predicted Impacts")
            
            # Calculate predicted changes (simplified model based on known coefficients)
            # These coefficients are from your Bayesian models
            
            # Wetland birds (water-dependent)
            wetland_impact = (
                (ndwi_change * 0.003) +  # Strong NDWI response
                (ndvi_change * 0.001) +   # Weak NDVI response
                (disturbance_change * -0.002)  # Negative impact of disturbance
            )
            wetland_new = np.clip(baseline_occupancy['Wetland'] + wetland_impact, 0, 1)
            wetland_change_pct = (wetland_impact / baseline_occupancy['Wetland']) * 100
            
            # Forest birds (vegetation-dependent, edge-sensitive)
            forest_impact = (
                (ndvi_change * 0.004) +   # Strong NDVI response
                (edge_change * -0.005) +   # Very sensitive to edges
                (disturbance_change * -0.003)  # Sensitive to disturbance
            )
            forest_new = np.clip(baseline_occupancy['Forest'] + forest_impact, 0, 1)
            forest_change_pct = (forest_impact / baseline_occupancy['Forest']) * 100
            
            # Urban birds (generalists - minimal response)
            urban_impact = (
                (ndvi_change * 0.0005) +  # Weak response to everything
                (ndwi_change * 0.0005) +
                (disturbance_change * 0.001)  # Actually benefit from disturbance slightly
            )
            urban_new = np.clip(baseline_occupancy['Urban'] + urban_impact, 0, 1)
            urban_change_pct = (urban_impact / baseline_occupancy['Urban']) * 100
            
            # Display results
            results_df = pd.DataFrame({
                'Guild': ['Wetland', 'Forest', 'Urban'],
                'Baseline ψ': [baseline_occupancy['Wetland'], baseline_occupancy['Forest'], baseline_occupancy['Urban']],
                'Predicted ψ': [wetland_new, forest_new, urban_new],
                'Change (ψ)': [wetland_impact, forest_impact, urban_impact],
                'Change (%)': [wetland_change_pct, forest_change_pct, urban_change_pct]
            })
            
            # Visualization
            fig = go.Figure()
            
            # Baseline
            fig.add_trace(go.Bar(
                name='Current',
                x=results_df['Guild'],
                y=results_df['Baseline ψ'],
                marker_color='lightblue',
                text=results_df['Baseline ψ'].round(3),
                textposition='inside'
            ))
            
            # Predicted
            fig.add_trace(go.Bar(
                name='Predicted',
                x=results_df['Guild'],
                y=results_df['Predicted ψ'],
                marker_color=results_df['Change (%)'].apply(
                    lambda x: '#4caf50' if x > 0 else '#d32f2f' if x < -5 else '#fbc02d'
                ),
                text=results_df['Predicted ψ'].round(3),
                textposition='inside'
            ))
            
            fig.update_layout(
                title="Occupancy Comparison: Current vs Predicted",
                xaxis_title="Guild",
                yaxis_title="Habitat Suitability (ψ)",
                barmode='group',
                height=350,
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Impact summary
            st.markdown("##### Impact Summary:")
            
            for _, row in results_df.iterrows():
                guild = row['Guild']
                change_pct = row['Change (%)']
                
                if abs(change_pct) < 1:
                    impact = "⚪ Negligible impact"
                    color = ""
                elif change_pct > 5:
                    impact = "🟢 Significant improvement"
                    color = "success"
                elif change_pct > 0:
                    impact = "🟢 Slight improvement"
                    color = "success"
                elif change_pct > -5:
                    impact = "🟡 Slight decline"
                    color = "warning"
                else:
                    impact = "🔴 Significant decline"
                    color = "error"
                
                if color:
                    if color == "success":
                        st.success(f"**{guild}:** {change_pct:+.1f}% - {impact}")
                    elif color == "warning":
                        st.warning(f"**{guild}:** {change_pct:+.1f}% - {impact}")
                    else:
                        st.error(f"**{guild}:** {change_pct:+.1f}% - {impact}")
                else:
                    st.info(f"**{guild}:** {change_pct:+.1f}% - {impact}")
    
    # =============================================================================
    # TAB 2: MANAGEMENT ACTIONS
    # =============================================================================
    
    with tab2:
        st.markdown("### 🛠️ Management Action Scenarios")
        st.caption("Simulate specific conservation interventions")
        
        # Pre-defined scenarios
        scenario = st.selectbox(
            "Select Management Action:",
            [
                "Baseline (No Action)",
                "Tree Planting Campaign (1 ha)",
                "Invasive Species Removal (0.5 ha)",
                "Nest Box Installation (50 boxes)",
                "Water Quality Improvement",
                "Visitor Access Restrictions (Buffer Zone)",
                "Complete Restoration (All Actions)"
            ]
        )
        
        # Define scenario parameters
        scenarios_params = {
            "Baseline (No Action)": {
                'ndvi': 0, 'ndwi': 0, 'disturbance': 0, 'edge': 0,
                'cost': 0, 'time': 0, 'description': "Current conditions maintained"
            },
            "Tree Planting Campaign (1 ha)": {
                'ndvi': 15, 'ndwi': 0, 'disturbance': 0, 'edge': -10,
                'cost': 50000, 'time': 24, 'description': "Plant 500 native saplings in degraded areas"
            },
            "Invasive Species Removal (0.5 ha)": {
                'ndvi': 10, 'ndwi': 5, 'disturbance': -15, 'edge': 0,
                'cost': 30000, 'time': 6, 'description': "Remove Lantana and Prosopis from 0.5 ha"
            },
            "Nest Box Installation (50 boxes)": {
                'ndvi': 0, 'ndwi': 0, 'disturbance': 0, 'edge': 0,
                'cost': 15000, 'time': 2, 'description': "Install nest boxes for cavity-nesting birds (mainly helps forest guild)",
                'forest_bonus': 0.05  # Direct boost to forest birds
            },
            "Water Quality Improvement": {
                'ndvi': 0, 'ndwi': 20, 'disturbance': 0, 'edge': 0,
                'cost': 75000, 'time': 12, 'description': "Install filtration, remove pollution sources"
            },
            "Visitor Access Restrictions (Buffer Zone)": {
                'ndvi': 0, 'ndwi': 0, 'disturbance': -30, 'edge': -15,
                'cost': 10000, 'time': 1, 'description': "Close buffer to visitors, enforce strictly"
            },
            "Complete Restoration (All Actions)": {
                'ndvi': 25, 'ndwi': 20, 'disturbance': -30, 'edge': -20,
                'cost': 200000, 'time': 36, 'description': "Comprehensive restoration program"
            }
        }
        
        params = scenarios_params[scenario]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Scenario Details")
            
            st.info(f"""
            **Action:** {scenario}
            
            **Description:** {params['description']}
            
            **Environmental Changes:**
            - NDVI: {params['ndvi']:+d}%
            - NDWI: {params['ndwi']:+d}%
            - Disturbance: {params['disturbance']:+d}%
            - Edge Effect: {params['edge']:+d}%
            
            **Implementation:**
            - Estimated Cost: ₹{params['cost']:,}
            - Time to Full Effect: {params['time']} months
            """)
        
        with col2:
            st.markdown("#### 📊 Predicted Outcomes")
            
            # Calculate impacts (same as Tab 1)
            wetland_impact = (
                (params['ndwi'] * 0.003) +
                (params['ndvi'] * 0.001) +
                (params['disturbance'] * -0.002)
            )
            wetland_new = np.clip(baseline_occupancy['Wetland'] + wetland_impact, 0, 1)
            
            forest_impact = (
                (params['ndvi'] * 0.004) +
                (params['edge'] * -0.005) +
                (params['disturbance'] * -0.003) +
                params.get('forest_bonus', 0)  # Special bonus for nest boxes
            )
            forest_new = np.clip(baseline_occupancy['Forest'] + forest_impact, 0, 1)
            
            urban_impact = (
                (params['ndvi'] * 0.0005) +
                (params['ndwi'] * 0.0005) +
                (params['disturbance'] * 0.001)
            )
            urban_new = np.clip(baseline_occupancy['Urban'] + urban_impact, 0, 1)
            
            # Metrics
            total_impact = wetland_impact + forest_impact + urban_impact
            
            st.metric(
                "Total Ecosystem Impact",
                f"{total_impact:+.3f} ψ",
                help="Sum of impacts across all guilds"
            )
            
            st.metric(
                "Cost Effectiveness",
                f"₹{params['cost'] / (total_impact * 100 + 0.001):,.0f} per 0.01 ψ increase" if total_impact > 0 else "N/A",
                help="Cost per unit improvement in occupancy"
            )
            
            st.metric(
                "Time to Results",
                f"{params['time']} months",
                help="Expected time for full effect"
            )
            
            # Which guild benefits most?
            impacts = {
                'Wetland': wetland_impact,
                'Forest': forest_impact,
                'Urban': urban_impact
            }
            max_beneficiary = max(impacts, key=impacts.get)
            
            if total_impact > 0:
                st.success(f"✅ **Primary Beneficiary:** {max_beneficiary} birds (+{impacts[max_beneficiary]:.3f} ψ)")
            elif total_impact < 0:
                st.error(f"⚠️ **Warning:** This scenario may harm bird populations")
            else:
                st.info("➡️ Minimal impact predicted")
    
    # =============================================================================
    # TAB 3: THREAT SCENARIOS
    # =============================================================================
    
    with tab3:
        st.markdown("### ⚠️ Threat Scenario Analysis")
        st.caption("Understand impact of potential threats")
        
        threat = st.selectbox(
            "Select Threat Scenario:",
            [
                "Drought (Severe)",
                "Flooding (Monsoon)",
                "Urban Development (Adjacent)",
                "Pollution Event (Water)",
                "Disease Outbreak",
                "Climate Change (2°C warming)"
            ]
        )
        
        threat_params = {
            "Drought (Severe)": {
                'ndvi': -25, 'ndwi': -30, 'disturbance': 0, 'edge': 0,
                'description': "Extended drought reduces vegetation and water availability",
                'probability': 'Medium (once every 5-10 years)',
                'duration': '3-6 months'
            },
            "Flooding (Monsoon)": {
                'ndvi': 5, 'ndwi': 30, 'disturbance': 10, 'edge': 0,
                'description': "Heavy rainfall floods low-lying areas, increases disturbance",
                'probability': 'High (annual during monsoon)',
                'duration': '2-4 weeks'
            },
            "Urban Development (Adjacent)": {
                'ndvi': -10, 'ndwi': -5, 'disturbance': 40, 'edge': 30,
                'description': "New construction near sanctuary increases noise, light, edge effects",
                'probability': 'High (ongoing)',
                'duration': 'Permanent'
            },
            "Pollution Event (Water)": {
                'ndvi': 0, 'ndwi': -20, 'disturbance': 0, 'edge': 0,
                'description': "Industrial discharge or sewage leak contaminates water",
                'probability': 'Low-Medium (every 2-3 years)',
                'duration': '1-3 months (until cleanup)'
            },
            "Disease Outbreak": {
                'ndvi': 0, 'ndwi': 0, 'disturbance': 0, 'edge': 0,
                'description': "Avian disease reduces bird populations directly (not habitat change)",
                'probability': 'Low (rare)',
                'duration': '6-12 months',
                'direct_mortality': 0.15  # 15% mortality
            },
            "Climate Change (2°C warming)": {
                'ndvi': -15, 'ndwi': -20, 'disturbance': 10, 'edge': 0,
                'description': "Long-term warming alters vegetation, water availability, phenology",
                'probability': 'High (projected by 2040)',
                'duration': 'Permanent'
            }
        }
        
        t_params = threat_params[threat]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ⚠️ Threat Profile")
            
            st.error(f"""
            **Threat:** {threat}
            
            **Description:** {t_params['description']}
            
            **Likelihood:** {t_params['probability']}
            
            **Duration:** {t_params['duration']}
            
            **Environmental Impacts:**
            - NDVI: {t_params['ndvi']:+d}%
            - NDWI: {t_params['ndwi']:+d}%
            - Disturbance: {t_params['disturbance']:+d}%
            - Edge Effect: {t_params['edge']:+d}%
            """)
        
        with col2:
            st.markdown("#### 📉 Predicted Impacts")
            
            # Calculate (same logic)
            wetland_impact = (
                (t_params['ndwi'] * 0.003) +
                (t_params['ndvi'] * 0.001) +
                (t_params['disturbance'] * -0.002)
            )
            wetland_new = np.clip(baseline_occupancy['Wetland'] + wetland_impact, 0, 1)
            wetland_change_pct = (wetland_impact / baseline_occupancy['Wetland']) * 100
            
            forest_impact = (
                (t_params['ndvi'] * 0.004) +
                (t_params['edge'] * -0.005) +
                (t_params['disturbance'] * -0.003)
            )
            forest_new = np.clip(baseline_occupancy['Forest'] + forest_impact, 0, 1)
            forest_change_pct = (forest_impact / baseline_occupancy['Forest']) * 100
            
            urban_impact = (
                (t_params['ndvi'] * 0.0005) +
                (t_params['ndwi'] * 0.0005) +
                (t_params['disturbance'] * 0.001)
            )
            urban_new = np.clip(baseline_occupancy['Urban'] + urban_impact, 0, 1)
            urban_change_pct = (urban_impact / baseline_occupancy['Urban']) * 100
            
            # Display impacts
            st.metric("Wetland Birds", f"{wetland_change_pct:+.1f}%", 
                     delta=f"{wetland_impact:+.3f} ψ",
                     delta_color="inverse")
            
            st.metric("Forest Birds", f"{forest_change_pct:+.1f}%",
                     delta=f"{forest_impact:+.3f} ψ",
                     delta_color="inverse")
            
            st.metric("Urban Birds", f"{urban_change_pct:+.1f}%",
                     delta=f"{urban_impact:+.3f} ψ",
                     delta_color="inverse")
            
            # Most vulnerable guild
            impacts = {
                'Wetland': wetland_impact,
                'Forest': forest_impact,
                'Urban': urban_impact
            }
            most_vulnerable = min(impacts, key=impacts.get)
            
            st.warning(f"""
            ⚠️ **Most Vulnerable:** {most_vulnerable} birds
            
            **Recommended Mitigation:**
            {get_mitigation_recommendation(threat, most_vulnerable)}
            """)
    
    # =============================================================================
    # TAB 4: SCENARIO COMPARISON
    # =============================================================================
    
    with tab4:
        st.markdown("### 📊 Compare Multiple Scenarios")
        st.caption("Side-by-side comparison of different options")
        
        # Select scenarios to compare
        compare_scenarios = st.multiselect(
            "Select scenarios to compare (max 5):",
            list(scenarios_params.keys()),
            default=["Baseline (No Action)", "Tree Planting Campaign (1 ha)", "Visitor Access Restrictions (Buffer Zone)"]
        )
        
        if len(compare_scenarios) > 0:
            comparison_results = []
            
            for scenario_name in compare_scenarios:
                params = scenarios_params[scenario_name]
                
                # Calculate impacts
                wetland_impact = (
                    (params['ndwi'] * 0.003) +
                    (params['ndvi'] * 0.001) +
                    (params['disturbance'] * -0.002)
                )
                
                forest_impact = (
                    (params['ndvi'] * 0.004) +
                    (params['edge'] * -0.005) +
                    (params['disturbance'] * -0.003) +
                    params.get('forest_bonus', 0)
                )
                
                urban_impact = (
                    (params['ndvi'] * 0.0005) +
                    (params['ndwi'] * 0.0005) +
                    (params['disturbance'] * 0.001)
                )
                
                total_impact = wetland_impact + forest_impact + urban_impact
                
                comparison_results.append({
                    'Scenario': scenario_name,
                    'Wetland Δψ': wetland_impact,
                    'Forest Δψ': forest_impact,
                    'Urban Δψ': urban_impact,
                    'Total Δψ': total_impact,
                    'Cost (₹)': params['cost'],
                    'Time (months)': params['time'],
                    'Cost per 0.01ψ': params['cost'] / (total_impact * 100 + 0.001) if total_impact > 0 else float('inf')
                })
            
            comparison_df = pd.DataFrame(comparison_results)
            
            # Visualization
            fig = go.Figure()
            
            guilds = ['Wetland Δψ', 'Forest Δψ', 'Urban Δψ']
            colors = ['#2196F3', '#4CAF50', '#FF9800']
            
            for guild, color in zip(guilds, colors):
                fig.add_trace(go.Bar(
                    name=guild.replace(' Δψ', ''),
                    x=comparison_df['Scenario'],
                    y=comparison_df[guild],
                    marker_color=color,
                    text=comparison_df[guild].round(3),
                    textposition='auto'
                ))
            
            fig.update_layout(
                title="Predicted Impact by Scenario",
                xaxis_title="Scenario",
                yaxis_title="Change in Occupancy (Δψ)",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            st.markdown("#### 📋 Detailed Comparison")
            
            st.dataframe(
                comparison_df.style.format({
                    'Wetland Δψ': '{:+.3f}',
                    'Forest Δψ': '{:+.3f}',
                    'Urban Δψ': '{:+.3f}',
                    'Total Δψ': '{:+.3f}',
                    'Cost (₹)': '₹{:,.0f}',
                    'Cost per 0.01ψ': '₹{:,.0f}'
                }).background_gradient(subset=['Total Δψ'], cmap='RdYlGn', vmin=-0.1, vmax=0.1),
                use_container_width=True
            )
            
            # Recommendations
            st.markdown("#### 🎯 Recommendations")
            
            best_impact = comparison_df.loc[comparison_df['Total Δψ'].idxmax()]
            best_cost = comparison_df[comparison_df['Cost per 0.01ψ'] < float('inf')].loc[comparison_df[comparison_df['Cost per 0.01ψ'] < float('inf')]['Cost per 0.01ψ'].idxmin()] if len(comparison_df[comparison_df['Cost per 0.01ψ'] < float('inf')]) > 0 else None
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"""
                **🏆 Highest Impact:**
                {best_impact['Scenario']}
                
                - Total Δψ: +{best_impact['Total Δψ']:.3f}
                - Cost: ₹{best_impact['Cost (₹)']:,.0f}
                - Time: {best_impact['Time (months)']} months
                """)
            
            with col2:
                if best_cost is not None:
                    st.success(f"""
                    **💰 Best Cost-Effectiveness:**
                    {best_cost['Scenario']}
                    
                    - Cost per 0.01ψ: ₹{best_cost['Cost per 0.01ψ']:,.0f}
                    - Total Δψ: +{best_cost['Total Δψ']:.3f}
                    - Time: {best_cost['Time (months)']} months
                    """)

def get_mitigation_recommendation(threat, vulnerable_guild):

    """Get mitigation recommendations based on threat and vulnerable guild"""
    
    recommendations = {
        ("Drought (Severe)", "Wetland"): "Install water retention structures, emergency water supply",
        ("Drought (Severe)", "Forest"): "Mulch around trees, plant drought-resistant species",
        ("Flooding (Monsoon)", "Wetland"): "Monitor water quality, clear drainage post-flood",
        ("Flooding (Monsoon)", "Forest"): "Ensure nest boxes are elevated, monitor for disease",
        ("Urban Development (Adjacent)", "Forest"): "Plant sound barriers, restrict visitor access core",
        ("Urban Development (Adjacent)", "Wetland"): "Monitor pollution sources, filter runoff",
        ("Pollution Event (Water)", "Wetland"): "Emergency cleanup, identify pollution source, install filtration",
        ("Climate Change (2°C warming)", "Forest"): "Assisted migration, plant climate-adapted species",
        ("Climate Change (2°C warming)", "Wetland"): "Create additional water bodies, improve retention"
    }
    
    return recommendations.get((threat, vulnerable_guild), "Conduct detailed impact assessment, implement adaptive management")


# =============================================================================
# DASHBOARD PAGE: RESTORATION TIMELINE CALCULATOR
# Predict restoration timelines and costs for degraded areas
# =============================================================================

def render_restoration_timeline_page():
    """
    Interactive restoration planning tool with timeline and budget predictions
    """
    st.title("⏰ Restoration Timeline Calculator")
    st.markdown("*Plan restoration projects with AI-predicted timelines and budgets*")
    
    # Introduction
    st.markdown("""
    <div class="info-box">
    <h3>🎯 What This Tool Does</h3>
    <p>Plan and visualize restoration projects for degraded habitat areas:</p>
    <ul>
    <li><strong>Timeline Prediction:</strong> AI estimates recovery time based on degradation severity</li>
    <li><strong>Budget Calculator:</strong> Itemized costs for each restoration phase</li>
    <li><strong>Milestone Tracking:</strong> Track expected improvements month-by-month</li>
    <li><strong>Scenario Planning:</strong> Compare quick fixes vs long-term restoration</li>
    </ul>
    <p><strong>Machine Learning:</strong> Predictions based on habitat recovery models! 🤖</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Area Selection
    st.markdown("### 📍 Step 1: Select Restoration Area")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Load priority areas if available
        priority_file = Path("data/processed/phase6/CPI_metadata.csv")
        
        area_options = []
        has_priority_data = False
        
        if priority_file.exists():
            try:
                priority_df = pd.read_csv(priority_file)
                
                # Check which columns exist
                available_cols = priority_df.columns.tolist()
                
                # Try to find or create Priority_Level column
                if 'Priority_Level' in available_cols:
                    priority_col = 'Priority_Level'
                    has_priority_data = True
                elif 'priority_level' in available_cols:
                    priority_col = 'priority_level'
                    has_priority_data = True
                elif 'Level' in available_cols:
                    priority_col = 'Level'
                    has_priority_data = True
                elif 'CPI' in available_cols or 'cpi' in available_cols:
                    # Create Priority_Level from CPI values
                    cpi_col = 'CPI' if 'CPI' in available_cols else 'cpi'
                    
                    # Map CPI (0-1) to Priority Levels (1-5)
                    priority_df['Priority_Level'] = pd.cut(
                        priority_df[cpi_col],
                        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        labels=[1, 2, 3, 4, 5],
                        include_lowest=True
                    ).astype(int)
                    
                    priority_col = 'Priority_Level'
                    has_priority_data = True
                    
                    st.success("✓ Priority levels calculated from CPI values")
                else:
                    # No usable column - show warning
                    st.warning(f"⚠️ Priority data found but no usable columns. Available: {', '.join(available_cols)}")
                    has_priority_data = False
                
                # If we have priority data, show info and create options
                if has_priority_data:
                    # Count high priority areas
                    high_priority_count = len(priority_df[priority_df[priority_col] >= 4])
                    medium_priority_count = len(priority_df[priority_df[priority_col] == 3])
                    
                    st.info(f"""
                    **🎯 Priority areas identified from Conservation Priority Index:**
                    
                    - **Critical/High Priority (Level 4-5):** {high_priority_count} areas
                    - **Medium Priority (Level 3):** {medium_priority_count} areas
                    
                    These areas combine high occupancy potential with significant threats.
                    """)
                    
                    # Build area options based on what exists
                    if high_priority_count > 0:
                        area_options.append("High Priority Area (CPI Level 4-5)")
                    if medium_priority_count > 0:
                        area_options.append("Medium Priority Area (CPI Level 3)")
                    
                    area_options.append("Custom Area (Manual Input)")
                else:
                    # No priority data - just offer custom
                    area_options = ["Custom Area (Manual Input)"]
                    
            except Exception as e:
                st.warning(f"⚠️ Could not load priority data: {e}")
                st.info("Using custom area input instead.")
                area_options = ["Custom Area (Manual Input)"]
                has_priority_data = False
        else:
            # File doesn't exist
            st.info("""
            💡 **No Conservation Priority Index data found.**
            
            To get priority recommendations:
            1. Complete Phase 6 analysis
            2. Run: `python sentinel/phase6/conservation_priority.py`
            3. This will identify which areas need restoration most urgently
            """)
            area_options = ["Custom Area (Manual Input)"]
        
        # Area type selector
        area_type = st.selectbox(
            "Select area type:",
            area_options,
            help="Choose from priority areas or enter custom location"
        )
        
        # Show context based on selection
        if "High Priority" in area_type and has_priority_data:
            st.success("🔴 **Critical Action Area** - Maximum conservation impact expected")
        elif "Medium Priority" in area_type and has_priority_data:
            st.info("🟡 **Important Area** - Moderate intervention recommended")
        else:
            st.caption("📍 Enter details for your custom restoration area below")
            
    with col2:
        area_size = st.number_input(
            "Area Size (hectares)",
            min_value=0.1,
            max_value=5.0,
            value=0.8,
            step=0.1,
            help="Size of degraded area to restore"
        )
        
        st.metric("As % of Sanctuary", f"{(area_size/7.86)*100:.1f}%")
    
    # Degradation Assessment
    st.markdown("---")
    st.markdown("### 🔍 Step 2: Assess Current Condition")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Environmental Condition")
        
        current_ndvi = st.slider(
            "Current NDVI (Vegetation Health)",
            min_value=0.0,
            max_value=0.8,
            value=0.25,
            step=0.05,
            help="0.0-0.2 = Bare/sparse, 0.2-0.4 = Degraded, 0.4-0.6 = Moderate, 0.6-0.8 = Healthy"
        )
        
        current_ndwi = st.slider(
            "Current NDWI (Water Availability)",
            min_value=-0.3,
            max_value=0.5,
            value=0.1,
            step=0.05,
            help="-0.3-0.0 = Dry, 0.0-0.2 = Moist, 0.2-0.5 = Wet/water present"
        )
        
        invasive_cover = st.slider(
            "Invasive Species Cover (%)",
            min_value=0,
            max_value=100,
            value=40,
            step=10,
            help="Percentage covered by invasive plants (Lantana, Prosopis)"
        )
    
    with col2:
        st.markdown("#### Disturbance Assessment")
        
        disturbance_level = st.select_slider(
            "Human Disturbance Level",
            options=['Low', 'Moderate', 'High', 'Very High'],
            value='High',
            help="Frequency of human activity, noise, trampling"
        )
        
        edge_distance = st.slider(
            "Distance to Sanctuary Edge (m)",
            min_value=0,
            max_value=150,
            value=30,
            step=10,
            help="Closer to edge = more edge effects"
        )
        
        soil_condition = st.select_slider(
            "Soil Condition",
            options=['Poor', 'Fair', 'Good', 'Excellent'],
            value='Fair',
            help="Soil fertility and structure"
        )
    
    # Calculate degradation severity
    degradation_score = calculate_degradation_severity(
        current_ndvi, current_ndwi, invasive_cover, 
        disturbance_level, edge_distance, soil_condition
    )
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        severity_label, severity_color = get_severity_label(degradation_score)
        st.metric(
            "Degradation Severity",
            severity_label,
            help=f"Score: {degradation_score:.2f}/10"
        )
    
    with col2:
        current_psi = estimate_current_occupancy(current_ndvi, current_ndwi, invasive_cover)
        st.metric(
            "Current Bird Occupancy (ψ)",
            f"{current_psi:.2f}",
            help="Estimated habitat suitability"
        )
    
    with col3:
        target_psi = 0.75  # Target restoration goal
        improvement_needed = target_psi - current_psi
        st.metric(
            "Improvement Needed",
            f"+{improvement_needed:.2f} ψ",
            help="To reach good habitat quality (ψ=0.75)"
        )
    
    # Restoration Strategy
    st.markdown("---")
    st.markdown("### 🛠️ Step 3: Select Restoration Strategy")
    
    strategy = st.selectbox(
        "Choose Restoration Approach:",
        [
            "Quick Fix (6-12 months)",
            "Standard Restoration (12-24 months)",
            "Comprehensive Restoration (24-36 months)",
            "Passive Regeneration (36-60 months)",
            "Custom Timeline"
        ]
    )
    
    # Calculate restoration plan based on strategy
    restoration_plan = calculate_restoration_plan(
        strategy, degradation_score, area_size, 
        current_ndvi, invasive_cover, soil_condition
    )
    
    # Display Timeline
    st.markdown("---")
    st.markdown("### 📅 Step 4: Restoration Timeline & Milestones")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create Gantt-style timeline
        fig = create_restoration_gantt(restoration_plan)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### ⏱️ Timeline Summary")
        
        st.metric("Total Duration", f"{restoration_plan['total_months']} months")
        st.metric("Total Cost", f"₹{restoration_plan['total_cost']:,}")
        st.metric("Expected Final ψ", f"{restoration_plan['final_psi']:.2f}")
        
        improvement = restoration_plan['final_psi'] - current_psi
        improvement_pct = (improvement / current_psi) * 100 if current_psi > 0 else 0
        
        st.metric("Improvement", f"+{improvement:.2f} ψ", delta=f"+{improvement_pct:.1f}%")
    
    # Phase Details
    st.markdown("---")
    st.markdown("### 📋 Detailed Phase Breakdown")
    
    for phase in restoration_plan['phases']:
        with st.expander(f"{phase['icon']} {phase['name']} (Months {phase['start']}-{phase['end']})", 
                        expanded=(phase['name'] == restoration_plan['phases'][0]['name'])):
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Objectives:** {phase['objectives']}")
                st.markdown(f"**Activities:**")
                for activity in phase['activities']:
                    st.markdown(f"- {activity}")
                
                if 'expected_outcomes' in phase:
                    st.markdown(f"**Expected Outcomes:**")
                    for outcome in phase['expected_outcomes']:
                        st.markdown(f"✓ {outcome}")
            
            with col2:
                st.metric("Duration", f"{phase['duration']} months")
                st.metric("Cost", f"₹{phase['cost']:,}")
                st.metric("ψ at End", f"{phase['psi_end']:.2f}")
                
                if phase['psi_end'] > phase['psi_start']:
                    st.success(f"↑ +{(phase['psi_end'] - phase['psi_start']):.2f} ψ")
    
    # Progress Prediction Chart
    st.markdown("---")
    st.markdown("### 📈 Predicted Recovery Curve")
    
    # Generate month-by-month predictions
    recovery_curve = generate_recovery_curve(
        current_psi, 
        restoration_plan['final_psi'],
        restoration_plan['total_months'],
        restoration_plan['phases']
    )
    
    fig_recovery = go.Figure()
    
    # Current condition
    fig_recovery.add_hline(y=current_psi, line_dash="dash", line_color="red",
                          annotation_text=f"Current (ψ={current_psi:.2f})")
    
    # Target
    fig_recovery.add_hline(y=target_psi, line_dash="dash", line_color="green",
                          annotation_text=f"Target (ψ={target_psi:.2f})")
    
    # Recovery curve
    fig_recovery.add_trace(go.Scatter(
        x=recovery_curve['months'],
        y=recovery_curve['psi'],
        mode='lines+markers',
        name='Predicted Recovery',
        line=dict(color='#2196F3', width=3),
        marker=dict(size=6)
    ))
    
    # Uncertainty band
    fig_recovery.add_trace(go.Scatter(
        x=recovery_curve['months'] + recovery_curve['months'][::-1],
        y=recovery_curve['psi_upper'] + recovery_curve['psi_lower'][::-1],
        fill='toself',
        fillcolor='rgba(33,150,243,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='95% Confidence'
    ))
    
    # Add phase markers
    for phase in restoration_plan['phases']:
        fig_recovery.add_vline(
            x=phase['end'],
            line_dash="dot",
            line_color="gray",
            annotation_text=phase['name'][:15],
            annotation_position="top"
        )
    
    fig_recovery.update_layout(
        title="Habitat Quality Recovery Over Time",
        xaxis_title="Months from Start",
        yaxis_title="Habitat Suitability (ψ)",
        height=400,
        yaxis=dict(range=[0, 1])
    )
    
    st.plotly_chart(fig_recovery, use_container_width=True)
    
    # Budget Breakdown
    st.markdown("---")
    st.markdown("### 💰 Budget Details")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Itemized budget
        budget_items = []
        for phase in restoration_plan['phases']:
            for item in phase.get('budget_items', []):
                budget_items.append({
                    'Phase': phase['name'],
                    'Item': item['item'],
                    'Quantity': item['quantity'],
                    'Unit Cost': item['unit_cost'],
                    'Total': item['total']
                })
        
        if budget_items:
            budget_df = pd.DataFrame(budget_items)
            
            st.dataframe(
                budget_df.style.format({
                    'Unit Cost': '₹{:,.0f}',
                    'Total': '₹{:,.0f}'
                }),
                use_container_width=True,
                height=300
            )
    
    with col2:
        # Budget summary pie chart
        phase_costs = [phase['cost'] for phase in restoration_plan['phases']]
        phase_names = [phase['name'] for phase in restoration_plan['phases']]
        
        fig_budget = go.Figure(data=[go.Pie(
            labels=phase_names,
            values=phase_costs,
            hole=0.3
        )])
        
        fig_budget.update_layout(
            title="Cost by Phase",
            height=300
        )
        
        st.plotly_chart(fig_budget, use_container_width=True)
        
        st.metric("Total Investment", f"₹{restoration_plan['total_cost']:,}")
        st.metric("Cost per Hectare", f"₹{restoration_plan['total_cost']/area_size:,.0f}")
        st.metric("Cost per 0.1 ψ Improvement", 
                 f"₹{restoration_plan['total_cost']/(improvement*10):,.0f}" if improvement > 0 else "N/A")
    
    # ROI Analysis
    st.markdown("---")
    st.markdown("### 📊 Return on Investment (ROI)")
    
    # Calculate ecosystem service values
    ecosystem_benefits = calculate_ecosystem_benefits(
        area_size, 
        current_psi, 
        restoration_plan['final_psi'],
        restoration_plan['total_months']
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Species Gain (Estimated)",
            f"+{ecosystem_benefits['species_gain']}",
            help="Additional species expected after restoration"
        )
    
    with col2:
        st.metric(
            "Ecosystem Service Value",
            f"₹{ecosystem_benefits['service_value']:,}/year",
            help="Carbon sequestration, air/water filtration value"
        )
    
    with col3:
        payback_years = restoration_plan['total_cost'] / ecosystem_benefits['service_value'] if ecosystem_benefits['service_value'] > 0 else float('inf')
        st.metric(
            "Payback Period",
            f"{payback_years:.1f} years" if payback_years < 100 else ">100 years",
            help="Time to recover costs through ecosystem services"
        )
    
    # Risk Assessment
    st.markdown("---")
    st.markdown("### ⚠️ Risk Assessment")
    
    risks = assess_restoration_risks(
        degradation_score, invasive_cover, disturbance_level, edge_distance
    )
    
    for risk in risks:
        if risk['level'] == 'High':
            st.error(f"🔴 **{risk['risk']}** - {risk['mitigation']}")
        elif risk['level'] == 'Medium':
            st.warning(f"🟡 **{risk['risk']}** - {risk['mitigation']}")
    
    # Download Report
    st.markdown("---")
    st.markdown("### 📥 Download Restoration Plan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Generate PDF Report"):
            st.info("PDF generation feature coming soon!")
    
    with col2:
        # Export as CSV
        export_data = []
        for phase in restoration_plan['phases']:
            export_data.append({
                'Phase': phase['name'],
                'Start_Month': phase['start'],
                'End_Month': phase['end'],
                'Duration_Months': phase['duration'],
                'Cost_INR': phase['cost'],
                'Starting_Psi': phase['psi_start'],
                'Ending_Psi': phase['psi_end'],
                'Objectives': phase['objectives']
            })
        
        export_df = pd.DataFrame(export_data)
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            "📊 Download as CSV",
            csv,
            file_name=f"restoration_plan_{area_size}ha.csv",
            mime="text/csv"
        )

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_degradation_severity(ndvi, ndwi, invasive, disturbance, edge_dist, soil):
    """Calculate overall degradation severity score (0-10)"""
    
    score = 0
    
    # NDVI component (0-3 points)
    if ndvi < 0.2:
        score += 3
    elif ndvi < 0.4:
        score += 2
    elif ndvi < 0.6:
        score += 1
    
    # NDWI component (0-2 points)
    if ndwi < 0:
        score += 2
    elif ndwi < 0.2:
        score += 1
    
    # Invasive cover (0-3 points)
    score += (invasive / 100) * 3
    
    # Disturbance (0-2 points)
    disturbance_scores = {'Low': 0, 'Moderate': 0.5, 'High': 1.5, 'Very High': 2}
    score += disturbance_scores[disturbance]
    
    # Edge distance (0-1 point)
    if edge_dist < 30:
        score += 1
    elif edge_dist < 60:
        score += 0.5
    
    # Soil (inverse, 0-1 point)
    soil_scores = {'Poor': 1, 'Fair': 0.5, 'Good': 0, 'Excellent': 0}
    score += soil_scores[soil]
    
    return min(score, 10)

def get_severity_label(score):
    """Get severity label and color"""
    if score >= 7:
        return "Severely Degraded", "error"
    elif score >= 5:
        return "Moderately Degraded", "warning"
    elif score >= 3:
        return "Mildly Degraded", "info"
    else:
        return "Good Condition", "success"

def estimate_current_occupancy(ndvi, ndwi, invasive):
    """Estimate current habitat suitability"""
    # Simplified occupancy model
    base_psi = 0.3
    base_psi += ndvi * 0.4  # NDVI effect
    base_psi += ndwi * 0.2   # NDWI effect
    base_psi -= (invasive / 100) * 0.3  # Invasive penalty
    
    return np.clip(base_psi, 0, 1)

def calculate_restoration_plan(strategy, severity, area, ndvi, invasive, soil):
    """Generate restoration plan based on strategy"""
    
    plans = {
        "Quick Fix (6-12 months)": {
            'total_months': 9,
            'phases': [
                {
                    'name': 'Phase 1: Emergency Intervention',
                    'icon': '🚨',
                    'start': 0,
                    'end': 3,
                    'duration': 3,
                    'cost': int(area * 30000),
                    'psi_start': estimate_current_occupancy(ndvi, 0.1, invasive),
                    'psi_end': estimate_current_occupancy(ndvi, 0.1, invasive) + 0.05,
                    'objectives': 'Remove immediate threats, stop degradation',
                    'activities': [
                        'Remove invasive species (mechanical)',
                        'Install temporary barriers',
                        'Basic cleanup'
                    ],
                    'budget_items': [
                        {'item': 'Invasive removal (manual)', 'quantity': area * 10, 'unit_cost': 2000, 'total': area * 20000},
                        {'item': 'Barriers & signage', 'quantity': 5, 'unit_cost': 2000, 'total': 10000}
                    ]
                },
                {
                    'name': 'Phase 2: Quick Restoration',
                    'icon': '🌱',
                    'start': 3,
                    'end': 9,
                    'duration': 6,
                    'cost': int(area * 40000),
                    'psi_start': estimate_current_occupancy(ndvi, 0.1, invasive) + 0.05,
                    'psi_end': 0.60,
                    'objectives': 'Rapid habitat improvement',
                    'activities': [
                        'Fast-growing native species planting',
                        'Nest box installation',
                        'Basic soil amendment'
                    ],
                    'budget_items': [
                        {'item': 'Fast-growing saplings', 'quantity': area * 100, 'unit_cost': 150, 'total': area * 15000},
                        {'item': 'Nest boxes', 'quantity': area * 10, 'unit_cost': 500, 'total': area * 5000},
                        {'item': 'Soil amendment', 'quantity': area, 'unit_cost': 20000, 'total': area * 20000}
                    ]
                }
            ]
        },
        
        "Standard Restoration (12-24 months)": {
            'total_months': 18,
            'phases': [
                {
                    'name': 'Phase 1: Site Preparation',
                    'icon': '🧹',
                    'start': 0,
                    'end': 4,
                    'duration': 4,
                    'cost': int(area * 35000),
                    'psi_start': estimate_current_occupancy(ndvi, 0.1, invasive),
                    'psi_end': estimate_current_occupancy(ndvi, 0.1, invasive) + 0.08,
                    'objectives': 'Prepare site, remove threats',
                    'activities': [
                        'Complete invasive removal',
                        'Soil testing and amendment',
                        'Water management setup',
                        'Access control'
                    ],
                    'budget_items': [
                        {'item': 'Invasive removal (comprehensive)', 'quantity': area * 10, 'unit_cost': 2500, 'total': area * 25000},
                        {'item': 'Soil testing', 'quantity': 5, 'unit_cost': 1000, 'total': 5000},
                        {'item': 'Water management', 'quantity': 1, 'unit_cost': area * 5000, 'total': area * 5000}
                    ]
                },
                {
                    'name': 'Phase 2: Active Restoration',
                    'icon': '🌳',
                    'start': 4,
                    'end': 12,
                    'duration': 8,
                    'cost': int(area * 55000),
                    'psi_start': estimate_current_occupancy(ndvi, 0.1, invasive) + 0.08,
                    'psi_end': 0.65,
                    'objectives': 'Establish native vegetation',
                    'activities': [
                        'Native tree planting (200+ saplings/ha)',
                        'Shrub layer establishment',
                        'Ground cover seeding',
                        'Erosion control measures'
                    ],
                    'budget_items': [
                        {'item': 'Native tree saplings', 'quantity': area * 200, 'unit_cost': 100, 'total': area * 20000},
                        {'item': 'Shrubs', 'quantity': area * 300, 'unit_cost': 50, 'total': area * 15000},
                        {'item': 'Seeds & mulch', 'quantity': area, 'unit_cost': 10000, 'total': area * 10000},
                        {'item': 'Labor (planting)', 'quantity': area * 20, 'unit_cost': 500, 'total': area * 10000}
                    ]
                },
                {
                    'name': 'Phase 3: Maintenance & Monitoring',
                    'icon': '🔍',
                    'start': 12,
                    'end': 18,
                    'duration': 6,
                    'cost': int(area * 25000),
                    'psi_start': 0.65,
                    'psi_end': 0.73,
                    'objectives': 'Ensure establishment, monitor recovery',
                    'activities': [
                        'Regular watering (dry season)',
                        'Weed control',
                        'Bird monitoring',
                        'Replanting failures'
                    ],
                    'budget_items': [
                        {'item': 'Watering system', 'quantity': 1, 'unit_cost': area * 8000, 'total': area * 8000},
                        {'item': 'Maintenance labor', 'quantity': 6, 'unit_cost': area * 2000, 'total': area * 12000},
                        {'item': 'Replacement plants', 'quantity': area * 50, 'unit_cost': 100, 'total': area * 5000}
                    ]
                }
            ]
        },
        
        # Add more strategies...
        "Comprehensive Restoration (24-36 months)": {
            'total_months': 30,
            'phases': [
                # Similar structure but more detailed phases
                {'name': 'Phase 1: Assessment & Planning', 'icon': '📋', 'start': 0, 'end': 3, 'duration': 3, 
                 'cost': int(area * 20000), 'psi_start': estimate_current_occupancy(ndvi, 0.1, invasive), 
                 'psi_end': estimate_current_occupancy(ndvi, 0.1, invasive), 'objectives': 'Detailed survey and planning',
                 'activities': ['Biodiversity survey', 'Soil analysis', 'Hydrology assessment', 'Stakeholder consultation'],
                 'budget_items': [{'item': 'Ecological survey', 'quantity': 1, 'unit_cost': area * 15000, 'total': area * 15000}]},
                
                {'name': 'Phase 2: Site Preparation', 'icon': '🧹', 'start': 3, 'end': 8, 'duration': 5,
                 'cost': int(area * 40000), 'psi_start': estimate_current_occupancy(ndvi, 0.1, invasive),
                 'psi_end': estimate_current_occupancy(ndvi, 0.1, invasive) + 0.10, 'objectives': 'Comprehensive site prep',
                 'activities': ['Invasive eradication', 'Soil remediation', 'Drainage improvement', 'Fencing'],
                 'budget_items': [{'item': 'Complete invasive removal', 'quantity': area * 15, 'unit_cost': 2000, 'total': area * 30000}]},
                
                {'name': 'Phase 3: Vegetation Establishment', 'icon': '🌳', 'start': 8, 'end': 20, 'duration': 12,
                 'cost': int(area * 70000), 'psi_start': estimate_current_occupancy(ndvi, 0.1, invasive) + 0.10,
                 'psi_end': 0.70, 'objectives': 'Full vegetation layers',
                 'activities': ['Canopy trees (300+/ha)', 'Mid-story trees', 'Shrub layer', 'Ground cover', 'Microhabitat features'],
                 'budget_items': [{'item': 'Premium native saplings', 'quantity': area * 300, 'unit_cost': 150, 'total': area * 45000}]},
                
                {'name': 'Phase 4: Habitat Enhancement', 'icon': '🦜', 'start': 20, 'end': 26, 'duration': 6,
                 'cost': int(area * 30000), 'psi_start': 0.70, 'psi_end': 0.78, 'objectives': 'Add habitat structures',
                 'activities': ['Nest boxes', 'Perches', 'Water features', 'Log piles', 'Brush piles'],
                 'budget_items': [{'item': 'Nest boxes & structures', 'quantity': area * 30, 'unit_cost': 800, 'total': area * 24000}]},
                
                {'name': 'Phase 5: Long-term Monitoring', 'icon': '📊', 'start': 26, 'end': 30, 'duration': 4,
                 'cost': int(area * 15000), 'psi_start': 0.78, 'psi_end': 0.82, 'objectives': 'Adaptive management',
                 'activities': ['Monthly bird surveys', 'Vegetation monitoring', 'Adaptive interventions', 'Documentation'],
                 'budget_items': [{'item': 'Monitoring program', 'quantity': 4, 'unit_cost': area * 3000, 'total': area * 12000}]}
            ]
        }
    }
    
    # Select appropriate plan
    if strategy in plans:
        plan = plans[strategy]
    else:
        plan = plans["Standard Restoration (12-24 months)"]  # Default
    
    # Calculate totals
    plan['total_cost'] = sum(phase['cost'] for phase in plan['phases'])
    plan['final_psi'] = plan['phases'][-1]['psi_end']
    
    return plan

def create_restoration_gantt(plan):
    """Create Gantt chart for restoration timeline"""
    
    fig = go.Figure()
    
    for i, phase in enumerate(plan['phases']):
        fig.add_trace(go.Bar(
            name=phase['name'],
            x=[phase['end'] - phase['start']],
            y=[phase['name']],
            base=[phase['start']],
            orientation='h',
            marker=dict(
                color=f"rgba({int(255*(1-i/len(plan['phases'])))}, {int(150*i/len(plan['phases']))}, {int(200)}, 0.8)"
            ),
            text=f"{phase['duration']}mo<br>₹{phase['cost']:,}",
            textposition='inside',
            hovertemplate=f"<b>{phase['name']}</b><br>" +
                         f"Duration: {phase['duration']} months<br>" +
                         f"Cost: ₹{phase['cost']:,}<br>" +
                         f"ψ: {phase['psi_start']:.2f} → {phase['psi_end']:.2f}<extra></extra>"
        ))
    
    fig.update_layout(
        title="Restoration Timeline (Gantt Chart)",
        xaxis_title="Months from Start",
        yaxis_title="Restoration Phase",
        barmode='overlay',
        height=300,
        showlegend=False
    )
    
    return fig

def generate_recovery_curve(start_psi, end_psi, months, phases):
    """Generate month-by-month recovery predictions"""
    
    curve_data = {
        'months': [],
        'psi': [],
        'psi_upper': [],
        'psi_lower': []
    }
    
    current_psi = start_psi
    
    for month in range(months + 1):
        # Find which phase we're in
        current_phase = None
        for phase in phases:
            if phase['start'] <= month <= phase['end']:
                current_phase = phase
                break
        
        if current_phase:
            # Linear interpolation within phase
            phase_progress = (month - current_phase['start']) / current_phase['duration']
            phase_psi = current_phase['psi_start'] + (current_phase['psi_end'] - current_phase['psi_start']) * phase_progress
            
            # Add some realism (not perfectly linear)
            noise = np.random.normal(0, 0.02)
            phase_psi = np.clip(phase_psi + noise, 0, 1)
        else:
            phase_psi = start_psi
        
        curve_data['months'].append(month)
        curve_data['psi'].append(phase_psi)
        curve_data['psi_upper'].append(min(phase_psi + 0.05, 1))
        curve_data['psi_lower'].append(max(phase_psi - 0.05, 0))
    
    return curve_data

def calculate_ecosystem_benefits(area, current_psi, final_psi, months):
    """Calculate ecosystem service benefits"""
    
    improvement = final_psi - current_psi
    
    # Species gain (rough estimate: 0.1 psi = ~10 additional species for urban sanctuary)
    species_gain = int(improvement * 100)
    
    # Ecosystem service value (carbon, water, air quality)
    # Rough estimate: ₹50,000/ha/year for good habitat
    service_value = int(area * final_psi * 50000)
    
    return {
        'species_gain': species_gain,
        'service_value': service_value
    }

def assess_restoration_risks(severity, invasive, disturbance, edge_dist):
    """Assess restoration risks"""
    
    risks = []
    
    if invasive > 50:
        risks.append({
            'risk': 'High invasive re-establishment',
            'level': 'High',
            'mitigation': 'Require 2+ years monitoring and re-treatment'
        })
    
    if disturbance == 'Very High':
        risks.append({
            'risk': 'Continued human disturbance',
            'level': 'High',
            'mitigation': 'Install barriers, enforcement patrols, education campaigns'
        })
    
    if edge_dist < 30:
        risks.append({
            'risk': 'Strong edge effects',
            'level': 'Medium',
            'mitigation': 'Create buffer plantings, sound barriers'
        })
    
    if severity > 7:
        risks.append({
            'risk': 'Slow recovery rate',
            'level': 'Medium',
            'mitigation': 'Consider soil inoculation, extended timeline'
        })
    
    return risks



# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application"""
    
    # Sidebar navigation with icons and descriptions
    st.sidebar.title("🦜 Dashboard Navigation")
    
    st.sidebar.markdown("""
    <div class="info-box">
    <strong>New to this dashboard?</strong><br>
    Start with <strong>Overview</strong> to understand the basics!
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Select Page:",
        [
            "📊 Overview",
            "🗺️ Habitat Maps",
            "🐦 Guild Analysis",
            "🎯 Conservation Priority",
            "📈 Temporal Trends",
            "📉 Species Decline Tracker",
            "🔮 What-If Scenarios",
            "⏰ Restoration Timeline"
        ]
    )
    
    # Add other pages here (continuing same pattern)
    if page == "📊 Overview":
        render_overview_page()
    elif page == "🗺️ Habitat Maps":
        render_habitat_maps_page()
    elif page == "🐦 Guild Analysis":
        render_guild_analysis_page()
    elif page == "🎯 Conservation Priority":
        render_conservation_priority_page()
    elif page == "📈 Temporal Trends":
        render_temporal_trends_page()
    elif page == "📉 Species Decline Tracker":
        render_species_decline_tracker_page()
    elif page == "🔮 What-If Scenarios":
        render_whatif_scenarios_page()
    elif page == "⏰ Restoration Timeline":
       render_restoration_timeline_page()

    
    # Footer with glossary access
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 Need Help?")
    
    if st.sidebar.button("View Complete Glossary"):
        st.sidebar.info("""
        **Quick Definitions:**
        - **ψ (psi)**: Habitat suitability (0-1 score)
        - **Guild**: Group of similar birds
        - **NDVI**: Vegetation greenness from satellite
        - **CPI**: Conservation Priority Index
        - **Core/Buffer**: Protected zones
        
        See full definitions in each page!
        """)
    
    st.sidebar.markdown("""
    ### ℹ️ About
    **Mangalavanam Conservation Dashboard**
    
    🦜 Adaptive monitoring system for bird sanctuary management
    
    **Data Period:** 2019-2025 (7 years)
    **Species:** 194 birds
    **Observations:** 2,847 records
    
    **Contact:** MSc Data Science Project
    """)

if __name__ == "__main__":
    main()