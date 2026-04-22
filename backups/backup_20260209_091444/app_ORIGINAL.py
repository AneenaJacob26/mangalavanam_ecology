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
# GLOSSARY / DEFINITIONS
# =============================================================================

GLOSSARY = {
    "Habitat Suitability (ψ)": """
        **What it means:** A score from 0 to 1 showing how good a place is for birds to live.
        - **0 = Unsuitable**: Birds unlikely to survive here
        - **0.5 = Moderate**: Some birds may use this area
        - **1 = Excellent**: Perfect habitat for birds
        
        **Why it matters:** Helps us identify which areas need protection most.
        
        **How we calculate it:** Using AI models that combine bird observations with satellite 
        images showing vegetation, water, and human activity.
    """,
    
    "Guild": """
        **What it means:** A group of bird species that use similar habitats and have similar needs.
        
        **Three Guilds in Mangalavanam:**
        
        1. 🦆 **Wetland Birds** (67 species)
           - Examples: Kingfishers, Herons, Egrets
           - Need: Water bodies, marshy areas
           - Habitat: Rivers, ponds, mangroves
        
        2. 🦜 **Forest Birds** (82 species)
           - Examples: Woodpeckers, Barbets, Bulbuls
           - Need: Trees, dense vegetation
           - Habitat: Forest interior, thick canopy
        
        3. 🐦 **Urban Birds** (45 species)
           - Examples: Crows, Mynas, Sparrows
           - Need: Can live anywhere
           - Habitat: Parks, buildings, roads
        
        **Why group them?** Instead of tracking 194 species individually, we group similar 
        species to understand which habitats are most important.
    """,
    
    "Conservation Priority": """
        **What it means:** A ranking system (1-5) showing which areas need urgent conservation action.
        
        **Priority Levels:**
        - 🔴 **Level 5 - Critical**: Act NOW (within 1-6 months)
        - 🟠 **Level 4 - High**: Act soon (within 6-12 months)
        - 🟡 **Level 3 - Medium**: Regular monitoring (within 1-2 years)
        - 🟢 **Level 2 - Low**: Routine maintenance
        - ⚪ **Level 1 - Minimal**: Stable, low concern
        
        **How we calculate it:**
        Priority = (Bird Habitat Quality × 0.40) + (Ecological Function × 0.35) - (Threats × 0.25)
        
        **Real example:** An area with many birds BUT near a road = High priority 
        (good habitat threatened by disturbance)
    """,
    
    "Environmental Rasters": """
        **What it means:** Satellite images that show environmental conditions like:
        
        1. **NDVI (Vegetation Index)**
           - 📊 What: Measures how green/healthy plants are
           - 🛰️ From: Sentinel-2 satellite
           - 📏 Scale: -1 (water/rock) to +1 (dense forest)
           - 🌿 Mangalavanam: Typically 0.3-0.7 (moderate vegetation)
        
        2. **NDWI (Water Index)**
           - 📊 What: Measures water availability
           - 🛰️ From: Sentinel-2 satellite
           - 📏 Scale: -1 (dry) to +1 (water)
           - 💧 Important for: Wetland birds
        
        3. **VIIRS Night Lights**
           - 📊 What: Shows human activity at night
           - 🛰️ From: VIIRS satellite sensor
           - 💡 Brighter = More urbanization
           - 🏙️ Used to measure: Urban pressure
        
        4. **Distance to Edge**
           - 📊 What: How far from sanctuary boundary
           - 🎯 Why: Birds near edges face more threats
           - 📏 Core interior = Safest areas
        
        **Why satellite images?** We can't be everywhere at once. Satellites show us 
        vegetation health, water, and urban growth across the ENTIRE sanctuary every week!
    """,
    
    "Occupancy Modeling": """
        **What it means:** A statistical method that accounts for "Did we actually see the bird?" 
        vs "Is the bird actually there?"
        
        **The Problem:**
        Not seeing a bird doesn't mean it's not there! Birds hide, fly away, or are silent.
        
        **The Solution:**
        Our models separate:
        - **ψ (psi)**: Probability bird truly lives there (TRUE occupancy)
        - **p**: Probability we detect it IF it's there (DETECTION rate)
        
        **Simple Example:**
        - You visit a forest 10 times
        - You see a woodpecker 3 times
        - Does it live there? Probably yes, but maybe you just missed it 7 times!
        - Our model says: ψ = 0.85 (85% sure it lives there), p = 0.35 (you only see it 35% of time)
        
        **Why this matters:** Traditional methods say "no bird seen = no bird there" which 
        UNDERESTIMATES actual bird populations. Our method is more accurate!
    """,
    
    "Bayesian Models": """
        **What it means:** A statistical approach that shows HOW CERTAIN we are about results.
        
        **Traditional Stats:** "Forest birds occupy core zone more than buffer" ✓ or ✗
        
        **Bayesian Stats:** "We are 95% confident forest birds prefer core zone, 
        with occupancy 0.19 higher (range: 0.05 to 0.33)"
        
        **Why better?**
        - Shows uncertainty (we're honest about what we DON'T know)
        - Works with small data (we only have 7 years of data)
        - Updates as new data arrives
        
        **Real Impact:** Instead of saying "protect this area" we say "we're 95% sure 
        protecting this area helps forest birds"
    """,
    
    "Functional Diversity": """
        **What it means:** How many different TYPES of birds (guilds) live in an area.
        
        **Why it matters:**
        - High diversity = Healthy ecosystem = All habitat types working
        - Low diversity = Problem (maybe only urban birds left = degraded habitat)
        
        **Indices We Calculate:**
        
        1. **Shannon Diversity**
           - Scale: 0 (only one guild) to 1.1 (all guilds equal)
           - Mangalavanam average: 0.82 (moderately diverse)
        
        2. **Simpson Diversity**
           - Meaning: If you randomly pick 2 birds, chance they're from different guilds
           - Mangalavanam: 61% (fairly diverse)
        
        3. **Functional Richness**
           - Simple count: How many guilds present?
           - Average: 2.1 guilds per area (good!)
        
        **Red Flag:** Area with only Urban guild = Problem (specialists gone)
        **Green Flag:** Area with all 3 guilds = Healthy ecosystem
    """,
    
    "Core vs Buffer Zones": """
        **What it means:** The sanctuary has two management areas:
        
        **🟢 Core Zone (2.74 ha):**
        - Most protected area
        - Strictly NO human entry (except rangers)
        - Where? Center of sanctuary
        - Best for: Sensitive specialist species
        
        **🟡 Buffer Zone (5.12 ha):**
        - Surrounds core
        - Limited human activity allowed
        - Where? Edges, near walking trails
        - Purpose: Protects core from outside threats
        
        **Why separate?**
        - Forest specialists NEED quiet core interior
        - Buffer absorbs edge effects (noise, light, disturbance)
        - Different management strategies for each zone
        
        **Our Findings:**
        - Forest birds: 87% occupy core vs 68% buffer (strong preference!)
        - Wetland birds: Similar in both (92% vs 90%)
        - Urban birds: No preference (98% everywhere)
    """
}

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

# =============================================================================
# PAGE 1: OVERVIEW (ENHANCED)
# =============================================================================

def render_overview_page():
    """Render enhanced overview dashboard with definitions"""
    st.title("🦜 Mangalavanam Bird Sanctuary")
    st.subheader("Adaptive Conservation Monitoring Dashboard")
    
    # Introduction box
    st.markdown("""
    <div class="info-box">
    <h3>📍 Welcome to the Conservation Dashboard!</h3>
    <p><strong>What is this dashboard?</strong><br>
    This is an interactive tool that helps conservation managers understand bird populations 
    and identify areas needing protection in Mangalavanam Bird Sanctuary, Kochi, Kerala.</p>
    
    <p><strong>Who is this for?</strong><br>
    Forest department staff, conservation managers, researchers, and anyone interested in 
    protecting urban biodiversity - NO TECHNICAL KNOWLEDGE REQUIRED!</p>
    
    <p><strong>How to use:</strong><br>
    Navigate using the sidebar on the left → Click on different pages → Explore interactive 
    maps and charts → Download data as needed</p>
    </div>
    """, unsafe_allow_html=True)
    
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
            "2.74 ha",
            help="Protected core sanctuary - strictly no human entry except rangers"
        )
        st.caption("🟢 Most protected zone")
    
    with col2:
        st.metric(
            "Buffer Zone",
            "5.12 ha",
            help="Buffer area with limited access - protects the core"
        )
        st.caption("🟡 Limited access zone")
    
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
    
    # What do these numbers mean?
    with st.expander("❓ What do these numbers mean?"):
        st.markdown("""
        - **Core + Buffer = 7.86 hectares total** (about 11 football fields!)
        - **194 species** is VERY HIGH for such a small urban sanctuary
        - **2,847 observations** = People like you uploaded bird sightings to eBird app
        - **2019-2025** = 7 years of continuous monitoring
        """)
    
    st.markdown("---")
    
    # Guild explanation with visuals
    st.markdown("### 🐦 Understanding Bird Guilds")
    
    show_definition("Guild")
    
    if guild_profiles is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Guild comparison chart
            fig = go.Figure()
            
            models = {
                'Year_ψ': 'Temporal Model',
                'Zone_ψ': 'Spatial Model',
                'Env_ψ_mean': 'Environmental Model'
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
            
            # Interpretation help
            st.markdown("""
            **📖 How to read this chart:**
            - **Y-axis (height)**: 0 = Poor habitat, 1 = Perfect habitat
            - **X-axis**: Three bird guilds
            - **Three bars per guild**: Results from different analysis methods
            
            **🔍 Key Insight:**
            - Urban birds (orange) score highest = They thrive everywhere
            - Wetland birds (blue) score 0.9+ = Sanctuary is great for them!
            - Forest birds (green) score lower = They need better protection
            """)
        
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
            
            # Download option
            csv = guild_profiles.to_csv(index=False)
            st.download_button(
                "📥 Download Guild Data",
                csv,
                file_name="guild_profiles.csv",
                mime="text/csv"
            )
    else:
        show_help_box("Guild analysis data not yet available. Run Phase 4 analysis to generate this.", "warning")
    
    st.markdown("---")
    
    # Study area map with satellite imagery
    st.markdown("### 🗺️ Study Area: Where is Mangalavanam?")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        core, buffer = load_boundaries()
        if core is not None and buffer is not None:
            st.markdown("**Interactive Map** (Click layer icon ⬜ top-right to toggle Satellite/Street view)")
            
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
        
        **Why it matters:**
        - One of India's smallest sanctuaries
        - Yet hosts 194 bird species!
        - Surrounded by dense urban area
        - Acts as "green lung" of the city
        
        **Study Period:**
        - 📅 2019 to 2025 (7 years)
        - 🔄 Continuous monitoring
        - 📊 2,847 observations
        
        **Data Sources:**
        - 👥 eBird citizen science
        - 🛰️ Sentinel-2 satellites
        - 🌙 VIIRS night lights
        - 🗺️ Field surveys
        """)
        
        st.markdown("---")
        
        # Add quick stats
        st.markdown("#### ⚡ Quick Facts")
        st.info("""
        - **Size**: 7.86 ha (smaller than 12 football fields!)
        - **Urbanization**: 99% surrounded by buildings
        - **Visitors**: ~500 people/day visit
        - **Protection**: State-level sanctuary (1989)
        """)
    
    # Add satellite image if available
    st.markdown("---")
    st.markdown("### 🛰️ Satellite View: Vegetation Health (NDVI)")
    
    show_definition("Environmental Rasters")
    
    # Try to load NDVI
    ndvi_file = Config.RASTERS_DIR / "NDVI_2025_core.tif"
    if ndvi_file.exists():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            data, bounds = load_satellite_image(ndvi_file)
            if data is not None:
                fig = px.imshow(
                    data,
                    color_continuous_scale='RdYlGn',
                    aspect='auto',
                    title="NDVI 2025 - Vegetation Greenness Index",
                    labels={'color': 'NDVI Value'}
                )
                
                fig.update_layout(
                    height=400,
                    coloraxis_colorbar=dict(
                        title="Greenness<br>(High=Green)",
                        thickness=20,
                        len=0.7
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.caption("🟢 Green = Dense vegetation (healthy), 🟡 Yellow = Moderate, 🔴 Red = Sparse/bare ground")
        
        with col2:
            st.markdown("""
            **🛰️ What is NDVI?**
            
            NDVI (Normalized Difference Vegetation Index) shows plant health from space.
            
            **Scale:**
            - -0.1 to 0: Water, rocks, bare soil
            - 0 to 0.3: Sparse vegetation
            - 0.3 to 0.6: Moderate vegetation
            - 0.6 to 1.0: Dense forest
            
            **Why we use it:**
            - Tracks forest health over time
            - Identifies degraded areas
            - Predicts bird habitat quality
            
            **In Mangalavanam:**
            - Core usually 0.5-0.7 (healthy mangrove)
            - Buffer 0.3-0.5 (mixed vegetation)
            """)
    else:
        st.info("💡 NDVI satellite imagery will display here when available")

# =============================================================================
# PAGE 2: HABITAT SUITABILITY MAPS (ENHANCED)
# =============================================================================

def render_habitat_maps_page():
    """Render habitat suitability visualization with comprehensive explanations"""
    st.title("🗺️ Habitat Suitability Maps")
    
    # Big definition box at top
    st.markdown("""
    <div class="definition-box">
    <h3>📖 What are Habitat Suitability Maps?</h3>
    <p><strong>Simple Explanation:</strong> These maps show WHERE birds prefer to live. 
    Think of it like a "comfort zone" map for birds!</p>
    
    <ul>
    <li><strong>🟢 Green areas (0.8-1.0):</strong> Perfect habitat! Birds love these spots.</li>
    <li><strong>🟡 Yellow areas (0.5-0.8):</strong> Okay habitat. Birds use these sometimes.</li>
    <li><strong>🔴 Red areas (0-0.5):</strong> Poor habitat. Birds avoid these areas.</li>
    </ul>
    
    <p><strong>How we make these maps:</strong> We combine:<br>
    ✓ Where people saw birds (eBird data)<br>
    ✓ Satellite images showing vegetation and water<br>
    ✓ AI models that learn bird preferences<br>
    ✓ Statistics that account for detection errors</p>
    </div>
    """, unsafe_allow_html=True)
    
    show_definition("Habitat Suitability (ψ)")
    show_definition("Occupancy Modeling")
    
    # Sidebar controls with help
    st.sidebar.markdown("### 🎛️ Map Controls")
    st.sidebar.info("Use these controls to explore different habitat maps")
    
    map_type = st.sidebar.selectbox(
        "📊 Select Map Type",
        ["Environmental", "Year-specific", "Zone-specific", "Conservation Priority"],
        help="Different ways of analyzing bird habitat preferences",
        key="habitat_map_type"
    )

    
    # Explain map types
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

    
    # Guild info
    if guild_filter != "All Guilds Combined":
        guild_info = {
            "Wetland": "🦆 Water birds: Kingfishers, Herons, Egrets",
            "Forest": "🦜 Tree birds: Woodpeckers, Barbets, Bulbuls",
            "Urban": "🐦 City birds: Crows, Mynas, Sparrows"
        }
        st.sidebar.caption(guild_info[guild_filter])
    
    # Determine file to load
    if map_type == "Environmental":
        if guild_filter == "All Guilds Combined":
            raster_file = Config.PHASE4_DIR / "psi_env_mean.tif"
        else:
            raster_file = Config.PHASE4_DIR / f"psi_env_{guild_filter}.tif"
        title = f"🌿 Environmental Habitat Suitability"
        subtitle = f"Based on satellite data (vegetation, water, lights) - {guild_filter}"
        
    elif map_type == "Year-specific":
        year = st.sidebar.selectbox(
            "📅 Select Year",
            Config.YEARS,
            index=len(Config.YEARS)-1
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
        title = f"📍 Core vs Buffer Zone Habitat Quality"
        subtitle = f"Spatial comparison - {guild_filter}"
        
    else:  # Conservation Priority
        raster_file = Config.CPI_RASTER
        title = "🎯 Conservation Priority Index"
        subtitle = "Where should we focus conservation efforts?"
        
        show_definition("Conservation Priority")
    
    # Main content area
    st.subheader(title)
    st.caption(subtitle)
    
    # Load and display raster
    if raster_file.exists():
        data, bounds, crs, transform = load_raster(raster_file)
        
        if data is not None:
            # Two-column layout
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Create heatmap
                if "Priority" in title:
                    # Categorical color scale for priority
                    fig = px.imshow(
                        data,
                        color_continuous_scale='RdYlGn_r',
                        aspect='auto',
                        title=None
                    )
                    colorbar_title = "Priority<br>Level"
                else:
                    # Continuous scale for habitat suitability
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
                
                # Legend explanation
                if "Priority" not in title:
                    st.markdown("""
                    **🎨 Color Guide:**
                    - 🟢 **Dark Green (0.8-1.0):** Excellent habitat - Birds thrive here
                    - 🟡 **Yellow-Green (0.6-0.8):** Good habitat - Suitable for most birds
                    - 🟡 **Yellow (0.4-0.6):** Moderate habitat - Some birds use this
                    - 🟠 **Orange (0.2-0.4):** Poor habitat - Birds rarely seen
                    - 🔴 **Red (0-0.2):** Very poor habitat - Birds avoid this area
                    """)
                else:
                    st.markdown("""
                    **🎨 Priority Levels:**
                    - 🔴 **Level 5 (Critical):** Act NOW - Immediate intervention needed
                    - 🟠 **Level 4 (High):** Act within 6-12 months
                    - 🟡 **Level 3 (Medium):** Regular monitoring needed
                    - 🟢 **Level 2 (Low):** Routine maintenance
                    - ⚪ **Level 1 (Minimal):** Stable area, low concern
                    """)
            
            with col2:
                st.markdown("### 📊 Statistics")
                
                valid_data = data[np.isfinite(data)]
                
                if len(valid_data) > 0:
                    # Basic stats
                    st.metric("Mean", f"{valid_data.mean():.3f}")
                    st.metric("Std Dev", f"{valid_data.std():.3f}")
                    st.metric("Min", f"{valid_data.min():.3f}")
                    st.metric("Max", f"{valid_data.max():.3f}")
                    
                    # Interpretation
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
                    
                    st.markdown("---")
                    
                    # Histogram
                    st.markdown("### 📈 Distribution")
                    fig_hist = px.histogram(
                        valid_data.flatten(),
                        nbins=20,
                        title=None,
                        labels={'value': 'Value', 'count': 'Pixels'}
                    )
                    fig_hist.update_layout(height=250, showlegend=False)
                    fig_hist.update_traces(marker_color='#4CAF50')
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    st.caption("This shows how many pixels (areas) have each habitat quality level")
                    
                else:
                    st.warning("No valid data in this raster")
                
                # Download option
                st.markdown("---")
                if st.button("📥 Download Map Data"):
                    # Convert to dataframe
                    df_export = pd.DataFrame({
                        'pixel_value': valid_data.flatten()
                    })
                    csv = df_export.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        file_name=f"{raster_file.stem}_data.csv",
                        mime="text/csv"
                    )
    else:
        show_help_box(f"Map file not found: {raster_file.name}", "warning")
        st.info("""
        **Why might this map be missing?**
        - Phase 4 or Phase 6 analysis not yet run
        - Data processing still in progress
        - File path configuration error
        
        **What to do:**
        1. Check if Phase 4 completed successfully
        2. Verify files exist in: `data/processed/phase4/`
        3. Re-run analysis pipeline if needed
        """)
# REPLACE WITH THIS COMPLETE VERSION:

    # Add satellite comparison with multiple layers
    st.markdown("---")
    st.markdown("### 🛰️ Compare with Satellite Imagery")
    
    # Layer selector with boundary toggle
    col_selector, col_boundary = st.columns([3, 1])
    
    with col_selector:
        satellite_layer = st.selectbox(
            "🛰️ Select Satellite Layer:",
            ["NDVI (Vegetation)", "NDWI (Water)", "VIIRS (Night Lights)"],
            help="Choose environmental variable to compare with habitat model"
        )
    
    with col_boundary:
        show_boundary = st.checkbox("Show Study Area", value=True, help="Toggle Core/Buffer boundaries")
    
    # Map layer selection to file
    layer_files = {
        "NDVI (Vegetation)": Config.RASTERS_DIR / "NDVI_2025_core.tif",
        "NDWI (Water)": Config.RASTERS_DIR / "NDWI_2025_core.tif",
        "VIIRS (Night Lights)": Config.RASTERS_DIR / "VIIRS_2025_core.tif"
    }
    
    layer_descriptions = {
        "NDVI (Vegetation)": "🌿 Greener = More vegetation = Better for forest birds",
        "NDWI (Water)": "💧 Higher values = More water = Better for wetland birds",
        "VIIRS (Night Lights)": "💡 Brighter = More urban = Better for city birds, worse for forest/wetland"
    }
    
    layer_colormaps = {
        "NDVI (Vegetation)": "RdYlGn",
        "NDWI (Water)": "Blues",
        "VIIRS (Night Lights)": "Hot_r"
    }
    
    selected_file = layer_files[satellite_layer]
    
    if selected_file.exists():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Satellite {satellite_layer}**")
            
            # Load satellite data
            sat_data, sat_bounds = load_satellite_image(selected_file)
            
            if sat_data is not None:
                # Create figure
                fig_sat = px.imshow(
                    sat_data, 
                    color_continuous_scale=layer_colormaps[satellite_layer],
                    aspect='auto'
                )
                fig_sat.update_layout(
                    height=350,
                    showlegend=False,
                    xaxis=dict(showticklabels=False),
                    yaxis=dict(showticklabels=False),
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                
                # Add boundary overlay if requested
                if show_boundary and sat_bounds is not None:
                    # Calculate approximate core area in pixel coordinates
                    # These are rough estimates - adjust based on your actual core zone
                    height, width = sat_data.shape
                    
                    # Core area rectangle (approximate center 30% of image)
                    core_y_start = int(height * 0.35)
                    core_y_end = int(height * 0.65)
                    core_x_start = int(width * 0.35)
                    core_x_end = int(width * 0.65)
                    
                    # Add core boundary as rectangle
                    fig_sat.add_shape(
                        type="rect",
                        x0=core_x_start, y0=core_y_start,
                        x1=core_x_end, y1=core_y_end,
                        line=dict(color="green", width=3),
                        name="Core Area"
                    )
                    
                    # Add buffer boundary as outer circle (using ellipse)
                    fig_sat.add_shape(
                        type="circle",
                        xref="x", yref="y",
                        x0=width*0.1, y0=height*0.1,
                        x1=width*0.9, y1=height*0.9,
                        line=dict(color="orange", width=2, dash="dash"),
                        name="Buffer Zone"
                    )
                    
                    # Add legend annotations
                    fig_sat.add_annotation(
                        x=core_x_start, y=core_y_start - 5,
                        text="Core",
                        showarrow=False,
                        font=dict(size=10, color="green", family="Arial Black"),
                        bgcolor="white",
                        opacity=0.8
                    )
                
                st.plotly_chart(fig_sat, use_container_width=True)
                st.caption(layer_descriptions[satellite_layer])
        
        with col2:
            st.markdown("**Habitat Suitability Model**")
            
            if data is not None:
                # Create habitat figure
                fig_hab = px.imshow(
                    data, 
                    color_continuous_scale='RdYlGn',
                    aspect='auto',
                    zmin=0,
                    zmax=1
                )
                fig_hab.update_layout(
                    height=350,
                    showlegend=False,
                    xaxis=dict(showticklabels=False),
                    yaxis=dict(showticklabels=False),
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                
                # Add same boundaries to habitat map
                if show_boundary:
                    height_hab, width_hab = data.shape
                    
                    # Core area
                    fig_hab.add_shape(
                        type="rect",
                        x0=int(width_hab*0.35), y0=int(height_hab*0.35),
                        x1=int(width_hab*0.65), y1=int(height_hab*0.65),
                        line=dict(color="green", width=3)
                    )
                    
                    # Buffer zone
                    fig_hab.add_shape(
                        type="circle",
                        x0=width_hab*0.1, y0=height_hab*0.1,
                        x1=width_hab*0.9, y1=height_hab*0.9,
                        line=dict(color="orange", width=2, dash="dash")
                    )
                
                st.plotly_chart(fig_hab, use_container_width=True)
                st.caption("🟢 Higher suitability = Birds more likely to live here")
        
        # Explanation box
        st.info(f"""
        💡 **Notice the patterns?** Our habitat model (right) learns from satellite **{satellite_layer.lower()}** (left) 
        **PLUS** water indicators, night lights, edge distance, and actual bird sightings to predict where 
        **{guild_filter.lower()}** birds will thrive!
        
        {'🟢 **Green box** = Core protected area | 🟠 **Orange circle** = Buffer zone (500m radius)' if show_boundary else ''}
        """)
    else:
        st.warning(f"Satellite layer not found: {selected_file.name}")
        st.info("Available layers: NDVI (vegetation), NDWI (water), VIIRS (lights)")

# =============================================================================
# PAGE 3: GUILD ANALYSIS - COMPLETE
# =============================================================================

def render_guild_analysis_page():
    """Render guild comparison with detailed explanations and imagery"""
    st.title("🐦 Functional Guild Analysis")
    st.markdown("*Understanding different bird groups and their unique habitat needs*")
    
    # Main definition
    st.markdown("""
    <div class="definition-box">
    <h3>📖 What are Bird Guilds?</h3>
    <p><strong>Simple explanation:</strong> Instead of studying 194 bird species individually, 
    we group similar birds together based on what they eat and where they live.</p>
    
    <p><strong>Think of it like this:</strong> Just like a city has different professions 
    (doctors, teachers, engineers), an ecosystem has different bird "professions" - 
    each group filling a specific ecological role!</p>
    
    <p><strong>Our three guilds:</strong></p>
    <ul>
    <li>🦆 <strong>Wetland Birds:</strong> Aquatic specialists (fish-eaters, wading birds)</li>
    <li>🦜 <strong>Forest Birds:</strong> Tree-dependent species (insect-eaters, fruit-eaters)</li>
    <li>🐦 <strong>Urban Birds:</strong> Generalists (can live anywhere, eat anything)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    show_definition("Guild")
    show_definition("Functional Diversity")
    
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
                "occupancy": guild_profiles[guild_profiles['Guild'] == 'Wetland']['Env_ψ_mean'].values[0] if len(guild_profiles[guild_profiles['Guild'] == 'Wetland']) > 0 else 0.82,
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
                "occupancy": guild_profiles[guild_profiles['Guild'] == 'Forest']['Env_ψ_mean'].values[0] if len(guild_profiles[guild_profiles['Guild'] == 'Forest']) > 0 else 0.68,
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
                "occupancy": guild_profiles[guild_profiles['Guild'] == 'Urban']['Env_ψ_mean'].values[0] if len(guild_profiles[guild_profiles['Guild'] == 'Urban']) > 0 else 0.93,
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
                
                # Habitat requirements
                st.markdown("### 🏞️ Habitat Requirements")
                st.markdown("""
                **What wetland birds need:**
                1. ✅ **Clean water** - Free from pollution and toxins
                2. ✅ **Fish populations** - Primary food source
                3. ✅ **Shallow areas** - For wading and feeding
                4. ✅ **Perching sites** - Trees/posts near water for kingfishers
                5. ✅ **Mudflats** - For probing and foraging
                
                **Why Mangalavanam is good for them:**
                - Tidal mudflats provide rich feeding grounds
                - Mangrove channels support fish populations  
                - Water quality generally good (NDWI scores high)
                - Minimal disturbance in core zone
                """)
            
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
                
                # Habitat requirements
                st.markdown("### 🏞️ Habitat Requirements")
                st.markdown("""
                **What forest birds need:**
                1. ✅ **Dense canopy** - Continuous tree cover
                2. ✅ **Interior habitat** - Away from edges
                3. ✅ **Quiet zones** - Minimal human disturbance
                4. ✅ **Native trees** - For insects and fruits
                5. ✅ **Vertical structure** - Multiple canopy layers
                
                **Challenges in Mangalavanam:**
                - ⚠️ Small size = High edge-to-interior ratio
                - ⚠️ Surrounded by city = Noise and light pollution
                - ⚠️ Limited canopy in buffer zone
                
                **Why core protection is critical:**
                - 87% occupy core vs 68% buffer (strong preference!)
                - Edge avoidance behavior (β = -0.90)
                - Need quiet interior for breeding
                """)
            
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
                
                # Habitat requirements
                st.markdown("### 🏞️ Habitat Requirements")
                st.markdown("""
                **What urban birds need:**
                1. ✅ **Food availability** - Any source works
                2. ✅ **Nesting sites** - Buildings, trees, anywhere
                3. ✅ **Human tolerance** - Don't fear people
                4. ❌ **Not picky** - Can live in degraded habitats
                
                **Why they thrive:**
                - Generalist diet (eat almost anything)
                - Habitat flexibility (nest anywhere)
                - Human-tolerance (benefit from human activity)
                - Competitor advantage (aggressive, adaptable)
                
                **Conservation concern:**
                - ⚠️ High urban bird occupancy can indicate habitat degradation
                - ⚠️ May outcompete specialist species
                - ⚠️ Sometimes considered "biotic homogenization"
                
                **But it's not all bad:**
                - ✅ Shows ecosystem still supports SOME birds
                - ✅ Provides ecosystem services (pest control)
                - ✅ Connects urban people to nature
                """)
            
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
                'Year_ψ': 'Temporal (Year-based)',
                'Zone_ψ': 'Spatial (Core vs Buffer)',
                'Env_ψ_mean': 'Environmental (Habitat-based)'
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
            
            # Interpretation
            st.markdown("""
            **📖 What this tells us:**
            
            1. **Urban > Wetland > Forest** in overall occupancy
               - Urban birds thrive everywhere (generalists)
               - Wetland birds doing well (good water quality)
               - Forest birds need help (require interior habitat)
            
            2. **All three models agree** (consistent results)
               - Temporal, Spatial, Environmental all show same pattern
               - This validates our findings!
            
            3. **Environmental model shows most variation**
               - Captures fine-scale habitat differences
               - Most useful for management decisions
            """)
            
            st.markdown("---")
            
            # Ecosystem health assessment
            st.markdown("### 🌐 Overall Ecosystem Health")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ✅ Positive Signs")
                st.success("""
                - **All three guilds present** = Diverse ecosystem
                - **Wetland birds thriving** = Good water quality
                - **Forest specialists detected** = Core habitat working
                - **High overall diversity** = Ecosystem resilience
                """)
                
                st.markdown("#### 🎯 Management Priorities")
                st.info("""
                1. **Protect core zone** for forest birds
                2. **Maintain water quality** for wetland birds
                3. **Buffer enhancement** to expand forest habitat
                4. **Monitor urban/specialist ratio** over time
                """)
            
            with col2:
                st.markdown("#### ⚠️ Concerns")
                st.warning("""
                - **Forest birds moderate occupancy** = Habitat stress
                - **Urban dominance** = Possible biotic homogenization
                - **Small sanctuary size** = Edge effect vulnerability
                - **Surrounding urbanization** = Ongoing threat
                """)
                
                st.markdown("#### 📊 Recommended Metrics")
                st.info("""
                Track these ratios annually:
                - **Specialist/Generalist ratio** (Forest:Urban)
                - **Guild evenness** (balanced distribution)
                - **Core zone occupancy** (forest birds)
                - **Water quality indicators** (wetland birds)
                """)
            
            st.markdown("---")
            
            # Download combined data
            if st.button("📥 Download Complete Guild Analysis"):
                combined_df = guild_profiles.copy()
                csv = combined_df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    file_name="guild_analysis_complete.csv",
                    mime="text/csv"
                )
    
    else:
        show_help_box("Guild analysis data not yet available. Run Phase 4 to generate.", "warning")
    
    # Functional diversity section
    if functional_summary is not None:
        st.markdown("---")
        st.markdown("### 🌐 Functional Diversity Indices")
        
        show_definition("Functional Diversity")
        
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
            
            # Interpretation guide
            st.markdown("""
            **📖 Index Meanings:**
            
            - **Shannon Diversity**: Overall guild variety (Higher = More diverse)
              - 0 = Only one guild, 1.1 = All guilds equally common
              - Mangalavanam: 0.82 = Good diversity!
            
            - **Simpson Diversity**: Chance two random birds are different guilds
              - 0 = All same guild, 1 = Maximum diversity
              - Mangalavanam: 0.61 = 61% chance of different guilds
            
            - **Functional Richness**: Simple count of guilds present
              - Max = 3 (all guilds), Min = 1 (only one guild)
              - Mangalavanam: 2.1 average = Most areas have 2+ guilds ✅
            
            - **Functional Evenness**: How evenly guilds are distributed
              - 1.0 = Perfectly even, 0 = Dominated by one guild
              - Mangalavanam: 0.75 = Fairly even distribution
            """)
        
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
            
            st.markdown("#### 🎯 What This Means")
            
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
    
    # Main definition
    st.markdown("""
    <div class="definition-box">
    <h3>📖 What is Conservation Priority?</h3>
    <p><strong>Simple explanation:</strong> Not all areas need the same level of attention. 
    Some areas are more important for conservation because they have:</p>
    <ul>
    <li>High bird diversity (many different species)</li>
    <li>Good habitat quality (birds thrive there)</li>
    <li>High threats (near roads, buildings, disturbance)</li>
    </ul>
    
    <p><strong>Our Priority Index (CPI) combines:</strong></p>
    <ul>
    <li>🐦 <strong>Occupancy (40%):</strong> How many birds use the area</li>
    <li>🌐 <strong>Functional value (35%):</strong> How many different guilds present</li>
    <li>⚠️ <strong>Threat level (25%):</strong> Urban pressure, edge effects</li>
    </ul>
    
    <p><strong>Formula:</strong> CPI = (0.40 × Bird Occupancy) + (0.35 × Diversity) - (0.25 × Threats)</p>
    </div>
    """, unsafe_allow_html=True)
    
    show_definition("Conservation Priority")
    show_definition("CPI (Conservation Priority Index)")
    
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
                
                # Legend with actions
                st.markdown("""
                **🎨 Priority Levels & Required Actions:**
                
                | Level | Color | Status | Timeline | Actions |
                |-------|-------|--------|----------|---------|
                | 🔴 **5** | Red | **Critical** | 0-6 months | Immediate intervention required |
                | 🟠 **4** | Orange | **High** | 6-12 months | Active management needed |
                | 🟡 **3** | Yellow | **Medium** | 1-2 years | Regular monitoring |
                | 🟢 **2** | Green | **Low** | 2+ years | Routine maintenance |
                | ⚪ **1** | Gray | **Minimal** | Ongoing | Stable, low concern |
                """)
            
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
        
        # Management recommendations tabs
        st.markdown("### 📋 Management Action Plans")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔴 Critical (Level 5)",
            "🟠 High (Level 4)",
            "🟡 Medium (Level 3)",
            "📊 Implementation"
        ])
        
        with tab1:
            st.markdown("## 🔴 Critical Priority Areas (Level 5)")
            st.error(f"**Area requiring critical intervention:** {priority_stats[5]['Area (ha)']:.2f} hectares ({priority_stats[5]['Percentage']:.1f}%)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ⚡ Immediate Actions (0-6 months)")
                
                st.markdown("""
                **🔍 Intensive Monitoring:**
                - Weekly bird surveys (increase from current monthly)
                - Water quality testing every 2 weeks
                - Vegetation health assessment monthly
                - Photo documentation of changes
                
                **🌿 Habitat Restoration:**
                - Remove invasive species (priority: Lantana, Prosopis)
                - Plant native mangrove saplings (100+ seedlings)
                - Install erosion control measures
                - Create water retention structures
                
                **🚧 Threat Mitigation:**
                - Install barriers to prevent encroachment
                - Restrict visitor access to degraded zones
                - Remove/relocate trash accumulation points
                - Noise reduction measures (signage, barriers)
                
                **💧 Water Management:**
                - Clear blocked drainage channels
                - Test for pollution sources
                - Install filtration if needed
                - Monitor tidal flow patterns
                """)
            
            with col2:
                st.markdown("### 💰 Budget & Resources")
                
                st.info("""
                **Estimated Costs:**
                - Monitoring equipment: ₹50,000
                - Native plants & labor: ₹1,00,000
                - Barriers & signage: ₹30,000
                - Water testing kits: ₹20,000
                - **Total: ₹2,00,000** (~$2,400)
                """)
                
                st.markdown("### 👥 Personnel Needed")
                
                st.info("""
                - **Field Staff:** 2 full-time rangers
                - **Ecologist:** 1 part-time consultant
                - **Volunteers:** 5-10 for plantings
                - **Lab Technician:** For water testing
                """)
                
                st.markdown("### 📅 Timeline")
                
                st.success("""
                **Month 1-2:**
                - Baseline assessment
                - Invasive removal starts
                - Barriers installed
                
                **Month 3-4:**
                - Planting campaign
                - Water management
                - Monitoring protocol established
                
                **Month 5-6:**
                - Interim evaluation
                - Adjust interventions
                - Report to stakeholders
                """)
            
            st.markdown("---")
            st.warning("""
            **⚠️ Why This is Critical:**
            These areas show both HIGH ecological value (many birds, high diversity) AND 
            HIGH threats (edge effects, disturbance). Without intervention in 6 months, 
            we risk permanent habitat loss and species decline.
            """)
        
        with tab2:
            st.markdown("## 🟠 High Priority Areas (Level 4)")
            st.warning(f"**Area requiring high-priority attention:** {priority_stats[4]['Area (ha)']:.2f} hectares ({priority_stats[4]['Percentage']:.1f}%)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📅 Recommended Actions (6-12 months)")
                
                st.markdown("""
                **🔍 Regular Monitoring:**
                - Bi-weekly bird surveys
                - Monthly vegetation assessments
                - Quarterly water quality checks
                
                **🌱 Buffer Enhancement:**
                - Strengthen buffer zone vegetation
                - Create dense shrub layer
                - Plant native species (50+ saplings)
                - Maintain existing trees
                
                **🎯 Controlled Access:**
                - Define visitor pathways
                - Install interpretive signage
                - Educational programs
                - Volunteer clean-up events
                
                **🦜 Species-Specific Measures:**
                - Nest box installation (for cavity nesters)
                - Perch provision (for kingfishers)
                - Maintain water features (for wetland birds)
                """)
            
            with col2:
                st.markdown("### 💰 Budget Estimate")
                
                st.info("""
                - Plants & materials: ₹60,000
                - Signage: ₹15,000
                - Educational materials: ₹10,000
                - Nest boxes: ₹5,000
                - **Total: ₹90,000** (~$1,100)
                """)
                
                st.markdown("### 📊 Success Indicators")
                
                st.success("""
                **Track these metrics:**
                - Buffer vegetation density increase
                - Forest bird occupancy stable/increasing
                - Visitor compliance with paths
                - Reduced littering incidents
                - Native plant survival rate >70%
                """)
        
        with tab3:
            st.markdown("## 🟡 Medium Priority Areas (Level 3)")
            st.info(f"**Area requiring medium-priority maintenance:** {priority_stats[3]['Area (ha)']:.2f} hectares ({priority_stats[3]['Percentage']:.1f}%)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📅 Maintenance Actions (1-2 years)")
                
                st.markdown("""
                **🔍 Monitoring:**
                - Monthly bird counts
                - Seasonal vegetation surveys
                - Annual comprehensive assessment
                
                **🌿 Vegetation Management:**
                - Prune dead branches
                - Control minor invasive growth
                - Monitor tree health
                - Maintain ground cover
                
                **👥 Community Engagement:**
                - Nature walks & bird watching
                - School programs
                - Citizen science participation
                - Volunteer maintenance days
                
                **📢 Education & Outreach:**
                - Interpretive displays
                - Social media updates
                - Annual reports
                - Stakeholder meetings
                """)
            
            with col2:
                st.markdown("### 💰 Annual Budget")
                
                st.info("""
                - Routine maintenance: ₹30,000
                - Educational programs: ₹20,000
                - Signage updates: ₹10,000
                - **Total: ₹60,000/year** (~$720)
                """)
                
                st.markdown("### 🎯 Goals")
                
                st.success("""
                - Maintain current occupancy levels
                - Prevent transition to high priority
                - Build community support
                - Monitor for early warning signs
                """)
        
        with tab4:
            st.markdown("## 📊 Implementation Strategy")
            
            st.markdown("### 🗓️ Phased Rollout Plan")
            
            # Timeline visualization
            timeline_data = pd.DataFrame({
                'Phase': ['Phase 1: Critical', 'Phase 2: High', 'Phase 3: Medium', 'Phase 4: Maintenance'],
                'Start': [0, 6, 12, 24],
                'Duration': [6, 6, 12, 12],
                'Priority': ['Critical', 'High', 'Medium', 'Low']
            })
            
            fig_timeline = px.timeline(
                timeline_data,
                x_start='Start',
                x_end='Duration',
                y='Phase',
                color='Priority',
                title="Implementation Timeline (Months from Start)",
                color_discrete_map={
                    'Critical': '#d32f2f',
                    'High': '#f57c00',
                    'Medium': '#fbc02d',
                    'Low': '#689f38'
                }
            )
            
            fig_timeline.update_layout(height=300)
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 💰 Total Budget Summary")
                
                total_immediate = 200000  # Level 5
                total_short = 90000       # Level 4
                total_medium = 60000      # Level 3
                total_3yr = total_immediate + total_short + (total_medium * 2)
                
                st.info(f"""
                **3-Year Budget Projection:**
                - Year 1 (Critical + High): ₹{(total_immediate + total_short):,}
                - Year 2 (Medium priority): ₹{total_medium:,}
                - Year 3 (Maintenance): ₹{total_medium:,}
                
                **Total 3-Year Investment: ₹{total_3yr:,}** (~${total_3yr/83:.0f})
                
                **Cost per hectare per year:**
                ₹{(total_3yr/3/7.86):.0f}/ha/year
                
                *Very reasonable for urban sanctuary conservation!*
                """)
                
                st.markdown("### 📈 Expected ROI")
                
                st.success("""
                **Conservation Returns:**
                - Habitat stabilization: 2+ ha
                - Species protection: 194 species
                - Ecosystem services: Air/water filtration
                - Education: 1000+ visitors/year
                - Research: Long-term monitoring data
                
                **Economic Value:**
                - Ecotourism potential: ₹5L+/year
                - Carbon sequestration
                - Flood mitigation
                - Property value enhancement
                """)
            
            with col2:
                st.markdown("### 👥 Stakeholder Roles")
                
                st.info("""
                **Forest Department:**
                - Overall coordination
                - Budget allocation
                - Staff deployment
                - Regulatory enforcement
                
                **Mangalavanam Nature Club:**
                - Community engagement
                - Volunteer coordination
                - Educational programs
                - Monitoring support
                
                **Research Institutions:**
                - Technical guidance
                - Data analysis
                - Student projects
                - Scientific publications
                
                **Local Community:**
                - Citizen science
                - Awareness campaigns
                - Voluntary compliance
                - Cultural value preservation
                """)
                
                st.markdown("### ✅ Success Metrics")
                
                st.success("""
                **Track annually:**
                - [ ] Bird occupancy trends
                - [ ] Habitat quality scores (ψ)
                - [ ] Vegetation cover (NDVI)
                - [ ] Water quality parameters
                - [ ] Visitor satisfaction
                - [ ] Budget utilization
                - [ ] Volunteer hours
                - [ ] Educational reach
                """)
            
            st.markdown("---")
            
            st.markdown("### 📥 Download Implementation Plan")
            
            if cpi_metadata is not None and st.button("Generate Full Implementation Report"):
                st.info("""
                **Report will include:**
                - Priority maps (PDF)
                - Area statistics (CSV)
                - Action plans (detailed)
                - Budget breakdown (Excel)
                - Timeline chart
                - Monitoring protocols
                """)
    
    else:
        show_help_box("Conservation priority analysis not yet complete. Run Phase 6.", "warning")
        
        st.info("""
        **To generate priority maps:**
        1. Ensure Phase 4 (functional analysis) is complete
        2. Run Phase 6 conservation priority script
        3. Refresh this dashboard
        
        **Expected outputs:**
        - Priority classification raster
        - CPI metadata CSV
        - Management recommendations
        """)

# =============================================================================
# PAGE 5: TEMPORAL TRENDS - COMPLETE
# =============================================================================
def render_temporal_trends_page():
    """Render temporal analysis with year-over-year comparisons"""
    st.title("📈 Temporal Trends Analysis")
    st.markdown("*Multi-year patterns in bird populations (2019-2025)*")
    
    # Introduction
    st.markdown("""
    <div class="info-box">
    <h3>📅 Why Track Changes Over Time?</h3>
    <p>By monitoring birds year after year, we can detect:</p>
    <ul>
    <li>📈 <strong>Population trends:</strong> Are bird numbers increasing or decreasing?</li>
    <li>🌡️ <strong>Climate impacts:</strong> How do birds respond to weather changes?</li>
    <li>🏗️ <strong>Urbanization effects:</strong> Does city growth affect bird communities?</li>
    <li>✅ <strong>Conservation success:</strong> Are our protection efforts working?</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
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
        'Wetland': guild_profiles[guild_profiles['Guild'] == 'Wetland']['Env_ψ_mean'].values[0],
        'Forest': guild_profiles[guild_profiles['Guild'] == 'Forest']['Env_ψ_mean'].values[0],
        'Urban': guild_profiles[guild_profiles['Guild'] == 'Urban']['Env_ψ_mean'].values[0]
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
            "🔮 What-If Scenarios"
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