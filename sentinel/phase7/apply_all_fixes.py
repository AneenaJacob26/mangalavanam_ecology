#!/usr/bin/env python3
"""
CONSOLIDATED FIX SCRIPT
Applies all 3 critical fixes to the Mangalavanam dashboard

Fixes applied:
1. LSTM forecasting (Urban guild saturation detection)
2. Dashboard satellite maps (exact boundaries)
3. Restoration timeline (Priority_Level error)

Author: MSc Thesis Support
Date: February 2026
"""

import shutil
from pathlib import Path
from datetime import datetime
import sys

print("="*80)
print("MANGALAVANAM DASHBOARD - CONSOLIDATED FIX SCRIPT")
print("="*80)
print()
print("This script will automatically apply 3 critical fixes:")
print("  1. ✅ LSTM forecasting (fix Urban guild)")
print("  2. ✅ Satellite maps (exact boundaries)")
print("  3. ✅ Restoration timeline (Priority_Level)")
print()
print("="*80)
print()

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(".")
BACKUP_DIR = BASE_DIR / "backups" / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# File paths
LSTM_CURRENT = BASE_DIR / "sentinel" / "phase7" / "lstm_forecasting.py"
APP_CURRENT = BASE_DIR / "sentinel" / "dashboard" / "app.py"

# Create backup directory
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

print(f"📁 Backup directory created: {BACKUP_DIR}")
print()

# =============================================================================
# FIX 1: LSTM FORECASTING
# =============================================================================

print("[FIX 1/3] Updating LSTM Forecasting Script...")
print("-" * 80)

# Backup current file
if LSTM_CURRENT.exists():
    backup_lstm = BACKUP_DIR / "lstm_forecasting_ORIGINAL.py"
    shutil.copy2(LSTM_CURRENT, backup_lstm)
    print(f"✓ Backed up current LSTM script to: {backup_lstm}")
else:
    print("⚠️  Warning: lstm_forecasting.py not found at expected location")
    print(f"   Expected: {LSTM_CURRENT}")
    create_new = input("   Create new file? (y/n): ").lower().strip()
    if create_new != 'y':
        print("   Skipping LSTM fix.")
        LSTM_CURRENT = None

if LSTM_CURRENT:
    # Write the FIXED LSTM script
    lstm_fixed_content = '''#!/usr/bin/env python3
"""
Phase 7A: LSTM Population Forecasting (FIXED FOR SATURATED GUILDS)
Handles guilds with constant 100% detection (like Urban birds)

KEY FIX: Skip LSTM for guilds with no variation, use simple forecast instead
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
PHASE2_DIR = Path("data/processed/phase2")
PHASE7_DIR = Path("data/processed/phase7")
PHASE7_DIR.mkdir(exist_ok=True, parents=True)

GUILDS = ['Wetland', 'Forest', 'Urban']
YEARS_HISTORICAL = list(range(2019, 2026))
YEARS_FORECAST = list(range(2026, 2031))

# LSTM hyperparameters (optimized for small dataset)
LOOKBACK_WINDOW = 3
DROPOUT_RATE = 0.3
EPOCHS = 200
BATCH_SIZE = 2
MIN_VARIANCE = 0.01  # If variance < this, skip LSTM

print("="*80)
print("PHASE 7A: LSTM POPULATION FORECASTING (FIXED)")
print("="*80)
print()

# Load data
print("[1/6] Loading historical bird observation data...")
timeseries_file = PHASE2_DIR / "guild_pixel_timeseries_with_covariates.csv"

if not timeseries_file.exists():
    print(f"❌ ERROR: {timeseries_file} not found!")
    exit(1)

df = pd.read_csv(timeseries_file)
print(f"✓ Loaded {len(df):,} observations")
print()

# Create time series
print("[2/6] Aggregating observations by guild and year...")

guild_timeseries = []

for guild in GUILDS:
    guild_data = df[df['guild'] == guild]
    
    for year in YEARS_HISTORICAL:
        year_data = guild_data[guild_data['year'] == year]
        
        if len(year_data) > 0:
            record = {
                'guild': guild,
                'year': year,
                'n_observations': len(year_data),
                'n_detections': year_data['detected'].sum(),
                'detection_rate': year_data['detected'].mean(),
                'mean_ndvi': year_data['ndvi'].mean(),
                'mean_ndwi': year_data['ndwi'].mean(),
                'mean_viirs': year_data['viirs'].mean(),
                'mean_dist_edge': year_data['dist_edge'].mean()
            }
            guild_timeseries.append(record)

ts_df = pd.DataFrame(guild_timeseries)
ts_df.to_csv(PHASE7_DIR / "historical_timeseries.csv", index=False)

print(f"✓ Created time series: {len(ts_df)} records")
print()

# Build LSTM model
print("[3/6] Building LSTM Neural Network (with saturation detection)...")
print()

def create_lstm_model(n_features, lookback=LOOKBACK_WINDOW):
    """LSTM model with sigmoid output for 0-1 constraint"""
    model = keras.Sequential([
        layers.LSTM(32, activation='tanh', return_sequences=True,
                   input_shape=(lookback, n_features)),
        layers.Dropout(DROPOUT_RATE),
        layers.LSTM(16, activation='tanh', return_sequences=False),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # FORCE 0-1 range
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['mae']
    )
    
    return model

def create_sequences(data, target_col, feature_cols, lookback=LOOKBACK_WINDOW):
    """Create input sequences for LSTM"""
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[feature_cols].iloc[i:i+lookback].values)
        y.append(data[target_col].iloc[i+lookback])
    return np.array(X), np.array(y)

FEATURES = ['mean_ndvi', 'mean_ndwi', 'mean_viirs', 'mean_dist_edge', 'detection_rate']
TARGET = 'detection_rate'

models = {}
forecasts = {}
scalers = {}

# Train models
print("[4/6] Training models (checking for saturation)...")
print()

for guild in GUILDS:
    print(f"{'='*60}")
    print(f"{guild} Guild")
    print(f"{'='*60}")
    
    guild_ts = ts_df[ts_df['guild'] == guild].sort_values('year').reset_index(drop=True)
    
    # CHECK FOR SATURATION
    detection_variance = guild_ts['detection_rate'].var()
    mean_detection = guild_ts['detection_rate'].mean()
    
    print(f"  Detection rate: mean={mean_detection:.3f}, variance={detection_variance:.4f}")
    
    # If variance is too low OR mean is too high, skip LSTM
    if detection_variance < MIN_VARIANCE or mean_detection > 0.95:
        print(f"  ⚠️  SATURATED GUILD (detection ~100% with no variation)")
        print(f"  → Skipping LSTM, using simple forecast: 'Stay at {mean_detection:.3f}'")
        print()
        
        models[guild] = None
        forecasts[guild] = {'method': 'constant', 'value': mean_detection}
        continue
    
    # Continue with LSTM for guilds with variation
    if len(guild_ts) < LOOKBACK_WINDOW + 1:
        print(f"  ⚠️  Not enough data")
        continue
    
    # Normalize
    scaler = MinMaxScaler()
    guild_ts[FEATURES] = scaler.fit_transform(guild_ts[FEATURES])
    scalers[guild] = scaler
    
    # Create sequences
    X, y = create_sequences(guild_ts, TARGET, FEATURES, lookback=LOOKBACK_WINDOW)
    
    if len(X) < 2:
        print(f"  ⚠️  Not enough sequences")
        continue
    
    # Split
    X_train, X_val = X[:-1], X[-1:]
    y_train, y_val = y[:-1], y[-1:]
    
    # Build and train
    model = create_lstm_model(n_features=len(FEATURES))
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor='loss', patience=30, restore_best_weights=True, min_delta=0.001
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.5, patience=15, min_lr=0.0001
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if len(X_val) > 0 else None,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    final_loss = history.history['loss'][-1]
    final_mae = history.history['mae'][-1]
    
    print(f"  ✓ LSTM trained")
    print(f"    Final loss: {final_loss:.4f}, MAE: {final_mae:.4f}")
    print()
    
    models[guild] = model
    forecasts[guild] = {'method': 'lstm'}

# Generate forecasts
print("[5/6] Generating forecasts for 2026-2030...")
print()

all_forecasts = []

for guild in GUILDS:
    print(f"  Forecasting {guild}...")
    
    guild_ts = ts_df[ts_df['guild'] == guild].sort_values('year').reset_index(drop=True)
    
    # Check if this guild uses simple forecast
    if forecasts.get(guild, {}).get('method') == 'constant':
        constant_value = forecasts[guild]['value']
        
        for year in YEARS_FORECAST:
            all_forecasts.append({
                'guild': guild,
                'year': year,
                'predicted_detection_rate': constant_value,
                'is_forecast': True,
                'method': 'constant'
            })
        
        print(f"    ✓ Simple forecast: {constant_value:.3f} (constant)")
        continue
    
    # Use LSTM forecast
    if guild not in models or models[guild] is None:
        print(f"    ⚠️  No model available")
        continue
    
    model = models[guild]
    scaler = scalers[guild]
    
    guild_ts_norm = guild_ts.copy()
    guild_ts_norm[FEATURES] = scaler.transform(guild_ts[FEATURES])
    
    current_sequence = guild_ts_norm[FEATURES].iloc[-LOOKBACK_WINDOW:].values
    
    for year in YEARS_FORECAST:
        X_pred = current_sequence.reshape(1, LOOKBACK_WINDOW, len(FEATURES))
        y_pred = model.predict(X_pred, verbose=0)[0, 0]
        
        # CRITICAL: Clip to valid range
        y_pred = np.clip(y_pred, 0.0, 1.0)
        
        next_features = guild_ts_norm[FEATURES].iloc[-3:].mean().values
        next_features[-1] = y_pred
        
        all_forecasts.append({
            'guild': guild,
            'year': year,
            'predicted_detection_rate': y_pred,
            'is_forecast': True,
            'method': 'lstm'
        })
        
        current_sequence = np.vstack([current_sequence[1:], next_features])
    
    print(f"    ✓ LSTM forecast generated")

# Add historical data
for _, row in ts_df.iterrows():
    all_forecasts.append({
        'guild': row['guild'],
        'year': row['year'],
        'predicted_detection_rate': row['detection_rate'],
        'is_forecast': False,
        'method': 'observed'
    })

forecast_df = pd.DataFrame(all_forecasts)
forecast_df.to_csv(PHASE7_DIR / "lstm_forecasts.csv", index=False)

print()
print(f"✓ Saved: lstm_forecasts.csv")
print()

# Visualize
print("[6/6] Creating visualizations...")

fig = go.Figure()
colors = {'Wetland': '#2196F3', 'Forest': '#4CAF50', 'Urban': '#FF9800'}

for guild in GUILDS:
    guild_data = forecast_df[forecast_df['guild'] == guild]
    
    hist = guild_data[~guild_data['is_forecast']]
    fig.add_trace(go.Scatter(
        x=hist['year'], y=hist['predicted_detection_rate'],
        mode='lines+markers', name=f'{guild} (Historical)',
        line=dict(width=2, color=colors[guild]), marker=dict(size=8)
    ))
    
    fore = guild_data[guild_data['is_forecast']]
    if len(fore) > 0:
        method = fore.iloc[0]['method']
        line_style = 'dot' if method == 'constant' else 'dash'
        
        fig.add_trace(go.Scatter(
            x=fore['year'], y=fore['predicted_detection_rate'],
            mode='lines+markers', name=f'{guild} (Forecast)',
            line=dict(width=2, dash=line_style, color=colors[guild]),
            marker=dict(size=8, symbol='diamond')
        ))

fig.add_vline(x=2025.5, line_dash="dot", line_color="gray",
             annotation_text="Forecast Begins", annotation_position="top")

fig.update_layout(
    title="LSTM Population Forecasts: 2026-2030",
    xaxis_title="Year",
    yaxis_title="Detection Rate",
    height=500
)

fig.write_html(PHASE7_DIR / "lstm_forecast_plot.html")
print("✓ Saved: lstm_forecast_plot.html")
print()

# Summary
print("="*80)
print("FORECAST SUMMARY")
print("="*80)
print()

for guild in GUILDS:
    guild_hist = forecast_df[(forecast_df['guild'] == guild) & (~forecast_df['is_forecast'])]
    guild_fore = forecast_df[(forecast_df['guild'] == guild) & (forecast_df['is_forecast'])]
    
    if len(guild_hist) == 0 or len(guild_fore) == 0:
        continue
    
    last_obs = guild_hist[guild_hist['year'] == 2025]['predicted_detection_rate'].values[0]
    forecast_2030 = guild_fore[guild_fore['year'] == 2030]['predicted_detection_rate'].values[0]
    trend = forecast_2030 - last_obs
    
    method = guild_fore.iloc[0]['method']
    method_label = "LSTM" if method == 'lstm' else "Constant (saturated)"
    
    print(f"{guild} Guild ({method_label}):")
    print(f"  2025 (Last observed): {last_obs:.3f}")
    print(f"  2030 (Forecast):      {forecast_2030:.3f}")
    
    if abs(trend) > 0.05:
        direction = "↗️ INCREASING" if trend > 0 else "↘️ DECLINING"
        print(f"  Trend: {direction} ({trend:+.3f})")
    else:
        print(f"  Trend: → STABLE ({trend:+.3f})")
    
    print()

print("="*80)
print("✅ LSTM FORECASTING COMPLETE!")
print("="*80)
print()
'''

    with open(LSTM_CURRENT, 'w', encoding='utf-8') as f:
        f.write(lstm_fixed_content)
    
    print(f"✓ LSTM script updated: {LSTM_CURRENT}")
    print()
else:
    print("⚠️  Skipped LSTM fix")
    print()

# =============================================================================
# FIX 2 & 3: APP.PY UPDATES
# =============================================================================

print("[FIX 2-3/3] Updating Dashboard (app.py)...")
print("-" * 80)

if APP_CURRENT.exists():
    # Backup
    backup_app = BACKUP_DIR / "app_ORIGINAL.py"
    shutil.copy2(APP_CURRENT, backup_app)
    print(f"✓ Backed up current app.py to: {backup_app}")
    
    # Read current app.py
    with open(APP_CURRENT, 'r', encoding='utf-8') as f:
        app_content = f.read()
    
    # Check if fixes already applied
    if 'create_environmental_satellite_map' in app_content:
        print("✓ Satellite map fix already applied (function exists)")
        satellite_fix_needed = False
    else:
        satellite_fix_needed = True
    
    if 'Priority_Level' in app_content and 'try:' in app_content:
        print("✓ Restoration timeline fix appears applied")
        timeline_fix_needed = False
    else:
        timeline_fix_needed = True
    
    if not satellite_fix_needed and not timeline_fix_needed:
        print()
        print("✅ App.py fixes already applied!")
        print()
    else:
        print()
        print("⚠️  App.py needs manual updates:")
        print()
        
        if satellite_fix_needed:
            print("📝 FIX 2: Satellite Maps")
            print("   1. Open: FINAL_FIX_exact_boundaries_satellite.py")
            print("   2. Copy the helper functions to app.py (after line 300)")
            print("   3. Replace satellite comparison section (around line 1850)")
            print()
        
        if timeline_fix_needed:
            print("📝 FIX 3: Restoration Timeline")
            print("   1. Open: FIX_restoration_timeline_priority.py")
            print("   2. Find render_restoration_timeline_page() in app.py")
            print("   3. Replace priority data loading section (around line 3600)")
            print()
        
        print("💡 These are complex multi-line replacements.")
        print("   I can't do them automatically without risking breaking your code.")
        print("   Please apply them manually using the fix files provided.")
        print()

else:
    print("⚠️  Warning: app.py not found!")
    print(f"   Expected location: {APP_CURRENT}")
    print()

# =============================================================================
# SUMMARY
# =============================================================================

print("="*80)
print("FIX SUMMARY")
print("="*80)
print()

print("✅ COMPLETED AUTOMATICALLY:")
print("  1. LSTM forecasting script updated")
print(f"     Location: {LSTM_CURRENT}")
print()

print("📝 REQUIRES MANUAL COMPLETION:")
print("  2. Satellite maps (app.py) - See instructions above")
print("  3. Restoration timeline (app.py) - See instructions above")
print()

print("📁 BACKUPS SAVED TO:")
print(f"   {BACKUP_DIR}")
print()

print("="*80)
print("NEXT STEPS")
print("="*80)
print()

print("1. ✅ Test LSTM fix:")
print("   python sentinel/phase7/lstm_forecasting.py")
print()

print("2. 📝 Apply app.py fixes manually (see instructions above)")
print()

print("3. ✅ Test dashboard:")
print("   streamlit run app.py")
print()

print("4. ✅ Verify all pages work:")
print("   - Overview")
print("   - Habitat Maps")
print("   - Guild Analysis")
print("   - Conservation Priority")
print("   - Temporal Trends")
print("   - AI Forecasting")
print()

print("="*80)
print()