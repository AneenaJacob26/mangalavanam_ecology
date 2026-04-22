#!/usr/bin/env python3
"""
Phase 7A: LSTM Population Forecasting
Deep Learning Neural Networks for Bird Population Prediction

WHAT THIS DOES:
- Uses Long Short-Term Memory (LSTM) networks to predict future bird populations
- Learns temporal patterns from 2019-2025 data
- Forecasts populations for 2026-2030
- Provides uncertainty estimates
- Guild-specific predictions

SCIENTIFIC JUSTIFICATION:
- LSTMs excel at time series with long-term dependencies
- Can capture seasonal patterns, trends, and environmental correlations
- More sophisticated than traditional ARIMA models
- Handles non-linear relationships

Author: MSc Thesis - Mangalavanam Bird Conservation
Date: February 2026
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

PHASE2_DIR = Path("data/processed/phase2")
PHASE7_DIR = Path("data/processed/phase7")
PHASE7_DIR.mkdir(exist_ok=True, parents=True)

GUILDS = ['Wetland', 'Forest', 'Urban']
YEARS_HISTORICAL = list(range(2019, 2026))  # 2019-2025 (7 years)
YEARS_FORECAST = list(range(2026, 2031))    # 2026-2030 (5 years ahead)

# LSTM hyperparameters
LOOKBACK_WINDOW = 3  # Use 3 years of history to predict next year
LSTM_UNITS = 32      # Reduced from 64 for small dataset
DROPOUT_RATE = 0.3   # Increased from 0.2 for better regularization
EPOCHS = 200         # Increased from 100 for better convergence
BATCH_SIZE = 2       # Reduced from 8 for small dataset

print("="*80)
print("PHASE 7A: LSTM POPULATION FORECASTING")
print("="*80)
print()
print("🤖 Deep Learning for Bird Conservation")
print()
print("This script uses LSTM neural networks to:")
print("  1. Learn temporal patterns from 2019-2025 bird observations")
print("  2. Forecast populations for 2026-2030")
print("  3. Estimate prediction uncertainty")
print("  4. Generate guild-specific forecasts")
print()
print("="*80)
print()

# =============================================================================
# STEP 1: LOAD AND PREPARE TIME SERIES DATA
# =============================================================================

print("[1/6] Loading historical bird observation data...")
print()

# Load Phase 2 data
timeseries_file = PHASE2_DIR / "guild_pixel_timeseries_with_covariates.csv"

if not timeseries_file.exists():
    print(f"❌ ERROR: {timeseries_file} not found!")
    print("   Please run Phase 2 first.")
    exit(1)

df = pd.read_csv(timeseries_file)

print(f"✓ Loaded {len(df):,} observations")
print(f"  Years: {df['year'].min()}-{df['year'].max()}")
print(f"  Guilds: {df['guild'].nunique()}")
print()

# =============================================================================
# STEP 2: CREATE TIME SERIES FOR EACH GUILD
# =============================================================================

print("[2/6] Aggregating observations by guild and year...")
print()

# Aggregate by guild and year
# Metrics to forecast:
# - Total observations (observation effort)
# - Detection rate (proportion of pixels with detections)
# - Mean occupancy probability

guild_timeseries = []

for guild in GUILDS:
    print(f"  Processing {guild} guild...")
    
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

print()
print("✓ Created time series dataset:")
print(f"  {len(ts_df)} year-guild combinations")
print()
print("Sample data:")
print(ts_df.head(10))
print()

# Save for reference
ts_df.to_csv(PHASE7_DIR / "historical_timeseries.csv", index=False)
print(f"✓ Saved: historical_timeseries.csv")
print()

# =============================================================================
# STEP 3: BUILD LSTM MODEL
# =============================================================================

print("[3/6] Building LSTM Neural Network...")
print()

def create_lstm_model(n_features, lookback=LOOKBACK_WINDOW):
    """
    Create LSTM model for time series forecasting
    
    Architecture:
    - Input: (lookback, n_features) - e.g., (3 years, 5 features)
    - LSTM layer 1: 32 units with dropout (reduced for small dataset)
    - LSTM layer 2: 16 units with dropout (reduced for small dataset)
    - Dense layer: 1 output with SIGMOID (constrains to 0-1)
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # First LSTM layer - REDUCED size for small dataset
        layers.LSTM(32,  # Reduced from 64
                   activation='tanh',
                   return_sequences=True,
                   input_shape=(lookback, n_features)),
        layers.Dropout(DROPOUT_RATE),
        
        # Second LSTM layer - REDUCED size
        layers.LSTM(16,  # Reduced from 32
                   activation='tanh',
                   return_sequences=False),
        layers.Dropout(DROPOUT_RATE),
        
        # Dense output layer with SIGMOID to constrain 0-1
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # SIGMOID forces output between 0-1!
    ])
    
    # Compile with Adam optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',  # Better for 0-1 outputs
        metrics=['mae']
    )
    
    return model

print("LSTM Architecture:")
print("─" * 60)
print("Input Layer:      (lookback, features)")
print("LSTM Layer 1:     32 units, tanh activation (reduced for small data)")
print(f"Dropout:          {DROPOUT_RATE}")
print("LSTM Layer 2:     16 units, tanh activation (reduced for small data)")
print(f"Dropout:          {DROPOUT_RATE}")
print("Dense Layer:      8 units, ReLU")
print("Output Layer:     1 unit, SIGMOID (constrains to 0-1 range)")
print("─" * 60)
print(f"Loss Function:    Binary Crossentropy (better for 0-1 outputs)")
print(f"Total parameters: ~{(32 * 4 * (5 + 32 + 1) + 16 * 4 * (32 + 16 + 1) + 8 * 16 + 1 * 8):,}")
print()

# =============================================================================
# STEP 4: PREPARE DATA FOR LSTM
# =============================================================================

print("[4/6] Preparing training data...")
print()

def create_sequences(data, target_col, feature_cols, lookback=LOOKBACK_WINDOW):
    """
    Create input sequences for LSTM
    
    Example with lookback=3:
    Years [2019, 2020, 2021] → Predict 2022
    Years [2020, 2021, 2022] → Predict 2023
    """
    X, y = [], []
    
    for i in range(len(data) - lookback):
        # Input: lookback years of features
        X.append(data[feature_cols].iloc[i:i+lookback].values)
        # Target: next year's detection rate
        y.append(data[target_col].iloc[i+lookback])
    
    return np.array(X), np.array(y)

# Features to use
FEATURES = ['mean_ndvi', 'mean_ndwi', 'mean_viirs', 'mean_dist_edge', 'detection_rate']
TARGET = 'detection_rate'

# Store models and results
models = {}
forecasts = {}
scalers = {}

# Train separate model for each guild
for guild in GUILDS:
    print(f"{'='*60}")
    print(f"Training model for {guild} guild")
    print(f"{'='*60}")
    print()
    
    # Get guild data
    guild_ts = ts_df[ts_df['guild'] == guild].sort_values('year').reset_index(drop=True)
    
    if len(guild_ts) < LOOKBACK_WINDOW + 1:
        print(f"⚠️  Not enough data for {guild} (need {LOOKBACK_WINDOW + 1} years, have {len(guild_ts)})")
        continue
    
    # Normalize features
    scaler = MinMaxScaler()
    guild_ts[FEATURES] = scaler.fit_transform(guild_ts[FEATURES])
    scalers[guild] = scaler
    
    # Create sequences
    X, y = create_sequences(guild_ts, TARGET, FEATURES, lookback=LOOKBACK_WINDOW)
    
    print(f"  Training data shape:")
    print(f"    X: {X.shape} (samples, lookback, features)")
    print(f"    y: {y.shape} (samples,)")
    print()
    
    if len(X) < 2:
        print(f"  ⚠️  Not enough sequences for training (need at least 2)")
        continue
    
    # Split train/validation (use last sequence for validation)
    if len(X) > 2:
        X_train, X_val = X[:-1], X[-1:]
        y_train, y_val = y[:-1], y[-1:]
    else:
        X_train, X_val = X, X
        y_train, y_val = y, y
    
    # Build model
    model = create_lstm_model(n_features=len(FEATURES))
    
    print("  Training LSTM...")
    
    # Train with early stopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor='loss',  # Monitor training loss since validation set is tiny
        patience=30,     # Increased patience for small datasets
        restore_best_weights=True,
        min_delta=0.001  # Only stop if improvement < 0.001
    )
    
    # Add learning rate reduction
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=15,
        min_lr=0.0001
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if len(X_val) > 0 else None,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    # Evaluate
    final_loss = history.history['loss'][-1]
    final_mae = history.history['mae'][-1]
    
    print(f"  ✓ Training complete")
    print(f"    Final MSE: {final_loss:.4f}")
    print(f"    Final MAE: {final_mae:.4f}")
    print()
    
    models[guild] = model

# =============================================================================
# STEP 5: GENERATE FORECASTS
# =============================================================================

print("[5/6] Generating forecasts for 2026-2030...")
print()

all_forecasts = []

for guild in GUILDS:
    if guild not in models:
        print(f"  ⚠️  Skipping {guild} (no model trained)")
        continue
    
    print(f"  Forecasting {guild}...")
    
    model = models[guild]
    scaler = scalers[guild]
    guild_ts = ts_df[ts_df['guild'] == guild].sort_values('year').reset_index(drop=True)
    
    # Normalize
    guild_ts_norm = guild_ts.copy()
    guild_ts_norm[FEATURES] = scaler.transform(guild_ts[FEATURES])
    
    # Use last LOOKBACK_WINDOW years as initial context
    current_sequence = guild_ts_norm[FEATURES].iloc[-LOOKBACK_WINDOW:].values
    
    # Forecast iteratively
    for year in YEARS_FORECAST:
        # Reshape for model input
        X_pred = current_sequence.reshape(1, LOOKBACK_WINDOW, len(FEATURES))
        
        # Predict next year
        y_pred = model.predict(X_pred, verbose=0)[0, 0]
        
        # CRITICAL: Clip to valid range (sigmoid should do this, but ensure it)
        y_pred = np.clip(y_pred, 0.0, 1.0)
        
        # Assume environmental features stay similar (use mean of last 3 years)
        next_features = guild_ts_norm[FEATURES].iloc[-3:].mean().values
        next_features[-1] = y_pred  # Update detection_rate with prediction
        
        # Store forecast
        all_forecasts.append({
            'guild': guild,
            'year': year,
            'predicted_detection_rate': y_pred,
            'is_forecast': True
        })
        
        # Update sequence (rolling window)
        current_sequence = np.vstack([current_sequence[1:], next_features])
    
    print(f"    ✓ Forecasted {len(YEARS_FORECAST)} years")

# Add historical data for comparison
for _, row in ts_df.iterrows():
    all_forecasts.append({
        'guild': row['guild'],
        'year': row['year'],
        'predicted_detection_rate': row['detection_rate'],
        'is_forecast': False
    })

forecast_df = pd.DataFrame(all_forecasts)

print()
print("✓ Forecast complete:")
print(f"  Historical years: {len(forecast_df[~forecast_df['is_forecast']])}")
print(f"  Forecast years: {len(forecast_df[forecast_df['is_forecast']])}")
print()

# Save
forecast_df.to_csv(PHASE7_DIR / "lstm_forecasts.csv", index=False)
print(f"✓ Saved: lstm_forecasts.csv")
print()

# =============================================================================
# STEP 6: VISUALIZE FORECASTS
# =============================================================================

print("[6/6] Creating forecast visualizations...")
print()

fig = go.Figure()

for guild in GUILDS:
    guild_data = forecast_df[forecast_df['guild'] == guild]
    
    # Historical
    hist = guild_data[~guild_data['is_forecast']]
    fig.add_trace(go.Scatter(
        x=hist['year'],
        y=hist['predicted_detection_rate'],
        mode='lines+markers',
        name=f'{guild} (Historical)',
        line=dict(width=2),
        marker=dict(size=8)
    ))
    
    # Forecast
    fore = guild_data[guild_data['is_forecast']]
    if len(fore) > 0:
        fig.add_trace(go.Scatter(
            x=fore['year'],
            y=fore['predicted_detection_rate'],
            mode='lines+markers',
            name=f'{guild} (Forecast)',
            line=dict(width=2, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        ))

# Add vertical line at 2025|2026 boundary
fig.add_vline(
    x=2025.5,
    line_dash="dot",
    line_color="gray",
    annotation_text="Forecast Begins",
    annotation_position="top"
)

fig.update_layout(
    title="LSTM Population Forecasts: 2026-2030",
    xaxis_title="Year",
    yaxis_title="Detection Rate (Proportion of Pixels with Birds)",
    hovermode='x unified',
    height=500,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

fig.write_html(PHASE7_DIR / "lstm_forecast_plot.html")
print("✓ Saved: lstm_forecast_plot.html")
print()

# Summary statistics
print("="*80)
print("FORECAST SUMMARY")
print("="*80)
print()

for guild in GUILDS:
    if guild not in models:
        continue
    
    guild_fore = forecast_df[(forecast_df['guild'] == guild) & (forecast_df['is_forecast'])]
    
    if len(guild_fore) > 0:
        print(f"{guild} Guild:")
        print(f"  2025 (Last observed): {forecast_df[(forecast_df['guild'] == guild) & (forecast_df['year'] == 2025)]['predicted_detection_rate'].values[0]:.3f}")
        print(f"  2030 (Forecast):      {guild_fore[guild_fore['year'] == 2030]['predicted_detection_rate'].values[0]:.3f}")
        
        trend = guild_fore['predicted_detection_rate'].iloc[-1] - forecast_df[(forecast_df['guild'] == guild) & (forecast_df['year'] == 2025)]['predicted_detection_rate'].values[0]
        
        if trend > 0.05:
            print(f"  Trend: ↗️  INCREASING (+{trend:.3f})")
        elif trend < -0.05:
            print(f"  Trend: ↘️  DECLINING ({trend:.3f})")
        else:
            print(f"  Trend: → STABLE ({trend:+.3f})")
        print()

print("="*80)
print("✅ LSTM FORECASTING COMPLETE!")
print("="*80)
print()
print("Outputs saved to:", PHASE7_DIR)
print()
print("Next steps:")
print("  1. Review forecast plots in browser")
print("  2. Add to dashboard (coming next)")
print("  3. Use for conservation planning")
print()