#!/usr/bin/env python3
"""
Mangalavanam Adaptive Sentinel
Phase 3B: Zone-Specific ψ–p Occupancy Model (CORRECTED)

IMPROVEMENTS:
- Guild-specific analysis support
- Better diagnostics and validation

Author: Senior Geospatial Data Scientist
Date: January 2026
"""

import pandas as pd
import pymc as pm
import arviz as az
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

CSV = Path("data/processed/phase2/guild_pixel_timeseries_with_covariates.csv")
OUT_DIR = Path("data/processed/phase3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# MAIN MODEL
# =============================================================================

def fit_zone_model(guild=None):
    """
    Fit zone-specific occupancy model.
    
    Args:
        guild: Specific guild to model (None = all guilds combined)
    """
    
    print("=" * 70)
    print("PHASE 3B — ZONE-SPECIFIC ψ–p MODEL")
    if guild:
        print(f"Guild: {guild}")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Load and filter data
    # -----------------------------------------------------------------
    print("\n[1/4] Loading data...")
    
    if not CSV.exists():
        print(f"❌ ERROR: {CSV} not found")
        return 1
    
    df = pd.read_csv(CSV)
    
    # Filter to specific guild if requested
    if guild:
        df = df[df["guild"] == guild].copy()
        print(f"  Filtered to {guild} guild")
    
    # Drop rows with missing data
    df = df.dropna(subset=["pixel_id", "zone", "detected", "effort"])
    
    # Encode zones: 0=buffer, 1=core
    df["zone_code"] = df["zone"].map({"buffer": 0, "core": 1})
    
    print(f"  ✓ Loaded {len(df):,} observations")
    
    if df["zone_code"].isna().any():
        print("  ⚠️ WARNING: Unknown zone values detected, filling with 0 (buffer)")
        df["zone_code"] = df["zone_code"].fillna(0)

    # -----------------------------------------------------------------
    # Prepare model data
    # -----------------------------------------------------------------
    print("\n[2/4] Preparing model data...")
    
    # Site indexing
    site_idx, sites = pd.factorize(df["pixel_id"])
    
    # Get zone for each site (should be constant per site)
    zone = df.groupby("pixel_id")["zone_code"].first().values

    # Response and predictors
    y = df["detected"].astype(int).values
    effort = df["effort"].values

    print(f"  Sites: {len(sites):,}")
    print(f"  Zone distribution:")
    print(f"    Buffer: {(zone == 0).sum():,} sites")
    print(f"    Core: {(zone == 1).sum():,} sites")
    print(f"  Observations: {len(df):,}")
    
    # Data checks
    print(f"\n  Data checks:")
    print(f"    Detections: {y.sum():,} / {len(y):,} ({y.sum()/len(y)*100:.1f}%)")
    buffer_detections = df[df["zone_code"] == 0]["detected"].sum()
    core_detections = df[df["zone_code"] == 1]["detected"].sum()
    print(f"    Buffer detections: {buffer_detections:,}")
    print(f"    Core detections: {core_detections:,}")

    # -----------------------------------------------------------------
    # Build and sample model
    # -----------------------------------------------------------------
    print("\n[3/4] Fitting Bayesian occupancy model...")
    print("  This may take 10-30 minutes...")

    with pm.Model() as model:

        # Occupancy model with zone effect
        alpha_psi = pm.Normal("alpha_psi", 0, 2)
        beta_zone = pm.Normal("beta_zone", 0, 2)

        # Site-level occupancy
        logit_psi = alpha_psi + beta_zone * zone
        psi = pm.Deterministic("psi", pm.math.sigmoid(logit_psi))

        # Latent occupancy state
        z = pm.Bernoulli("z", psi, shape=len(sites))

        # Detection model
        alpha_p = pm.Normal("alpha_p", 0, 2)
        beta_p = pm.Normal("beta_p", 0, 2)

        logit_p = alpha_p + beta_p * effort
        p = pm.math.sigmoid(logit_p)

        # Observation model
        pm.Bernoulli("y_obs", p=z[site_idx] * p, observed=y)

        # Sample
        print("  → Sampling posterior (4 chains × 2000 draws)...")
        trace = pm.sample(
            draws=2000,
            tune=2000,
            chains=4,
            cores=1,
            target_accept=0.9,
            progressbar=True
        )

    # -----------------------------------------------------------------
    # Save and summarize
    # -----------------------------------------------------------------
    print("\n[4/4] Saving results...")
    
    # Construct output filename
    if guild:
        out_file = OUT_DIR / f"posterior_zone_{guild}.nc"
    else:
        out_file = OUT_DIR / "posterior_zone.nc"
    
    az.to_netcdf(trace, out_file)
    print(f"  ✓ Saved: {out_file}")

    # Print diagnostics
    print("\n" + "=" * 70)
    print("CONVERGENCE DIAGNOSTICS")
    print("=" * 70)
    
    summary = az.summary(
        trace,
        var_names=["alpha_psi", "beta_zone", "alpha_p", "beta_p"],
        hdi_prob=0.95
    )
    print(summary)
    
    # Zone effects
    print("\n" + "=" * 70)
    print("ZONE EFFECTS")
    print("=" * 70)
    
    alpha_samples = trace.posterior["alpha_psi"].values.flatten()
    beta_samples = trace.posterior["beta_zone"].values.flatten()
    
    # Buffer (zone_code = 0)
    psi_buffer = 1 / (1 + np.exp(-alpha_samples))
    print(f"Buffer ψ: {psi_buffer.mean():.3f} [{np.percentile(psi_buffer, 2.5):.3f}, {np.percentile(psi_buffer, 97.5):.3f}]")
    
    # Core (zone_code = 1)
    psi_core = 1 / (1 + np.exp(-(alpha_samples + beta_samples)))
    print(f"Core ψ: {psi_core.mean():.3f} [{np.percentile(psi_core, 2.5):.3f}, {np.percentile(psi_core, 97.5):.3f}]")
    
    # Difference
    diff = psi_core - psi_buffer
    print(f"Core - Buffer: {diff.mean():.3f} [{np.percentile(diff, 2.5):.3f}, {np.percentile(diff, 97.5):.3f}]")
    
    if diff.mean() > 0:
        print("  → Core has higher occupancy than Buffer")
    else:
        print("  → Buffer has higher occupancy than Core")
    
    print("\n✅ PHASE 3B COMPLETE")
    print("=" * 70)

    return 0


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fit zone-specific occupancy model'
    )
    parser.add_argument(
        '--guild',
        type=str,
        choices=['Wetland', 'Forest', 'Urban'],
        default=None,
        help='Specific guild to model (default: all guilds combined)'
    )
    
    args = parser.parse_args()
    
    # Need numpy for calculations
    import numpy as np
    globals()['np'] = np
    
    return fit_zone_model(guild=args.guild)


if __name__ == "__main__":
    exit(main())