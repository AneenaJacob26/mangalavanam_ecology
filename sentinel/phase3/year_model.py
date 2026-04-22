#!/usr/bin/env python3
"""
Mangalavanam Adaptive Sentinel
Phase 3A: Year-Varying ψ–p Occupancy Model (CORRECTED)

IMPROVEMENTS:
- Proper observation-level year indexing
- Guild-specific analysis support
- Better error handling and diagnostics

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

def fit_year_model(guild=None):
    """
    Fit year-varying occupancy model.
    
    Args:
        guild: Specific guild to model (None = all guilds combined)
    """
    
    print("=" * 70)
    print("PHASE 3A — YEAR-VARYING ψ–p MODEL")
    if guild:
        print(f"Guild: {guild}")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Load and filter data
    # -----------------------------------------------------------------
    print("\n[1/4] Loading data...")
    
    if not CSV.exists():
        print(f"❌ ERROR: {CSV} not found")
        print("Please run Phase 2b first.")
        return 1
    
    df = pd.read_csv(CSV)
    
    # Filter to specific guild if requested
    if guild:
        df = df[df["guild"] == guild].copy()
        print(f"  Filtered to {guild} guild")
    
    # Drop rows with missing critical data
    df = df.dropna(subset=["pixel_id", "year", "detected", "effort"])
    
    print(f"  ✓ Loaded {len(df):,} observations")
    
    if len(df) == 0:
        print("❌ ERROR: No valid observations after filtering")
        return 1

    # -----------------------------------------------------------------
    # Prepare model data
    # -----------------------------------------------------------------
    print("\n[2/4] Preparing model data...")
    
    # Site indexing
    site_idx, sites = pd.factorize(df["pixel_id"])
    
    # Year indexing for each observation
    year_idx, years = pd.factorize(df["year"])
    
    # Get dominant year per site (for occupancy calculation)
    site_year_map = df.groupby("pixel_id")["year"].agg(lambda x: x.mode()[0])
    site_year_idx = site_year_map.map({y: i for i, y in enumerate(years)}).values

    # Response and predictors
    y = df["detected"].astype(int).values
    effort = df["effort"].values

    print(f"  Sites: {len(sites):,}")
    print(f"  Years: {list(years)}")
    print(f"  Observations: {len(df):,}")
    
    # Check for data issues
    print(f"\n  Data checks:")
    print(f"    Detections: {y.sum():,} / {len(y):,} ({y.sum()/len(y)*100:.1f}%)")
    print(f"    Effort = 1: {(effort == 1).sum():,} ({(effort == 1).sum()/len(effort)*100:.1f}%)")
    
    if y.sum() == 0:
        print("  ⚠️ WARNING: No detections in data!")
    if y.sum() == len(y):
        print("  ⚠️ WARNING: Species detected everywhere!")

    # -----------------------------------------------------------------
    # Build and sample model
    # -----------------------------------------------------------------
    print("\n[3/4] Fitting Bayesian occupancy model...")
    print("  This may take 10-30 minutes...")

    with pm.Model() as model:

        # Occupancy model with year effect
        alpha_psi = pm.Normal("alpha_psi", 0, 2)
        beta_year = pm.Normal("beta_year", 0, 1, shape=len(years))

        # Site-level occupancy varies by site's dominant year
        logit_psi = alpha_psi + beta_year[site_year_idx]
        psi = pm.Deterministic("psi", pm.math.sigmoid(logit_psi))

        # Latent occupancy state
        z = pm.Bernoulli("z", psi, shape=len(sites))

        # Detection model
        alpha_p = pm.Normal("alpha_p", 0, 2)
        beta_p = pm.Normal("beta_p", 0, 1)

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
            cores=1,  # Important for Windows compatibility
            target_accept=0.9,
            progressbar=True
        )

    # -----------------------------------------------------------------
    # Save and summarize
    # -----------------------------------------------------------------
    print("\n[4/4] Saving results...")
    
    # Construct output filename
    if guild:
        out_file = OUT_DIR / f"posterior_year_{guild}.nc"
    else:
        out_file = OUT_DIR / "posterior_year.nc"
    
    az.to_netcdf(trace, out_file)
    print(f"  ✓ Saved: {out_file}")

    # Print diagnostics
    print("\n" + "=" * 70)
    print("CONVERGENCE DIAGNOSTICS")
    print("=" * 70)
    
    summary = az.summary(
        trace,
        var_names=["alpha_psi", "beta_year", "alpha_p", "beta_p"],
        hdi_prob=0.95
    )
    print(summary)
    
    # Year effects
    print("\n" + "=" * 70)
    print("YEAR EFFECTS (beta_year)")
    print("=" * 70)
    beta_samples = trace.posterior["beta_year"].values
    for i, year in enumerate(years):
        mean_effect = beta_samples[:, :, i].mean()
        hdi_low = beta_samples[:, :, i].flatten()
        hdi_low = pd.Series(hdi_low).quantile(0.025)
        hdi_high = pd.Series(hdi_low).quantile(0.975)
        print(f"  Year {year}: {mean_effect:+.3f} [{hdi_low:+.3f}, {hdi_high:+.3f}]")
    
    # Occupancy prevalence
    psi_samples = trace.posterior["psi"].values
    mean_psi = psi_samples.mean()
    print(f"\nMean occupancy (ψ): {mean_psi:.3f}")
    
    print("\n✅ PHASE 3A COMPLETE")
    print("=" * 70)

    return 0


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fit year-varying occupancy model'
    )
    parser.add_argument(
        '--guild',
        type=str,
        choices=['Wetland', 'Forest', 'Urban'],
        default=None,
        help='Specific guild to model (default: all guilds combined)'
    )
    
    args = parser.parse_args()
    
    return fit_year_model(guild=args.guild)


if __name__ == "__main__":
    exit(main())