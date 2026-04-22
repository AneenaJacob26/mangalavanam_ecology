
"""
Mangalavanam Adaptive Sentinel
Phase 3C: Environmental ψ–p Occupancy Model (CORRECTED)

IMPROVEMENTS:
- Guild-specific analysis support
- Better covariate handling and validation
- Proper standardization

Author: Senior Geospatial Data Scientist
Date: January 2026
"""

import pandas as pd
import numpy as np
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

# Environmental covariates to use
COVARS = [
    "ndvi",
    "ndwi",
    "dist_edge",
    "dist_drainage",
    "viirs"
]

# =============================================================================
# MAIN MODEL
# =============================================================================

def fit_environmental_model(guild=None):
    """
    Fit environmental occupancy model.
    
    Args:
        guild: Specific guild to model (None = all guilds combined)
    """
    
    print("=" * 70)
    print("PHASE 3C — ENVIRONMENTAL ψ–p MODEL")
    if guild:
        print(f"Guild: {guild}")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Load and filter data
    # -----------------------------------------------------------------
    print("\n[1/5] Loading data...")
    
    if not CSV.exists():
        print(f"❌ ERROR: {CSV} not found")
        return 1
    
    df = pd.read_csv(CSV)
    
    # Filter to specific guild if requested
    if guild:
        df = df[df["guild"] == guild].copy()
        print(f"  Filtered to {guild} guild")
    
    # Check which covariates are available
    available_covars = [c for c in COVARS if c in df.columns]
    missing_covars = [c for c in COVARS if c not in df.columns]
    
    if missing_covars:
        print(f"  ⚠️ Missing covariates: {', '.join(missing_covars)}")
    
    if not available_covars:
        print(f"❌ ERROR: No environmental covariates found!")
        return 1
    
    print(f"  Using covariates: {', '.join(available_covars)}")
    
    # Drop rows with missing critical data
    required_cols = ["pixel_id", "detected", "effort"] + available_covars
    df = df.dropna(subset=required_cols)
    
    print(f"  ✓ Loaded {len(df):,} observations with complete data")
    
    if len(df) == 0:
        print("❌ ERROR: No observations with complete covariate data")
        return 1

    # -----------------------------------------------------------------
    # Prepare model data
    # -----------------------------------------------------------------
    print("\n[2/5] Preparing model data...")
    
    # Site indexing
    site_idx, sites = pd.factorize(df["pixel_id"])
    
    # Covariate matrix
    X = df[available_covars].values
    
    # Standardize covariates (CRITICAL for convergence)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1  # Prevent division by zero
    X_standardized = (X - X_mean) / X_std
    
    print(f"  Covariate standardization:")
    for i, cov in enumerate(available_covars):
        print(f"    {cov}: mean={X_mean[i]:.3f}, std={X_std[i]:.3f}")

    # Response and detection effort
    y = df["detected"].astype(int).values
    effort = df["effort"].values

    print(f"\n  Model dimensions:")
    print(f"    Sites: {len(sites):,}")
    print(f"    Observations: {len(df):,}")
    print(f"    Covariates: {len(available_covars)}")
    
    # Data checks
    print(f"\n  Data checks:")
    print(f"    Detections: {y.sum():,} / {len(y):,} ({y.sum()/len(y)*100:.1f}%)")
    print(f"    Mean effort: {effort.mean():.3f}")

    # --------------------------------------------------
    # SITE vs OBSERVATION STRUCTURE (CRITICAL FIX)
    # --------------------------------------------------

    # Number of sites
    n_sites = len(sites)

    # Observation-level inputs (p)
    X_p = X_standardized
    y_obs = y

    # Site-level covariates (ψ): mean across visits per site
    X_psi = (
        pd.DataFrame(X_standardized, columns=available_covars)
        .assign(site_idx=site_idx)
        .groupby("site_idx")
        .mean()
        .sort_index()
        .values
    )

    # Safety checks
    assert X_psi.shape[0] == n_sites
    assert X_p.shape[0] == len(y_obs)
    assert site_idx.max() == n_sites - 1

    # -----------------------------------------------------------------
    # Build and sample model
    # -----------------------------------------------------------------
    print("\n[3/5] Fitting Bayesian occupancy model...")
    print("  This may take 15-45 minutes...")

    with pm.Model() as model:

        # -----------------------------
        # Occupancy (ψ) — SITE LEVEL
        # -----------------------------
        alpha_psi = pm.Normal("alpha_psi", 0, 1)
        beta_psi = pm.Normal("beta_psi", 0, 1, shape=X_psi.shape[1])

        logit_psi = alpha_psi + pm.math.dot(X_psi, beta_psi)
        psi_site = pm.Deterministic("psi_site", pm.math.sigmoid(logit_psi))

        z = pm.Bernoulli("z", p=psi_site, shape=n_sites)

        # -----------------------------
        # Detection (p) — OBS LEVEL
        # -----------------------------
        alpha_p = pm.Normal("alpha_p", 0, 1)
        beta_p = pm.Normal("beta_p", 0, 1, shape=X_p.shape[1])

        logit_p = alpha_p + pm.math.dot(X_p, beta_p)
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p))

        # -----------------------------
        # Observation model (THE FIX)
        # -----------------------------
        pm.Bernoulli(
            "y",
            p=z[site_idx] * p,
            observed=y_obs
        )

        # -----------------------------
        # Sampling
        # -----------------------------
        trace = pm.sample(
            draws=2000,
            tune=2000,
            chains=4,
            target_accept=0.9,
            random_seed=42
        )

    # -----------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------
    print("\n[4/5] Saving results...")
    
    # Construct output filename
    if guild:
        out_file = OUT_DIR / f"posterior_env_{guild}.nc"
    else:
        out_file = OUT_DIR / "posterior_env.nc"
    
    az.to_netcdf(trace, out_file)
    print(f"  ✓ Saved: {out_file}")
    
    # Also save standardization parameters for prediction
    std_params = pd.DataFrame({
        'covariate': available_covars,
        'mean': X_mean,
        'std': X_std
    })
    
    if guild:
        std_file = OUT_DIR / f"covariate_standardization_{guild}.csv"
    else:
        std_file = OUT_DIR / "covariate_standardization.csv"
    
    std_params.to_csv(std_file, index=False)
    print(f"  ✓ Saved standardization params: {std_file}")

    # -----------------------------------------------------------------
    # Summarize and interpret
    # -----------------------------------------------------------------
    print("\n[5/5] Model diagnostics...")
    
    print("\n" + "=" * 70)
    print("CONVERGENCE DIAGNOSTICS")
    print("=" * 70)
    
    summary = az.summary(
        trace,
        var_names=["alpha_psi", "beta_psi", "alpha_p", "beta_p"],
        hdi_prob=0.95
    )
    print(summary)
    
    # Covariate effects
    print("\n" + "=" * 70)
    print("COVARIATE EFFECTS ON OCCUPANCY")
    print("=" * 70)
    
    beta_samples = trace.posterior["beta_psi"].values  # (chain, draw, covariate)
    print("Covariate order:", available_covars)
    
    for i, cov in enumerate(available_covars):
        beta_i = beta_samples[:, :, i].flatten()
        mean_beta = beta_i.mean()
        hdi_low = np.percentile(beta_i, 2.5)
        hdi_high = np.percentile(beta_i, 97.5)
        prob_positive = (beta_i > 0).mean()
        
        print(f"\n  {cov}:")
        print(f"    β = {mean_beta:+.3f} [{hdi_low:+.3f}, {hdi_high:+.3f}]")
        print(f"    P(β > 0) = {prob_positive:.3f}")
        
        if prob_positive > 0.95:
            print(f"    → Strong POSITIVE effect ✓")
        elif prob_positive < 0.05:
            print(f"    → Strong NEGATIVE effect ✓")
        else:
            print(f"    → Weak or uncertain effect")
    
    # Occupancy summary
    psi_samples = trace.posterior["psi_site"].values.flatten()
    print(f"\n" + "=" * 70)
    print(f"OCCUPANCY SUMMARY")
    print("=" * 70)
    print(f"Mean ψ: {psi_samples.mean():.3f}")
    print(f"Median ψ: {np.median(psi_samples):.3f}")
    print(f"Range: [{psi_samples.min():.3f}, {psi_samples.max():.3f}]")
    
    print("\n✅ PHASE 3C COMPLETE")
    print("=" * 70)

    return 0


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fit environmental occupancy model'
    )
    parser.add_argument(
        '--guild',
        type=str,
        choices=['Wetland', 'Forest', 'Urban'],
        default=None,
        help='Specific guild to model (default: all guilds combined)'
    )
    
    args = parser.parse_args()
    
    return fit_environmental_model(guild=args.guild)


if __name__ == "__main__":
    exit(main())