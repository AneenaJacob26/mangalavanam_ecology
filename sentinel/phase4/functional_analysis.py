#!/usr/bin/env python3
"""
Mangalavanam Adaptive Sentinel
Phase 4: Functional Analysis (PAPER VERSION - SMALL DATASET)

For papers with limited spatial data (n < 100 pixels):
- Extracts occupancy from posteriors (not broken rasters)
- Calculates Shannon entropy from posterior predictions
- Provides site-level functional diversity metrics
- Honest about limitations

Author: Aneena Jacob
Date: March 2026
"""

import numpy as np
import pandas as pd
import arviz as az
from pathlib import Path
from scipy.special import expit
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

PHASE3_DIR = Path("data/processed/phase3")
PHASE4_DIR = Path("data/processed/phase4")
PHASE4_DIR.mkdir(exist_ok=True, parents=True)

GUILDS = ['Wetland', 'Forest', 'Urban']

print("=" * 80)
print("PHASE 4 — FUNCTIONAL ANALYSIS (SMALL DATASET VERSION)")
print("=" * 80)
print()
print("NOTE: This version extracts occupancy directly from posteriors")
print("      Appropriate for datasets with < 100 pixels")
print()

# =============================================================================
# PART 1: EXTRACT GUILD OCCUPANCIES FROM POSTERIORS
# =============================================================================

print("=" * 80)
print("PART 1: EXTRACTING GUILD OCCUPANCIES FROM POSTERIORS")
print("=" * 80)
print()

guild_occupancies = {}
guild_profiles = []

for guild in GUILDS:
    print(f"--- {guild} Guild ---")
    
    profile = {'Guild': guild}
    
    # ================================================================
    # ENVIRONMENTAL MODEL (PRIMARY)
    # ================================================================
    env_file = PHASE3_DIR / f"posterior_env_{guild}.nc"
    
    if env_file.exists():
        try:
            idata = az.from_netcdf(env_file)
            
            if 'psi_site' in idata.posterior:
                # Extract site-level predictions
                psi_samples = idata.posterior['psi_site'].values  # (chain, draw, site)
                psi_mean = psi_samples.mean(axis=(0, 1))  # Mean across MCMC
                psi_std = psi_samples.std(axis=(0, 1))
                
                # Store for functional diversity calculation
                guild_occupancies[guild] = psi_mean
                
                # Guild profile statistics
                profile['Env_mean'] = float(psi_mean.mean())
                profile['Env_std'] = float(psi_std.mean())
                
                print(f"  ✓ Environmental model:")
                print(f"    Sites: {len(psi_mean)}")
                print(f"    Mean ψ: {psi_mean.mean():.3f}")
                print(f"    Std ψ: {psi_std.mean():.3f}")
                print(f"    Range: [{psi_mean.min():.3f}, {psi_mean.max():.3f}]")
            else:
                print(f"  ⚠️ No psi_site in posterior")
                profile['Env_mean'] = np.nan
                profile['Env_std'] = np.nan
        except Exception as e:
            print(f"  ❌ Error loading environmental model: {e}")
            profile['Env_mean'] = np.nan
            profile['Env_std'] = np.nan
    else:
        print(f"  ❌ Environmental model not found")
        profile['Env_mean'] = np.nan
        profile['Env_std'] = np.nan
    
    # ================================================================
    # YEAR MODEL (OPTIONAL)
    # ================================================================
    year_file = PHASE3_DIR / f"posterior_year_{guild}.nc"
    
    if year_file.exists():
        try:
            idata = az.from_netcdf(year_file)
            
            if 'alpha_psi' in idata.posterior:
                alpha = idata.posterior['alpha_psi'].values
                
                if 'beta_year' in idata.posterior:
                    beta_year = idata.posterior['beta_year'].values
                    # Mean effect across years
                    mean_beta = beta_year.mean(axis=2) if beta_year.ndim > 2 else beta_year
                    psi = expit(alpha + mean_beta)
                else:
                    psi = expit(alpha)
                
                profile['Year'] = float(psi.mean())
                print(f"  ✓ Year model: ψ = {profile['Year']:.3f}")
            else:
                profile['Year'] = np.nan
        except:
            profile['Year'] = np.nan
    else:
        profile['Year'] = np.nan
    
    # ================================================================
    # ZONE MODEL (OPTIONAL)
    # ================================================================
    zone_file = PHASE3_DIR / f"posterior_zone_{guild}.nc"
    
    if zone_file.exists():
        try:
            idata = az.from_netcdf(zone_file)
            
            if 'alpha_psi' in idata.posterior:
                alpha = idata.posterior['alpha_psi'].values
                psi = expit(alpha)
                profile['Zone'] = float(psi.mean())
                print(f"  ✓ Zone model: ψ = {profile['Zone']:.3f}")
            else:
                profile['Zone'] = np.nan
        except:
            profile['Zone'] = np.nan
    else:
        profile['Zone'] = np.nan
    
    guild_profiles.append(profile)
    print()

# Check if we have all guild occupancies
if len(guild_occupancies) != len(GUILDS):
    print("❌ ERROR: Missing occupancy predictions for some guilds")
    print(f"   Have: {list(guild_occupancies.keys())}")
    print(f"   Need: {GUILDS}")
    exit(1)

print(f"✓ All guild occupancies extracted successfully")
print()

# =============================================================================
# PART 2: CALCULATE SHANNON ENTROPY & FUNCTIONAL DIVERSITY
# =============================================================================

print("=" * 80)
print("PART 2: FUNCTIONAL DIVERSITY INDICES")
print("=" * 80)
print()

# Get occupancy arrays
psi_wetland = guild_occupancies['Wetland']
psi_forest = guild_occupancies['Forest']
psi_urban = guild_occupancies['Urban']

n_sites = len(psi_wetland)

print(f"Calculating functional diversity for {n_sites} sites...")
print()

# ================================================================
# SHANNON ENTROPY
# ================================================================
def calculate_shannon_entropy(p_w, p_f, p_u):
    """
    Shannon entropy: H = -Σ (p_i * log(p_i))
    where p_i = ψ_i / Σψ (normalized proportions)
    """
    total = p_w + p_f + p_u
    
    # Avoid division by zero
    total_safe = np.where(total == 0, 1e-10, total)
    
    # Normalize to proportions
    p_w_norm = p_w / total_safe
    p_f_norm = p_f / total_safe
    p_u_norm = p_u / total_safe
    
    # Calculate entropy
    H = np.zeros_like(p_w)
    for p in [p_w_norm, p_f_norm, p_u_norm]:
        mask = p > 0
        H[mask] -= p[mask] * np.log(p[mask])
    
    # Set to 0 where no guilds present
    H[total == 0] = 0
    
    return H

shannon_H = calculate_shannon_entropy(psi_wetland, psi_forest, psi_urban)

print("Shannon Diversity (H):")
print(f"  Formula: H = -Σ (p_g · log(p_g))")
print(f"  Mean: {shannon_H.mean():.3f}")
print(f"  Std: {shannon_H.std():.3f}")
print(f"  Range: [{shannon_H.min():.3f}, {shannon_H.max():.3f}]")
print(f"  Interpretation: Higher values = more even guild distribution")
print()

# ================================================================
# SIMPSON DIVERSITY
# ================================================================
def calculate_simpson_diversity(p_w, p_f, p_u):
    """Simpson's D = 1 - Σ(p_i²)"""
    total = p_w + p_f + p_u
    total_safe = np.where(total == 0, 1e-10, total)
    
    p_w_norm = p_w / total_safe
    p_f_norm = p_f / total_safe
    p_u_norm = p_u / total_safe
    
    D = 1 - (p_w_norm**2 + p_f_norm**2 + p_u_norm**2)
    
    return D

simpson_D = calculate_simpson_diversity(psi_wetland, psi_forest, psi_urban)

print("Simpson Diversity (D):")
print(f"  Formula: D = 1 - Σ(p_i²)")
print(f"  Mean: {simpson_D.mean():.3f}")
print(f"  Std: {simpson_D.std():.3f}")
print(f"  Range: [{simpson_D.min():.3f}, {simpson_D.max():.3f}]")
print()

# ================================================================
# FUNCTIONAL RICHNESS
# ================================================================
def calculate_functional_richness(p_w, p_f, p_u, threshold=0.5):
    """Number of guilds with ψ > threshold"""
    richness = (p_w > threshold).astype(int) + \
               (p_f > threshold).astype(int) + \
               (p_u > threshold).astype(int)
    return richness

richness = calculate_functional_richness(psi_wetland, psi_forest, psi_urban)

print("Functional Richness:")
print(f"  Threshold: ψ > 0.5")
print(f"  Mean: {richness.mean():.2f} guilds/site")
print(f"  Std: {richness.std():.2f}")
print(f"  Range: [{richness.min():.0f}, {richness.max():.0f}]")
print()

# ================================================================
# FUNCTIONAL EVENNESS
# ================================================================
def calculate_functional_evenness(H, richness):
    """J = H / log(S) where S = richness"""
    evenness = np.zeros_like(H)
    mask = richness > 0
    
    log_richness = np.log(richness[mask])
    log_richness_safe = np.where(log_richness == 0, 1, log_richness)
    
    evenness[mask] = H[mask] / log_richness_safe
    
    return evenness

evenness = calculate_functional_evenness(shannon_H, richness)

print("Functional Evenness (J):")
print(f"  Formula: J = H / log(richness)")
print(f"  Mean: {evenness.mean():.3f}")
print(f"  Std: {evenness.std():.3f}")
print()

# ================================================================
# TOTAL OCCUPANCY
# ================================================================
total_occ = psi_wetland + psi_forest + psi_urban
mean_occ = total_occ / 3

print("Total & Mean Occupancy:")
print(f"  Total occupancy (sum): {total_occ.mean():.2f} ± {total_occ.std():.2f}")
print(f"  Mean occupancy (avg): {mean_occ.mean():.3f} ± {mean_occ.std():.3f}")
print()

# ================================================================
# FUNCTIONAL IMPORTANCE
# ================================================================
functional_importance = mean_occ * shannon_H

print("Functional Importance:")
print(f"  Formula: ψ̄ × H")
print(f"  Mean: {functional_importance.mean():.3f}")
print(f"  Std: {functional_importance.std():.3f}")
print()

# =============================================================================
# PART 3: SAVE OUTPUTS
# =============================================================================

print("=" * 80)
print("PART 3: SAVING OUTPUTS")
print("=" * 80)
print()

# ================================================================
# GUILD PROFILES
# ================================================================
profiles_df = pd.DataFrame(guild_profiles)
profiles_file = PHASE4_DIR / "guild_profiles.csv"
profiles_df.to_csv(profiles_file, index=False)

print("Guild Profiles:")
print(profiles_df.to_string(index=False))
print()
print(f"✓ Saved: {profiles_file.name}")
print()

# ================================================================
# FUNCTIONAL INDICES SUMMARY
# ================================================================
indices_summary = pd.DataFrame({
    'Index': [
        'Shannon Diversity',
        'Simpson Diversity',
        'Functional Richness',
        'Functional Evenness',
        'Total Occupancy',
        'Mean Occupancy',
        'Functional Importance'
    ],
    'Mean': [
        shannon_H.mean(),
        simpson_D.mean(),
        richness.mean(),
        evenness.mean(),
        total_occ.mean(),
        mean_occ.mean(),
        functional_importance.mean()
    ],
    'Std': [
        shannon_H.std(),
        simpson_D.std(),
        richness.std(),
        evenness.std(),
        total_occ.std(),
        mean_occ.std(),
        functional_importance.std()
    ],
    'Min': [
        shannon_H.min(),
        simpson_D.min(),
        richness.min(),
        evenness.min(),
        total_occ.min(),
        mean_occ.min(),
        functional_importance.min()
    ],
    'Max': [
        shannon_H.max(),
        simpson_D.max(),
        richness.max(),
        evenness.max(),
        total_occ.max(),
        mean_occ.max(),
        functional_importance.max()
    ]
})

indices_file = PHASE4_DIR / "functional_indices_summary.csv"
indices_summary.to_csv(indices_file, index=False)

print("Functional Diversity Indices:")
print(indices_summary.to_string(index=False))
print()
print(f"✓ Saved: {indices_file.name}")
print()

# ================================================================
# SITE-LEVEL DATA (for CPI calculation)
# ================================================================
site_data = pd.DataFrame({
    'site_id': range(n_sites),
    'psi_wetland': psi_wetland,
    'psi_forest': psi_forest,
    'psi_urban': psi_urban,
    'shannon_H': shannon_H,
    'simpson_D': simpson_D,
    'richness': richness,
    'evenness': evenness,
    'total_occ': total_occ,
    'mean_occ': mean_occ,
    'functional_importance': functional_importance
})

site_file = PHASE4_DIR / "site_level_diversity.csv"
site_data.to_csv(site_file, index=False)

print(f"✓ Saved: {site_file.name}")
print(f"  Contains site-level values for {n_sites} sites")
print()

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 80)
print("PHASE 4 COMPLETE")
print("=" * 80)
print()

print("Generated files:")
print(f"  ✓ guild_profiles.csv - Guild-level occupancy statistics")
print(f"  ✓ functional_indices_summary.csv - Diversity metrics")
print(f"  ✓ site_level_diversity.csv - Site-level values for CPI")
print()

print("Key Results:")
print(f"  Number of sites: {n_sites}")
print(f"  Shannon Diversity: {shannon_H.mean():.3f} ± {shannon_H.std():.3f}")
print(f"  Simpson Diversity: {simpson_D.mean():.3f} ± {simpson_D.std():.3f}")
print(f"  Mean Richness: {richness.mean():.2f} guilds/site")
print(f"  Mean Occupancy: {mean_occ.mean():.3f} ± {mean_occ.std():.3f}")
print()

print("For Paper:")
print(f"  Use these values in Table IV (Functional Diversity)")
print(f"  Shannon entropy validates CPI diversity component")
print(f"  Site-level data feeds into Phase 6 (CPI calculation)")
print()

print("=" * 80)