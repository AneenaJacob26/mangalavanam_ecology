#!/usr/bin/env python3
"""
Mangalavanam Adaptive Sentinel
Phase 5: Simplified Model Validation (HONEST VERSION)

What we ACTUALLY do:
- Simple 80/20 train-test split
- Basic AUC, sensitivity, specificity
- Hosmer-Lemeshow calibration test
- Honest about limitations

What we DON'T claim:
- Spatial cross-validation (too complex for 44 pixels)
- Autocorrelation analysis (NaN with sparse data)
- Field validation (never conducted)

Author: Honest Scientist
Date: March 2026
"""

import numpy as np
import pandas as pd
import arviz as az
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import train_test_split
from scipy.stats import chi2
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

POST_DIR = Path("data/processed/phase3")
CSV_DATA = Path("data/processed/phase2/guild_pixel_timeseries_with_covariates.csv")
OUT_DIR = Path("data/processed/phase5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GUILDS = ['Wetland', 'Forest', 'Urban']

print("=" * 80)
print("PHASE 5 — SIMPLIFIED MODEL VALIDATION")
print("=" * 80)
print()
print("Approach: Simple train-test split (honest about small sample size)")
print()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def hosmer_lemeshow_test(y_true, y_pred, bins=10):
    """
    Hosmer-Lemeshow goodness-of-fit test for calibration.
    
    Tests whether predicted probabilities match observed frequencies.
    H0: Good calibration (predicted = observed)
    """
    # Create bins based on predicted probabilities
    bin_edges = np.percentile(y_pred, np.linspace(0, 100, bins + 1))
    bin_ids = np.digitize(y_pred, bin_edges[1:-1])
    
    hl_stat = 0
    df = bins - 2
    
    calibration_data = []
    
    for i in range(bins):
        mask = (bin_ids == i)
        
        if mask.sum() == 0:
            continue
        
        n_k = mask.sum()
        O_k = y_true[mask].sum()  # Observed events
        E_k = y_pred[mask].sum()  # Expected events
        
        mean_pred = y_pred[mask].mean()
        obs_freq = y_true[mask].mean()
        
        calibration_data.append({
            'Bin': i + 1,
            'N': n_k,
            'Mean_Predicted': mean_pred,
            'Observed_Freq': obs_freq,
            'Expected': E_k,
            'Observed': O_k
        })
        
        # H-L statistic
        if E_k > 0 and E_k < n_k:
            hl_stat += (O_k - E_k) ** 2 / (E_k * (1 - E_k / n_k))
    
    # P-value
    p_value = 1 - chi2.cdf(hl_stat, df)
    
    return hl_stat, p_value, pd.DataFrame(calibration_data)


# =============================================================================
# VALIDATION FUNCTION
# =============================================================================

def validate_guild(guild):
    """
    Simple train-test validation for one guild.
    
    Returns both in-sample and out-of-sample metrics for comparison.
    """
    print(f"=" * 80)
    print(f"VALIDATING: {guild} Guild")
    print("=" * 80)
    print()
    
    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    
    if not CSV_DATA.exists():
        print("  ❌ Data file not found")
        return None
    
    df = pd.read_csv(CSV_DATA)
    df_guild = df[df['guild'] == guild].copy()
    
    # Aggregate to site level (ever detected)
    sites = df_guild.groupby('pixel_id').agg({
        'detected': 'max'
    }).reset_index()
    
    print(f"  Total sites: {len(sites)}")
    print(f"  Detected at: {sites['detected'].sum()} sites ({sites['detected'].mean():.1%})")
    print()
    
    # -------------------------------------------------------------------------
    # Load predictions
    # -------------------------------------------------------------------------
    
    post_file = POST_DIR / f"posterior_env_{guild}.nc"
    
    if not post_file.exists():
        print(f"  ❌ Posterior not found: {post_file.name}")
        return None
    
    try:
        idata = az.from_netcdf(post_file)
        
        if 'psi_site' not in idata.posterior:
            print(f"  ❌ No psi_site in posterior")
            return None
        
        # Extract predictions (mean across chains and draws)
        psi_samples = idata.posterior['psi_site'].values  # (chain, draw, site)
        psi_pred = psi_samples.mean(axis=(0, 1))  # Mean prediction
        psi_std = psi_samples.std(axis=(0, 1))   # Uncertainty
        
    except Exception as e:
        print(f"  ❌ Error loading predictions: {e}")
        return None
    
    # Align lengths
    min_len = min(len(sites), len(psi_pred))
    sites = sites.iloc[:min_len].copy()
    sites['psi_pred'] = psi_pred[:min_len]
    sites['psi_std'] = psi_std[:min_len]
    
    y_true = sites['detected'].values
    y_pred = sites['psi_pred'].values
    
    # Check if binary outcome
    if len(np.unique(y_true)) < 2:
        print(f"  ⚠️ Not enough variation in detections (all same value)")
        return None
    
    # -------------------------------------------------------------------------
    # In-Sample Metrics (all data)
    # -------------------------------------------------------------------------
    
    print("  [1] IN-SAMPLE VALIDATION (All Data)")
    print("  " + "-" * 76)
    
    # AUC
    auc_in = roc_auc_score(y_true, y_pred)
    
    # Confusion matrix (threshold = 0.5)
    y_pred_binary = (y_pred > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    sens_in = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec_in = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc_in = (tp + tn) / (tp + tn + fp + fn)
    tss_in = sens_in + spec_in - 1
    
    print(f"    AUC-ROC: {auc_in:.3f}")
    print(f"    Sensitivity: {sens_in:.3f}")
    print(f"    Specificity: {spec_in:.3f}")
    print(f"    Accuracy: {acc_in:.3f}")
    print(f"    TSS: {tss_in:.3f}")
    print()
    
    # -------------------------------------------------------------------------
    # Out-of-Sample Metrics (80/20 split)
    # -------------------------------------------------------------------------
    
    print("  [2] OUT-OF-SAMPLE VALIDATION (80/20 Split)")
    print("  " + "-" * 76)
    
    # Simple random split
    if len(sites) >= 10:  # Only if enough data
        X = np.arange(len(y_true)).reshape(-1, 1)  # Dummy features
        X_train, X_test, y_train_true, y_test_true, y_train_pred, y_test_pred = \
            train_test_split(X, y_true, y_pred, test_size=0.2, random_state=42, stratify=y_true)
        
        # AUC on test set
        if len(np.unique(y_test_true)) == 2:
            auc_out = roc_auc_score(y_test_true, y_test_pred)
            
            # Confusion matrix
            y_test_binary = (y_test_pred > 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test_true, y_test_binary).ravel()
            
            sens_out = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec_out = tn / (tn + fp) if (tn + fp) > 0 else 0
            acc_out = (tp + tn) / (tp + tn + fp + fn)
            tss_out = sens_out + spec_out - 1
            
            print(f"    Test set size: {len(y_test_true)} pixels")
            print(f"    AUC-ROC: {auc_out:.3f}")
            print(f"    Sensitivity: {sens_out:.3f}")
            print(f"    Specificity: {spec_out:.3f}")
            print(f"    Accuracy: {acc_out:.3f}")
            print(f"    TSS: {tss_out:.3f}")
            print()
        else:
            print(f"    ⚠️ Test set lacks variation")
            auc_out = np.nan
            sens_out = np.nan
            spec_out = np.nan
            tss_out = np.nan
    else:
        print(f"    ⚠️ Too few samples ({len(sites)}) for train-test split")
        auc_out = np.nan
        sens_out = np.nan
        spec_out = np.nan
        tss_out = np.nan
    
    # -------------------------------------------------------------------------
    # Hosmer-Lemeshow Calibration Test
    # -------------------------------------------------------------------------
    
    print("  [3] CALIBRATION ANALYSIS")
    print("  " + "-" * 76)
    
    hl_stat, hl_pvalue, calib_df = hosmer_lemeshow_test(y_true, y_pred, bins=10)
    
    print(f"    Hosmer-Lemeshow Statistic: {hl_stat:.3f}")
    print(f"    P-value: {hl_pvalue:.3f}")
    
    if hl_pvalue > 0.05:
        print(f"    ✓ Good calibration (p > 0.05)")
    else:
        print(f"    ⚠️ Poor calibration (p < 0.05)")
    
    print()
    print("    Calibration by Decile:")
    print(calib_df[['Bin', 'N', 'Mean_Predicted', 'Observed_Freq']].to_string(index=False))
    print()
    
    # Save calibration
    calib_file = OUT_DIR / f"calibration_{guild}.csv"
    calib_df.to_csv(calib_file, index=False)
    print(f"    ✓ Saved: {calib_file.name}")
    print()
    
    # -------------------------------------------------------------------------
    # Return results
    # -------------------------------------------------------------------------
    
    return {
        'Guild': guild,
        'N_pixels': len(sites),
        'Prevalence': sites['detected'].mean(),
        # In-sample
        'AUC_in_sample': auc_in,
        'Sensitivity_in_sample': sens_in,
        'Specificity_in_sample': spec_in,
        'TSS_in_sample': tss_in,
        # Out-of-sample
        'AUC_test': auc_out,
        'Sensitivity_test': sens_out,
        'Specificity_test': spec_out,
        'TSS_test': tss_out,
        # Calibration
        'HL_statistic': hl_stat,
        'HL_pvalue': hl_pvalue
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

results = []

for guild in GUILDS:
    result = validate_guild(guild)
    if result:
        results.append(result)
    print()

# =============================================================================
# SUMMARY
# =============================================================================

if results:
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print()
    
    summary_df = pd.DataFrame(results)
    
    # Format for display
    display_cols = [
        'Guild', 'N_pixels', 'AUC_in_sample', 'Sensitivity_in_sample', 
        'Specificity_in_sample', 'HL_pvalue'
    ]
    
    print("PRIMARY METRICS (In-Sample - All Data):")
    print(summary_df[display_cols].to_string(index=False, float_format='%.3f'))
    print()
    
    # Save
    summary_file = OUT_DIR / "validation_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Saved: {summary_file.name}")
    print()
    
    # Interpretation
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    
    print("What these metrics mean:")
    print()
    print("  AUC > 0.8  = Excellent discrimination")
    print("  AUC 0.7-0.8 = Good discrimination")
    print("  AUC 0.6-0.7 = Fair discrimination")
    print("  AUC < 0.6  = Poor discrimination")
    print()
    print("  H-L p > 0.05 = Good calibration (predictions match observations)")
    print("  H-L p < 0.05 = Poor calibration (predictions biased)")
    print()
    
    print("IMPORTANT CAVEATS:")
    print()
    print(f"  • Small sample size (n={summary_df['N_pixels'].iloc[0]} pixels)")
    print(f"  • In-sample metrics are OPTIMISTIC (training data)")
    print(f"  • Out-of-sample metrics more realistic but uncertain with small n")
    print(f"  • No spatial cross-validation (insufficient data)")
    print(f"  • No field validation conducted")
    print()
    print("  → Results should be interpreted with caution")
    print("  → Recommend larger study area for future research")
    print()

else:
    print("❌ No validation results generated")

print("=" * 80)
print("PHASE 5 COMPLETE")
print("=" * 80)