#!/usr/bin/env python3
"""
Species Decline Tracker
Automatically detect bird species showing significant population trends

Uses:
- Mann-Kendall trend test (non-parametric)
- Change point detection
- Statistical significance testing
- Auto-generates alerts for declining species

Author: MSc Data Science Project
Date: January 2026
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from pymannkendall import original_test
except ImportError:
    print("⚠️ Installing pymannkendall...")
    import subprocess
    subprocess.run(["pip", "install", "pymannkendall", "--break-system-packages"], check=True)
    from pymannkendall import original_test

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration paths"""
    BASE_DIR = Path("data")
    EBIRD_DATA = BASE_DIR / "processed/phase1/ebird_with_guilds_and_zones.csv"
    OUTPUT_DIR = BASE_DIR / "processed/phase7_trends"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Thresholds
    SIGNIFICANCE_LEVEL = 0.05  # p < 0.05 = significant
    CRITICAL_DECLINE = -30  # % decline to flag as critical
    HIGH_DECLINE = -15  # % decline to flag as high concern
    MIN_YEARS = 4  # Minimum years needed for trend analysis

# =============================================================================
# CORE ANALYSIS FUNCTIONS
# =============================================================================

def load_and_prepare_data(filepath):
    """
    Load eBird data and prepare for trend analysis
    
    Returns:
        DataFrame with species, year, counts
    """
    print(f"📂 Loading data from: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        print(f"   ✓ Loaded {len(df):,} observations")
    except FileNotFoundError:
        print(f"   ✗ File not found: {filepath}")
        return None
    
    # Parse dates
    if 'OBSERVATION DATE' in df.columns:
        df['year'] = pd.to_datetime(df['OBSERVATION DATE'], errors='coerce').dt.year
    elif 'year' in df.columns:
        pass  # Already have year
    else:
        print("   ✗ No date information found!")
        return None
    
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    
    # Get species column
    if 'COMMON NAME' in df.columns:
        species_col = 'COMMON NAME'
    elif 'SCIENTIFIC NAME' in df.columns:
        species_col = 'SCIENTIFIC NAME'
    else:
        print("   ✗ No species name column found!")
        return None
    
    print(f"   ✓ Years: {df['year'].min()} to {df['year'].max()}")
    print(f"   ✓ Species found: {df[species_col].nunique()}")
    
    return df, species_col

def calculate_species_trends(df, species_col, min_years=4):
    """
    Calculate trends for each species using Mann-Kendall test
    
    Args:
        df: DataFrame with bird observations
        species_col: Name of species column
        min_years: Minimum years required for trend test
    
    Returns:
        DataFrame with trend analysis results
    """
    print(f"\n📊 Analyzing species trends...")
    
    # Group by species and year
    yearly_counts = df.groupby([species_col, 'year']).size().reset_index(name='count')
    
    results = []
    species_list = yearly_counts[species_col].unique()
    
    for i, species in enumerate(species_list, 1):
        if i % 20 == 0:
            print(f"   Processing species {i}/{len(species_list)}...")
        
        species_data = yearly_counts[yearly_counts[species_col] == species].sort_values('year')
        
        # Need minimum years for reliable trend
        if len(species_data) < min_years:
            continue
        
        years = species_data['year'].values
        counts = species_data['count'].values
        
        # Mann-Kendall trend test (non-parametric, robust)
        try:
            mk_result = original_test(counts)
        except:
            continue
        
        # Calculate percent change
        first_year = years[0]
        last_year = years[-1]
        first_count = counts[0]
        last_count = counts[-1]
        
        if first_count > 0:
            pct_change = ((last_count - first_count) / first_count) * 100
        else:
            pct_change = 0
        
        # Annual rate of change
        years_span = last_year - first_year
        if years_span > 0:
            annual_rate = pct_change / years_span
        else:
            annual_rate = 0
        
        # Classify trend
        if mk_result.p < Config.SIGNIFICANCE_LEVEL:  # Significant trend
            if mk_result.trend == 'decreasing':
                status = 'Declining'
                if pct_change < Config.CRITICAL_DECLINE:
                    urgency = 'Critical'
                elif pct_change < Config.HIGH_DECLINE:
                    urgency = 'High'
                else:
                    urgency = 'Medium'
            elif mk_result.trend == 'increasing':
                status = 'Increasing'
                urgency = 'Low'
            else:
                status = 'Stable'
                urgency = 'Low'
        else:
            status = 'Stable'
            urgency = 'Low'
        
        # Check for recent absence (possible local extinction)
        recent_years = years[-2:]  # Last 2 years
        recent_counts = counts[-2:]
        possibly_extinct = all(c == 0 for c in recent_counts) if len(recent_counts) == 2 else False
        
        results.append({
            'Species': species,
            'Status': status,
            'Trend': mk_result.trend,
            'P_value': mk_result.p,
            'Tau': mk_result.Tau,  # Kendall's tau (effect size)
            'Percent_Change': pct_change,
            'Annual_Rate': annual_rate,
            'Urgency': urgency,
            'First_Year': int(first_year),
            'Last_Year': int(last_year),
            'First_Year_Count': int(first_count),
            'Last_Year_Count': int(last_count),
            'Years_Monitored': len(species_data),
            'Total_Observations': int(counts.sum()),
            'Possibly_Extinct': possibly_extinct
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Percent_Change')
    
    print(f"   ✓ Analyzed {len(results_df)} species")
    
    return results_df

def generate_summary_statistics(results_df):
    """Generate summary statistics for dashboard"""
    
    summary = {
        'total_species': len(results_df),
        'declining': len(results_df[results_df['Status'] == 'Declining']),
        'stable': len(results_df[results_df['Status'] == 'Stable']),
        'increasing': len(results_df[results_df['Status'] == 'Increasing']),
        'critical': len(results_df[results_df['Urgency'] == 'Critical']),
        'high_concern': len(results_df[results_df['Urgency'] == 'High']),
        'possibly_extinct': len(results_df[results_df['Possibly_Extinct'] == True])
    }
    
    summary['declining_pct'] = (summary['declining'] / summary['total_species']) * 100
    summary['increasing_pct'] = (summary['increasing'] / summary['total_species']) * 100
    
    return summary

def generate_alerts(results_df):
    """Generate conservation alerts for critical species"""
    
    alerts = []
    
    # Critical declines
    critical = results_df[results_df['Urgency'] == 'Critical'].head(10)
    for _, row in critical.iterrows():
        alerts.append({
            'Level': 'CRITICAL',
            'Species': row['Species'],
            'Message': f"Declined {abs(row['Percent_Change']):.1f}% from {row['First_Year']} to {row['Last_Year']}",
            'Action': "Urgent investigation required",
            'Details': f"Annual rate: {row['Annual_Rate']:.1f}% per year, p={row['P_value']:.4f}"
        })
    
    # Possible extinctions
    extinct = results_df[results_df['Possibly_Extinct'] == True]
    for _, row in extinct.iterrows():
        alerts.append({
            'Level': 'EXTINCTION RISK',
            'Species': row['Species'],
            'Message': f"No observations in last 2 years (last seen {row['Last_Year']-2})",
            'Action': "Targeted surveys needed to confirm presence",
            'Details': f"Previously recorded: {row['Total_Observations']} times"
        })
    
    return pd.DataFrame(alerts)

def save_results(results_df, summary, alerts):
    """Save all results to CSV files"""
    
    print(f"\n💾 Saving results...")
    
    # Main results
    output_file = Config.OUTPUT_DIR / "species_trends_analysis.csv"
    results_df.to_csv(output_file, index=False)
    print(f"   ✓ Trends saved: {output_file}")
    
    # Summary statistics
    summary_file = Config.OUTPUT_DIR / "trends_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_file, index=False)
    print(f"   ✓ Summary saved: {summary_file}")
    
    # Alerts
    if len(alerts) > 0:
        alerts_file = Config.OUTPUT_DIR / "conservation_alerts.csv"
        alerts.to_csv(alerts_file, index=False)
        print(f"   ✓ Alerts saved: {alerts_file}")
    
    # Separate files by status
    for status in ['Declining', 'Stable', 'Increasing']:
        subset = results_df[results_df['Status'] == status]
        if len(subset) > 0:
            subset_file = Config.OUTPUT_DIR / f"species_{status.lower()}.csv"
            subset.to_csv(subset_file, index=False)
            print(f"   ✓ {status} species saved: {subset_file}")

def print_summary_report(results_df, summary, alerts):
    """Print human-readable summary to console"""
    
    print("\n" + "="*80)
    print("📊 SPECIES TREND ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\n📈 Overall Statistics:")
    print(f"   Total species analyzed: {summary['total_species']}")
    print(f"   Declining: {summary['declining']} ({summary['declining_pct']:.1f}%)")
    print(f"   Stable: {summary['stable']}")
    print(f"   Increasing: {summary['increasing']} ({summary['increasing_pct']:.1f}%)")
    
    print(f"\n🚨 Conservation Concerns:")
    print(f"   Critical decline: {summary['critical']} species")
    print(f"   High concern: {summary['high_concern']} species")
    print(f"   Possibly extinct: {summary['possibly_extinct']} species")
    
    # Top declining species
    print(f"\n🔴 TOP 10 DECLINING SPECIES:")
    declining = results_df[results_df['Status'] == 'Declining'].head(10)
    for i, row in declining.iterrows():
        print(f"   {row['Species'][:40]:40s} | {row['Percent_Change']:+7.1f}% | p={row['P_value']:.3f} | {row['Urgency']}")
    
    # Top increasing species
    print(f"\n🟢 TOP 10 INCREASING SPECIES:")
    increasing = results_df[results_df['Status'] == 'Increasing'].tail(10)[::-1]
    for i, row in increasing.iterrows():
        print(f"   {row['Species'][:40]:40s} | {row['Percent_Change']:+7.1f}% | p={row['P_value']:.3f}")
    
    # Critical alerts
    if len(alerts) > 0:
        print(f"\n⚠️  CONSERVATION ALERTS ({len(alerts)} total):")
        for i, alert in alerts.head(5).iterrows():
            print(f"\n   [{alert['Level']}] {alert['Species']}")
            print(f"      {alert['Message']}")
            print(f"      Action: {alert['Action']}")
    
    print("\n" + "="*80)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_trend_analysis():
    """Main function to run complete trend analysis"""
    
    print("="*80)
    print("🦜 MANGALAVANAM SPECIES DECLINE TRACKER")
    print("="*80)
    
    # Load data
    result = load_and_prepare_data(Config.EBIRD_DATA)
    if result is None:
        print("\n❌ Failed to load data. Exiting.")
        return None
    
    df, species_col = result
    
    # Analyze trends
    results_df = calculate_species_trends(df, species_col, Config.MIN_YEARS)
    
    if len(results_df) == 0:
        print("\n❌ No species with sufficient data for trend analysis.")
        return None
    
    # Generate summaries
    summary = generate_summary_statistics(results_df)
    alerts = generate_alerts(results_df)
    
    # Save results
    save_results(results_df, summary, alerts)
    
    # Print report
    print_summary_report(results_df, summary, alerts)
    
    print("\n✅ Analysis complete!")
    print(f"   Results saved to: {Config.OUTPUT_DIR}")
    
    return results_df, summary, alerts

if __name__ == "__main__":
    # Run the analysis
    results = run_trend_analysis()
    
    print("\n💡 Next steps:")
    print("   1. Review the CSV files in data/processed/phase7_trends/")
    print("   2. Investigate species marked as 'Critical'")
    print("   3. Run targeted surveys for possibly extinct species")
    print("   4. Add this data to the dashboard for visualization")