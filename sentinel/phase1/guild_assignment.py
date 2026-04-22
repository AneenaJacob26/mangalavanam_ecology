#!/usr/bin/env python3
"""
Mangalavanam Adaptive Sentinel
Phase 1b: Guild Assignment

Purpose:
- Attach functional guild labels to cleaned eBird observations
- Uses trait-based classification from EltonTraits
- Maintains zone assignments from Phase 1a

Author: Senior Geospatial Data Scientist
Date: January 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def normalize_columns(df):
    """Normalize column names: uppercase, spaces instead of underscores."""
    df.columns = df.columns.str.upper().str.replace('_', ' ')
    return df


def assign_guild(row):
    """
    Trait-based guild assignment.
    
    Classification logic:
    - Wetland: High water foraging OR high fish diet
    - Forest: High canopy foraging AND high invertebrate diet
    - Urban: All others (generalists, ground feeders, human-adapted)
    
    Args:
        row: Species trait row from functional traits database
        
    Returns:
        str: Guild label ('Wetland', 'Forest', or 'Urban')
    """
    
    # Water foraging behavior
    water_cols = [c for c in row.index if 'WAT' in str(c).upper()]
    total_water = sum(row[c] for c in water_cols if pd.notnull(row[c]))
    
    # Fish diet
    diet_fish = row.get('DIET-VFISH', row.get('DIET VFISH', 0))
    if pd.isnull(diet_fish):
        diet_fish = 0
    
    # Canopy foraging
    canopy_cols = [
        c for c in row.index
        if 'CANOPY' in str(c).upper() or 'MIDHIGH' in str(c).upper()
    ]
    total_canopy = sum(row[c] for c in canopy_cols if pd.notnull(row[c]))
    
    # Invertebrate diet
    diet_inv = row.get('DIET-INV', row.get('DIET INV', 0))
    if pd.isnull(diet_inv):
        diet_inv = 0
    
    # Classification thresholds
    if total_water > 40 or diet_fish > 20:
        return 'Wetland'
    elif total_canopy > 40 and diet_inv > 40:
        return 'Forest'
    else:
        return 'Urban'


def load_and_process_traits(trait_file):
    """
    Load functional traits and assign guilds at species level.
    
    Args:
        trait_file: Path to bird_functional_traits.csv
        
    Returns:
        pd.DataFrame: Species guild lookup table
    """
    print("\n[1/3] Loading functional traits...")
    
    traits = pd.read_csv(trait_file)
    traits = normalize_columns(traits)
    
    print(f"  ✓ Loaded {len(traits):,} species trait records")
    
    # Assign guilds to each species
    print("  → Classifying species into guilds...")
    traits['GUILD'] = traits.apply(assign_guild, axis=1)
    
    # Show guild distribution
    guild_counts = traits['GUILD'].value_counts()
    print(f"  ✓ Guild classification:")
    for guild, count in guild_counts.items():
        print(f"    - {guild}: {count} species")
    
    # Create lookup table
    guild_lookup = traits[['SCIENTIFIC', 'GUILD']].copy()
    guild_lookup.columns = ['SCIENTIFIC NAME', 'GUILD']
    
    return guild_lookup


def merge_guilds_with_observations(df, guild_lookup):
    """
    Merge guild assignments onto observation data.
    
    Args:
        df: Cleaned eBird observations with zones
        guild_lookup: Species-guild mapping
        
    Returns:
        pd.DataFrame: Observations with guild labels
    """
    print("\n[2/3] Merging guilds onto observations...")
    
    initial_count = len(df)
    unique_species_before = df['SCIENTIFIC NAME'].nunique()
    
    # Left join to keep all observations
    df = df.merge(
        guild_lookup,
        on='SCIENTIFIC NAME',
        how='left'
    )
    
    # Handle species not in trait database (assign to Urban as generalists)
    missing_guilds = df['GUILD'].isna().sum()
    if missing_guilds > 0:
        print(f"  ⚠ {missing_guilds:,} observations without trait data")
        print(f"    → Assigning to 'Urban' guild (generalist assumption)")
        df['GUILD'] = df['GUILD'].fillna('Urban')
    
    # Verify no data loss
    assert len(df) == initial_count, "Row count changed during merge!"
    assert df['SCIENTIFIC NAME'].nunique() == unique_species_before, "Species lost during merge!"
    
    print(f"  ✓ Guild assignment complete")
    print(f"\n  Observation counts by guild:")
    for guild, count in df['GUILD'].value_counts().items():
        print(f"    - {guild}: {count:,} observations")
    
    return df


def validate_output(df):
    """
    Perform data quality checks on final output.
    
    Args:
        df: Final dataset
    """
    print("\n[3/3] Validating output...")
    
    checks_passed = True
    
    # Check 1: Required columns exist
    required_cols = [
        'SAMPLING EVENT IDENTIFIER',
        'SCIENTIFIC NAME',
        'COMMON NAME',
        'OBSERVATION COUNT',
        'LATITUDE',
        'LONGITUDE',
        'OBSERVATION DATE',
        'YEAR',
        'ZONE',
        'GUILD'
    ]
    
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        print(f"  ✗ Missing columns: {missing_cols}")
        checks_passed = False
    else:
        print(f"  ✓ All required columns present")
    
    # Check 2: No null values in critical columns
    critical_cols = ['SCIENTIFIC NAME', 'YEAR', 'ZONE', 'GUILD']
    null_counts = df[critical_cols].isnull().sum()
    
    if null_counts.sum() > 0:
        print(f"  ✗ Null values found:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"    - {col}: {count}")
        checks_passed = False
    else:
        print(f"  ✓ No null values in critical columns")
    
    # Check 3: Valid year range
    if df['YEAR'].min() < 2019 or df['YEAR'].max() > 2025:
        print(f"  ✗ Invalid year range: {df['YEAR'].min()} - {df['YEAR'].max()}")
        checks_passed = False
    else:
        print(f"  ✓ Valid year range: {df['YEAR'].min()} - {df['YEAR'].max()}")
    
    # Check 4: Valid zones
    valid_zones = {'Core', 'Buffer', 'Outside'}
    invalid_zones = set(df['ZONE'].unique()) - valid_zones
    if invalid_zones:
        print(f"  ✗ Invalid zones: {invalid_zones}")
        checks_passed = False
    else:
        print(f"  ✓ Valid zones: {', '.join(sorted(df['ZONE'].unique()))}")
    
    # Check 5: Valid guilds
    valid_guilds = {'Wetland', 'Forest', 'Urban'}
    invalid_guilds = set(df['GUILD'].unique()) - valid_guilds
    if invalid_guilds:
        print(f"  ✗ Invalid guilds: {invalid_guilds}")
        checks_passed = False
    else:
        print(f"  ✓ Valid guilds: {', '.join(sorted(df['GUILD'].unique()))}")
    
    if checks_passed:
        print(f"\n  ✅ All validation checks passed")
    else:
        print(f"\n  ❌ Some validation checks failed")
    
    return checks_passed


def main():
    """Main execution pipeline."""
    
    print("=" * 70)
    print("PHASE 1b — GUILD ASSIGNMENT")
    print("=" * 70)
    
    # =====================================================================
    # CONFIGURATION
    # =====================================================================
    INPUT_FILE = Path("data/processed/phase1/ebird_cleaned_with_zones.csv")
    TRAIT_FILE = Path("data/raw/bird_functional_traits.csv")
    OUTPUT_FILE = Path("data/processed/phase1/ebird_with_guilds_and_zones.csv")
    
    # =====================================================================
    # EXECUTION
    # =====================================================================
    
    try:
        # Load cleaned eBird data from Phase 1a
        print(f"\nLoading cleaned eBird data...")
        df = pd.read_csv(INPUT_FILE)
        print(f"  ✓ Loaded {len(df):,} observations from Phase 1a")
        
        # Load and process traits
        guild_lookup = load_and_process_traits(TRAIT_FILE)
        
        # Merge guilds onto observations
        df = merge_guilds_with_observations(df, guild_lookup)
        
        # Validate output
        validation_passed = validate_output(df)
        
        if not validation_passed:
            print("\n⚠ Warning: Validation checks failed but continuing...")
        
        # Save output
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        
        # ================================================================
        # SUMMARY
        # ================================================================
        print("\n" + "=" * 70)
        print("PHASE 1b COMPLETE")
        print("=" * 70)
        print(f"Output: {OUTPUT_FILE}")
        print(f"\nDataset summary:")
        print(f"  Total observations: {len(df):,}")
        print(f"  Unique species: {df['SCIENTIFIC NAME'].nunique():,}")
        print(f"  Unique checklists: {df['SAMPLING EVENT IDENTIFIER'].nunique():,}")
        print(f"  Year range: {df['YEAR'].min()} - {df['YEAR'].max()}")
        
        print(f"\nGuild × Zone cross-tabulation:")
        crosstab = pd.crosstab(df['GUILD'], df['ZONE'], margins=True)
        print(crosstab)
        
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Required file not found")
        print(f"   {e}")
        print(f"\nPlease ensure Phase 1a has been run and created:")
        print(f"  - {INPUT_FILE}")
        print(f"\nAnd that the trait file exists:")
        print(f"  - {TRAIT_FILE}")
        return 1
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())