#!/usr/bin/env python3
"""
Mangalavanam Adaptive Sentinel - Phase 2-6 Master Pipeline
==========================================================
Executes occupancy modeling, validation, and conservation analysis pipeline

Author: Senior Geospatial Data Scientist
Date: January 2026
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import argparse


# =============================================================================
# CONFIGURATION
# =============================================================================

GUILDS = ["Wetland", "Forest", "Urban"]

PHASE_2_SCRIPTS = [
    {
        'script': 'sentinel/phase2/raster_to_matrix.py',
        'name': 'Phase 2a: Raster → Occupancy Matrix',
        'required': True
    },
    {
        'script': 'sentinel/phase2/extract_covariates.py',
        'name': 'Phase 2b: Extract Environmental Covariates',
        'required': True
    }
]

PHASE_3_SCRIPTS = [
    {
        'script': 'sentinel/phase3/year_model.py',
        'name': 'Phase 3a: Year-Varying Occupancy Model',
        'per_guild': True,
        'required': True
    },
    {
        'script': 'sentinel/phase3/zone_model.py',
        'name': 'Phase 3b: Zone-Specific Occupancy Model',
        'per_guild': True,
        'required': True
    },
    {
        'script': 'sentinel/phase3/env_model.py',
        'name': 'Phase 3c: Environmental Occupancy Model',
        'per_guild': True,
        'required': True
    },
    {
        'script': 'sentinel/phase3/export_all_psi_maps.py',
        'name': 'Phase 3d: Export ψ Maps',
        'per_guild': False,
        'required': True
    }
]

PHASE_4_SCRIPT = {
    'script': 'sentinel/phase4/functional_analysis.py',
    'name': 'Phase 4: Functional & Conservation Analysis',
    'required': True
}

PHASE_5_SCRIPT = {
    'script': 'sentinel/phase5/validation.py',
    'name': 'Phase 5: Model Validation',
    'required': True
}

PHASE_6_SCRIPT = {
    'script': 'sentinel/phase6/conservation_priority.py',
    'name': 'Phase 6: Conservation Priority Index',
    'required': True
}

# =============================================================================
# UTILITIES
# =============================================================================

def print_banner(text):
    """Print formatted banner."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def run_script(script_path, phase_name, guild=None):
    """
    Run a Python script and capture exit code.
    
    Args:
        script_path: Path to script
        phase_name: Human-readable name
        guild: Optional guild name for per-guild scripts
        
    Returns:
        bool: Success status
    """
    if guild:
        display_name = f"{phase_name} - {guild} Guild"
    else:
        display_name = phase_name
    
    print_banner(f"RUNNING: {display_name}")
    print(f"Script: {script_path}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Build command
        cmd = [sys.executable, str(script_path)]
        
        # Add guild argument if needed
        if guild:
            cmd.extend(['--guild', guild])
        
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False
        )
        
        print(f"\n✅ {display_name} completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {display_name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ {display_name} failed with error: {e}")
        return False

def check_prerequisites():
    """Check if Phase 1 outputs exist and script directories are set up."""
    print_banner("PREREQUISITE CHECK")
    
    print("Checking Phase 1 outputs...")
    required_files = [
        "data/processed/phase1/guild_occupancy_rasters/Wetland_2019_Core.tif",
        "data/processed/phase1/ebird_with_guilds_and_zones.csv",
        "data/raw/rasters/NDVI_2025_core.tif",
        "data/raw/rasters/DIST_EDGE.tif"
    ]
    
    all_exist = True
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (MISSING)")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  Phase 1 outputs missing!")
        print("Please run Phase 1 pipeline first.")
        return False
    
    # Check script directory structure
    print("\nChecking script directories...")
    script_dirs = ['sentinel/phase2', 'sentinel/phase3', 'sentinel/phase4', 'sentinel/phase5', 'sentinel/phase6']
    
    for dir_name in script_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            print(f"  ✅ {dir_name}/ directory exists")
        else:
            print(f"  ❌ {dir_name}/ directory NOT FOUND")
            print(f"     Please create: mkdir {dir_name}")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  Script directories missing!")
        print("Please organize your scripts into phase directories.")
        print("\nExpected structure:")
        print("  phase2/")
        print("    ├── raster_to_matrix.py")
        print("    └── extract_covariates.py")
        print("  phase3/")
        print("    ├── year_model.py")
        print("    ├── zone_model.py")
        print("    ├── env_model.py")
        print("    └── export_psi_maps_corrected.py")
        print("  phase4/")
        print("    └── functional_analysis.py")
        print("  phase5/")
        print("    └── validation.py")
        print("  phase6/")
        print("    └── conservation_priority.py")
        return False
    
    print("\n✅ All Phase 1 outputs and script directories found!")
    return True

# =============================================================================
# PHASE EXECUTORS
# =============================================================================

def run_phase_2():
    """Execute Phase 2 scripts."""
    print_banner("PHASE 2 — OCCUPANCY MATRIX & COVARIATE EXTRACTION")
    
    for phase in PHASE_2_SCRIPTS:
        script_path = Path(phase['script'])
        
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            if phase['required']:
                return False
            continue
        
        success = run_script(script_path, phase['name'])
        
        if not success and phase['required']:
            print(f"\n❌ Phase 2 failed at {phase['name']}")
            return False
    
    return True

def run_phase_3(guilds_to_run):
    """Execute Phase 3 occupancy models."""
    print_banner("PHASE 3 — HIERARCHICAL OCCUPANCY MODELS")
    
    for guild in guilds_to_run:
        print(f"\n{'='*70}")
        print(f"  GUILD: {guild}")
        print(f"{'='*70}")
        
        for phase in PHASE_3_SCRIPTS:
            script_path = Path(phase['script'])
            
            if not script_path.exists():
                print(f"❌ Script not found: {script_path}")
                if phase['required']:
                    return False
                continue
            
            # Run per-guild or global
            if phase['per_guild']:
                success = run_script(script_path, phase['name'], guild=guild)
            else:
                # Only run once for non-guild scripts
                if guild == guilds_to_run[0]:
                    success = run_script(script_path, phase['name'])
                else:
                    continue
            
            if not success and phase['required']:
                print(f"\n❌ Phase 3 failed at {phase['name']} for {guild}")
                return False
    
    return True

def run_phase_4():
    """Execute Phase 4 functional analysis."""
    print_banner("PHASE 4 — FUNCTIONAL & CONSERVATION ANALYSIS")
    
    script_path = Path(PHASE_4_SCRIPT['script'])
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return not PHASE_4_SCRIPT['required']
    
    return run_script(script_path, PHASE_4_SCRIPT['name'])

def run_phase_5():
    """Execute Phase 5 validation."""
    print_banner("PHASE 5 — MODEL VALIDATION")
    
    script_path = Path(PHASE_5_SCRIPT['script'])
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return not PHASE_5_SCRIPT['required']
    
    return run_script(script_path, PHASE_5_SCRIPT['name'])

def run_phase_6():
    """Execute Phase 6 conservation analysis."""
    print_banner("PHASE 6 — CONSERVATION PRIORITY ANALYSIS")
    
    script_path = Path(PHASE_6_SCRIPT['script'])
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return not PHASE_6_SCRIPT['required']
    
    return run_script(script_path, PHASE_6_SCRIPT['name'])

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main pipeline execution."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run Mangalavanam Phase 2-6 analysis pipeline'
    )
    parser.add_argument(
        '--guilds',
        nargs='+',
        choices=GUILDS + ['all'],
        default=['all'],
        help='Guilds to process (default: all)'
    )
    parser.add_argument(
        '--skip-phase-2',
        action='store_true',
        help='Skip Phase 2 (if already completed)'
    )
    parser.add_argument(
        '--skip-functional',
        action='store_true',
        help='Skip Phase 4 functional analysis'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip Phase 5 validation'
    )
    parser.add_argument(
        '--skip-cpi',
        action='store_true',
        help='Skip Phase 6 CPI calculation'
    )
    
    args = parser.parse_args()
    
    # Determine guilds to run
    if 'all' in args.guilds:
        guilds_to_run = GUILDS
    else:
        guilds_to_run = args.guilds
    
    # Header
    print_banner("MANGALAVANAM ADAPTIVE SENTINEL - PHASE 2-6 PIPELINE")
    print("Hierarchical Occupancy Modeling & Conservation Analysis")
    print(f"\nGuilds to process: {', '.join(guilds_to_run)}")
    print(f"Expected runtime: 30-90 minutes (depending on data size)")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Pipeline aborted due to missing prerequisites")
        return 1
    
    # Track results
    results = []
    start_time = datetime.now()
    
    # Phase 2
    if not args.skip_phase_2:
        success = run_phase_2()
        results.append(('Phase 2', success))
        if not success:
            print("\n❌ Pipeline stopped at Phase 2")
            return 1
    else:
        print("\n⊘ Skipping Phase 2 (as requested)")
        results.append(('Phase 2', 'SKIPPED'))
    
    # Phase 3
    success = run_phase_3(guilds_to_run)
    results.append(('Phase 3', success))
    if not success:
        print("\n❌ Pipeline stopped at Phase 3")
        return 1
    
    # Phase 4
    if not args.skip_functional:
        success = run_phase_4()
        results.append(('Phase 4', success))
        # Continue even if Phase 4 fails (it's analytical, not critical)
    else:
        print("\n⊘ Skipping Phase 4 (as requested)")
        results.append(('Phase 4', 'SKIPPED'))
    
    # Phase 5
    if not args.skip_validation:
        success = run_phase_5()
        results.append(('Phase 5', success))
        # Continue even if validation fails
    else:
        print("\n⊘ Skipping Phase 5 (as requested)")
        results.append(('Phase 5', 'SKIPPED'))
    
    # Phase 6
    if not args.skip_cpi:
        success = run_phase_6()
        results.append(('Phase 6', success))
    else:
        print("\n⊘ Skipping Phase 6 (as requested)")
        results.append(('Phase 6', 'SKIPPED'))
    
    # Final summary
    end_time = datetime.now()
    runtime = end_time - start_time
    
    print_banner("PIPELINE EXECUTION SUMMARY")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {runtime}")
    print(f"\nPhase Results:")
    print("-" * 70)
    
    for phase, status in results:
        if status == 'SKIPPED':
            symbol = '⊘'
            status_text = 'SKIPPED'
        elif status:
            symbol = '✅'
            status_text = 'SUCCESS'
        else:
            symbol = '❌'
            status_text = 'FAILED'
        
        print(f"  {symbol} {phase}: {status_text}")
    
    # Check overall success
    failed_count = sum(1 for _, status in results if status is False)
    success_count = sum(1 for _, status in results if status is True)
    
    print("-" * 70)
    print(f"\nTotal: {len(results)} phases")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Skipped: {len(results) - success_count - failed_count}")
    
    if failed_count == 0:
        print("\n" + "=" * 70)
        print("  🎉 PHASE 2-6 PIPELINE COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 70)
        print("\nOutputs ready for conservation dashboard:")
        print("  - Occupancy time series: data/processed/phase2/")
        print("  - ψ maps: data/processed/phase3/psi_maps/")
        print("  - Functional indices: data/processed/phase4/")
        print("  - Validation: data/processed/phase5/")
        print("  - Conservation priorities: data/processed/phase6/")
        print("\nNext steps:")
        print("  1. Review model diagnostics in phase5/")
        print("  2. Review functional diversity in phase4/")
        print("  3. Visualize ψ maps in QGIS or dashboard")
        print("  4. Launch Streamlit dashboard: streamlit run app.py")
        print("=" * 70)
        return 0
    else:
        print("\n❌ Pipeline completed with errors")
        print("Please review error messages above and fix issues.")
        return 1


if __name__ == "__main__":
    exit(main())