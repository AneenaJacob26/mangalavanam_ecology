#!/usr/bin/env python3
"""
Mangalavanam Adaptive Sentinel - Phase 1 Master Pipeline
Executes all Phase 1 scripts in correct order with progress tracking

Author: Senior Geospatial Data Scientist
Date: January 2026
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


def print_banner(text):
    """Print a formatted banner."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def run_script(script_path, phase_name):
    """
    Run a Python script and capture its exit code.
    
    Args:
        script_path: Path to Python script
        phase_name: Human-readable phase name
        
    Returns:
        bool: True if successful, False if failed
    """
    print_banner(f"RUNNING: {phase_name}")
    print(f"Script: {script_path}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False  # Show output in real-time
        )
        
        print(f"\n✅ {phase_name} completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {phase_name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ {phase_name} failed with error: {e}")
        return False


def check_prerequisites():
    """Check if required input files exist."""
    print_banner("PREREQUISITE CHECK")
    
    required_files = [
        "data/raw/occurrences/EBD/ebd_occurrence.csv",
        "data/raw/occurrences/EBD/ebd_metadata.csv",
        "data/raw/bird_functional_traits.csv",
        "data/raw/shapes/Mangalavanam_Core.geojson",
        "data/raw/shapes/Mangalavanam_Buffer.geojson",
        "data/raw/rasters/NDVI_2025_core.tif"
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
        print("\n⚠️  Some required files are missing!")
        print("Please ensure all input data is in place before running the pipeline.")
        return False
    
    print("\n✅ All prerequisite files found!")
    return True


def main():
    """Main pipeline execution."""
    
    print_banner("MANGALAVANAM ADAPTIVE SENTINEL - PHASE 1 PIPELINE")
    print("This will run all Phase 1 scripts in the correct order.")
    print("Expected runtime: 5-15 minutes depending on data size.")
    print("\nPhases:")
    print("  1a. Data Cleaning & Zone Assignment")
    print("  1b. Guild Assignment")
    print("  1c. Occupancy Rasterization")
    print("  1d. Spatial Proxies (independent)")
    print("  1e. Validation")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Pipeline aborted due to missing prerequisites")
        return 1
    
    # Get user confirmation
    print("\n" + "-" * 70)
    response = input("Proceed with pipeline execution? (y/n): ").lower()
    if response != 'y':
        print("Pipeline cancelled by user.")
        return 0
    
    # Define pipeline phases
    phases = [
        {
            'script': 'data_cleaning_with_zones.py',
            'name': 'Phase 1a: Data Cleaning & Zone Assignment',
            'optional': False
        },
        {
            'script': 'guild_assignment.py',
            'name': 'Phase 1b: Guild Assignment',
            'optional': False
        },
        {
            'script': 'occupancy_rasterization.py',
            'name': 'Phase 1c: Occupancy Rasterization',
            'optional': False
        },
        {
            'script': 'spatial_proxies.py',
            'name': 'Phase 1d: Spatial Proxies',
            'optional': False  # Set to True if you want to skip this
        },
        {
            'script': 'validation.py',
            'name': 'Phase 1e: Validation',
            'optional': False
        }
    ]
    
    # Track results
    results = []
    start_time = datetime.now()
    
    # Execute phases
    for i, phase in enumerate(phases, 1):
        script_path = Path(__file__).parent / phase['script']
        
        if not script_path.exists():
            print(f"\n❌ Script not found: {script_path}")
            if phase['optional']:
                print(f"⚠️  Skipping optional phase: {phase['name']}")
                results.append({'phase': phase['name'], 'status': 'SKIPPED'})
                continue
            else:
                print(f"❌ Required phase cannot be executed!")
                return 1
        
        success = run_script(script_path, phase['name'])
        results.append({
            'phase': phase['name'],
            'status': 'SUCCESS' if success else 'FAILED'
        })
        
        if not success and not phase['optional']:
            print(f"\n❌ Pipeline stopped at {phase['name']}")
            print("Fix the error and re-run the pipeline.")
            return 1
    
    # Calculate runtime
    end_time = datetime.now()
    runtime = end_time - start_time
    
    # Final summary
    print_banner("PIPELINE EXECUTION SUMMARY")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {runtime}")
    print(f"\nPhase Results:")
    print("-" * 70)
    
    for result in results:
        status_symbol = {
            'SUCCESS': '✅',
            'FAILED': '❌',
            'SKIPPED': '⊘'
        }[result['status']]
        print(f"  {status_symbol} {result['phase']}: {result['status']}")
    
    # Check if all succeeded
    failed_count = sum(1 for r in results if r['status'] == 'FAILED')
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    
    print("-" * 70)
    print(f"\nTotal: {len(results)} phases")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Skipped: {len(results) - success_count - failed_count}")
    
    if failed_count == 0:
        print("\n" + "=" * 70)
        print("  🎉 PHASE 1 PIPELINE COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 70)
        print("\nOutputs are ready for Phase 2 analysis:")
        print("  - data/processed/phase1/ebird_with_guilds_and_zones.csv")
        print("  - data/processed/phase1/guild_occupancy_rasters/*.tif")
        print("  - data/raw/rasters/DIST_EDGE.tif")
        print("  - data/raw/rasters/DIST_DRAINAGE.tif")
        print("\nNext steps:")
        print("  1. Review validation reports in data/processed/phase1/validation/")
        print("  2. Proceed to Phase 2 (SDM fitting)")
        print("=" * 70)
        return 0
    else:
        print("\n❌ Pipeline completed with errors")
        print("Please review the error messages above and fix issues.")
        return 1


if __name__ == "__main__":
    exit(main())