# sentinel/phase0_spatial_audit.py

import rasterio
from pathlib import Path
from collections import defaultdict

# --------------------------------------------------
# CONFIGURATION (EDIT ONLY HERE)
# --------------------------------------------------

RAW_RASTER_ROOT = Path("data/raw/rasters")

VARIABLES = ["NDVI", "NDWI", "VIIRS", "HeightExposure"]
YEARS = list(range(2019, 2026))
ZONES = ["core", "buffer"]

# Spatial Master Grid (SMG)
MASTER_VARIABLE = "NDVI"
MASTER_YEAR = 2025
MASTER_ZONE = "core"

EXPECTED_EPSG = 32643
EXPECTED_RES = (30, 30)  # meters

# --------------------------------------------------
# Logging helpers
# --------------------------------------------------

def fail(msg: str):
    print("\n❌ PHASE 0 FAILED")
    print(msg)
    raise SystemExit(1)

def info(msg: str):
    print(f"ℹ️  {msg}")

def success(msg: str):
    print(f"✅ {msg}")

# --------------------------------------------------
# 1️⃣ Locate Spatial Master Grid
# --------------------------------------------------

def locate_master_raster() -> Path:
    master_path = (
        RAW_RASTER_ROOT / f"{MASTER_VARIABLE}_{MASTER_YEAR}_{MASTER_ZONE}.tif"
    )

    if not master_path.exists():
        fail(f"Spatial Master Grid not found:\n{master_path}")

    success(f"Spatial Master Grid found → {master_path}")
    return master_path

# --------------------------------------------------
# 2️⃣ Read spatial signature
# --------------------------------------------------

def read_spatial_signature(raster_path: Path) -> dict:
    with rasterio.open(raster_path) as ds:
        return {
            "crs": ds.crs.to_epsg(),
            "transform": ds.transform,
            "width": ds.width,
            "height": ds.height,
            "res": tuple(map(abs, ds.res)),
        }

# --------------------------------------------------
# 3️⃣ Enforce no duplicates
# --------------------------------------------------

def enforce_no_duplicates():
    seen = defaultdict(list)

    for var in VARIABLES:
        var_dir = RAW_RASTER_ROOT / var
        if not var_dir.exists():
            fail(f"Missing folder: {var_dir}")

        for tif in var_dir.glob("*.tif"):
            parts = tif.stem.split("_")
            if len(parts) != 3:
                fail(f"Invalid filename format: {tif.name}")

            seen[tuple(parts)].append(tif)

    duplicates = {k: v for k, v in seen.items() if len(v) > 1}

    if duplicates:
        msg = "Duplicate rasters detected:\n"
        for k, files in duplicates.items():
            msg += f"\n{k}:\n"
            for f in files:
                msg += f"  - {f}"
        fail(msg)

    success("No duplicate rasters detected")

# --------------------------------------------------
# 4️⃣ Spatial consistency audit
# --------------------------------------------------

def audit_spatial_consistency(master_sig: dict):
    failures = []

    for var in VARIABLES:
        for year in YEARS:
            for zone in ZONES:
                raster_path = RAW_RASTER_ROOT / var / f"{var}_{year}_{zone}.tif"

                if not raster_path.exists():
                    fail(f"Missing raster: {raster_path}")

                sig = read_spatial_signature(raster_path)

                if sig["crs"] != EXPECTED_EPSG:
                    failures.append((raster_path, "CRS mismatch"))

                if sig["res"] != EXPECTED_RES:
                    failures.append((raster_path, "Resolution mismatch"))

                if (
                    sig["width"] != master_sig["width"]
                    or sig["height"] != master_sig["height"]
                ):
                    failures.append((raster_path, "Grid shape mismatch"))

                if sig["transform"] != master_sig["transform"]:
                    failures.append((raster_path, "Pixel alignment mismatch"))

    if failures:
        msg = "Spatial inconsistencies detected:\n"
        for f, reason in failures:
            msg += f"\n{f} → {reason}"
        fail(msg)

    success("All rasters perfectly aligned to Spatial Master Grid")

# --------------------------------------------------
# 5️⃣ Entry point
# --------------------------------------------------

def run_phase0():
    print("\n🧪 PHASE 0 — SPATIAL AUDIT & ENFORCEMENT\n")

    enforce_no_duplicates()

    master_path = locate_master_raster()
    master_sig = read_spatial_signature(master_path)

    if master_sig["crs"] != EXPECTED_EPSG:
        fail("Master raster CRS is incorrect")

    if master_sig["res"] != EXPECTED_RES:
        fail("Master raster resolution is incorrect")

    audit_spatial_consistency(master_sig)

    print("\n🎉 PHASE 0 PASSED")
    print("All spatial data are synchronized and safe for modeling.\n")

# --------------------------------------------------
# CLI
# --------------------------------------------------

if __name__ == "__main__":
    run_phase0()
