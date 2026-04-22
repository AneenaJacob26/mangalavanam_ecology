# sentinel/phase1_alignment_sanity_check.py

import rasterio
from pathlib import Path

# --------------------------------------------------
# Files to compare (edit if needed)
# --------------------------------------------------

REFERENCE_RASTER = Path(
    "data/raw/rasters/HeightExposure_2025_core.tif"
)

TEST_RASTERS = [
    Path("data/processed/phase1/guild_occupancy_rasters/Wetland_2025_Core.tif"),
    Path("data/processed/phase1/guild_occupancy_rasters/Forest_2025_Core.tif"),
    Path("data/processed/phase1/guild_occupancy_rasters/Urban_2025_Core.tif"),
]

# --------------------------------------------------
# Alignment checker
# --------------------------------------------------

def check_alignment(ref_path, test_paths):
    with rasterio.open(ref_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_shape = ref.shape
        ref_res = ref.res

        print("\nREFERENCE GRID")
        print(f" CRS       : {ref_crs}")
        print(f" Transform : {ref_transform}")
        print(f" Shape     : {ref_shape}")
        print(f" Resolution: {ref_res}")

        for path in test_paths:
            print(f"\nCHECKING → {path.name}")

            with rasterio.open(path) as tst:
                assert tst.crs == ref_crs, "❌ CRS mismatch"
                assert tst.transform == ref_transform, "❌ Grid transform mismatch"
                assert tst.shape == ref_shape, "❌ Raster shape mismatch"
                assert tst.res == ref_res, "❌ Pixel resolution mismatch"

                print(" ✅ Alignment OK")

    print("\n🎉 ALL RASTERS PERFECTLY ALIGNED\n")

# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    print("\n🧪 PHASE 1b — ALIGNMENT SANITY CHECK\n")
    check_alignment(REFERENCE_RASTER, TEST_RASTERS)
