"""
This script is used to stack the DTM, hillshade, slope and SLRM on each other to reduce storage space
"""
from pathlib import Path
import rasterio
import numpy as np


def sanitize_band(band, clip_abs=1e5, nodata_values=(-9999, 9999)):
    band = band.astype(np.float32)
    band = np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0)
    for nd in nodata_values:
        band[band == nd] = 0
    band[np.abs(band) > clip_abs] = 0
    return band


base_dir = Path("D:/AHN")
hillshade_dir = Path("D:/AHN-hillshade")
slope_dir = Path("D:/AHN-slope")
slrm_dir = Path("D:/AHN-slrm")
output_dir = Path("D:/AHN-processed")

for base_file in base_dir.glob("*.tif"):
    name = base_file.stem

    hs_file = hillshade_dir / f"hs_{name}.tif"
    sl_file = slope_dir / f"sl_{name}.tif"
    slrm_file = slrm_dir / f"slrm_{name}.tif"

    out_file = output_dir / f"{name}_stacked.tif"

    if out_file.exists():
        print(f"Stack already exists for {name}")

        for f in [hs_file, sl_file, slrm_file]:
            if f.exists():
                f.unlink()
                print(f"Deleted {f.name}")

        continue

    if not (hs_file.exists() and sl_file.exists() and slrm_file.exists()):
        print(f"Missing companion files for {name}")
        continue

    print(f"Processing {name}")

    try:
        with rasterio.open(base_file) as src1, \
             rasterio.open(hs_file) as src2, \
             rasterio.open(sl_file) as src3, \
             rasterio.open(slrm_file) as src4:

            band1 = sanitize_band(src1.read(1))
            band2 = sanitize_band(src2.read(1))
            band3 = sanitize_band(src3.read(1))
            band4 = sanitize_band(src4.read(1))

            meta = src1.meta.copy()
            meta.update({
                "count": 4
            })

            with rasterio.open(out_file, "w", **meta) as dst:
                dst.write(band1, 1)
                dst.write(band2, 2)
                dst.write(band3, 3)
                dst.write(band4, 4)

                dst.set_band_description(1, "AHN")
                dst.set_band_description(2, "Hillshade")
                dst.set_band_description(3, "Slope")
                dst.set_band_description(4, "SLRM")

        if out_file.exists():
            for f in [hs_file, sl_file, slrm_file]:
                if f.exists():
                    f.unlink()

            print(f"Deleted intermediate files for {name}")

    except Exception as e:
        print(f"Failed processing {name}: {e}")

print("Done.")
