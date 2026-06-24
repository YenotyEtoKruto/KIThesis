"""
This script is used to create hillshade and slope models
"""
import os
from osgeo import gdal

gdal.UseExceptions()

input_folder = "D:/AHN"
output_hillshade = "D:/AHN-hillshade"
output_slope = "D:/AHN-slope"

os.makedirs(output_hillshade, exist_ok=True)
os.makedirs(output_slope, exist_ok=True)

files = os.listdir(input_folder)
print(f"{len(files)} files found")

for file in files:
    if file.endswith(".TIF"):
        in_path = os.path.join(input_folder, file)

        hillshade_out = os.path.join(output_hillshade, f"hs_{file}")
        slope_out = os.path.join(output_slope, f"sl_{file}")

        gdal.DEMProcessing(
            hillshade_out,
            in_path,
            "hillshade",
            azimuth=315,
            altitude=45
        )

        gdal.DEMProcessing(
            slope_out,
            in_path,
            "slope",
            slopeFormat="degree"
        )

        print(f"Processed {file}")
