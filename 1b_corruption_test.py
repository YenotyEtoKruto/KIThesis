"""
This script is used to check for corruptions in the AHN files
"""
import os
from osgeo import gdal

gdal.UseExceptions()

input_folder = "D:/AHN"

bad_files = []

for file in os.listdir(input_folder):
    if file.endswith(".TIF"):
        path = os.path.join(input_folder, file)

        try:
            ds = gdal.Open(path)
            if ds is None:
                bad_files.append(file)

        except RuntimeError as e:
            print(f"Corrupted: {file}")
            bad_files.append(file)

print("\nBroken files:")
for f in bad_files:
    print(f)