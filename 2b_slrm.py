"""
This script is used to create a simplified local relief model
"""
from rvt.vis import slrm
import numpy as np
from osgeo import gdal
import os

gdal.UseExceptions()

input_folder = "D:/AHN"
output_slrm = "D:/AHN-slrm"

os.makedirs(output_slrm, exist_ok=True)

files = os.listdir(input_folder)
print(f"{len(files)} files found")

for file in files:
    if not file.endswith(".TIF"):
        continue

    in_path = os.path.join(input_folder, file)
    slrm_out = os.path.join(output_slrm, f"slrm_{file}")

    if os.path.exists(slrm_out):
        print(f"Skipping {file} (already processed)")
        continue

    try:
        ds = gdal.Open(in_path)
        if ds is None:
            print(f"Cannot open: {file}")
            continue

        band = ds.GetRasterBand(1)
        dtm = band.ReadAsArray().astype(np.float32)
        nodata = band.GetNoDataValue()
        if nodata is not None:
            dtm[dtm == nodata] = 0

        slrm_result = slrm(
            dem=dtm,
            radius_cell=20,
            ve_factor=1
        )

        slrm_result = np.nan_to_num(
            slrm_result,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(
            slrm_out,
            ds.RasterXSize,
            ds.RasterYSize,
            1,
            gdal.GDT_Float32
        )

        out_ds.SetGeoTransform(ds.GetGeoTransform())
        out_ds.SetProjection(ds.GetProjection())

        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(slrm_result)

        out_ds.FlushCache()

        print(f"Processed {file}")

    except Exception as e:
        print(f"Error with {file}: {e}")
