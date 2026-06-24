"""
This script is used to clean the DTM, to decrease problems in later derivations
"""
import os
import numpy as np
from osgeo import gdal

input_folder = r"D:\AHN"
clip_abs = 1e5

for file_name in os.listdir(input_folder):
    if not file_name.lower().endswith(".tif"):
        continue

    input_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(input_folder, f"cleaned_{file_name}")

    ds = gdal.Open(input_path)
    if ds is None:
        print(f"Cannot open {file_name}")
        continue

    band = ds.GetRasterBand(1)
    dtm = band.ReadAsArray().astype(np.float32)

    nodata = band.GetNoDataValue()
    if nodata is not None:
        dtm[dtm == nodata] = np.nan

    dtm[np.abs(dtm) > clip_abs] = np.nan

    dtm_cleaned = np.nan_to_num(dtm, nan=0.0)

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_path, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(dtm_cleaned)
    out_band.SetNoDataValue(0)
    out_ds.FlushCache()
    out_ds = None
    ds = None

    os.remove(input_path)
    print(f"Cleaned and replaced {file_name} with {output_path}")