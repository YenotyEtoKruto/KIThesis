"""
This script is used to write a summary of tiles to tile_index.csv to reduce storage needed
"""
import csv
from random import random, shuffle

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
from pathlib import Path

data_folder = Path("D:/AHN-processed")    # 4 layers (DTM, Hillshade, Slope, SLRM)
label_folder = Path("D:/AHN-mask")    # 1 layer (label)
output_csv = "tile_index.csv"

#tile_size = 128
#stride = 96  # overlap (128 - 96 = 32 px overlap)
tile_size = 256
stride = 192
#min_positive_pixels = 200  # skip tiles with barely any mound
min_positive_pixels = 200
positive_tiles = 0
negative_tiles = 0

with open(output_csv, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["image", "mask", "x", "y", "positive_pixels", "presence"])

    for input_path in data_folder.glob("*_stacked.tif"):
        raster = input_path.stem.replace("_stacked", "")
        raster = raster.replace("cleaned_", "")
        mask_path = label_folder / f"{raster}_mask.tif"
        if not mask_path.exists():
            continue

        print(f"\nProcessing: {raster}")

        with rasterio.open(input_path) as src_img, rasterio.open(mask_path) as src_mask:
            assert src_img.width == src_mask.width
            assert src_img.height == src_mask.height
            assert src_img.transform == src_mask.transform

            width = src_img.width
            height = src_img.height

            for y in tqdm(range(0, height - tile_size + 1, stride)):
                for x in range(0, width - tile_size + 1, stride):
                    window = Window(x,y, tile_size, tile_size)

                    mask_tile = src_mask.read(1, window=window)
                    if np.all(mask_tile == 0):
                        presence = 0
                    else:
                        presence = int(np.sum(mask_tile > 0) >= min_positive_pixels)

                    keep_tile = False
                    if presence:
                        positive_tiles += 1
                        keep_tile = True

                    if keep_tile:
                        pos_pix = int(np.sum(mask_tile > 0))

                        writer.writerow([str(input_path), str(mask_path), x, y, pos_pix, presence])

print("\nFinished!")

print(f"Positive tiles: {positive_tiles}")
print(f"Negative tiles: {negative_tiles}")

total = positive_tiles + negative_tiles

print(f"Total tiles: {total}")
print(f"Positive ratio: {positive_tiles / total:.2%}")
