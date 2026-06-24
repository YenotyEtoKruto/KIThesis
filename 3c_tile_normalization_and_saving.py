"""
This script is used to normalize the tiles and save them in a compressed format
"""
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

csv_path = "tile_index_split.csv"
tile_size = 256
output_dir = Path("C:/Users/Friso/AHN-tiles-normalized")
df = pd.read_csv(csv_path)

for split in ["train", "val", "test"]:
    (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / split / "masks").mkdir(parents=True, exist_ok=True)


def sanitize_tile(img, clip_abs=1e5):
    """
    Cleans nodata/extreme values.
    img shape: (C, H, W)
    """
    img = img.astype(np.float32)
    img[np.isnan(img)] = 0
    img[np.isinf(img)] = 0
    img[img == -9999] = 0
    img[img == 9999] = 0
    img[np.abs(img) > clip_abs] = 0
    return img


print("\nComputing global training statistics...")

train_df = df[df["split"] == "train"]

n_channels = None
channel_sum = None
channel_sq_sum = None
num_pixels = 0

for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
    x, y = int(row["x"]), int(row["y"])
    image_path = Path(row["image"])

    with rasterio.open(image_path) as src:
        window = rasterio.windows.Window(x, y, tile_size, tile_size)
        img_tile = src.read(window=window).astype(np.float32)

    img_tile = sanitize_tile(img_tile)

    if n_channels is None:
        n_channels = img_tile.shape[0]
        channel_sum = np.zeros(n_channels, dtype=np.float64)
        channel_sq_sum = np.zeros(n_channels, dtype=np.float64)

    channel_sum += img_tile.sum(axis=(1, 2))
    channel_sq_sum += (img_tile ** 2).sum(axis=(1, 2))
    num_pixels += img_tile.shape[1] * img_tile.shape[2]

means = channel_sum / num_pixels
stds = np.sqrt((channel_sq_sum / num_pixels) - (means ** 2))

print("\nGLOBAL MEANS:")
print(means)

print("\nGLOBAL STDS:")
print(stds)

stds = np.maximum(stds, 1e-6)

print("\nNormalizing and saving tiles...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    split = row["split"]
    x, y = int(row["x"]), int(row["y"])

    image_path = Path(row["image"])
    mask_path = Path(row["mask"])

    tile_id = f"{image_path.stem}_{x}_{y}"

    with rasterio.open(image_path) as src:
        window = rasterio.windows.Window(x, y, tile_size, tile_size)
        img_tile = src.read(window=window).astype(np.float32)

    with rasterio.open(mask_path) as src:
        mask_tile = src.read(1, window=window).astype(np.uint8)

    mask_tile = (mask_tile > 0).astype(np.uint8)

    img_tile = sanitize_tile(img_tile)
    img_tile = (img_tile - means[:, None, None]) / stds[:, None, None]

    image_out = (output_dir / split / "images" / f"{tile_id}.npz")
    mask_out = (output_dir / split / "masks" / f"{tile_id}.npz")

    np.savez_compressed(image_out, image=img_tile.astype(np.float32))
    np.savez_compressed(mask_out, mask=mask_tile.astype(np.uint8))
