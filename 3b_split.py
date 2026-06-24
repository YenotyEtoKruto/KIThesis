"""
This script is used to define what tiles are to be used for what part of the training
"""
import pandas as pd
from pathlib import Path

df = pd.read_csv("tile_index.csv")

df["raster_id"] = df["image"].apply(lambda x: Path(x).stem.replace("_stacked", ""))

stats = (df.groupby("raster_id").agg(total_tiles=("presence", "count"), positive_tiles=("presence", "sum")).reset_index())
stats = stats.sort_values(
    by="total_tiles",
    ascending=False
)

total_tiles = stats["total_tiles"].sum()

train_target = total_tiles * 0.7
val_target = total_tiles * 0.2
test_target = total_tiles * 0.1

splits = {
    "train": [],
    "val": [],
    "test": []
}

split_counts = {
    "train": 0,
    "val": 0,
    "test": 0
}

for _, row in stats.iterrows():

    raster = row["raster_id"]
    count = row["total_tiles"]

    train_gap = train_target - split_counts["train"]
    val_gap = val_target - split_counts["val"]
    test_gap = test_target - split_counts["test"]

    train_gap = train_gap/7
    val_gap = val_gap/2

    gaps = {
        "train": train_gap,
        "val": val_gap,
        "test": test_gap
    }

    best_split = max(gaps, key=gaps.get)
    splits[best_split].append(raster)
    split_counts[best_split] += count


def assign_split(raster):
    for split, rasters in splits.items():
        if raster in rasters:
            return split


df["split"] = df["raster_id"].apply(assign_split)
df.to_csv("tile_index_split.csv", index=False)

print("\nTile counts:")
print(df["split"].value_counts())
print("\nPositive tiles:")
print(df[df["presence"] == 1]["split"].value_counts())
print(df.groupby(["split", "raster_id"])["presence"].sum())
