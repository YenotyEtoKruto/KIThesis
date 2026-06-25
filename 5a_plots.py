"""
This script is used to compare computational costs
"""
import matplotlib.pyplot as plt
import pandas as pd
import json


with open("unet_results_tf.json", "r") as unet_file:
    unet_results = json.load(unet_file)
    unet_results = pd.DataFrame(unet_results)
    unet_results = unet_results.iloc[::-1]
    unet_results["model"] = "UNet"

with open("transunet_results_tf.json", "r") as transunet_file:
    transunet_results = json.load(transunet_file)
    transunet_results = pd.DataFrame(transunet_results)
    transunet_results = transunet_results.iloc[::-1]
    transunet_results["model"] = "TransUNet"

with open("deeplab_results_tf.json", "r") as deeplab_results_file:
    deeplab_results = json.load(deeplab_results_file)
    deeplab_results = pd.DataFrame(deeplab_results)
    deeplab_results = deeplab_results.iloc[::-1]
    deeplab_results["model"] = "DeepLab"

with open("densetrans_results_tf.json", "r") as densetrans_file:
    densetrans_results = json.load(densetrans_file)
    densetrans_results = pd.DataFrame(densetrans_results)
    densetrans_results = densetrans_results.iloc[::-1]
    densetrans_results["model"] = "DenseTrans"


results = pd.concat([unet_results, transunet_results, deeplab_results, densetrans_results], ignore_index=True)
print(results.columns)

metrics = ["iou", "dice_score", "precision", "recall"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

models = results["model"].unique()

for ax, metric in zip(axes.flat, metrics):

    data = [results.loc[results["model"] == model, metric] for model in models]

    ax.boxplot(data, labels=models)

    ax.set_title(metric.replace("_", " ").title())
    if metric == "iou":
        ax.set_title("IoU")
    elif metric == "dice_score":
        ax.set_title("DSC")
    ax.set_ylabel("Score")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

eff_metrics = ["training_time_sec", "model_size"]

summary = results.groupby("model")[eff_metrics].agg(["mean", "std"])

summary.columns = [
    f"{col}_{stat}" for col, stat in summary.columns
]

summary = summary.reset_index()

summary["Training Time (s)"] = (
    summary["training_time_sec_mean"].round(2).astype(str)
    + " ± " +
    summary["training_time_sec_std"].round(2).astype(str)
)

summary["Model Size (MB)"] = (
    summary["model_size_mean"].round(2).astype(str)
    + " ± " +
    summary["model_size_std"].round(2).astype(str)
)

final_table = summary[[
    "model",
    "Training Time (s)",
    "Model Size (MB)"
]]

print(final_table.to_string(index=False))
