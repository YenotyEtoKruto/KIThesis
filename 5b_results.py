"""
This script is used to compare performances of the models
"""
import json
import pandas as pd

with open("unet_results_tf.json", "r") as unet_file:
    unet_results = json.load(unet_file)
    unet_results = pd.DataFrame(unet_results)
    unet_results = unet_results.iloc[::-1]
    unet_results["model"] = "UNet"

with open("transunet_rerun_results_tf.json", "r") as transunet_file:
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


results = pd.concat({"UNet": unet_results,
                     "TransUNet": transunet_results,
                     "DeepLab": deeplab_results,
                     "DenseTrans": densetrans_results}, axis=1)

results = results.swaplevel(0, 1, axis=1)
results = results.sort_index(axis=1)
results = results.drop(columns=["combination", "channels", "model"])
results.insert(0, ("info", "combination"), unet_results["combination"])
results.insert(1, ("info", "channels"), unet_results["channels"])

higher_is_better = ['iou', 'dice_score', 'precision', 'recall']
lower_is_better = ['training_time_sec', 'model_size', 'combo_loss']


def highlight_best(row):
    result = [''] * len(row)

    metrics = row.index.get_level_values(0).unique()

    for metric in metrics:

        if metric not in higher_is_better + lower_is_better:
            continue

        cols = [c for c in row.index if c[0] == metric]
        vals = row[cols]

        best = vals.max() if metric in higher_is_better else vals.min()

        for c in cols:
            if pd.notna(row[c]) and row[c] == best:
                idx = row.index.get_loc(c)
                result[idx] = (
                    "background-color: lightgreen; "
                    "font-weight: bold"
                )

    return result


result = results.style.apply(highlight_best, axis=1).set_table_styles([
        {
            "selector": "th",
            "props": [
                ("border", "1px solid black"),
                ("padding", "5px")
            ]
        },
        {
            "selector": "td",
            "props": [
                ("border", "1px solid #ccc"),
                ("padding", "5px")
            ]
        }
    ])
result.to_html("results.html")
