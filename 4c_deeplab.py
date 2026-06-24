"""
This script is used to train the DeepLab models
"""
import gc
import os
import json
import time
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
import albumentations as A
from tensorflow.keras.layers import (Input, Conv2D, Activation, MaxPooling2D, Add, Concatenate, SeparableConv2D,
                                     BatchNormalization, GlobalAveragePooling2D, Reshape, UpSampling2D)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt

mixed_precision.set_global_policy('mixed_float16')

tile_csv = "tile_index_split.csv"
tile_dir = Path("C:/Users/Friso/AHN-tiles-normalized")

batch_size = 8
learning_rate = 0.001
epochs = 100

output_dir = "models_deeplab_tf"
results_file = "deeplab_results_tf.csv"

os.makedirs(output_dir, exist_ok=True)

# combinations
layers = ["dtm", "hillshade", "slope", "slrm"]
band_map = {k: i for i, k in enumerate(layers)}
all_combinations = []
for r in range(1, len(layers) + 1):
    all_combinations.extend(combinations(layers, r))
all_combinations.reverse()

train_transforms = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)])


def load_dataset(split, transforms=None):
    df = pd.read_csv(tile_csv)
    df = df[df["split"] == split].reset_index(drop=True)

    images = []
    masks = []

    for _, row in df.iterrows():
        image_path = tile_dir / row[
            "split"] / "images" / f"{Path(row['image']).stem}_{int(row['x'])}_{int(row['y'])}.npz"
        mask_path = tile_dir / row["split"] / "masks" / f"{Path(row['image']).stem}_{int(row['x'])}_{int(row['y'])}.npz"

        image = np.load(image_path)["image"]
        mask = np.load(mask_path)["mask"]

        image = np.transpose(image, (1, 2, 0))

        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)

        if transforms:
            augmented = transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        images.append(image.astype(np.float32))
        masks.append(mask.astype(np.float32))

    return np.array(images), np.array(masks)


def ASPP(x):
    x1 = SeparableConv2D(128, 1, padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    x2 = SeparableConv2D(128, 3, dilation_rate=6, padding="same")(x)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)

    x3 = SeparableConv2D(128, 3, dilation_rate=12, padding="same")(x)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)

    x4 = SeparableConv2D(128, 3, dilation_rate=18, padding="same")(x)
    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)

    x5 = GlobalAveragePooling2D()(x)
    x5 = Reshape((1, 1, x5.shape[-1]))(x5)
    x5 = Conv2D(128, 1, padding='same')(x5)
    x5 = UpSampling2D(size=(x.shape[1], x.shape[2]), interpolation="bilinear")(x5)

    y = Concatenate()([x1, x2, x3, x4, x5])
    y = Conv2D(256, 1)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    return y


def entryblock(input, filters, stride=1):
    shortcut = input

    x = Activation("relu")(input)
    x = SeparableConv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    x = Activation("relu")(x)
    x = SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    if stride != 1 or input.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(input)
        shortcut = BatchNormalization()(shortcut)

    output = Add()([x, shortcut])

    return output


def entry(input):
    x = Conv2D(32, 3, strides=2, padding="same", use_bias=False)(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = entryblock(x, 128, stride=2)
    low_level = x

    x = entryblock(x, 256, stride=2)
    x = entryblock(x, 728, stride=2)

    return x, low_level


def middle(input):
    x = Activation("relu")(input)
    x = SeparableConv2D(728, 3, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    x = Activation("relu")(x)
    x = SeparableConv2D(728, 3, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    x = Activation("relu")(x)
    x = SeparableConv2D(728, 3, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    output = Add()([x, input])
    return output


def end(input):
    x = Activation("relu")(input)
    x = SeparableConv2D(728, 3, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    x = Activation("relu")(x)
    x = SeparableConv2D(1024, 3, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(3, strides=1, padding="same")(x)
    y = Conv2D(1024, 1, strides=1, padding="same", use_bias=False)(input)
    y = BatchNormalization()(y)
    x = Add()([x, y])

    x = Activation("relu")(x)
    x = SeparableConv2D(1536, 3, dilation_rate=2, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    x = Activation("relu")(x)
    x = SeparableConv2D(1536, 3, dilation_rate=2, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    x = Activation("relu")(x)
    x = SeparableConv2D(2048, 3, dilation_rate=2, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    output = Activation("relu")(x)

    return output


def xception65(input):
    x, low_level = entry(input)
    for _ in range(8):
        x = middle(x)
    output = end(x)
    return output, low_level


def deeplabv3plus(input_channels):

    input_layer = Input(shape=(256, 256, input_channels))

    # Xception-65 backbone
    high_level, low_level = xception65(input_layer)

    # ASPP
    x = ASPP(high_level)

    # Decoder
    x = UpSampling2D(size=(4, 4), interpolation="bilinear")(x)

    low_level = Conv2D(48, 1, use_bias=False)(low_level)
    low_level = BatchNormalization()(low_level)
    low_level = Activation("relu")(low_level)

    x = Concatenate()([x, low_level])

    x = SeparableConv2D(256, 3, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = SeparableConv2D(256, 3, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D(size=(4, 4), interpolation="bilinear")(x)

    output_layer = Conv2D(1, 1, activation="sigmoid", dtype="float32")(x)
    return Model(input_layer, output_layer)


def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    denominator = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth

    return (2.0 * intersection + smooth) / denominator


def dice_loss(y_true, y_pred, smooth=1e-6):
    return 1.0 - dice_coef(y_true, y_pred)


def weighted_bce(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    pos_weight = 10.0

    loss = -(pos_weight * y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    return tf.reduce_mean(loss)


def combo_loss(y_true, y_pred):
    return 0.5 * dice_loss(y_true, y_pred) + 0.5 * weighted_bce(y_true, y_pred)


X_train_full, y_train = load_dataset("train", train_transforms)
X_val_full, y_val = load_dataset("val")
X_test_full, y_test = load_dataset("test")


def train_combination(combo):
    print(f"\n Training: {combo}")

    combo_name = "_".join(combo)
    channel_indices = [band_map[c] for c in combo]

    X_train = X_train_full[:, :, :, channel_indices]
    X_val = X_val_full[:, :, :, channel_indices]
    X_test = X_test_full[:, :, :, channel_indices]

    model_path = os.path.join(output_dir, f"deeplab_{combo_name}.keras")

    start_time = time.time()

    if os.path.exists(model_path):
        print(f"Loading existing model: {model_path}")

        model = load_model(model_path, custom_objects={"combo_loss": combo_loss, "dice_loss": dice_loss,
                                                       "dice_coef": dice_coef, "weighted_bce": weighted_bce})
    else:
        print(f"Training new model: {combo_name}")

        model = deeplabv3plus(len(channel_indices))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=combo_loss, metrics=[dice_coef])
        model.summary()
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
                            verbose=2)
        model.save(model_path)
        print(f"Saved model to {model_path}")

        tf.keras.backend.clear_session()
        gc.collect()

        model = load_model(model_path, custom_objects={
            "combo_loss": combo_loss,
            "dice_loss": dice_loss,
            "dice_coef": dice_coef,
            "weighted_bce": weighted_bce})

    preds = model.predict(X_test, verbose=2)

    print(
        "Prediction stats:",
        preds.min(),
        preds.max(),
        preds.mean()
    )

    preds_binary = (preds > 0.55).astype(np.uint8)
    y_true = y_test.astype(np.uint8)
    iou = jaccard_score(y_true.flatten(), preds_binary.flatten())
    precision = precision_score(y_true.flatten(), preds_binary.flatten(), zero_division=0)
    recall = recall_score(y_true.flatten(), preds_binary.flatten(), zero_division=0)
    dice = f1_score(y_true.flatten(), preds_binary.flatten(), zero_division=0)
    combo_l = combo_loss(y_true.astype(np.float32), preds.astype(np.float32)).numpy()
    elapsed_time = time.time() - start_time
    model_size = os.path.getsize(model_path) / (1024 * 1024)

    """
    for i in range(5):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(X_test[i][:, :, 0], cmap="terrain")
        ax[0].set_title("Input")

        ax[1].imshow(y_test[i][:, :, 0], cmap="gray")
        ax[1].set_title("Ground Truth")

        ax[2].imshow(preds_binary[i][:, :, 0], cmap="gray")
        ax[2].set_title(f"Prediction ({combo_name})")
        plt.show()
    """

    return {
        "combination": combo_name,
        "channels": len(channel_indices),
        "iou": float(iou),
        "dice_score": float(dice),
        "combo_loss": float(combo_l),
        "precision": float(precision),
        "recall": float(recall),
        "training_time_sec": float(elapsed_time),
        "model_size": float(model_size)
    }


all_results = []

for combo in all_combinations:

    result = train_combination(combo)
    all_results.append(result)
    pd.DataFrame(all_results).to_csv(results_file, index=False)

    tf.keras.backend.clear_session()
    gc.collect()

with open("deeplab_results_tf.json", "w") as f:
    json.dump(all_results, f, indent=4)