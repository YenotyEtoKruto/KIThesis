"""
This script is used to train the DenseTrans models
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
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Conv2DTranspose, MultiHeadAttention,
                                     LayerNormalization, Activation, Concatenate, DepthwiseConv2D, Reshape)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
from swin_function import SwinTransformerBlock, WindowAttention, StochasticDepth
import matplotlib.pyplot as plt


mixed_precision.set_global_policy('float32')

tile_csv = "tile_index_split.csv"
tile_dir = Path("C:/Users/Friso/AHN-tiles-normalized")

batch_size = 8
learning_rate = 0.001
epochs = 100

output_dir = "models_densetrans_tf"
results_file = "densetrans_results_tf.csv"

os.makedirs(output_dir, exist_ok=True)

# 15 combinations
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


def densetrans(input_channels):
    # Input
    input_layer = Input(shape=(256, 256, input_channels))

    # E1
    conv1 = DepthwiseConv2D(3, padding="same")(input_layer)
    conv1 = Conv2D(16, 1, padding="same")(conv1)
    conv1 = LayerNormalization(epsilon=1e-6)(conv1)
    conv1 = Activation("relu")(conv1)
    pool1 = MaxPooling2D(2, 2, padding="same")(conv1)

    # E2
    conv2 = DepthwiseConv2D(3, padding="same")(pool1)
    conv2 = Conv2D(32, 1, padding="same")(conv2)
    conv2 = LayerNormalization(epsilon=1e-6)(conv2)
    conv2 = Activation("relu")(conv2)
    pool2 = MaxPooling2D(2, 2, padding="same")(conv2)

    # E3
    conv3 = DepthwiseConv2D(3, padding="same")(pool2)
    conv3 = Conv2D(64, 1, padding="same")(conv3)
    conv3 = LayerNormalization(epsilon=1e-6)(conv3)
    conv3 = Activation("relu")(conv3)
    pool3 = MaxPooling2D(2, 2, padding="same")(conv3)

    # E4
    conv4 = DepthwiseConv2D(3, padding="same")(pool3)
    conv4 = Conv2D(128, 1, padding="same")(conv4)
    conv4 = LayerNormalization(epsilon=1e-6)(conv4)
    conv4 = Activation("relu")(conv4)
    pool4 = MaxPooling2D(2, 2, padding="same")(conv4)

    # Swin
    x = Conv2D(128, 2, strides=2)(conv4)
    print(x.shape)
    x = Reshape((16 * 16, 128))(x)
    print(x.shape)
    x = SwinTransformerBlock(128, (16, 16), 4, window_size=4)(x)
    x = SwinTransformerBlock(128, (16, 16), 4, window_size=4, shift_size=2)(x)
    x = SwinTransformerBlock(128, (16, 16), 4, window_size=4)(x)
    x = SwinTransformerBlock(128, (16, 16), 4, window_size=4, shift_size=2)(x)
    x = Reshape((16, 16, 128))(x)
    x = Conv2D(128, 1)(x)

    # Bottleneck
    bottleneck = DepthwiseConv2D(3, padding="same")(pool4)
    bottleneck = Conv2D(256, 1, padding="same")(bottleneck)
    bottleneck = LayerNormalization(epsilon=1e-6)(bottleneck)
    bottleneck = Activation("relu")(bottleneck)
    merge1 = Concatenate()([bottleneck, x])
    merge1 = Conv2D(256, 1, padding="same")(merge1)

    # D1
    up1 = Conv2DTranspose(128, 3, 2, padding="same")(merge1)
    up1 = Concatenate()([up1, conv4])
    up1 = DepthwiseConv2D(3, padding="same")(up1)
    up1 = Conv2D(128, 1, padding="same")(up1)
    up1 = LayerNormalization()(up1)
    up1 = Activation("relu")(up1)

    # D2
    up2 = Conv2DTranspose(64, 3, 2, padding="same")(up1)
    up2 = Concatenate()([up2, conv3])
    up2 = DepthwiseConv2D(3, padding="same")(up2)
    up2 = Conv2D(128, 1, padding="same")(up2)
    up2 = LayerNormalization()(up2)
    up2 = Activation("relu")(up2)

    # D3
    up3 = Conv2DTranspose(32, 3, 2, padding="same")(up2)
    up3 = Concatenate()([up3, conv2])
    up3 = DepthwiseConv2D(3, padding="same")(up3)
    up3 = Conv2D(128, 1, padding="same")(up3)
    up3 = LayerNormalization()(up3)
    up3 = Activation("relu")(up3)

    # D4
    up4 = Conv2DTranspose(16, 3, 2, padding="same")(up3)
    up4 = Concatenate()([up4, conv1])
    up4 = DepthwiseConv2D(3, padding="same")(up4)
    up4 = Conv2D(128, 1, padding="same")(up4)
    up4 = LayerNormalization()(up4)
    up4 = Activation("relu")(up4)

    # Output
    output_layer = Conv2D(1, 1, padding="same", activation="sigmoid", dtype="float32")(up4)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model


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
    print(f"\nTraining: {combo}")

    combo_name = "_".join(combo)
    channel_indices = [band_map[c] for c in combo]

    X_train = X_train_full[:, :, :, channel_indices]
    X_val = X_val_full[:, :, :, channel_indices]
    X_test = X_test_full[:, :, :, channel_indices]

    model_path = os.path.join(output_dir, f"densetrans_{combo_name}.keras")

    start_time = time.time()

    if os.path.exists(model_path):
        print(f"Loading existing model: {model_path}")

        model = load_model(model_path, custom_objects={
            "combo_loss": combo_loss,
            "dice_loss": dice_loss,
            "dice_coef": dice_coef,
            "weighted_bce": weighted_bce,
            "SwinTransformerBlock": SwinTransformerBlock,
            "WindowAttention": WindowAttention,
            "StochasticDepth": StochasticDepth})
    else:
        print(f"Training new model: {combo_name}")

        model = densetrans(len(channel_indices))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=combo_loss, metrics=[dice_coef])
        model.summary()
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=2)
        model.save(model_path)
        print(f"Saved model to {model_path}")

        tf.keras.backend.clear_session()
        gc.collect()

        model = load_model(model_path, custom_objects={
            "combo_loss": combo_loss,
            "dice_loss": dice_loss,
            "dice_coef": dice_coef,
            "weighted_bce": weighted_bce,
            "SwinTransformerBlock": SwinTransformerBlock,
            "WindowAttention": WindowAttention,
            "StochasticDepth": StochasticDepth})

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
    n_examples = min(5, len(X_test))

    for i in range(n_examples):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(X_test[i][:, :, 0], cmap="terrain")
        ax[0].set_title("Input")

        ax[1].imshow(y_test[i][:, :, 0], cmap="gray")
        ax[1].set_title("Ground Truth")

        ax[2].imshow(preds_binary[i][:, :, 0], cmap="gray")
        ax[2].set_title(f"Prediction ({combo_name})")

        plt.tight_layout()
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

with open("densetrans_results_tf.json", "w") as f:
    json.dump(all_results, f, indent=4)
