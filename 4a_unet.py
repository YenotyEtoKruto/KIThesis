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
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt

mixed_precision.set_global_policy('float32')

tile_csv = "tile_index_split.csv"
tile_dir = Path("C:/Users/Friso/AHN-tiles-normalized")

batch_size = 8
learning_rate = 0.001
epochs = 100

output_dir = "models_unet_tf"
results_file = "unet_results_tf.csv"

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


def unet(input_channels):
    input_layer = Input(shape=(256, 256, input_channels))
    # Encoder
    conv1 = Conv2D(16, 3, activation="relu", padding="same", kernel_initializer="he_normal")(input_layer)
    conv1 = Conv2D(16, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(32, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool1)
    conv2 = Conv2D(32, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool2)
    conv3 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool3)
    conv4 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    # bottleneck
    bottleneck = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool4)
    bottleneck = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(bottleneck)

    # decoder
    up1 = Conv2DTranspose(128, 2, strides=2, padding="same")(bottleneck)
    merge1 = concatenate([up1, conv4])
    conv5 = Conv2D(128, 3, activation="relu", padding="same")(merge1)
    conv5 = Conv2D(128, 3, activation="relu", padding="same")(conv5)

    up2 = Conv2DTranspose(64, 2, strides=2, padding="same")(conv5)
    merge2 = concatenate([up2, conv3])
    conv6 = Conv2D(64, 3, activation="relu", padding="same")(merge2)
    conv6 = Conv2D(64, 3, activation="relu", padding="same")(conv6)

    up3 = Conv2DTranspose(32, 2, strides=2, padding="same")(conv6)
    merge3 = concatenate([up3, conv2])
    conv7 = Conv2D(32, 3, activation="relu", padding="same")(merge3)
    conv7 = Conv2D(32, 3, activation="relu", padding="same")(conv7)

    up4 = Conv2DTranspose(16, 2, strides=2, padding="same")(conv7)
    merge4 = concatenate([up4, conv1])
    conv8 = Conv2D(16, 3, activation="relu", padding="same")(merge4)
    conv8 = Conv2D(16, 3, activation="relu", padding="same")(conv8)
    output_layer = Conv2D(1, 1, activation="sigmoid", dtype="float32")(conv8)

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
    print(f"\n Training: {combo}")

    combo_name = "_".join(combo)
    channel_indices = [band_map[c] for c in combo]

    X_train = X_train_full[:, :, :, channel_indices]
    X_val = X_val_full[:, :, :, channel_indices]
    X_test = X_test_full[:, :, :, channel_indices]

    model_path = os.path.join(output_dir, f"unet_{combo_name}.keras")

    start_time = time.time()

    if os.path.exists(model_path):
        print(f"Loading existing model: {model_path}")

        model = load_model(model_path, custom_objects={"combo_loss": combo_loss, "dice_loss": dice_loss,
                                                       "dice_coef": dice_coef, "weighted_bce": weighted_bce})
    else:
        print(f"Training new model: {combo_name}")

        model = unet(len(channel_indices))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=combo_loss, metrics=[dice_coef])
        # model.summary()
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

with open("unet_results_tf.json", "w") as f:
    json.dump(all_results, f, indent=4)
