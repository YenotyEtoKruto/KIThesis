"""
This script is used to train the TransUNet models
"""
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
                                     Dense, LayerNormalization, Dropout, Reshape, BatchNormalization,
                                     Activation, Add, Concatenate)
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

output_dir = "models_transunet_tf"
results_file = "transunet_results_tf.csv"

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


class PositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_patches, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.pos_emb = self.add_weight(
            shape=(1, num_patches, embed_dim),
            initializer="random_normal",
            trainable=True,
            name="pos_emb"
        )
        self.num_patches = num_patches
        self.embed_dim = embed_dim

    def call(self, x):
        return x + self.pos_emb

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "embed_dim": self.embed_dim})
        return config


def bottleneck_block(input, filter, stride=1):
    x = Conv2D(filter, 1, strides=stride, padding="same")(input)
    x = BatchNormalization(dtype="float32")(x)
    x = Activation("relu")(x)

    x = Conv2D(filter, 3, strides=1, padding="same")(x)
    x = BatchNormalization(dtype="float32")(x)
    x = Activation("relu")(x)

    x = Conv2D(4*filter, 1, strides=1, padding="same")(x)
    x = BatchNormalization(dtype="float32")(x)

    if stride != 1 or input.shape[-1] != 4 * filter:
        shortcut = Conv2D(4 * filter, 1, strides=stride, padding="same", use_bias=False)(input)
        shortcut = BatchNormalization(dtype="float32")(shortcut)
    else:
        shortcut = input

    out = Add()([x, shortcut])
    out = Activation("relu")(out)

    return out


def transformer_block(x, num_heads, mlp_dim, dropout=0.1):
    x1 = LayerNormalization(epsilon=1e-6)(x)
    x1 = MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1] // num_heads, dropout=dropout)(x1, x1)
    x1 = Add()([x1, x])

    x2 = LayerNormalization(epsilon=1e-6)(x1)
    x2 = Dense(mlp_dim, activation="gelu")(x2)
    x2 = Dropout(dropout)(x2)
    x2 = Dense(x1.shape[-1])(x2)
    x2 = Dropout(dropout)(x2)
    x2 = Add()([x1, x2])

    return x2


def new_transunet(input_channels):
    input_layer = Input(shape=(256, 256, input_channels))

    # Encoder (ResNet50-style)
    conv1 = Conv2D(64, 7, 2, padding="same")(input_layer)
    conv1 = BatchNormalization(dtype="float32")(conv1)
    conv1 = Activation("relu")(conv1)

    conv2_1 = MaxPooling2D(3, 2, padding="same")(conv1)
    conv2_2 = bottleneck_block(conv2_1, 64)
    conv2_3 = bottleneck_block(conv2_2, 64)
    conv2_4 = bottleneck_block(conv2_3, 64)
    skip1 = conv2_4

    conv3_1 = bottleneck_block(conv2_4, 128, 2)
    conv3_2 = bottleneck_block(conv3_1, 128)
    conv3_3 = bottleneck_block(conv3_2, 128)
    conv3_4 = bottleneck_block(conv3_3, 128)
    skip2 = conv3_4

    conv4_1 = bottleneck_block(conv3_4, 256, 2)
    conv4_2 = bottleneck_block(conv4_1, 256)
    conv4_3 = bottleneck_block(conv4_2, 256)
    conv4_4 = bottleneck_block(conv4_3, 256)
    conv4_5 = bottleneck_block(conv4_4, 256)
    conv4_6 = bottleneck_block(conv4_5, 256)

    # Encoder (ViT)
    embedding = Conv2D(256, 1, padding="same")(conv4_6)
    h, w, = embedding.shape[1], embedding.shape[2]
    embedding = Reshape((h * w, 256))(embedding)

    x = PositionEmbedding(h * w, 256)(embedding)
    for _ in range(4):
        x = transformer_block(x, num_heads=4, mlp_dim=768)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Reshape((h, w, 256))(x)
    conv5 = Conv2D(512, 3, padding="same")(x)
    conv5 = BatchNormalization(dtype="float32")(conv5)
    conv5 = Activation("relu")(conv5)

    # Decoder
    up1 = Conv2DTranspose(256, 3, 2, padding="same")(conv5)
    merge1 = Concatenate()([up1, skip2])
    conv6 = Conv2D(256, 3, padding="same")(merge1)
    conv6 = BatchNormalization(dtype="float32")(conv6)
    conv6 = Activation("relu")(conv6)

    up2 = Conv2DTranspose(128, 3, 2, padding="same")(conv6)
    merge2 = Concatenate()([up2, skip1])
    conv7 = Conv2D(128, 3, padding="same")(merge2)
    conv7 = BatchNormalization(dtype="float32")(conv7)
    conv7 = Activation("relu")(conv7)

    up3 = Conv2DTranspose(64, 3, 2, padding="same")(conv7)
    conv8 = Conv2D(64, 3, padding="same")(up3)
    conv8 = BatchNormalization(dtype="float32")(conv8)
    conv8 = Activation("relu")(conv8)

    up4 = Conv2DTranspose(32, 3, 2, padding="same")(conv8)
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

    model_path = os.path.join(output_dir, f"transunet_{combo_name}.keras")

    start_time = time.time()

    if os.path.exists(model_path):
        print(f"Loading existing model: {model_path}")

        model = load_model(model_path, custom_objects=custom_objects)

    else:
        print(f"Training new model: {combo_name}")

        model = new_transunet(len(channel_indices))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=combo_loss, metrics=[dice_coef])
        model.summary()
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=2)
        model.save(model_path)
        print(f"Saved model to {model_path}")

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

custom_objects = {
    "PositionEmbedding": PositionEmbedding,
    "dice_coef": dice_coef,
    "dice_loss": dice_loss,
    "weighted_bce": weighted_bce,
    "combo_loss": combo_loss
}

all_results = []

for combo in all_combinations:

    result = train_combination(combo)

    all_results.append(result)

    pd.DataFrame(all_results).to_csv(results_file, index=False)

with open("transunet_results_tf.json", "w") as f:
    json.dump(all_results, f, indent=4)
