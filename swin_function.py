"""
Code copied from
https://github.com/sayakpaul/swin-transformers-tf/tree/main
"""

import collections.abc
from functools import partial
from typing import Dict, Union
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
import collections.abc
from typing import Tuple, Union
from tensorflow.keras import layers
from typing import List


@tf.keras.utils.register_keras_serializable()
def mlp_block(dropout_rate: float, hidden_units: List[int], name: str = "mlp"):
    """FFN for a Transformer block."""
    ffn = keras.Sequential(name=name)
    for (idx, units) in enumerate(hidden_units):
        ffn.add(
            layers.Dense(
                units,
                activation=tf.nn.gelu if idx == 0 else None,
                bias_initializer=keras.initializers.RandomNormal(stddev=1e-6),
            )
        )
        ffn.add(layers.Dropout(dropout_rate))
    return ffn


@tf.keras.utils.register_keras_serializable()
def window_partition(x: tf.Tensor, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

    x = tf.reshape(
        x, (B, H // window_size, window_size, W // window_size, window_size, C)
    )
    windows = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    windows = tf.reshape(windows, (-1, window_size, window_size, C))
    return windows


@tf.keras.utils.register_keras_serializable()
def window_reverse(windows: tf.Tensor, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = tf.shape(windows)[0] // tf.cast(
        H * W / window_size / window_size, dtype="int32"
    )

    x = tf.reshape(
        windows,
        (
            B,
            H // window_size,
            W // window_size,
            window_size,
            window_size,
            -1,
        ),
    )
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    return tf.reshape(x, (B, H, W, -1))


@tf.keras.utils.register_keras_serializable()
def get_relative_position_index(win_h, win_w):
    # get pair-wise relative position index for each token inside the window
    xx, yy = tf.meshgrid(range(win_h), range(win_w))
    coords = tf.stack([yy, xx], axis=0)  # [2, Wh, Ww]
    coords_flatten = tf.reshape(coords, [2, -1])  # [2, Wh*Ww]

    relative_coords = (
        coords_flatten[:, :, None] - coords_flatten[:, None, :]
    )  # [2, Wh*Ww, Wh*Ww]
    relative_coords = tf.transpose(
        relative_coords, perm=[1, 2, 0]
    )  # [Wh*Ww, Wh*Ww, 2]

    xx = (relative_coords[:, :, 0] + win_h - 1) * (2 * win_w - 1)
    yy = relative_coords[:, :, 1] + win_w - 1
    relative_coords = tf.stack([xx, yy], axis=-1)

    return tf.reduce_sum(relative_coords, axis=-1)  # [Wh*Ww, Wh*Ww]


@tf.keras.utils.register_keras_serializable()
class WindowAttention(layers.Layer):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        head_dim (int): Number of channels per head (dim // num_heads if not set)
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        num_heads,
        head_dim=None,
        window_size=7,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.dim = dim
        self.window_size = (
            window_size
            if isinstance(window_size, collections.abc.Iterable)
            else (window_size, window_size)
        )  # Wh, Ww
        self.win_h, self.win_w = self.window_size
        self.window_area = self.win_h * self.win_w
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.attn_dim = self.head_dim * num_heads
        self.scale = self.head_dim ** -0.5

        # get pair-wise relative position index for each token inside the window
        self.relative_position_index = get_relative_position_index(
            self.win_h, self.win_w
        )

        self.qkv = layers.Dense(
            self.attn_dim * 3, use_bias=qkv_bias, name="attention_qkv"
        )
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim, name="attention_projection")
        self.proj_drop = layers.Dropout(proj_drop)

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(
            shape=((2 * self.win_h - 1) * (2 * self.win_w - 1), self.num_heads),
            initializer="zeros",
            trainable=True,
            name="relative_position_bias_table",
        )
        super().build(input_shape)

    def _get_rel_pos_bias(self) -> tf.Tensor:
        relative_position_bias = tf.gather(
            self.relative_position_bias_table,
            self.relative_position_index,
            axis=0,
        )
        return tf.transpose(relative_position_bias, [2, 0, 1])

    def call(
        self, x, mask=None, return_attns=False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, (B_, N, 3, self.num_heads, -1))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))

        q, k, v = tf.unstack(qkv, 3)

        scale = tf.cast(self.scale, dtype=qkv.dtype)
        q = q * scale
        attn = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2]))
        attn = attn + self._get_rel_pos_bias()

        if mask is not None:
            num_win = tf.shape(mask)[0]
            attn = tf.reshape(
                attn, (B_ // num_win, num_win, self.num_heads, N, N)
            )
            attn = attn + tf.expand_dims(mask, 1)[None, ...]

            attn = tf.reshape(attn, (-1, self.num_heads, N, N))
            attn = tf.nn.softmax(attn, -1)
        else:
            attn = tf.nn.softmax(attn, -1)

        attn = self.attn_drop(attn)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (B_, N, C))

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attns:
            return x, attn
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "window_size": self.window_size,
                "win_h": self.win_h,
                "win_w": self.win_w,
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "attn_dim": self.attn_dim,
                "scale": self.scale,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable()
class StochasticDepth(layers.Layer):
    def __init__(self, drop_prop, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = float(drop_prop)

    def call(self, x, training=False):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],)+(1,)*(tf.shape(x).shape[0]-1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"drop_prob": self.drop_prob})
        return config


@tf.keras.utils.register_keras_serializable()
class SwinTransformerBlock(layers.Layer):
    """Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        window_size (int): Window size.
        num_heads (int): Number of attention heads.
        head_dim (int): Enforce the number of channels per head
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        norm_layer (layers.Layer, optional): Normalization layer.  Default: layers.LayerNormalization
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads=4,
        head_dim=None,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=partial(L.LayerNormalization, epsilon=1e-5),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer()
        self.attn = WindowAttention(
            dim,
            num_heads=num_heads,
            head_dim=head_dim,
            window_size=window_size
            if isinstance(window_size, collections.abc.Iterable)
            else (window_size, window_size),
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            name="window_attention",
        )

        self.drop_path = (
            StochasticDepth(drop_path) if drop_path > 0.0 else tf.identity
        )
        self.norm2 = norm_layer()
        self.mlp = mlp_block(
            dropout_rate=drop, hidden_units=[int(dim * mlp_ratio), dim]
        )

        if self.shift_size > 0:
            # `get_attn_mask()` uses NumPy to make in-place assignments.
            # Since this is done during initialization, it's okay.
            self.attn_mask = self.get_attn_mask()
        else:
            self.attn_mask = None

    def get_config(self):
        config = super().get_config()

        config.update({
            "dim": self.dim,
            "input_resolution": self.input_resolution,
            "window_size": self.window_size,
            "shift_size": self.shift_size,
            "mlp_ratio": self.mlp_ratio,
        })

        return config

    def get_attn_mask(self):
        # calculate attention mask for SW-MSA
        H, W = self.input_resolution
        img_mask = np.zeros((1, H, W, 1))  # [1, H, W, 1]
        cnt = 0
        for h in (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        ):
            for w in (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            ):
                img_mask[:, h, w, :] = cnt
                cnt += 1

        img_mask = tf.convert_to_tensor(img_mask, dtype="float32")
        mask_windows = window_partition(
            img_mask, self.window_size
        )  # [num_win, window_size, window_size, 1]
        mask_windows = tf.reshape(
            mask_windows, (-1, self.window_size * self.window_size)
        )
        attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(
            mask_windows, 2
        )
        attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
        return tf.where(attn_mask == 0, 0.0, attn_mask)

    def call(
        self, x, return_attns=False
    ) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
        H, W = self.input_resolution
        B, L, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        shortcut = x
        x = self.norm1(x)
        x = tf.reshape(x, (B, H, W, C))

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # [num_win*B, window_size, window_size, C]
        x_windows = tf.reshape(
            x_windows, (-1, self.window_size * self.window_size, C)
        )  # [num_win*B, window_size*window_size, C]

        # W-MSA/SW-MSA
        if not return_attns:
            attn_windows = self.attn(
                x_windows, mask=self.attn_mask
            )  # [num_win*B, window_size*window_size, C]
        else:
            attn_windows, attn_scores = self.attn(
                x_windows, mask=self.attn_mask, return_attns=True
            )  # [num_win*B, window_size*window_size, C]
        # merge windows
        attn_windows = tf.reshape(
            attn_windows, (-1, self.window_size, self.window_size, C)
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, H, W
        )  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x,
                shift=(self.shift_size, self.shift_size),
                axis=(1, 2),
            )
        else:
            x = shifted_x
        x = tf.reshape(x, (B, H * W, C))

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if return_attns:
            return x, attn_scores
        else:
            return x