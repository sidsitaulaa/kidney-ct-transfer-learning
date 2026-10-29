"""Frozen ImageNet backbones as feature extractors.

Each network receives 0-255 RGB pixels passed through its own
`preprocess_input`, so the input matches what that network saw during
ImageNet pre-training.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def backbone_registry():
    import tensorflow as tf

    apps = tf.keras.applications
    return {
        "VGG16": (apps.VGG16, apps.vgg16.preprocess_input),
        "MobileNet": (apps.MobileNet, apps.mobilenet.preprocess_input),
        "MobileNetV2": (apps.MobileNetV2, apps.mobilenet_v2.preprocess_input),
        "ResNet50": (apps.ResNet50, apps.resnet50.preprocess_input),
        "InceptionV3": (apps.InceptionV3, apps.inception_v3.preprocess_input),
        "Xception": (apps.Xception, apps.xception.preprocess_input),
        "EfficientNetB0": (apps.EfficientNetB0, apps.efficientnet.preprocess_input),
    }


# Three depths per network for multi-level features (mid-level features often transfer better to
# medical images than the last, most ImageNet-specific block).
MULTILEVEL_LAYERS = {
    "VGG16": ("block3_pool", "block4_pool", "block5_pool"),
    "MobileNet": ("conv_pw_5_relu", "conv_pw_11_relu", "conv_pw_13_relu"),
    "MobileNetV2": ("block_6_expand_relu", "block_13_expand_relu", "out_relu"),
    "ResNet50": ("conv3_block4_out", "conv4_block6_out", "conv5_block3_out"),
    "InceptionV3": ("mixed2", "mixed7", "mixed10"),
    "Xception": ("add_2", "add_10", "block14_sepconv2_act"),
    "EfficientNetB0": ("block3b_add", "block5c_add", "top_activation"),
}


def available_backbones() -> list[str]:
    return list(backbone_registry())


def load_image(path, img_size: int, crop: bool):
    """Decode one file to an img_size x img_size float32 tensor in 0-255 (optionally body-cropped first)."""
    import tensorflow as tf

    from .preprocess import body_crop_square

    image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    if crop:
        image = tf.numpy_function(body_crop_square, [image], tf.uint8)
        image.set_shape([None, None, 3])
    return tf.image.resize(tf.cast(image, tf.float32), (img_size, img_size))


def _cache_path(cache_dir: Path, name: str, img_size: int, weights: str | None, paths: Sequence[str],
                variant: str, crop: bool = False) -> Path:
    digest = hashlib.sha1("\n".join(paths).encode()).hexdigest()[:10]
    suffix = ("" if variant == "final" else f"_{variant}") + ("_crop" if crop else "")
    return Path(cache_dir) / f"features_{name}_{img_size}_{weights or 'random'}{suffix}_{digest}.npy"


def extract_features(root: Path, paths: Sequence[str], name: str, img_size: int, weights: str | None,
                     batch_size: int, cache_dir: Path, variant: str = "final", crop: bool = False) -> np.ndarray:
    """Global-average-pooled backbone features for every image, cached on disk.

    variant="final" pools the last block; variant="multilevel" concatenates pooled
    features from the three layers in MULTILEVEL_LAYERS.
    """
    if variant not in ("final", "multilevel"):
        raise ValueError(f"unknown feature variant {variant!r}")
    cache = _cache_path(cache_dir, name, img_size, weights, paths, variant, crop)
    if cache.exists():
        feats = np.load(cache)
        if len(feats) == len(paths):
            return feats

    import tensorflow as tf

    constructor, preprocess = backbone_registry()[name]
    if variant == "final":
        backbone = constructor(include_top=False, weights=weights, input_shape=(img_size, img_size, 3),
                               pooling="avg")
    else:
        base = constructor(include_top=False, weights=weights, input_shape=(img_size, img_size, 3))
        pooled = [tf.keras.layers.GlobalAveragePooling2D()(base.get_layer(layer).output)
                  for layer in MULTILEVEL_LAYERS[name]]
        backbone = tf.keras.Model(base.input, tf.keras.layers.Concatenate()(pooled))
    backbone.trainable = False

    files = [str(Path(root) / p) for p in paths]
    dataset = (tf.data.Dataset.from_tensor_slices(files)
               .map(lambda p: load_image(p, img_size, crop), num_parallel_calls=2)
               .batch(batch_size)
               .map(preprocess)
               .prefetch(1))
    feats = backbone.predict(dataset, verbose=0).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, feats)
    tf.keras.backend.clear_session()
    return feats
