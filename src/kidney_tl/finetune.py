"""Stage 3: fine-tune a backbone end to end under the same group-level folds.

Frozen ImageNet features plateau well below what the data allows, so the top of the
network is adapted to CT. To keep this feasible on a CPU:

* training and validation use every `stride`-th slice of each scan (neighbouring slices
  correlate at r ~ 0.999, so little information is lost); every test image is still predicted;
* training runs in two phases: the new head alone, then the top `unfreeze` layers as well;
* batch-norm layers of the backbone stay in inference mode, the usual practice for small data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from .features import backbone_registry, load_image
from .utils import set_seed


@dataclass(frozen=True)
class FinetuneConfig:
    backbone: str = "MobileNet"
    img_size: int = 192
    stride: int = 3
    unfreeze: int = 30
    head_epochs: int = 4
    finetune_epochs: int = 12
    patience: int = 3
    batch_size: int = 32
    head_lr: float = 1e-3
    finetune_lr: float = 1e-4
    dropout: float = 0.5
    crop: bool = False  # crop to the body and pad to a square before resizing (see preprocess.py)


def thin_by_scan(idx: np.ndarray, groups: np.ndarray, slices: np.ndarray, stride: int) -> np.ndarray:
    """Keep every `stride`-th slice (in slice order) of each scan group within `idx`."""
    if stride <= 1:
        return idx
    frame = pd.DataFrame({"i": idx, "g": groups[idx], "s": slices[idx]}).sort_values(["g", "s"])
    keep = frame.groupby("g").cumcount() % stride == 0
    return np.sort(frame.loc[keep, "i"].to_numpy())


def _dataset(files: Sequence[str], labels: np.ndarray | None, cfg: FinetuneConfig, training: bool):
    import tensorflow as tf

    def load(path):
        return tf.cast(load_image(path, cfg.img_size, cfg.crop), tf.uint8)

    images = tf.data.Dataset.from_tensor_slices(list(files)).map(load, num_parallel_calls=2)
    ds = images if labels is None else tf.data.Dataset.zip((images, tf.data.Dataset.from_tensor_slices(labels)))
    ds = ds.cache()  # decoded, resized uint8 images stay in memory after the first epoch
    if training:
        ds = ds.shuffle(2048, reshuffle_each_iteration=True)
    return ds.batch(cfg.batch_size).prefetch(1)


def build_model(cfg: FinetuneConfig, n_classes: int, weights: str | None = "imagenet"):
    import tensorflow as tf

    constructor, preprocess = backbone_registry()[cfg.backbone]
    base = constructor(include_top=False, weights=weights, input_shape=(cfg.img_size, cfg.img_size, 3),
                       pooling="avg")
    base.trainable = False
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.03),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
        tf.keras.layers.RandomContrast(0.15),
    ], name="augment")
    inputs = tf.keras.Input((cfg.img_size, cfg.img_size, 3), dtype=tf.uint8)
    x = augment(tf.cast(inputs, tf.float32))
    x = tf.keras.layers.Lambda(preprocess, name="preprocess")(x)
    x = base(x, training=False)  # keeps batch-norm statistics frozen even after unfreezing
    x = tf.keras.layers.Dropout(cfg.dropout)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base


def train_and_predict(root: Path, paths: Sequence[str], y: np.ndarray, train: np.ndarray, val: np.ndarray,
                      test: np.ndarray, cfg: FinetuneConfig, n_classes: int, seed: int,
                      weights: str | None = "imagenet", log=print,
                      predict_val: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fine-tune on `train`, early-stop on `val`; return (test probabilities, validation probabilities, history).

    Validation probabilities are for `predict_val` when given (e.g. the full, un-thinned validation set),
    otherwise for `val`.
    """
    import tensorflow as tf

    set_seed(seed)
    files = np.array([str(Path(root) / p) for p in paths])
    ds_train = _dataset(files[train], y[train], cfg, training=True)
    ds_val = _dataset(files[val], y[val], cfg, training=False)
    weights_by_class = compute_class_weight("balanced", classes=np.arange(n_classes), y=y[train])
    class_weight = dict(enumerate(weights_by_class))

    model, base = build_model(cfg, n_classes, weights)
    model.compile(tf.keras.optimizers.Adam(cfg.head_lr), "sparse_categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(ds_train, validation_data=ds_val, epochs=cfg.head_epochs, class_weight=class_weight, verbose=0)
    log(f"    head phase: val accuracy {h1.history['val_accuracy'][-1]:.4f}")

    base.trainable = True
    for layer in base.layers[:-cfg.unfreeze]:
        layer.trainable = False
    model.compile(tf.keras.optimizers.Adam(cfg.finetune_lr), "sparse_categorical_crossentropy", metrics=["accuracy"])
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.patience, restore_best_weights=True)
    h2 = model.fit(ds_train, validation_data=ds_val, epochs=cfg.finetune_epochs, class_weight=class_weight,
                   callbacks=[stop], verbose=0)
    log(f"    fine-tune phase: {len(h2.history['loss'])} epochs, best val loss {min(h2.history['val_loss']):.4f}, "
        f"val accuracy {h2.history['val_accuracy'][int(np.argmin(h2.history['val_loss']))]:.4f}")

    val_idx = val if predict_val is None else predict_val
    p_val = model.predict(_dataset(files[val_idx], None, cfg, training=False), verbose=0)
    p_test = model.predict(_dataset(files[test], None, cfg, training=False), verbose=0)
    history = {"head": h1.history, "finetune": h2.history}
    tf.keras.backend.clear_session()
    return p_test, p_val, history
