"""Classifier head trained on frozen backbone features."""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from .metrics import fast_macro_f1
from .utils import set_seed


@dataclass(frozen=True)
class HeadConfig:
    units: int
    dropout: float
    lr: float


def grid(units, dropouts, lrs) -> list[HeadConfig]:
    return [HeadConfig(u, d, lr) for u, d, lr in itertools.product(units, dropouts, lrs)]


def standardize(train: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """Z-score with statistics from the training subset only."""
    mean, std = train.mean(axis=0), train.std(axis=0) + 1e-6
    return [(x - mean) / std for x in (train, *others)]


def predict_proba(model, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Class probabilities via direct calls (avoids `Model.predict` retracing inside loops)."""
    return np.concatenate([model(x[i:i + batch_size], training=False).numpy() for i in range(0, len(x), batch_size)])


def build_head(input_dim: int, n_classes: int, cfg: HeadConfig):
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input((input_dim,)),
        tf.keras.layers.Dense(cfg.units, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(cfg.dropout),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.lr), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_head(x_train, y_train, x_val, y_val, cfg: HeadConfig, n_classes: int, seed: int, *,
               max_epochs: int, patience: int, batch_size: int, class_weights: bool):
    """Fit with early stopping on validation loss; returns (model, epochs_trained)."""
    import tensorflow as tf

    set_seed(seed)
    weights = None
    if class_weights:
        w = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train)
        weights = dict(enumerate(w))
    model = build_head(x_train.shape[1], n_classes, cfg)
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=max_epochs,
                        batch_size=batch_size, class_weight=weights, callbacks=[stop], verbose=0)
    return model, len(history.history["loss"])


def select_head(x_train, y_train, x_val, y_val, configs: list[HeadConfig], n_classes: int, seed: int,
                **train_kwargs) -> tuple[HeadConfig, list[dict]]:
    """Pick the head settings with the best validation macro-F1. The test set is never used here."""
    import tensorflow as tf

    rows, best, best_score = [], None, -1.0
    for cfg in configs:
        model, epochs = train_head(x_train, y_train, x_val, y_val, cfg, n_classes, seed, **train_kwargs)
        score = fast_macro_f1(y_val, predict_proba(model, x_val).argmax(axis=1), n_classes)
        rows.append({**asdict(cfg), "val_macro_f1": score, "epochs": epochs})
        if score > best_score:
            best, best_score = cfg, score
        tf.keras.backend.clear_session()
    return best, rows
