"""Equal-budget fine-tuning of every backbone: train its final stage on cached trunk activations.

Fully fine-tuning VGG16 or Xception for five folds is not feasible on a CPU, but the expensive
part of fine-tuning only the top of a network is the frozen trunk, whose output never changes.
So each network is split at the start of its final (lowest-resolution) stage:

* the **trunk** (everything before the cut) runs once per image; its activations are cached;
* the **top** (the final stage, re-built from the pretrained layers) plus a new classifier head is
  trained on those activations, with the same recipe for every model.

Batch-norm layers in the top stay frozen (inference mode), and augmentation is applied to the cached
activation maps (horizontal flip and small shifts), so the protocol is identical across networks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from .features import backbone_registry, load_image
from .utils import set_seed

# First layer of each network's final stage is the one *after* these cut layers.
CUTS = {
    "VGG16": "block4_pool",           # trains block5 (3 convs at 14x14) + pool
    "MobileNet": "conv_pw_11_relu",   # trains the stride-2 depthwise block and block 13 (7x7)
    "ResNet50": "conv4_block6_out",   # trains conv5 (three bottleneck blocks, 7x7)
    "InceptionV3": "mixed7",          # trains mixed8-mixed10 (the last reduction and two blocks)
    "Xception": "add_10",             # trains the exit flow (blocks 13 and 14)
}


@dataclass(frozen=True)
class StageConfig:
    img_size: int = 224
    crop: bool = True
    stride: int = 2
    head_epochs: int = 2
    epochs: int = 12
    patience: int = 3
    batch_size: int = 32
    head_lr: float = 1e-3
    lr: float = 1e-4
    dropout: float = 0.5
    multilevel_head: bool = True  # also feed pooled cut-stage activations to the head (mid-level features)


def split_backbone(name: str, img_size: int, weights: str | None = "imagenet"):
    """Return (trunk, top, preprocess): trunk maps images to cut activations, top maps those to pooled features."""
    import tensorflow as tf

    tf.keras.backend.clear_session()  # layer names such as Xception's "add_10" restart from zero
    constructor, preprocess = backbone_registry()[name]
    base = constructor(include_top=False, weights=weights, input_shape=(img_size, img_size, 3), pooling="avg")
    cut = CUTS[name]
    trunk = tf.keras.Model(base.input, base.get_layer(cut).output, name=f"{name}_trunk")

    names = [layer.name for layer in base.layers]
    new_input = tf.keras.Input(shape=base.get_layer(cut).output.shape[1:], name="cut_activations")
    tensors = {cut: new_input}
    for layer in base.layers[names.index(cut) + 1:]:
        inbound = layer._inbound_nodes[0].inbound_layers
        inbound = inbound if isinstance(inbound, list) else [inbound]
        try:
            args = [tensors[parent.name] for parent in inbound]
        except KeyError as err:  # a skip connection crosses the cut: not a valid stage boundary
            raise ValueError(f"{name}: layer {layer.name} depends on {err} from before the cut {cut}") from None
        tensors[layer.name] = layer(args[0] if len(args) == 1 else args)
    top = tf.keras.Model(new_input, tensors[names[-1]], name=f"{name}_top")
    for layer in top.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False  # inference-mode batch norm, as in full fine-tuning on small data
    return trunk, top, preprocess


def _drop_page_cache(fh) -> None:
    """Flush written pages to disk and ask the OS to drop them from RAM (keeps WSL memory bounded)."""
    import os

    fh.flush()
    os.fsync(fh.fileno())
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def cache_trunk(root: Path, paths: list[str], name: str, cfg: StageConfig, cache_dir: Path,
                weights: str | None = "imagenet", batch_size: int = 32, log=print) -> np.ndarray:
    """Trunk activations for every image, as a read-only float16 memory map (computed once).

    Written with plain file writes that are flushed to disk and evicted from the page cache every
    few batches, so a multi-gigabyte cache never accumulates in RAM. An interrupted run resumes from
    the last flushed image (recorded in a .progress file).
    """
    import tensorflow as tf

    trunk, _, preprocess = split_backbone(name, cfg.img_size, weights)
    shape = (len(paths), *trunk.output.shape[1:])
    row_bytes = int(np.prod(shape[1:])) * 2
    tag = f"trunk_{name}_{cfg.img_size}{'_crop' if cfg.crop else ''}_{weights or 'random'}_{len(paths)}"
    data_file = Path(cache_dir) / f"{tag}.f16"
    done_file = Path(cache_dir) / f"{tag}.done"
    progress_file = Path(cache_dir) / f"{tag}.progress"
    if not done_file.exists():
        data_file.parent.mkdir(parents=True, exist_ok=True)
        start = int(progress_file.read_text()) if progress_file.exists() and data_file.exists() else 0
        mode = "r+b" if start else "wb"
        with open(data_file, mode) as fh:
            if not start:
                fh.truncate(len(paths) * row_bytes)
            fh.seek(start * row_bytes)
            files = [str(Path(root) / p) for p in paths[start:]]
            dataset = (tf.data.Dataset.from_tensor_slices(files)
                       .map(lambda p: load_image(p, cfg.img_size, cfg.crop), num_parallel_calls=2)
                       .batch(batch_size).map(preprocess).prefetch(1))
            if start:
                log(f"    {name} trunk cache: resuming at {start}/{len(paths)}")
            done = start
            for i, batch in enumerate(dataset, start=1):
                act = trunk(batch, training=False).numpy().astype(np.float16)
                fh.write(act.tobytes())
                done += len(act)
                if i % 20 == 0:
                    _drop_page_cache(fh)
                    progress_file.write_text(str(done))
                if i % 50 == 0:
                    log(f"    {name} trunk cache: {done}/{len(paths)}")
            _drop_page_cache(fh)
        if done != len(paths):
            raise RuntimeError(f"{name}: wrote {done} of {len(paths)} activations")
        done_file.touch()
        progress_file.unlink(missing_ok=True)
        tf.keras.backend.clear_session()
    return np.memmap(data_file, dtype=np.float16, mode="r", shape=shape)


def build_model(name: str, cfg: StageConfig, n_classes: int, weights: str | None = "imagenet"):
    """Top stage + head, taking cached float16 activations as input."""
    import tensorflow as tf

    _, top, _ = split_backbone(name, cfg.img_size, weights)
    inputs = tf.keras.Input(shape=top.input.shape[1:], dtype=tf.float16)
    x = tf.cast(inputs, tf.float32)
    x = tf.keras.layers.RandomFlip("horizontal")(x)
    x = tf.keras.layers.RandomTranslation(0.08, 0.08, fill_mode="constant")(x)
    pooled = top(x)
    if len(pooled.shape) == 4:
        pooled = tf.keras.layers.GlobalAveragePooling2D()(pooled)
    if cfg.multilevel_head:  # mid-level features transferred better than the last stage alone (preprocessing study)
        pooled = tf.keras.layers.Concatenate()([tf.keras.layers.GlobalAveragePooling2D()(x), pooled])
    x = tf.keras.layers.Dropout(cfg.dropout)(pooled)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), top


def _stream(acts: np.ndarray, y: np.ndarray, idx: np.ndarray, batch_size: int, sample_weight: np.ndarray | None,
            shuffle: bool, seed: int):
    """tf.data pipeline that reads batches from the (memory-mapped) activation cache, so RAM stays bounded."""
    import tensorflow as tf

    rng = np.random.default_rng(seed)
    shape = acts.shape[1:]

    def generate():
        order = rng.permutation(idx) if shuffle else idx
        for i in range(0, len(order), batch_size):
            b = np.sort(order[i:i + batch_size])  # sorted reads are faster on a memory map
            x = np.asarray(acts[b])
            if sample_weight is None:
                yield x, y[b]
            else:
                yield x, y[b], sample_weight[y[b]].astype(np.float32)

    spec = [tf.TensorSpec((None, *shape), tf.float16), tf.TensorSpec((None,), tf.int64)]
    if sample_weight is not None:
        spec.append(tf.TensorSpec((None,), tf.float32))
    return tf.data.Dataset.from_generator(generate, output_signature=tuple(spec)).prefetch(2)


def train_fold(acts: np.ndarray, y: np.ndarray, train: np.ndarray, val_es: np.ndarray, val_full: np.ndarray,
               test: np.ndarray, name: str, cfg: StageConfig, n_classes: int, seed: int,
               weights: str | None = "imagenet", log=print):
    """Train on cached activations; early-stop on `val_es`; return (test probs, full-val probs, history)."""
    import tensorflow as tf

    set_seed(seed)
    model, top = build_model(name, cfg, n_classes, weights)
    w = compute_class_weight("balanced", classes=np.arange(n_classes), y=y[train])  # per class
    y = y.astype(np.int64)
    ds_train = _stream(acts, y, train, cfg.batch_size, w, shuffle=True, seed=seed)
    ds_val = _stream(acts, y, val_es, cfg.batch_size, None, shuffle=False, seed=seed)

    top.trainable = False
    model.compile(tf.keras.optimizers.Adam(cfg.head_lr), "sparse_categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(ds_train, validation_data=ds_val, epochs=cfg.head_epochs, verbose=0)

    top.trainable = True
    for layer in top.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(tf.keras.optimizers.Adam(cfg.lr), "sparse_categorical_crossentropy", metrics=["accuracy"])
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.patience, restore_best_weights=True)
    h2 = model.fit(ds_train, validation_data=ds_val, epochs=cfg.epochs, callbacks=[stop], verbose=0)
    best = int(np.argmin(h2.history["val_loss"]))
    log(f"    head val acc {h1.history['val_accuracy'][-1]:.4f}; fine-tune {len(h2.history['loss'])} epochs, "
        f"best val acc {h2.history['val_accuracy'][best]:.4f}")

    def predict(idx):
        return np.concatenate([model(np.asarray(acts[idx[i:i + 256]]), training=False).numpy()
                               for i in range(0, len(idx), 256)])

    p_test, p_val = predict(test), predict(val_full)
    tf.keras.backend.clear_session()
    return p_test, p_val, {"head": h1.history, "finetune": h2.history}
