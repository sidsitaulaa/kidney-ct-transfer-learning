"""Input pipeline for the image-level experiment, shared by scripts/run_image_level.py and
notebooks/image_level_colab.ipynb so both feed the networks identically.

Every image is
1. decoded to 3-channel RGB (`decode_image(channels=3)`: grey is repeated to three
   channels, RGBA loses alpha),
2. resized to 150x150 with bilinear interpolation and no aspect-ratio preservation (the
   behaviour of `image_dataset_from_directory`),
3. kept as float32 in 0-255 and passed **once** through the network's own
   `preprocess_input` (never divided by 255 first).

TensorFlow is imported inside the functions so the module is cheap to import.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

IMG_SIZE = 150
CLASS_INDEX = {"Cyst": 0, "Normal": 1, "Stone": 2, "Tumor": 3}
MODELS = ("VGG16", "MobileNet", "ResNet50", "InceptionV3", "Xception")

# name -> (keras.applications attribute, preprocessing module, description, expected output range)
# "caffe": RGB->BGR, then subtract the ImageNet channel means (B 103.939, G 116.779, R 123.68), no scaling.
# "tf": x / 127.5 - 1, i.e. [-1, 1].
_CAFFE_RANGE = (-123.68 - 1e-3, 255.0 - 103.939 + 1e-3)
PREPROCESSING = {
    "VGG16": ("VGG16", "vgg16", "caffe: RGB->BGR, subtract ImageNet mean per channel", _CAFFE_RANGE),
    "MobileNet": ("MobileNet", "mobilenet", "tf: x/127.5 - 1 -> [-1, 1]", (-1.0, 1.0)),
    "ResNet50": ("ResNet50", "resnet50", "caffe: RGB->BGR, subtract ImageNet mean per channel", _CAFFE_RANGE),
    "InceptionV3": ("InceptionV3", "inception_v3", "tf: x/127.5 - 1 -> [-1, 1]", (-1.0, 1.0)),
    "Xception": ("Xception", "xception", "tf: x/127.5 - 1 -> [-1, 1]", (-1.0, 1.0)),
}


def label_of(path: str) -> int:
    """Class index from the class folder of a relative path such as 'Cyst/Cyst- (1).jpg'."""
    return CLASS_INDEX[Path(path).parts[-2]]


def check_labels(paths: Sequence[str], labels: Sequence[int]) -> None:
    """Every file's label must match its folder under the fixed CLASS_INDEX map."""
    bad = [p for p, y in zip(paths, labels, strict=True) if label_of(p) != int(y)]
    if bad:
        raise AssertionError(f"{len(bad)} labels do not match their folder, e.g. {bad[:3]}")


def audit_files(root: Path, paths: Sequence[str]) -> dict:
    """Format, colour mode, bit depth and size of every file, plus any that fail to decode."""
    from PIL import Image

    fmt, mode, sizes, bits, failed, grey_rgb = Counter(), Counter(), Counter(), Counter(), [], 0
    for i, p in enumerate(paths):
        try:
            with Image.open(Path(root) / p) as im:
                im.load()
                fmt[im.format] += 1
                mode[im.mode] += 1
                sizes[im.size] += 1
                bits[{"1": 1, "L": 8, "P": 8, "RGB": 8, "RGBA": 8, "I;16": 16, "I": 32, "F": 32}.get(im.mode, -1)] += 1
                if i % 50 == 0 and im.mode == "RGB":  # are the three channels identical (grey CT stored as RGB)?
                    a = np.asarray(im)
                    grey_rgb += int((a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all())
        except Exception as exc:  # noqa: BLE001 - log every failure, never stop the audit
            failed.append({"path": p, "error": repr(exc)})
    ratios = np.array([w / h for (w, h), n in sizes.items() for _ in range(n)])
    return {
        "formats": dict(fmt), "modes": dict(mode), "bit_depth_per_channel": {str(k): v for k, v in bits.items()},
        "decode_failures": failed,
        "distinct_sizes": len(sizes),
        "most_common_sizes": [[f"{w}x{h}", n] for (w, h), n in sizes.most_common(8)],
        "width_range": [min(w for w, _ in sizes), max(w for w, _ in sizes)],
        "height_range": [min(h for _, h in sizes), max(h for _, h in sizes)],
        "aspect_ratio_w_over_h": {"min": float(ratios.min()), "median": float(np.median(ratios)),
                                  "max": float(ratios.max()),
                                  "share_outside_0.8_to_1.25": float(((ratios < 0.8) | (ratios > 1.25)).mean())},
        "sampled_rgb_with_identical_channels": f"{grey_rgb} of {sum(1 for i in range(len(paths)) if i % 50 == 0)}",
    }


def backbone(name: str, weights: str | None = "imagenet"):
    """The notebook's frozen backbone (`classes=4` dropped: it is ignored with include_top=False)."""
    import tensorflow as tf

    attr, module, _, _ = PREPROCESSING[name]
    net = getattr(tf.keras.applications, attr)(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                               pooling="max", weights=weights)
    net.trainable = False
    return net


def preprocess_fn(name: str):
    import tensorflow as tf

    return getattr(tf.keras.applications, PREPROCESSING[name][1]).preprocess_input


def decode_resize(path):
    """File path tensor -> 150x150x3 float32 RGB in 0-255."""
    import tensorflow as tf

    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return tf.image.resize(tf.cast(img, tf.float32), (IMG_SIZE, IMG_SIZE), method="bilinear")


def image_dataset(files: Sequence[str], batch_size: int, labels: Sequence[int] | None = None,
                  model: str | None = None, shuffle: bool = False, seed: int = 123):
    """Batched images in file order (shuffle only for training batches), optionally with labels
    and with the model's preprocess_input applied once. Without `model`, pixels stay 0-255."""
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices(list(files) if labels is None else (list(files), np.asarray(labels)))
    if shuffle:
        ds = ds.shuffle(len(files), seed=seed, reshuffle_each_iteration=True)
    pre = preprocess_fn(model) if model else None

    def load(*item):
        x = decode_resize(item[0])
        if pre is not None:
            x = pre(x)
        return x if labels is None else (x, item[1])

    return ds.map(load, num_parallel_calls=2).batch(batch_size).prefetch(1)


def preprocessing_stats(name: str, batch_0_255: np.ndarray) -> dict:
    """min/max/mean/std of a 0-255 batch after the model's preprocess_input, with a range check."""
    import tensorflow as tf

    x = tf.constant(batch_0_255, dtype=tf.float32)
    if float(tf.reduce_max(x)) <= 1.0:
        raise AssertionError("input looks already scaled to [0, 1]; the pipeline must feed 0-255")
    y = preprocess_fn(name)(tf.identity(x)).numpy()
    lo, hi = PREPROCESSING[name][3]
    stats = {"model": name, "mode": PREPROCESSING[name][2], "min": float(y.min()), "max": float(y.max()),
             "mean": float(y.mean()), "std": float(y.std()), "expected_range": [lo, hi]}
    if not (y.min() >= lo - 1e-4 and y.max() <= hi + 1e-4):
        raise AssertionError(f"{name}: preprocessed range [{y.min()}, {y.max()}] outside [{lo}, {hi}]")
    if PREPROCESSING[name][2].startswith("caffe"):
        # BGR order: channel 0 must be blue-minus-mean; check against a manual computation
        manual = x.numpy()[..., ::-1] - np.array([103.939, 116.779, 123.68], dtype=np.float32)
        if not np.allclose(manual, y, atol=1e-3):
            raise AssertionError(f"{name}: caffe preprocessing does not match RGB->BGR minus mean")
    else:
        if not np.allclose(x.numpy() / 127.5 - 1.0, y, atol=1e-5):
            raise AssertionError(f"{name}: tf preprocessing does not match x/127.5 - 1")
    return stats


def assert_aligned(feature_paths: Sequence[str], split_paths: Sequence[str]) -> None:
    """Rows of a feature cache must be in exactly the order of split.csv."""
    if list(feature_paths) != list(split_paths):
        raise AssertionError("feature rows are not aligned with split.csv")
