"""Input pipeline of the image-level experiment: preprocessing range per model, label map, row alignment.

Needs TensorFlow (run inside the docker/Dockerfile.image_level image); skipped without it.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

tf = pytest.importorskip("tensorflow")

from kidney_tl.image_level_pipeline import (  # noqa: E402
    CLASS_INDEX,
    MODELS,
    assert_aligned,
    check_labels,
    image_dataset,
    label_of,
    preprocessing_stats,
)

ROOT = Path(__file__).resolve().parents[1]


def _batch(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 255, (4, 150, 150, 3)).astype(np.float32)
    x[0] = 0.0
    x[1] = 255.0  # hit both ends of the pixel range
    return x


@pytest.mark.parametrize("name", MODELS)
def test_preprocessing_range_per_model(name):
    st = preprocessing_stats(name, _batch())
    if name in ("MobileNet", "InceptionV3", "Xception"):
        assert st["min"] == pytest.approx(-1.0) and st["max"] == pytest.approx(1.0)
    else:  # caffe: BGR minus ImageNet means
        assert st["min"] == pytest.approx(-123.68, abs=1e-3)
        assert st["max"] == pytest.approx(255 - 103.939, abs=1e-3)


@pytest.mark.parametrize("name", MODELS)
def test_already_scaled_input_is_rejected(name):
    with pytest.raises(AssertionError):
        preprocessing_stats(name, _batch() / 255.0)


def test_class_index_map_and_folder_labels():
    assert CLASS_INDEX == {"Cyst": 0, "Normal": 1, "Stone": 2, "Tumor": 3}
    paths = ["Cyst/Cyst- (1).jpg", "Normal/Normal- (2).jpg", "Stone/Stone- (3).jpg", "Tumor/Tumor- (4).jpg"]
    assert [label_of(p) for p in paths] == [0, 1, 2, 3]
    check_labels(paths, [0, 1, 2, 3])
    with pytest.raises(AssertionError):
        check_labels(paths, [0, 1, 3, 2])


def test_decoding_to_rgb_and_order_is_kept(tmp_path):
    """Grey PNG -> three equal channels, RGBA -> alpha dropped, files come back in the given order."""
    files = []
    for i in range(6):
        p = tmp_path / f"img{i}.png"
        if i % 3 == 0:
            Image.fromarray(np.full((40, 52), 20 * i, np.uint8), "L").save(p)
        elif i % 3 == 1:
            Image.fromarray(np.full((40, 40, 4), 20 * i, np.uint8), "RGBA").save(p)
        else:
            Image.fromarray(np.full((60, 40, 3), 20 * i, np.uint8), "RGB").save(p)
        files.append(str(p))
    x = np.concatenate([b.numpy() for b in image_dataset(files, 4)])
    assert x.shape == (6, 150, 150, 3) and x.dtype == np.float32
    assert np.allclose(x.mean(axis=(1, 2, 3)), [20 * i for i in range(6)], atol=0.5)  # order kept, 0-255 kept
    assert np.allclose(x[..., 0], x[..., 2])


def test_feature_rows_must_match_split(tmp_path):
    split_paths = ["Cyst/a.jpg", "Normal/b.jpg", "Stone/c.jpg"]
    assert_aligned(split_paths, split_paths)
    with pytest.raises(AssertionError):
        assert_aligned(split_paths[::-1], split_paths)

    spec = importlib.util.spec_from_file_location("run_image_level", ROOT / "scripts" / "run_image_level.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    df = pd.DataFrame({"path": split_paths})
    cache = tmp_path / "image_level_X_150_max.npy"
    np.save(cache, np.arange(6, dtype=np.float32).reshape(3, 2))
    with pytest.raises(RuntimeError):  # no row manifest: refuse to use the cache
        mod.load_features(cache, df)
    (tmp_path / "image_level_X_150_max.paths.txt").write_text("\n".join(split_paths[::-1]) + "\n")
    with pytest.raises(AssertionError):
        mod.load_features(cache, df)
    (tmp_path / "image_level_X_150_max.paths.txt").write_text("\n".join(split_paths) + "\n")
    assert mod.load_features(cache, df).shape == (3, 2)
