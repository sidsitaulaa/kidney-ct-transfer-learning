"""Equal-budget stage-wise fine-tuning: correct split and an end-to-end run on synthetic data."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kidney_tl.stagewise import split_backbone

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_stagewise.py"


def test_split_reproduces_the_full_network():
    from kidney_tl.features import backbone_registry

    x = np.random.default_rng(0).uniform(-1, 1, (2, 96, 96, 3)).astype("float32")
    trunk, top, _ = split_backbone("MobileNet", 96, weights=None)
    split = top(trunk(x, training=False), training=False).numpy()
    base = backbone_registry()["MobileNet"][0](include_top=False, weights=None, input_shape=(96, 96, 3),
                                               pooling="avg")
    base.set_weights(trunk.get_weights() + top.get_weights())
    np.testing.assert_allclose(split, base(x, training=False).numpy(), atol=1e-5)


def test_run_stagewise_end_to_end_and_resume(synthetic_dir, tmp_path):
    cmd = [sys.executable, str(SCRIPT), "--data-dir", str(synthetic_dir), "--results-dir", str(tmp_path / "res"),
           "--cache-dir", str(tmp_path / "cache"), "--models", "MobileNet", "ResNet50", "--img-size", "96",
           "--weights", "none", "--epochs", "1", "--patience", "1", "--folds", "3", "--threads", "2", "--nice", "0"]
    first = subprocess.run(cmd, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr[-2000:]
    for model in ("MobileNet", "ResNet50"):
        preds = pd.read_csv(tmp_path / "res" / "stagewise" / model / "test_predictions.csv.gz")
        assert len(preds) == 176 and preds["path"].is_unique
    second = subprocess.run(cmd, capture_output=True, text=True)
    assert second.returncode == 0 and "skipping" in second.stdout
