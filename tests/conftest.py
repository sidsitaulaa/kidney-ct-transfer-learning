import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kidney_tl.synthetic import make_dataset  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory) -> Path:
    """4 classes x 10 scans x 4 near-identical slices = 160 images, nested one level like Kaggle's archive."""
    root = tmp_path_factory.mktemp("kidney") / "CT-KIDNEY" / "CT-KIDNEY"
    return make_dataset(root, scans_per_class=10, slices_per_scan=4, size=64)
