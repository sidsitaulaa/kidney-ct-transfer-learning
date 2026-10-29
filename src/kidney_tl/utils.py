"""Runtime helpers: resource limits, seeding, logging, environment capture."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np


def limit_resources(threads: int, nice: int = 10) -> None:
    """Cap CPU threads and lower process priority so the machine stays responsive.

    Must run before TensorFlow creates its thread pools. `threads <= 0` means
    "let TensorFlow decide" (use on a dedicated GPU machine such as Colab).
    """
    if threads > 0:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS"):
            os.environ[var] = str(threads)
        os.environ["TF_NUM_INTEROP_THREADS"] = "2"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if nice and hasattr(os, "nice"):
        try:
            os.nice(nice)
        except OSError:
            pass
    import tensorflow as tf

    if threads > 0:
        try:
            tf.config.threading.set_intra_op_parallelism_threads(threads)
            tf.config.threading.set_inter_op_parallelism_threads(2)
        except RuntimeError:  # TensorFlow already initialised; environment variables still apply
            pass


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow, and request deterministic kernels."""
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def get_logger(log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("kidney_tl")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, mode="w")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def environment_info() -> dict:
    """Library versions and hardware, saved next to the results."""
    import scipy
    import sklearn
    import tensorflow as tf

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
        "cpu_count": os.cpu_count(),
    }


def write_json(obj: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=str))
