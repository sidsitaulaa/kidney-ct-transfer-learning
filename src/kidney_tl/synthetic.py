"""Tiny synthetic dataset that mimics the structure of the real one, for tests and smoke runs.

Each class has several 'scans'; each scan is a smooth random texture plus a class
marker, and its 'slices' differ only by small noise. Files are numbered in slice
order, and the first scan of each class is stored a second time at the end
(byte-identical copies), as happens in the real dataset.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ("Cyst", "Normal", "Stone", "Tumor")


def make_dataset(root: Path, scans_per_class: int = 10, slices_per_scan: int = 4, size: int = 64,
                 seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    root = Path(root)
    for ci, cls in enumerate(CLASSES):
        folder = root / cls
        folder.mkdir(parents=True, exist_ok=True)
        number = 1
        for _ in range(scans_per_class):
            low = rng.integers(0, 256, (8, 8)).astype(np.uint8)
            base = np.asarray(Image.fromarray(low).resize((size, size), Image.BILINEAR), dtype=np.int16)
            base[ci * size // 5: ci * size // 5 + size // 6, :] = 255
            for _ in range(slices_per_scan):
                img = np.clip(base + rng.integers(-2, 3, base.shape), 0, 255).astype(np.uint8)
                Image.fromarray(img).save(folder / f"{cls}- ({number}).jpg", quality=95)
                number += 1
        for s in range(1, slices_per_scan + 1):  # scan 1 stored again under new numbers
            shutil.copyfile(folder / f"{cls}- ({s}).jpg", folder / f"{cls}- ({number}).jpg")
            number += 1
    return root
