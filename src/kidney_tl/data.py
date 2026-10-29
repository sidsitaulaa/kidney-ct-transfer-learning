"""Dataset indexing and per-image signatures (byte hash and grey thumbnail)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
THUMB_SIZE = 32
_NUMBER = re.compile(r"\((\d+)\)")


def find_class_root(data_dir: Path) -> Path:
    """Return the directory whose sub-folders are the classes (the Kaggle archive nests them)."""
    current = Path(data_dir)
    for _ in range(4):
        subdirs = sorted(p for p in current.iterdir() if p.is_dir() and not p.name.startswith("."))
        has_images = any(f.suffix.lower() in IMAGE_EXTENSIONS for d in subdirs for f in list(d.iterdir())[:50])
        if len(subdirs) > 1 and has_images:
            return current
        if len(subdirs) != 1:
            break
        current = subdirs[0]
    raise FileNotFoundError(f"No class folders with images found under {data_dir}")


def index_images(data_dir: Path) -> pd.DataFrame:
    """One row per image: relative path, integer label, class name and slice number.

    File names look like ``Cyst- (123).jpg``; the number follows slice order within
    the class, which is what lets us recover scans (see grouping.py).
    """
    root = find_class_root(data_dir)
    classes = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    rows = [
        (str(f.relative_to(root)), label, name)
        for label, name in enumerate(classes)
        for f in sorted((root / name).iterdir())
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    df = pd.DataFrame(rows, columns=["path", "label", "cls"])
    numbers = df["path"].str.extract(_NUMBER)[0]
    if numbers.isna().any():
        raise ValueError("every file name must contain its slice number in parentheses, e.g. 'Cyst- (12).jpg'")
    df["slice"] = numbers.astype(int)
    df.attrs["root"] = str(root)
    df.attrs["classes"] = classes
    return df


def file_md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def thumbnail(path: Path, size: int = THUMB_SIZE) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32).ravel()


def image_signatures(df: pd.DataFrame, cache_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """MD5 per file (exact copies) and unit-norm, mean-centred thumbnails (dot product = correlation)."""
    root = Path(df.attrs["root"])
    cache = Path(cache_dir) / f"signatures_{THUMB_SIZE}.npz" if cache_dir else None
    if cache is not None and cache.exists():
        stored = np.load(cache, allow_pickle=False)
        if stored["paths"].tolist() == df["path"].tolist():
            return stored["md5"], stored["thumbs"]
    md5 = np.array([file_md5(root / p) for p in df["path"]])
    thumbs = np.stack([thumbnail(root / p) for p in df["path"]])
    thumbs -= thumbs.mean(axis=1, keepdims=True)
    thumbs /= np.linalg.norm(thumbs, axis=1, keepdims=True) + 1e-8
    thumbs = thumbs.astype(np.float32)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, paths=df["path"].to_numpy(dtype=str), md5=md5, thumbs=thumbs)
    return md5, thumbs
