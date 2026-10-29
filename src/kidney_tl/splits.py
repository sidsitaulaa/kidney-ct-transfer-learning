"""Evaluation protocols (cross-validation folds), disjointness checks and leakage measurement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from .grouping import max_similarity

GROUP_CV = "group-level CV"
IMAGE_CV = "image-level CV"


@dataclass(frozen=True)
class Split:
    """Index arrays into the dataset table for one fold of one protocol."""

    protocol: str
    fold: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def subsets(self) -> dict[str, np.ndarray]:
        return {"train": self.train, "val": self.val, "test": self.test}


def group_cv(labels: np.ndarray, groups: np.ndarray, n_folds: int, val_frac: float, seed: int) -> list[Split]:
    """Stratified group k-fold. Each scan group is tested exactly once and never seen in training;
    the validation set is carved out of the training fold by group as well."""
    idx = np.arange(len(labels))
    outer = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    inner_k = max(2, round(1 / val_frac))
    folds = []
    for k, (trval, test) in enumerate(outer.split(idx, labels, groups)):
        inner = StratifiedGroupKFold(n_splits=inner_k, shuffle=True, random_state=seed + k)
        tr, va = next(inner.split(trval, labels[trval], groups[trval]))
        folds.append(Split(GROUP_CV, k, np.sort(trval[tr]), np.sort(trval[va]), np.sort(test)))
    return folds


def image_cv(labels: np.ndarray, n_folds: int, val_frac: float, seed: int) -> list[Split]:
    """Stratified k-fold by image: disjoint by file, but slices of one scan straddle the split."""
    idx = np.arange(len(labels))
    outer = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for k, (trval, test) in enumerate(outer.split(idx, labels)):
        tr, va = train_test_split(trval, test_size=val_frac, stratify=labels[trval], random_state=seed + k)
        folds.append(Split(IMAGE_CV, k, np.sort(tr), np.sort(va), np.sort(test)))
    return folds


def assert_disjoint(split: Split, groups: np.ndarray | None = None) -> None:
    """Raise if any image (or, when given, any scan group) appears in two subsets of a fold."""
    parts = split.subsets()
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if np.intersect1d(parts[a], parts[b]).size:
                raise AssertionError(f"{split.protocol} fold {split.fold}: images shared between {a} and {b}")
            if groups is not None and np.intersect1d(groups[parts[a]], groups[parts[b]]).size:
                raise AssertionError(f"{split.protocol} fold {split.fold}: scan groups shared between {a} and {b}")


def leakage_report(folds: list[Split], groups: np.ndarray, md5: np.ndarray, thumbs: np.ndarray,
                   duplicate_r: float) -> dict:
    """How much of the test data was effectively seen in training, pooled over folds."""
    seen_file, seen_bytes, seen_near, seen_group, n_test = 0, 0, 0, 0, 0
    for f in folds:
        seen_file += np.isin(f.test, f.train).sum()
        seen_bytes += np.isin(md5[f.test], md5[f.train]).sum()
        seen_near += (max_similarity(f.test, f.train, thumbs) >= duplicate_r).sum()
        seen_group += np.isin(groups[f.test], groups[f.train]).sum()
        n_test += len(f.test)
    pct = lambda n: 100 * n / n_test  # noqa: E731
    return {
        "protocol": folds[0].protocol, "folds": len(folds),
        "train_per_fold": int(np.mean([len(f.train) for f in folds])),
        "val_per_fold": int(np.mean([len(f.val) for f in folds])),
        "test_per_fold": int(np.mean([len(f.test) for f in folds])),
        "test_file_in_train_pct": pct(seen_file),
        "test_byte_identical_in_train_pct": pct(seen_bytes),
        "test_near_identical_in_train_pct": pct(seen_near),
        "test_scan_in_train_pct": pct(seen_group),
    }


def write_manifest(folds: list[Split], df: pd.DataFrame, groups: np.ndarray, path: Path) -> None:
    """Which file was in which subset of which fold, so every result can be audited."""
    rows = []
    for f in folds:
        for subset, idx in f.subsets().items():
            rows.append(pd.DataFrame({"fold": f.fold, "subset": subset, "path": df["path"].iloc[idx].to_numpy(),
                                      "class": df["cls"].iloc[idx].to_numpy(), "scan_group": groups[idx]}))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)
