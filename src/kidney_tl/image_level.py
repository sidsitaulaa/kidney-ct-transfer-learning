"""Helpers for the image-level experiment (scripts/run_image_level.py).

The split is by image, so slices of one scan can fall into both training and test.
Nothing here imports TensorFlow, so the functions are cheap to unit-test.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from .metrics import classification_metrics

SUBSETS = ("train", "val", "test")


def duplicate_groups(md5: Sequence[str]) -> np.ndarray:
    """Integer group id per image; byte-identical files (same MD5) share an id."""
    _, inverse = np.unique(np.asarray(md5), return_inverse=True)
    return inverse.astype(np.int64)


def image_level_split(labels: np.ndarray, md5: Sequence[str], fractions=(0.70, 0.15, 0.15),
                      seed: int = 123) -> np.ndarray:
    """Stratified train/val/test split by image, with byte-identical files kept together.

    The unit of splitting is a duplicate group (one file, or a set of byte-identical
    copies), so the exact same file can never be in two subsets. It is still an
    image-level split: neighbouring slices of one scan can land in different subsets.
    Returns an array of subset names, one per image.
    """
    labels = np.asarray(labels)
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    groups = duplicate_groups(md5)
    unit_label = pd.Series(labels).groupby(groups).agg(["first", "nunique"])
    if (unit_label["nunique"] > 1).any():
        raise ValueError("byte-identical files with different class labels")
    units = unit_label.index.to_numpy()
    y = unit_label["first"].to_numpy()
    _, val_frac, test_frac = fractions
    trval, test = train_test_split(units, test_size=test_frac, stratify=y, random_state=seed)
    y_trval = pd.Series(y, index=units).loc[trval].to_numpy()
    train, val = train_test_split(trval, test_size=val_frac / (1 - test_frac), stratify=y_trval,
                                  random_state=seed)
    unit_subset = pd.Series("", index=units, dtype=object)
    unit_subset.loc[train], unit_subset.loc[val], unit_subset.loc[test] = "train", "val", "test"
    return unit_subset.loc[groups].to_numpy()


def assert_split_ok(subset: np.ndarray, md5: Sequence[str]) -> None:
    """Every image in exactly one subset, subsets disjoint, and no MD5 in two subsets."""
    subset = np.asarray(subset)
    if not set(np.unique(subset)) <= set(SUBSETS):
        raise AssertionError(f"unknown subset names: {set(np.unique(subset)) - set(SUBSETS)}")
    idx = {s: np.flatnonzero(subset == s) for s in SUBSETS}
    md5 = np.asarray(md5)
    for i, a in enumerate(SUBSETS):
        for b in SUBSETS[i + 1:]:
            if np.intersect1d(idx[a], idx[b]).size:
                raise AssertionError(f"images shared between {a} and {b}")
            if np.intersect1d(md5[idx[a]], md5[idx[b]]).size:
                raise AssertionError(f"byte-identical files shared between {a} and {b}")
    if sum(len(v) for v in idx.values()) != len(subset):
        raise AssertionError("some images are in no subset")


def evaluate(y_true: np.ndarray, prob: np.ndarray, class_names: Sequence[str], n_boot: int = 1000,
             seed: int = 0) -> dict:
    """Test-set metrics from predicted probabilities. Predictions are the argmax label."""
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    y_pred = prob.argmax(axis=1)
    m = {k: (float(v) if not isinstance(v, int) else v)
         for k, v in classification_metrics(y_true, y_pred, prob, class_names).items()}
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=np.arange(len(class_names)),
                                                 average="weighted", zero_division=0)
    m.update(weighted_precision=float(p), weighted_recall=float(r), weighted_f1=float(f))
    lo, hi = bootstrap_accuracy_ci(y_true, y_pred, n_boot, seed)
    m.update(accuracy_ci_low=lo, accuracy_ci_high=hi, n_test=int(len(y_true)))
    m["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names))).tolist()
    return m


def bootstrap_accuracy_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000, seed: int = 0,
                          alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI for accuracy, resampling test images with replacement."""
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    rng = np.random.default_rng(seed)
    stats = correct[rng.integers(0, len(correct), (n_boot, len(correct)))].mean(axis=1)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
