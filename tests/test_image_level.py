"""Split, duplicate grouping and metrics for the image-level experiment (scripts/run_image_level.py)."""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from kidney_tl.image_level import (
    assert_split_ok,
    bootstrap_accuracy_ci,
    duplicate_groups,
    evaluate,
    image_level_split,
)


def _toy(n_per_class=(400, 500, 150, 250), n_dup_pairs=60, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.concatenate([np.full(n, c) for c, n in enumerate(n_per_class)])
    md5 = np.array([f"h{i}" for i in range(len(labels))], dtype=object)
    # make byte-identical pairs inside a class (as in the real data)
    for _ in range(n_dup_pairs):
        c = rng.integers(0, len(n_per_class))
        a, b = rng.choice(np.flatnonzero(labels == c), 2, replace=False)
        md5[b] = md5[a]
    return labels, md5


def test_duplicate_groups_share_ids():
    g = duplicate_groups(["x", "y", "x", "z", "y"])
    assert g[0] == g[2] and g[1] == g[4] and len(set(g)) == 3


def test_split_is_disjoint_stratified_and_deterministic():
    labels, md5 = _toy()
    subset = image_level_split(labels, md5, seed=123)
    assert_split_ok(subset, md5)
    for s, frac in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
        assert np.mean(subset == s) == pytest.approx(frac, abs=0.02)
        for c in np.unique(labels):  # stratified: each class keeps the same proportions
            assert np.mean(subset[labels == c] == s) == pytest.approx(frac, abs=0.04)
    assert (image_level_split(labels, md5, seed=123) == subset).all()


def test_duplicates_stay_in_one_subset():
    labels, md5 = _toy(n_dup_pairs=200)
    subset = image_level_split(labels, md5, seed=7)
    for h in np.unique(md5):
        assert len(set(subset[md5 == h])) == 1


def test_assert_split_ok_catches_shared_duplicates():
    md5 = np.array(["a", "a", "b", "c"])
    with pytest.raises(AssertionError):
        assert_split_ok(np.array(["train", "test", "val", "test"]), md5)


def test_split_rejects_duplicates_across_classes():
    with pytest.raises(ValueError):
        image_level_split(np.array([0, 1, 0, 1] * 10), ["same"] * 2 + [f"u{i}" for i in range(38)])


def test_evaluate_uses_argmax_labels():
    y = np.array([0, 0, 1, 1, 2, 3, 3, 3])
    prob = np.full((8, 4), 0.1)
    pred = np.array([0, 1, 1, 1, 2, 3, 3, 0])
    prob[np.arange(8), pred] = 0.7
    m = evaluate(y, prob, ["Cyst", "Normal", "Stone", "Tumor"], n_boot=200)
    assert m["accuracy"] == pytest.approx(accuracy_score(y, pred))
    assert m["macro_f1"] == pytest.approx(f1_score(y, pred, average="macro"))
    assert m["weighted_f1"] == pytest.approx(f1_score(y, pred, average="weighted"))
    assert np.array(m["confusion_matrix"]).sum() == 8 and np.trace(m["confusion_matrix"]) == 6
    assert m["accuracy_ci_low"] <= m["accuracy"] <= m["accuracy_ci_high"]
    assert 0 <= m["roc_auc_macro_ovr"] <= 1
    assert m["Stone_recall"] == 1.0


def test_bootstrap_ci_is_narrow_for_large_samples():
    y = np.zeros(5000, dtype=int)
    pred = np.where(np.arange(5000) < 4500, 0, 1)
    lo, hi = bootstrap_accuracy_ci(y, pred, n_boot=1000, seed=1)
    assert lo < 0.9 < hi and hi - lo < 0.025
