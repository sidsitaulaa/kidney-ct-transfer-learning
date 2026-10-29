import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score

from kidney_tl.metrics import (
    classification_metrics,
    cluster_bootstrap_ci,
    fast_balanced_accuracy,
    fast_macro_f1,
    holm_adjust,
    paired_cluster_bootstrap,
)


def test_perfect_predictions():
    y = np.array([0, 0, 1, 1, 2, 2])
    m = classification_metrics(y, y, np.eye(3)[y], ["a", "b", "c"])
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "cohen_kappa", "mcc", "roc_auc_macro_ovr"):
        assert m[key] == pytest.approx(1.0)
    assert m["a_specificity"] == pytest.approx(1.0)


def test_macro_f1_penalises_a_missed_minority_class():
    y = np.array([0] * 8 + [1] * 2)
    m = classification_metrics(y, np.zeros(10, dtype=int), None, ["major", "minor"])
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["minor_recall"] == 0
    assert m["macro_f1"] < 0.5
    assert "roc_auc_macro_ovr" not in m


def test_fast_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 500)
    p = np.where(rng.random(500) < 0.7, y, rng.integers(0, 4, 500))
    assert fast_macro_f1(y, p, 4) == pytest.approx(f1_score(y, p, average="macro"))
    assert fast_balanced_accuracy(y, p, 4) == pytest.approx(balanced_accuracy_score(y, p))


def test_cluster_bootstrap_is_wider_than_image_bootstrap():
    """Correlated images inside clusters must widen the interval."""
    rng = np.random.default_rng(1)
    clusters = np.repeat(np.arange(40), 25)
    y = np.zeros(1000, dtype=int)
    cluster_correct = rng.random(40) < 0.8
    p = np.where(cluster_correct[clusters], 0, 1)
    acc = lambda a, b: float(np.mean(a == b))  # noqa: E731
    lo_c, hi_c = cluster_bootstrap_ci(y, p, clusters, acc, n_samples=500)
    lo_i, hi_i = cluster_bootstrap_ci(y, p, np.arange(1000), acc, n_samples=500)
    assert lo_c <= acc(y, p) <= hi_c
    assert (hi_c - lo_c) > 2 * (hi_i - lo_i)


def test_paired_bootstrap_detects_a_real_difference_and_not_a_fake_one():
    rng = np.random.default_rng(2)
    clusters = np.repeat(np.arange(100), 10)
    y = rng.integers(0, 2, 1000)
    good = np.where(rng.random(1000) < 0.95, y, 1 - y)
    bad = np.where(rng.random(1000) < 0.6, y, 1 - y)
    acc = lambda a, b: float(np.mean(a == b))  # noqa: E731
    assert paired_cluster_bootstrap(y, good, bad, clusters, acc, n_samples=500)["p_value"] < 0.01
    same = paired_cluster_bootstrap(y, good, good, clusters, acc, n_samples=500)
    assert same["difference"] == 0 and same["p_value"] == 1.0


def test_holm_adjustment():
    np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
