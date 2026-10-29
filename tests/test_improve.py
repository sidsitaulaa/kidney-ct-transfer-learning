import numpy as np
import pytest

from kidney_tl.improve import run_candidates, scan_level
from kidney_tl.splits import group_cv


def test_scan_level_averages_log_probabilities():
    groups = np.array([0, 0, 0, 1, 1])
    y = np.array([1, 1, 1, 0, 0])
    prob = np.array([[0.6, 0.4], [0.2, 0.8], [0.1, 0.9], [0.7, 0.3], [0.4, 0.6]])
    ids, scan_y, scan_pred = scan_level(prob, groups, y)
    np.testing.assert_array_equal(ids, [0, 1])
    np.testing.assert_array_equal(scan_y, [1, 0])
    np.testing.assert_array_equal(scan_pred, [1, 0])


def test_scan_level_rejects_mixed_class_groups():
    with pytest.raises(AssertionError):
        scan_level(np.full((2, 2), 0.5), np.array([0, 0]), np.array([0, 1]))


def test_candidates_cover_every_image_and_select_per_fold():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(60), 5)
    y = np.repeat(rng.integers(0, 3, 60), 5)
    signal = np.eye(3)[y] * 2
    feats = {"A": signal + rng.normal(size=(300, 3)), "B": rng.normal(size=(300, 8))}
    folds = group_cv(y, groups, n_folds=3, val_frac=0.2, seed=0)
    cands = run_candidates(feats, folds, y, 3, cs=(0.1, 1.0), fusion_pca=4, log=lambda *_: None)
    assert set(cands) == {"LogReg · A", "LogReg · B", "LogReg · fusion of all backbones",
                          "Ensemble · mean of backbone LogRegs", "Selected per fold on validation"}
    for c in cands.values():
        assert not np.isnan(c.prob).any()
    informative = (cands["LogReg · A"].prob.argmax(1) == y).mean()
    noise = (cands["LogReg · B"].prob.argmax(1) == y).mean()
    assert informative > noise
    assert len(cands["Selected per fold on validation"].chosen) == 3
