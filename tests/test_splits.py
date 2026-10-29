import numpy as np
import pytest

from kidney_tl.data import image_signatures, index_images
from kidney_tl.grouping import assign_groups
from kidney_tl.splits import Split, assert_disjoint, group_cv, image_cv, leakage_report


@pytest.fixture(scope="module")
def data(synthetic_dir):
    df = index_images(synthetic_dir)
    md5, thumbs = image_signatures(df)
    groups, _ = assign_groups(df, md5, thumbs, 0.9, 0.995)
    return df, md5, thumbs, groups


def _check_partition(folds, n):
    tested = np.concatenate([f.test for f in folds])
    assert np.unique(tested).size == tested.size == n  # every image tested exactly once


def test_group_cv_keeps_scans_together(data):
    df, md5, thumbs, groups = data
    folds = group_cv(df["label"].to_numpy(), groups, n_folds=3, val_frac=0.2, seed=0)
    _check_partition(folds, len(df))
    for f in folds:
        assert_disjoint(f, groups)
        assert set(df["label"].iloc[f.test]) == {0, 1, 2, 3}
    report = leakage_report(folds, groups, md5, thumbs, 0.995)
    assert report["test_scan_in_train_pct"] == 0
    assert report["test_byte_identical_in_train_pct"] == 0
    assert report["test_near_identical_in_train_pct"] == 0


def test_image_cv_leaks_scans(data):
    """Why the group-level protocol exists: image-level folds put sibling slices in train and test."""
    df, md5, thumbs, groups = data
    folds = image_cv(df["label"].to_numpy(), n_folds=3, val_frac=0.2, seed=0)
    _check_partition(folds, len(df))
    for f in folds:
        assert_disjoint(f)
    report = leakage_report(folds, groups, md5, thumbs, 0.995)
    assert report["test_file_in_train_pct"] == 0
    assert report["test_scan_in_train_pct"] > 80
    assert report["test_near_identical_in_train_pct"] > 80


def test_assert_disjoint_detects_overlap():
    with pytest.raises(AssertionError):
        assert_disjoint(Split("bad", 0, np.array([0, 1, 2]), np.array([3]), np.array([2, 4])))
    groups = np.array([0, 0, 1, 1, 2])
    with pytest.raises(AssertionError):
        assert_disjoint(Split("bad", 0, np.array([0]), np.array([2]), np.array([1, 4])), groups)
