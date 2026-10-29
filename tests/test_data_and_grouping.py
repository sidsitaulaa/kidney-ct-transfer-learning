import numpy as np
import pytest

from kidney_tl.config import ExperimentConfig
from kidney_tl.data import image_signatures, index_images
from kidney_tl.grouping import assign_groups, scan_segments


def test_index_finds_nested_class_root(synthetic_dir):
    df = index_images(synthetic_dir.parent.parent)  # two levels above the class folders
    assert df.attrs["classes"] == ["Cyst", "Normal", "Stone", "Tumor"]
    assert len(df) == 4 * (10 + 1) * 4
    assert df.groupby("cls")["slice"].agg(lambda s: sorted(s) == list(range(1, 45))).all()


def test_signatures_are_unit_correlation_vectors_and_cached(synthetic_dir, tmp_path):
    df = index_images(synthetic_dir)
    md5, thumbs = image_signatures(df, tmp_path)
    np.testing.assert_allclose(np.linalg.norm(thumbs, axis=1), 1, atol=1e-5)
    np.testing.assert_allclose(thumbs.mean(axis=1), 0, atol=1e-5)
    md5_again, thumbs_again = image_signatures(df, tmp_path)
    np.testing.assert_array_equal(md5, md5_again)
    np.testing.assert_array_equal(thumbs, thumbs_again)


def test_segments_follow_scans(synthetic_dir):
    df = index_images(synthetic_dir)
    _, thumbs = image_signatures(df)
    seg = scan_segments(df, thumbs, break_r=0.9)
    scan = (df["slice"] - 1) // 4 + 100 * df["label"]
    pairs = set(zip(scan, seg, strict=True))
    assert len(pairs) == scan.nunique() == len(set(seg))  # one-to-one


def test_duplicated_scan_is_merged_and_classes_never_mix(synthetic_dir):
    df = index_images(synthetic_dir)
    md5, thumbs = image_signatures(df)
    groups, info = assign_groups(df, md5, thumbs, break_r=0.9, duplicate_r=0.995)
    first = df["slice"] <= 4
    copy = df["slice"] >= 41
    for label in range(4):
        g = set(groups[(df["label"] == label) & (first | copy)])
        assert len(g) == 1, "the stored-twice scan must form one group"
    assert df.assign(g=groups).groupby("g")["label"].nunique().max() == 1
    assert (info["groups"] == 10).all() and (info["scan_segments"] == 11).all()


@pytest.mark.parametrize("kwargs", [{"n_folds": 1}, {"val_frac": 0.6}, {"scan_break_r": 0.999, "duplicate_r": 0.99}])
def test_config_validation(kwargs, tmp_path):
    with pytest.raises(ValueError):
        ExperimentConfig(data_dir=tmp_path, **kwargs)
