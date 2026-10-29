import numpy as np

from kidney_tl.data import image_signatures, index_images
from kidney_tl.finetune import FinetuneConfig, thin_by_scan, train_and_predict
from kidney_tl.grouping import assign_groups
from kidney_tl.splits import group_cv


def test_thin_by_scan_keeps_every_nth_slice_per_scan():
    idx = np.arange(10)
    groups = np.array([0] * 6 + [1] * 4)
    slices = np.array([6, 5, 4, 3, 2, 1, 10, 11, 12, 13])
    kept = thin_by_scan(idx, groups, slices, stride=3)
    assert set(slices[kept]) == {1, 4, 10, 13}
    np.testing.assert_array_equal(thin_by_scan(idx, groups, slices, stride=1), idx)


def test_finetune_runs_and_returns_probabilities(synthetic_dir):
    df = index_images(synthetic_dir)
    y = df["label"].to_numpy()
    md5, thumbs = image_signatures(df)
    groups, _ = assign_groups(df, md5, thumbs, 0.9, 0.995)
    fold = group_cv(y, groups, n_folds=3, val_frac=0.2, seed=0)[0]
    cfg = FinetuneConfig(backbone="MobileNet", img_size=32, stride=2, unfreeze=5, head_epochs=1, finetune_epochs=1,
                         patience=1, batch_size=16)
    p_test, p_val, history = train_and_predict(df.attrs["root"], df["path"].tolist(), y,
                                               thin_by_scan(fold.train, groups, df["slice"].to_numpy(), 2),
                                               fold.val, fold.test, cfg, 4, seed=0, weights=None,
                                               log=lambda *_: None)
    assert p_test.shape == (len(fold.test), 4) and p_val.shape == (len(fold.val), 4)
    np.testing.assert_allclose(p_test.sum(axis=1), 1, rtol=1e-4)
    assert set(history) == {"head", "finetune"}
