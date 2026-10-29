#!/usr/bin/env python3
"""Choose the input preprocessing and feature depth on validation scans (frozen backbone + logistic regression).

    python scripts/preprocessing_study.py --data-dir data/ct-kidney --backbone MobileNet

For each variant (body crop on/off x last-layer or multi-level features) a logistic regression is
tuned per group-level fold on that fold's validation scans. The decision uses the mean validation
macro-F1 only; pooled test scores are written for information. Output: results/preprocessing_study.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kidney_tl.config import ExperimentConfig  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    p.add_argument("--backbone", default="MobileNet")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--nice", type=int, default=10)
    args = p.parse_args()
    from kidney_tl.utils import limit_resources

    limit_resources(args.threads, args.nice)

    import numpy as np
    import pandas as pd

    from kidney_tl.data import image_signatures, index_images
    from kidney_tl.features import extract_features
    from kidney_tl.grouping import assign_groups
    from kidney_tl.improve import fit_logreg
    from kidney_tl.metrics import fast_accuracy, fast_macro_f1
    from kidney_tl.splits import group_cv
    from kidney_tl.utils import get_logger

    cfg = ExperimentConfig(data_dir=args.data_dir, output_dir=args.results_dir, cache_dir=args.cache_dir)
    log = get_logger(cfg.output_dir / "preprocessing_study.log")
    df = index_images(cfg.data_dir)
    y = df["label"].to_numpy()
    n_cls = len(df.attrs["classes"])
    md5, thumbs = image_signatures(df, cfg.cache_dir)
    groups, _ = assign_groups(df, md5, thumbs, cfg.scan_break_r, cfg.duplicate_r)
    folds = group_cv(y, groups, cfg.n_folds, cfg.val_frac, cfg.seed)

    rows = []
    for variant in ("final", "multilevel"):
        for crop in (False, True):
            feats = extract_features(df.attrs["root"], df["path"].tolist(), args.backbone, cfg.img_size, cfg.weights,
                                     cfg.feature_batch_size, cfg.cache_dir, variant=variant, crop=crop)
            pred = np.full(len(df), -1)
            val_scores = []
            for f in folds:
                val_f1, _, _, p_te = fit_logreg(feats[f.train], y[f.train], feats[f.val], y[f.val], feats[f.test],
                                                 (1e-3, 1e-2, 1e-1, 1.0), n_cls, seed=f.fold)
                pred[f.test] = p_te.argmax(axis=1)
                val_scores.append(val_f1)
            row = {"backbone": args.backbone, "features": variant, "body_crop": crop, "dim": feats.shape[1],
                   "mean_val_macro_f1": float(np.mean(val_scores)),
                   "test_accuracy": fast_accuracy(y, pred), "test_macro_f1": fast_macro_f1(y, pred, n_cls)}
            rows.append(row)
            log.info("%s", row)
    table = pd.DataFrame(rows).sort_values("mean_val_macro_f1", ascending=False)
    table.to_csv(cfg.output_dir / "preprocessing_study.csv", index=False)
    log.info("Chosen on validation: %s", table.iloc[0].to_dict())


if __name__ == "__main__":
    main()
