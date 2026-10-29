#!/usr/bin/env python3
"""Stage 3: fine-tune a backbone under the group-level folds of stage 1.

    python scripts/run_finetune.py --data-dir data/ct-kidney --backbone MobileNet          # all folds
    python scripts/run_finetune.py --data-dir data/ct-kidney --backbone MobileNet --fold 0 # one-fold pilot

Outputs go to results/finetune/<backbone>/: per-fold validation scores, training histories and,
once every fold has run, out-of-fold test predictions that scripts/run_improvements.py picks up.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kidney_tl.config import ExperimentConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    p.add_argument("--backbone", default="MobileNet")
    p.add_argument("--img-size", type=int, default=192)
    p.add_argument("--stride", type=int, default=3, help="use every n-th slice of each scan for training")
    p.add_argument("--unfreeze", type=int, default=30, help="number of top backbone layers to fine-tune")
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--head-epochs", type=int, default=4)
    p.add_argument("--crop", action="store_true", help="crop to the body and pad to a square before resizing")
    p.add_argument("--finetune-epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--tag", help="output sub-folder name (default: the backbone name)")
    p.add_argument("--seed-offset", type=int, default=0, help="change to train another ensemble member")
    p.add_argument("--fold", type=int, nargs="*", help="run only these folds (default: all)")
    p.add_argument("--force", action="store_true", help="retrain folds whose results already exist")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--weights", default="imagenet")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--nice", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from kidney_tl.utils import limit_resources

    limit_resources(args.threads, args.nice)

    import numpy as np
    import pandas as pd

    from kidney_tl.data import image_signatures, index_images
    from kidney_tl.finetune import FinetuneConfig, thin_by_scan, train_and_predict
    from kidney_tl.grouping import assign_groups
    from kidney_tl.metrics import classification_metrics, fast_macro_f1
    from kidney_tl.splits import GROUP_CV, group_cv
    from kidney_tl.utils import get_logger, write_json

    cfg = ExperimentConfig(data_dir=args.data_dir, output_dir=args.results_dir, cache_dir=args.cache_dir,
                           n_folds=args.folds, threads=args.threads)
    ft = FinetuneConfig(backbone=args.backbone, img_size=args.img_size, stride=args.stride, unfreeze=args.unfreeze,
                        finetune_lr=args.finetune_lr, finetune_epochs=args.finetune_epochs, patience=args.patience,
                        head_epochs=args.head_epochs, crop=args.crop)
    weights = None if args.weights.lower() == "none" else args.weights
    out = cfg.output_dir / "finetune" / (args.tag or ft.backbone)
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger(out / "run.log")

    df = index_images(cfg.data_dir)
    classes = df.attrs["classes"]
    y = df["label"].to_numpy()
    slices = df["slice"].to_numpy()
    md5, thumbs = image_signatures(df, cfg.cache_dir)
    groups, _ = assign_groups(df, md5, thumbs, cfg.scan_break_r, cfg.duplicate_r)
    folds = group_cv(y, groups, cfg.n_folds, cfg.val_frac, cfg.seed)
    settings = {**ft.__dict__, "weights": weights}
    config_file = out / "finetune_config.json"
    previous = json.loads(config_file.read_text()) if config_file.exists() else None
    if previous is not None:  # settings added after a run was made take their default values
        previous = {**{k: v for k, v in FinetuneConfig().__dict__.items() if k not in previous}, **previous}
    resume = previous == settings and not args.force
    if previous is not None and previous != settings and not args.force:
        raise SystemExit(f"{out} holds results for different settings; use another --tag or --force")
    write_json(settings, config_file)

    for f in folds:
        if args.fold and f.fold not in args.fold:
            continue
        fold_files = (f"fold{f.fold}.json", f"fold{f.fold}_test_predictions.csv.gz")
        done_before = all((out / name).exists() for name in fold_files)
        if resume and done_before:
            log.info("fold %d: results with identical settings already exist, skipping (use --force to retrain)",
                     f.fold)
            continue
        t0 = time.time()
        train = thin_by_scan(f.train, groups, slices, ft.stride)
        val = thin_by_scan(f.val, groups, slices, ft.stride)  # for early stopping
        log.info("fold %d: %d training and %d validation images (1 in every %d slices), %d test images",
                 f.fold, len(train), len(val), ft.stride, len(f.test))
        p_test, p_val, history = train_and_predict(df.attrs["root"], df["path"].tolist(), y, train, val, f.test, ft,
                                                   len(classes), seed=cfg.seed + f.fold + 1000 * args.seed_offset,
                                                   weights=weights, log=log.info, predict_val=f.val)
        val_f1 = fast_macro_f1(y[f.val], p_val.argmax(1), len(classes))  # full validation set, as in stage 2
        thinned = np.isin(f.val, val)
        val_f1_thinned = fast_macro_f1(y[val], p_val[thinned].argmax(1), len(classes))  # comparable with pilots
        val_frame = pd.DataFrame(p_val, columns=[f"prob_{c}" for c in classes])
        val_frame.insert(0, "path", df["path"].iloc[f.val].to_numpy())
        val_frame.to_csv(out / f"fold{f.fold}_val_predictions.csv.gz", index=False, float_format="%.5f")
        test_m = classification_metrics(y[f.test], p_test.argmax(1), p_test, classes)
        frame = pd.DataFrame(p_test, columns=[f"prob_{c}" for c in classes])
        frame.insert(0, "path", df["path"].iloc[f.test].to_numpy())
        frame.to_csv(out / f"fold{f.fold}_test_predictions.csv.gz", index=False, float_format="%.5f")
        write_json({"fold": f.fold, "val_macro_f1": val_f1, "val_macro_f1_thinned": val_f1_thinned,
                    "test_accuracy": test_m["accuracy"],
                    "test_macro_f1": test_m["macro_f1"], "minutes": (time.time() - t0) / 60, "history": history},
                   out / f"fold{f.fold}.json")
        log.info("fold %d done in %.1f min: val macro-F1 %.4f (full) / %.4f (thinned)", f.fold,
                 (time.time() - t0) / 60, val_f1, val_f1_thinned)

    done = sorted(out.glob("fold*_test_predictions.csv.gz"))
    if len(done) == cfg.n_folds:
        merged = pd.concat([pd.read_csv(p) for p in done], ignore_index=True)
        merged.insert(0, "model", f"Fine-tuned {args.tag or ft.backbone}")
        merged.insert(0, "protocol", GROUP_CV)
        merged.to_csv(out / "test_predictions.csv.gz", index=False, float_format="%.5f")
        val_scores = {json.loads(p.read_text())["fold"]: json.loads(p.read_text())["val_macro_f1"]
                      for p in out.glob("fold*.json")}
        write_json(val_scores, out / "val_macro_f1.json")
        log.info("All %d folds done; out-of-fold predictions written for stage-2 comparison", cfg.n_folds)


if __name__ == "__main__":
    main()
