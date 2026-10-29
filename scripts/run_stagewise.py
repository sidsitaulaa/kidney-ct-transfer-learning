#!/usr/bin/env python3
"""Equal-budget fine-tuning of all five backbones (final stage + head) under the group-level folds.

    python scripts/run_stagewise.py --data-dir data/ct-kidney                    # all five models
    python scripts/run_stagewise.py --data-dir data/ct-kidney --models MobileNet --fold 0

Every model gets the same input (224 px, body crop), folds, augmentation, optimiser, epochs and
early stopping; only the network differs. Outputs per model in results/stagewise/<model>/;
folds already finished with identical settings are skipped (use --force to retrain).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kidney_tl.config import PAPER_MODELS, ExperimentConfig  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    p.add_argument("--models", nargs="+", default=list(PAPER_MODELS))
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--no-crop", action="store_true", help="resize the whole frame instead of cropping to the body")
    p.add_argument("--stride", type=int, default=2, help="use 1 in every n slices of each scan for training")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold", type=int, nargs="*", help="run only these folds")
    p.add_argument("--weights", default="imagenet")
    p.add_argument("--force", action="store_true")
    p.add_argument("--cache-batch", type=int, default=32, help="images per batch when caching trunk activations")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--nice", type=int, default=10)
    args = p.parse_args()
    from kidney_tl.utils import limit_resources

    limit_resources(args.threads, args.nice)

    import numpy as np
    import pandas as pd

    from kidney_tl.data import image_signatures, index_images
    from kidney_tl.finetune import thin_by_scan
    from kidney_tl.grouping import assign_groups
    from kidney_tl.metrics import classification_metrics, fast_macro_f1
    from kidney_tl.splits import GROUP_CV, group_cv
    from kidney_tl.stagewise import CUTS, StageConfig, cache_trunk, train_fold
    from kidney_tl.utils import get_logger, write_json

    weights = None if args.weights.lower() == "none" else args.weights
    cfg = ExperimentConfig(data_dir=args.data_dir, output_dir=args.results_dir, cache_dir=args.cache_dir,
                           n_folds=args.folds, threads=args.threads)
    sc = StageConfig(img_size=args.img_size, crop=not args.no_crop, stride=args.stride, epochs=args.epochs,
                     patience=args.patience, lr=args.lr)
    log = get_logger(cfg.output_dir / "stagewise" / "run.log")

    df = index_images(cfg.data_dir)
    classes = df.attrs["classes"]
    y, slices = df["label"].to_numpy(), df["slice"].to_numpy()
    md5, thumbs = image_signatures(df, cfg.cache_dir)
    groups, _ = assign_groups(df, md5, thumbs, cfg.scan_break_r, cfg.duplicate_r)
    folds = group_cv(y, groups, cfg.n_folds, cfg.val_frac, cfg.seed)

    for name in args.models:
        out = cfg.output_dir / "stagewise" / name
        out.mkdir(parents=True, exist_ok=True)
        settings = {**sc.__dict__, "model": name, "cut_after": CUTS[name], "weights": weights}
        config_file = out / "config.json"
        previous = json.loads(config_file.read_text()) if config_file.exists() else None
        if previous is not None and previous != settings and not args.force:
            raise SystemExit(f"{out} holds results for different settings; use --force to replace them")
        write_json(settings, config_file)

        t0 = time.time()
        acts = cache_trunk(df.attrs["root"], df["path"].tolist(), name, sc, cfg.cache_dir, weights,
                           batch_size=args.cache_batch, log=log.info)
        log.info("%s: trunk activations %s ready in %.0fs", name, acts.shape, time.time() - t0)
        for f in folds:
            if args.fold and f.fold not in args.fold:
                continue
            fold_files = (out / f"fold{f.fold}.json", out / f"fold{f.fold}_test_predictions.csv.gz")
            if previous == settings and not args.force and all(x.exists() for x in fold_files):
                log.info("%s fold %d: already done with identical settings, skipping", name, f.fold)
                continue
            t1 = time.time()
            train = thin_by_scan(f.train, groups, slices, sc.stride)
            val_es = thin_by_scan(f.val, groups, slices, sc.stride)
            log.info("%s fold %d: %d training, %d validation (early stopping), %d test images",
                     name, f.fold, len(train), len(val_es), len(f.test))
            p_test, p_val, history = train_fold(acts, y, train, val_es, f.val, f.test, name, sc, len(classes),
                                                seed=cfg.seed + f.fold, weights=weights, log=log.info)
            for subset, idx, prob in (("test", f.test, p_test), ("val", f.val, p_val)):
                frame = pd.DataFrame(prob, columns=[f"prob_{c}" for c in classes])
                frame.insert(0, "path", df["path"].iloc[idx].to_numpy())
                frame.to_csv(out / f"fold{f.fold}_{subset}_predictions.csv.gz", index=False, float_format="%.5f")
            val_f1 = fast_macro_f1(y[f.val], p_val.argmax(1), len(classes))
            test_m = classification_metrics(y[f.test], p_test.argmax(1), p_test, classes)
            write_json({"fold": f.fold, "val_macro_f1": val_f1, "test_accuracy": test_m["accuracy"],
                        "test_macro_f1": test_m["macro_f1"], "minutes": (time.time() - t1) / 60,
                        "history": history}, out / f"fold{f.fold}.json")
            log.info("%s fold %d done in %.1f min: val macro-F1 %.4f, test accuracy %.4f", name, f.fold,
                     (time.time() - t1) / 60, val_f1, test_m["accuracy"])

        done = sorted(out.glob("fold*_test_predictions.csv.gz"))
        if len(done) == cfg.n_folds:
            merged = pd.concat([pd.read_csv(x) for x in done], ignore_index=True)
            merged.insert(0, "model", name)
            merged.insert(0, "protocol", GROUP_CV)
            merged.to_csv(out / "test_predictions.csv.gz", index=False, float_format="%.5f")
            vals = {int(json.loads(x.read_text())["fold"]): json.loads(x.read_text())["val_macro_f1"]
                    for x in out.glob("fold*.json")}
            write_json(vals, out / "val_macro_f1.json")
            log.info("%s: all %d folds done (mean val macro-F1 %.4f)", name, cfg.n_folds, np.mean(list(vals.values())))


if __name__ == "__main__":
    main()
