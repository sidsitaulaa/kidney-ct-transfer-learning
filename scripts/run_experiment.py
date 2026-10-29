#!/usr/bin/env python3
"""Run the full benchmark.

Examples
--------
    python scripts/run_experiment.py --data-dir data/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone
    python scripts/run_experiment.py --smoke            # 2-minute synthetic dry run
    python scripts/run_experiment.py --data-dir ... --threads 0 --nice 0   # dedicated GPU machine (Colab)
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kidney_tl.config import PAPER_MODELS, ExperimentConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, help="folder containing the class sub-folders (searched recursively)")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    p.add_argument("--models", nargs="+", default=list(PAPER_MODELS))
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=6, help="CPU threads for TensorFlow; 0 = no limit")
    p.add_argument("--nice", type=int, default=10, help="lower process priority by this much (0 = unchanged)")
    p.add_argument("--weights", default="imagenet", help="'imagenet' or 'none' (random initialisation)")
    p.add_argument("--quick", action="store_true", help="tiny head search and few epochs (for checks, not results)")
    p.add_argument("--smoke", action="store_true", help="synthetic data, small models, untrained weights")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from kidney_tl.utils import limit_resources

    limit_resources(args.threads, args.nice)  # before TensorFlow builds anything
    from kidney_tl.experiment import run

    if args.smoke:
        from kidney_tl.synthetic import make_dataset

        tmp = Path(tempfile.mkdtemp(prefix="kidney_smoke_"))
        out = args.output_dir if args.output_dir != ROOT / "results" else tmp / "results"
        cfg = ExperimentConfig(
            data_dir=make_dataset(tmp / "data"), output_dir=out, cache_dir=tmp / "cache",
            models=("MobileNet", "ResNet50"), img_size=64, weights=None, n_folds=3, grid_units=(32,),
            grid_lr=(1e-3,), max_epochs=5, patience=2, bootstrap_samples=200, threads=args.threads)
    else:
        if args.data_dir is None:
            raise SystemExit("--data-dir is required (or use --smoke)")
        cfg = ExperimentConfig(data_dir=args.data_dir, output_dir=args.output_dir, cache_dir=args.cache_dir,
                               models=tuple(args.models), img_size=args.img_size, n_folds=args.folds, seed=args.seed,
                               threads=args.threads,
                               weights=None if args.weights.lower() == "none" else args.weights,
                               **(dict(grid_units=(16,), grid_lr=(1e-3,), max_epochs=3, patience=1,
                                       bootstrap_samples=100) if args.quick else {}))
    run(cfg)
    print(f"\nReport written to {cfg.output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
