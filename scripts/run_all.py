#!/usr/bin/env python3
"""Run the whole study in order: stage 1 (benchmark), stage 3 (fine-tuning), stage 2 (comparison report).

    python scripts/run_all.py --data-dir data/ct-kidney     # the full study (hours on a CPU)
    python scripts/run_all.py --smoke                        # every stage on tiny synthetic data (minutes)

Each stage is its own script, so any one can also be re-run alone. Fine-tuning runs
once per entry in FINETUNE_RUNS; the mean of those runs is the pre-declared final model.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Fine-tuning runs of the final study. Settings were chosen among seven pilots on fold-0 validation
# scans only (results/IMPROVEMENTS.md, "Fine-tuning settings tried"). The mean of these runs is the
# pre-declared final model (a two-member deep ensemble).
_COMMON_FT = ["--backbone", "MobileNet", "--stride", "2", "--unfreeze", "60", "--finetune-lr", "5e-5",
              "--finetune-epochs", "15", "--patience", "4"]
FINETUNE_RUNS = [
    ("MobileNet", [*_COMMON_FT, "--img-size", "192"]),       # pilot B settings
    ("MobileNet_224", [*_COMMON_FT, "--img-size", "224"]),   # pilot D settings (best on validation)
]


def stage(script: str, *args: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *args]
    print(f"\n=== {script} {' '.join(args)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    p.add_argument("--threads", default="6")
    p.add_argument("--nice", default="10")
    p.add_argument("--smoke", action="store_true", help="run every stage on synthetic data with tiny settings")
    args = p.parse_args()

    common = ["--threads", args.threads, "--nice", args.nice]
    if args.smoke:
        from kidney_tl.synthetic import make_dataset

        tmp = Path(tempfile.mkdtemp(prefix="kidney_all_"))
        data = make_dataset(tmp / "data")
        res = args.results_dir if args.results_dir != ROOT / "results" else tmp / "results"
        paths = ["--data-dir", str(data), "--results-dir", str(res), "--cache-dir", str(tmp / "cache")]
        tiny = ["--img-size", "32", "--weights", "none", "--folds", "3"]
        models = ["--models", "MobileNet", "ResNet50"]
        stage("run_experiment.py", "--data-dir", str(data), "--output-dir", str(res), "--cache-dir",
              str(tmp / "cache"), *tiny, *models, "--quick", *common)
        for offset, (tag, _) in enumerate(FINETUNE_RUNS):
            stage("run_finetune.py", *paths, *tiny, "--backbone", "MobileNet", "--stride", "2", "--unfreeze", "5",
                  "--head-epochs", "1", "--finetune-epochs", "1", "--patience", "1", "--tag", tag,
                  "--seed-offset", str(offset), *common)
        stage("run_improvements.py", *paths, *tiny, *models, "--bootstrap", "100", *common)
        print(f"\nSmoke run complete: {res}")
        return

    if args.data_dir is None:
        raise SystemExit("--data-dir is required (or use --smoke)")
    paths = ["--data-dir", str(args.data_dir), "--cache-dir", str(args.cache_dir)]
    stage("run_experiment.py", *paths, "--output-dir", str(args.results_dir), *common)
    for tag, settings in FINETUNE_RUNS:
        stage("run_finetune.py", *paths, "--results-dir", str(args.results_dir), *settings, "--tag", tag, *common)
    stage("run_improvements.py", *paths, "--results-dir", str(args.results_dir), *common)
    print(f"\nDone. Reports: {args.results_dir / 'REPORT.md'} and {args.results_dir / 'IMPROVEMENTS.md'}")


if __name__ == "__main__":
    main()
