"""The single entry point runs every stage end to end (synthetic data, a few minutes on CPU)."""

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_run_all_smoke(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_all.py"
    out = tmp_path / "results"
    subprocess.run([sys.executable, str(script), "--smoke", "--results-dir", str(out), "--threads", "2", "--nice", "0"],
                   check=True, capture_output=True)
    for name in ("REPORT.md", "IMPROVEMENTS.md", "finetune/MobileNet/test_predictions.csv.gz",
                 "finetune/MobileNet_224/test_predictions.csv.gz", "improvements/pipelines.csv"):
        assert (out / name).exists(), name
    pipelines = set(pd.read_csv(out / "improvements" / "pipelines.csv").pipeline)
    assert {"Fine-tuned MobileNet", "Fine-tuned MobileNet_224", "Fine-tuned ensemble (mean of 2 runs)",
            "Selected per fold on validation (incl. fine-tuned)"} <= pipelines
