"""End-to-end run on synthetic data with tiny untrained backbones (under a minute on CPU)."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

from kidney_tl.config import ExperimentConfig
from kidney_tl.experiment import run
from kidney_tl.splits import GROUP_CV, IMAGE_CV


def test_full_pipeline(synthetic_dir, tmp_path):
    cfg = ExperimentConfig(
        data_dir=synthetic_dir, output_dir=tmp_path / "results", cache_dir=tmp_path / "cache",
        models=("MobileNet", "ResNet50"), img_size=32, weights=None, n_folds=3, grid_units=(16,),
        grid_lr=(1e-3,), max_epochs=3, patience=1, bootstrap_samples=50, threads=2)
    tables = run(cfg)
    out = cfg.output_dir
    for name in ("REPORT.md", "pooled_metrics.csv", "per_fold_metrics.csv", "bootstrap_ci.csv",
                 "model_comparisons.csv", "leakage_report.csv", "test_predictions.csv.gz",
                 "figures/confusion_group-level.png", "figures/protocol_comparison.png",
                 "splits/group-level_folds.csv"):
        assert (out / name).exists(), name
    pooled = tables["pooled"].set_index(["protocol", "model"])
    assert set(pooled.index.get_level_values(0)) == {GROUP_CV, IMAGE_CV}
    assert (pooled.loc[GROUP_CV, "n_test_images"] == 176).all()      # every image tested once
    assert pooled["accuracy"].between(0, 1).all()
    leak = tables["leakage"].set_index("protocol")
    assert leak.loc[GROUP_CV, "test_scan_in_train_pct"] == 0
    assert leak.loc[IMAGE_CV, "test_scan_in_train_pct"] > 0          # image-level folds share scans
    folds = pd.read_csv(out / "splits/group-level_folds.csv")
    test_rows = folds[folds.subset == "test"]
    assert test_rows["path"].is_unique and len(test_rows) == 176
    assert "Headline" in (out / "REPORT.md").read_text()


def test_stage2_script_runs_on_stage1_outputs(synthetic_dir, tmp_path):
    cfg = ExperimentConfig(
        data_dir=synthetic_dir, output_dir=tmp_path / "results", cache_dir=tmp_path / "cache",
        models=("MobileNet", "ResNet50"), img_size=32, weights=None, n_folds=3, grid_units=(16,),
        grid_lr=(1e-3,), max_epochs=3, patience=1, bootstrap_samples=50, threads=2)
    run(cfg)
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_improvements.py"
    subprocess.run([sys.executable, str(script), "--data-dir", str(synthetic_dir), "--results-dir",
                    str(cfg.output_dir), "--cache-dir", str(cfg.cache_dir), "--models", "MobileNet", "ResNet50",
                    "--img-size", "32", "--weights", "none", "--folds", "3", "--threads", "2", "--nice", "0"],
                   check=True, capture_output=True)
    table = pd.read_csv(cfg.output_dir / "improvements" / "pipelines.csv")
    assert "Selected per fold on validation" in set(table.pipeline)
    assert {"Stage 1 MLP · MobileNet", "Stage 1 MLP · ResNet50"} <= set(table.pipeline)
    assert table.scan_accuracy.between(0, 1).all()
    assert "Nested-selection estimate" in (cfg.output_dir / "IMPROVEMENTS.md").read_text()
