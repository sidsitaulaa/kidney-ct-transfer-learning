"""run_finetune.py skips folds already trained with identical settings and refuses mismatched settings."""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_finetune.py"


def _run(synthetic_dir, tmp_path, *extra):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--data-dir", str(synthetic_dir), "--results-dir", str(tmp_path / "res"),
         "--cache-dir", str(tmp_path / "cache"), "--img-size", "32", "--weights", "none", "--folds", "3",
         "--stride", "2", "--unfreeze", "5", "--head-epochs", "1", "--finetune-epochs", "1", "--patience", "1",
         "--fold", "0", "--threads", "2", "--nice", "0", *extra], capture_output=True, text=True)


def test_resume_and_mismatch(synthetic_dir, tmp_path):
    first = _run(synthetic_dir, tmp_path)
    assert first.returncode == 0, first.stderr[-2000:]
    second = _run(synthetic_dir, tmp_path)
    assert second.returncode == 0 and "skipping" in second.stdout
    mismatch = _run(synthetic_dir, tmp_path, "--unfreeze", "3")
    assert mismatch.returncode != 0 and "different settings" in mismatch.stderr
