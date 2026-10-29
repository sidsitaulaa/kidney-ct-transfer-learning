"""Experiment configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

PAPER_MODELS = ("VGG16", "MobileNet", "ResNet50", "InceptionV3", "Xception")


@dataclass
class ExperimentConfig:
    """All settings for one run. Defaults reproduce the version-2 experiment."""

    data_dir: Path
    output_dir: Path = Path("results")
    cache_dir: Path = Path("cache")

    # Backbones
    models: tuple[str, ...] = PAPER_MODELS
    img_size: int = 224
    weights: str | None = "imagenet"
    feature_batch_size: int = 32

    # Scan grouping (see grouping.py)
    scan_break_r: float = 0.9      # adjacent slices correlating below this start a new scan
    duplicate_r: float = 0.995     # images this similar (or byte-identical) mark the same scan stored twice

    # Evaluation
    n_folds: int = 5
    val_frac: float = 0.15         # share of each training fold held out for model selection and early stopping
    seed: int = 42

    # Classifier head
    grid_units: tuple[int, ...] = (256, 512)
    grid_dropout: tuple[float, ...] = (0.5,)
    grid_lr: tuple[float, ...] = (1e-3, 1e-4)
    max_epochs: int = 100
    patience: int = 8
    batch_size: int = 256
    class_weights: bool = True

    bootstrap_samples: int = 2000
    threads: int = 6

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.cache_dir = Path(self.cache_dir)
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        if not 0 < self.val_frac < 0.5:
            raise ValueError("val_frac must be in (0, 0.5)")
        if not 0 < self.scan_break_r < self.duplicate_r <= 1:
            raise ValueError("need 0 < scan_break_r < duplicate_r <= 1")

    def to_json(self, path: Path) -> None:
        payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}
        Path(path).write_text(json.dumps(payload, indent=2))
