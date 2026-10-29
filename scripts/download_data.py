#!/usr/bin/env python3
"""Download the CT KIDNEY DATASET (Normal / Cyst / Tumor / Stone, 12,446 images) into ./data.

Sources
-------
kaggle (default, canonical)
    nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone, via kagglehub. Public, but
    anonymous downloads can be slow; a Kaggle API token in ~/.kaggle/kaggle.json helps.
huggingface (fast mirror)
    mhmad240/kidney-ct-classification: the same 12,446 images stored as parquet. They are
    written back to <class>/<file> folders. The script checks the class counts against the
    published Kaggle counts and refuses to continue if they differ.

Usage
-----
    python scripts/download_data.py                       # Kaggle
    python scripts/download_data.py --source huggingface  # mirror
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_ID = "nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone"
HF_ID = "mhmad240/kidney-ct-classification"
HF_LABEL_NAMES = ("Cyst", "Normal", "Stone", "Tumor")  # ClassLabel order on the mirror's dataset card
EXPECTED_COUNTS = {"Cyst": 3709, "Normal": 5077, "Stone": 1377, "Tumor": 2283}  # Kaggle dataset card


def from_kaggle(target: Path) -> Path:
    os.environ.setdefault("KAGGLEHUB_CACHE", str(ROOT / "data" / ".kagglehub"))
    import kagglehub

    path = Path(kagglehub.dataset_download(KAGGLE_ID))
    if not target.exists():
        try:
            target.symlink_to(path, target_is_directory=True)
        except OSError:
            shutil.copytree(path, target)
    return target


def from_huggingface(target: Path) -> Path:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    from PIL import Image

    files = sorted(f for f in HfApi().list_repo_files(HF_ID, repo_type="dataset") if f.endswith(".parquet"))
    tmp = target.with_name(target.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    counts: Counter[str] = Counter()
    for remote in files:
        local = hf_hub_download(HF_ID, remote, repo_type="dataset", cache_dir=str(ROOT / "data" / ".hf_cache"))
        table = pq.ParquetFile(local)
        for batch in table.iter_batches(batch_size=256, columns=["image", "label"]):
            for image, label in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist(), strict=True):
                cls = HF_LABEL_NAMES[label]
                name = Path(image.get("path") or f"{cls}-{counts[cls]}.jpg").name
                out = tmp / cls / name
                out.parent.mkdir(parents=True, exist_ok=True)
                if out.exists():
                    raise RuntimeError(f"duplicate file name in mirror: {cls}/{name}")
                data = image["bytes"]
                if Image.open(io.BytesIO(data)).format in ("JPEG", "PNG"):
                    out.write_bytes(data)  # keep the original encoded bytes
                else:
                    Image.open(io.BytesIO(data)).convert("RGB").save(out.with_suffix(".jpg"), quality=95)
                counts[cls] += 1
        print(f"  {remote}: {sum(counts.values())} images so far")
    if dict(counts) != EXPECTED_COUNTS:
        raise RuntimeError(f"class counts {dict(counts)} differ from the Kaggle dataset {EXPECTED_COUNTS}")
    shutil.rmtree(target, ignore_errors=True)
    tmp.rename(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("kaggle", "huggingface"), default="kaggle")
    args = parser.parse_args()
    target = ROOT / "data" / "ct-kidney"
    target.parent.mkdir(parents=True, exist_ok=True)
    path = from_kaggle(target) if args.source == "kaggle" else from_huggingface(target)
    print(f"Dataset ready at {path}")


if __name__ == "__main__":
    main()
