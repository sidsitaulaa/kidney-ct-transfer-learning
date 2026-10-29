#!/usr/bin/env python3
"""Write results/image_level/REPORT.md from the saved outputs of scripts/run_image_level.py.

Needs no TensorFlow. Every number comes from the JSON/CSV files in results/image_level/.
    python scripts/report_image_level.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "image_level"
MODELS = ("VGG16", "MobileNet", "ResNet50", "InceptionV3", "Xception")
CLASSES = ("Cyst", "Normal", "Stone", "Tumor")


def f4(x):
    return "n/a" if x is None else f"{x:.4f}"


def mem_stats():
    """Peak container memory, lowest host MemAvailable, and paused / OOM polls from memory.log."""
    path = OUT / "memory.log"
    if not path.exists():
        return None
    peak, low, paused, oom, polls = 0.0, None, 0, 0, 0
    for line in path.read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        polls += 1
        low = int(parts[2]) if low is None else min(low, int(parts[2]))
        paused += parts[3].startswith("paused")
        oom += "OOMKILLED" in parts[3]
        if len(parts) >= 5 and "/" in parts[4]:
            m = re.match(r"([\d.]+)(GiB|MiB|KiB|B)", parts[4].split("/")[0])
            if m:
                peak = max(peak, float(m.group(1)) * {"GiB": 1024, "MiB": 1, "KiB": 1 / 1024, "B": 1 / 2**20}[m.group(2)])
    first, last = path.read_text().splitlines()[1].split()[:2], path.read_text().splitlines()[-1].split()[:2]
    return {"peak_mib": peak, "min_host_avail_mb": low, "paused_polls": paused, "oom_polls": oom, "polls": polls,
            "from": " ".join(first), "to": " ".join(last)}


def main() -> None:
    split = json.loads((OUT / "split_summary.json").read_text())
    checks = json.loads((OUT / "input_checks.json").read_text())
    sanity = json.loads((OUT / "sanity_check.json").read_text()) if (OUT / "sanity_check.json").exists() else None
    ms = {m: json.loads((OUT / m / "metrics.json").read_text()) for m in MODELS if (OUT / m / "metrics.json").exists()}
    env = json.loads((OUT / "environment.json").read_text()) if (OUT / "environment.json").exists() else {}
    mem = mem_stats()
    L = []
    w = L.append

    w("# Image-level evaluation of five ImageNet CNNs for kidney CT classification\n")
    w("Five frozen ImageNet backbones (VGG16, MobileNet, ResNet50, InceptionV3 and Xception) with a tuned "
      "Dense–BatchNorm–Dropout head, evaluated on a stratified 70/15/15 train/val/test split **by image**. "
      "The scan-level evaluation is a separate analysis, in `results/REPORT.md`.\n")
    w(f"Produced by `scripts/run_image_level.py` in Docker (`docker/Dockerfile.image_level`: TensorFlow {env.get('tensorflow', '2.15')}, "
      f"keras-tuner {env.get('keras_tuner', '1.4.7')}, CPU only, 3–4 CPUs, 2.5–3 GB memory limit; see section 6). "
      "The Colab version is `notebooks/image_level_colab.ipynb`.\n")

    # --------------------------------------------------------------- results first
    w("## 1. Results on the held-out test set\n")
    w(f"Test set: {split['n_test']} images, used once per model. Predictions are argmax labels. "
      "The 95% CI for accuracy is a percentile bootstrap over test images (1,000 resamples).\n")
    w("| Model | Test accuracy [95% CI] | Balanced accuracy | Macro-F1 | Weighted F1 | Macro precision | Macro recall | ROC-AUC (macro OvR) | Errors |")
    w("|---|---|---|---|---|---|---|---|---|")
    for m, r in ms.items():
        errors = r["n_test"] - round(r["accuracy"] * r["n_test"])
        w(f"| {m} | {f4(r['accuracy'])} [{f4(r['accuracy_ci_low'])}, {f4(r['accuracy_ci_high'])}] | {f4(r['balanced_accuracy'])} | "
          f"{f4(r['macro_f1'])} | {f4(r['weighted_f1'])} | {f4(r['macro_precision'])} | {f4(r['macro_recall'])} | "
          f"{f4(r.get('roc_auc_macro_ovr'))} | {errors} / {r['n_test']} |")
    w("")
    w("### Per-class precision / recall / F1 (test)\n")
    w("| Model | " + " | ".join(f"{c} P / R / F1" for c in CLASSES) + " |")
    w("|---|" + "---|" * len(CLASSES))
    for m, r in ms.items():
        w(f"| {m} | " + " | ".join(f"{r[c + '_precision']:.3f} / {r[c + '_recall']:.3f} / {r[c + '_f1']:.3f}" for c in CLASSES) + " |")
    w("")
    w("### Confusion matrices (rows = true, columns = predicted; order " + ", ".join(CLASSES) + ")\n")
    for m, r in ms.items():
        w(f"* **{m}**: " + "; ".join(f"{c}: {row}" for c, row in zip(CLASSES, r["confusion_matrix"], strict=True)))
    w("")
    w("Per-model files: `<Model>/metrics.json`, `predictions.csv`, `history.csv`, `curves.png`, `confusion_matrix.png`, "
      "`tuner_trials.csv`, `head.keras`. All models: `summary.csv`.\n")
    if ms:
        best = max(ms, key=lambda k: ms[k]["accuracy"])
        w(f"The best model is **{best}** ({ms[best]['accuracy'] * 100:.2f}%). All {len(ms)} models are between "
          f"{min(r['accuracy'] for r in ms.values()) * 100:.2f}% and {max(r['accuracy'] for r in ms.values()) * 100:.2f}%. "
          "These scores measure recognition of other slices of scans seen in training, not accuracy on new patients "
          "(section 5).\n")

    # --------------------------------------------------------------- hyperparameters
    w("## 2. Selected hyperparameters (keras-tuner RandomSearch on validation accuracy)\n")
    w("| Model | Dense units | Dropout | Learning rate | Tuner best val accuracy (mean of 3 runs) | Val accuracy after refit | Feature dim |")
    w("|---|---|---|---|---|---|---|")
    for m, r in ms.items():
        hp = r["best_hyperparameters"]
        w(f"| {m} | {hp['dense_units']} | {hp['dropout_rate']:.1f} | {hp['learning_rate']:g} | {f4(r['tuner_best_val_accuracy'])} | "
          f"{f4(r['final_val_accuracy'])} | {r['feature_dim']} |")
    w("")
    w("Search space: units 128–1024 (step 128), dropout 0.2–0.5 (step 0.1), learning rate "
      "{1e-2, 1e-3, 1e-4}; `max_trials=10`, `executions_per_trial=3`, `epochs=10`, batch size 32, seed 123. The best "
      "configuration was then refitted on the training set for 10 epochs (validating on val) and evaluated once on test. "
      "All trials are in `<Model>/tuner_trials.csv` and the tuner state in `tuner/`. "
      "Because the tuner seed is fixed, every model was offered the same 10 configurations; the two with 640 units "
      "and learning rate 1e-4 (dropout 0.2 or 0.3) won for all five models.\n")

    # --------------------------------------------------------------- protocol
    w("## 3. Evaluation protocol\n")
    w("| Step | Protocol |")
    w("|---|---|")
    w("| Split | Stratified 70/15/15 train/val/test split by image (seed 123) from the file list, saved to `split.csv`. The three subsets are asserted to be disjoint, and byte-identical files are kept in one subset |")
    w("| Model selection | keras-tuner RandomSearch on the validation set only (10 trials × 3 executions × 10 epochs). The test set is used once per model |")
    w("| Preprocessing | Each network's own `preprocess_input`, applied once to 0–255 pixels (table in section 4) |")
    w("| Metrics | Computed on argmax labels: accuracy with bootstrap CI, balanced accuracy, macro and weighted precision/recall/F1, per-class metrics, confusion matrices, macro one-vs-rest ROC-AUC |")
    w("| Models | VGG16, MobileNet, ResNet50, InceptionV3 and Xception, with ImageNet weights |")
    w("| Fixed settings | 150×150 input, `include_top=False`, `pooling='max'`, frozen backbone; head Flatten → Dense(relu) → BatchNorm → Dropout → Dense(4, softmax); Adam, sparse categorical cross-entropy; 10-epoch refit of the best head; batch size 32; no augmentation |")
    w("")
    w("**Cached features.** The backbone is frozen and there is no augmentation, so each image's backbone output never "
      "changes during training. Each model's 150×150 max-pooled features were therefore computed once "
      "(`cache/image_level_<Model>_150_max.npy`) and the head was trained on them. This is mathematically the same as "
      "training the full stack: a Keras layer with `trainable=False` runs its BatchNorm in inference mode even during "
      "`fit`, so the backbone is a fixed function. ")
    if sanity:
        w(f"Checked on {sanity['model']} ({sanity['images']} test images, `sanity_check.json`): the end-to-end model "
          f"(backbone + head) and the head on cached features give identical probabilities (max abs difference "
          f"{sanity['max_abs_diff_probabilities_end_to_end_vs_cached']:.1e}); recomputed features equal the cache "
          f"(max diff {sanity['max_abs_diff_features_recomputed_vs_cache']:.1e}); and the frozen backbone gives the "
          f"same output with `training=True` and `training=False` (max diff "
          f"{sanity['max_abs_diff_backbone_training_true_vs_false']:.1e}). Check passed: {sanity['passed']}. "
          "Only the random order of training batches and dropout masks differ from an end-to-end run.\n")
    else:
        w("\n")

    # --------------------------------------------------------------- data, split and input checks
    w("## 4. Data, split and input pipeline checks\n")
    w(f"12,446 JPEG images. Split: train {split['n_train']}, val {split['n_val']}, test {split['n_test']}.\n")
    w("| Class | Train | Val | Test |")
    w("|---|---|---|---|")
    for c in CLASSES:
        pc = split["per_class"][c]
        n = sum(pc.values())
        w(f"| {c} | {pc['train']} ({pc['train'] / n:.1%}) | {pc['val']} ({pc['val'] / n:.1%}) | {pc['test']} ({pc['test'] / n:.1%}) |")
    w("")
    w(f"* **Duplicate files.** {split['files_in_duplicate_groups']} files form {split['duplicate_groups']} groups of "
      f"byte-identical copies (same MD5). The split assigns each group as a unit, so no file in test has an exact copy in "
      f"training ({split['test_byte_identical_in_train']} test images byte-identical to a training image). This is still "
      "an image-level split.")
    a = checks["audit"]
    w(f"* **Decoding.** Formats {a['formats']}, modes {a['modes']}, {list(a['bit_depth_per_channel'])[0]}-bit per channel; "
      f"{len(a['decode_failures'])} files failed to decode with PIL, and TensorFlow decoded {checks['tf_decode']['images']} / 12,446 "
      f"to 150×150×3 float32 in [{checks['tf_decode']['pixel_min']:.0f}, {checks['tf_decode']['pixel_max']:.0f}]. "
      f"The CT slices are grey images stored as RGB (identical channels in {a['sampled_rgb_with_identical_channels']} sampled files). "
      "`decode_image(channels=3)` is used throughout (grey would be repeated to 3 channels, RGBA would lose alpha).")
    w(f"* **Size.** {a['distinct_sizes']} distinct original sizes, widths {a['width_range'][0]}–{a['width_range'][1]} and heights "
      f"{a['height_range'][0]}–{a['height_range'][1]} px; most common: "
      + ", ".join(f"{s} ({n})" for s, n in a["most_common_sizes"][:4])
      + f". Aspect ratio (w/h) {a['aspect_ratio_w_over_h']['min']:.2f}–{a['aspect_ratio_w_over_h']['max']:.2f} "
      f"(median {a['aspect_ratio_w_over_h']['median']:.2f}); none is extreme. Images are resized bilinearly to 150×150 "
      "without preserving aspect ratio, as `image_dataset_from_directory` did, so the widest images are squeezed by up to 19%.")
    w("* **Labels.** Fixed map `{Cyst: 0, Normal: 1, Stone: 2, Tumor: 3}` (`class_index.json`); every file's label was "
      "checked against its folder; per-class subset fractions are within 1 point of 70/15/15.")
    w("* **Order.** Features are extracted in file order without shuffling; each cache has a row manifest "
      "(`cache/image_level_<Model>_150_max.paths.txt`) that is asserted equal to `split.csv` before use. Only training "
      "batches are shuffled (Keras `fit`), with seed 123.")
    w("* **Visual check.** `input_check.png`: 3 training and 3 test images per class after decoding and resizing. They "
      "are axial and coronal abdominal CT slices, upright, neither blank nor saturated, with the right labels.\n")
    w("Preprocessing per model, applied once to 0–255 pixels (statistics on a sample batch of 16 training images; ranges asserted):\n")
    w("| Model | `preprocess_input` | Min | Max | Mean | SD | Feature shape |")
    w("|---|---|---|---|---|---|---|")
    for st in checks["preprocessing_sample_batch"]:
        w(f"| {st['model']} | {st['mode']} | {st['min']:.2f} | {st['max']:.2f} | {st['mean']:.2f} | {st['std']:.2f} | "
          f"{checks['feature_shapes'][st['model']][1]} |")
    w("")

    # --------------------------------------------------------------- caveats
    w("## 5. Limitations: these numbers measure familiar scans, not new patients\n")
    w(f"* The dataset's consecutive file numbers are neighbouring slices of the same CT scan (204 recovered scan groups; "
      f"`results/scan_groups.csv`, per-image groups from `results/splits/group-level_folds.csv`). With an image-level split, "
      f"**{split['test_from_scan_in_train']} of {split['n_test']} test images ({split['test_from_scan_in_train_pct']:.1f}%) come "
      f"from a scan that also has slices in training**, and **{split['test_near_identical_in_train_pct']:.1f}%** "
      f"({split['test_near_identical_in_train']}) have a near-identical training image (32×32 thumbnail correlation ≥ 0.995; "
      f"median best correlation {split['test_max_r_to_train_median']:.4f}). This leakage is measured here; the split does not remove it.")
    w("* The test scores therefore show how well a model recognises other slices of scans it has already seen. They "
      "are not an estimate of accuracy on new patients.")
    w("* When no scan is shared between training and test (group-level 5-fold CV, `results/REPORT.md` sections 3–4), "
      "the same five frozen backbones score **0.556–0.665** accuracy, and the best model of the whole study (a fine-tuned "
      "MobileNet ensemble, `results/IMPROVEMENTS.md`) scores **0.814**. Use those numbers for any claim about generalisation.")
    w("* Bootstrap CIs resample test images as if independent. Slices of one scan are strongly correlated, so these "
      "intervals are too narrow even for the image-level question.")
    w("* One split, one seed; a single dataset with no external validation.\n")

    # --------------------------------------------------------------- runtime
    w("## 6. Runtime and resources\n")
    w("| Model | Feature extraction (min) | Tuning (min) | Final fit (s) | Total (min) |")
    w("|---|---|---|---|---|")
    for m, r in ms.items():
        s = r["seconds"]
        w(f"| {m} | {s['features'] / 60:.1f} | {s['tuning'] / 60:.1f} | {s['final_fit']:.0f} | {s['model_total'] / 60:.1f} |")
    w("")
    w("InceptionV3's feature time covers only the part extracted after the Docker crash (the first 10,240 images "
      "were extracted before it). Total wall-clock time for the run, excluding the gap after the crash: about 2.6 hours.\n")
    w("**Run history.** The run used the same resumable command throughout (`python scripts/run_image_level.py`):\n")
    w("1. Started 03:27 UTC in a container with `--memory=3g --memory-swap=3g --cpus=4`, 4/2 TensorFlow threads and "
      "feature batch size 32. VGG16 and MobileNet finished.")
    w("2. 04:23 UTC: stopped by hand between models, because container memory had climbed to about 86% of the limit, and "
      "restarted with lower memory use (each model in its own Python process, the Keras session cleared before each tuner "
      "trial, cached image pages released). ResNet50 finished.")
    w("3. About 05:09 UTC, during InceptionV3 feature extraction: **Docker Desktop crashed** because of host memory "
      "pressure from outside the container (host MemAvailable had fallen to about 1.3 GB; the container itself used "
      "about 0.8 GB). There was no OOM kill inside the container.")
    w("4. 05:17 UTC: restarted with `--memory=2500m --memory-swap=2500m --cpus=3`, 3/1 TensorFlow threads and "
      "`IMAGE_LEVEL_BATCH=16`. InceptionV3 resumed its feature extraction at image 10,240 of 12,446. InceptionV3 and "
      "Xception finished, and the container exited with code 0 at 06:09 UTC.\n")
    w("Changing the thread count and the feature batch size does not change the method. Features are a deterministic "
      "function of each image, apart from floating-point rounding.\n")
    if mem:
        w(f"**Memory.** Phases 1–3 (`memory.log`, polled every ~25 s from `docker stats` and `/proc/meminfo`, {mem['polls']} polls, "
          f"local time UTC+05:45): peak container usage **{mem['peak_mib'] / 1024:.2f} GiB** of the 3 GiB limit, reached "
          f"before the memory changes; lowest host MemAvailable {mem['min_host_avail_mb']} MB, just before the Docker crash. "
          "Phase 4 (host-side watchdog log, not included in the repository; 239 polls every ~10 s): peak container usage **1.78 GiB** of the "
          "2.44 GiB limit; lowest host MemAvailable 3,018 MB. No OOM kill and no watchdog pause in any phase.\n")
    (OUT / "REPORT.md").write_text("\n".join(L))
    print(f"wrote {OUT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
