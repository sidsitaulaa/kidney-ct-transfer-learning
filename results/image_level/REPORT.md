# Image-level evaluation of five ImageNet CNNs for kidney CT classification

Five frozen ImageNet backbones (VGG16, MobileNet, ResNet50, InceptionV3 and Xception) with a tuned Dense–BatchNorm–Dropout head, evaluated on a stratified 70/15/15 train/val/test split **by image**. The scan-level evaluation is a separate analysis, in `results/REPORT.md`.

Produced by `scripts/run_image_level.py` in Docker (`docker/Dockerfile.image_level`: TensorFlow 2.15.0, keras-tuner 1.4.7, CPU only, 3–4 CPUs, 2.5–3 GB memory limit; see section 6). The Colab version is `notebooks/image_level_colab.ipynb`.

## 1. Results on the held-out test set

Test set: 1853 images, used once per model. Predictions are argmax labels. The 95% CI for accuracy is a percentile bootstrap over test images (1,000 resamples).

| Model | Test accuracy [95% CI] | Balanced accuracy | Macro-F1 | Weighted F1 | Macro precision | Macro recall | ROC-AUC (macro OvR) | Errors |
|---|---|---|---|---|---|---|---|---|
| VGG16 | 0.9984 [0.9962, 1.0000] | 0.9973 | 0.9981 | 0.9984 | 0.9989 | 0.9973 | 1.0000 | 3 / 1853 |
| MobileNet | 0.9973 [0.9946, 0.9995] | 0.9968 | 0.9958 | 0.9973 | 0.9948 | 0.9968 | 1.0000 | 5 / 1853 |
| ResNet50 | 0.9989 [0.9973, 1.0000] | 0.9981 | 0.9986 | 0.9989 | 0.9992 | 0.9981 | 1.0000 | 2 / 1853 |
| InceptionV3 | 0.9870 [0.9811, 0.9919] | 0.9878 | 0.9827 | 0.9871 | 0.9780 | 0.9878 | 0.9997 | 24 / 1853 |
| Xception | 0.9930 [0.9887, 0.9968] | 0.9904 | 0.9909 | 0.9930 | 0.9914 | 0.9904 | 0.9999 | 13 / 1853 |

### Per-class precision / recall / F1 (test)

| Model | Cyst P / R / F1 | Normal P / R / F1 | Stone P / R / F1 | Tumor P / R / F1 |
|---|---|---|---|---|
| VGG16 | 0.998 / 1.000 / 0.999 | 0.997 / 1.000 / 0.999 | 1.000 / 0.995 / 0.998 | 1.000 / 0.994 / 0.997 |
| MobileNet | 0.998 / 0.996 / 0.997 | 1.000 / 0.999 / 0.999 | 0.981 / 0.995 / 0.988 | 1.000 / 0.997 / 0.999 |
| ResNet50 | 0.998 / 1.000 / 0.999 | 0.999 / 1.000 / 0.999 | 1.000 / 0.995 / 0.998 | 1.000 / 0.997 / 0.999 |
| InceptionV3 | 0.996 / 0.989 / 0.993 | 0.997 / 0.982 / 0.989 | 0.935 / 0.981 / 0.958 | 0.983 / 1.000 / 0.991 |
| Xception | 0.989 / 0.998 / 0.994 | 0.996 / 0.993 / 0.995 | 0.981 / 0.976 / 0.978 | 1.000 / 0.994 / 0.997 |

### Confusion matrices (rows = true, columns = predicted; order Cyst, Normal, Stone, Tumor)

* **VGG16**: Cyst: [546, 0, 0, 0]; Normal: [0, 757, 0, 0]; Stone: [1, 0, 206, 0]; Tumor: [0, 2, 0, 341]
* **MobileNet**: Cyst: [544, 0, 2, 0]; Normal: [0, 756, 1, 0]; Stone: [1, 0, 206, 0]; Tumor: [0, 0, 1, 342]
* **ResNet50**: Cyst: [546, 0, 0, 0]; Normal: [0, 757, 0, 0]; Stone: [1, 0, 206, 0]; Tumor: [0, 1, 0, 342]
* **InceptionV3**: Cyst: [540, 1, 4, 1]; Normal: [1, 743, 10, 3]; Stone: [1, 1, 203, 2]; Tumor: [0, 0, 0, 343]
* **Xception**: Cyst: [545, 0, 1, 0]; Normal: [2, 752, 3, 0]; Stone: [2, 3, 202, 0]; Tumor: [2, 0, 0, 341]

Per-model files: `<Model>/metrics.json`, `predictions.csv`, `history.csv`, `curves.png`, `confusion_matrix.png`, `tuner_trials.csv`, `head.keras`. All models: `summary.csv`.

The best model is **ResNet50** (99.89%). All 5 models are between 98.70% and 99.89%. These scores measure recognition of other slices of scans seen in training, not accuracy on new patients (section 5).

## 2. Selected hyperparameters (keras-tuner RandomSearch on validation accuracy)

| Model | Dense units | Dropout | Learning rate | Tuner best val accuracy (mean of 3 runs) | Val accuracy after refit | Feature dim |
|---|---|---|---|---|---|---|
| VGG16 | 640 | 0.3 | 0.0001 | 0.9989 | 0.9984 | 512 |
| MobileNet | 640 | 0.3 | 0.0001 | 0.9956 | 0.9984 | 1024 |
| ResNet50 | 640 | 0.2 | 0.0001 | 0.9982 | 0.9989 | 2048 |
| InceptionV3 | 640 | 0.3 | 0.0001 | 0.9897 | 0.9878 | 2048 |
| Xception | 640 | 0.2 | 0.0001 | 0.9954 | 0.9931 | 2048 |

Search space: units 128–1024 (step 128), dropout 0.2–0.5 (step 0.1), learning rate {1e-2, 1e-3, 1e-4}; `max_trials=10`, `executions_per_trial=3`, `epochs=10`, batch size 32, seed 123. The best configuration was then refitted on the training set for 10 epochs (validating on val) and evaluated once on test. All trials are in `<Model>/tuner_trials.csv` and the tuner state in `tuner/`. Because the tuner seed is fixed, every model was offered the same 10 configurations; the two with 640 units and learning rate 1e-4 (dropout 0.2 or 0.3) won for all five models.

## 3. Evaluation protocol

| Step | Protocol |
|---|---|
| Split | Stratified 70/15/15 train/val/test split by image (seed 123) from the file list, saved to `split.csv`. The three subsets are asserted to be disjoint, and byte-identical files are kept in one subset |
| Model selection | keras-tuner RandomSearch on the validation set only (10 trials × 3 executions × 10 epochs). The test set is used once per model |
| Preprocessing | Each network's own `preprocess_input`, applied once to 0–255 pixels (table in section 4) |
| Metrics | Computed on argmax labels: accuracy with bootstrap CI, balanced accuracy, macro and weighted precision/recall/F1, per-class metrics, confusion matrices, macro one-vs-rest ROC-AUC |
| Models | VGG16, MobileNet, ResNet50, InceptionV3 and Xception, with ImageNet weights |
| Fixed settings | 150×150 input, `include_top=False`, `pooling='max'`, frozen backbone; head Flatten → Dense(relu) → BatchNorm → Dropout → Dense(4, softmax); Adam, sparse categorical cross-entropy; 10-epoch refit of the best head; batch size 32; no augmentation |

**Cached features.** The backbone is frozen and there is no augmentation, so each image's backbone output never changes during training. Each model's 150×150 max-pooled features were therefore computed once (`cache/image_level_<Model>_150_max.npy`) and the head was trained on them. This is mathematically the same as training the full stack: a Keras layer with `trainable=False` runs its BatchNorm in inference mode even during `fit`, so the backbone is a fixed function. 
Checked on MobileNet (64 test images, `sanity_check.json`): the end-to-end model (backbone + head) and the head on cached features give identical probabilities (max abs difference 0.0e+00); recomputed features equal the cache (max diff 0.0e+00); and the frozen backbone gives the same output with `training=True` and `training=False` (max diff 0.0e+00). Check passed: True. Only the random order of training batches and dropout masks differ from an end-to-end run.

## 4. Data, split and input pipeline checks

12,446 JPEG images. Split: train 8713, val 1880, test 1853.

| Class | Train | Val | Test |
|---|---|---|---|
| Cyst | 2599 (70.1%) | 564 (15.2%) | 546 (14.7%) |
| Normal | 3552 (70.0%) | 768 (15.1%) | 757 (14.9%) |
| Stone | 964 (70.0%) | 206 (15.0%) | 207 (15.0%) |
| Tumor | 1598 (70.0%) | 342 (15.0%) | 343 (15.0%) |

* **Duplicate files.** 1034 files form 517 groups of byte-identical copies (same MD5). The split assigns each group as a unit, so no file in test has an exact copy in training (0 test images byte-identical to a training image). This is still an image-level split.
* **Decoding.** Formats {'JPEG': 12446}, modes {'RGB': 12446}, 8-bit per channel; 0 files failed to decode with PIL, and TensorFlow decoded 12446 / 12,446 to 150×150×3 float32 in [0, 255]. The CT slices are grey images stored as RGB (identical channels in 249 of 249 sampled files). `decode_image(channels=3)` is used throughout (grey would be repeated to 3 channels, RGBA would lose alpha).
* **Size.** 77 distinct original sizes, widths 512–1371 and heights 451–1110 px; most common: 512x512 (7709), 798x646 (240), 804x651 (179), 768x622 (177). Aspect ratio (w/h) 1.00–1.24 (median 1.00); none is extreme. Images are resized bilinearly to 150×150 without preserving aspect ratio, as `image_dataset_from_directory` did, so the widest images are squeezed by up to 19%.
* **Labels.** Fixed map `{Cyst: 0, Normal: 1, Stone: 2, Tumor: 3}` (`class_index.json`); every file's label was checked against its folder; per-class subset fractions are within 1 point of 70/15/15.
* **Order.** Features are extracted in file order without shuffling; each cache has a row manifest (`cache/image_level_<Model>_150_max.paths.txt`) that is asserted equal to `split.csv` before use. Only training batches are shuffled (Keras `fit`), with seed 123.
* **Visual check.** `input_check.png`: 3 training and 3 test images per class after decoding and resizing. They are axial and coronal abdominal CT slices, upright, neither blank nor saturated, with the right labels.

Preprocessing per model, applied once to 0–255 pixels (statistics on a sample batch of 16 training images; ranges asserted):

| Model | `preprocess_input` | Min | Max | Mean | SD | Feature shape |
|---|---|---|---|---|---|---|
| VGG16 | caffe: RGB->BGR, subtract ImageNet mean per channel | -123.68 | 151.06 | -79.97 | 60.59 | 512 |
| MobileNet | tf: x/127.5 - 1 -> [-1, 1] | -1.00 | 1.00 | -0.73 | 0.47 | 1024 |
| ResNet50 | caffe: RGB->BGR, subtract ImageNet mean per channel | -123.68 | 151.06 | -79.97 | 60.59 | 2048 |
| InceptionV3 | tf: x/127.5 - 1 -> [-1, 1] | -1.00 | 1.00 | -0.73 | 0.47 | 2048 |
| Xception | tf: x/127.5 - 1 -> [-1, 1] | -1.00 | 1.00 | -0.73 | 0.47 | 2048 |

## 5. Limitations: these numbers measure familiar scans, not new patients

* The dataset's consecutive file numbers are neighbouring slices of the same CT scan (204 recovered scan groups; `results/scan_groups.csv`, per-image groups from `results/splits/group-level_folds.csv`). With an image-level split, **1852 of 1853 test images (99.9%) come from a scan that also has slices in training**, and **97.4%** (1805) have a near-identical training image (32×32 thumbnail correlation ≥ 0.995; median best correlation 0.9992). This leakage is measured here; the split does not remove it.
* The test scores therefore show how well a model recognises other slices of scans it has already seen. They are not an estimate of accuracy on new patients.
* When no scan is shared between training and test (group-level 5-fold CV, `results/REPORT.md` sections 3–4), the same five frozen backbones score **0.556–0.665** accuracy, and the best model of the whole study (a fine-tuned MobileNet ensemble, `results/IMPROVEMENTS.md`) scores **0.814**. Use those numbers for any claim about generalisation.
* Bootstrap CIs resample test images as if independent. Slices of one scan are strongly correlated, so these intervals are too narrow even for the image-level question.
* One split, one seed; a single dataset with no external validation.

## 6. Runtime and resources

| Model | Feature extraction (min) | Tuning (min) | Final fit (s) | Total (min) |
|---|---|---|---|---|
| VGG16 | 28.6 | 9.7 | 20 | 38.6 |
| MobileNet | 2.4 | 13.9 | 25 | 16.7 |
| ResNet50 | 9.8 | 26.5 | 60 | 37.3 |
| InceptionV3 | 2.2 | 19.3 | 33 | 22.1 |
| Xception | 11.0 | 18.2 | 31 | 29.9 |

InceptionV3's feature time covers only the part extracted after the Docker crash (the first 10,240 images were extracted before it). Total wall-clock time for the run, excluding the gap after the crash: about 2.6 hours.

**Run history.** The run used the same resumable command throughout (`python scripts/run_image_level.py`):

1. Started 03:27 UTC in a container with `--memory=3g --memory-swap=3g --cpus=4`, 4/2 TensorFlow threads and feature batch size 32. VGG16 and MobileNet finished.
2. 04:23 UTC: stopped by hand between models, because container memory had climbed to about 86% of the limit, and restarted with lower memory use (each model in its own Python process, the Keras session cleared before each tuner trial, cached image pages released). ResNet50 finished.
3. About 05:09 UTC, during InceptionV3 feature extraction: **Docker Desktop crashed** because of host memory pressure from outside the container (host MemAvailable had fallen to about 1.3 GB; the container itself used about 0.8 GB). There was no OOM kill inside the container.
4. 05:17 UTC: restarted with `--memory=2500m --memory-swap=2500m --cpus=3`, 3/1 TensorFlow threads and `IMAGE_LEVEL_BATCH=16`. InceptionV3 resumed its feature extraction at image 10,240 of 12,446. InceptionV3 and Xception finished, and the container exited with code 0 at 06:09 UTC.

Changing the thread count and the feature batch size does not change the method. Features are a deterministic function of each image, apart from floating-point rounding.

**Memory.** Phases 1–3 (`memory.log`, polled every ~25 s from `docker stats` and `/proc/meminfo`, 258 polls, local time UTC+05:45): peak container usage **2.59 GiB** of the 3 GiB limit, reached before the memory changes; lowest host MemAvailable 1335 MB, just before the Docker crash. Phase 4 (host-side watchdog log, not included in the repository; 239 polls every ~10 s): peak container usage **1.78 GiB** of the 2.44 GiB limit; lowest host MemAvailable 3,018 MB. No OOM kill and no watchdog pause in any phase.
