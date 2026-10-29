#!/usr/bin/env python3
"""Kidney CT transfer-learning experiment with an image-level train/val/test split.

Five frozen ImageNet backbones (VGG16, MobileNet, ResNet50, InceptionV3, Xception) at
150x150 with global max pooling, and a Dense-BatchNorm-Dropout head tuned with keras-tuner
RandomSearch. Evaluation protocol:

1. a stratified 70/15/15 train/val/test split by image; byte-identical files are kept in
   one subset and the subsets are asserted to be disjoint;
2. each network gets its own `preprocess_input`, applied once to 0-255 pixels;
3. hyperparameters are chosen on the validation set only;
4. metrics use argmax labels.

The backbone is frozen and there is no augmentation, so each backbone's 150x150 max-pooled
features are computed once and cached; training the head on them is equivalent to training
the full stack (checked by --sanity-check). The test set is used once per model.

Resumable: the split, each feature cache (checkpointed every few batches), every keras-tuner
trial and every finished model are saved, so re-running the same command continues where it
stopped. Designed to run in Docker (docker/Dockerfile.image_level) with a 3 GB memory limit.

    python scripts/run_image_level.py                  # full run (all five models)
    python scripts/run_image_level.py --sanity-check   # end-to-end vs cached logits, one model
    python scripts/run_image_level.py --report-only    # rebuild REPORT tables from saved results
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kidney_tl.data import index_images  # noqa: E402
from kidney_tl.grouping import max_similarity  # noqa: E402
from kidney_tl.image_level import assert_split_ok, duplicate_groups, evaluate, image_level_split  # noqa: E402
from kidney_tl.image_level_pipeline import (  # noqa: E402
    CLASS_INDEX,
    PREPROCESSING,
    assert_aligned,
    audit_files,
    backbone,
    check_labels,
    image_dataset,
    preprocess_fn,
    preprocessing_stats,
)

from kidney_tl.image_level_pipeline import IMG_SIZE as IMG, MODELS  # noqa: E402
SEED = 123
# relative cost of each stage, for the percent/ETA in status.json
W_FEAT, W_TRIAL, W_FINAL = 6.0, 0.3, 1.0
MAX_TRIALS = 10


# --------------------------------------------------------------------------- progress
class Progress:
    """progress.log (one line per stage / trial) and status.json (model, stage, percent, ETA)."""

    def __init__(self, out: Path):
        self.log_path, self.status_path = out / "progress.log", out / "status.json"
        self.start = time.time()
        self.units_total = len(MODELS) * (W_FEAT + MAX_TRIALS * W_TRIAL + W_FINAL)
        self.units_done = 0.0       # including work done in earlier (interrupted) runs
        self.units_session = 0.0    # done in this run, for the ETA
        self.model, self.stage = "", "starting"

    def log(self, msg: str) -> None:
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as fh:
            fh.write(line + "\n")

    def advance(self, units: float, session: bool = True) -> None:
        self.units_done += units
        if session:
            self.units_session += units
        self.write()

    def set(self, model: str | None = None, stage: str | None = None) -> None:
        self.model = model if model is not None else self.model
        self.stage = stage if stage is not None else self.stage
        self.write()

    def write(self, done: bool = False) -> None:
        elapsed = time.time() - self.start
        pct = 100.0 * self.units_done / self.units_total
        eta = None
        if self.units_session > 0 and not done:
            eta = elapsed / self.units_session * (self.units_total - self.units_done)
        status = {
            "model": self.model, "stage": "finished" if done else self.stage,
            "percent": round(100.0 if done else pct, 1),
            "elapsed": str(timedelta(seconds=int(elapsed))),
            "eta": None if eta is None else str(timedelta(seconds=int(eta))),
            "eta_clock": None if eta is None else f"{datetime.now() + timedelta(seconds=eta):%H:%M}",
            "updated": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "pid": os.getpid(),
        }
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=2))
        tmp.replace(self.status_path)


def drop_page_cache(paths) -> None:
    """Ask the kernel to drop clean cached pages of these files. Page cache is charged to the
    container's memory cgroup, so reading 12,446 JPEGs per model would otherwise push the
    container towards its limit (and the host watchdog's 92% pause threshold)."""
    if not hasattr(os, "posix_fadvise"):
        return
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def write_json(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    tmp.replace(path)


# --------------------------------------------------------------------------- split
def load_md5(df: pd.DataFrame, cache_dir: Path) -> np.ndarray:
    """MD5 per image from cache/image_hashes.csv, falling back to hashing the files."""
    hashes = cache_dir / "image_hashes.csv"
    if hashes.exists():
        h = pd.read_csv(hashes).set_index("path")["md5"]
        if set(h.index) == set(df["path"]):
            return h.loc[df["path"]].to_numpy()
    from kidney_tl.data import file_md5

    return np.array([file_md5(Path(df.attrs["root"]) / p) for p in df["path"]])


def make_split(df: pd.DataFrame, md5: np.ndarray, out: Path, prog: Progress) -> np.ndarray:
    path = out / "split.csv"
    if path.exists():
        saved = pd.read_csv(path)
        if saved["path"].tolist() != df["path"].tolist():
            raise RuntimeError(f"{path} does not match the dataset file list; move it away to regenerate")
        subset = saved["subset"].to_numpy()
        prog.log(f"split: reusing {path.relative_to(ROOT)}")
    else:
        subset = image_level_split(df["label"].to_numpy(), md5, seed=SEED)
        pd.DataFrame({"path": df["path"], "class": df["cls"], "subset": subset}).to_csv(path, index=False)
        prog.log(f"split: wrote {path.relative_to(ROOT)}")
    assert_split_ok(subset, md5)
    return subset


def leakage_summary(df: pd.DataFrame, md5: np.ndarray, subset: np.ndarray, cache_dir: Path) -> dict:
    """Measure (do not fix) how much of the test set comes from scans seen in training."""
    groups = duplicate_groups(md5)
    sizes = np.bincount(groups)
    tr, va, te = (np.flatnonzero(subset == s) for s in ("train", "val", "test"))
    info = {
        "n_images": int(len(df)),
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "per_class": pd.crosstab(df["cls"], subset).loc[:, ["train", "val", "test"]].to_dict(orient="index"),
        "duplicate_groups": int((sizes > 1).sum()),
        "files_in_duplicate_groups": int(sizes[sizes > 1].sum()),
        "test_byte_identical_in_train": int(np.isin(md5[te], md5[tr]).sum()),
    }
    # Per-image scan groups (204 recovered scans; per-class summary in results/scan_groups.csv).
    folds = ROOT / "results" / "splits" / "group-level_folds.csv"
    if folds.exists():
        g = pd.read_csv(folds)
        g = g[g["fold"] == 0].set_index("path")["scan_group"]
        if set(g.index) == set(df["path"]):
            scan = g.loc[df["path"]].to_numpy()
            seen = np.isin(scan[te], scan[tr])
            info.update(scan_groups=int(np.unique(scan).size), test_from_scan_in_train=int(seen.sum()),
                        test_from_scan_in_train_pct=100.0 * seen.mean(),
                        val_from_scan_in_train_pct=100.0 * np.isin(scan[va], scan[tr]).mean())
    sig = cache_dir / "signatures_32.npz"
    if sig.exists():
        stored = np.load(sig, allow_pickle=False)
        if stored["paths"].tolist() == df["path"].tolist():
            r = max_similarity(te, tr, stored["thumbs"])
            info.update(test_near_identical_in_train=int((r >= 0.995).sum()),
                        test_near_identical_in_train_pct=100.0 * float((r >= 0.995).mean()),
                        test_max_r_to_train_median=float(np.median(r)))
    return info


# --------------------------------------------------------------------------- input checks
EXPECTED_DIM = {"VGG16": 512, "MobileNet": 1024, "ResNet50": 2048, "InceptionV3": 2048, "Xception": 2048}


def label_and_split_checks(df: pd.DataFrame, subset: np.ndarray, out: Path) -> dict:
    """Fixed class-index map, folder labels, and per-class 70/15/15 stratification (cheap; run every time)."""
    found = {c: i for i, c in enumerate(df.attrs["classes"])}
    if found != CLASS_INDEX:
        raise AssertionError(f"class folders {found} differ from the fixed map {CLASS_INDEX}")
    check_labels(df["path"].tolist(), df["label"].tolist())
    write_json(CLASS_INDEX, out / "class_index.json")
    frac = pd.crosstab(df["cls"], subset, normalize="index")[["train", "val", "test"]]
    for s, target in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
        if (frac[s] - target).abs().max() > 0.01:
            raise AssertionError(f"subset {s} is not stratified: {frac[s].to_dict()}")
    return {"class_index": CLASS_INDEX, "labels_match_folders": True,
            "subset_fraction_per_class": frac.round(4).to_dict(orient="index")}


def input_checks(tf, df: pd.DataFrame, subset: np.ndarray, out: Path, prog: Progress) -> dict:
    """Decode audit, 0-255 range, per-model preprocessing range, feature shapes and a picture grid."""
    path = out / "input_checks.json"
    if path.exists():
        return json.loads(path.read_text())
    prog.log("input checks: auditing every file (format, mode, bit depth, size, decode errors)")
    root = Path(df.attrs["root"])
    audit = audit_files(root, df["path"].tolist())
    if audit["decode_failures"]:
        prog.log(f"input checks: {len(audit['decode_failures'])} files failed to decode: {audit['decode_failures'][:5]}")
    # also decode every file with the TF pipeline itself, in batches (catches anything PIL accepts but TF rejects)
    files = [str(root / p) for p in df["path"]]
    n_ok, lo, hi = 0, np.inf, -np.inf
    for x in image_dataset(files, 64):
        if x.shape[1:] != (IMG, IMG, 3):
            raise AssertionError(f"decoded batch has shape {x.shape}")
        n_ok += int(x.shape[0])
        lo, hi = min(lo, float(tf.reduce_min(x))), max(hi, float(tf.reduce_max(x)))
    if n_ok != len(files) or lo < 0 or hi > 255 or hi <= 1:
        raise AssertionError(f"TF decode: {n_ok}/{len(files)} images, pixel range [{lo}, {hi}]")
    prog.log(f"input checks: TF decoded {n_ok}/{len(files)} images to {IMG}x{IMG}x3 float32, range [{lo:.0f}, {hi:.0f}]")

    # per-model preprocessing on one sample batch (16 training images, 4 per class)
    rng = np.random.default_rng(SEED)
    sample = np.concatenate([rng.choice(np.flatnonzero((subset == "train") & (df["label"].to_numpy() == c)), 4,
                                        replace=False) for c in range(4)])
    batch = next(iter(image_dataset([files[i] for i in sample], len(sample)))).numpy()
    pre_stats, shapes = [], {}
    for name in MODELS:
        st = preprocessing_stats(name, batch)
        pre_stats.append(st)
        prog.log(f"input checks: {name:11s} {st['mode']}: min {st['min']:.2f} max {st['max']:.2f} "
                 f"mean {st['mean']:.2f} std {st['std']:.2f} (expected within {st['expected_range'][0]:.2f}"
                 f"..{st['expected_range'][1]:.2f})")
        net = backbone(name, weights=None)  # shape check only; weights do not change shapes
        y = net(preprocess_fn(name)(tf.constant(batch[:2])), training=False)
        shapes[name] = [int(d) for d in y.shape]
        if shapes[name] != [2, EXPECTED_DIM[name]]:
            raise AssertionError(f"{name}: feature shape {shapes[name]}, expected [2, {EXPECTED_DIM[name]}]")
        del net, y
        tf.keras.backend.clear_session()
        gc.collect()
    prog.log(f"input checks: feature shapes at {IMG}x{IMG} with pooling='max': {shapes}")

    # picture grid: 3 train + 3 test images per class, after decoding and resizing
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 6, figsize=(12, 8.6))
    for c, cls in enumerate(df.attrs["classes"]):
        picks = [(s, i) for s in ("train", "test") for i in
                 rng.choice(np.flatnonzero((subset == s) & (df["label"].to_numpy() == c)), 3, replace=False)]
        imgs = next(iter(image_dataset([files[i] for _, i in picks], 6))).numpy()
        for j, ((s, i), img) in enumerate(zip(picks, imgs, strict=True)):
            ax = axes[c, j]
            ax.imshow(img.astype(np.uint8))
            ax.set_title(f"{cls} | {s}\n{Path(df['path'].iloc[i]).name}", fontsize=8)
            ax.axis("off")
    fig.suptitle(f"Decoded and resized to {IMG}x{IMG} RGB (0-255, before preprocess_input)")
    fig.tight_layout()
    fig.savefig(out / "input_check.png", dpi=90)
    plt.close(fig)

    result = {"audit": audit, "tf_decode": {"images": n_ok, "pixel_min": lo, "pixel_max": hi},
              "preprocessing_sample_batch": pre_stats, "feature_shapes": shapes,
              "preprocessing": {m: PREPROCESSING[m][2] for m in MODELS}}
    write_json(result, path)
    return result


# --------------------------------------------------------------------------- TF setup
def setup_tf():
    threads = int(os.environ.get("TF_NUM_INTRAOP_THREADS", "4"))
    inter = int(os.environ.get("TF_NUM_INTEROP_THREADS", "2"))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(threads))
    try:
        os.nice(10)
    except OSError:
        pass
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(inter)
    return tf


# --------------------------------------------------------------------------- features
def feature_path(cache_dir: Path, name: str) -> Path:
    return cache_dir / f"image_level_{name}_{IMG}_max.npy"


def extract_features(tf, name: str, df: pd.DataFrame, cache_dir: Path, batch: int, prog: Progress) -> Path:
    """Max-pooled features of every image, streamed in batches into a memory-mapped file that is
    checkpointed every ~1000 images, so an interrupted extraction resumes."""
    final = feature_path(cache_dir, name)
    if final.exists():
        return final
    partial = final.with_name(final.stem + ".partial.npy")
    marker = final.with_name(final.stem + ".progress.json")
    files = [str(Path(df.attrs["root"]) / p) for p in df["path"]]
    net = backbone(name)
    dim = int(net.output_shape[-1])
    done = 0
    if partial.exists() and marker.exists():
        state = json.loads(marker.read_text())
        if state.get("n") == len(files) and state.get("dim") == dim:
            done = int(state["done"])
    if done == 0:
        feats = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=(len(files), dim))
    else:
        feats = np.load(partial, mmap_mode="r+")
        prog.log(f"{name} features: resuming at image {done}/{len(files)}")
    t0, last = time.time(), done
    # file order, no shuffling; preprocess_input applied once inside the dataset
    for x in image_dataset(files[done:], batch, model=name):
        out = net(x, training=False).numpy().astype(np.float32)
        feats[done:done + len(out)] = out
        done += len(out)
        if done - last >= 1024 or done == len(files):
            feats.flush()
            write_json({"n": len(files), "dim": dim, "done": done}, marker)
            rate = (done - last) / max(time.time() - t0, 1e-6)
            prog.advance(W_FEAT * (done - last) / len(files))
            prog.log(f"{name} features: {done}/{len(files)} images ({rate:.1f} img/s)")
            t0, last = time.time(), done
    feats.flush()
    del feats, net
    gc.collect()
    tf.keras.backend.clear_session()
    arr = np.load(partial, mmap_mode="r")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"non-finite values in {partial}")
    del arr
    drop_page_cache(files)
    final.with_name(final.stem + ".paths.txt").write_text("\n".join(df["path"]) + "\n")
    partial.replace(final)
    marker.unlink(missing_ok=True)
    return final


def load_features(path: Path, df: pd.DataFrame) -> np.ndarray:
    """Load one model's cache and assert its rows follow split.csv / the dataset index order."""
    rows = path.with_name(path.stem + ".paths.txt")
    if not rows.exists():
        raise RuntimeError(f"{rows} missing: cannot verify that feature rows match split.csv")
    assert_aligned(rows.read_text().splitlines(), df["path"].tolist())
    feats = np.load(path).astype(np.float32, copy=False)
    drop_page_cache([path])
    if feats.shape[0] != len(df):
        raise AssertionError(f"{path}: {feats.shape[0]} rows for {len(df)} images")
    return feats


# --------------------------------------------------------------------------- head + tuner
def head_builder(tf, dim: int):
    """The notebook's head on top of the (cached) frozen backbone output."""
    layers = tf.keras.layers

    def build(hp):
        model = tf.keras.Sequential([
            layers.Input(shape=(dim,)),
            layers.Flatten(),  # a no-op after max pooling, kept to mirror the notebook
            layers.Dense(units=hp.Int("dense_units", min_value=128, max_value=1024, step=128), activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(rate=hp.Float("dropout_rate", min_value=0.2, max_value=0.5, step=0.1)),
            layers.Dense(4, activation="softmax"),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model
    return build


def make_tuner(tf, build, directory: Path, project: str, prog: Progress, model_name: str):
    import keras_tuner as kt

    class LoggedRandomSearch(kt.RandomSearch):
        def on_trial_begin(self, trial):
            # free the previous trial's graphs and models (otherwise memory grows trial by trial)
            tf.keras.backend.clear_session()
            gc.collect()
            super().on_trial_begin(trial)

        def on_trial_end(self, trial):
            super().on_trial_end(trial)
            score = trial.score if trial.score is not None else float("nan")
            hp = trial.hyperparameters.values
            prog.advance(W_TRIAL)
            prog.log(f"{model_name} tuner trial {trial.trial_id} {trial.status}: val_accuracy={score:.4f} "
                     f"units={hp.get('dense_units')} dropout={hp.get('dropout_rate'):.1f} "
                     f"lr={hp.get('learning_rate'):g}")

    return LoggedRandomSearch(build, objective="val_accuracy", max_trials=MAX_TRIALS, executions_per_trial=3,
                              directory=str(directory), project_name=project, overwrite=False, seed=SEED)


def plot_curves(history: pd.DataFrame, name: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    epochs = np.arange(1, len(history) + 1)
    for i, metric in enumerate(("loss", "accuracy")):
        ax[i].plot(epochs, history[metric], label=f"training {metric}")
        ax[i].plot(epochs, history[f"val_{metric}"], label=f"validation {metric}")
        ax[i].set_xlabel("Epoch")
        ax[i].set_ylabel(metric.capitalize())
        ax[i].legend()
    fig.suptitle(f"{name}: final refit on train (frozen backbone, cached features)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(cm, classes, name: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name}: test confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_model(tf, name: str, df: pd.DataFrame, subset: np.ndarray, out: Path, cache_dir: Path, batch: int,
              prog: Progress) -> None:
    mdir = out / name
    if (mdir / "metrics.json").exists():
        prog.log(f"{name}: already complete, skipping")
        prog.advance(W_FEAT + MAX_TRIALS * W_TRIAL + W_FINAL, session=False)
        return
    mdir.mkdir(parents=True, exist_ok=True)
    t_model = time.time()
    classes = df.attrs["classes"]
    y = df["label"].to_numpy()

    # 1. features (resumable)
    if feature_path(cache_dir, name).exists():
        prog.advance(W_FEAT, session=False)
    else:
        marker = feature_path(cache_dir, name).with_name(f"image_level_{name}_{IMG}_max.progress.json")
        if marker.exists():
            prog.advance(W_FEAT * json.loads(marker.read_text())["done"] / len(df), session=False)
    prog.set(name, "features")
    prog.log(f"{name}: stage features")
    t0 = time.time()
    feats_path = extract_features(tf, name, df, cache_dir, batch, prog)
    t_feat = time.time() - t0
    feats = load_features(feats_path, df)
    idx = {s: np.flatnonzero(subset == s) for s in ("train", "val", "test")}
    x = {s: feats[i] for s, i in idx.items()}
    ys = {s: y[i] for s, i in idx.items()}
    del feats
    gc.collect()

    # 2. random search (resumes trial by trial)
    tuner_dir = out / "tuner"
    proj = tuner_dir / f"{name}_hyperparameter_tuning"
    already = len(list(proj.glob("trial_*/trial.json"))) if proj.exists() else 0
    already = min(already, MAX_TRIALS)
    prog.advance(already * W_TRIAL, session=False)
    prog.set(name, "tuning")
    prog.log(f"{name}: stage tuning ({already}/{MAX_TRIALS} trials already done)")
    tf.keras.utils.set_random_seed(SEED)
    build = head_builder(tf, x["train"].shape[1])
    tuner = make_tuner(tf, build, tuner_dir, f"{name}_hyperparameter_tuning", prog, name)
    t0 = time.time()
    tuner.search(x["train"], ys["train"], validation_data=(x["val"], ys["val"]), epochs=10, batch_size=32,
                 verbose=0)
    t_tune = time.time() - t0
    best = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_trial = tuner.oracle.get_best_trials(1)[0]
    trials = []
    for t in tuner.oracle.trials.values():
        trials.append({"trial": t.trial_id, **t.hyperparameters.values, "val_accuracy": t.score})
    pd.DataFrame(trials).sort_values("trial").to_csv(mdir / "tuner_trials.csv", index=False)
    del tuner
    gc.collect()

    # 3. refit the best configuration on train for 10 epochs, validating on val
    prog.set(name, "final training")
    prog.log(f"{name}: stage final training with {best.values}")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    model = build(best)
    t0 = time.time()
    hist = model.fit(x["train"], ys["train"], validation_data=(x["val"], ys["val"]), epochs=10, batch_size=32,
                     verbose=0)
    t_fit = time.time() - t0
    history = pd.DataFrame(hist.history)
    history.index.name = "epoch"
    history.to_csv(mdir / "history.csv")
    model.save(mdir / "head.keras")
    plot_curves(history, name, mdir / "curves.png")

    # 4. single evaluation on the test set
    prog.set(name, "test evaluation")
    prob = model.predict(x["test"], batch_size=256, verbose=0)
    metrics = evaluate(ys["test"], prob, classes, n_boot=1000, seed=SEED)
    val_prob = model.predict(x["val"], batch_size=256, verbose=0)
    metrics.update(
        model=name, best_hyperparameters=best.values, best_trial=best_trial.trial_id,
        tuner_best_val_accuracy=float(best_trial.score),
        final_val_accuracy=float((val_prob.argmax(1) == ys["val"]).mean()),
        final_train_accuracy_last_epoch=float(history["accuracy"].iloc[-1]),
        feature_dim=int(x["train"].shape[1]),
        seconds={"features": round(t_feat, 1), "tuning": round(t_tune, 1), "final_fit": round(t_fit, 1),
                 "model_total": round(time.time() - t_model, 1)},
    )
    test_df = df.iloc[idx["test"]]
    pred = pd.DataFrame({"path": test_df["path"].to_numpy(), "class": test_df["cls"].to_numpy(),
                         "y_true": ys["test"], "y_pred": prob.argmax(1)})
    for i, c in enumerate(classes):
        pred[f"p_{c}"] = prob[:, i].round(6)
    pred.to_csv(mdir / "predictions.csv", index=False)
    plot_confusion(metrics["confusion_matrix"], classes, name, mdir / "confusion_matrix.png")
    write_json(metrics, mdir / "metrics.json")  # written last: marks the model as complete
    prog.advance(W_FINAL)
    prog.log(f"{name}: done in {timedelta(seconds=int(time.time() - t_model))}: test accuracy "
             f"{metrics['accuracy']:.4f} [{metrics['accuracy_ci_low']:.4f}, {metrics['accuracy_ci_high']:.4f}], "
             f"macro-F1 {metrics['macro_f1']:.4f}")
    del model, x, ys, prob, val_prob
    tf.keras.backend.clear_session()
    gc.collect()


# --------------------------------------------------------------------------- sanity check
def sanity_check(tf, df: pd.DataFrame, subset: np.ndarray, out: Path, cache_dir: Path, name: str) -> dict:
    """Show that the cached-feature head equals the notebook's end-to-end frozen model.

    Builds the notebook's full Sequential (frozen backbone -> Flatten -> Dense -> BN -> Dropout
    -> Dense), copies its head weights into a head-only model, and compares outputs on 64 test
    images: end-to-end vs head(cached features). It also calls the backbone with training=True
    to confirm that a frozen backbone's BatchNorm stays in inference mode during fit.
    """
    import keras_tuner as kt

    cache = feature_path(cache_dir, name)
    if not cache.exists():
        raise SystemExit(f"{cache} not found; run the main script first (it creates the feature cache)")
    test_idx = np.flatnonzero(subset == "test")[:64]
    files = [str(Path(df.attrs["root"]) / p) for p in df["path"].iloc[test_idx]]
    cached = load_features(cache, df)[test_idx]

    tf.keras.utils.set_random_seed(SEED)
    net = backbone(name)
    hp = kt.HyperParameters()
    hp.Fixed("dense_units", 256)
    hp.Fixed("dropout_rate", 0.3)
    hp.Fixed("learning_rate", 1e-3)
    head = head_builder(tf, int(net.output_shape[-1]))(hp)
    # give BatchNorm non-trivial statistics so the comparison is not vacuous
    bn = [layer for layer in head.layers if isinstance(layer, tf.keras.layers.BatchNormalization)][0]
    rng = np.random.default_rng(0)
    bn.set_weights([w + rng.normal(0, 0.1, w.shape).astype(np.float32) if i < 3 else
                    np.abs(w + rng.normal(0, 0.1, w.shape)).astype(np.float32) for i, w in enumerate(bn.get_weights())])
    full = tf.keras.Sequential([net] + [layer for layer in head.layers])  # notebook-style end-to-end model

    e2e, feats_eval, feats_train = [], [], []
    for x in image_dataset(files, 16, model=name):
        e2e.append(full(x, training=False).numpy())
        feats_eval.append(net(x, training=False).numpy())
        feats_train.append(net(x, training=True).numpy())
    e2e, feats_eval, feats_train = map(np.concatenate, (e2e, feats_eval, feats_train))
    from_cache = head(cached, training=False).numpy()
    result = {
        "model": name, "images": len(test_idx),
        "max_abs_diff_probabilities_end_to_end_vs_cached": float(np.abs(e2e - from_cache).max()),
        "max_abs_diff_features_recomputed_vs_cache": float(np.abs(feats_eval - cached).max()),
        "max_abs_diff_backbone_training_true_vs_false": float(np.abs(feats_train - feats_eval).max()),
        "argmax_agreement": float((e2e.argmax(1) == from_cache.argmax(1)).mean()),
        "feature_scale_max": float(np.abs(cached).max()),
    }
    result["passed"] = bool(result["max_abs_diff_probabilities_end_to_end_vs_cached"] < 1e-4
                            and result["max_abs_diff_backbone_training_true_vs_false"] == 0.0
                            and result["argmax_agreement"] == 1.0)
    write_json(result, out / "sanity_check.json")
    print(json.dumps(result, indent=2))
    return result


# --------------------------------------------------------------------------- summary
def write_summary(out: Path, classes) -> pd.DataFrame | None:
    rows = []
    for name in MODELS:
        p = out / name / "metrics.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        row = {"model": name}
        for k in ("accuracy", "accuracy_ci_low", "accuracy_ci_high", "balanced_accuracy", "macro_precision",
                  "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1",
                  "roc_auc_macro_ovr", "tuner_best_val_accuracy", "final_val_accuracy"):
            row[k] = m.get(k)
        for c in classes:
            for k in ("precision", "recall", "f1"):
                row[f"{c}_{k}"] = m.get(f"{c}_{k}")
        row.update({f"hp_{k}": v for k, v in m["best_hyperparameters"].items()})
        row["minutes"] = round(m["seconds"]["model_total"] / 60, 1)
        rows.append(row)
    if not rows:
        return None
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "ct-kidney")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "image_level")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("IMAGE_LEVEL_BATCH", "32")),
                    help="images per batch for feature extraction (lower it after an OOM kill)")
    ap.add_argument("--sanity-check", action="store_true", help="compare end-to-end and cached outputs, then exit")
    ap.add_argument("--sanity-model", default="MobileNet", choices=MODELS)
    ap.add_argument("--input-check", action="store_true", help="run the input-pipeline checks only, then exit")
    ap.add_argument("--report-only", action="store_true", help="only rebuild summary.csv from saved results")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)  # set by the per-model orchestration
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tuner").mkdir(exist_ok=True)
    prog = Progress(args.out)
    df = index_images(args.data_dir)
    if args.report_only:
        print(write_summary(args.out, df.attrs["classes"]))
        return

    prog.log(f"run_image_level start: models={args.models} batch={args.batch_size} "
             f"threads={os.environ.get('TF_NUM_INTRAOP_THREADS', '4')}")
    prog.set("", "split")
    md5 = load_md5(df, args.cache_dir)
    subset = make_split(df, md5, args.out, prog)
    leak = leakage_summary(df, md5, subset, args.cache_dir)
    write_json(leak, args.out / "split_summary.json")
    prog.log(f"split: train {leak['n_train']}, val {leak['n_val']}, test {leak['n_test']}; "
             f"{leak['duplicate_groups']} duplicate groups ({leak['files_in_duplicate_groups']} files) kept together; "
             f"test from a scan seen in training: {leak.get('test_from_scan_in_train_pct', float('nan')):.1f}%; "
             f"near-identical (r>=0.995) to a training image: {leak.get('test_near_identical_in_train_pct', float('nan')):.1f}%")

    checks = label_and_split_checks(df, subset, args.out)
    prog.log(f"labels: class map {checks['class_index']} fixed; every label matches its folder; "
             f"per-class train/val/test fractions within 1 point of 70/15/15")
    if not (args.child or args.sanity_check or args.input_check):
        orchestrate(args, prog, df)
        return
    tf = setup_tf()
    tf.keras.utils.set_random_seed(SEED)
    input_checks(tf, df, subset, args.out, prog)
    if args.input_check:
        return
    if args.sanity_check:
        sanity_check(tf, df, subset, args.out, args.cache_dir, args.sanity_model)
        return
    import keras_tuner

    write_json({"tensorflow": tf.__version__, "keras_tuner": keras_tuner.__version__, "numpy": np.__version__,
                "pandas": pd.__version__, "python": sys.version.split()[0], "cpu_count": os.cpu_count(),
                "intra_threads": os.environ.get("TF_NUM_INTRAOP_THREADS"), "batch_size": args.batch_size},
               args.out / "environment.json")
    for name in MODELS:  # count models finished earlier, so percent and ETA cover the whole run
        if name not in args.models and (args.out / name / "metrics.json").exists():
            prog.advance(W_FEAT + MAX_TRIALS * W_TRIAL + W_FINAL, session=False)
    for name in args.models:
        run_model(tf, name, df, subset, args.out, args.cache_dir, args.batch_size, prog)


def orchestrate(args, prog: Progress, df: pd.DataFrame) -> None:
    """Run the input checks and then each model in a fresh Python process, so TensorFlow and
    keras-tuner memory is returned to the system after every model. This process never imports
    TensorFlow. A failed child stops the run; re-running the same command resumes it."""
    import subprocess

    base = [sys.executable, str(Path(__file__).resolve()), "--data-dir", str(args.data_dir), "--out", str(args.out),
            "--cache-dir", str(args.cache_dir), "--batch-size", str(args.batch_size)]
    if not (args.out / "input_checks.json").exists():
        subprocess.run(base + ["--input-check"], check=True)
    for name in args.models:
        if (args.out / name / "metrics.json").exists():
            prog.log(f"{name}: already complete, skipping")
            continue
        rc = subprocess.run(base + ["--child", "--models", name]).returncode
        if rc != 0:
            prog.log(f"{name}: child process exited with code {rc}; re-run the same command to resume")
            raise SystemExit(rc)
    write_summary(args.out, df.attrs["classes"])
    prog.write(done=True)
    prog.log(f"run_image_level finished (this session {timedelta(seconds=int(time.time() - prog.start))})")


if __name__ == "__main__":
    main()
