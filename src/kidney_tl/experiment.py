"""End-to-end experiment: index, group scans, build folds, extract features, train, evaluate, report."""

from __future__ import annotations

import itertools
import time
from functools import partial

import numpy as np
import pandas as pd

from . import report
from .config import ExperimentConfig
from .data import image_signatures, index_images
from .features import extract_features
from .grouping import assign_groups
from .head import grid, predict_proba, select_head, standardize, train_head
from .metrics import (
    classification_metrics,
    cluster_bootstrap_ci,
    fast_accuracy,
    fast_balanced_accuracy,
    fast_macro_f1,
    holm_adjust,
    paired_cluster_bootstrap,
)
from .splits import (
    GROUP_CV,
    IMAGE_CV,
    Split,
    assert_disjoint,
    group_cv,
    image_cv,
    leakage_report,
    write_manifest,
)
from .utils import environment_info, get_logger, write_json

POOLED_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall", "weighted_f1",
                  "cohen_kappa", "mcc", "roc_auc_macro_ovr")


def build_protocols(cfg: ExperimentConfig, df: pd.DataFrame, groups: np.ndarray) -> dict[str, list[Split]]:
    """Group-level CV (primary) and image-level CV."""
    labels = df["label"].to_numpy()
    protocols = {
        GROUP_CV: group_cv(labels, groups, cfg.n_folds, cfg.val_frac, cfg.seed),
        IMAGE_CV: image_cv(labels, cfg.n_folds, cfg.val_frac, cfg.seed),
    }
    for name, folds in protocols.items():
        for f in folds:
            assert_disjoint(f, groups if name == GROUP_CV else None)
        tested = np.concatenate([f.test for f in folds])
        if np.unique(tested).size != tested.size:
            raise AssertionError(f"{name}: an image is tested in more than one fold")
    return protocols


def _bootstrap_statistics(cfg, protocols, oof, y, groups, n_classes):
    """Scan-level bootstrap CIs for each model, and paired comparisons between all model pairs."""
    boot_metrics = {"accuracy": fast_accuracy,
                    "balanced_accuracy": partial(fast_balanced_accuracy, n_classes=n_classes),
                    "macro_f1": partial(fast_macro_f1, n_classes=n_classes)}
    ci_rows, compare_rows = [], []
    for (name, model_name), (idx, pred, _) in oof.items():
        for metric_name, fn in boot_metrics.items():
            lo, hi = cluster_bootstrap_ci(y[idx], pred, groups[idx], fn, cfg.bootstrap_samples, seed=0)
            ci_rows.append({"protocol": name, "model": model_name, "metric": metric_name,
                            "value": fn(y[idx], pred), "ci_low": lo, "ci_high": hi})
    pairs = list(itertools.combinations(cfg.models, 2))
    for name in protocols:
        for metric_name in ("macro_f1", "accuracy"):
            tests = []
            for a, b in pairs:
                idx, pred_a, _ = oof[(name, a)]
                pred_b = oof[(name, b)][1]
                tests.append(paired_cluster_bootstrap(y[idx], pred_a, pred_b, groups[idx],
                                                      boot_metrics[metric_name], cfg.bootstrap_samples, seed=1))
            adjusted = holm_adjust([t["p_value"] for t in tests]) if tests else []
            compare_rows += [{"protocol": name, "metric": metric_name, "model_a": a, "model_b": b, **t, "p_holm": p}
                             for (a, b), t, p in zip(pairs, tests, adjusted, strict=True)]
    return pd.DataFrame(ci_rows), pd.DataFrame(compare_rows)


def _save_predictions(oof, df, groups, classes, path) -> None:
    """Every test prediction (one row per image per model and protocol), for auditing."""
    frames = []
    for (name, model_name), (idx, pred, prob) in oof.items():
        frame = pd.DataFrame(prob, columns=[f"prob_{c}" for c in classes])
        frame.insert(0, "pred", np.asarray(classes)[pred])
        frame.insert(0, "true", df["cls"].iloc[idx].to_numpy())
        frame.insert(0, "scan_group", groups[idx])
        frame.insert(0, "path", df["path"].iloc[idx].to_numpy())
        frame.insert(0, "model", model_name)
        frame.insert(0, "protocol", name)
        frames.append(frame)
    pd.concat(frames).to_csv(path, index=False, float_format="%.5f")


def run(cfg: ExperimentConfig) -> dict[str, pd.DataFrame]:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger(out / "run.log")
    t_start = time.time()
    cfg.to_json(out / "config.json")
    write_json(environment_info(), out / "environment.json")

    # 1. Data, signatures and scan groups
    df = index_images(cfg.data_dir)
    classes = df.attrs["classes"]
    md5, thumbs = image_signatures(df, cfg.cache_dir)
    groups, group_info = assign_groups(df, md5, thumbs, cfg.scan_break_r, cfg.duplicate_r)
    n_dup = int(pd.Series(md5).duplicated(keep=False).sum())
    log.info("Indexed %d images in %s; %d byte-identical files; %d scan groups",
             len(df), classes, n_dup, np.unique(groups).size)
    group_info.to_csv(out / "scan_groups.csv", index=False)

    # 2. Protocols and leakage
    protocols = build_protocols(cfg, df, groups)
    leakage = pd.DataFrame([leakage_report(folds, groups, md5, thumbs, cfg.duplicate_r)
                            for folds in protocols.values()])
    leakage.to_csv(out / "leakage_report.csv", index=False)
    for name, folds in protocols.items():
        write_manifest(folds, df, groups, out / "splits" / f"{name.split(' ')[0]}_folds.csv")
    log.info("Leakage check:\n%s", leakage.round(2).to_string(index=False))

    # 3. Features once per backbone; per fold: choose the head on validation, train, predict the test fold
    y = df["label"].to_numpy()
    configs = grid(cfg.grid_units, cfg.grid_dropout, cfg.grid_lr)
    train_kwargs = dict(max_epochs=cfg.max_epochs, patience=cfg.patience, batch_size=cfg.batch_size,
                        class_weights=cfg.class_weights)
    fold_rows, search_rows, oof = [], [], {}
    for model_name in cfg.models:
        t0 = time.time()
        feats = extract_features(df.attrs["root"], df["path"].tolist(), model_name, cfg.img_size, cfg.weights,
                                 cfg.feature_batch_size, cfg.cache_dir)
        log.info("%s: features %s in %.0fs", model_name, feats.shape, time.time() - t0)
        for name, folds in protocols.items():
            pred = np.full(len(df), -1)
            prob = np.full((len(df), len(classes)), np.nan, dtype=np.float32)
            for f in folds:
                x_tr, x_va, x_te = standardize(feats[f.train], feats[f.val], feats[f.test])
                seed = cfg.seed + f.fold
                best, rows = select_head(x_tr, y[f.train], x_va, y[f.val], configs, len(classes), seed,
                                         **train_kwargs)
                search_rows += [{"protocol": name, "model": model_name, "fold": f.fold, **r} for r in rows]
                head, epochs = train_head(x_tr, y[f.train], x_va, y[f.val], best, len(classes), seed, **train_kwargs)
                p = predict_proba(head, x_te)
                prob[f.test], pred[f.test] = p, p.argmax(axis=1)
                fold_rows.append({"protocol": name, "model": model_name, "fold": f.fold, "epochs": epochs,
                                  "head_units": best.units, "head_dropout": best.dropout, "head_lr": best.lr,
                                  **classification_metrics(y[f.test], pred[f.test], p, classes)})
            tested = np.unique(np.concatenate([f.test for f in folds]))
            oof[(name, model_name)] = (tested, pred[tested], prob[tested])
            accs = " ".join(f"{r['accuracy']:.4f}" for r in fold_rows[-len(folds):])
            log.info("  %-20s accuracy per fold: %s", name, accs)

    # 4. Pooled out-of-fold metrics, scan-level bootstrap intervals, paired model comparisons
    pooled = pd.DataFrame([{"protocol": name, "model": model_name, "n_test_images": len(idx),
                            "n_test_scans": int(np.unique(groups[idx]).size),
                            **classification_metrics(y[idx], pred, prob, classes)}
                           for (name, model_name), (idx, pred, prob) in oof.items()])
    ci, comparisons = _bootstrap_statistics(cfg, protocols, oof, y, groups, len(classes))
    folds_df = pd.DataFrame(fold_rows)
    present = [m for m in POOLED_METRICS if m in folds_df]
    fold_summary = folds_df.groupby(["protocol", "model"])[present].agg(["mean", "std"])
    fold_summary.columns = [f"{m}_{s}" for m, s in fold_summary.columns]

    tables = {"pooled": pooled, "folds": folds_df, "fold_summary": fold_summary.reset_index(), "ci": ci,
              "comparisons": comparisons, "search": pd.DataFrame(search_rows), "leakage": leakage,
              "groups": group_info}
    files = {"pooled": "pooled_metrics.csv", "folds": "per_fold_metrics.csv", "fold_summary": "fold_summary.csv",
             "ci": "bootstrap_ci.csv", "comparisons": "model_comparisons.csv", "search": "head_search.csv"}
    for key, filename in files.items():
        tables[key].to_csv(out / filename, index=False)
    _save_predictions(oof, df, groups, classes, out / "test_predictions.csv.gz")

    report.write_report(cfg, tables, oof, protocols, df, groups, classes, time.time() - t_start)
    log.info("Done in %.1f min. Report: %s", (time.time() - t_start) / 60, out / "REPORT.md")
    return tables
