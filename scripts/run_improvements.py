#!/usr/bin/env python3
"""Stage 2: try stronger classifiers on the cached features and report them at image and scan level.

Run after scripts/run_experiment.py (it reuses that run's cached features, folds and
predictions):

    python scripts/run_improvements.py --data-dir data/ct-kidney
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kidney_tl.config import PAPER_MODELS, ExperimentConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    p.add_argument("--models", nargs="+", default=list(PAPER_MODELS))
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--weights", default="imagenet", help="'imagenet' or 'none' (must match stage 1)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples for CIs and comparisons")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--nice", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from kidney_tl.utils import limit_resources

    limit_resources(args.threads, args.nice)

    import numpy as np
    import pandas as pd

    from kidney_tl.data import image_signatures, index_images
    from kidney_tl.features import extract_features
    from kidney_tl.grouping import assign_groups
    from kidney_tl.improve import run_candidates, scan_level
    from kidney_tl.metrics import (
        classification_metrics,
        cluster_bootstrap_ci,
        fast_accuracy,
        fast_balanced_accuracy,
        fast_macro_f1,
        holm_adjust,
        paired_cluster_bootstrap,
    )
    from kidney_tl.report import md_table
    from kidney_tl.splits import GROUP_CV, group_cv
    from kidney_tl.utils import get_logger

    cfg = ExperimentConfig(data_dir=args.data_dir, output_dir=args.results_dir, cache_dir=args.cache_dir,
                           models=tuple(args.models), img_size=args.img_size, n_folds=args.folds,
                           weights=None if args.weights.lower() == "none" else args.weights, threads=args.threads,
                           bootstrap_samples=args.bootstrap)
    out = cfg.output_dir / "improvements"
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger(out / "run.log")
    t0 = time.time()

    df = index_images(cfg.data_dir)
    classes = df.attrs["classes"]
    n_cls = len(classes)
    y = df["label"].to_numpy()
    md5, thumbs = image_signatures(df, cfg.cache_dir)
    groups, _ = assign_groups(df, md5, thumbs, cfg.scan_break_r, cfg.duplicate_r)
    folds = group_cv(y, groups, cfg.n_folds, cfg.val_frac, cfg.seed)
    features = {m: extract_features(df.attrs["root"], df["path"].tolist(), m, cfg.img_size, cfg.weights,
                                    cfg.feature_batch_size, cfg.cache_dir) for m in cfg.models}
    log.info("Loaded cached features for %s", ", ".join(cfg.models))

    stage2 = run_candidates(features, folds, y, n_cls, log=log.info)
    candidates = {name: c.prob for name, c in stage2.items()}
    chosen = stage2["Selected per fold on validation"].chosen

    # Stage-1 heads (MLP per backbone) as baselines, from the saved out-of-fold predictions
    stage1 = pd.read_csv(cfg.output_dir / "test_predictions.csv.gz")
    stage1 = stage1[stage1.protocol == GROUP_CV]
    position = pd.Series(np.arange(len(df)), index=df["path"])
    for model, d in stage1.groupby("model", sort=False):
        prob = np.full((len(df), n_cls), np.nan)
        prob[position[d.path].to_numpy()] = d[[f"prob_{c}" for c in classes]].to_numpy()
        candidates[f"Stage 1 MLP · {model}"] = prob

    # Stage-3 fine-tuned models (runs that covered every fold). Their mean is the pre-declared final model.
    stage2_selected = stage2["Selected per fold on validation"]
    runs = []
    for pred_file in sorted((cfg.output_dir / "finetune").glob("*/test_predictions.csv.gz")):
        d = pd.read_csv(pred_file)
        prob = np.full((len(df), n_cls), np.nan)
        prob[position[d.path].to_numpy()] = d[[f"prob_{c}" for c in classes]].to_numpy()
        val_f1 = {int(k): v for k, v in json.loads((pred_file.parent / "val_macro_f1.json").read_text()).items()}
        runs.append((d["model"].iloc[0], prob, val_f1))
        candidates[d["model"].iloc[0]] = prob
        log.info("Included %s", d["model"].iloc[0])
    if len(runs) > 1:
        candidates[f"Fine-tuned ensemble (mean of {len(runs)} runs)"] = np.mean([p for _, p, _ in runs], axis=0)
    if runs:
        combined = stage2_selected.prob.copy()
        for f in folds:
            best_name, best_prob, best_f1 = None, None, stage2_selected.val_f1[f.fold]
            for name, prob, val_f1 in runs:
                if val_f1[f.fold] > best_f1:
                    best_name, best_prob, best_f1 = name, prob, val_f1[f.fold]
            if best_name is not None:
                combined[f.test] = best_prob[f.test]
                chosen[f.fold] = best_name
        candidates["Selected per fold on validation (incl. fine-tuned)"] = combined

    acc = fast_accuracy
    bal = partial(fast_balanced_accuracy, n_classes=n_cls)
    mf1 = partial(fast_macro_f1, n_classes=n_cls)
    rows, scan_preds = [], {}
    for name, prob in candidates.items():
        assert not np.isnan(prob).any(), name
        pred = prob.argmax(axis=1)
        m = classification_metrics(y, pred, prob, classes)
        s_ids, s_y, s_pred = scan_level(prob, groups, y)
        scan_preds[name] = s_pred
        row = {"pipeline": name, "slice_accuracy": m["accuracy"], "slice_balanced_accuracy": m["balanced_accuracy"],
               "slice_macro_f1": m["macro_f1"], "slice_mcc": m["mcc"], "slice_roc_auc": m.get("roc_auc_macro_ovr")}
        for key, fn in (("slice_accuracy", acc), ("slice_macro_f1", mf1)):
            row[f"{key}_ci"] = cluster_bootstrap_ci(y, pred, groups, fn, cfg.bootstrap_samples)
        row["scan_accuracy"] = acc(s_y, s_pred)
        row["scan_balanced_accuracy"] = bal(s_y, s_pred)
        row["scan_macro_f1"] = mf1(s_y, s_pred)
        for key, fn in (("scan_accuracy", acc), ("scan_macro_f1", mf1)):
            row[f"{key}_ci"] = cluster_bootstrap_ci(s_y, s_pred, s_ids, fn, cfg.bootstrap_samples)
        row["n_scans"] = len(s_ids)
        for i, c in enumerate(classes):
            row[f"slice_{c}_f1"] = m[f"{c}_f1"]
            row[f"scan_{c}_recall"] = float(np.mean(s_pred[s_y == i] == i)) if (s_y == i).any() else np.nan
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("slice_macro_f1", ascending=False)
    table.to_csv(out / "pipelines.csv", index=False)

    # Paired comparisons of every stage-2 pipeline against the best stage-1 model, at both levels
    best_stage1 = table[table.pipeline.str.startswith("Stage 1")].iloc[0].pipeline
    s_ids, s_y, _ = scan_level(candidates[best_stage1], groups, y)
    comps = []
    for name in table.pipeline:
        if name == best_stage1:
            continue
        sl = paired_cluster_bootstrap(y, candidates[name].argmax(1), candidates[best_stage1].argmax(1), groups, mf1,
                                      cfg.bootstrap_samples, seed=1)
        sc = paired_cluster_bootstrap(s_y, scan_preds[name], scan_preds[best_stage1], s_ids, mf1,
                                      cfg.bootstrap_samples, seed=1)
        comps.append({"pipeline": name, "slice_diff": sl["difference"], "slice_ci": (sl["ci_low"], sl["ci_high"]),
                      "slice_p": sl["p_value"], "scan_diff": sc["difference"], "scan_ci": (sc["ci_low"], sc["ci_high"]),
                      "scan_p": sc["p_value"]})
    comps = pd.DataFrame(comps)
    comps["slice_p_holm"] = holm_adjust(comps.slice_p)
    comps["scan_p_holm"] = holm_adjust(comps.scan_p)
    comps.to_csv(out / "comparisons_vs_stage1.csv", index=False)

    def ci(v, c):
        return f"{v:.4f} [{c[0]:.4f}, {c[1]:.4f}]"

    main_tbl = pd.DataFrame({
        "Pipeline": table.pipeline,
        "Slice accuracy [95% CI]": [ci(v, c) for v, c in zip(table.slice_accuracy, table.slice_accuracy_ci,
                                                                strict=True)],
        "Slice macro-F1 [95% CI]": [ci(v, c) for v, c in zip(table.slice_macro_f1, table.slice_macro_f1_ci,
                                                                strict=True)],
        "Slice balanced acc.": table.slice_balanced_accuracy.map("{:.4f}".format),
        "Slice MCC": table.slice_mcc.map("{:.4f}".format),
        "Scan accuracy [95% CI]": [ci(v, c) for v, c in zip(table.scan_accuracy, table.scan_accuracy_ci, strict=True)],
        "Scan macro-F1 [95% CI]": [ci(v, c) for v, c in zip(table.scan_macro_f1, table.scan_macro_f1_ci,
                                                              strict=True)],
    })
    per_class = pd.DataFrame({"Pipeline": table.pipeline,
                              **{f"{c} slice F1": table[f"slice_{c}_f1"].map("{:.3f}".format) for c in classes},
                              **{f"{c} scan recall": table[f"scan_{c}_recall"].map("{:.3f}".format)
                                 for c in classes}})
    comp_tbl = pd.DataFrame({
        "Pipeline": comps.pipeline,
        "Δ slice macro-F1 [95% CI]": [f"{d:+.4f} [{c[0]:+.4f}, {c[1]:+.4f}]"
                                      for d, c in zip(comps.slice_diff, comps.slice_ci, strict=True)],
        "p (Holm)": comps.slice_p_holm.map("{:.3g}".format),
        "Δ scan macro-F1 [95% CI]": [f"{d:+.4f} [{c[0]:+.4f}, {c[1]:+.4f}]"
                                     for d, c in zip(comps.scan_diff, comps.scan_ci, strict=True)],
        "p (Holm) ": comps.scan_p_holm.map("{:.3g}".format),
    })
    pilot_rows = []
    for fold0 in sorted((cfg.output_dir / "finetune").glob("*/fold0.json")):
        if (fold0.parent / "test_predictions.csv.gz").exists():
            continue  # the final all-fold run, reported above; this table lists the pilots
        info = json.loads(fold0.read_text())
        setup = json.loads((fold0.parent / "finetune_config.json").read_text())
        pilot_rows.append({"Run": fold0.parent.name, "Backbone": setup["backbone"],
                           "Slices used": f"1 in {setup['stride']}", "Unfrozen layers": setup["unfreeze"],
                           "Image size": setup["img_size"], "Fine-tune LR": f"{setup['finetune_lr']:g}",
                           "Fold-0 val macro-F1 (thinned val)": f"{info.get('val_macro_f1_thinned', info['val_macro_f1']):.4f}",
                           "Fold-0 val macro-F1 (full val)": (f"{info['val_macro_f1']:.4f}"
                                                              if "val_macro_f1_thinned" in info else "not recorded"),
                           "Minutes": f"{info['minutes']:.0f}"})
    pilot_md = md_table(pd.DataFrame(pilot_rows)) if pilot_rows else "_No fine-tuning runs found._"

    sel_name = ("Selected per fold on validation (incl. fine-tuned)"
                if "Selected per fold on validation (incl. fine-tuned)" in set(table.pipeline)
                else "Selected per fold on validation")
    sel = table[table.pipeline == sel_name].iloc[0]
    text = f"""# Stage 2: stronger classifiers on the same group-level folds

All pipelines use the {GROUP_CV} folds of stage 1 ({cfg.n_folds} folds, {int(sel.n_scans)} scans, {len(df):,} images).
Each is tuned on the validation scans of each training fold; the test fold is scored once. "Selected per fold on
validation" picks, in every fold, the stage-2 pipeline with the best validation macro-F1, so its score is the unbiased
estimate for the whole tuning procedure. **Scan level**: the log-probabilities of all slices of a scan are averaged
and the scan receives one label. 95% CIs come from a bootstrap over scans ({cfg.bootstrap_samples:,} resamples).

## Results (sorted by slice-level macro-F1)

{md_table(main_tbl)}

**Nested-selection estimate ({sel_name}):** slice accuracy
{ci(sel.slice_accuracy, sel.slice_accuracy_ci)}, scan accuracy {ci(sel.scan_accuracy, sel.scan_accuracy_ci)}. Pipeline chosen in each fold:
{"; ".join(f"fold {k}: {v}" for k, v in sorted(chosen.items()))}.

## Per class

{md_table(per_class)}

## Fine-tuning settings tried (chosen on fold-0 validation scans only)

Scores are on the fold-0 validation scans, using the same 1-in-*stride* slices that guided early stopping
(runs with stride 2 share exactly the same validation images).

{pilot_md}

## Difference from the best stage-1 model ({best_stage1}), paired scan-level bootstrap, Holm-corrected

{md_table(comp_tbl)}
"""
    (cfg.output_dir / "IMPROVEMENTS.md").write_text(text)
    log.info("Done in %.1f min. Report: %s", (time.time() - t0) / 60, cfg.output_dir / "IMPROVEMENTS.md")


if __name__ == "__main__":
    main()
