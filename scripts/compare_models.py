#!/usr/bin/env python3
"""Five-model comparison report (results/MODEL_COMPARISON.md) from the finished runs.

    python scripts/compare_models.py --data-dir data/ct-kidney

Uses: stage-1 results (results/pooled_metrics.csv), stage-2 frozen-feature pipelines
(results/improvements/pipelines.csv) and the equal-budget fine-tuning runs (results/stagewise/<model>/).
All numbers on unseen scans come from the same group-level folds.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kidney_tl.config import PAPER_MODELS, ExperimentConfig  # noqa: E402

# Accuracy published for this dataset with random (image-level) splits; leakage makes these optimistic.
LITERATURE = [
    ("VGG16", "98.20%", "Islam et al., Sci. Rep. 2022 (dataset paper), random 80/20 split",
     "https://pmc.ncbi.nlm.nih.gov/articles/PMC9259587/"),
    ("ResNet50", "73.80%", "Islam et al., Sci. Rep. 2022, random 80/20 split",
     "https://pmc.ncbi.nlm.nih.gov/articles/PMC9259587/"),
    ("InceptionV3", "61.60%", "Islam et al., Sci. Rep. 2022, random 80/20 split",
     "https://pmc.ncbi.nlm.nih.gov/articles/PMC9259587/"),
    ("InceptionV3", "99.82%", "NCI Dublin MSc thesis 2024, random split", "https://norma.ncirl.ie/8792/"),
    ("MobileNet (V2)", "99.64% (val.)", "BEEI 2025, random split", "https://beei.org/index.php/EEI/article/view/10084"),
    ("VGG16 / InceptionV3 / MobileNetV2", "99.21% / 97.38% / 95.29%", "BRAC University thesis 2022, random split",
     "https://dspace.bracu.ac.bd/items/0e04b749-0932-44bf-8525-52bdac06b2f5"),
    ("ResNet-50 / Inception-v3 / MobileNet-v3", "82.35% / 87.64% / 85.30%",
     "NephroNet benchmark 2025, file-name-based patient grouping", "https://pmc.ncbi.nlm.nih.gov/articles/PMC13337462"),
    ("Kidney-stone CT detector (other dataset)", "99.1% random vs 90.5% patient-wise",
     "SSRN leakage reassessment, three centres", "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7117201"),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    p.add_argument("--models", nargs="+", default=list(PAPER_MODELS))
    p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import confusion_matrix

    from kidney_tl.data import image_signatures, index_images
    from kidney_tl.grouping import assign_groups
    from kidney_tl.improve import scan_level
    from kidney_tl.metrics import (
        classification_metrics,
        cluster_bootstrap_ci,
        fast_accuracy,
        fast_macro_f1,
        holm_adjust,
        paired_cluster_bootstrap,
    )
    from kidney_tl.report import md_table
    from kidney_tl.splits import GROUP_CV, IMAGE_CV

    cfg = ExperimentConfig(data_dir=args.data_dir, output_dir=args.results_dir, cache_dir=args.cache_dir,
                           bootstrap_samples=args.bootstrap)
    out, fig_dir = cfg.output_dir, cfg.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = index_images(cfg.data_dir)
    classes = df.attrs["classes"]
    n_cls = len(classes)
    y = df["label"].to_numpy()
    md5, thumbs = image_signatures(df, cfg.cache_dir)
    groups, _ = assign_groups(df, md5, thumbs, cfg.scan_break_r, cfg.duplicate_r)
    position = pd.Series(np.arange(len(df)), index=df["path"])
    mf1 = partial(fast_macro_f1, n_classes=n_cls)

    def load_probs(path: Path) -> np.ndarray:
        d = pd.read_csv(path)
        prob = np.full((len(df), n_cls), np.nan)
        prob[position[d.path].to_numpy()] = d[[f"prob_{c}" for c in classes]].to_numpy()
        assert not np.isnan(prob).any(), path
        return prob

    probs = {m: load_probs(out / "stagewise" / m / "test_predictions.csv.gz") for m in args.models
             if (out / "stagewise" / m / "test_predictions.csv.gz").exists()}
    missing = [m for m in args.models if m not in probs]
    if missing:
        raise SystemExit(f"stage-wise runs not finished for: {missing}")
    ens_name = f"Ensemble of all {len(args.models)}"
    probs[ens_name] = np.mean([probs[m] for m in args.models], axis=0)

    def summarise(name: str, prob: np.ndarray) -> dict:
        pred = prob.argmax(axis=1)
        m = classification_metrics(y, pred, prob, classes)
        s_ids, s_y, s_pred = scan_level(prob, groups, y)
        row = {"model": name, **m, "scan_accuracy": fast_accuracy(s_y, s_pred), "scan_macro_f1": mf1(s_y, s_pred)}
        row["acc_ci"] = cluster_bootstrap_ci(y, pred, groups, fast_accuracy, cfg.bootstrap_samples)
        row["f1_ci"] = cluster_bootstrap_ci(y, pred, groups, mf1, cfg.bootstrap_samples)
        row["scan_acc_ci"] = cluster_bootstrap_ci(s_y, s_pred, s_ids, fast_accuracy, cfg.bootstrap_samples)
        return row

    rows = [summarise(name, prob) for name, prob in probs.items()]
    ft = pd.DataFrame(rows)
    ft.drop(columns=["acc_ci", "f1_ci", "scan_acc_ci"]).to_csv(out / "model_comparison_finetuned.csv", index=False)

    def ci(v, c):
        return f"{v:.4f} [{c[0]:.4f}, {c[1]:.4f}]"

    main_tbl = pd.DataFrame({
        "Model": ft.model,
        "Slice accuracy [95% CI]": [ci(v, c) for v, c in zip(ft.accuracy, ft.acc_ci, strict=True)],
        "Slice macro-F1 [95% CI]": [ci(v, c) for v, c in zip(ft.macro_f1, ft.f1_ci, strict=True)],
        "Balanced acc.": ft.balanced_accuracy.map("{:.4f}".format),
        "Cohen κ": ft.cohen_kappa.map("{:.4f}".format),
        "MCC": ft.mcc.map("{:.4f}".format),
        "ROC-AUC": ft.roc_auc_macro_ovr.map("{:.4f}".format),
        "Scan accuracy [95% CI]": [ci(v, c) for v, c in zip(ft.scan_accuracy, ft.scan_acc_ci, strict=True)],
        "Scan macro-F1": ft.scan_macro_f1.map("{:.4f}".format),
    })
    per_class = pd.DataFrame({"Model": ft.model, **{f"{c} {k}": ft[f"{c}_{k}"].map("{:.3f}".format)
                                                    for c in classes for k in ("precision", "recall", "f1")}})

    # Every evaluation of each model, so the effect of each step is visible
    pooled = pd.read_csv(out / "pooled_metrics.csv")
    pipes = pd.read_csv(out / "improvements" / "pipelines.csv").set_index("pipeline")
    ladder = []
    for m in args.models:
        img = pooled[(pooled.protocol == IMAGE_CV) & (pooled.model == m)].iloc[0]
        grp = pooled[(pooled.protocol == GROUP_CV) & (pooled.model == m)].iloc[0]
        lr = pipes.loc[f"LogReg · {m}"]
        f = ft[ft.model == m].iloc[0]
        ladder.append({"Model": m,
                       "Image-level CV, frozen (leaky)": f"{img.accuracy:.4f}",
                       "Unseen scans: frozen + MLP": f"{grp.accuracy:.4f}",
                       "Unseen scans: frozen + LogReg": f"{lr.slice_accuracy:.4f}",
                       "Unseen scans: fine-tuned (equal budget)": f"{f.accuracy:.4f}",
                       "Unseen scans: fine-tuned, per scan": f"{f.scan_accuracy:.4f}"})
    best = ft[ft.model.isin(args.models)].sort_values("macro_f1", ascending=False)

    comps = []
    for a, b in itertools.combinations(args.models, 2):
        r = paired_cluster_bootstrap(y, probs[a].argmax(1), probs[b].argmax(1), groups, mf1, cfg.bootstrap_samples,
                                     seed=1)
        comps.append({"Model A": a, "Model B": b, **r})
    comps = pd.DataFrame(comps)
    comps["p_holm"] = holm_adjust(comps.p_value)
    comp_tbl = pd.DataFrame({"Model A": comps["Model A"], "Model B": comps["Model B"],
                             "Δ macro-F1 (A − B) [95% CI]": [f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]" for d, lo, hi in
                                                             zip(comps.difference, comps.ci_low, comps.ci_high,
                                                                 strict=True)],
                             "p (Holm)": comps.p_holm.map("{:.3g}".format),
                             "Significant": np.where(comps.p_holm < 0.05, "yes", "no")})
    comps.to_csv(out / "model_comparison_pairwise.csv", index=False)

    names = list(probs)
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.7), squeeze=False)
    for ax, name in zip(axes[0], names, strict=True):
        cm = confusion_matrix(y, probs[name].argmax(1), labels=range(n_cls))
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(n_cls):
            for j in range(n_cls):
                ax.text(j, i, f"{100 * norm[i, j]:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.6 else "black")
        ax.set_xticks(range(n_cls), classes, rotation=45, fontsize=7)
        ax.set_yticks(range(n_cls), classes, fontsize=7)
        ax.set_title(name, fontsize=9)
    fig.suptitle("Fine-tuned models on unseen scans: recall per class (rows = true class)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "model_comparison_confusion.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(names))
    for k, (col, cicol, label) in enumerate((("accuracy", "acc_ci", "slice accuracy"),
                                             ("scan_accuracy", "scan_acc_ci", "scan accuracy"))):
        vals = ft.set_index("model").loc[names]
        err = np.array([[v - c[0], c[1] - v] for v, c in zip(vals[col], vals[cicol], strict=True)]).T
        ax.bar(x + (k - 0.5) * 0.38, vals[col], 0.38, yerr=err, capsize=3, label=label)
    ax.axhline(0.9, color="grey", ls="--", lw=1)
    ax.set_xticks(x, names, rotation=15)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Accuracy on unseen scans")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "model_comparison_accuracy.png", dpi=150)
    plt.close(fig)

    # Selective prediction: per fold, the confidence threshold is chosen on that fold's validation scans as the
    # lowest one whose retained validation slices reach the target accuracy; it is then applied to the test fold.
    from kidney_tl.splits import group_cv

    folds = group_cv(y, groups, cfg.n_folds, cfg.val_frac, cfg.seed)
    val_probs = {}
    for m in args.models:
        vp = np.full((len(df), n_cls), np.nan)
        for f in folds:
            d = pd.read_csv(out / "stagewise" / m / f"fold{f.fold}_val_predictions.csv.gz")
            vp[position[d.path].to_numpy()] = d[[f"prob_{c}" for c in classes]].to_numpy()
        val_probs[m] = vp
    val_probs[ens_name] = None  # filled per fold below

    def selective(name: str, target: float) -> dict:
        kept = np.zeros(len(df), dtype=bool)
        for f in folds:
            if name == ens_name:
                pv = np.mean([val_probs[m][f.val] for m in args.models], axis=0)
            else:
                pv = val_probs[name][f.val]
            conf, correct = pv.max(axis=1), pv.argmax(axis=1) == y[f.val]
            thr = 1.0
            for t in np.unique(conf):
                if correct[conf >= t].mean() >= target:
                    thr = t
                    break
            kept[f.test] = probs[name][f.test].max(axis=1) >= thr
        pred = probs[name].argmax(axis=1)
        acc = fast_accuracy(y[kept], pred[kept]) if kept.any() else float("nan")
        lo, hi = cluster_bootstrap_ci(y[kept], pred[kept], groups[kept], fast_accuracy, cfg.bootstrap_samples)
        per_class_cov = {f"{c} coverage": f"{kept[y == i].mean():.3f}" for i, c in enumerate(classes)}
        return {"Model": name, "Target (validation)": f"{target:.0%}", "Test accuracy on answered slices [95% CI]":
                f"{acc:.4f} [{lo:.4f}, {hi:.4f}]", "Coverage (share answered)": f"{kept.mean():.3f}", **per_class_cov}

    sel_tbl = pd.DataFrame([selective(n, t) for t in (0.90, 0.95, 0.99) for n in probs])
    sel_tbl.to_csv(out / "model_comparison_selective.csv", index=False)

    lit = pd.DataFrame(LITERATURE, columns=["Model", "Reported accuracy", "Source and split", "Link"])
    import json

    settings = json.loads((out / "stagewise" / args.models[0] / "config.json").read_text())
    stage_cfg = ", ".join(f"{k}={v}" for k, v in settings.items() if k not in ("model", "cut_after"))
    top = best.iloc[0]
    ens = ft[ft.model == ens_name].iloc[0]
    text = f"""# Five-model comparison under identical conditions

**Protocol (identical for every model):** group-level 5-fold CV on {len(df):,} images from {np.unique(groups).size} scans
(no scan in both training and test); 224×224 input after cropping to the body and padding to a square; each network's
own ImageNet weights and `preprocess_input`; the final stage of each network is fine-tuned (layers after the cut in
`src/kidney_tl/stagewise.py`) with a classifier head that also sees mid-level (cut-stage) features; class-balanced loss;
Adam, 2 warm-up epochs at 1e-3 then up to 12 at 1e-4 with early stopping on validation scans; flip and shift augmentation.
95% CIs come from a scan-level bootstrap ({cfg.bootstrap_samples:,} resamples). Settings: `{stage_cfg}`.

## 1. Results on unseen scans (fine-tuned, equal budget)

{md_table(main_tbl)}

**Ranking by macro-F1:** {", ".join(f"{r.model} ({r.macro_f1:.3f})" for r in best.itertuples())}.
Best single model: **{top.model}**, slice accuracy {top.accuracy:.4f}, scan accuracy {top.scan_accuracy:.4f}.
The pre-declared ensemble ({ens_name}) reaches slice accuracy {ens.accuracy:.4f} and scan accuracy {ens.scan_accuracy:.4f}.

![Accuracy per model](figures/model_comparison_accuracy.png)

## 2. Per class

{md_table(per_class)}

![Confusion matrices](figures/model_comparison_confusion.png)

## 3. Are the differences real? (paired scan-level bootstrap on macro-F1, Holm-corrected)

{md_table(comp_tbl)}

## 4. Reaching 90%+ honestly: answer only when confident (selective prediction)

Thresholds are chosen on validation scans only; slices below the threshold are referred for human review.

{md_table(sel_tbl)}

## 5. How each model's accuracy changes with the evaluation and training protocol

{md_table(pd.DataFrame(ladder))}

## 6. Published accuracies for this dataset (random image-level splits unless stated)

{md_table(lit)}

Random-split scores (95–100%) are matched by this study's image-level CV (≥ 99.8% for every model), because adjacent
slices of the same scan land in both training and test. They measure recognition of seen scans, not generalisation
to new patients.
"""
    (out / "MODEL_COMPARISON.md").write_text(text)
    print(f"Written {out / 'MODEL_COMPARISON.md'}")


if __name__ == "__main__":
    main()
