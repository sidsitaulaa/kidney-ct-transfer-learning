"""Figures and the markdown results report."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from .splits import GROUP_CV, IMAGE_CV  # noqa: E402


def md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table (no extra dependencies)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _with_ci(ci: pd.DataFrame, protocol: str, model: str, metric: str) -> str:
    r = ci[(ci.protocol == protocol) & (ci.model == model) & (ci.metric == metric)].iloc[0]
    return f"{r.value:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}]"


def results_table(pooled: pd.DataFrame, ci: pd.DataFrame, fold_summary: pd.DataFrame, protocol: str) -> pd.DataFrame:
    """Pooled out-of-fold metrics with scan-level 95% CIs, plus the spread across folds."""
    p = pooled[pooled.protocol == protocol].sort_values("macro_f1", ascending=False)
    fs = fold_summary[fold_summary.protocol == protocol].set_index("model")
    rows = []
    for r in p.itertuples(index=False):
        f = fs.loc[r.model]
        auc = getattr(r, "roc_auc_macro_ovr", np.nan)
        spread = f"{f.accuracy_mean:.4f}" if pd.isna(f.accuracy_std) else f"{f.accuracy_mean:.4f} ± {f.accuracy_std:.4f}"
        rows.append({
            "Model": r.model,
            "Accuracy [95% CI]": _with_ci(ci, protocol, r.model, "accuracy"),
            "Balanced accuracy [95% CI]": _with_ci(ci, protocol, r.model, "balanced_accuracy"),
            "Macro-F1 [95% CI]": _with_ci(ci, protocol, r.model, "macro_f1"),
            "Macro precision": f"{r.macro_precision:.4f}",
            "Macro recall": f"{r.macro_recall:.4f}",
            "Cohen κ": f"{r.cohen_kappa:.4f}",
            "MCC": f"{r.mcc:.4f}",
            "ROC-AUC": "" if pd.isna(auc) else f"{auc:.4f}",
            "Accuracy per fold (mean ± SD)": spread,
        })
    return pd.DataFrame(rows)


def plot_confusions(oof, protocol, models, classes, y, path: Path) -> None:
    """Row-normalised confusion matrices (recall per class) with counts, one panel per model."""
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.9), squeeze=False)
    for ax, model in zip(axes[0], models, strict=True):
        idx, pred, _ = oof[(protocol, model)]
        cm = confusion_matrix(y[idx], pred, labels=range(len(classes)))
        norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{cm[i, j]}\n{100 * norm[i, j]:.1f}%", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(classes)), classes, rotation=45, fontsize=8)
        ax.set_yticks(range(len(classes)), classes, fontsize=8)
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=8)
    axes[0][0].set_ylabel("True", fontsize=8)
    fig.suptitle(f"Out-of-fold confusion matrices, {protocol}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_protocol_comparison(ci: pd.DataFrame, models, protocols, path: Path) -> None:
    """Accuracy and macro-F1 per model under each protocol, with scan-level 95% CIs."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    width = 0.8 / len(protocols)
    x = np.arange(len(models))
    for ax, metric in zip(axes, ("accuracy", "macro_f1"), strict=True):
        for k, protocol in enumerate(protocols):
            s = ci[(ci.protocol == protocol) & (ci.metric == metric)].set_index("model").reindex(models)
            err = np.vstack([s.value - s.ci_low, s.ci_high - s.value])
            ax.bar(x + k * width - 0.4 + width / 2, s.value, width, yerr=err, capsize=2, label=protocol)
        ax.set_xticks(x, models, rotation=20)
        ax.set_title(metric.replace("_", "-"))
        ax.set_ylim(max(0.0, ci[ci.metric == metric].ci_low.min() - 0.05), 1.0)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Pooled test score")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_class_f1(pooled: pd.DataFrame, protocol: str, classes, path: Path) -> None:
    s = pooled[pooled.protocol == protocol].set_index("model")
    data = s[[f"{c}_f1" for c in classes]].to_numpy()
    fig, ax = plt.subplots(figsize=(1.4 * len(classes) + 2, 0.5 * len(s) + 1.5))
    ax.imshow(data, cmap="viridis", vmin=max(0.0, data.min() - 0.05), vmax=1)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", color="white", fontsize=9)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(s)), s.index)
    ax.set_title(f"Per-class F1, {protocol} (pooled out-of-fold)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _comparison_table(comparisons: pd.DataFrame, protocol: str, metric: str) -> str:
    if comparisons.empty:
        return "_Only one model was run, so there is nothing to compare._"
    c = comparisons[(comparisons.protocol == protocol) & (comparisons.metric == metric)]
    rows = [{"Model A": r.model_a, "Model B": r.model_b, "Difference (A − B)": f"{r.difference:+.4f}",
             "95% CI": f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]", "p": f"{r.p_value:.3g}", "p (Holm)": f"{r.p_holm:.3g}",
             "Significant at 0.05": "yes" if r.p_holm < 0.05 else "no"} for r in c.itertuples(index=False)]
    return md_table(pd.DataFrame(rows))


def _head_choices(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (protocol, model), d in folds.groupby(["protocol", "model"], sort=False):
        choices = ", ".join(f"{u}/{lr:g}" for u, lr in zip(d.head_units, d.head_lr, strict=True))
        rows.append({"Protocol": protocol, "Model": model, "Units / learning rate chosen in each fold": choices})
    return pd.DataFrame(rows)


def write_report(cfg, tables, oof, protocols, df, groups, classes, seconds: float) -> None:
    out = Path(cfg.output_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    y = df["label"].to_numpy()
    models = list(cfg.models)
    names = list(protocols)
    pooled, ci, fold_summary = tables["pooled"], tables["ci"], tables["fold_summary"]

    for name in names:
        plot_confusions(oof, name, models, classes, y, fig_dir / f"confusion_{name.split(' ')[0]}.png")
    plot_protocol_comparison(ci, models, names, fig_dir / "protocol_comparison.png")
    plot_per_class_f1(pooled, GROUP_CV, classes, fig_dir / "per_class_f1.png")

    env = json.loads((out / "environment.json").read_text())
    leak = tables["leakage"].rename(columns={
        "protocol": "Protocol", "folds": "Folds", "train_per_fold": "Train / fold", "val_per_fold": "Val / fold",
        "test_per_fold": "Test / fold", "test_file_in_train_pct": "Test file also in train (%)",
        "test_byte_identical_in_train_pct": "Test byte-identical to a train image (%)",
        "test_near_identical_in_train_pct": f"Test near-identical to a train image (r ≥ {cfg.duplicate_r}, %)",
        "test_scan_in_train_pct": "Test scan also in train (%)"}).round(1)
    groups_tbl = tables["groups"].rename(columns={
        "class": "Class", "images": "Images", "scan_segments": "Slice sequences", "groups": "Scan groups",
        "median_group_size": "Median images / group", "largest_group": "Largest group"})

    proto_rows = []
    for m in models:
        row = {"Model": m}
        for name in names:
            row[f"{name}: accuracy"] = _with_ci(ci, name, m, "accuracy")
            row[f"{name}: macro-F1"] = _with_ci(ci, name, m, "macro_f1")
        proto_rows.append(row)

    pc = pooled[pooled.protocol == GROUP_CV].set_index("model").reindex(models)
    pc_tbl = pd.DataFrame({"Model": models})
    for c in classes:
        for k in ("precision", "recall", "f1", "specificity"):
            pc_tbl[f"{c} {k}"] = pc[f"{c}_{k}"].map(lambda v: f"{v:.3f}").to_numpy()

    best = pooled[pooled.protocol == GROUP_CV].sort_values("macro_f1", ascending=False).iloc[0]
    n_settings = len(cfg.grid_units) * len(cfg.grid_dropout) * len(cfg.grid_lr)
    hardware = f"GPU {', '.join(env['gpus'])}" if env["gpus"] else f"CPU, {cfg.threads} threads"
    text = f"""# Results: leakage-aware transfer learning for kidney CT classification

Generated {dt.datetime.now():%Y-%m-%d %H:%M} in {seconds / 60:.0f} min · TensorFlow {env['tensorflow']} · {hardware} ·
{len(df):,} images · {int(np.unique(groups).size)} scan groups · classes {', '.join(classes)}

**Headline ({GROUP_CV}, the primary estimate):** the best model is **{best.model}**, with accuracy
**{_with_ci(ci, GROUP_CV, best.model, 'accuracy')}** and macro-F1 **{_with_ci(ci, GROUP_CV, best.model, 'macro_f1')}**
(pooled over {cfg.n_folds} folds; 95% CIs from {cfg.bootstrap_samples:,} scan-level bootstrap resamples).

## 1. Protocol

* {len(models)} {"ImageNet-pretrained" if cfg.weights else "randomly initialised"} backbones ({', '.join(models)}),
  frozen, each with its own `preprocess_input`, {cfg.img_size}×{cfg.img_size} input, global average pooling.
* Head: Dense → BatchNorm → Dropout → softmax, class-balanced loss, batch {cfg.batch_size},
  early stopping on validation loss (patience {cfg.patience}).
  The head setting ({n_settings} candidates) is chosen in each fold by **validation macro-F1**; the test fold is never used.
* **Scan groups.** Within each class, file numbers follow slice order. A new scan starts where consecutive slices stop
  correlating (r < {cfg.scan_break_r}); scans stored twice (byte-identical or r ≥ {cfg.duplicate_r} slices) are merged.
* **Protocols**
  * **{GROUP_CV} (primary):** stratified group {cfg.n_folds}-fold; every scan is tested once and never seen in training.
  * **{IMAGE_CV}:** stratified {cfg.n_folds}-fold by image, as in most published work on this dataset.
* Metrics are pooled over all out-of-fold predictions. CIs and model comparisons resample **whole scan groups**,
  because slices of one scan are not independent.

## 2. Data and scan groups

{md_table(groups_tbl)}

## 3. Leakage check

{md_table(leak)}

## 4. Main results: {GROUP_CV}

{md_table(results_table(pooled, ci, fold_summary, GROUP_CV))}

## 5. {IMAGE_CV}

{md_table(results_table(pooled, ci, fold_summary, IMAGE_CV))}

## 6. How much the evaluation protocol changes the numbers

{md_table(pd.DataFrame(proto_rows))}

![Protocol comparison](figures/protocol_comparison.png)

## 7. Per-class results, {GROUP_CV}

{md_table(pc_tbl)}

![Per-class F1](figures/per_class_f1.png)

![Confusion matrices](figures/confusion_group-level.png)

## 8. Are the differences between models real? ({GROUP_CV}, paired scan-level bootstrap, Holm-corrected)

Macro-F1:

{_comparison_table(tables['comparisons'], GROUP_CV, 'macro_f1')}

Accuracy:

{_comparison_table(tables['comparisons'], GROUP_CV, 'accuracy')}

## 9. Selected head settings

{md_table(_head_choices(tables['folds']))}

## 10. Files

`pooled_metrics.csv` (all metrics, pooled), `per_fold_metrics.csv`, `fold_summary.csv`, `bootstrap_ci.csv`,
`model_comparisons.csv`, `head_search.csv`, `leakage_report.csv`, `scan_groups.csv`,
`splits/*_folds.csv` (which file was in which subset of which fold), `test_predictions.csv.gz`
(every out-of-fold prediction with class probabilities), `config.json`, `environment.json`, `run.log`.
"""
    (out / "REPORT.md").write_text(text)
