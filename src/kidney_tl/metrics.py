"""Classification metrics and scan-level (cluster) bootstrap statistics.

Slices from one scan are strongly correlated, so treating them as independent
would make confidence intervals far too narrow. All intervals and model
comparisons therefore resample whole scan groups.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

Metric = Callable[[np.ndarray, np.ndarray], float]


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None,
                           class_names: Sequence[str]) -> dict[str, float]:
    """Overall and per-class metrics. `y_pred` must hold class indices, not probabilities.

    Macro averages weight every class equally, so a weak minority class (Stone)
    cannot hide behind the large Normal class.
    """
    labels = np.arange(len(class_names))
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn = cm.sum() - cm.sum(axis=0) - cm.sum(axis=1) + np.diag(cm)
    fp = cm.sum(axis=0) - np.diag(cm)
    specificity = np.divide(tn, tn + fp, out=np.zeros(len(labels)), where=(tn + fp) > 0)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": prec.mean(),
        "macro_recall": rec.mean(),
        "macro_f1": f1.mean(),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_prob is not None and np.unique(y_true).size == len(labels):
        out["roc_auc_macro_ovr"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=labels)
    for i, name in enumerate(class_names):
        out[f"{name}_precision"] = prec[i]
        out[f"{name}_recall"] = rec[i]
        out[f"{name}_f1"] = f1[i]
        out[f"{name}_specificity"] = specificity[i]
        out[f"{name}_support"] = int(support[i])
    return out


def fast_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4) -> float:
    """Macro-F1 from a bincount confusion matrix (fast enough for thousands of bootstrap draws)."""
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(n_classes), where=denom > 0)
    return float(f1.mean())


def fast_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4) -> float:
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    support = cm.sum(axis=1)
    recall = np.divide(np.diag(cm), support, out=np.zeros(n_classes), where=support > 0)
    return float(recall[support > 0].mean())


def _cluster_draws(clusters: np.ndarray, n_samples: int, seed: int):
    """Yield index arrays, each a resample of whole clusters with replacement."""
    rng = np.random.default_rng(seed)
    ids, inverse = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(ids))]
    for _ in range(n_samples):
        pick = rng.integers(0, len(ids), len(ids))
        yield np.concatenate([members[i] for i in pick])


def cluster_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, clusters: np.ndarray, metric: Metric,
                         n_samples: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI for a metric, resampling scan groups rather than images."""
    stats = np.array([metric(y_true[s], y_pred[s]) for s in _cluster_draws(clusters, n_samples, seed)])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_cluster_bootstrap(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, clusters: np.ndarray,
                             metric: Metric, n_samples: int = 2000, seed: int = 0) -> dict[str, float]:
    """Difference metric(A) - metric(B) on the same images, with a scan-level bootstrap CI and
    two-sided p-value (the share of resamples on the other side of zero, doubled)."""
    diffs = np.array([metric(y_true[s], pred_a[s]) - metric(y_true[s], pred_b[s])
                      for s in _cluster_draws(clusters, n_samples, seed)])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"difference": metric(y_true, pred_a) - metric(y_true, pred_b), "ci_low": float(lo),
            "ci_high": float(hi), "p_value": float(min(1.0, max(p, 1 / n_samples)))}


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values for multiple pairwise comparisons."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted
