"""Stage 2: stronger classifiers on the cached backbone features, under the same group-level folds.

Every candidate is trained on the training part of each outer fold and tuned on that
fold's validation scans only; the outer test fold is scored once. An extra "selected"
pipeline picks, inside each fold, whichever candidate has the best validation macro-F1,
so its score is an honest estimate of the whole tuning procedure (nested selection).

Two prediction levels are reported:
* slice level - one prediction per image, as in stage 1;
* scan level  - the log-probabilities of all slices of a scan are averaged and the scan
  gets one label, which is how a model would be used on a patient's study.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .metrics import fast_macro_f1
from .splits import Split


@dataclass
class Candidate:
    """Out-of-fold test probabilities plus per-fold validation scores for one pipeline."""

    name: str
    prob: np.ndarray
    val_f1: dict[int, float] = field(default_factory=dict)
    chosen: dict[int, str] = field(default_factory=dict)


def fit_logreg(x_tr, y_tr, x_va, y_va, x_te, cs, n_classes, pca_dim=None, seed=0):
    """Standardise (and optionally PCA) on the training fold, choose C on validation macro-F1."""
    scaler = StandardScaler().fit(x_tr)
    x_tr, x_va, x_te = scaler.transform(x_tr), scaler.transform(x_va), scaler.transform(x_te)
    if pca_dim is not None and pca_dim < x_tr.shape[1]:
        dim = min(pca_dim, x_tr.shape[0] - 1)  # PCA cannot have more components than training samples
        pca = PCA(n_components=dim, random_state=seed).fit(x_tr)
        x_tr, x_va, x_te = pca.transform(x_tr), pca.transform(x_va), pca.transform(x_te)
    best = None
    for c in cs:
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=3000, random_state=seed)
        clf.fit(x_tr, y_tr)
        p_va = clf.predict_proba(x_va)
        f1 = fast_macro_f1(y_va, p_va.argmax(axis=1), n_classes)
        if best is None or f1 > best[0]:
            best = (f1, c, p_va, clf.predict_proba(x_te))
    return best  # (val_f1, C, val_prob, test_prob)


def run_candidates(features: dict[str, np.ndarray], folds: list[Split], y: np.ndarray, n_classes: int,
                   cs=(1e-3, 1e-2, 1e-1, 1.0), fusion_pca: int = 512, log=print) -> dict[str, Candidate]:
    """Per-backbone logistic regression, feature fusion, probability ensemble, and nested selection."""
    n = len(y)
    names = list(features)
    cands = {f"LogReg · {b}": Candidate(f"LogReg · {b}", np.full((n, n_classes), np.nan)) for b in names}
    cands["LogReg · fusion of all backbones"] = Candidate("LogReg · fusion of all backbones",
                                                          np.full((n, n_classes), np.nan))
    cands["Ensemble · mean of backbone LogRegs"] = Candidate("Ensemble · mean of backbone LogRegs",
                                                             np.full((n, n_classes), np.nan))
    selected = Candidate("Selected per fold on validation", np.full((n, n_classes), np.nan))

    for f in folds:
        val_probs, test_probs = {}, {}
        for b in names:
            x = features[b]
            f1, c, p_va, p_te = fit_logreg(x[f.train], y[f.train], x[f.val], y[f.val], x[f.test], cs, n_classes,
                                            seed=f.fold)
            key = f"LogReg · {b}"
            cands[key].prob[f.test] = p_te
            cands[key].val_f1[f.fold] = f1
            val_probs[key], test_probs[key] = p_va, p_te
            log(f"  fold {f.fold} {key:28s} C={c:g} val macro-F1 {f1:.4f}")

        fused = np.concatenate([features[b] for b in names], axis=1)
        f1, c, p_va, p_te = fit_logreg(fused[f.train], y[f.train], fused[f.val], y[f.val], fused[f.test], cs,
                                        n_classes, pca_dim=fusion_pca, seed=f.fold)
        key = "LogReg · fusion of all backbones"
        cands[key].prob[f.test], cands[key].val_f1[f.fold] = p_te, f1
        val_probs[key], test_probs[key] = p_va, p_te
        log(f"  fold {f.fold} {key:28s} C={c:g} val macro-F1 {f1:.4f}")

        backbone_keys = [f"LogReg · {b}" for b in names]
        p_va = np.mean([val_probs[k] for k in backbone_keys], axis=0)
        p_te = np.mean([test_probs[k] for k in backbone_keys], axis=0)
        key = "Ensemble · mean of backbone LogRegs"
        cands[key].prob[f.test] = p_te
        cands[key].val_f1[f.fold] = fast_macro_f1(y[f.val], p_va.argmax(axis=1), n_classes)
        val_probs[key], test_probs[key] = p_va, p_te

        winner = max(val_probs, key=lambda k: cands[k].val_f1[f.fold])
        selected.prob[f.test] = test_probs[winner]
        selected.val_f1[f.fold] = cands[winner].val_f1[f.fold]
        selected.chosen[f.fold] = winner
        log(f"  fold {f.fold} selected on validation: {winner}")

    cands[selected.name] = selected
    return cands


def scan_level(prob: np.ndarray, groups: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average log-probabilities over each scan group. Returns (scan ids, scan labels, scan predictions).

    Every scan has a single class in this dataset, which the assertion checks.
    """
    frame = pd.DataFrame({"g": groups, "y": y})
    if frame.groupby("g")["y"].nunique().max() != 1:
        raise AssertionError("a scan group contains more than one class")
    logp = pd.DataFrame(np.log(np.clip(prob, 1e-7, 1.0)))
    mean_logp = logp.groupby(groups).mean()
    scan_y = frame.groupby("g")["y"].first().reindex(mean_logp.index).to_numpy()
    return mean_logp.index.to_numpy(), scan_y, mean_logp.to_numpy().argmax(axis=1)
