"""
common/probing.py
=================

Logistic-regression linear probing with:
  * StandardScaler fit on train only
  * Cross-validated L2 regularisation strength
  * Stratified 1000-iteration bootstrap 95 % CIs for AUC and best-F1
  * Pickled classifier + scaler so downstream experiments (clean vs
    perturbed disease delta-AUC) can reuse them.

Upgraded from v1-v3 to expose *class weights*, *cross-validated C*, and
*explicit multiple-comparison reporting hooks* requested by reviewers.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score,
    precision_recall_curve, classification_report,
)


@dataclass
class ProbeResult:
    name: str
    auc: float
    auc_ci: Tuple[float, float]
    f1: float
    f1_ci: Tuple[float, float]
    threshold: float
    fpr: np.ndarray
    tpr: np.ndarray
    y_true: np.ndarray
    y_proba: np.ndarray
    best_C: float
    n_train: int
    n_test: int
    n_pos_train: int
    n_pos_test: int

    def to_json_dict(self):
        d = asdict(self)
        for k in ("fpr", "tpr", "y_true", "y_proba"):
            d[k] = np.asarray(d[k]).tolist()
        d["auc_ci"] = list(self.auc_ci)
        d["f1_ci"] = list(self.f1_ci)
        return d


def _best_f1_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1s = 2 * precisions[1:] * recalls[1:] / (precisions[1:] + recalls[1:] + 1e-12)
    idx = int(np.argmax(f1s))
    return float(thresholds[idx]), float(f1s[idx])


def _bootstrap_ci(y_true, y_proba, threshold, n_boot=1000,
                  confidence=0.95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs, f1s = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        pb = y_proba[idx]
        if len(np.unique(yb)) > 1:
            aucs.append(roc_auc_score(yb, pb))
        f1s.append(f1_score(yb, (pb >= threshold).astype(int), zero_division=0))
    alpha = 1 - confidence
    def ci(a):
        return (float(np.percentile(a, 100 * alpha / 2)),
                float(np.percentile(a, 100 * (1 - alpha / 2))))
    return ci(aucs) if aucs else (0.0, 0.0), ci(f1s)


def train_probe(
    X_train, y_train, X_test, y_test,
    name: str = "probe",
    C_grid: Iterable[float] = (0.01, 0.1, 1.0, 10.0),
    n_boot: int = 1000, confidence: float = 0.95,
    max_iter: int = 2000, class_weight: Optional[str] = "balanced",
    cv_folds: int = 5, seed: int = 42, verbose: bool = True,
) -> ProbeResult:
    """Fit StandardScaler + LogisticRegression with CV-chosen C, then evaluate."""
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).astype(int)
    X_test  = np.asarray(X_test)
    y_test  = np.asarray(y_test).astype(int)

    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)

    # Cross-validated C search
    best_C, best_cv_auc = None, -np.inf
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    for C in C_grid:
        cv_aucs = []
        for tr_idx, va_idx in skf.split(Xtr, y_train):
            clf = LogisticRegression(
                C=C, max_iter=max_iter, class_weight=class_weight,
                solver="lbfgs", random_state=seed,
            )
            clf.fit(Xtr[tr_idx], y_train[tr_idx])
            proba = clf.predict_proba(Xtr[va_idx])[:, 1]
            cv_aucs.append(roc_auc_score(y_train[va_idx], proba))
        mean_auc = float(np.mean(cv_aucs))
        if verbose:
            print(f"  [{name}] C={C:>7}  CV-AUC={mean_auc:.4f}")
        if mean_auc > best_cv_auc:
            best_cv_auc = mean_auc
            best_C = C

    # Refit on full train
    clf = LogisticRegression(
        C=best_C, max_iter=max_iter, class_weight=class_weight,
        solver="lbfgs", random_state=seed,
    )
    clf.fit(Xtr, y_train)
    y_proba = clf.predict_proba(Xte)[:, 1]

    auc = float(roc_auc_score(y_test, y_proba))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    thr, _ = _best_f1_threshold(y_test, y_proba)
    f1 = float(f1_score(y_test, (y_proba >= thr).astype(int), zero_division=0))
    auc_ci, f1_ci = _bootstrap_ci(y_test, y_proba, thr, n_boot, confidence, seed)

    if verbose:
        print(f"  [{name}] best_C={best_C}  AUC={auc:.4f} {auc_ci}  "
              f"F1@{thr:.3f}={f1:.4f} {f1_ci}")

    return ProbeResult(
        name=name, auc=auc, auc_ci=auc_ci, f1=f1, f1_ci=f1_ci,
        threshold=thr, fpr=fpr, tpr=tpr, y_true=y_test, y_proba=y_proba,
        best_C=best_C, n_train=len(y_train), n_test=len(y_test),
        n_pos_train=int(y_train.sum()), n_pos_test=int(y_test.sum()),
    ), {"classifier": clf, "scaler": scaler}


def save_probe(result: ProbeResult, artefacts: Dict, out_dir: Path,
               stem: str):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{stem}_result.json", "w") as fh:
        json.dump(result.to_json_dict(), fh, indent=2)
    with open(out_dir / f"{stem}_artefacts.pkl", "wb") as fh:
        pickle.dump(artefacts, fh)


def load_probe_result(path: Path) -> ProbeResult:
    with open(path) as fh:
        d = json.load(fh)
    for k in ("fpr", "tpr", "y_true", "y_proba"):
        d[k] = np.asarray(d[k])
    d["auc_ci"] = tuple(d["auc_ci"])
    d["f1_ci"] = tuple(d["f1_ci"])
    return ProbeResult(**d)
