"""
common/stats.py
===============

Formal statistical tests for model comparisons, addressing Reviewer 3 (RYAI)
and Reviewer 1 (Meta-Radiology) concerns:

  * DeLong test for two correlated AUCs on the SAME test set
  * Permutation test on AUC differences (distribution-free backup)
  * Benjamini-Hochberg FDR correction across the full comparison matrix
  * Paired bootstrap CIs on delta-AUC for clean vs perturbed contrasts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# DeLong test (fast algorithm by Sun & Xu, 2014)
# ---------------------------------------------------------------------------

def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T
    return T2


def _fast_delong(y_true, y_scores_a, y_scores_b):
    """
    Fast DeLong for TWO correlated ROC curves on the same y_true.
    Returns (auc_a, auc_b, delta, var_delta).
    """
    y_true = np.asarray(y_true).astype(int)
    positives = y_true == 1
    negatives = ~positives
    m = int(positives.sum())
    n = int(negatives.sum())
    assert m > 0 and n > 0

    def _aucs_and_cov(pred_a, pred_b):
        tx = np.stack([pred_a[positives], pred_b[positives]])
        ty = np.stack([pred_a[negatives], pred_b[negatives]])
        tz = np.stack([pred_a, pred_b])

        tx_rank = np.stack([_compute_midrank(tx[0]), _compute_midrank(tx[1])])
        ty_rank = np.stack([_compute_midrank(ty[0]), _compute_midrank(ty[1])])
        tz_rank = np.stack([_compute_midrank(tz[0]), _compute_midrank(tz[1])])

        aucs = np.zeros(2)
        for k in range(2):
            aucs[k] = (tz_rank[k][positives].sum() - m * (m + 1) / 2) / (m * n)

        v01 = np.zeros((2, m))
        v10 = np.zeros((2, n))
        for k in range(2):
            v01[k] = (tz_rank[k][positives] - tx_rank[k]) / n
            v10[k] = 1 - (tz_rank[k][negatives] - ty_rank[k]) / m

        sx = np.cov(v01, bias=False)
        sy = np.cov(v10, bias=False)
        delongcov = sx / m + sy / n
        return aucs, delongcov

    aucs, cov = _aucs_and_cov(np.asarray(y_scores_a), np.asarray(y_scores_b))
    delta = aucs[0] - aucs[1]
    var_delta = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    return float(aucs[0]), float(aucs[1]), float(delta), var_delta


def delong_test(y_true, y_scores_a, y_scores_b) -> Dict:
    """Two-sided DeLong test for paired AUCs."""
    auc_a, auc_b, delta, var = _fast_delong(y_true, y_scores_a, y_scores_b)
    if var <= 0 or np.isnan(var):
        return dict(auc_a=auc_a, auc_b=auc_b, delta=delta,
                    se=0.0, z=0.0, p_value=1.0, ci_low=delta, ci_high=delta)
    se = float(np.sqrt(var))
    z = delta / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    ci_low = delta - 1.96 * se
    ci_high = delta + 1.96 * se
    return dict(auc_a=auc_a, auc_b=auc_b, delta=delta,
                se=se, z=z, p_value=float(p),
                ci_low=float(ci_low), ci_high=float(ci_high))


# ---------------------------------------------------------------------------
# Permutation test on AUC difference (distribution-free backup)
# ---------------------------------------------------------------------------

def permutation_auc_test(y_true, y_scores_a, y_scores_b,
                         n_permutations=5000, seed=42) -> Dict:
    """
    Randomly swap per-sample scores between A and B; compute AUC diff null.
    Useful when DeLong's normal approximation is questionable
    (e.g. very small samples or near-ceiling AUCs).
    """
    y_true = np.asarray(y_true).astype(int)
    sa = np.asarray(y_scores_a, dtype=float)
    sb = np.asarray(y_scores_b, dtype=float)
    from sklearn.metrics import roc_auc_score

    obs = roc_auc_score(y_true, sa) - roc_auc_score(y_true, sb)
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        mask = rng.random(len(y_true)) < 0.5
        a_perm = np.where(mask, sb, sa)
        b_perm = np.where(mask, sa, sb)
        null[i] = roc_auc_score(y_true, a_perm) - roc_auc_score(y_true, b_perm)
    p = float((np.abs(null) >= abs(obs)).mean())
    return dict(obs_delta=float(obs), p_value=p,
                null_mean=float(null.mean()), null_std=float(null.std()))


# ---------------------------------------------------------------------------
# Multiple comparisons
# ---------------------------------------------------------------------------

def benjamini_hochberg(p_values, alpha=0.05) -> Dict:
    """BH-FDR.  Returns adjusted p-values and per-test pass/fail."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    # Enforce monotonicity
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    adj_out = np.empty_like(adj)
    adj_out[order] = adj
    return dict(
        p_adjusted=adj_out.tolist(),
        rejected=(adj_out <= alpha).tolist(),
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Paired bootstrap on delta-AUC (clean vs perturbed disease classification)
# ---------------------------------------------------------------------------

def paired_bootstrap_delta_auc(y_true, y_proba_clean, y_proba_perturbed,
                               n_boot=1000, seed=42) -> Dict:
    from sklearn.metrics import roc_auc_score
    y_true = np.asarray(y_true).astype(int)
    pc = np.asarray(y_proba_clean, dtype=float)
    pp = np.asarray(y_proba_perturbed, dtype=float)
    assert pc.shape == pp.shape == y_true.shape

    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas, clean_aucs, pert_aucs = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        a_c = roc_auc_score(yb, pc[idx])
        a_p = roc_auc_score(yb, pp[idx])
        clean_aucs.append(a_c)
        pert_aucs.append(a_p)
        deltas.append(a_c - a_p)
    deltas = np.asarray(deltas)

    return dict(
        auc_clean=float(roc_auc_score(y_true, pc)),
        auc_perturbed=float(roc_auc_score(y_true, pp)),
        delta=float(roc_auc_score(y_true, pc) - roc_auc_score(y_true, pp)),
        delta_ci=(float(np.percentile(deltas, 2.5)),
                  float(np.percentile(deltas, 97.5))),
        delta_mean=float(deltas.mean()),
        delta_sd=float(deltas.std()),
    )
