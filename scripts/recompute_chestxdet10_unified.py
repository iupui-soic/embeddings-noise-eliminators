#!/usr/bin/env python3
"""Produce one canonical AUC + 95% bootstrap-CI table for ChestX-Det10 bbox-level
classification, derived from the same forward pass that backs the paired-bootstrap
inference in ``paired_predictions.npz``.

For each (model, class) cell:
  * CLS pool predictions   --> reuse ``y_cls`` from paired_predictions.npz
  * patch-local predictions --> reuse ``y_pl``  from paired_predictions.npz
  * patch-mean predictions  --> re-train an L2-LR probe on the cached
                                image-level patch-mean features, identical
                                seed/C-grid/CV to the headline pipeline.

For each cell we compute the point AUC plus a 1,000-sample unpaired bootstrap
95% CI on the test set.  Output: ``outputs/v4_exp_chestxdet10/unified_bbox_results.parquet``
with 45 rows (5 models x 3 classes x 3 pools).  All downstream tables (main
manuscript Table~12, supplementary Tables~S10 and S11) should derive from this
parquet so the prose, table cells, and inferential q-values stay in lock-step.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path("/home/saptpurk/embeddings-noise-eliminators")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from common import PARAMS, train_probe  # noqa: E402
from run_chestxdet10_smalllesion import (  # noqa: E402
    DEFAULT_CLASSES, DEFAULT_MODELS, build_bbox_dataset, load_metadata,
)

OUT = REPO / "outputs" / "v4_exp_chestxdet10"
PRED = OUT / "paired_predictions.npz"
CACHE = OUT / "cache"


def boot_auc_ci(y_true: np.ndarray, y_proba: np.ndarray,
                n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, y_proba[idx]))
    pt = float(roc_auc_score(y_true, y_proba))
    aucs = np.asarray(aucs, dtype=float)
    return pt, float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main():
    meta_full = load_metadata()
    file_names = sorted(meta_full.keys())
    meta = {fn: meta_full[fn] for fn in file_names}
    idx_by_fn = {fn: i for i, fn in enumerate(file_names)}

    preds = np.load(PRED, allow_pickle=False)
    rows = []
    for model_name in DEFAULT_MODELS:
        # patch-mean pool features come from the cached image-level dump.
        cache_path = CACHE / f"{model_name}_image_pools.npz"
        if not cache_path.exists():
            print(f"!! missing cache for {model_name}; skipping", file=sys.stderr)
            continue
        d = np.load(cache_path, allow_pickle=True)
        cached_fns = list(d["file_names"])
        assert cached_fns == file_names, f"cache file_names mismatch for {model_name}"
        pm = d["patch_mean"]  # (3543, dim)

        for cls_name in DEFAULT_CLASSES:
            key = f"{model_name}__{cls_name}"
            y_test = preds[f"{key}__y_test"].astype(int)
            y_cls = preds[f"{key}__y_cls"].astype(float)
            y_pl = preds[f"{key}__y_pl"].astype(float)
            n_pos_test = int(y_test.sum())

            records = build_bbox_dataset(
                meta, cls_name, n_neg_per_pos=2, seed=PARAMS.random_seed)
            roles = np.array([r[2] for r in records], dtype=np.int8)
            splits = np.array([r[3] for r in records], dtype=object)
            tr = (splits == "train")
            te = (splits == "test")
            pm_feats = np.stack([pm[idx_by_fn[r[0]]] for r in records])
            assert int(roles[te].sum()) == n_pos_test, (
                f"role count mismatch on {key}: {int(roles[te].sum())} vs {n_pos_test}")

            pr_pm, _ = train_probe(
                pm_feats[tr], roles[tr], pm_feats[te], roles[te],
                name=f"bbox_{model_name}_{cls_name}_patch_mean", verbose=False)
            y_pm = pr_pm.y_proba.astype(float)

            for pool, yp in (("cls", y_cls), ("patch_mean", y_pm),
                             ("patch_local", y_pl)):
                auc, lo, hi = boot_auc_ci(y_test, yp)
                rows.append({
                    "model": model_name, "class": cls_name, "pool": pool,
                    "auc": auc, "auc_lo": lo, "auc_hi": hi,
                    "n_pos_train": int(roles[tr].sum()),
                    "n_pos_test": n_pos_test,
                })
                print(f"{model_name:11s} {cls_name:13s} {pool:11s} "
                      f"AUC={auc:.3f} [{lo:.3f}, {hi:.3f}]")

    df = pd.DataFrame(rows)
    out_parquet = OUT / "unified_bbox_results.parquet"
    df.to_parquet(out_parquet, index=False)
    print(f"\nSaved: {out_parquet}  ({len(df)} rows)")
    print("\nDeltas (patch_local - cls) per cell:")
    wide = df.pivot_table(index=["model", "class"], columns="pool", values="auc")
    wide["delta"] = wide["patch_local"] - wide["cls"]
    print(wide.to_string())
    print(f"\nΔ range: [{wide['delta'].min():+.3f}, {wide['delta'].max():+.3f}]")
    print(f"min patch_local AUC: {wide['patch_local'].min():.3f}")
    print(f"min Mass patch_local AUC: "
          f"{wide.loc[wide.index.get_level_values('class') == 'Mass', 'patch_local'].min():.3f}")


if __name__ == "__main__":
    main()
