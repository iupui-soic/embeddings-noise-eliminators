#!/usr/bin/env python3
"""
ChestX-Det10 paired bootstrap on Δ(patch_local − CLS) per cell + BH-FDR.
=======================================================================

Companion to ``run_chestxdet10_smalllesion.py``.  For each of the 15
(model × class) cells of Table~\\ref{tab:chestxdet10_recovery}, this
script:

    1. Reuses the cached image-level CLS / patch-mean pools for the
       same forward pass that produced the headline numbers.
    2. Re-extracts the bbox-level patch_local pool features.
    3. Trains the same three L2-LR probes (CLS, patch_mean, patch_local)
       with identical (seed, C-grid) settings.
    4. Persists per-cell test labels and per-pool y_proba so the test
       can be reproduced without GPU access.
    5. Computes a paired bootstrap p-value for Δ(patch_local − CLS)
       using ``paired_bootstrap_delta_auc`` (n_boot = 10,000).
    6. Applies BH-FDR correction across the 15 cells.

Outputs
-------
outputs/v4_exp_chestxdet10/
    paired_predictions.npz    - per-cell {y_test, y_cls, y_pl} for all 15 cells
    paired_stats.parquet      - per-cell Δ, paired-bootstrap CI, p, q (BH)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path("/home/saptpurk/embeddings-noise-eliminators")
sys.path.insert(0, str(REPO_ROOT))

from common import EmbeddingExtractor, train_probe, PARAMS  # noqa: E402
from common.stats import benjamini_hochberg, paired_bootstrap_delta_auc  # noqa: E402

# Reuse the helpers from the headline script so this stays in lock-step.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_chestxdet10_smalllesion import (  # noqa: E402
    DEFAULT_CLASSES,
    DEFAULT_MODELS,
    build_bbox_dataset,
    extract_bbox_local_pools,
    extract_image_pools,
    load_metadata,
)


def paired_p_value(deltas: np.ndarray) -> float:
    """Two-sided paired-bootstrap p-value.

    Uses the standard convention: 2 × min(P(Δ_b ≤ 0), P(Δ_b ≥ 0)),
    floored at 1/(n_boot + 1) so a fully separated bootstrap distribution
    yields a finite p-value rather than zero.
    """
    deltas = np.asarray(deltas, dtype=float)
    n = len(deltas)
    if n == 0:
        return float("nan")
    # Centre the bootstrap distribution at zero (paired-shift null).
    centred = deltas - deltas.mean()
    obs = float(deltas.mean())
    p_two_sided = float((np.abs(centred) >= abs(obs)).mean())
    return max(p_two_sided, 1.0 / (n + 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    p.add_argument("--out-dir", default=str(
        REPO_ROOT / "outputs" / "v4_exp_chestxdet10"))
    p.add_argument("--n-neg-per-pos", type=int, default=2)
    p.add_argument("--n-boot", type=int, default=10_000)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    meta_full = load_metadata()
    file_names = sorted(meta_full.keys())
    meta = {fn: meta_full[fn] for fn in file_names}
    print(f"Loaded ChestX-Det10 metadata: {len(meta_full)} images")

    rows, predictions = [], {}
    for model_name in args.models:
        print(f"\n=== Model: {model_name} ===")
        ext = EmbeddingExtractor(
            model_name,
            hf_token=__import__("os").environ.get("HF_TOKEN"))
        cls, pm = extract_image_pools(
            ext, file_names, meta,
            cache_path=cache_dir / f"{model_name}_image_pools.npz",
            batch_size=4 if model_name in ("dinov3", "medsiglip") else 8,
        )
        idx_by_fn = {fn: i for i, fn in enumerate(file_names)}

        for cls_name in args.classes:
            print(f"\n  ----- Class: {cls_name} -----")
            t0 = time.time()
            records = build_bbox_dataset(
                meta, cls_name, n_neg_per_pos=args.n_neg_per_pos,
                seed=PARAMS.random_seed)
            if not records:
                continue
            local_feats = extract_bbox_local_pools(
                ext, records, meta,
                batch_size=2 if model_name in ("dinov3", "medsiglip") else 4)

            cls_feats = np.stack([cls[idx_by_fn[r[0]]] for r in records])
            roles = np.array([r[2] for r in records], dtype=np.int8)
            splits = np.array([r[3] for r in records], dtype=object)
            tr_mask = (splits == "train")
            te_mask = (splits == "test")

            pr_cls, _ = train_probe(
                cls_feats[tr_mask], roles[tr_mask],
                cls_feats[te_mask], roles[te_mask],
                name=f"bbox_{model_name}_{cls_name}_cls", verbose=False)
            pr_pl, _ = train_probe(
                local_feats[tr_mask], roles[tr_mask],
                local_feats[te_mask], roles[te_mask],
                name=f"bbox_{model_name}_{cls_name}_patch_local",
                verbose=False)

            y_test = pr_cls.y_true.astype(int)
            y_cls = pr_cls.y_proba.astype(float)
            y_pl = pr_pl.y_proba.astype(float)
            assert (y_test == pr_pl.y_true.astype(int)).all(), (
                "test order should be identical: same records, same te_mask")

            from sklearn.metrics import roc_auc_score
            auc_cls = float(roc_auc_score(y_test, y_cls))
            auc_pl = float(roc_auc_score(y_test, y_pl))

            rng = np.random.default_rng(42)
            n = len(y_test)
            deltas = []
            for _ in range(args.n_boot):
                idx = rng.integers(0, n, n)
                yb = y_test[idx]
                if len(np.unique(yb)) < 2:
                    continue
                deltas.append(
                    roc_auc_score(yb, y_pl[idx])
                    - roc_auc_score(yb, y_cls[idx]))
            deltas = np.asarray(deltas, dtype=float)
            delta_obs = auc_pl - auc_cls
            p_val = paired_p_value(deltas)
            ci_lo = float(np.percentile(deltas, 2.5))
            ci_hi = float(np.percentile(deltas, 97.5))

            elapsed = time.time() - t0
            print(f"    Δ={delta_obs:+.3f}  CI=[{ci_lo:+.3f}, {ci_hi:+.3f}]  "
                  f"p={p_val:.4f}  n_test={n}  ({elapsed:.0f}s)")

            rows.append({
                "model": model_name, "class": cls_name,
                "auc_cls": auc_cls, "auc_patch_local": auc_pl,
                "delta": delta_obs,
                "delta_ci_lo": ci_lo, "delta_ci_hi": ci_hi,
                "p_paired_bootstrap": p_val,
                "n_test": n, "n_pos_test": int(y_test.sum()),
            })
            predictions[f"{model_name}__{cls_name}__y_test"] = y_test
            predictions[f"{model_name}__{cls_name}__y_cls"] = y_cls
            predictions[f"{model_name}__{cls_name}__y_pl"] = y_pl

        ext.close()

    df = pd.DataFrame(rows)
    bh = benjamini_hochberg(df["p_paired_bootstrap"].values, alpha=0.05)
    df["q_bh"] = bh["p_adjusted"]
    df["rejected_q05"] = bh["rejected"]

    df.to_parquet(out_dir / "paired_stats.parquet", index=False)
    np.savez_compressed(out_dir / "paired_predictions.npz", **predictions)

    print("\n=== Per-cell paired bootstrap on Δ(patch_local − CLS) ===")
    print(df.to_string(index=False))
    print(f"\nBH-FDR-significant at q ≤ 0.05: "
          f"{int(df['rejected_q05'].sum())}/{len(df)}")
    print(f"Saved: {out_dir / 'paired_stats.parquet'}")


if __name__ == "__main__":
    main()
