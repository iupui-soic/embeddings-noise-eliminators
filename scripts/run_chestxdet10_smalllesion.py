#!/usr/bin/env python3
"""
ChestX-Det10 small-lesion bbox-stratified patch-local probing.
==============================================================

Two complementary analyses on the same 3,543-image ChestX-Det10 cohort
(NIH-CXR14 subset with bbox annotations from Deepwise AILab):

    (A) Image-level binary detection
        Per (FM, class), train an L2-LR probe on "image has class X
        yes/no" using global pool (cls / patch_mean), then stratify
        test-set AUC by within-image bbox area (small / large median split).

    (B) Bbox-level region-aware patch-pool comparison
        Per (FM, class, pool_mode), train an L2-LR probe on
        (image, ROI) pairs:
            positives = real class-X bboxes (one row per bbox)
            negatives = matched-area random regions sampled from
                        class-X-negative images
        Pool modes:
            cls         : image-level CLS pool (ignores ROI; baseline)
            patch_mean  : image-level mean of patch tokens (ignores ROI)
            patch_local : ROI-restricted mean of patch tokens that
                          intersect the bbox (region-aware)
        Stratify test-set AUC by bbox area.

The mechanism in the main paper predicts that for small-lesion classes
(Calcification, Nodule), patch_local AUC will exceed cls AUC; the gap
should shrink for the larger Mass class.

Outputs
-------
outputs/v4_exp_chestxdet10/
    cache/<model>_image_pools.npz   - per-image cls + patch_mean caches
    image_level_results.parquet     - analysis A long-form results
    bbox_level_results.parquet      - analysis B long-form results
    summary.txt                     - human-readable run summary
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path("/home/saptpurk/embeddings-noise-eliminators")
sys.path.insert(0, str(REPO_ROOT))

from common import EmbeddingExtractor, train_probe, PARAMS  # noqa: E402
from common.bbox_pool import random_negative_bbox  # noqa: E402

CHESTXDET10_ROOT = Path("/data0/chestx-det10")
IMAGE_HW = (1024, 1024)
DEFAULT_CLASSES = ["Nodule", "Calcification", "Mass"]
DEFAULT_MODELS = ["raddino", "dinov2", "dinov3", "biomedclip", "medsiglip"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_metadata() -> dict:
    out = {}
    for split in ("train", "test"):
        with open(CHESTXDET10_ROOT / f"{split}.json") as f:
            d = json.load(f)
        for ent in d:
            out[ent["file_name"]] = {
                "syms": ent.get("syms") or [],
                "boxes": ent.get("boxes") or [],
                "split": split,
            }
    return out


def open_image_array(file_name: str, split: str) -> np.ndarray:
    folder = "images_train" if split == "train" else "images_test"
    p = CHESTXDET10_ROOT / folder / file_name
    img = Image.open(p).convert("RGB")
    return np.array(img)


def has_class(entry: dict, target_class: str) -> bool:
    return target_class in entry["syms"]


def class_bboxes(entry: dict, target_class: str):
    return [b for s, b in zip(entry["syms"], entry["boxes"]) if s == target_class]


# ---------------------------------------------------------------------------
# Image-level pool extraction (cls + patch_mean), cached on disk
# ---------------------------------------------------------------------------

def extract_image_pools(extractor: EmbeddingExtractor, file_names, meta,
                        cache_path: Path, batch_size: int = 8) -> tuple:
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        cached_fns = list(d["file_names"])
        if cached_fns == file_names:
            print(f"    cache hit: {cache_path.name}  shape={d['cls'].shape}")
            return d["cls"], d["patch_mean"]

    cls_list, pm_list = [], []
    n = len(file_names)
    t0 = time.time()
    for i in range(0, n, batch_size):
        batch = file_names[i:i + batch_size]
        images = [open_image_array(fn, meta[fn]["split"]) for fn in batch]
        result = extractor.extract_all(
            images, [[] for _ in images], image_hw=IMAGE_HW)
        cls_list.append(result["cls"])
        pm_list.append(result["patch_mean"])
        if (i // batch_size) % 25 == 0 or i + batch_size >= n:
            elapsed = time.time() - t0
            rate = (i + len(batch)) / max(1e-3, elapsed)
            print(f"    image pools {i+len(batch):>4}/{n}  "
                  f"({rate:.1f} img/s, {elapsed:.0f}s elapsed)")

    cls = np.concatenate(cls_list, axis=0)
    pm = np.concatenate(pm_list, axis=0)
    np.savez_compressed(cache_path, file_names=np.array(file_names, dtype=object),
                        cls=cls, patch_mean=pm)
    return cls, pm


# ---------------------------------------------------------------------------
# Bbox-level extraction (per region patch-local pool from one forward pass)
# ---------------------------------------------------------------------------

def build_bbox_dataset(meta: dict, target_class: str, n_neg_per_pos: int,
                       seed: int):
    """Return list of records [(file_name, bbox, role, split)] where role
    is 1 for real-bbox positives and 0 for matched-random negatives."""
    rng = np.random.default_rng(seed)

    pos = []
    for fn, m in meta.items():
        for box in class_bboxes(m, target_class):
            pos.append((fn, tuple(box), 1, m["split"]))

    neg_pool = {"train": [], "test": []}
    for fn, m in meta.items():
        if not has_class(m, target_class):
            neg_pool[m["split"]].append(fn)

    neg = []
    for fn, box, _, split in pos:
        x1, y1, x2, y2 = box
        bw, bh = max(1, int(x2 - x1)), max(1, int(y2 - y1))
        if not neg_pool[split]:
            continue
        for _ in range(n_neg_per_pos):
            nfn = neg_pool[split][rng.integers(0, len(neg_pool[split]))]
            nbox = random_negative_bbox(rng, IMAGE_HW, bh, bw)
            neg.append((nfn, tuple(nbox), 0, split))

    return pos + neg


def extract_bbox_local_pools(extractor: EmbeddingExtractor,
                             records, meta, batch_size: int = 4) -> np.ndarray:
    """For each record, return its patch-local pool by forwarding the host
    image once and masking the patch tokens that intersect the record's bbox."""
    by_fn = defaultdict(list)
    for ridx, (fn, box, role, split) in enumerate(records):
        by_fn[fn].append((ridx, box, split))

    pools = None
    sorted_fns = sorted(by_fn.keys())
    n = len(sorted_fns)
    t0 = time.time()
    for i in range(0, n, batch_size):
        batch_fns = sorted_fns[i:i + batch_size]
        images = [open_image_array(fn, meta[fn]["split"]) for fn in batch_fns]
        _, patches, (gh, gw) = extractor._forward_tokens(images)
        if pools is None:
            pools = np.zeros((len(records), patches.shape[-1]), dtype=np.float32)
        cell_h = IMAGE_HW[0] / gh
        cell_w = IMAGE_HW[1] / gw
        for bi, fn in enumerate(batch_fns):
            P = patches[bi]
            for (ridx, box, _split) in by_fn[fn]:
                x1, y1, x2, y2 = box
                r0 = max(0, int(np.floor(y1 / cell_h)))
                r1 = min(gh, int(np.ceil(y2 / cell_h)))
                c0 = max(0, int(np.floor(x1 / cell_w)))
                c1 = min(gw, int(np.ceil(x2 / cell_w)))
                if r1 > r0 and c1 > c0:
                    mask = np.zeros((gh, gw), dtype=bool)
                    mask[r0:r1, c0:c1] = True
                    pools[ridx] = P[mask.reshape(-1)].mean(axis=0)
                else:
                    pools[ridx] = P.mean(axis=0)
        if (i // batch_size) % 25 == 0 or i + batch_size >= n:
            elapsed = time.time() - t0
            rate = (i + len(batch_fns)) / max(1e-3, elapsed)
            print(f"    bbox pools {i+len(batch_fns):>4}/{n}  "
                  f"({rate:.1f} img/s, {elapsed:.0f}s elapsed)")
    return pools


# ---------------------------------------------------------------------------
# Stratified AUC
# ---------------------------------------------------------------------------

def stratified_aucs(y_test: np.ndarray, y_proba: np.ndarray,
                    test_areas: np.ndarray) -> dict:
    """Compute small / large test-set AUCs by median bbox area among positives.

    Negatives are held constant across strata."""
    pos_mask = (y_test == 1)
    neg_mask = (y_test == 0)
    if pos_mask.sum() < 4:
        return {"auc_small": None, "auc_large": None,
                "n_small": int(pos_mask.sum()), "n_large": 0,
                "median_area": None}
    pos_areas = test_areas[pos_mask]
    median_a = float(np.median(pos_areas))
    small_pos = pos_mask & (test_areas < median_a)
    large_pos = pos_mask & (test_areas >= median_a)

    def auc_for(pos_idx_mask):
        sub = pos_idx_mask | neg_mask
        if pos_idx_mask.sum() == 0 or neg_mask.sum() == 0:
            return None
        try:
            return float(roc_auc_score(y_test[sub], y_proba[sub]))
        except ValueError:
            return None

    return {
        "auc_small": auc_for(small_pos),
        "auc_large": auc_for(large_pos),
        "n_small": int(small_pos.sum()),
        "n_large": int(large_pos.sum()),
        "median_area": median_a,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_for_model(model_name: str, classes, meta, file_names, out_dir: Path,
                  cache_dir: Path, n_neg_per_pos: int, smoke: bool):
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

    rows_image, rows_bbox = [], []

    for cls_name in classes:
        print(f"\n  ----- Class: {cls_name} -----")

        # === Analysis A: image-level binary ===
        train_idx = np.array([i for i, fn in enumerate(file_names)
                              if meta[fn]["split"] == "train"])
        test_idx = np.array([i for i, fn in enumerate(file_names)
                             if meta[fn]["split"] == "test"])
        train_y = np.array([int(has_class(meta[file_names[i]], cls_name))
                            for i in train_idx])
        test_y = np.array([int(has_class(meta[file_names[i]], cls_name))
                           for i in test_idx])
        # Median bbox area per positive image (used for stratified test AUC).
        test_areas = np.zeros(len(test_idx), dtype=np.float64)
        for k, i in enumerate(test_idx):
            boxes = class_bboxes(meta[file_names[i]], cls_name)
            if boxes:
                areas = [(b[2]-b[0]) * (b[3]-b[1]) for b in boxes]
                test_areas[k] = float(np.median(areas))

        for pool, X in (("cls", cls), ("patch_mean", pm)):
            if train_y.sum() == 0 or test_y.sum() == 0:
                continue
            pr, _ = train_probe(
                X[train_idx], train_y, X[test_idx], test_y,
                name=f"img_{model_name}_{cls_name}_{pool}", verbose=False)
            strata = stratified_aucs(test_y, pr.y_proba, test_areas)
            print(f"    [img]  {pool:11s} AUC={pr.auc:.3f} "
                  f"[{pr.auc_ci[0]:.3f}, {pr.auc_ci[1]:.3f}]   "
                  f"small={strata['auc_small']}  large={strata['auc_large']}  "
                  f"n_pos={pr.n_pos_train}/{pr.n_pos_test}")
            rows_image.append({
                "model": model_name, "class": cls_name, "pool": pool,
                "auc": pr.auc, "auc_lo": pr.auc_ci[0], "auc_hi": pr.auc_ci[1],
                "auc_small": strata["auc_small"], "auc_large": strata["auc_large"],
                "n_small": strata["n_small"], "n_large": strata["n_large"],
                "median_area": strata["median_area"],
                "best_C": pr.best_C,
                "n_train": pr.n_train, "n_test": pr.n_test,
                "n_pos_train": pr.n_pos_train, "n_pos_test": pr.n_pos_test,
            })

        # === Analysis B: bbox-level region-aware ===
        records = build_bbox_dataset(
            meta, cls_name, n_neg_per_pos=n_neg_per_pos,
            seed=PARAMS.random_seed)
        if smoke:
            # Keep both train and test records so the probe has a test set.
            train_recs = [r for r in records if r[3] == "train"][:24]
            test_recs = [r for r in records if r[3] == "test"][:8]
            records = train_recs + test_recs
        if not records:
            continue

        local_feats = extract_bbox_local_pools(
            ext, records, meta,
            batch_size=2 if model_name in ("dinov3", "medsiglip") else 4)

        # Map each record to its image-level cls / patch_mean
        cls_feats = np.stack([cls[idx_by_fn[r[0]]] for r in records])
        pm_feats = np.stack([pm[idx_by_fn[r[0]]] for r in records])
        roles = np.array([r[2] for r in records], dtype=np.int8)
        splits = np.array([r[3] for r in records], dtype=object)
        rec_areas = np.array([(r[1][2]-r[1][0]) * (r[1][3]-r[1][1])
                              for r in records], dtype=np.float64)
        tr_mask = (splits == "train")
        te_mask = (splits == "test")

        for pool, feats in (("cls", cls_feats),
                            ("patch_mean", pm_feats),
                            ("patch_local", local_feats)):
            if roles[tr_mask].sum() == 0 or roles[te_mask].sum() == 0:
                continue
            pr, _ = train_probe(
                feats[tr_mask], roles[tr_mask],
                feats[te_mask], roles[te_mask],
                name=f"bbox_{model_name}_{cls_name}_{pool}", verbose=False)
            test_areas_b = rec_areas[te_mask]
            strata = stratified_aucs(roles[te_mask], pr.y_proba, test_areas_b)
            print(f"    [bbox] {pool:11s} AUC={pr.auc:.3f} "
                  f"[{pr.auc_ci[0]:.3f}, {pr.auc_ci[1]:.3f}]   "
                  f"small={strata['auc_small']}  large={strata['auc_large']}  "
                  f"n_pos={pr.n_pos_train}/{pr.n_pos_test}")
            rows_bbox.append({
                "model": model_name, "class": cls_name, "pool": pool,
                "auc": pr.auc, "auc_lo": pr.auc_ci[0], "auc_hi": pr.auc_ci[1],
                "auc_small": strata["auc_small"], "auc_large": strata["auc_large"],
                "n_small": strata["n_small"], "n_large": strata["n_large"],
                "median_area": strata["median_area"],
                "best_C": pr.best_C,
                "n_train": pr.n_train, "n_test": pr.n_test,
                "n_pos_train": pr.n_pos_train, "n_pos_test": pr.n_pos_test,
            })

    ext.close()
    return rows_image, rows_bbox


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    p.add_argument("--out-dir", default=str(
        REPO_ROOT / "outputs" / "v4_exp_chestxdet10"))
    p.add_argument("--n-neg-per-pos", type=int, default=2)
    p.add_argument("--smoke", action="store_true",
                   help="Use a tiny dataset slice for end-to-end verification.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    meta_full = load_metadata()
    file_names = sorted(meta_full.keys())
    if args.smoke:
        # Restrict to images that have at least one bbox of any kind, plus
        # a handful of clean negatives, to give the smoke test something to fit.
        with_bbox = [fn for fn in file_names if meta_full[fn]["syms"]][:80]
        no_bbox = [fn for fn in file_names if not meta_full[fn]["syms"]][:20]
        file_names = sorted(with_bbox + no_bbox)
    # Restrict meta to file_names so bbox-dataset construction never references
    # images outside the cohort being processed.
    meta = {fn: meta_full[fn] for fn in file_names}
    print(f"Loaded ChestX-Det10 metadata: {len(meta_full)} images "
          f"(processing {len(file_names)})")

    all_image_rows, all_bbox_rows = [], []
    for model_name in args.models:
        try:
            r_img, r_bbox = run_for_model(
                model_name, args.classes, meta, file_names,
                out_dir=out_dir, cache_dir=cache_dir,
                n_neg_per_pos=args.n_neg_per_pos, smoke=args.smoke)
            all_image_rows.extend(r_img)
            all_bbox_rows.extend(r_bbox)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"!! Model {model_name} failed: {e}")
            continue
        # Persist after each model so partial runs are useful
        pd.DataFrame(all_image_rows).to_parquet(
            out_dir / "image_level_results.parquet", index=False)
        pd.DataFrame(all_bbox_rows).to_parquet(
            out_dir / "bbox_level_results.parquet", index=False)
        print(f"  saved partial results "
              f"(img={len(all_image_rows)} rows, bbox={len(all_bbox_rows)} rows)")

    print("\n=== DONE ===")
    print(f"Image-level results: {len(all_image_rows)} rows")
    print(f"Bbox-level  results: {len(all_bbox_rows)} rows")
    print(f"Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
