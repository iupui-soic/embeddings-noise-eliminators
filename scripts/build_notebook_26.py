#!/usr/bin/env python3
"""
Author notebook 26 ChestXDet10 SmallLesion PatchPool from a single-source
Python script. Mirrors the cell-tag conventions of notebook 21
(`parameters` tag on the params cell, `papermill-injection` on the apply
cell) so `bash scripts/run_papermill_all.sh` orchestrates it cleanly.

Run this generator once after editing; the resulting .ipynb is the
canonical, version-controlled artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path("/home/saptpurk/embeddings-noise-eliminators/notebooks/26_ChestXDet10_SmallLesion_PatchPool.ipynb")


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str, tags=None) -> dict:
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": tags or []},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }
    return cell


CELLS = [
    md("""# Notebook 26 ChestXDet10 SmallLesion PatchPool

Real-CXR small-lesion bbox-stratified analysis on **ChestX-Det10**
(NIH-CXR14 subset with bbox annotations from Deepwise AI Lab; 3,543 images,
1,462 small-lesion bboxes across Calcification, Nodule, Mass).

Two complementary probes per (foundation model, lesion class):

1. **Image-level binary detection** (CLS / patch-mean), test-set AUC
   stratified by within-image median bbox area.
2. **Bbox-level region-aware patch-pool comparison** — positives are real
   bboxes; negatives are matched-area random regions sampled from
   class-negative images. Three pool modes from the same forward pass:
   CLS, patch-mean, patch-local (mean of patch tokens whose receptive
   field intersects the bbox).

The mechanism predicted in §6 (synthetic-perturbation battery) implies
that patch-local pooling should recover small-lesion classification AUC
that the global CLS pool dilutes. This notebook tests that prediction on
real CXR clinical-task data without injecting any perturbations.
"""),
    code("""# === Papermill parameters (override via `papermill -p NAME VALUE`) ===
DATASET = "chestxdet10"      # not used for label loading; cohort lives at /data0/chestx-det10
SEED = 42
OUTPUTS_DIR = "/home/saptpurk/embeddings-noise-eliminators/outputs"
REPO_ROOT_OVERRIDE = "/home/saptpurk/embeddings-noise-eliminators"
HF_TOKEN_OVERRIDE = None     # set non-None when running gated models locally
MODELS = "raddino,dinov2,dinov3,biomedclip,medsiglip"
CLASSES = "Nodule,Calcification,Mass"
N_NEG_PER_POS = 2            # bbox-level: random-region negatives per real bbox
SMOKE = False                # set True to restrict to ~100 images for plumbing check
""", tags=["parameters"]),
    code("""# Apply papermill parameters to environment
import os
os.environ.setdefault("DATASET", DATASET)
os.environ.setdefault("OUTPUTS_DIR", OUTPUTS_DIR)
os.environ.setdefault("REPO_ROOT", REPO_ROOT_OVERRIDE)
if HF_TOKEN_OVERRIDE:
    os.environ["HF_TOKEN"] = HF_TOKEN_OVERRIDE
""", tags=["papermill-injection"]),
    code("""import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(os.environ.get("REPO_ROOT", REPO_ROOT_OVERRIDE))
sys.path.insert(0, str(REPO_ROOT))

from common import EmbeddingExtractor, train_probe, PARAMS  # noqa: E402
from common.bbox_pool import random_negative_bbox  # noqa: E402

CHESTXDET10_ROOT = Path("/data0/chestx-det10")
IMAGE_HW = (1024, 1024)
OUT = Path(OUTPUTS_DIR) / "v4_exp_chestxdet10"
CACHE = OUT / "cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(exist_ok=True)

models_list = [m.strip() for m in MODELS.split(",") if m.strip()]
classes_list = [c.strip() for c in CLASSES.split(",") if c.strip()]
print(f"models  : {models_list}")
print(f"classes : {classes_list}")
print(f"OUT     : {OUT}")
"""),
    code("""def load_metadata() -> dict:
    \"\"\"file_name -> {syms, boxes, split} for all 3,543 ChestX-Det10 images.\"\"\"
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
    img = Image.open(CHESTXDET10_ROOT / folder / file_name).convert("RGB")
    return np.array(img)


def has_class(entry: dict, target_class: str) -> bool:
    return target_class in entry["syms"]


def class_bboxes(entry: dict, target_class: str):
    return [b for s, b in zip(entry["syms"], entry["boxes"]) if s == target_class]


meta_full = load_metadata()
file_names = sorted(meta_full.keys())
if SMOKE:
    with_bbox = [fn for fn in file_names if meta_full[fn]["syms"]][:80]
    no_bbox = [fn for fn in file_names if not meta_full[fn]["syms"]][:20]
    file_names = sorted(with_bbox + no_bbox)
# Restrict meta to the cohort actually being processed so bbox-dataset
# construction never references images outside file_names.
meta = {fn: meta_full[fn] for fn in file_names}
print(f"Loaded ChestX-Det10 metadata: {len(meta_full)} images "
      f"(processing {len(file_names)})")

# Cohort statistics
for c in classes_list:
    n_img = sum(1 for fn in file_names if has_class(meta[fn], c))
    n_bbox = sum(len(class_bboxes(meta[fn], c)) for fn in file_names)
    print(f"  {c:14s}: positive_images={n_img:>4}  bboxes={n_bbox:>4}")
"""),
    code("""def extract_image_pools(extractor, file_names, meta, cache_path: Path,
                        batch_size: int = 8):
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        cached_fns = list(d["file_names"])
        if cached_fns == file_names:
            print(f"    cache hit: {cache_path.name}  shape={d['cls'].shape}")
            return d["cls"], d["patch_mean"]
    cls_list, pm_list = [], []
    n = len(file_names); t0 = time.time()
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
                  f"({rate:.1f} img/s, {elapsed:.0f}s)")
    cls = np.concatenate(cls_list, axis=0)
    pm = np.concatenate(pm_list, axis=0)
    np.savez_compressed(
        cache_path,
        file_names=np.array(file_names, dtype=object),
        cls=cls, patch_mean=pm)
    return cls, pm


def build_bbox_dataset(meta: dict, target_class: str,
                       n_neg_per_pos: int, seed: int):
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


def extract_bbox_local_pools(extractor, records, meta, batch_size: int = 4):
    by_fn = defaultdict(list)
    for ridx, (fn, box, role, split) in enumerate(records):
        by_fn[fn].append((ridx, box, split))
    pools = None
    sorted_fns = sorted(by_fn.keys())
    n = len(sorted_fns); t0 = time.time()
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
                  f"({rate:.1f} img/s, {elapsed:.0f}s)")
    return pools


def stratified_aucs(y_test, y_proba, test_areas):
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
"""),
    code("""rows_image = []
rows_bbox = []

for model_name in models_list:
    print(f"\\n=== Model: {model_name} ===")
    ext = EmbeddingExtractor(model_name, hf_token=os.environ.get("HF_TOKEN"))

    cls, pm = extract_image_pools(
        ext, file_names, meta,
        cache_path=CACHE / f"{model_name}_image_pools.npz",
        batch_size=4 if model_name in ("dinov3", "medsiglip") else 8,
    )
    idx_by_fn = {fn: i for i, fn in enumerate(file_names)}

    for cls_name in classes_list:
        print(f"\\n  ----- Class: {cls_name} -----")

        # ----- Analysis A: image-level binary -----
        train_idx = np.array([i for i, fn in enumerate(file_names)
                              if meta[fn]["split"] == "train"])
        test_idx = np.array([i for i, fn in enumerate(file_names)
                             if meta[fn]["split"] == "test"])
        train_y = np.array([int(has_class(meta[file_names[i]], cls_name))
                            for i in train_idx])
        test_y = np.array([int(has_class(meta[file_names[i]], cls_name))
                           for i in test_idx])
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
                "auc_small": strata["auc_small"],
                "auc_large": strata["auc_large"],
                "n_small": strata["n_small"], "n_large": strata["n_large"],
                "median_area": strata["median_area"],
                "best_C": pr.best_C,
                "n_train": pr.n_train, "n_test": pr.n_test,
                "n_pos_train": pr.n_pos_train, "n_pos_test": pr.n_pos_test,
            })

        # ----- Analysis B: bbox-level region-aware -----
        records = build_bbox_dataset(
            meta, cls_name, n_neg_per_pos=N_NEG_PER_POS, seed=PARAMS.random_seed)
        if SMOKE:
            train_recs = [r for r in records if r[3] == "train"][:24]
            test_recs = [r for r in records if r[3] == "test"][:8]
            records = train_recs + test_recs
        if not records:
            continue

        local_feats = extract_bbox_local_pools(
            ext, records, meta,
            batch_size=2 if model_name in ("dinov3", "medsiglip") else 4)
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
            strata = stratified_aucs(roles[te_mask], pr.y_proba, rec_areas[te_mask])
            print(f"    [bbox] {pool:11s} AUC={pr.auc:.3f} "
                  f"[{pr.auc_ci[0]:.3f}, {pr.auc_ci[1]:.3f}]   "
                  f"small={strata['auc_small']}  large={strata['auc_large']}  "
                  f"n_pos={pr.n_pos_train}/{pr.n_pos_test}")
            rows_bbox.append({
                "model": model_name, "class": cls_name, "pool": pool,
                "auc": pr.auc, "auc_lo": pr.auc_ci[0], "auc_hi": pr.auc_ci[1],
                "auc_small": strata["auc_small"],
                "auc_large": strata["auc_large"],
                "n_small": strata["n_small"], "n_large": strata["n_large"],
                "median_area": strata["median_area"],
                "best_C": pr.best_C,
                "n_train": pr.n_train, "n_test": pr.n_test,
                "n_pos_train": pr.n_pos_train, "n_pos_test": pr.n_pos_test,
            })

    ext.close()
    pd.DataFrame(rows_image).to_parquet(
        OUT / "image_level_results.parquet", index=False)
    pd.DataFrame(rows_bbox).to_parquet(
        OUT / "bbox_level_results.parquet", index=False)
    print(f"  saved partial results "
          f"(img={len(rows_image)} rows, bbox={len(rows_bbox)} rows)")

print("\\n=== DONE ===")
print(f"Image-level results: {len(rows_image)} rows")
print(f"Bbox-level  results: {len(rows_bbox)} rows")
print(f"Outputs in: {OUT}")
"""),
    code("""# Quick summary print: per-FM mean patch_local-vs-CLS gap across the 3 classes
df_bb = pd.DataFrame(rows_bbox)
if not df_bb.empty:
    piv = df_bb.pivot_table(
        index=["model", "class"], columns="pool", values="auc")
    if "patch_local" in piv.columns and "cls" in piv.columns:
        gap = (piv["patch_local"] - piv["cls"]).reset_index()
        gap.columns = ["model", "class", "gap_local_minus_cls"]
        print("Per-(FM, class) bbox-level patch_local - CLS gap:")
        print(gap.round(3).to_string(index=False))
        print()
        print("Per-FM mean gap across classes:")
        print(gap.groupby("model")["gap_local_minus_cls"].mean().round(3).to_string())
"""),
]


def main():
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NB_PATH.open("w") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {NB_PATH}")


if __name__ == "__main__":
    main()
