"""
common/native_resolution.py
===========================

Direct native-resolution perturbation control for the v4 mechanism paper.

The main pipeline injects perturbations into a $1{,}024 \\times 1{,}024$ canvas
and lets each model's HuggingFace image processor handle the resize down to
the model's pretraining input resolution.  This module bypasses the resize
upstream of the processor: each clean image is resized to the model's native
post-processor input resolution *first*, the perturbation is applied at that
native pixel scale, and the resulting array is fed to the model.  The model
therefore sees the perturbation at the absolute pixel scale we specified,
not a resize-attenuated version of it.

The function below produces one AUC row per (model, perturbation) cell --
CLS pool and patch-local pool from the same forward pass, plus 1{,}000-iteration
unpaired bootstrap 95% CIs from probe predictions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from common.embedding_utils import EmbeddingExtractor
from common.perturbations import make_injector, sample_patch_origin
from common.probing import train_probe

# Effective post-processor spatial input per model.  The HuggingFace
# AutoImageProcessor's resize+crop pipeline lands the input image at this
# resolution before patch tokenisation; pre-resizing the source image to
# this size means the processor's resize step is (close to) a no-op, so the
# pixel-scale of the injected perturbation is preserved end-to-end.
NATIVE_INPUT_HW = {
    "raddino":     (518, 518),
    "dinov2":      (224, 224),
    "dinov3":      (224, 224),
    "biomedclip":  (224, 224),
    "medsiglip":   (448, 448),
}

DEFAULT_DATASET_ROOTS = {
    "nih":   Path("/data0/NIH-CXR14/images"),
    "mimic": None,   # not wired up here
    "emory": None,
}


def _resize_to_native(image: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Resize a HxWx3 (or HxW) uint8 image to target_hw with LANCZOS4."""
    h, w = target_hw
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[:2] == (h, w):
        return image.astype(np.uint8)
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4).astype(np.uint8)


def _load_clean_images(dataset: str, n: int, seed: int = 42) -> Sequence[Tuple[str, np.ndarray]]:
    root = DEFAULT_DATASET_ROOTS.get(dataset)
    if root is None or not Path(root).exists():
        raise FileNotFoundError(
            f"Dataset root for '{dataset}' not configured / not found ({root})")
    files = sorted(Path(root).glob("*.png"))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(files), size=min(n, len(files)), replace=False)
    out = []
    for i in idx:
        f = files[int(i)]
        img = np.array(Image.open(f).convert("RGB"))
        out.append((str(f), img))
    return out


def _bootstrap_auc_ci(y_true: np.ndarray, y_proba: np.ndarray,
                      n_boot: int = 1000, seed: int = 42) -> Tuple[float, float]:
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, y_proba[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _forward_batches(extractor: EmbeddingExtractor,
                     images: Sequence[np.ndarray],
                     patch_locs: Sequence[Sequence[dict]],
                     image_hw: Tuple[int, int],
                     batch_size: int = 8):
    cls_list, pl_list = [], []
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i + batch_size]
        batch_locs = patch_locs[i:i + batch_size]
        out = extractor.extract_all(batch_imgs, batch_locs, image_hw=image_hw)
        cls_list.append(out["cls"])
        pl_list.append(out["patch_local"])
    return np.concatenate(cls_list, axis=0), np.concatenate(pl_list, axis=0)


def run_native_resolution_sweep(
    models: Iterable[str] = ("raddino", "dinov2", "dinov3", "biomedclip", "medsiglip"),
    perturbations: Sequence[Tuple[str, int]] = (
        ("iso_blur", 4), ("iso_blur", 16), ("reticular", 32),
    ),
    dataset: str = "nih",
    n_clean: int = 200,
    out_dir: Optional[Path] = None,
    seed: int = 42,
    hf_token: Optional[str] = None,
):
    """Run the matched-native-resolution control sweep.

    For each (model, perturbation): build a balanced (clean, perturbed)
    binary-classification dataset at the model's native input resolution,
    extract CLS and patch-local pools from one frozen forward pass, train
    L2-LR probes on each pool, return AUC + 95% bootstrap CI.

    Returns a list of dicts, one per (model, perturbation) cell.  Persists
    the same as a parquet to ``out_dir / native_resolution_results.parquet``.
    """
    out_dir = Path(out_dir) if out_dir else Path(
        "/home/saptpurk/embeddings-noise-eliminators/outputs/v4_exp15_native_resolution")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load + cache clean images once at the largest native size we need; we
    # then resize per-model.  (LANCZOS4 down-/up-sample is cheap relative to
    # the GPU forward passes that dominate runtime.)
    clean_records = _load_clean_images(dataset, n_clean, seed=seed)
    print(f"loaded {len(clean_records)} clean images from {dataset}")

    rows = []
    for model_name in models:
        if model_name not in NATIVE_INPUT_HW:
            print(f"!! skip unknown model {model_name}")
            continue
        native_hw = NATIVE_INPUT_HW[model_name]
        print(f"\n=== {model_name} @ native {native_hw[0]}x{native_hw[1]} ===")
        ext = EmbeddingExtractor(model_name, hf_token=hf_token)

        # Resize all clean images once for this model
        clean_native = [(p, _resize_to_native(im, native_hw)) for p, im in clean_records]

        for pert_name, patch_size in perturbations:
            print(f"  --- perturbation {pert_name} patch={patch_size}px ---")
            injector = make_injector(pert_name, seed=seed)

            # Generate (clean, perturbed) at native resolution.  Each clean image
            # contributes one perturbed twin; deterministic patch placement
            # via the injector's per-image seed.
            clean_imgs, perturbed_imgs, locs = [], [], []
            for path, clean_im in clean_native:
                noisy_im, meta = injector(
                    clean_im, patch_size=patch_size, image_path=path)
                clean_imgs.append(clean_im)
                perturbed_imgs.append(noisy_im)
                locs.append(meta["patch_locations"])

            cls_clean, pl_clean = _forward_batches(
                ext, clean_imgs, locs, native_hw,
                batch_size=2 if model_name in ("dinov3", "medsiglip") else 4)
            cls_pert, pl_pert = _forward_batches(
                ext, perturbed_imgs, locs, native_hw,
                batch_size=2 if model_name in ("dinov3", "medsiglip") else 4)

            X_cls = np.concatenate([cls_clean, cls_pert], axis=0)
            X_pl = np.concatenate([pl_clean, pl_pert], axis=0)
            y = np.concatenate([np.zeros(len(cls_clean)),
                                np.ones(len(cls_pert))]).astype(int)

            # Image-level split: each clean image and its paired perturbed
            # twin always land on the same side of the train/test split,
            # so the probe cannot identify perturbed status by recognising
            # the underlying image (the clean[i] and perturbed[i] embeddings
            # are near-identical when the perturbation is sub-patch).
            n_imgs = len(cls_clean)
            rng = np.random.default_rng(seed)
            img_perm = rng.permutation(n_imgs)
            n_tr_imgs = n_imgs // 2
            tr_imgs = img_perm[:n_tr_imgs]
            te_imgs = img_perm[n_tr_imgs:]
            # clean indices in concatenated array are [0..n_imgs);
            # perturbed indices are [n_imgs..2*n_imgs).
            tr = np.concatenate([tr_imgs, n_imgs + tr_imgs])
            te = np.concatenate([te_imgs, n_imgs + te_imgs])

            pr_cls, _ = train_probe(
                X_cls[tr], y[tr], X_cls[te], y[te],
                name=f"native_{model_name}_{pert_name}{patch_size}_cls",
                verbose=False)
            pr_pl, _ = train_probe(
                X_pl[tr], y[tr], X_pl[te], y[te],
                name=f"native_{model_name}_{pert_name}{patch_size}_local",
                verbose=False)

            cls_lo, cls_hi = _bootstrap_auc_ci(pr_cls.y_true, pr_cls.y_proba)
            pl_lo, pl_hi = _bootstrap_auc_ci(pr_pl.y_true, pr_pl.y_proba)

            row = {
                "model": model_name,
                "native_h": native_hw[0],
                "native_w": native_hw[1],
                "perturbation": pert_name,
                "patch_size_px": patch_size,
                "n_clean": len(cls_clean),
                "n_test": len(te),
                "auc_cls": pr_cls.auc,
                "auc_cls_lo": cls_lo, "auc_cls_hi": cls_hi,
                "auc_patch_local": pr_pl.auc,
                "auc_patch_local_lo": pl_lo, "auc_patch_local_hi": pl_hi,
                "delta": pr_pl.auc - pr_cls.auc,
            }
            rows.append(row)
            print(
                f"    CLS  AUC={pr_cls.auc:.3f} [{cls_lo:.3f}, {cls_hi:.3f}]   "
                f"patch_local AUC={pr_pl.auc:.3f} [{pl_lo:.3f}, {pl_hi:.3f}]   "
                f"Δ={pr_pl.auc - pr_cls.auc:+.3f}"
            )

        ext.close()

    df = pd.DataFrame(rows)
    out_pq = out_dir / "native_resolution_results.parquet"
    df.to_parquet(out_pq, index=False)
    print(f"\nSaved: {out_pq}  ({len(df)} rows)")
    return rows
