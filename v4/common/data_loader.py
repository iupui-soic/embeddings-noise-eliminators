"""
common/data_loader.py
=====================

Dataset-agnostic loader for image paths + binary disease labels.

NIH-CXR14 uses a pipe-separated "Finding Labels" column; Emory CXR uses
one binary column per disease.  This module hides that difference so the
experiment notebooks work unchanged on either server.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


def load_disease_labels(cfg, diseases: List[str],
                        split_csv: Optional[Path] = None) -> pd.DataFrame:
    """
    Return a dataframe with columns:
        image_path, split (optional), <disease1>, <disease2>, ...

    Dataset differences are handled here:
      * NIH   : parse "Finding Labels" via pipe-separator
      * Emory : use the explicit binary columns
      * MIMIC : CheXpert labels joined to metadata, delegated to helper
    """
    # MIMIC has a multi-CSV join that doesn't fit the "one CSV, known image
    # column" pattern; delegate up front so the generic path checks below
    # don't trip on it.
    if cfg.label_mode == "mimic_chexpert":
        return _load_mimic_chexpert(cfg, diseases)

    df = pd.read_csv(split_csv or cfg.raw_csv)

    # Image path
    img_col = cfg.image_col
    if img_col not in df.columns:
        raise KeyError(f"Expected image column {img_col!r} in {split_csv or cfg.raw_csv}")
    df["image_path"] = df[img_col].apply(lambda x: str(cfg.img_dir / x)
                                         if not str(x).startswith("/") else str(x))

    # Disease labels
    if cfg.label_mode == "nih_finding_labels":
        if "Finding Labels" not in df.columns:
            raise KeyError("NIH CSV must contain 'Finding Labels'")
        for d in diseases:
            col = cfg.disease_columns[d]
            df[d] = df["Finding Labels"].fillna("").apply(
                lambda s: int(col in s.split("|")))
    elif cfg.label_mode == "binary_columns":
        for d in diseases:
            col = cfg.disease_columns[d]
            if col not in df.columns:
                raise KeyError(
                    f"Expected Emory binary column {col!r} for disease {d!r}")
            df[d] = df[col].astype(int)
    else:
        raise ValueError(f"Unknown label_mode: {cfg.label_mode}")

    keep = ["image_path"] + diseases
    if "split" in df.columns:
        keep.append("split")
    return df[keep].reset_index(drop=True)


def _load_mimic_chexpert(cfg, diseases: List[str]) -> pd.DataFrame:
    """Load MIMIC-CXR-JPG labels + metadata + build per-image paths."""
    labels_path = cfg.raw_csv
    meta_path   = cfg.metadata_csv
    if meta_path is None or not Path(meta_path).exists():
        raise FileNotFoundError(
            f"MIMIC metadata CSV not found at {meta_path}. "
            f"Expected mimic-cxr-2.0.0-metadata.csv[.gz] from the JPG release."
        )
    lab = pd.read_csv(labels_path)
    meta = pd.read_csv(meta_path)

    # Normalize column names we rely on
    for c in ("subject_id", "study_id"):
        if c not in lab.columns:
            raise KeyError(f"Missing {c!r} in {labels_path}")
        if c not in meta.columns:
            raise KeyError(f"Missing {c!r} in {meta_path}")
    if "dicom_id" not in meta.columns:
        raise KeyError(f"Missing 'dicom_id' in {meta_path}")

    # Filter metadata to requested views (AP/PA by default) BEFORE join, cheap
    if cfg.view_filter and "ViewPosition" in meta.columns:
        meta = meta[meta["ViewPosition"].isin(cfg.view_filter)].copy()

    # Join per-image metadata with per-study labels
    df = meta.merge(lab, on=["subject_id", "study_id"], how="inner")

    # CheXpert convention: -1 uncertain, 0 negative, 1 positive, NaN not-mentioned.
    # U-Zero mapping: uncertain → 0 (conservative default, matches Irvin et al. 2019).
    for d in diseases:
        col = cfg.disease_columns[d]
        if col not in df.columns:
            raise KeyError(
                f"Expected CheXpert label column {col!r} for disease {d!r} "
                f"in {labels_path}; available: {[c for c in df.columns if c not in ('subject_id','study_id','dicom_id','ViewPosition')][:14]}")
        df[d] = (df[col].fillna(0).astype(float) > 0.5).astype(int)

    # Build per-image path:  <img_dir>/p{subj[:2]}/p{subj}/s{study}/{dicom}.jpg
    def _mkpath(r):
        subj = str(int(r["subject_id"]))
        study = str(int(r["study_id"]))
        grp = "p" + subj[:2]
        return str(Path(cfg.img_dir) / grp / f"p{subj}" / f"s{study}" / f"{r['dicom_id']}.jpg")
    df["image_path"] = df.apply(_mkpath, axis=1)

    # Merge official split if available (train/validate/test) — preserves
    # comparability with published MIMIC-CXR results.
    split_csv = getattr(cfg, "split_csv", None)
    if split_csv is not None and Path(split_csv).exists():
        spl = pd.read_csv(split_csv)
        # Normalise: MIMIC uses 'split' with values train/validate/test
        if "split" in spl.columns and {"subject_id", "study_id", "dicom_id"}.issubset(spl.columns):
            df = df.merge(spl[["subject_id", "study_id", "dicom_id", "split"]],
                          on=["subject_id", "study_id", "dicom_id"], how="left")

    # Optional subsample filter. If MIMIC_SUBSAMPLE_IDS is set and points at a
    # parquet with a `dicom_id` column, keep only those rows. The manifest is
    # built by v4/build_mimic_subsample.py: probe-training is subsampled within
    # the official train split; official validate/test splits are retained in
    # full.
    sub_path = os.environ.get("MIMIC_SUBSAMPLE_IDS", "").strip()
    if sub_path and Path(sub_path).exists():
        sub = pd.read_parquet(sub_path, columns=["dicom_id"])
        n_before = len(df)
        df = df[df["dicom_id"].isin(sub["dicom_id"])].copy()
        print(f"[mimic subsample] {sub_path} -> kept {len(df):,}/{n_before:,} rows")

    keep = ["image_path"] + diseases
    if "split" in df.columns:
        keep.append("split")
    return df[keep].reset_index(drop=True)


def load_and_pad(image_path: str,
                 target_size: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]
    if (h, w) != target_size:
        max_dim = max(h, w, target_size[0])
        pad_h, pad_w = max_dim - h, max_dim - w
        top, bot = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        value = [0, 0, 0] if img.ndim == 3 else 0
        img = cv2.copyMakeBorder(img, top, bot, left, right,
                                 cv2.BORDER_CONSTANT, value=value)
        if img.shape[:2] != target_size:
            img = cv2.resize(img, target_size[::-1],
                             interpolation=cv2.INTER_LANCZOS4)
    return img


def stratified_split(df: pd.DataFrame, stratify_col: str,
                     test_frac: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    test_idx = []
    for v, g in df.groupby(stratify_col):
        n_test = max(1, int(round(len(g) * test_frac)))
        sampled = rng.choice(g.index.values, size=n_test, replace=False)
        test_idx.extend(sampled.tolist())
    test_mask = df.index.isin(test_idx)
    return df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Parallel image loader (used by exp02-08 to decouple cv2.imread from GPU)
# ---------------------------------------------------------------------------
# cv2.imread on 1024x1024 PNGs pegs at ~50 ms/image single-threaded, making
# image I/O the dominant bottleneck for frozen-inference embedding jobs.
# A DataLoader with worker processes (each running cv2 in its own interpreter)
# typically recovers ~3-5x throughput on a 16-core host.

class CXRDataset:
    """Dataset of (clean_image, perturbed_image_or_None, path) for an injector.

    Yields numpy arrays that the caller batches into the extractor.  Returns
    arrays rather than tensors so we match the existing extract_cls contract.
    """
    def __init__(self, paths, target_size, injector=None, patch_size=None):
        self.paths = list(paths)
        self.target_size = target_size
        self.injector = injector
        self.patch_size = patch_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        clean = load_and_pad(p, self.target_size)
        if self.injector is None:
            return clean, None, p
        noisy, _ = self.injector(clean, patch_size=self.patch_size,
                                 num_patches=1, image_path=p)
        return clean, noisy, p


def _np_collate(batch):
    """Keep images as numpy lists; no stacking (extractor handles processors)."""
    clean = [b[0] for b in batch]
    noisy = [b[1] for b in batch] if batch[0][1] is not None else None
    paths = [b[2] for b in batch]
    return clean, noisy, paths


def parallel_iter(paths, target_size, batch_size: int = 16,
                  num_workers: int = 4, injector=None, patch_size: int = None):
    """Yield (clean_imgs, perturbed_imgs_or_None, paths) tuples with parallel I/O.

    Deterministic order (shuffle=False) so cached embeddings align with paths.
    """
    import torch
    from torch.utils.data import DataLoader
    ds = CXRDataset(paths, target_size, injector, patch_size)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=_np_collate,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )
    for clean_imgs, noisy_imgs, paths in dl:
        yield clean_imgs, noisy_imgs, paths
