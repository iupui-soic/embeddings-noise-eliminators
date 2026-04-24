#!/usr/bin/env python3
"""
Build the MIMIC-CXR-JPG subsample manifest.

Produces mimic_subsample_ids.parquet, a reproducible manifest of the exact
(dicom_id, subject_id, study_id, split) rows used by every MIMIC experiment
in the pipeline.

Design:
  - Frontal views only (AP/PA), matching NIH / Emory protocol.
  - Official MIMIC-CXR train/validate/test split is preserved verbatim.
    Patient-disjointness across splits is therefore guaranteed by PhysioNet.
  - Evaluation (test + validate) uses the FULL official sets - no subsampling.
  - Probe-training is subsampled within the official train split:
        * stratification key: 3-bit label pattern
          (Cardiomegaly, Edema, Lung Lesion; CheXpert U-Zero convention).
        * target n = TARGET_N_TRAIN (default 50,000)
        * seed = 42 (fixed)
  - Output columns kept at image (dicom_id) level so downstream code filters
    with a single isin() call.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
TARGET_N_TRAIN = 50_000
MIMIC_ROOT = Path("/data0/MIMIC-CXR")
OUT_PATH = Path(
    "/home/saptpurk/embeddings-noise-eliminators/v4_work/mimic_subsample_ids.parquet"
)


def build(target_n_train: int = TARGET_N_TRAIN, seed: int = SEED,
          out_path: Path = OUT_PATH) -> pd.DataFrame:
    meta = pd.read_csv(MIMIC_ROOT / "mimic-cxr-2.0.0-metadata.csv.gz")
    lab = pd.read_csv(MIMIC_ROOT / "mimic-cxr-2.0.0-chexpert.csv.gz")
    spl = pd.read_csv(MIMIC_ROOT / "mimic-cxr-2.0.0-split.csv.gz")

    meta = meta[meta["ViewPosition"].isin(["AP", "PA"])].copy()

    df = meta.merge(lab, on=["subject_id", "study_id"], how="inner")
    df = df.merge(
        spl[["subject_id", "study_id", "dicom_id", "split"]],
        on=["subject_id", "study_id", "dicom_id"], how="left",
    )
    df = df[df["split"].isin(["train", "validate", "test"])].copy()

    for d_col in ["Cardiomegaly", "Edema", "Lung Lesion"]:
        df[d_col] = (df[d_col].fillna(0).astype(float) > 0.5).astype(int)

    df["strata"] = (df["Cardiomegaly"].astype(str)
                    + df["Edema"].astype(str)
                    + df["Lung Lesion"].astype(str))

    keep_test = df[df["split"] == "test"].copy()
    keep_val = df[df["split"] == "validate"].copy()

    train = df[df["split"] == "train"].copy()
    frac = min(1.0, target_n_train / len(train))

    def _pick(g):
        n = max(1, int(round(len(g) * frac)))
        return g.sample(n=min(n, len(g)), random_state=seed)

    train_sub = (train.groupby("strata", group_keys=False)
                      .apply(_pick)
                      .reset_index(drop=True))

    out = pd.concat(
        [train_sub, keep_val, keep_test], ignore_index=True
    )
    out = out.assign(subsampled=True)
    out = out[["dicom_id", "subject_id", "study_id", "split", "strata",
               "Cardiomegaly", "Edema", "Lung Lesion", "subsampled"]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print(f"Wrote {len(out):,} rows to {out_path}")
    print("\nPer-split counts (frontal, post-subsample):")
    print(out.groupby("split").size().to_string())
    print("\nPer-split label prevalence:")
    for s in ["train", "validate", "test"]:
        sub = out[out["split"] == s]
        row = {d: f"{sub[d].mean():.3f}" for d in
               ("Cardiomegaly", "Edema", "Lung Lesion")}
        print(f"  {s:8s}  n={len(sub):7,d}  {row}")

    full_train = df[df["split"] == "train"]
    print("\nFull-train vs subsample prevalence drift (pp):")
    for d in ("Cardiomegaly", "Edema", "Lung Lesion"):
        drift = (train_sub[d].mean() - full_train[d].mean()) * 100
        print(f"  {d:15s}  full={full_train[d].mean():.3f}  "
              f"sub={train_sub[d].mean():.3f}  Δ={drift:+.2f}pp")

    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=TARGET_N_TRAIN)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()
    build(target_n_train=args.n_train, seed=args.seed, out_path=args.out)
