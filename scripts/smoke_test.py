"""End-to-end smoke test for v4 pipeline on NIH-CXR14.

Small, fast: 200 images, RAD-DINO only (public, no HF token needed),
one perturbation, linear probe AUC on cardiomegaly.
Runs in ~2-3 min on one RTX 6000 Ada.
"""
import os, sys, time
os.environ["DATASET"] = "nih"

REPO_ROOT = "/home/saptpurk/embeddings-noise-eliminators"
sys.path.insert(0, os.path.join(REPO_ROOT, "v4"))

import numpy as np
import pandas as pd
import torch

from common import (
    get_config, PARAMS,
    LocalizedBlurInjector, EmbeddingExtractor,
    load_disease_labels, load_and_pad, stratified_split,
    train_probe,
)

t0 = time.time()
CFG = get_config()
print(f"[+{time.time()-t0:.1f}s] Dataset: {CFG.name}  img_dir={CFG.img_dir}  work_dir={CFG.work_dir}")
print(f"[+{time.time()-t0:.1f}s] CUDA: {torch.cuda.is_available()} devices={torch.cuda.device_count()}")

# Load labels
df = load_disease_labels(CFG, ["cardiomegaly"])
print(f"[+{time.time()-t0:.1f}s] Loaded labels: {len(df)} rows, positives={int(df['cardiomegaly'].sum())}")

# Subsample 200 rows with both classes represented
pos = df[df["cardiomegaly"] == 1].sample(n=100, random_state=42)
neg = df[df["cardiomegaly"] == 0].sample(n=100, random_state=42)
small = pd.concat([pos, neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)
train_df, test_df = stratified_split(small, "cardiomegaly", test_frac=0.3, seed=42)
print(f"[+{time.time()-t0:.1f}s] Split: train={len(train_df)} test={len(test_df)}")

# Test a single image load
img = load_and_pad(train_df.iloc[0]["image_path"], CFG.target_size)
print(f"[+{time.time()-t0:.1f}s] First image loaded: shape={img.shape} dtype={img.dtype}")

# Load RAD-DINO
print(f"[+{time.time()-t0:.1f}s] Loading RAD-DINO...")
ext = EmbeddingExtractor("raddino", hf_token=None)
print(f"[+{time.time()-t0:.1f}s] RAD-DINO loaded")

def extract(df_):
    out = []
    bs = PARAMS.batch_size
    for i in range(0, len(df_), bs):
        imgs = [load_and_pad(p, CFG.target_size) for p in df_.iloc[i:i+bs]["image_path"]]
        out.append(ext.extract_cls(imgs))
    return np.vstack(out)

Xtr_clean = extract(train_df)
Xte_clean = extract(test_df)
print(f"[+{time.time()-t0:.1f}s] Clean embeddings: train={Xtr_clean.shape} test={Xte_clean.shape}")

# Perturb test set with iso-blur patch=4
inj = LocalizedBlurInjector(seed=42)
def extract_pert(df_):
    out = []
    bs = PARAMS.batch_size
    for i in range(0, len(df_), bs):
        imgs = []
        for _, row in df_.iloc[i:i+bs].iterrows():
            clean = load_and_pad(row["image_path"], CFG.target_size)
            noisy, _ = inj(clean, patch_size=4, num_patches=1, image_path=row["image_path"])
            imgs.append(noisy)
        out.append(ext.extract_cls(imgs))
    return np.vstack(out)

Xte_pert = extract_pert(test_df)
print(f"[+{time.time()-t0:.1f}s] Perturbed test embeddings: {Xte_pert.shape}")

# Linear probe on cardiomegaly (clean)
ytr = train_df["cardiomegaly"].values
yte = test_df["cardiomegaly"].values
probe, art = train_probe(
    Xtr_clean, ytr, Xte_clean, yte,
    name="smoke_raddino_cardiomegaly_clean",
    C_grid=(0.1, 1.0), n_boot=100, max_iter=1000, seed=42, verbose=False,
)
print(f"[+{time.time()-t0:.1f}s] AUC (clean test) = {probe.auc:.4f}  [{probe.auc_ci[0]:.4f}, {probe.auc_ci[1]:.4f}]")

# Perturbation detectability probe (clean vs perturbed test)
X_det = np.vstack([Xte_clean, Xte_pert])
y_det = np.concatenate([np.zeros(len(Xte_clean)), np.ones(len(Xte_pert))])
# Use the same samples for train/test — just a sanity check, not a proper probe
probe2, _ = train_probe(
    X_det, y_det, X_det, y_det,
    name="smoke_iso_blur_detect", C_grid=(1.0,), n_boot=50, max_iter=500, seed=42, verbose=False,
)
print(f"[+{time.time()-t0:.1f}s] iso-blur detectability AUC (in-sample) = {probe2.auc:.4f}")

ext.close()
print(f"\n[+{time.time()-t0:.1f}s] SMOKE TEST PASSED")
