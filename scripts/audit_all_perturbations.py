"""Cross-experiment audit — every injector used in every experiment.

Checks for silent failures beyond the clip bug:
  1. Pixel-diff localisation (only patch region changes)
  2. Artifact SIGNAL STRENGTH (is the perturbation visible after resize?)
  3. Clean vs perturbed embedding distinguishable (per model)
  4. Cross-perturbation embeddings actually differ (sanity vs exp02 look-alike)
  5. Deterministic seeding through the DataLoader workers
"""
import os, sys
os.environ["DATASET"] = "nih"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, "/home/saptpurk/embeddings-noise-eliminators/v4")

import numpy as np
from common import (
    get_config, MODELS, HF_TOKEN,
    CircleInjector, SquareInjector, DiagonalLineInjector,
    LocalizedBlurInjector, DirectionalMotionBlurInjector,
    ReticularPatternInjector, GroundGlassInjector,
    load_and_pad, load_disease_labels, parallel_iter,
    EmbeddingExtractor,
)

CFG = get_config()
df = load_disease_labels(CFG, ["cardiomegaly"])

# Full injector catalogue, tagged by which experiment uses them
CATALOGUE = [
    # exp02 (geometric)
    ("exp02/C1",               CircleInjector(seed=42, radius=1), 4),
    ("exp02/C2",               CircleInjector(seed=42, radius=2), 8),
    ("exp02/S4",               SquareInjector(seed=42), 4),
    ("exp02/S8",               SquareInjector(seed=42), 8),
    ("exp02/L4",               DiagonalLineInjector(seed=42), 4),
    ("exp02/L8",               DiagonalLineInjector(seed=42), 8),
    # exp03 (iso motion blur)
    ("exp03/iso_p4",           LocalizedBlurInjector(seed=42), 4),
    ("exp03/iso_p8",           LocalizedBlurInjector(seed=42), 8),
    # exp04 / exp06 (perts used in clean-vs-perturbed + patch-token)
    ("exp04/dir_cranio_p32",   DirectionalMotionBlurInjector(seed=42, angle_deg=90.0,
                                                             kernel_length=21), 32),
    ("exp04/dir_cranio_p64",   DirectionalMotionBlurInjector(seed=42, angle_deg=90.0,
                                                             kernel_length=21), 64),
    ("exp04/dir_lateral_p32",  DirectionalMotionBlurInjector(seed=42, angle_deg=0.0,
                                                             kernel_length=21), 32),
    ("exp04/dir_lateral_p64",  DirectionalMotionBlurInjector(seed=42, angle_deg=0.0,
                                                             kernel_length=21), 64),
    ("exp04/reticular_fine_p32",
        ReticularPatternInjector(seed=42, period_px=3, amplitude=0.08), 32),
    ("exp04/reticular_coarse_p64",
        ReticularPatternInjector(seed=42, period_px=6, amplitude=0.08), 64),
    ("exp04/ground_glass_p64",
        GroundGlassInjector(seed=42, sigma_px=12.0, amplitude=0.06), 64),
    # exp08 (directional kernel grid — spot-check corners)
    ("exp08/dir_k11_p16",      DirectionalMotionBlurInjector(seed=42, kernel_length=11, angle_deg=90.0), 16),
    ("exp08/dir_k31_p64",      DirectionalMotionBlurInjector(seed=42, kernel_length=31, angle_deg=0.0), 64),
]

# Pick 5 different sample images to make the audit more robust than 1 image
SAMPLES = df["image_path"].sample(n=5, random_state=123).tolist()

# ---------------------------------------------------------------------
# Section 1 — pixel-diff localisation check (post-bug-fix)
# ---------------------------------------------------------------------
print("=" * 80)
print("1. PIXEL-DIFF LOCALISATION (should change only within patch region)")
print("=" * 80)
print(f"{'injector':>28s} {'median_px_changed':>18s} {'max_px':>8s} {'patch_area':>12s} {'median_max_diff':>18s}")
results = {}
for name, inj, ps in CATALOGUE:
    n_changed_list, max_diff_list = [], []
    for path in SAMPLES:
        img = load_and_pad(path, CFG.target_size)
        noisy, _ = inj(img, patch_size=ps, num_patches=1, image_path=path)
        diff = np.abs(noisy.astype(int) - img.astype(int))
        n = (diff.sum(axis=-1 if diff.ndim == 3 else 0) > 0).sum()
        n_changed_list.append(n)
        max_diff_list.append(diff.max())
    med_n = int(np.median(n_changed_list))
    max_n = int(max(n_changed_list))
    med_d = int(np.median(max_diff_list))
    area = ps * ps
    results[name] = (med_n, max_n, med_d, area)
    status = "OK" if max_n <= area * 1.3 else "WARN: changes exceed patch"
    print(f"{name:>28s} {med_n:>18d} {max_n:>8d} {area:>12d} {med_d:>18d}  {status}")

# ---------------------------------------------------------------------
# Section 2 — signal strength per injector
# ---------------------------------------------------------------------
print()
print("=" * 80)
print("2. SIGNAL STRENGTH (is the artifact actually visible?)")
print("=" * 80)
print("  A small-radius circle on a featureless bright region can be nearly")
print("  invisible even with a correct injection — check max_diff vs local_std.")
print()
weak_threshold = 3  # abs diff < 3 is below PNG quantisation/noise floor
warnings = []
for name, (_, _, med_d, _) in results.items():
    if med_d < weak_threshold:
        warnings.append((name, med_d))
if warnings:
    print(f"  LOW SIGNAL ({len(warnings)} injectors with median max_diff < {weak_threshold}):")
    for n, d in warnings:
        print(f"    {n:>28s}  median max_diff = {d}")
else:
    print("  All injectors produce median max_diff >= 3 intensity units.")

# ---------------------------------------------------------------------
# Section 3 — embedding-level clean vs perturbed distinguishability
# ---------------------------------------------------------------------
print()
print("=" * 80)
print("3. EMBEDDING-LEVEL DISTINGUISHABILITY (on 40 clean + 40 perturbed images)")
print("=" * 80)
print("  Reports cosine similarity of clean vs perturbed embeddings per injector.")
print("  If ~1.0, the artifact is essentially invisible to that model.")

N = 40
probe_paths = df["image_path"].sample(n=N, random_state=456).tolist()

for model_name in ["raddino", "dinov2", "biomedclip", "medsiglip"]:
    tok = HF_TOKEN if MODELS[model_name].get("requires_token") else None
    try:
        ext = EmbeddingExtractor(model_name, hf_token=tok)
    except Exception as e:
        print(f"\n  [{model_name}] load failed: {e}")
        continue
    # Extract clean CLS for N images once
    clean_imgs = [load_and_pad(p, CFG.target_size) for p in probe_paths]
    X_clean = ext.extract_cls(clean_imgs)
    print(f"\n  {model_name}:")
    print(f"  {'injector':>28s}  {'cos(clean,pert)':>18s}  {'||Δ||/||clean||':>18s}")
    for name, inj, ps in CATALOGUE:
        # Build perturbed set using same images
        pert_imgs = []
        for p in probe_paths:
            img = load_and_pad(p, CFG.target_size)
            noisy, _ = inj(img, patch_size=ps, num_patches=1, image_path=p)
            pert_imgs.append(noisy)
        X_pert = ext.extract_cls(pert_imgs)
        # Per-pair cosine similarity
        def norm(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
        cos = (norm(X_clean) * norm(X_pert)).sum(axis=1)
        rel_l2 = np.linalg.norm(X_clean - X_pert, axis=1) / (np.linalg.norm(X_clean, axis=1) + 1e-9)
        print(f"  {name:>28s}  {cos.mean():>18.5f}  {rel_l2.mean():>18.5f}")
    ext.close()

# ---------------------------------------------------------------------
# Section 4 — cross-perturbation distinguishability
# ---------------------------------------------------------------------
print()
print("=" * 80)
print("4. CROSS-PERTURBATION DIFFERENCES (should all differ; previously identical)")
print("=" * 80)
print("  Pairs of perturbed embeddings across different perturbation types.")
print("  If equal, perturbations are indistinguishable to the model (bad).")

ext = EmbeddingExtractor("raddino", hf_token=None)
N2 = 20
paths = df["image_path"].sample(n=N2, random_state=789).tolist()
pert_embs = {}
subset = [("C1", CircleInjector(seed=42, radius=1), 4),
          ("C2", CircleInjector(seed=42, radius=2), 8),
          ("S4", SquareInjector(seed=42), 4),
          ("S8", SquareInjector(seed=42), 8),
          ("L4", DiagonalLineInjector(seed=42), 4),
          ("L8", DiagonalLineInjector(seed=42), 8)]
for name, inj, ps in subset:
    pert_imgs = []
    for p in paths:
        img = load_and_pad(p, CFG.target_size)
        noisy, _ = inj(img, patch_size=ps, num_patches=1, image_path=p)
        pert_imgs.append(noisy)
    pert_embs[name] = ext.extract_cls(pert_imgs)
ext.close()

print(f"\n  L2 distances between perturbation types (raddino CLS):")
names = list(pert_embs.keys())
for i in range(len(names)):
    for j in range(i+1, len(names)):
        d = np.linalg.norm(pert_embs[names[i]] - pert_embs[names[j]])
        print(f"    {names[i]:>4s} vs {names[j]:>4s}: {d:>8.3f}")

print()
print("=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)
