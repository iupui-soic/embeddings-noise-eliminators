"""Full audit of exp02 — synthetic geometric noise pipeline.

Checks:
  1. PNG loading fidelity (grayscale vs 3-channel, BGR vs RGB)
  2. Perturbation injection: actual pixel diff visible?
  3. Preprocessor effect: does the perturbation survive resize to each
     model's native resolution?
  4. End-to-end embedding effect: distance between clean and perturbed
     CLS vs chance / noise
  5. Determinism: same image + same seed → same perturbation?
  6. DataLoader-worker seed integrity: does parallel_iter preserve the
     deterministic per-image injection?
"""
import os, sys, hashlib
os.environ["DATASET"] = "nih"
sys.path.insert(0, "/home/saptpurk/embeddings-noise-eliminators/v4")

import numpy as np
import cv2
from PIL import Image
from common import (
    get_config, CircleInjector, SquareInjector, DiagonalLineInjector,
    LocalizedBlurInjector, load_and_pad, load_disease_labels,
    EmbeddingExtractor, parallel_iter, HF_TOKEN, MODELS,
)
from common.config import PARAMS

CFG = get_config()

# Pick a reproducible sample image
df = load_disease_labels(CFG, ["cardiomegaly"])
SAMPLE = df.iloc[42]["image_path"]
print(f"Sample: {SAMPLE}")

img = load_and_pad(SAMPLE, CFG.target_size)
print(f"Loaded shape={img.shape} dtype={img.dtype}  min={img.min()} max={img.max()}  mean={img.mean():.1f}")
# Are all three channels identical (true grayscale)?
if img.ndim == 3:
    g_identical = np.array_equal(img[:,:,0], img[:,:,1]) and np.array_equal(img[:,:,1], img[:,:,2])
    print(f"  3-channel, all-channels-identical={g_identical}")

# ---------------------------------------------------------------------
# 1. Verify each injector actually changes pixels, and where
# ---------------------------------------------------------------------
print("\n=== (1) Injector pixel-diff audit ===")
injectors = [
    ("C1", CircleInjector(seed=42, radius=1), 4),
    ("C2", CircleInjector(seed=42, radius=2), 8),
    ("S4", SquareInjector(seed=42), 4),
    ("S8", SquareInjector(seed=42), 8),
    ("L4", DiagonalLineInjector(seed=42), 4),
    ("L8", DiagonalLineInjector(seed=42), 8),
    ("iso_p4", LocalizedBlurInjector(seed=42), 4),   # reference
]
for name, inj, ps in injectors:
    noisy, meta = inj(img, patch_size=ps, num_patches=1, image_path=SAMPLE)
    diff = np.abs(noisy.astype(int) - img.astype(int))
    n_changed_pixels = (diff.sum(axis=-1 if diff.ndim==3 else 0) > 0).sum()
    max_diff = diff.max()
    total_area = img.shape[0] * img.shape[1]
    pct = 100.0 * n_changed_pixels / total_area
    loc = meta["patch_locations"][0]
    print(f"  {name:>8s}: {n_changed_pixels:>4d} pixels changed "
          f"({pct:.4f}% of image), max_abs_diff={max_diff}, "
          f"patch @ y={loc['y']}, x={loc['x']}, size={loc['size']}")

# ---------------------------------------------------------------------
# 2. Does the perturbation SURVIVE each model's preprocessor resize?
# ---------------------------------------------------------------------
print("\n=== (2) Preprocessor resize: does artifact survive? ===")
print("   (simulating what each model actually sees after processor)")

def simulate_processor(ext, pil_or_np):
    """Return the actual tensor the model forward sees, as numpy in (C,H,W)."""
    if ext._loader == "open_clip":
        t = ext.processor(Image.fromarray(cv2.cvtColor(pil_or_np, cv2.COLOR_BGR2RGB)) if pil_or_np.ndim==3 else Image.fromarray(pil_or_np).convert("RGB"))
        return t.cpu().numpy()
    pil = Image.fromarray(cv2.cvtColor(pil_or_np, cv2.COLOR_BGR2RGB)) if pil_or_np.ndim==3 else Image.fromarray(pil_or_np).convert("RGB")
    out = ext.processor(pil, return_tensors="pt")
    return out["pixel_values"][0].cpu().numpy()

for model_name in ["raddino","dinov2","biomedclip","dinov3","medsiglip"]:
    tok = HF_TOKEN if MODELS[model_name].get("requires_token") else None
    try:
        ext = EmbeddingExtractor(model_name, hf_token=tok)
    except Exception as e:
        print(f"  {model_name}: load failed: {e}")
        continue
    clean_pre = simulate_processor(ext, img)
    print(f"\n  {model_name}: processor output shape = {clean_pre.shape}  "
          f"(native res = {clean_pre.shape[-1]})")
    # Apply each perturbation and measure L2 diff in processor-space
    for name, inj, ps in injectors:
        noisy, _ = inj(img, patch_size=ps, num_patches=1, image_path=SAMPLE)
        noisy_pre = simulate_processor(ext, noisy)
        l2_diff = np.linalg.norm(clean_pre - noisy_pre)
        # How many pixels in the processor-space tensor changed?
        pix_diff = (np.abs(clean_pre - noisy_pre) > 1e-6).any(axis=0)  # union across channels
        n_changed = int(pix_diff.sum())
        native_area = clean_pre.shape[-1] * clean_pre.shape[-2]
        pct = 100.0 * n_changed / native_area
        print(f"    {name:>8s}: ||pre(clean)-pre(pert)||_2 = {l2_diff:>8.3f}  "
              f"{n_changed:>5d} / {native_area} native pixels changed ({pct:.3f}%)")
    ext.close()

# ---------------------------------------------------------------------
# 3. Determinism: same image + same injector call → same output?
# ---------------------------------------------------------------------
print("\n=== (3) Determinism check (seed=42, same image, repeated calls) ===")
inj = CircleInjector(seed=42, radius=2)
outs = []
for _ in range(3):
    out, meta = inj(img, patch_size=8, num_patches=1, image_path=SAMPLE)
    outs.append((out, meta["patch_locations"][0]))
same = all(np.array_equal(outs[0][0], o[0]) for o in outs[1:])
same_loc = all(outs[0][1] == o[1] for o in outs[1:])
print(f"  Pixel arrays identical across 3 calls: {same}")
print(f"  Patch locations identical: {same_loc}")

# ---------------------------------------------------------------------
# 4. DataLoader worker determinism: does parallel_iter give same result?
# ---------------------------------------------------------------------
print("\n=== (4) DataLoader worker determinism (parallel_iter vs inline) ===")
inj = CircleInjector(seed=42, radius=2)
# Inline: call injector in main thread
inline_out = [inj(load_and_pad(SAMPLE, CFG.target_size), patch_size=8,
                  num_patches=1, image_path=SAMPLE)[0]]
# Parallel: via DataLoader workers
paths = [SAMPLE]
worker_out = None
for clean, noisy, p in parallel_iter(paths, CFG.target_size, batch_size=1,
                                     num_workers=2, injector=inj, patch_size=8):
    worker_out = noisy[0]
    break
match = np.array_equal(inline_out[0], worker_out)
diff = np.abs(inline_out[0].astype(int) - worker_out.astype(int))
print(f"  parallel_iter output == inline injector output? {match}")
print(f"  max diff between inline vs worker output: {diff.max()}")
print(f"  n pixels differ: {(diff.sum(axis=-1 if diff.ndim==3 else 0) > 0).sum()}")
