"""5-minute sanity check for every loadable model.

For each model in MODELS_TO_RUN (env; defaults to raddino,dinov2,biomedclip
since dinov3 is access-gated):
  - loads without error
  - returns expected CLS embedding dim
  - patch_local produces non-NaN output
  - reports actual patch grid for the given input resolution

Catches silent preprocessor / embedding-head / grid bugs before burning GPU hours.
"""
import os, sys, time
os.environ["DATASET"] = "nih"
sys.path.insert(0, "/home/saptpurk/embeddings-noise-eliminators")

import numpy as np
from common import (
    get_config, MODELS, HF_TOKEN, models_to_run,
    EmbeddingExtractor, load_disease_labels, load_and_pad,
)

CFG = get_config()
df = load_disease_labels(CFG, ["cardiomegaly"])
paths = df["image_path"].sample(n=10, random_state=42).tolist()
imgs = [load_and_pad(p, CFG.target_size) for p in paths]

# Fake patch_locations per image for patch_local pooling
H, W = CFG.target_size
fake_locs = [[{"y": H // 2 - 32, "x": W // 2 - 32, "size": 64}] for _ in paths]

targets = models_to_run()
print(f"Testing {targets} at image_hw=({H}, {W})\n")

all_pass = True
for mn in targets:
    t0 = time.time()
    print(f"=== {mn} ===")
    try:
        tok = HF_TOKEN if MODELS[mn]["requires_token"] else None
        ext = EmbeddingExtractor(mn, hf_token=tok)
        out = ext.extract_all(imgs, fake_locs, (H, W))
        dt = time.time() - t0
        expected = MODELS[mn]["cls_dim"]
        shape_ok = out["cls"].shape == (10, expected)
        no_nan = not np.isnan(out["patch_local"]).any()
        print(f"  cls shape     : {out['cls'].shape}   (expected (10, {expected}))  {'OK' if shape_ok else 'FAIL'}")
        print(f"  patch grid    : {out['grid_hw']}")
        print(f"  patch_local   : shape={out['patch_local'].shape} NaN-free={no_nan}")
        print(f"  load+infer    : {dt:.1f}s\n")
        if not (shape_ok and no_nan):
            all_pass = False
        ext.close()
    except Exception as e:
        all_pass = False
        print(f"  FAIL: {type(e).__name__}: {e}\n")

print("ALL MODELS PASSED" if all_pass else "SOME MODELS FAILED")
sys.exit(0 if all_pass else 1)
