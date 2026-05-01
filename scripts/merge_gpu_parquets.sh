#!/usr/bin/env bash
# After both GPU sub-pipelines finish, concatenate the per-GPU parquets into
# the canonical filenames fill_placeholders.py expects.
set -eu
DS="${1:-nih}"
WORK="/home/saptpurk/embeddings-noise-eliminators/outputs"
source /home/saptpurk/embeddings-noise-eliminators/.venv/bin/activate

python3 - <<PY
import pandas as pd
from pathlib import Path
WORK = Path("$WORK")
DS = "$DS"

jobs = [
    ("exp01_disease_classification", f"exp01_{DS}_results"),
    ("exp02_geometric",              f"exp02_{DS}_results"),
    ("exp03_iso_motion_blur",        f"exp03_{DS}_results"),
    ("exp04_clean_vs_perturbed",     f"exp04_{DS}_results"),
    ("exp05_embedding_viz",          f"exp05_{DS}_embedding_separation"),
    ("exp06_patch_probing",          f"exp06_{DS}_patch_probing"),
    ("exp07_rawpixel_baseline",      f"exp07_{DS}_rawpixel_baseline"),
    ("exp08_directional_blur",       f"exp08_{DS}_directional_blur"),
]
for exp_dir, stem in jobs:
    d = WORK / f"v4_{exp_dir}_{DS}"
    if not d.exists():
        print(f"skip {exp_dir}: dir missing")
        continue
    parts = sorted(d.glob(f"{stem}_*.parquet"))
    if not parts:
        # maybe already merged (no suffix variant)
        print(f"skip {exp_dir}: no per-GPU parts")
        continue
    frames = []
    for p in parts:
        try:
            frames.append(pd.read_parquet(p))
            print(f"  read {p.name}: {len(frames[-1])} rows")
        except Exception as e:
            print(f"  FAILED to read {p.name}: {e}")
    if not frames:
        continue
    merged = pd.concat(frames, ignore_index=True)
    # Deduplicate: for each experiment, the (model,*) key should be unique.
    # pooling (exp06) and mode (exp07) distinguish rows that share
    # (model, perturbation, ...) but report different pooling/baseline variants;
    # without them, cls/patch_mean rows get collapsed into patch_local.
    dedup_cols = [c for c in ["model","disease","pattern","patch_size",
                              "perturbation","direction","kernel_length",
                              "pooling","mode"]
                  if c in merged.columns]
    if dedup_cols:
        merged = merged.drop_duplicates(subset=dedup_cols, keep="last")
    out = d / f"{stem}.parquet"
    merged.to_parquet(out, index=False)
    print(f"  -> {out.name}: {len(merged)} rows merged")
PY
