# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Experiment 07 — Raw-Pixel Baseline at Insertion Site
#
# **Purpose.**  Answer Reviewer 3 (RYAI, #11) directly: "train a simple
# linear classifier on raw pixel intensities at the insertion site."
#
# If a logistic regression on raw pixel values around the known patch
# location achieves high AUC, then the artefact is a trivially detectable
# signal in pixel space.  ViT foundation-model embeddings failing to
# detect it is therefore a *representational* failure rather than a
# task-difficulty issue.
#
# **Important.**  We do NOT crop using ground-truth location for
# evaluation in the main paper — we use a fixed image-level feature
# (flattened downsampled image) OR a "known-location" oracle crop.
# Reporting both lets reviewers see the full picture:
#
#  * `raw_global`   — flattened 64x64 downsample of the whole image
#  * `raw_oracle`   — flattened 32x32 window centred on the injected patch
#                     (ceiling: what a detector *could* achieve with perfect
#                     location prior)
#
# **How to run.**
#     export DATASET=nih
#     jupyter nbconvert --execute --to notebook 07_RawPixel_Baseline.ipynb
#
# Output: `exp07_<dataset>_rawpixel_baseline.parquet`

# %%
import os, sys, json
from pathlib import Path
REPO_ROOT = Path(os.environ.get("REPO_ROOT", "/home/saptpurk/embeddings-noise-eliminators/v4"))
sys.path.insert(0, str(REPO_ROOT))

from common import (
    get_config, PARAMS,
    LocalizedBlurInjector, DirectionalMotionBlurInjector,
    ReticularPatternInjector, GroundGlassInjector,
    train_probe, load_disease_labels, load_and_pad, stratified_split,
)

CFG = get_config()
OUT = CFG.output_dir("exp07_rawpixel_baseline")
print(f"Dataset: {CFG.name}  |  Output: {OUT}")

# %%
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

SEED = PARAMS.random_seed
GLOBAL_DS = 64       # downsample the whole image to 64x64 = 4096 features
ORACLE_PATCH = 32    # 32x32 window centred on the artefact = 1024 features

PERTURBATIONS = [
    ("iso_blur_p4",          LocalizedBlurInjector(seed=SEED), 4),
    ("iso_blur_p8",          LocalizedBlurInjector(seed=SEED), 8),
    ("dir_blur_cranio_p64",
        DirectionalMotionBlurInjector(seed=SEED, angle_deg=90.0,
                                      kernel_length=PARAMS.directional_kernel_length),
        64),
    ("reticular_fine_p32",
        ReticularPatternInjector(seed=SEED, period_px=3,
                                 amplitude=PARAMS.reticular_amplitude), 32),
    ("ground_glass_p64",
        GroundGlassInjector(seed=SEED, sigma_px=PARAMS.ground_glass_sigma,
                            amplitude=PARAMS.ground_glass_amplitude), 64),
]

# %%
df_all = load_disease_labels(CFG, ["cardiomegaly"])
rng = np.random.default_rng(SEED)
n_target = min(20_000, len(df_all))     # raw-pixel features are cheap; smaller sample OK
idx = rng.choice(len(df_all), size=n_target, replace=False)
df = df_all.iloc[idx].reset_index(drop=True)
df["_stratum"] = "0"
train_df, test_df = stratified_split(df, "_stratum", test_frac=0.2, seed=SEED)
print(f"train={len(train_df)}  test={len(test_df)}")

# %%
def _to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def global_features(img):
    g = _to_gray(img).astype(np.float32) / 255.0
    ds = cv2.resize(g, (GLOBAL_DS, GLOBAL_DS), interpolation=cv2.INTER_AREA)
    return ds.reshape(-1)

def oracle_features(img, patch_locations):
    """Extract a 32x32 window centred on the (first) injected patch."""
    g = _to_gray(img).astype(np.float32) / 255.0
    H, W = g.shape
    if not patch_locations:
        cy, cx = H // 2, W // 2
    else:
        loc = patch_locations[0]
        cy = loc["y"] + loc["size"] // 2
        cx = loc["x"] + loc["size"] // 2
    hh = ORACLE_PATCH // 2
    y0 = max(0, cy - hh); y1 = min(H, cy + hh)
    x0 = max(0, cx - hh); x1 = min(W, cx + hh)
    crop = g[y0:y1, x0:x1]
    if crop.shape != (ORACLE_PATCH, ORACLE_PATCH):
        crop = cv2.resize(crop, (ORACLE_PATCH, ORACLE_PATCH))
    return crop.reshape(-1)

# %%
def build_features(df, injector, patch_size):
    X_global_clean, X_global_pert = [], []
    X_oracle_clean, X_oracle_pert = [], []
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"features {injector.__class__.__name__}/{patch_size}"):
        clean = load_and_pad(row["image_path"], CFG.target_size)
        noisy, meta = injector(clean, patch_size=patch_size, num_patches=1,
                               image_path=row["image_path"])
        X_global_clean.append(global_features(clean))
        X_global_pert.append(global_features(noisy))
        X_oracle_clean.append(oracle_features(clean, meta["patch_locations"]))
        X_oracle_pert.append(oracle_features(noisy, meta["patch_locations"]))
    return (np.stack(X_global_clean), np.stack(X_global_pert),
            np.stack(X_oracle_clean), np.stack(X_oracle_pert))

# %%
records = []
for pert_name, injector, patch_size in PERTURBATIONS:
    print(f"\n--- {pert_name} ---")
    Xg_tr_c, Xg_tr_p, Xo_tr_c, Xo_tr_p = build_features(train_df, injector, patch_size)
    Xg_te_c, Xg_te_p, Xo_te_c, Xo_te_p = build_features(test_df,  injector, patch_size)

    for mode, (Xtr_c, Xtr_p, Xte_c, Xte_p) in [
        ("raw_global", (Xg_tr_c, Xg_tr_p, Xg_te_c, Xg_te_p)),
        ("raw_oracle", (Xo_tr_c, Xo_tr_p, Xo_te_c, Xo_te_p)),
    ]:
        Xtr = np.vstack([Xtr_c, Xtr_p])
        ytr = np.concatenate([np.zeros(len(Xtr_c)), np.ones(len(Xtr_p))]).astype(int)
        Xte = np.vstack([Xte_c, Xte_p])
        yte = np.concatenate([np.zeros(len(Xte_c)), np.ones(len(Xte_p))]).astype(int)

        res, _ = train_probe(
            Xtr, ytr, Xte, yte,
            name=f"{pert_name}/{mode}",
            C_grid=PARAMS.lr_C_grid,
            n_boot=PARAMS.n_bootstrap,
            max_iter=PARAMS.lr_max_iter,
            seed=SEED, verbose=False,
        )
        records.append(dict(
            dataset=CFG.dataset, perturbation=pert_name, mode=mode,
            auc=res.auc, auc_ci_low=res.auc_ci[0], auc_ci_high=res.auc_ci[1],
            f1=res.f1, threshold=res.threshold, best_C=res.best_C,
            n_features=int(Xtr.shape[1]),
            n_train=res.n_train, n_test=res.n_test,
        ))
        print(f"  {mode:>10s}  AUC={res.auc:.4f} "
              f"[{res.auc_ci[0]:.4f}, {res.auc_ci[1]:.4f}]  "
              f"features={Xtr.shape[1]}")

# %%
df_out = pd.DataFrame(records)
_run_tag = os.environ.get("RUN_TAG", "")
_suffix = ("_" + _run_tag) if _run_tag else ""
out_path = OUT / f"exp07_{CFG.dataset}_rawpixel_baseline{_suffix}.parquet"
df_out.to_parquet(out_path, index=False)
print(f"\nSaved -> {out_path}")
df_out
