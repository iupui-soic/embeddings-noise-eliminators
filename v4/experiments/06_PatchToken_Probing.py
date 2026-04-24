# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Experiment 06 — Patch-Token Probing (CLS vs patch_mean vs patch_local)
#
# **Purpose.**  Arbitrate the patch-token-dilution hypothesis directly.
# Reviewer 3 (RYAI, #11) noted that CLS-only probing may miss information
# that survives in patch tokens.  This experiment probes three pooling
# modes on the *same* perturbed inputs:
#
#  * `cls`         — as in v1–v3
#  * `patch_mean`  — mean over all patch tokens (global non-CLS)
#  * `patch_local` — mean over patch tokens intersecting the injected patch
#
# If `patch_local` AUC >> `cls` AUC, the paper's "dilution" mechanism is
# *directly supported* by data.  If all three are low, the paper's
# "representational suppression" claim strengthens.
#
# **How to run.**
#     export DATASET=nih
#     jupyter nbconvert --execute --to notebook 06_PatchToken_Probing.ipynb
#
# Output: `exp06_<dataset>_patch_probing.parquet`.

# %%
import os, sys, gc, json
from pathlib import Path
REPO_ROOT = Path(os.environ.get("REPO_ROOT", "/home/saptpurk/embeddings-noise-eliminators/v4"))
sys.path.insert(0, str(REPO_ROOT))

from common import (
    get_config, PARAMS, MODELS, HF_TOKEN, models_to_run,
    LocalizedBlurInjector, DirectionalMotionBlurInjector,
    ReticularPatternInjector, GroundGlassInjector,
    EmbeddingExtractor, train_probe,
    load_disease_labels, load_and_pad, stratified_split,
)

CFG = get_config()
OUT = CFG.output_dir("exp06_patch_probing")
print(f"Dataset: {CFG.name}  |  Output: {OUT}")

# %%
import numpy as np, pandas as pd, torch
from tqdm.auto import tqdm

SEED       = PARAMS.random_seed
MODEL_NAMES = models_to_run()
print(f"Running models: {MODEL_NAMES}")

PERTURBATIONS = [
    ("iso_blur_p4",          LocalizedBlurInjector(seed=SEED), 4),
    ("iso_blur_p8",          LocalizedBlurInjector(seed=SEED), 8),
    ("dir_blur_cranio_p64",
        DirectionalMotionBlurInjector(seed=SEED, angle_deg=90.0,
                                      kernel_length=PARAMS.directional_kernel_length),
        64),
    ("dir_blur_lateral_p64",
        DirectionalMotionBlurInjector(seed=SEED, angle_deg=0.0,
                                      kernel_length=PARAMS.directional_kernel_length),
        64),
    ("reticular_fine_p32",
        ReticularPatternInjector(seed=SEED, period_px=3,
                                 amplitude=PARAMS.reticular_amplitude), 32),
    ("ground_glass_p64",
        GroundGlassInjector(seed=SEED, sigma_px=PARAMS.ground_glass_sigma,
                            amplitude=PARAMS.ground_glass_amplitude), 64),
]

# %% [markdown]
# ## 1. Balanced noise-detection dataset (50:50 clean vs perturbed)

# %%
df_all = load_disease_labels(CFG, ["cardiomegaly"])   # disease unused here
rng = np.random.default_rng(SEED)

# Sample a manageable subset for a detection task (target ~30-40k total)
n_target = min(40_000, len(df_all))
idx = rng.choice(len(df_all), size=n_target, replace=False)
df = df_all.iloc[idx].reset_index(drop=True)

# Split 80/20 for probe training
df["_stratum"] = "0"
train_df, test_df = stratified_split(df, "_stratum", test_frac=0.2, seed=SEED)
print(f"train={len(train_df)}  test={len(test_df)}")

# %% [markdown]
# ## 2. Extract CLS, patch-mean, patch-local for clean vs perturbed
#
# Note we make one forward pass per image-perturbation combo and save all
# three pooling modes together — cheapest use of GPU time.

# %%
def extract_three_pools(extractor, df, injector, patch_size, tag):
    """
    Return three dicts keyed by (tag, variant) -> (N, D) array, plus
    a parallel noise-label array (0 clean / 1 perturbed).  Writes per-
    model parquet cache so rerunning is cheap.
    """
    cache = OUT / f"{extractor.model_name}_{tag}.npz"
    if cache.exists():
        npz = np.load(cache)
        return {k: npz[k] for k in npz.files}

    pools = {"cls_clean":[], "cls_pert":[],
             "pm_clean":[], "pm_pert":[],
             "pl_clean":[], "pl_pert":[]}

    # parallel_iter returns clean + perturbed images; we still need the
    # per-image patch_locations for patch_local pooling, so recompute them
    # deterministically in the main thread using the same injector + seed.
    from common import parallel_iter
    clean_rows_cls, clean_rows_pm, clean_rows_pl = [], [], []
    pert_rows_cls,  pert_rows_pm,  pert_rows_pl  = [], [], []
    n_batches_total = (len(df) + PARAMS.batch_size - 1) // PARAMS.batch_size
    pbar = tqdm(parallel_iter(df["image_path"].tolist(), CFG.target_size,
                              batch_size=PARAMS.batch_size,
                              num_workers=PARAMS.num_workers,
                              injector=injector, patch_size=patch_size),
                total=n_batches_total, desc=f"{extractor.model_name}/{tag}")
    for clean_imgs, pert_imgs, paths in pbar:
        # Recompute patch_locations deterministically for patch_local pooling
        pert_locs = []
        for cln, p in zip(clean_imgs, paths):
            _, meta = injector(cln, patch_size=patch_size, num_patches=1,
                               image_path=p)
            pert_locs.append(meta["patch_locations"])

        # Clean pass (no artefact location -> patch_local == patch_mean)
        no_loc = [[] for _ in clean_imgs]
        out_c = extractor.extract_all(clean_imgs, no_loc, CFG.target_size)
        # Perturbed pass with artefact locations
        out_p = extractor.extract_all(pert_imgs, pert_locs, CFG.target_size)

        pools["cls_clean"].append(out_c["cls"])
        pools["cls_pert"].append(out_p["cls"])
        pools["pm_clean"].append(out_c["patch_mean"])
        pools["pm_pert"].append(out_p["patch_mean"])
        pools["pl_clean"].append(out_c["patch_mean"])   # == patch_mean for clean
        pools["pl_pert"].append(out_p["patch_local"])

    stacked = {k: np.vstack(v).astype(np.float32) for k, v in pools.items()}
    np.savez_compressed(cache, **stacked)
    return stacked


def probe_three_modes(pools_tr, pools_te, tag):
    """Train + evaluate linear probes for cls / patch_mean / patch_local."""
    def build(pools):
        X_clean = pools["cls_clean"]       # any mode has same N
        N = len(X_clean)
        y = np.concatenate([np.zeros(N), np.ones(N)]).astype(int)
        sets = {}
        for mode, ck, pk in [("cls", "cls_clean", "cls_pert"),
                             ("patch_mean", "pm_clean", "pm_pert"),
                             ("patch_local", "pl_clean", "pl_pert")]:
            X = np.vstack([pools[ck], pools[pk]])
            sets[mode] = (X, y)
        return sets

    sets_tr = build(pools_tr)
    sets_te = build(pools_te)

    out = []
    for mode, (Xtr, ytr) in sets_tr.items():
        Xte, yte = sets_te[mode]
        res, _ = train_probe(
            Xtr, ytr, Xte, yte,
            name=f"{tag}/{mode}",
            C_grid=PARAMS.lr_C_grid,
            n_boot=PARAMS.n_bootstrap,
            max_iter=PARAMS.lr_max_iter,
            seed=SEED, verbose=False,
        )
        out.append(dict(pooling=mode, auc=res.auc,
                        auc_ci_low=res.auc_ci[0], auc_ci_high=res.auc_ci[1],
                        f1=res.f1, threshold=res.threshold,
                        best_C=res.best_C,
                        n_train=res.n_train, n_test=res.n_test))
    return out

# %%
records = []
for model_name in MODEL_NAMES:
    print(f"\n=== {model_name.upper()} ===")
    ext = EmbeddingExtractor(
        model_name,
        hf_token=HF_TOKEN if MODELS[model_name]["requires_token"] else None,
    )
    for pert_name, injector, patch_size in PERTURBATIONS:
        tr_tag = f"train_{pert_name}"
        te_tag = f"test_{pert_name}"
        pools_tr = extract_three_pools(ext, train_df, injector, patch_size, tr_tag)
        pools_te = extract_three_pools(ext, test_df,  injector, patch_size, te_tag)

        for row in probe_three_modes(pools_tr, pools_te, f"{model_name}/{pert_name}"):
            row.update(dict(dataset=CFG.dataset, model=model_name,
                            perturbation=pert_name, patch_size=patch_size))
            records.append(row)
            print(f"  {model_name:>8s} {pert_name:>22s} {row['pooling']:>12s}  "
                  f"AUC={row['auc']:.4f} "
                  f"[{row['auc_ci_low']:.4f}, {row['auc_ci_high']:.4f}]")
    ext.close()
    del ext; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# %%
df_out = pd.DataFrame(records)
_run_tag = os.environ.get("RUN_TAG", "")
_suffix = ("_" + _run_tag) if _run_tag else ""
out_path = OUT / f"exp06_{CFG.dataset}_patch_probing{_suffix}.parquet"
df_out.to_parquet(out_path, index=False)
print(f"\nSaved -> {out_path}")
df_out
