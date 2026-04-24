# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Experiment 03 — Isotropic Motion Blur Detection (Table 3)
#
# Replicates v1-v3 Table 3 on frozen CLS embeddings.  Consumed by
# `fill_placeholders.py :: load_exp03`.
#
# Patches: iso_p4 (4x4 patch of isotropic Gaussian blur) and iso_p8 (8x8).
#
# **Run.**
#     export DATASET=nih            # or mimic / emory
#     export HF_TOKEN=<token>
#     export MODELS_TO_RUN=raddino,dinov2,dinov3,biomedclip,medsiglip
#     jupyter nbconvert --execute --to notebook --inplace notebooks/03_IsoMotionBlur.ipynb
#
# Output: `exp03_<dataset>_results.parquet` with columns:
#   model, patch_size, auc, auc_ci_low, auc_ci_high, n_test, n_pos_test

# %%
import os, sys, gc
from pathlib import Path
REPO_ROOT = Path(os.environ.get("REPO_ROOT", "/home/saptpurk/embeddings-noise-eliminators/v4"))
sys.path.insert(0, str(REPO_ROOT))

from common import (
    get_config, PARAMS, MODELS, HF_TOKEN, models_to_run,
    LocalizedBlurInjector,
    EmbeddingExtractor, train_probe,
    load_disease_labels, load_and_pad, stratified_split,
)

CFG = get_config()
OUT = CFG.output_dir("exp03_iso_motion_blur")
print(f"Dataset: {CFG.name}  |  Output: {OUT}")

# %%
import numpy as np, pandas as pd, torch
from tqdm.auto import tqdm

SEED = PARAMS.random_seed
MODEL_NAMES = models_to_run()
print(f"Running models: {MODEL_NAMES}")

PATCH_SIZES = [4, 8]

# %%
df_all = load_disease_labels(CFG, ["cardiomegaly"])
rng = np.random.default_rng(SEED)
n_target = min(40_000, len(df_all))
idx = rng.choice(len(df_all), size=n_target, replace=False)
df = df_all.iloc[idx].reset_index(drop=True)
df["_stratum"] = "0"
train_df, test_df = stratified_split(df, "_stratum", test_frac=0.2, seed=SEED)
print(f"train={len(train_df)}  test={len(test_df)}")

# %%
def extract_iso(ext, df_, patch_size, tag):
    cache = OUT / f"emb_{ext.model_name}_p{patch_size}_{tag}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["clean"], d["pert"]
    from common import parallel_iter
    injector = LocalizedBlurInjector(seed=SEED)
    clean_rows, pert_rows = [], []
    n_batches = (len(df_) + PARAMS.batch_size - 1) // PARAMS.batch_size
    pbar = tqdm(parallel_iter(df_["image_path"].tolist(), CFG.target_size,
                              batch_size=PARAMS.batch_size,
                              num_workers=PARAMS.num_workers,
                              injector=injector, patch_size=patch_size),
                total=n_batches, desc=f"{ext.model_name} iso_p{patch_size} / {tag}")
    for clean_imgs, pert_imgs, _ in pbar:
        clean_rows.append(ext.extract_cls(clean_imgs))
        pert_rows.append(ext.extract_cls(pert_imgs))
    clean = np.vstack(clean_rows)
    pert  = np.vstack(pert_rows)
    np.savez_compressed(cache, clean=clean, pert=pert)
    return clean, pert

records = []
for model_name in MODEL_NAMES:
    print(f"\n=== {model_name.upper()} ===")
    tok = HF_TOKEN if MODELS[model_name].get("requires_token") else None
    ext = EmbeddingExtractor(model_name, hf_token=tok)
    for ps in PATCH_SIZES:
        cln_tr, prt_tr = extract_iso(ext, train_df, ps, "train")
        cln_te, prt_te = extract_iso(ext, test_df,  ps, "test")
        Xtr = np.vstack([cln_tr, prt_tr])
        ytr = np.concatenate([np.zeros(len(cln_tr)), np.ones(len(prt_tr))])
        Xte = np.vstack([cln_te, prt_te])
        yte = np.concatenate([np.zeros(len(cln_te)), np.ones(len(prt_te))])
        probe, _ = train_probe(
            Xtr, ytr, Xte, yte,
            name=f"exp03_{model_name}_p{ps}",
            C_grid=PARAMS.lr_C_grid,
            n_boot=PARAMS.n_bootstrap,
            max_iter=PARAMS.lr_max_iter,
            seed=SEED, verbose=False,
        )
        records.append(dict(
            dataset=CFG.dataset, model=model_name, patch_size=ps,
            auc=probe.auc,
            auc_ci_low=probe.auc_ci[0], auc_ci_high=probe.auc_ci[1],
            n_test=int(len(yte)), n_pos_test=int(yte.sum()),
        ))
        print(f"{model_name:>10s} | iso_p{ps}  AUC={probe.auc:.4f} "
              f"[{probe.auc_ci[0]:.4f}, {probe.auc_ci[1]:.4f}]")
    ext.close()
    del ext; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# %%
results_df = pd.DataFrame(records)
_run_tag = os.environ.get("RUN_TAG", "")
_suffix = ("_" + _run_tag) if _run_tag else ""
out_path = OUT / f"exp03_{CFG.dataset}_results{_suffix}.parquet"
results_df.to_parquet(out_path, index=False)
print(f"\nSaved {len(results_df)} rows -> {out_path}")
print(results_df.pivot(index="patch_size", columns="model", values="auc"))
