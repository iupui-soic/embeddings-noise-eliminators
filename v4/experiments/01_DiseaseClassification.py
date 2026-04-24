# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Experiment 01 — Disease Classification Baselines (clean images)
#
# **Purpose.**  Table 1 of the manuscript: linear-probe AUC + F1 for every
# foundation model on every disease, trained and evaluated on CLEAN images.
# Directly consumed by `fill_placeholders.py :: load_exp01`.
#
# **Run.**
#     export DATASET=nih            # or mimic / emory
#     export HF_TOKEN=<token>       # for gated models (dinov3, medsiglip)
#     export MODELS_TO_RUN=raddino,dinov2,dinov3,biomedclip,medsiglip
#     jupyter nbconvert --execute --to notebook --inplace notebooks/01_DiseaseClassification.ipynb
#
# Output: `exp01_<dataset>_results.parquet` with columns:
#   model, disease, auc, f1, auc_ci_low, auc_ci_high, f1_ci_low, f1_ci_high, best_C

# %%
import os, sys, gc
from pathlib import Path
REPO_ROOT = Path(os.environ.get("REPO_ROOT", "/home/saptpurk/embeddings-noise-eliminators/v4"))
sys.path.insert(0, str(REPO_ROOT))

from common import (
    get_config, PARAMS, MODELS, HF_TOKEN, models_to_run,
    EmbeddingExtractor, train_probe,
    load_disease_labels, load_and_pad, stratified_split,
)

CFG = get_config()
OUT = CFG.output_dir("exp01_disease_classification")
print(f"Dataset: {CFG.name}  |  Output: {OUT}")

# %%
import numpy as np, pandas as pd, torch
from tqdm.auto import tqdm

DISEASES = ["cardiomegaly", "edema", "lung_lesion"]
MODEL_NAMES = models_to_run()
SEED = PARAMS.random_seed
print(f"Running models: {MODEL_NAMES}")

# %% [markdown]
# ## 1. Build disease-stratified split (shared with exp04 via disk cache)

# %%
SHARED_SPLIT = CFG.output_dir("exp04_clean_vs_perturbed") / "disease_split.csv"
SPLIT_PATH = OUT / "disease_split.csv"

if SHARED_SPLIT.exists():
    df_all = pd.read_csv(SHARED_SPLIT)
    print(f"Reusing shared split from exp04: {len(df_all)} rows")
elif SPLIT_PATH.exists():
    df_all = pd.read_csv(SPLIT_PATH)
    print(f"Loaded cached split: {len(df_all)} rows")
else:
    df_all = load_disease_labels(CFG, DISEASES)
    df_all["_stratum"] = df_all[DISEASES].astype(str).agg("".join, axis=1)
    train_df, test_df = stratified_split(df_all, "_stratum",
                                         test_frac=0.2, seed=SEED)
    train_df["split"], test_df["split"] = "train", "test"
    df_all = pd.concat([train_df, test_df], ignore_index=True)
    df_all.drop(columns=["_stratum"], inplace=True)
    df_all.to_csv(SPLIT_PATH, index=False)
    print(f"Created split: train={(df_all.split=='train').sum()} test={(df_all.split=='test').sum()}")

train_df = df_all[df_all["split"] == "train"].reset_index(drop=True)
test_df  = df_all[df_all["split"] == "test"].reset_index(drop=True)

# %% [markdown]
# ## 2. Extract CLEAN CLS embeddings and fit linear probes per (model, disease)

# %%
def extract_clean(ext, df, tag):
    """Cache per-model clean embeddings in the shared exp04 directory so
    exp01 and exp04 reuse the same extraction."""
    cache_dir = CFG.output_dir("exp04_clean_vs_perturbed")
    emb_path = cache_dir / f"emb_clean_{ext.model_name}_{tag}.npy"
    if emb_path.exists():
        return np.load(emb_path)
    from common import parallel_iter
    out = []
    paths = df["image_path"].tolist()
    n_batches = (len(paths) + PARAMS.batch_size - 1) // PARAMS.batch_size
    for clean_imgs, _, _ in tqdm(
            parallel_iter(paths, CFG.target_size,
                          batch_size=PARAMS.batch_size,
                          num_workers=PARAMS.num_workers),
            total=n_batches,
            desc=f"CLEAN {tag} / {ext.model_name}"):
        out.append(ext.extract_cls(clean_imgs))
    X = np.vstack(out)
    np.save(emb_path, X)
    return X

records = []
for model_name in MODEL_NAMES:
    print(f"\n=== {model_name.upper()} ===")
    tok = HF_TOKEN if MODELS[model_name].get("requires_token") else None
    ext = EmbeddingExtractor(model_name, hf_token=tok)
    Xtr = extract_clean(ext, train_df, "train")
    Xte = extract_clean(ext, test_df,  "test")
    for disease in DISEASES:
        ytr = train_df[disease].values
        yte = test_df[disease].values
        if ytr.sum() < 10 or yte.sum() < 10:
            print(f"  skip {disease}: too few positives")
            continue
        probe, _ = train_probe(
            Xtr, ytr, Xte, yte,
            name=f"exp01_{model_name}_{disease}",
            C_grid=PARAMS.lr_C_grid,
            n_boot=PARAMS.n_bootstrap,
            max_iter=PARAMS.lr_max_iter,
            seed=SEED, verbose=False,
        )
        records.append(dict(
            dataset=CFG.dataset, model=model_name, disease=disease,
            auc=probe.auc,
            auc_ci_low=probe.auc_ci[0], auc_ci_high=probe.auc_ci[1],
            f1=probe.f1,
            f1_ci_low=probe.f1_ci[0], f1_ci_high=probe.f1_ci[1],
            best_C=probe.best_C,
            n_train=probe.n_train, n_pos_train=probe.n_pos_train,
            n_test=probe.n_test, n_pos_test=probe.n_pos_test,
        ))
        print(f"  {disease:>12s}: AUC={probe.auc:.4f} "
              f"[{probe.auc_ci[0]:.4f}, {probe.auc_ci[1]:.4f}]  "
              f"F1={probe.f1:.3f}  best_C={probe.best_C:g}")
    ext.close()
    del ext; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# %%
results_df = pd.DataFrame(records)
_run_tag = os.environ.get("RUN_TAG", "")
_suffix = ("_" + _run_tag) if _run_tag else ""
out_path = OUT / f"exp01_{CFG.dataset}_results{_suffix}.parquet"
results_df.to_parquet(out_path, index=False)
print(f"\nSaved {len(results_df)} rows -> {out_path}")
print(results_df.pivot_table(index="model", columns="disease", values="auc"))
