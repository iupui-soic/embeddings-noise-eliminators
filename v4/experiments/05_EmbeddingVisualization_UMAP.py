# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Experiment 05 — Embedding-space Visualisation (UMAP + PCA + Silhouette)
#
# **Purpose.**  RYAI Reviewer 3 (#9) and Meta-Rad Reviewer 2 (#3) both ask
# whether clean and perturbed embeddings *separate* in feature space.
# This experiment produces:
#
#  * UMAP projection of clean vs perturbed CLS embeddings per model
#  * Silhouette score quantifying separation
#  * Inter-cluster Mahalanobis distance (dimension-normalised)
#  * PCA top-2 as a simple baseline visual
#
# **Inputs.**  Uses embeddings produced by exp04 (clean + perturbed CLS).
#
# **How to run.**
#     export DATASET=nih                 # or 'emory'
#     jupyter nbconvert --execute --to notebook 05_EmbeddingVisualization_UMAP.ipynb
#
# Output: `exp05_<dataset>_embedding_separation.parquet` + PNGs.

# %%
import os, sys, gc, json
from pathlib import Path
REPO_ROOT = Path(os.environ.get("REPO_ROOT", "/home/saptpurk/embeddings-noise-eliminators/v4"))
sys.path.insert(0, str(REPO_ROOT))

from common import get_config, PARAMS, MODELS

CFG = get_config()
EXP04_DIR = CFG.output_dir("exp04_clean_vs_perturbed")
OUT       = CFG.output_dir("exp05_embedding_viz")
print(f"Dataset: {CFG.name}  |  Reading from: {EXP04_DIR}  |  Writing: {OUT}")

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("umap-learn not installed; falling back to PCA only.")
    print("    pip install umap-learn")

from common import models_to_run
MODEL_NAMES   = models_to_run()
print(f"Running models: {MODEL_NAMES}")
PERTURBATIONS = [
    "iso_blur_p4", "iso_blur_p8",
    "dir_blur_cranio_p32", "dir_blur_cranio_p64",
    "dir_blur_lateral_p32", "dir_blur_lateral_p64",
    "reticular_fine_p32", "reticular_coarse_p64",
    "ground_glass_p64",
]

# %%
def mahalanobis_between(A, B):
    """
    Robust inter-cluster Mahalanobis distance using pooled covariance shrinkage.
    """
    mu_a, mu_b = A.mean(0), B.mean(0)
    pooled = 0.5 * (np.cov(A.T) + np.cov(B.T))
    # Shrinkage to avoid singularity in high-D
    trace = np.trace(pooled) / pooled.shape[0]
    pooled = 0.9 * pooled + 0.1 * trace * np.eye(pooled.shape[0])
    try:
        inv = np.linalg.pinv(pooled)
        d = (mu_a - mu_b) @ inv @ (mu_a - mu_b).T
        return float(np.sqrt(max(d, 0.0)))
    except np.linalg.LinAlgError:
        return float(np.linalg.norm(mu_a - mu_b))


def silhouette_binary(X, y, subsample=5000, seed=42):
    """Binary silhouette on a subsample (full silhouette is O(n^2) memory)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    if n > subsample:
        idx = rng.choice(n, size=subsample, replace=False)
        X, y = X[idx], y[idx]
    try:
        return float(silhouette_score(X, y, metric="cosine"))
    except Exception:
        return float("nan")


# %%
records = []
for model_name in MODEL_NAMES:
    clean_path = EXP04_DIR / f"emb_clean_{model_name}_test.npy"
    if not clean_path.exists():
        print(f"Skipping {model_name}: exp04 outputs not found.")
        continue
    X_clean = np.load(clean_path)

    for pert in PERTURBATIONS:
        pert_path = EXP04_DIR / f"emb_pert_{model_name}_{pert}_test.npy"
        if not pert_path.exists():
            continue
        X_pert = np.load(pert_path)

        # Stack with labels
        X = np.vstack([X_clean, X_pert]).astype(np.float32)
        y = np.concatenate([np.zeros(len(X_clean)), np.ones(len(X_pert))]).astype(int)

        # Metrics
        sil = silhouette_binary(X, y)
        maha = mahalanobis_between(X_clean, X_pert)

        # PCA 2-D
        pca = PCA(n_components=2, random_state=PARAMS.random_seed)
        Z_pca = pca.fit_transform(X)

        # UMAP 2-D (sub-sample for speed)
        if HAS_UMAP:
            rng = np.random.default_rng(PARAMS.random_seed)
            max_plot = min(10000, len(X))
            idx = rng.choice(len(X), size=max_plot, replace=False)
            reducer = umap.UMAP(n_components=2, n_neighbors=30,
                                min_dist=0.1, metric="cosine",
                                random_state=PARAMS.random_seed)
            Z_umap = reducer.fit_transform(X[idx])
            y_plot = y[idx]
        else:
            Z_umap, y_plot = None, None

        # Plot
        fig, axes = plt.subplots(1, 2 if HAS_UMAP else 1,
                                 figsize=(10 if HAS_UMAP else 5, 4))
        if not HAS_UMAP: axes = [axes]
        axes[0].scatter(Z_pca[y == 0, 0], Z_pca[y == 0, 1],
                        s=3, alpha=0.5, label="clean")
        axes[0].scatter(Z_pca[y == 1, 0], Z_pca[y == 1, 1],
                        s=3, alpha=0.5, label="perturbed", c="crimson")
        axes[0].set_title(f"PCA | {model_name} | {pert}")
        axes[0].legend()
        if HAS_UMAP:
            axes[1].scatter(Z_umap[y_plot == 0, 0], Z_umap[y_plot == 0, 1],
                            s=3, alpha=0.5, label="clean")
            axes[1].scatter(Z_umap[y_plot == 1, 0], Z_umap[y_plot == 1, 1],
                            s=3, alpha=0.5, label="perturbed", c="crimson")
            axes[1].set_title(f"UMAP | silhouette={sil:.3f}")
            axes[1].legend()
        plt.tight_layout()
        png = OUT / f"viz_{model_name}_{pert}.png"
        plt.savefig(png, dpi=180, bbox_inches="tight")
        plt.close()

        records.append(dict(
            dataset=CFG.dataset, model=model_name, perturbation=pert,
            silhouette=sil, mahalanobis=maha,
            n_clean=int(len(X_clean)), n_perturbed=int(len(X_pert)),
            plot_path=str(png.relative_to(OUT)),
        ))
        print(f"{model_name:>8s} {pert:>22s}  silhouette={sil:+.3f}  mahalanobis={maha:.3f}")

# %%
df = pd.DataFrame(records)
_run_tag = os.environ.get("RUN_TAG", "")
_suffix = ("_" + _run_tag) if _run_tag else ""
out_path = OUT / f"exp05_{CFG.dataset}_embedding_separation{_suffix}.parquet"
df.to_parquet(out_path, index=False)
print(f"\nSaved -> {out_path}")
df
