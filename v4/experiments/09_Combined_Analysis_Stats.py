# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Experiment 09 — Combined Statistical Analysis Across Both Servers
#
# **Purpose.**  Run DeLong, permutation, and BH-FDR tests on the full
# comparison matrix (addresses RYAI R3 #8 and Meta-Rad R1 #3).
#
# This notebook is designed to run on a **local analysis machine** after
# you have pulled the parquet outputs from the two servers.  The NIH
# and Emory servers produce identical directory layouts; copy them both
# under a shared `./results/` folder:
#
#     results/
#       nih/
#         exp04_clean_vs_perturbed/exp04_nih_results.parquet
#         exp05_embedding_viz/exp05_nih_embedding_separation.parquet
#         exp06_patch_probing/exp06_nih_patch_probing.parquet
#         exp07_rawpixel_baseline/exp07_nih_rawpixel_baseline.parquet
#         exp08_directional_blur/exp08_nih_directional_blur.parquet
#       emory/
#         ... same structure ...
#
# Also copy the per-disease-classifier `*.json` files from exp04 — they
# contain `y_true` and `y_proba` arrays needed for DeLong tests.
#
# **How to run.**
#     export RESULTS_ROOT=/path/to/results    # see layout above
#     jupyter nbconvert --execute --to notebook 09_Combined_Analysis_Stats.ipynb

# %%
import json, os, sys
from pathlib import Path

RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", "./results")).resolve()
print(f"Reading results from: {RESULTS_ROOT}")

# Only need local utilities, not the full server-side 'common' package.
REPO_ROOT = Path(os.environ.get("REPO_ROOT", "/home/saptpurk/embeddings-noise-eliminators/v4"))
sys.path.insert(0, str(REPO_ROOT))
from common.stats import (
    delong_test, permutation_auc_test,
    benjamini_hochberg, paired_bootstrap_delta_auc,
)
from common.probing import load_probe_result

# %%
import numpy as np
import pandas as pd

OUT = Path("./results/combined_analysis")
OUT.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Aggregate exp04 ΔAUC tables across datasets

# %%
exp04_frames = []
for ds in ("nih", "emory"):
    p = RESULTS_ROOT / ds / "exp04_clean_vs_perturbed" / f"exp04_{ds}_results.parquet"
    if p.exists():
        exp04_frames.append(pd.read_parquet(p))
    else:
        print(f"  missing: {p}")
exp04 = pd.concat(exp04_frames, ignore_index=True) if exp04_frames else pd.DataFrame()
print(f"exp04 rows: {len(exp04)}")
exp04.to_csv(OUT / "exp04_combined.csv", index=False)
exp04.head()

# %% [markdown]
# ## 2. DeLong tests on model-to-model CLS-probe AUCs (exp06) with BH-FDR
#
# For each (dataset, perturbation, pooling mode), compare RAD-DINO vs DINOv3
# on the SAME test set.  This requires the per-sample y_proba and y_true,
# which we stored as JSON in exp06.  (If you didn't save the JSONs, you can
# still run DeLong on exp04 disease classifications.)

# %%
def _load_exp06_result(ds, model, pert, pooling):
    # Expected filename (experiment 06 writes them via save_probe)
    p = (RESULTS_ROOT / ds / "exp06_patch_probing"
         / f"{model}_{pert}_{pooling}_result.json")
    if p.exists():
        return load_probe_result(p)
    return None


pairs = []
for ds in ("nih", "emory"):
    exp06_path = RESULTS_ROOT / ds / "exp06_patch_probing" / f"exp06_{ds}_patch_probing.parquet"
    if not exp06_path.exists():
        print(f"missing: {exp06_path}")
        continue
    summary = pd.read_parquet(exp06_path)
    perts = summary["perturbation"].unique()
    pools = summary["pooling"].unique()

    for pert in perts:
        for pool in pools:
            rr_rad = _load_exp06_result(ds, "raddino", pert, pool)
            rr_d3  = _load_exp06_result(ds, "dinov3",  pert, pool)
            if rr_rad is None or rr_d3 is None:
                continue
            # DeLong requires identical y_true
            if not np.array_equal(rr_rad.y_true, rr_d3.y_true):
                print(f"mismatch y_true for {ds}/{pert}/{pool}")
                continue
            dl = delong_test(rr_rad.y_true, rr_rad.y_proba, rr_d3.y_proba)
            pm = permutation_auc_test(rr_rad.y_true, rr_rad.y_proba, rr_d3.y_proba,
                                      n_permutations=5000)
            pairs.append(dict(
                dataset=ds, perturbation=pert, pooling=pool,
                auc_raddino=dl["auc_a"], auc_dinov3=dl["auc_b"],
                delta=dl["delta"], delong_z=dl["z"],
                delong_p=dl["p_value"], perm_p=pm["p_value"],
                ci_low=dl["ci_low"], ci_high=dl["ci_high"],
            ))

pairs_df = pd.DataFrame(pairs)
if len(pairs_df):
    bh = benjamini_hochberg(pairs_df["delong_p"].values, alpha=0.05)
    pairs_df["delong_p_bh"] = bh["p_adjusted"]
    pairs_df["delong_rejected"] = bh["rejected"]
    pairs_df.to_csv(OUT / "raddino_vs_dinov3_delong.csv", index=False)
    print(f"Pairwise comparisons (RAD-DINO vs DINOv3): {len(pairs_df)}")
else:
    print("No exp06 result JSONs found — paired DeLong skipped.")
pairs_df

# %% [markdown]
# ## 3. Clean-vs-perturbed disease ΔAUC significance (exp04)
#
# DeLong on the same disease classifier's predictions on clean vs perturbed
# test sets.  Because these share y_true, DeLong applies directly.

# %%
def _load_exp04_clean(ds, model, disease):
    p = (RESULTS_ROOT / ds / "exp04_clean_vs_perturbed"
         / f"disease_{model}_{disease}_clean_result.json")
    return load_probe_result(p) if p.exists() else None

# Full DeLong per (dataset, model, disease, perturbation):
disease_rows = []
if len(exp04):
    for _, row in exp04.iterrows():
        rr_clean = _load_exp04_clean(row["dataset"], row["model"], row["disease"])
        if rr_clean is None:
            continue
        # Re-run the perturbed predictions from the saved classifier would
        # require the perturbed embeddings — simpler path: we already have
        # `delong_p` from exp04 itself.  So just collect those:
        disease_rows.append(dict(
            dataset=row["dataset"], model=row["model"], disease=row["disease"],
            perturbation=row["perturbation"],
            delta_auc=row["delta_auc"], delong_p=row["delong_p"],
        ))
    disease_df = pd.DataFrame(disease_rows)
    if len(disease_df):
        bh = benjamini_hochberg(disease_df["delong_p"].values, alpha=0.05)
        disease_df["delong_p_bh"] = bh["p_adjusted"]
        disease_df["rejected"] = bh["rejected"]
        disease_df.to_csv(OUT / "clean_vs_perturbed_disease_bh.csv", index=False)
        print(f"Disease ΔAUC comparisons: {len(disease_df)}  "
              f"rejected at BH-FDR 0.05: {int(disease_df['rejected'].sum())}")
        disease_df.head(10)

# %% [markdown]
# ## 4. Patch-dilution summary
#
# If `patch_local` AUC >> `cls` AUC for a perturbation, that directly
# supports the patch-dilution hypothesis for that perturbation.

# %%
pool_frames = []
for ds in ("nih", "emory"):
    p = RESULTS_ROOT / ds / "exp06_patch_probing" / f"exp06_{ds}_patch_probing.parquet"
    if p.exists(): pool_frames.append(pd.read_parquet(p))
pool_df = pd.concat(pool_frames, ignore_index=True) if pool_frames else pd.DataFrame()

if len(pool_df):
    wide = pool_df.pivot_table(
        index=["dataset", "model", "perturbation"],
        columns="pooling", values="auc",
    ).reset_index()
    wide["local_minus_cls"] = wide.get("patch_local", 0) - wide.get("cls", 0)
    wide.to_csv(OUT / "patch_dilution_summary.csv", index=False)
    print("\nPatch-dilution evidence (patch_local - cls AUC, sorted):")
    print(wide.sort_values("local_minus_cls", ascending=False).head(20))

# %% [markdown]
# ## 5. Compose a single LaTeX-ready summary table

# %%
def fmt_ci(v, lo, hi, digits=3):
    return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"

summary_rows = []
for _, row in exp04.iterrows():
    summary_rows.append(dict(
        Dataset=row["dataset"].upper(),
        Model=row["model"],
        Disease=row["disease"],
        Perturbation=row["perturbation"],
        **{"AUC clean":     f"{row['auc_clean']:.3f}",
           "AUC perturbed": f"{row['auc_perturbed']:.3f}",
           "ΔAUC [95% CI]": fmt_ci(row["delta_auc"],
                                   row["delta_ci_low"], row["delta_ci_high"]),
           "DeLong p":      f"{row['delong_p']:.3g}"},
    ))

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "paper_table_main.csv", index=False)
print("\nMain manuscript table preview:")
print(summary.head(20).to_string(index=False))
