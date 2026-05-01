#!/usr/bin/env bash
set -u
cd /home/saptpurk/embeddings-noise-eliminators
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=1
export DATASET=mimic
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set in environment (see REPRODUCE.md section 2)}"
export MODELS_TO_RUN=dinov3,medsiglip
export REPO_ROOT=/home/saptpurk/embeddings-noise-eliminators
export V4_WORK_DIR=/home/saptpurk/embeddings-noise-eliminators/outputs
export RUN_TAG=gpu1
export NUM_WORKERS=4
# Pre-registered MIMIC-CXR subsample manifest (see build_mimic_subsample.py).
# Evaluation uses full official test+validate splits; probe-training is
# stratified random subsample of the official train split (seed=42, n=50k).
export MIMIC_SUBSAMPLE_IDS=/home/saptpurk/embeddings-noise-eliminators/manifests/mimic_subsample_ids.parquet

LOG_ROOT=/home/saptpurk/embeddings-noise-eliminators/outputs
ORDER=(
  "notebooks/01_DiseaseClassification.ipynb"
  "notebooks/02_SyntheticGeometric.ipynb"
  "notebooks/03_IsoMotionBlur.ipynb"
  "notebooks/04_CleanVsPerturbed_DiseaseClassification.ipynb"
  "notebooks/06_PatchToken_Probing.ipynb"
  "notebooks/08_DirectionalMotionBlur.ipynb"
  "notebooks/07_RawPixel_Baseline.ipynb"
  "notebooks/05_EmbeddingVisualization_UMAP.ipynb"
)
for nb in "${ORDER[@]}"; do
  name=$(basename "$nb" .ipynb)
  echo "[$(date -Is)] === mimic-gpu1 START $name ==="
  if jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 \
        "$nb" 2>&1 | tee -a "$LOG_ROOT/mimic_gpu1_${name}.log"; then
    echo "[$(date -Is)] === mimic-gpu1 OK  $name ==="
  else
    echo "[$(date -Is)] === mimic-gpu1 FAIL $name ==="
  fi
done
echo "[$(date -Is)] mimic-gpu1 ALL EXPERIMENTS FINISHED"
