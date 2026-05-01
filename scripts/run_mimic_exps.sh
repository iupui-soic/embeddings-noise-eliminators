#!/usr/bin/env bash
# Run the v4 experiment pipeline on MIMIC-CXR-JPG for every non-gated model + DINOv3.
# Intended to run AFTER the NIH pipeline finishes and AFTER the MIMIC S3 sync completes.
set -u

cd /home/saptpurk/embeddings-noise-eliminators
source .venv/bin/activate

export DATASET=mimic
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set in environment (see REPRODUCE.md section 2)}"
export MODELS_TO_RUN=raddino,dinov2,biomedclip,dinov3,medsiglip
export REPO_ROOT=/home/saptpurk/embeddings-noise-eliminators/v4
export V4_WORK_DIR=/home/saptpurk/embeddings-noise-eliminators/v4_work

cd v4
LOG_ROOT=/home/saptpurk/embeddings-noise-eliminators/v4_work

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
  echo "[$(date -Is)] === START mimic $name ==="
  if jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 \
        "$nb" 2>&1 | tee -a "$LOG_ROOT/mimic_${name}.log"; then
    echo "[$(date -Is)] === OK  mimic $name ==="
  else
    echo "[$(date -Is)] === FAIL mimic $name ==="
    echo "CONTINUE_ON_FAIL=1 so the next experiment still runs" | tee -a "$LOG_ROOT/mimic_${name}.log"
  fi
done

echo "[$(date -Is)] ALL MIMIC EXPERIMENTS FINISHED"
