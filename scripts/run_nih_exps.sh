#!/usr/bin/env bash
# Run the v4 experiment pipeline on NIH-CXR14 for every non-gated model.
# DINOv3 is deferred until HF access is granted.
set -u

cd /home/saptpurk/embeddings-noise-eliminators
source .venv/bin/activate

export DATASET=nih
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set in environment (see REPRODUCE.md section 2)}"
export MODELS_TO_RUN=raddino,dinov2,biomedclip,dinov3,medsiglip
export REPO_ROOT=/home/saptpurk/embeddings-noise-eliminators
export V4_WORK_DIR=/home/saptpurk/embeddings-noise-eliminators/outputs

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
  echo "[$(date -Is)] === START $name ==="
  if jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 \
        "$nb" 2>&1 | tee -a "$LOG_ROOT/${name}.log"; then
    echo "[$(date -Is)] === OK  $name ==="
  else
    echo "[$(date -Is)] === FAIL $name ==="
    echo "CONTINUE_ON_FAIL=1 so the next experiment still runs" | tee -a "$LOG_ROOT/${name}.log"
  fi
done

echo "[$(date -Is)] ALL EXPERIMENTS FINISHED"
