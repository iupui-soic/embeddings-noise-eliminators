#!/usr/bin/env bash
# GPU 0 subset: the three faster models.
set -u
cd /home/saptpurk/embeddings-noise-eliminators
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export DATASET=nih
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set in environment (see REPRODUCE.md section 2)}"
export MODELS_TO_RUN=raddino,dinov2,biomedclip
export REPO_ROOT=/home/saptpurk/embeddings-noise-eliminators
export V4_WORK_DIR=/home/saptpurk/embeddings-noise-eliminators/outputs
export RUN_TAG=gpu0
export NUM_WORKERS=4

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
  echo "[$(date -Is)] === gpu0 START $name ==="
  if jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 \
        "$nb" 2>&1 | tee -a "$LOG_ROOT/gpu0_${name}.log"; then
    echo "[$(date -Is)] === gpu0 OK  $name ==="
  else
    echo "[$(date -Is)] === gpu0 FAIL $name ==="
  fi
done
echo "[$(date -Is)] gpu0 ALL EXPERIMENTS FINISHED"
