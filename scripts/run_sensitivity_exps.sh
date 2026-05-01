#!/usr/bin/env bash
# Runs MedIA-preempting sensitivity experiments after the main NIH+MIMIC+Emory
# pipelines (exp01-exp08) complete. Each notebook below adds one defense:
#
#   10 MLPProbe_CLS               nonlinear probe sanity check (tier-1)
#   11 ClusteredBootstrap         patient-level bootstrap for exp04 (tier-1)
#   12 DemographicSubgroups       sex/age AUC stratification (tier-1)
#   13 JPEGSensitivity            NIH-only JPEG quality sweep (tier-1)
#   14 MultiSeed                  n=3 seed variability for headline cells (tier-1)
#   15 NativeResolution           resolution ablation (descoped: RAD-DINO vs DINOv3, NIH only)
#   16 LabelNoiseSensitivity      high-confidence MIMIC subset (tier-2)
#   17 ADI_vs_DownstreamDelta     paper-only: ADI predicts exp04 ΔAUC (tier-1)
#   18 PatchFootprintTable        paper-only: artifact-to-token ratio table (tier-2)
#   19 SmallLesionStratifiedDelta paper-only: NIH bbox subset, ΔAUC by lesion area (tier-1)
#
# Paper-only notebooks (11,12,17,18,19) can run on CPU / any machine.
# Embedding-producing notebooks (13,15) need a GPU; 10/14 reuse cached embeddings
# and need only a GPU for probe fitting (lightweight).
set -u
cd /home/saptpurk/embeddings-noise-eliminators
source .venv/bin/activate

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set in environment (see REPRODUCE.md section 2)}"
export REPO_ROOT=/home/saptpurk/embeddings-noise-eliminators/v4
export V4_WORK_DIR=/home/saptpurk/embeddings-noise-eliminators/v4_work
export MIMIC_SUBSAMPLE_IDS=/home/saptpurk/embeddings-noise-eliminators/manuscript/mimic_subsample_ids.parquet
export NUM_WORKERS=4

cd v4
LOG_ROOT=/home/saptpurk/embeddings-noise-eliminators/v4_work

ORDER=(
  "notebooks/11_ClusteredBootstrap.ipynb"
  "notebooks/17_ADI_vs_DownstreamDelta.ipynb"
  "notebooks/18_PatchFootprintTable.ipynb"
  "notebooks/12_DemographicSubgroups.ipynb"
  "notebooks/16_LabelNoiseSensitivity.ipynb"
  "notebooks/10_MLPProbe_CLS.ipynb"
  "notebooks/14_MultiSeed.ipynb"
  "notebooks/13_JPEGSensitivity.ipynb"
  "notebooks/15_NativeResolution.ipynb"
  "notebooks/19_SmallLesionStratifiedDelta.ipynb"
)

for nb in "${ORDER[@]}"; do
  name=$(basename "$nb" .ipynb)
  echo "[$(date -Is)] === sensitivity START $name ==="
  if jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 \
        "$nb" 2>&1 | tee -a "$LOG_ROOT/sensitivity_${name}.log"; then
    echo "[$(date -Is)] === sensitivity OK  $name ==="
  else
    echo "[$(date -Is)] === sensitivity FAIL $name ==="
  fi
done
echo "[$(date -Is)] sensitivity ALL EXPERIMENTS FINISHED"
