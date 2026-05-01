#!/usr/bin/env bash
# =============================================================================
# scripts/run_papermill_all.sh
#
# Sequential papermill execution of every notebook in notebooks/ for the
# default DATASET (env var DATASET=nih if unset). Each executed notebook is
# written to outputs/papermill/, and stdout/stderr are tee'd to outputs/logs/.
#
# Override defaults:
#   DATASET=mimic bash scripts/run_papermill_all.sh
#   MODELS=raddino,dinov3 bash scripts/run_papermill_all.sh
#
# For multi-GPU parallel runs, see scripts/run_nih_exps_gpu0.sh / _gpu1.sh.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATASET="${DATASET:-nih}"
MODELS="${MODELS:-raddino,dinov2,dinov3,biomedclip,medsiglip}"
SEED="${SEED:-42}"
OUTPUTS_DIR="${OUTPUTS_DIR:-${REPO_ROOT}/outputs}"

mkdir -p "${OUTPUTS_DIR}/papermill" "${OUTPUTS_DIR}/logs"

NOTEBOOKS=(
  "00_BuildMimicSubsample"
  "01_DiseaseClassification"
  "02_SyntheticGeometric"
  "03_IsoMotionBlur"
  "04_CleanVsPerturbed_DiseaseClassification"
  "05_EmbeddingVisualization_UMAP"
  "06_PatchToken_Probing"
  "07_RawPixel_Baseline"
  "08_DirectionalMotionBlur"
  "09_Combined_Analysis_Stats"
  "10_MLPProbe_CLS"
  "11_ClusteredBootstrap"
  "12_DemographicSubgroups"
  "13_JPEGSensitivity"
  "14_MultiSeed"
  "15_NativeResolution"
  "16_LabelNoiseSensitivity"
  "17_ADI_vs_DownstreamDelta"
  "18_PatchFootprintTable"
  "19_SmallLesionStratifiedDelta"
  "20_DinoResNet50_Battery"
  "22_ResNet50_Oracle"
  "23_ResNet50_Baseline"
  "24_ResNet50_GlobalIsoDir"
  "25_IsoBlur_DeLong"
)

# notebook 21_VinDr_SmallNodule needs DATASET=vindr; handled separately at end

if [ "${DATASET}" = "nih" ]; then
  # Skip the MIMIC subsample build for non-MIMIC runs
  NOTEBOOKS=("${NOTEBOOKS[@]:1}")
fi

for nb in "${NOTEBOOKS[@]}"; do
  src="notebooks/${nb}.ipynb"
  dst="${OUTPUTS_DIR}/papermill/${nb}__${DATASET}.ipynb"
  log="${OUTPUTS_DIR}/logs/${nb}__${DATASET}.log"

  if [ ! -f "${src}" ]; then
    echo "[skip] missing ${src}"
    continue
  fi

  echo "=========================================================="
  echo "[papermill] ${nb}  DATASET=${DATASET}  (log: ${log})"
  echo "=========================================================="

  papermill "${src}" "${dst}" \
            -p DATASET "${DATASET}" \
            -p MODELS "${MODELS}" \
            -p SEED "${SEED}" \
            -p OUTPUTS_DIR "${OUTPUTS_DIR}" \
            -p REPO_ROOT_OVERRIDE "${REPO_ROOT}" \
            --log-output --log-level INFO \
            2>&1 | tee "${log}"
done

# VinDr-CXR confirmatory cohort (special-case DATASET)
echo "=========================================================="
echo "[papermill] 21_VinDr_SmallNodule  DATASET=vindr"
echo "=========================================================="
papermill notebooks/21_VinDr_SmallNodule.ipynb \
          "${OUTPUTS_DIR}/papermill/21_VinDr_SmallNodule__vindr.ipynb" \
          -p DATASET vindr \
          -p MODELS "raddino,dinov3" \
          -p SEED "${SEED}" \
          -p OUTPUTS_DIR "${OUTPUTS_DIR}" \
          -p REPO_ROOT_OVERRIDE "${REPO_ROOT}" \
          --log-output --log-level INFO \
          2>&1 | tee "${OUTPUTS_DIR}/logs/21_VinDr_SmallNodule__vindr.log"

echo ""
echo "All notebooks completed. Result parquets are in:"
echo "  ${OUTPUTS_DIR}/results/${DATASET}/"
echo ""
echo "Next: fill the manuscript and compile:"
echo "  python manuscript/fill_placeholders.py --results-root ${OUTPUTS_DIR}/results/"
echo "  cd manuscript && pdflatex manuscript.tex && bibtex manuscript && pdflatex manuscript.tex && pdflatex manuscript.tex"
