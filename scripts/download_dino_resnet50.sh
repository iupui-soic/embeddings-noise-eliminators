#!/usr/bin/env bash
# Download the DINO-pretrained ResNet-50 weights (Caron et al. 2021),
# used as the matched architectural control in notebook
# 20_DinoResNet50_Battery.ipynb.
#
# Reference: Mathilde Caron et al., "Emerging Properties in Self-Supervised
# Vision Transformers", ICCV 2021.
# https://github.com/facebookresearch/dino

set -euo pipefail

DEST_DIR="${1:-outputs}"
DEST_FILE="${DEST_DIR}/dino_resnet50_pretrain.pth"
URL="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth"

mkdir -p "${DEST_DIR}"

if [ -f "${DEST_FILE}" ]; then
  size_mb=$(du -m "${DEST_FILE}" | awk '{print $1}')
  echo "[skip] ${DEST_FILE} already present (${size_mb} MB)"
  exit 0
fi

echo "[download] DINO-pretrained ResNet-50 weights (~94 MB)"
echo "  source: ${URL}"
echo "  dest:   ${DEST_FILE}"

if command -v curl >/dev/null 2>&1; then
  curl -L --fail --progress-bar -o "${DEST_FILE}" "${URL}"
elif command -v wget >/dev/null 2>&1; then
  wget --show-progress -O "${DEST_FILE}" "${URL}"
else
  echo "Error: neither curl nor wget is available" >&2
  exit 1
fi

size_mb=$(du -m "${DEST_FILE}" | awk '{print $1}')
echo "[done]  wrote ${DEST_FILE} (${size_mb} MB)"
