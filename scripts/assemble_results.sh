#!/usr/bin/env bash
# Assemble per-dataset experiment parquets into the layout expected by
# manuscript/fill_placeholders.py:
#
#   results/<dataset>/<exp_dir>/<parquet>
#
# Source paths live in V4_WORK_DIR/v4_<exp_id>_<dataset>/<parquet>
# Dataset must be passed as $1 (nih|mimic|emory).
set -eu
DS="${1:-nih}"
WORK="/home/saptpurk/embeddings-noise-eliminators/v4_work"
OUT="/home/saptpurk/embeddings-noise-eliminators/v4_work/results/${DS}"

mkdir -p "$OUT"

for exp in \
  "exp01_disease_classification:exp01_${DS}_results.parquet" \
  "exp02_geometric:exp02_${DS}_results.parquet" \
  "exp03_iso_motion_blur:exp03_${DS}_results.parquet" \
  "exp04_clean_vs_perturbed:exp04_${DS}_results.parquet" \
  "exp05_embedding_viz:exp05_${DS}_embedding_separation.parquet" \
  "exp06_patch_probing:exp06_${DS}_patch_probing.parquet" \
  "exp07_rawpixel_baseline:exp07_${DS}_rawpixel_baseline.parquet" \
  "exp08_directional_blur:exp08_${DS}_directional_blur.parquet"; do
  expdir="${exp%%:*}"
  fname="${exp##*:}"
  src="$WORK/v4_${expdir}_${DS}/${fname}"
  dst_dir="$OUT/${expdir}"
  mkdir -p "$dst_dir"
  if [ -f "$src" ]; then
    cp "$src" "$dst_dir/${fname}"
    echo "copied $fname"
  else
    echo "(missing) $src"
  fi
done

# exp19 is NIH-only by design (requires bbox annotations).
if [ "$DS" = "nih" ]; then
  exp19_src="$WORK/v4_exp19_small_lesion_strata"
  exp19_dst="$OUT/exp19_small_lesion_strata"
  if [ -d "$exp19_src" ]; then
    mkdir -p "$exp19_dst"
    for f in "$exp19_src"/exp19_smalllesion_*.parquet; do
      [ -f "$f" ] || continue
      cp "$f" "$exp19_dst/$(basename "$f")"
      echo "copied $(basename "$f")"
    done
  fi
fi
echo "done -> $OUT"
