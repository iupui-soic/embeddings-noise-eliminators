# Reproduction guide

This document is the end-to-end runbook for reproducing every number, table,
and figure in the manuscript. It assumes a fresh `git clone` of this
repository.

If anything in this document doesn't work as written, please open an issue.

## 1. Hardware and OS

The full pipeline was developed and tested on:

| Component | Tested configuration | Notes |
|---|---|---|
| OS | Ubuntu 22.04 LTS | any Linux with CUDA support; macOS works for analysis-only |
| Python | 3.10 | 3.9–3.12 should work |
| GPU | NVIDIA RTX 6000 Ada (48 GB) | DINOv3 ViT-7B requires ≥40 GB VRAM at fp16; smaller models fit on ≥16 GB |
| RAM | 64 GB | 32 GB is enough for analysis-only; embedding extraction is memory-light per image |
| Disk | 500 GB free | NIH ≈ 45 GB, MIMIC ≈ 470 GB, VinDr ≈ 80 GB; outputs ≈ 30 GB |

A multi-GPU machine cuts wall-time roughly in half via the `scripts/run_*_gpu0.sh`
/ `_gpu1.sh` pair. Single-GPU runs use the unified `scripts/run_*.sh`.

## 2. Environment setup

```bash
git clone https://github.com/iupui-soic/embeddings-noise-eliminators.git
cd embeddings-noise-eliminators

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Optional: verify the env
python -c "import common; from common import MODELS; print(list(MODELS))"
```

Expected output:

```
['raddino', 'dinov3', 'dinov3_vits', 'dinov2', 'biomedclip', 'medsiglip']
```

For HuggingFace gated models (DINOv3, MedSigLIP), set `HF_TOKEN`:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

## 3. Datasets

Download each dataset to a path of your choice and edit `common/config.py` to
point to the actual locations. The defaults assume `/data0/`.

### NIH-CXR14 (public)

```bash
# Download from Kaggle: https://www.kaggle.com/datasets/nih-chest-xrays/data
# Expected layout:
#   /data0/NIH-CXR14/images/00000001_000.png …
#   /data0/NIH-CXR14/Data_Entry_2017_v2020.csv
```

### MIMIC-CXR-JPG v2.0.0 (PhysioNet credentialed)

Requires a PhysioNet account with the MIMIC data-use agreement signed.

```bash
# https://physionet.org/content/mimic-cxr-jpg/2.0.0/
# Expected layout (s3 sync recommended):
#   /data0/MIMIC-CXR/files/p10/p10000032/s50414267/02aa804e-...jpg
#   /data0/MIMIC-CXR/mimic-cxr-2.0.0-metadata.csv.gz
#   /data0/MIMIC-CXR/mimic-cxr-2.0.0-chexpert.csv.gz
#   /data0/MIMIC-CXR/mimic-cxr-2.0.0-split.csv.gz
```

The pre-registered probe-training subsample manifest is committed in
`manifests/mimic_subsample_ids.parquet`. Either use it directly or rebuild it:

```bash
papermill notebooks/00_BuildMimicSubsample.ipynb \
          outputs/papermill/00_BuildMimicSubsample.ipynb \
          -p MIMIC_ROOT /data0/MIMIC-CXR \
          -p OUT_PATH manifests/mimic_subsample_ids.parquet
```

### VinDr-CXR (PhysioNet credentialed)

```bash
# https://physionet.org/content/vindr-cxr/1.0.0/
# Expected layout:
#   /data0/VinDr-CXR/train/<study_id>.dicom
#   /data0/VinDr-CXR/test/<study_id>.dicom
#   /data0/VinDr-CXR/annotations/image_labels_train.csv
#   /data0/VinDr-CXR/annotations/annotations_test.csv
```

### Emory CXR (institutional, restricted)

Emory CXR is institutional. Reproduction of the Emory-specific rows requires
access to the on-premises PHI-compatible Emory infrastructure under the IRB
protocol cited in the manuscript. The remainder of the pipeline runs end-to-end
on NIH + MIMIC + VinDr alone.

## 4. Auxiliary model weights

The frozen DINO-pretrained ResNet-50 (Caron et al. 2021) — the matched
architectural control in the manuscript — is fetched from the official
Facebook AI release:

```bash
bash scripts/download_dino_resnet50.sh
# downloads outputs/dino_resnet50_pretrain.pth (~94 MB)
```

ViT foundation-model weights (RAD-DINO, DINOv2, DINOv3, BiomedCLIP, MedSigLIP)
are pulled lazily by HuggingFace on first use; ensure `HF_TOKEN` is set for the
gated DINOv3 and MedSigLIP repositories.

## 5. Smoke test (~30 min, single GPU)

Before kicking off the full pipeline, validate end-to-end integrity on a
200-image subset:

```bash
python scripts/smoke_test.py
```

Expected output:

- `outputs/smoke/exp01_smoke_results.parquet` — non-empty
- `outputs/smoke/exp03_smoke_results.parquet` — non-empty
- a "smoke test PASSED" line in stdout

If this fails, fix it before scaling up. Check `outputs/smoke/*.log`.

## 6. Full pipeline

The canonical run order. **Each step is a papermill-parameterized notebook**;
substitute `DATASET=mimic` or `DATASET=emory` for the per-dataset reruns.
Approximate wall-times are for a single RTX 6000 Ada at the released splits.

| Step | Notebook | NIH wall-time | Output |
|---|---|---|---|
| 0 | `00_BuildMimicSubsample` | 5 min | `manifests/mimic_subsample_ids.parquet` |
| 1 | `01_DiseaseClassification` | 30 min | `outputs/results/<DS>/exp01_disease_classification/exp01_<DS>_results.parquet` |
| 2 | `02_SyntheticGeometric` | 90 min | `outputs/results/<DS>/exp02_geometric/exp02_<DS>_results.parquet` |
| 3 | `03_IsoMotionBlur` | 25 min | `…/exp03_iso_motion_blur/exp03_<DS>_results.parquet` |
| 4 | `04_CleanVsPerturbed_DiseaseClassification` | 60 min | `…/exp04_clean_vs_perturbed/exp04_<DS>_results.parquet` |
| 5 | `05_EmbeddingVisualization_UMAP` | 35 min | `…/exp05_embedding_viz/*.png + .parquet` |
| 6 | `06_PatchToken_Probing` | 70 min | `…/exp06_patch_probing/exp06_<DS>_patch_probing.parquet` |
| 7 | `07_RawPixel_Baseline` | 15 min | `…/exp07_rawpixel_baseline/exp07_<DS>_rawpixel_baseline.parquet` |
| 8 | `08_DirectionalMotionBlur` | 8 hours | `…/exp08_directional_blur/exp08_<DS>_directional_blur.parquet` |
| 9 | `09_Combined_Analysis_Stats` | 5 min | DeLong + BH-FDR aggregates |
| 10–18 | sensitivity analyses (lightweight) | 15 min each | `outputs/v4_exp{10..18}_*` |
| 19 | `19_SmallLesionStratifiedDelta` | 20 min | NIH BBox stratified ΔAUC |
| 20 | `20_DinoResNet50_Battery` | 90 min | DINO-ResNet-50 architectural control rows |
| 21 | `21_VinDr_SmallNodule` (DATASET=vindr) | 60 min | VinDr stratified replication |
| 22 | `22_ResNet50_Oracle` | 30 min | 32×32 oracle pixel positive control |
| 23 | `23_ResNet50_Baseline` | 25 min | whole-image ResNet-50 reticular + ground-glass |
| 24 | `24_ResNet50_GlobalIsoDir` | 30 min | whole-image ResNet-50 iso + directional |
| 25 | `25_IsoBlur_DeLong` | 5 min | paired DeLong RAD-DINO vs DINOv3 |

Three execution patterns:

### A. Single-GPU sequential (simplest)

```bash
bash scripts/run_papermill_all.sh                  # all 26 notebooks, default DATASET=nih
DATASET=mimic bash scripts/run_papermill_all.sh    # then MIMIC
# Emory rerun is on-prem only.
```

### B. Multi-GPU parallel (NIH on two GPUs, then MIMIC)

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_nih_exps_gpu0.sh &
CUDA_VISIBLE_DEVICES=1 bash scripts/run_nih_exps_gpu1.sh &
wait
CUDA_VISIBLE_DEVICES=0 bash scripts/run_mimic_exps_gpu0.sh &
CUDA_VISIBLE_DEVICES=1 bash scripts/run_mimic_exps_gpu1.sh &
wait
bash scripts/merge_gpu_parquets.sh                 # consolidate per-GPU shards
bash scripts/run_sensitivity_exps.sh               # notebooks 10–18
```

### C. One notebook at a time (debugging / partial reruns)

```bash
papermill notebooks/06_PatchToken_Probing.ipynb \
          outputs/papermill/06_PatchToken_Probing_nih.ipynb \
          -p DATASET nih \
          -p MODELS "raddino,dinov3"
```

## 7. Fill the manuscript

Once the parquet results are in `outputs/results/{nih,mimic,emory}/exp01–08/`:

```bash
python manuscript/fill_placeholders.py \
       --results-root outputs/results/ \
       --main         manuscript/manuscript.tex \
       --supp         manuscript/supplementary.tex \
       --main-out     manuscript/manuscript_filled.tex \
       --supp-out     manuscript/supplementary_filled.tex
```

The manuscript ships with all numbers already populated, so this step is
relevant only when reproducing from a fresh data run.

Then compile:

```bash
cd manuscript
pdflatex manuscript.tex
bibtex   manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex

pdflatex supplementary.tex
bibtex   supplementary
pdflatex supplementary.tex
pdflatex supplementary.tex
```

## 8. Expected outputs

After a complete run on NIH + MIMIC + VinDr:

```
outputs/
├── results/
│   ├── nih/
│   │   ├── exp01_disease_classification/exp01_nih_results.parquet
│   │   ├── exp02_geometric/exp02_nih_results.parquet
│   │   ├── exp03_iso_motion_blur/exp03_nih_results.parquet
│   │   ├── exp04_clean_vs_perturbed/exp04_nih_results.parquet
│   │   ├── exp05_embedding_viz/exp05_nih_embedding_separation.parquet
│   │   ├── exp05_embedding_viz/viz_*.png
│   │   ├── exp06_patch_probing/exp06_nih_patch_probing.parquet
│   │   ├── exp07_rawpixel_baseline/exp07_nih_rawpixel_baseline.parquet
│   │   └── exp08_directional_blur/exp08_nih_directional_blur.parquet
│   ├── mimic/                       (same layout as nih/)
│   └── emory/                       (same layout, on-prem only)
├── v4_exp10_mlp_probe/              (sensitivity)
├── v4_exp11_clustered_bootstrap/
├── v4_exp12_demographic_subgroups/
├── v4_exp13_jpeg_sensitivity/
├── v4_exp14_multiseed/
├── v4_exp15_native_resolution/
├── v4_exp16_label_noise/
├── v4_exp17_adi_vs_delta/
├── v4_exp18_patch_footprint/
├── v4_exp19_small_lesion_strata/    (NIH BBox stratified ΔAUC)
├── v4_exp_vindr_smallnodule/        (VinDr confirmatory)
├── papermill/                       (executed notebook copies)
└── logs/                            (per-notebook stdout/stderr)
```

## 9. Manuscript-table mapping

Each manuscript table is built from these specific parquet files:

| Manuscript element | Source parquet(s) |
|---|---|
| Table 1 (disease AUC) | `outputs/results/{nih,mimic,emory}/exp01_disease_classification/*.parquet` |
| Table 2 (synthetic geometric) | `…/exp02_geometric/*.parquet` |
| Table 3 (iso-blur) | `…/exp03_iso_motion_blur/*.parquet` + `outputs/v4_iso_delong/*.parquet` |
| Table 4 (directional blur) | `…/exp08_directional_blur/*.parquet` |
| Table 5 (pathology patterns) | `…/exp06_patch_probing/*.parquet` (CLS rows for reticular/ground-glass) |
| Table 6 (three-pooling) | `…/exp06_patch_probing/*.parquet` |
| Table 7 (raw-pixel baselines) | `…/exp07_rawpixel_baseline/*.parquet` |
| Table 9 (clean-vs-perturbed) | `…/exp04_clean_vs_perturbed/*.parquet` |
| Table 11 (NIH BBox stratified) | `outputs/v4_exp19_small_lesion_strata/*.parquet` |
| Table 12 (VinDr stratified) | `outputs/v4_exp_vindr_smallnodule/*.parquet` |
| Table 13 (SFDI) | derived in `09_Combined_Analysis_Stats.ipynb` from exp06 CLS rows |

`fill_placeholders.py` automates this mapping; the manuscript is the single
source of truth for which token reads from which parquet column.

## 10. Common gotchas

- **MIMIC subsample is mandatory** for the manuscript splits. Running MIMIC
  experiments without setting `MIMIC_SUBSAMPLE_IDS` will train on the full 240k
  rows and produce different (but still scientifically valid) numbers.
- **DINOv3 and MedSigLIP are gated** on HuggingFace. Without a valid `HF_TOKEN`
  these models will be silently skipped and the corresponding rows will be
  empty — leading to mismatched-shape errors in `09_Combined_Analysis_Stats`.
- **Determinism:** the SHA-256 placement is deterministic across reruns *given
  the same image filenames*, but if you rename or re-export images, placements
  change. Always inject perturbations on the original released filenames.
- **Patch sizes:** RAD-DINO and DINOv2-B/14 use patch 14; DINOv3 ViT-7B,
  BiomedCLIP, and MedSigLIP use patch 16. The patch-local probe automatically
  uses each model's native patch size from `MODELS[…]['patch_size']`.

## 11. Verifying the results match the published numbers

After a complete fresh run, compare:

```bash
python -c "
import pandas as pd
df = pd.read_parquet('outputs/results/nih/exp01_disease_classification/exp01_nih_results.parquet')
print(df[['model','disease','auc']].sort_values(['disease','model']).to_string(index=False))
"
```

Expected (manuscript Table 1, NIH column):

```
     model       disease   auc
biomedclip  cardiomegaly  0.877
    dinov2  cardiomegaly  0.818
    dinov3  cardiomegaly  0.836
 medsiglip  cardiomegaly  0.913
   raddino  cardiomegaly  0.900
… (15 rows total)
```

A bit-exact match is not expected (CUDA non-determinism in convolutions);
±0.005 AUC drift run-to-run is normal.

If your numbers differ by more than that, see Section 10 (Common gotchas) or
open an issue.
