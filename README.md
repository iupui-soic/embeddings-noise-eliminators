# Foundation Model Embeddings Discard Small-Scale Signal in Chest Radiography

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

This repository contains the code, data manifests, and LaTeX submission package
for the manuscript:

> **Foundation Model Embeddings Discard Small-Scale Signal in Chest
> Radiography: A Mechanistic Analysis of CLS-Aggregation Pooling.**
> Muthyala R., Yin Z., Jilla A., Li F., Dapamede T., Khosravi B.,
> Chavoshi M., Gichoya J., Purkayastha S. *Medical Image Analysis* (under
> review).

## Headline result

Across **5 frozen ViT foundation models** (RAD-DINO, DINOv2-B/14, DINOv3
ViT-7B, BiomedCLIP, MedSigLIP) and **3 large chest-radiography datasets**
(NIH-CXR14 *n* = 112 120; MIMIC-CXR *n* = 243 324; Emory CXR *n* = 137 280),
**every model's CLS embedding is at or near the chance floor** when probed
for small-scale (sub-percent of image area) perturbations — synthetic
geometric patterns, isotropic and physics-motivated directional motion blur,
and pathology-mimicking reticular and ground-glass patterns.

The same models maintain disease-classification AUCs of 0.642–0.913 on
cardiomegaly, edema, and lung lesion. The disconnect between disease-AUC
performance and small-scale-signal sensitivity is a property of the embedding
space — not the input — and we localize the loss to the **CLS-aggregation
step of the frozen forward pass**: across all 5 models, patch-local probes
recover AUC ≈ 1.0 on the same forward pass while the global CLS probe is at
chance.

The mechanism produces a measurable, recoverable effect on real CXR small
lesions. On **ChestX-Det10** (*n* = 3 543 NIH-CXR14 images with 1 462 bbox
annotations across Calcification, Nodule, Mass), CLS-pool image-level
classification shows a within-class small/large stratum gap up to **+0.218
AUC** across 15 (FM × class) cells. A region-aware bbox-level **patch-local
probe** on the same forward pass recovers **AUC ≥ 0.91** on every cell with
a patch-local-vs-CLS gap of **+0.126 to +0.350**, eliminating the
size-stratified disadvantage and yielding a worked, deployable mitigation.

We propose a **Small-Feature Detectability Index (SFDI)** — the geometric
mean of per-perturbation linear-probe AUCs over a standardized small-scale
battery — as a representational-property summary. Under the proposed
battery $\mathcal{B}_\text{CXR}$, all five foundation models score within
**0.505–0.524** across all three datasets, near the 0.5 chance floor.

## What is in this repository

```
.
├── README.md                  # this file
├── REPRODUCE.md               # end-to-end reproduction runbook
├── LICENSE                    # Apache 2.0
├── CITATION.cff               # citation metadata
├── requirements.txt           # Python dependencies (papermill-ready)
├── common/                    # shared modules used by every notebook
│   ├── config.py              # dataset paths, model registry, hyperparameters
│   ├── data_loader.py         # NIH / MIMIC / Emory label normalization
│   ├── embedding_utils.py     # frozen ViT extractor; CLS / patch-mean / patch-local
│   ├── perturbations.py       # SHA-256-deterministic injectors
│   ├── probing.py             # L2 logistic regression + bootstrap CIs
│   └── stats.py               # DeLong, permutation, BH-FDR, paired bootstrap
├── notebooks/                 # papermill-parameterized notebooks
│   ├── 00_BuildMimicSubsample.ipynb       # MIMIC subsample manifest
│   ├── 01_DiseaseClassification.ipynb     # main Table 1
│   ├── 02_SyntheticGeometric.ipynb        # 90-cell geometric battery
│   ├── 03_IsoMotionBlur.ipynb             # 30-cell iso-blur null
│   ├── 04_CleanVsPerturbed_DiseaseClassification.ipynb  # 405-cell clinical loop
│   ├── 05_EmbeddingVisualization_UMAP.ipynb
│   ├── 06_PatchToken_Probing.ipynb        # CLS / patch-mean / patch-local
│   ├── 07_RawPixel_Baseline.ipynb         # raw-pixel oracle
│   ├── 08_DirectionalMotionBlur.ipynb     # 270-cell directional sweep
│   ├── 09_Combined_Analysis_Stats.ipynb   # DeLong + BH-FDR
│   ├── 10–18                              # 9 sensitivity analyses
│   ├── 20_DinoResNet50_Battery.ipynb      # frozen DINO-ResNet-50 architectural control
│   ├── 22_ResNet50_Oracle.ipynb           # 32×32 oracle pixel positive control
│   ├── 23_ResNet50_Baseline.ipynb         # whole-image ResNet-50 (reticular, ground-glass)
│   ├── 24_ResNet50_GlobalIsoDir.ipynb     # whole-image ResNet-50 (iso, dir-blur)
│   ├── 25_IsoBlur_DeLong.ipynb            # paired DeLong RAD-DINO vs DINOv3
│   └── 26_ChestXDet10_SmallLesion_PatchPool.ipynb  # ChestX-Det10 bbox region-aware patch-pool
├── scripts/                   # build & audit utilities (non-notebook)
│   ├── build_mimic_subsample.py           # legacy CLI alternative to notebook 00
│   ├── build_sensitivity_notebooks.py     # regenerator for notebooks 10–18
│   ├── build_notebook_26.py               # regenerator for notebook 26 (ChestX-Det10)
│   ├── audit_all_perturbations.py         # cross-experiment integrity audit
│   ├── audit_exp02.py                     # exp02-specific PNG/injection audit
│   ├── model_sanity.py                    # 5-min model load + dim check
│   ├── smoke_test.py                      # 200-image end-to-end smoke
│   ├── run_*.sh                           # multi-GPU orchestrators (NIH/MIMIC/sensitivity)
│   ├── run_chestxdet10_smalllesion.py     # CLI driver for notebook 26's pipeline
│   ├── assemble_results.sh, merge_gpu_parquets.sh
│   └── download_dino_resnet50.sh          # auxiliary SSL-CNN weights download
├── manifests/                 # reproducibility-frozen data artifacts
│   └── mimic_subsample_ids.parquet        # MIMIC pre-registered probe-train subsample
├── manuscript/                # MedIA submission package
│   ├── manuscript.tex
│   ├── supplementary.tex
│   ├── refs.bib
│   ├── fill_placeholders.py               # parquet → LaTeX token resolver
│   ├── README_TEX.md
│   ├── Meta-Radiology_reviews.txt
│   └── Radiology-AI_reviews.txt
├── legacy/                    # 50 v3-era notebooks, archived (not for reproduction)
└── outputs/                   # gitignored: result parquets, logs, model weights
```

## Quick start

See **[REPRODUCE.md](REPRODUCE.md)** for the full reproducibility runbook,
including dataset acquisition, hardware requirements, expected runtimes, and
papermill orchestration. The condensed version:

```bash
git clone https://github.com/iupui-soic/embeddings-noise-eliminators.git
cd embeddings-noise-eliminators
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_dino_resnet50.sh        # auxiliary SSL-CNN weights

# Smoke test (~30 min on a single GPU, 200 images)
python scripts/smoke_test.py

# Full pipeline (papermill-orchestrated)
bash scripts/run_papermill_all.sh
python manuscript/fill_placeholders.py --results-root outputs/results/
cd manuscript && pdflatex manuscript.tex && bibtex manuscript && \
                pdflatex manuscript.tex && pdflatex manuscript.tex
```

## Datasets

| Dataset | Access | Used for |
|---|---|---|
| [NIH-CXR14](https://www.kaggle.com/datasets/nih-chest-xrays/data) | Public | Primary (probe training, all 9 perturbation experiments) |
| [MIMIC-CXR-JPG v2.0.0](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) | PhysioNet credentialed access | Primary (subsampled probe-training; full validate/test) |
| [ChestX-Det10](https://github.com/Deepwise-AILab/ChestX-Det10-Dataset) | Public (annotations on GitHub; images from Deepwise mirror) | ChestX-Det10 small-lesion bbox-stratified analysis (notebook 26) |
| Emory CXR | Institutional, IRB-restricted | Primary (PHI-compatible on-prem only) |

## Citation

If you use this code or its findings in your research, please cite:

```bibtex
@article{muthyala2026embeddings,
  title   = {Foundation Model Embeddings Discard Small-Scale Signal in
             Chest Radiography: A Mechanistic Analysis of CLS-Aggregation
             Pooling},
  author  = {Muthyala, Raajitha and Yin, Zhenan and Jilla, Alekhya and
             Li, Frank and Dapamede, Theo and Khosravi, Bardia and
             Chavoshi, Mohammadreza and Gichoya, Judy and
             Purkayastha, Saptarshi},
  journal = {Medical Image Analysis},
  year    = {2026},
  note    = {Under review}
}
```

A machine-readable `CITATION.cff` is also provided.

## Contact

**Saptarshi Purkayastha, PhD**
Department of Biomedical Engineering and Informatics
Indiana University, Indianapolis, IN, USA
[saptpurk@iu.edu](mailto:saptpurk@iu.edu)

## License

Apache 2.0. See [`LICENSE`](LICENSE).
