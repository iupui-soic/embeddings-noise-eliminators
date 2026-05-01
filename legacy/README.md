# Legacy v3-era notebooks

These 50 notebooks are archival copies from the v3 study iteration. They are
**not** part of the current reproduction pipeline.

The current reproducible pipeline lives in:

- `notebooks/00_BuildMimicSubsample.ipynb` … `notebooks/25_IsoBlur_DeLong.ipynb`
- `common/` (shared utility modules)
- `scripts/` (audit and smoke-test utilities)

Run order, dependencies, and expected outputs are documented in the project
root `REPRODUCE.md`.

## What is here, by family

| Pattern | What it is |
|---|---|
| `00_DataCuration_*` | v3 synthetic-noise injection workflows (Circle, Line4/8, Square4/8) |
| `00_EmoryCXR_DataCuration*` | v3 Emory CXR data curation (cardiomegaly, lung lesion) |
| `01_Embeddings_DINOV3_*` | v3 DINOv3 embedding extraction per perturbation type |
| `01_Embeddings_RADDINO_*` | v3 RAD-DINO embedding extraction per perturbation type |
| `01_EmoryCXR_*Noise_*` | v3 Emory perturbation pipelines (DINO/RAD-DINO + ResNet50) |
| `01_NIH-CXR14_ResNet50_*` | v3 ResNet50 baselines on NIH per perturbation |
| `02_*`, `03_*` | v3 disease classification and localized blur analyses |

These were superseded by the unified, parameterized v4 pipeline. They are kept
in this folder for transparency and to preserve the historical record of the
study, but `git log -- legacy/` is the most useful way to inspect them.
