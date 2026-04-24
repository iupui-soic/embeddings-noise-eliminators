"""
common/config.py
================

Shared configuration for v4 experiments.

The same notebook runs on BOTH the NIH server and the Emory PHI-compatible
server. The only thing that changes is the `DATASET` environment variable
(or the top-of-notebook override).

On each server:
    export DATASET=nih        # NIH-CXR14 server
    export DATASET=emory      # Emory PHI-compatible server

Or inside a notebook:
    import os; os.environ["DATASET"] = "nih"
    from common.config import CFG
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Per-server path configuration
# ---------------------------------------------------------------------------
# Edit ONLY these blocks when moving to a different server.  Everything
# downstream is dataset-agnostic.

NIH_CONFIG = {
    "name": "NIH-CXR14",
    "img_dir": Path("/data0/NIH-CXR14/images"),
    "raw_csv": Path("/data0/NIH-CXR14/Data_Entry_2017_v2020.csv"),
    "work_dir": Path(os.environ.get(
        "V4_WORK_DIR",
        "/home/saptpurk/embeddings-noise-eliminators/v4_work",
    )),
    "target_size": (1024, 1024),
    # Disease label columns in the raw CSV
    "disease_columns": {
        "cardiomegaly": "Cardiomegaly",
        "edema":        "Edema",
        "lung_lesion":  "Nodule",       # NIH uses "Nodule" for lung-lesion equiv.
    },
    # How disease labels are encoded in the raw CSV.  NIH uses a pipe-
    # separated "Finding Labels" column.
    "label_mode": "nih_finding_labels",
    # Which image column to use
    "image_col": "Image Index",
}

EMORY_CONFIG = {
    "name": "EmoryCXR",
    # >>> REPLACE these with the actual Emory paths on your PHI server <<<
    "img_dir": Path("/path/to/emory/images"),
    "raw_csv": Path("/path/to/emory/labels.csv"),
    "work_dir": Path("/path/to/emory/work"),
    "target_size": (1024, 1024),
    "disease_columns": {
        "cardiomegaly": "cardiomegaly",
        "edema":        "edema",
        "lung_lesion":  "lung_lesion",
    },
    "label_mode": "binary_columns",     # one 0/1 column per disease
    "image_col": "image_path",
}


MIMIC_CONFIG = {
    "name": "MIMIC-CXR-JPG",
    # Root containing p10/, p11/, ... (after s3 sync of mimic-cxr-jpg/2.1.0/)
    "img_dir": Path("/data0/MIMIC-CXR/files"),
    # CheXpert-style per-study labels shipped with mimic-cxr-jpg/2.1.0/
    "raw_csv": Path("/data0/MIMIC-CXR/mimic-cxr-2.0.0-chexpert.csv.gz"),
    # Per-image metadata (subject_id, study_id, dicom_id, ViewPosition, ...)
    "metadata_csv": Path("/data0/MIMIC-CXR/mimic-cxr-2.0.0-metadata.csv.gz"),
    # Official train/validate/test split — preferred over re-splitting for
    # comparability to published MIMIC-CXR results.
    "split_csv": Path("/data0/MIMIC-CXR/mimic-cxr-2.0.0-split.csv.gz"),
    # Only keep frontal views to match NIH / Emory protocol
    "view_filter": ("AP", "PA"),
    "work_dir": Path(os.environ.get(
        "V4_WORK_DIR",
        "/home/saptpurk/embeddings-noise-eliminators/v4_work",
    )),
    "target_size": (1024, 1024),
    "disease_columns": {
        "cardiomegaly": "Cardiomegaly",
        "edema":        "Edema",
        "lung_lesion":  "Lung Lesion",
    },
    # MIMIC CheXpert uses -1=uncertain, 0=negative, 1=positive, NaN=not mentioned.
    # Mapping to 0/1 is done in data_loader.py (U-Zero: uncertain → 0, default).
    "label_mode": "mimic_chexpert",
    "image_col": "image_path",        # constructed inside load_disease_labels
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODELS = {
    "raddino": {
        "hf_id": "microsoft/rad-dino",
        "requires_token": False,
        "cls_dim": 768,
        "patch_size": 14,
        "description": "Radiology-specific DINOv2-based ViT-B/14",
        "loader": "hf_automodel",
    },
    "dinov3": {
        # High-capacity DINOv3 (paper's original claim rests on this variant's 4096-d embedding).
        "hf_id": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        "requires_token": True,
        "cls_dim": 4096,
        "patch_size": 16,
        "description": "General-purpose DINOv3 ViT-7B/16",
        "loader": "hf_automodel",
    },
    "dinov3_vits": {
        # Small DINOv3 variant — adds the low-dimensional natural-image control
        # to the dimension gradient (384-d, smaller than RAD-DINO's 768-d).
        "hf_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "requires_token": True,
        "cls_dim": 384,
        "patch_size": 16,
        "description": "General-purpose DINOv3 ViT-S/16 (low-dim natural-image control)",
        "loader": "hf_automodel",
    },
    # v4 additions: mechanistic and domain-breadth controls
    "dinov2": {
        # Natural-image DINOv2 ViT-B/14 — pairs exactly with RAD-DINO
        # (same architecture, same dim, same patch) to isolate the
        # pretraining-domain effect from dimensionality.
        "hf_id": "facebook/dinov2-base",
        "requires_token": False,
        "cls_dim": 768,
        "patch_size": 14,
        "description": "General-purpose DINOv2 ViT-B/14 (natural images)",
        "loader": "hf_automodel",
    },
    "biomedclip": {
        # Microsoft BiomedCLIP ViT-B/16 (biomedical image-text pretraining).
        # Broadens the "medical FM" category beyond RAD-DINO so the
        # findings can't be dismissed as RAD-DINO-specific.
        "hf_id": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "requires_token": False,
        # We take the pre-projection 768-d transformer CLS (not the 512-d
        # projected image embedding) so patch_local pooling math is
        # identical across models.
        "cls_dim": 768,
        "patch_size": 16,
        "description": "Microsoft BiomedCLIP ViT-B/16 (biomedical CLIP)",
        "loader": "open_clip",
    },
    "medsiglip": {
        # Google MedSigLIP-448 — recent (2024) medical SigLIP trained on
        # large-scale medical image-text pairs.  Provides a second
        # modern medical FM class (SigLIP-family) next to BiomedCLIP (CLIP-family).
        # Note: effective patch grid is 32x32 at 448 input (patch size 14).
        "hf_id": "google/medsiglip-448",
        "requires_token": True,            # google models are usually gated
        "cls_dim": 1152,                   # SigLIP-L image pooled output
        "patch_size": 14,
        "description": "Google MedSigLIP-448 (medical SigLIP-L at 448px)",
        "loader": "hf_automodel",
    },
}

# HF token is read from env.  NEVER commit a token in code.
HF_TOKEN = os.environ.get("HF_TOKEN", None)


# ---------------------------------------------------------------------------
# Experimental parameters (must be identical across servers for comparability)
# ---------------------------------------------------------------------------

@dataclass
class ExperimentParams:
    # Perturbation geometry
    placement_margin: float = 0.20          # central 60% of image
    noise_min_intensity: int = 20
    noise_max_intensity: int = 235
    # Isotropic motion blur (v1-v3 artifact)
    iso_blur_ksize: int = 21
    iso_blur_sigma: float = 0.0             # 0 => auto from ksize (~3.5)
    iso_patch_sizes: tuple = (4, 8)
    # Directional motion blur (NEW, v4 physics-motivated)
    directional_kernel_length: int = 21
    directional_angles: tuple = (0, 90)     # 0 = cranio-caudal, 90 = lateral
    directional_patch_sizes: tuple = (16, 32, 64)
    # Subtle pathology-like patterns (NEW, v4)
    reticular_period_px: tuple = (3, 6)     # fine vs coarse reticular
    reticular_amplitude: float = 0.08       # modulation as fraction of local mean
    ground_glass_sigma: float = 12.0        # broad low-freq bump
    ground_glass_amplitude: float = 0.06
    pathology_patch_sizes: tuple = (32, 64)
    # Probing
    n_bootstrap: int = 1000
    confidence_level: float = 0.95
    random_seed: int = 42
    # Classifier
    lr_C_grid: tuple = (0.01, 0.1, 1.0, 10.0)
    lr_max_iter: int = 2000
    # Batching
    batch_size: int = 16
    # DataLoader worker count for parallel image loading (Option C optimization).
    # Env override: NUM_WORKERS.  0 disables multiprocessing (safe fallback).
    num_workers: int = int(os.environ.get("NUM_WORKERS", "4"))


PARAMS = ExperimentParams()


# ---------------------------------------------------------------------------
# Runtime resolution
# ---------------------------------------------------------------------------

def _resolve_dataset() -> str:
    ds = os.environ.get("DATASET", "").strip().lower()
    if ds not in ("nih", "emory", "mimic"):
        raise RuntimeError(
            "Environment variable DATASET must be set to 'nih', 'emory', or "
            "'mimic' before importing common.config.  Example:\n"
            "    os.environ['DATASET'] = 'nih'\n"
            "then re-import."
        )
    return ds


@dataclass
class ResolvedConfig:
    dataset: str
    name: str
    img_dir: Path
    raw_csv: Path
    work_dir: Path
    target_size: tuple
    disease_columns: Dict[str, str]
    label_mode: str
    image_col: str
    # Optional fields used only by some datasets (e.g., MIMIC)
    metadata_csv: Optional[Path] = None
    view_filter: Optional[tuple] = None
    split_csv: Optional[Path] = None

    def output_dir(self, experiment_id: str) -> Path:
        """Per-experiment output directory on THIS server."""
        p = self.work_dir / f"v4_{experiment_id}_{self.dataset}"
        p.mkdir(parents=True, exist_ok=True)
        return p


def get_config() -> ResolvedConfig:
    ds = _resolve_dataset()
    src = {"nih": NIH_CONFIG, "emory": EMORY_CONFIG, "mimic": MIMIC_CONFIG}[ds]
    return ResolvedConfig(dataset=ds, **src)


# ---------------------------------------------------------------------------
# Which models to run
# ---------------------------------------------------------------------------
# Experiments consult MODELS_TO_RUN (comma-separated) so we can stage runs
# as access to each model is granted.  Defaults to running everything.
def models_to_run() -> List[str]:
    raw = os.environ.get("MODELS_TO_RUN", "").strip()
    if not raw:
        return list(MODELS.keys())
    out = [m.strip().lower() for m in raw.split(",") if m.strip()]
    bad = [m for m in out if m not in MODELS]
    if bad:
        raise ValueError(f"Unknown model(s) in MODELS_TO_RUN: {bad}. "
                         f"Known: {list(MODELS.keys())}")
    return out


# Lazy singleton for convenience
try:
    CFG = get_config()
except RuntimeError:
    CFG = None  # Resolved on-demand in notebook if DATASET not yet set
