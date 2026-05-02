"""v4 shared utilities for both NIH and Emory servers.

Submodules with heavy dependencies (`embedding_utils` needs torch) are
imported lazily so that analysis machines that only do stats / plotting
can still `from common import ...` the lighter parts.
"""
from .config import CFG, PARAMS, MODELS, HF_TOKEN, get_config, models_to_run
from .perturbations import (
    LocalizedBlurInjector,
    DirectionalMotionBlurInjector,
    ReticularPatternInjector,
    GroundGlassInjector,
    CircleInjector,
    SquareInjector,
    DiagonalLineInjector,
    make_injector,
)
from .stats import (
    delong_test, permutation_auc_test,
    benjamini_hochberg, paired_bootstrap_delta_auc,
)
from .data_loader import (
    load_disease_labels, load_and_pad, stratified_split,
    CXRDataset, parallel_iter,
)

# probing uses sklearn only; safe on analysis machines
from .probing import train_probe, save_probe, load_probe_result, ProbeResult

# Torch-dependent import is optional
try:
    from .embedding_utils import EmbeddingExtractor
    from .bbox_pool import extract_all_bbox, random_negative_bbox
except ImportError as _e:  # torch not installed - analysis-only machine
    import warnings as _w
    _w.warn(
        f"common.embedding_utils unavailable ({_e}); "
        "EmbeddingExtractor will not be importable on this machine. "
        "This is fine for statistics/plotting notebooks."
    )
    EmbeddingExtractor = None  # type: ignore
    extract_all_bbox = None  # type: ignore
    random_negative_bbox = None  # type: ignore
