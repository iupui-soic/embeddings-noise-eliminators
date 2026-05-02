"""
common/bbox_pool.py
====================

Region-aware patch-local pooling for non-square bounding boxes.

This extends :func:`common.embedding_utils.EmbeddingExtractor.extract_all`
with support for per-image lists of ``[x1, y1, x2, y2]`` bboxes (rather
than the square ``{'y','x','size'}`` artefact-patch format used by the
synthetic-perturbation experiments).

The pool is the mean of the final-layer patch tokens whose grid cells
intersect the union of the supplied bboxes for the image. If an image
has no bboxes (e.g. negative samples), the pool falls back to the global
patch-mean pool.

Returned dict keys mirror :func:`extract_all`:
  cls, patch_mean, patch_local, grid_hw
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .embedding_utils import EmbeddingExtractor

Bbox = Sequence[float]  # (x1, y1, x2, y2) in pixel coords of the input image


def extract_all_bbox(
    extractor: EmbeddingExtractor,
    images,
    bboxes_per_image: List[List[Bbox]],
    image_hw: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    """Same forward pass, three pools, with patch-local masked by ``bboxes``.

    Parameters
    ----------
    extractor       : a loaded ``EmbeddingExtractor``
    images          : list of HxWxC numpy arrays (BGR or RGB; ``_to_pil`` handles)
    bboxes_per_image: per-image list of ``[x1,y1,x2,y2]`` bboxes in image coords
    image_hw        : (H, W) of the input images before the HF processor
    """
    cls, patches, (gh, gw) = extractor._forward_tokens(images)
    H, W = image_hw
    cell_h = H / gh
    cell_w = W / gw

    patch_mean = patches.mean(axis=1)
    patch_local = np.zeros_like(patch_mean)

    for bi, bboxes in enumerate(bboxes_per_image):
        if not bboxes:
            patch_local[bi] = patches[bi].mean(axis=0)
            continue
        mask = np.zeros((gh, gw), dtype=bool)
        for bb in bboxes:
            x1, y1, x2, y2 = bb
            r0 = max(0, int(np.floor(y1 / cell_h)))
            r1 = min(gh, int(np.ceil(y2 / cell_h)))
            c0 = max(0, int(np.floor(x1 / cell_w)))
            c1 = min(gw, int(np.ceil(x2 / cell_w)))
            if r1 > r0 and c1 > c0:
                mask[r0:r1, c0:c1] = True
        flat = mask.reshape(-1)
        if flat.sum() == 0:
            patch_local[bi] = patches[bi].mean(axis=0)
        else:
            patch_local[bi] = patches[bi][flat].mean(axis=0)

    return {
        "cls":         cls,
        "patch_mean":  patch_mean,
        "patch_local": patch_local,
        "grid_hw":     (gh, gw),
    }


def random_negative_bbox(
    rng: np.random.Generator,
    image_hw: Tuple[int, int],
    target_h: int,
    target_w: int,
) -> Bbox:
    """Sample a random bbox of the given (h, w) inside the image.

    Used as a matched-area random window for hard-negative-region
    construction in bbox-level patch-local probes.
    """
    H, W = image_hw
    th = max(1, min(target_h, H))
    tw = max(1, min(target_w, W))
    y1 = int(rng.integers(0, max(1, H - th)))
    x1 = int(rng.integers(0, max(1, W - tw)))
    return (x1, y1, x1 + tw, y1 + th)
