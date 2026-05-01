"""
common/perturbations.py
=======================

All image-space perturbations used in v4 experiments, consolidated.

Backwards compatibility: the isotropic Gaussian blur pipeline is byte-identical
to the v1-v3 `LocalizedBlurInjector`, so existing parquet files remain
comparable.

v4 adds:
    * DirectionalMotionBlur  - linear-kernel motion blur with physically
      motivated angle (cranio-caudal for diaphragm, lateral for cardiac)
    * ReticularPatternInjector - narrow-band sinusoidal grid approximating
      fine interstitial reticulation
    * GroundGlassInjector - low-frequency Gaussian bump approximating
      ground-glass opacity

All injectors share the same interface:
    noisy, meta = injector(image, image_path=..., patch_size=..., ...)

where `meta` contains the exact patch location(s) and artefact parameters.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Deterministic seeding - identical to v1-v3 for reproducibility
# ---------------------------------------------------------------------------

def deterministic_seed(base_seed: int, image_path) -> int:
    """SHA-256(base_seed || filename) truncated to 32-bit int."""
    filename = os.path.basename(str(image_path))
    h = hashlib.sha256(f"{base_seed}_{filename}".encode())
    return int(h.hexdigest()[:8], 16)


def sample_patch_origin(rng, image_h, image_w, patch_size, margin=0.20):
    """Top-left (y, x) inside the central (1-2*margin) fraction."""
    y_min = int(image_h * margin)
    y_max = max(y_min + 1, int(image_h * (1 - margin)) - patch_size)
    x_min = int(image_w * margin)
    x_max = max(x_min + 1, int(image_w * (1 - margin)) - patch_size)
    return int(rng.integers(y_min, y_max)), int(rng.integers(x_min, x_max))


# ---------------------------------------------------------------------------
# Isotropic Gaussian motion-blur patch (v1-v3 compatible)
# ---------------------------------------------------------------------------

class LocalizedBlurInjector:
    """
    Apply an isotropic Gaussian blur (fixed k=21) globally, then replace
    a small patch of the clean image with the blurred patch.  Identical
    to the v1-v3 pipeline.

    IMPORTANT: the Gaussian *kernel* is 21x21.  The 4 and 8 values describe
    the *patch footprint* that is pasted back.  Earlier manuscript versions
    mislabelled these; v4 fixes the language.
    """

    def __init__(self, seed: int = 42,
                 blur_ksize: int = 21,
                 blur_sigma: float = 0.0,
                 placement_margin: float = 0.20,
                 clip_range: Tuple[int, int] = (20, 235)):
        self.seed = seed
        self.blur_ksize = blur_ksize
        self.blur_sigma = blur_sigma
        self.margin = placement_margin
        self.clip_range = clip_range

    def __call__(self, image, patch_size: int, num_patches: int = 1,
                 image_path: Optional[str] = None):
        rng = np.random.default_rng(
            deterministic_seed(self.seed, image_path) if image_path else self.seed
        )
        noisy = image.copy()
        h, w = image.shape[:2]
        blurred_full = cv2.GaussianBlur(
            image, (self.blur_ksize, self.blur_ksize), self.blur_sigma
        )

        patch_log = []
        for _ in range(num_patches):
            y, x = sample_patch_origin(rng, h, w, patch_size, self.margin)
            noisy[y:y + patch_size, x:x + patch_size] = \
                blurred_full[y:y + patch_size, x:x + patch_size]
            patch_log.append({"y": y, "x": x, "size": patch_size})

        # BUGFIX v4.1: apply clip only within each modified patch.  The prior
        # global clip silently shifted every dark-border pixel 0->20 and every
        # white 255->235, which the probe then detected as a global 'was this
        # image clipped?' signal instead of the local artifact.
        for _loc in patch_log:
            _y, _x, _s = _loc['y'], _loc['x'], _loc['size']
            noisy[_y:_y + _s, _x:_x + _s] = np.clip(
                noisy[_y:_y + _s, _x:_x + _s], *self.clip_range)
        noisy = noisy.astype(np.uint8)
        return noisy, {
            "artefact":      "localized_gaussian_blur",
            "kernel_size":   self.blur_ksize,
            "kernel_sigma":  self.blur_sigma,
            "patch_size":    patch_size,
            "patch_locations": patch_log,
        }


# ---------------------------------------------------------------------------
# NEW v4: Directional motion blur (physics-motivated)
# ---------------------------------------------------------------------------

def _linear_motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    """
    Build a 1-D line-shaped kernel of total size `length x length` with a
    unit-integral streak along `angle_deg` (0 = horizontal right, 90 = up).

    Clinically motivated angles in CXR:
      * angle_deg = 90  -> cranio-caudal (diaphragmatic excursion)
      * angle_deg = 0   -> left-right (cardiac wall motion)
    """
    k = np.zeros((length, length), dtype=np.float32)
    centre = (length - 1) / 2.0
    theta = np.deg2rad(angle_deg)
    dx, dy = np.cos(theta), -np.sin(theta)  # image y grows downward
    for t in np.linspace(-centre, centre, length):
        xi = int(round(centre + t * dx))
        yi = int(round(centre + t * dy))
        if 0 <= xi < length and 0 <= yi < length:
            k[yi, xi] = 1.0
    if k.sum() == 0:
        k[int(centre), int(centre)] = 1.0
    return k / k.sum()


class DirectionalMotionBlurInjector:
    """
    Physics-motivated directional motion blur patch.  Mimics the linear PSF
    that respiratory or cardiac motion imparts to a stationary detector over
    a finite exposure time.
    """

    def __init__(self, seed: int = 42,
                 kernel_length: int = 21,
                 angle_deg: float = 90.0,          # 0 lateral, 90 cranio-caudal
                 placement_margin: float = 0.20,
                 clip_range: Tuple[int, int] = (20, 235)):
        self.seed = seed
        self.kernel_length = kernel_length
        self.angle_deg = angle_deg
        self.margin = placement_margin
        self.clip_range = clip_range
        self._kernel = _linear_motion_kernel(kernel_length, angle_deg)

    def __call__(self, image, patch_size: int, num_patches: int = 1,
                 image_path: Optional[str] = None):
        rng = np.random.default_rng(
            deterministic_seed(self.seed, image_path) if image_path else self.seed
        )
        noisy = image.copy()
        h, w = image.shape[:2]

        # Convolve once globally then extract the patch (matches iso-blur style)
        blurred_full = cv2.filter2D(image, -1, self._kernel,
                                    borderType=cv2.BORDER_REFLECT)

        patch_log = []
        for _ in range(num_patches):
            y, x = sample_patch_origin(rng, h, w, patch_size, self.margin)
            noisy[y:y + patch_size, x:x + patch_size] = \
                blurred_full[y:y + patch_size, x:x + patch_size]
            patch_log.append({"y": y, "x": x, "size": patch_size})

        # BUGFIX v4.1: apply clip only within each modified patch.  The prior
        # global clip silently shifted every dark-border pixel 0->20 and every
        # white 255->235, which the probe then detected as a global 'was this
        # image clipped?' signal instead of the local artifact.
        for _loc in patch_log:
            _y, _x, _s = _loc['y'], _loc['x'], _loc['size']
            noisy[_y:_y + _s, _x:_x + _s] = np.clip(
                noisy[_y:_y + _s, _x:_x + _s], *self.clip_range)
        noisy = noisy.astype(np.uint8)
        return noisy, {
            "artefact":       "directional_motion_blur",
            "kernel_length":  self.kernel_length,
            "angle_deg":      self.angle_deg,
            "patch_size":     patch_size,
            "patch_locations": patch_log,
        }


# ---------------------------------------------------------------------------
# NEW v4: Subtle pathology-mimicking patterns
# ---------------------------------------------------------------------------

class ReticularPatternInjector:
    """
    Inject a narrow-band sinusoidal grid over a patch, as a simplified
    model of fine reticular interstitial markings.  Amplitude is scaled
    by the local patch mean so it stays within clinically plausible
    contrast.
    """

    def __init__(self, seed: int = 42,
                 period_px: int = 4,
                 amplitude: float = 0.08,
                 placement_margin: float = 0.20,
                 clip_range: Tuple[int, int] = (20, 235)):
        self.seed = seed
        self.period_px = period_px
        self.amplitude = amplitude
        self.margin = placement_margin
        self.clip_range = clip_range

    def __call__(self, image, patch_size: int, num_patches: int = 1,
                 image_path: Optional[str] = None):
        rng = np.random.default_rng(
            deterministic_seed(self.seed, image_path) if image_path else self.seed
        )
        noisy = image.copy().astype(np.float32)
        h, w = image.shape[:2]

        yy, xx = np.meshgrid(np.arange(patch_size), np.arange(patch_size),
                             indexing="ij")
        grid = np.sin(2 * np.pi * xx / self.period_px) + \
               np.sin(2 * np.pi * yy / self.period_px)
        grid /= np.abs(grid).max() + 1e-9     # in [-1, 1]

        patch_log = []
        for _ in range(num_patches):
            y, x = sample_patch_origin(rng, h, w, patch_size, self.margin)
            tile = noisy[y:y + patch_size, x:x + patch_size]
            if tile.ndim == 3:
                local_mean = tile.mean(axis=(0, 1), keepdims=True)
                modulation = (grid[..., None] * local_mean * self.amplitude)
            else:
                local_mean = tile.mean()
                modulation = grid * local_mean * self.amplitude
            noisy[y:y + patch_size, x:x + patch_size] = tile + modulation
            patch_log.append({"y": y, "x": x, "size": patch_size})

        # BUGFIX v4.1: apply clip only within each modified patch.  The prior
        # global clip silently shifted every dark-border pixel 0->20 and every
        # white 255->235, which the probe then detected as a global 'was this
        # image clipped?' signal instead of the local artifact.
        for _loc in patch_log:
            _y, _x, _s = _loc['y'], _loc['x'], _loc['size']
            noisy[_y:_y + _s, _x:_x + _s] = np.clip(
                noisy[_y:_y + _s, _x:_x + _s], *self.clip_range)
        noisy = noisy.astype(np.uint8)
        return noisy, {
            "artefact":   "reticular_pattern",
            "period_px":  self.period_px,
            "amplitude":  self.amplitude,
            "patch_size": patch_size,
            "patch_locations": patch_log,
        }


class GroundGlassInjector:
    """
    Inject a broad, low-frequency Gaussian bump that raises local
    mean intensity slightly, as a simplified proxy for ground-glass
    opacity.  Amplitude is scaled by local mean.
    """

    def __init__(self, seed: int = 42,
                 sigma_px: float = 12.0,
                 amplitude: float = 0.06,
                 placement_margin: float = 0.20,
                 clip_range: Tuple[int, int] = (20, 235)):
        self.seed = seed
        self.sigma_px = sigma_px
        self.amplitude = amplitude
        self.margin = placement_margin
        self.clip_range = clip_range

    def __call__(self, image, patch_size: int, num_patches: int = 1,
                 image_path: Optional[str] = None):
        rng = np.random.default_rng(
            deterministic_seed(self.seed, image_path) if image_path else self.seed
        )
        noisy = image.copy().astype(np.float32)
        h, w = image.shape[:2]

        yy, xx = np.meshgrid(np.arange(patch_size), np.arange(patch_size),
                             indexing="ij")
        cy, cx = patch_size / 2, patch_size / 2
        bump = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * self.sigma_px ** 2))

        patch_log = []
        for _ in range(num_patches):
            y, x = sample_patch_origin(rng, h, w, patch_size, self.margin)
            tile = noisy[y:y + patch_size, x:x + patch_size]
            if tile.ndim == 3:
                local_mean = tile.mean(axis=(0, 1), keepdims=True)
                modulation = bump[..., None] * local_mean * self.amplitude
            else:
                local_mean = tile.mean()
                modulation = bump * local_mean * self.amplitude
            noisy[y:y + patch_size, x:x + patch_size] = tile + modulation
            patch_log.append({"y": y, "x": x, "size": patch_size})

        # BUGFIX v4.1: apply clip only within each modified patch.  The prior
        # global clip silently shifted every dark-border pixel 0->20 and every
        # white 255->235, which the probe then detected as a global 'was this
        # image clipped?' signal instead of the local artifact.
        for _loc in patch_log:
            _y, _x, _s = _loc['y'], _loc['x'], _loc['size']
            noisy[_y:_y + _s, _x:_x + _s] = np.clip(
                noisy[_y:_y + _s, _x:_x + _s], *self.clip_range)
        noisy = noisy.astype(np.uint8)
        return noisy, {
            "artefact":   "ground_glass",
            "sigma_px":   self.sigma_px,
            "amplitude":  self.amplitude,
            "patch_size": patch_size,
            "patch_locations": patch_log,
        }


# ---------------------------------------------------------------------------
# v1-v3 synthetic geometric patterns (replicated for Table 2 coverage
# across newly-added foundation models)
# ---------------------------------------------------------------------------
#
# These match the original v1-v3 perturbation specs (circles C1/C2,
# squares S4/S8, diagonal-line tiles L4/L8).  Pixel intensities follow
# the v1-v3 adaptive sampling rule: modal_intensity +/- 20 gray levels,
# clipped to [20, 235].


def _adaptive_intensity(rng, local_region, spread: int = 20,
                        clip: Tuple[int, int] = (20, 235)) -> int:
    """v1-v3 adaptive-intensity sampling: modal +/- spread, clipped."""
    flat = np.asarray(local_region).reshape(-1)
    if flat.size == 0:
        modal = 128
    else:
        # Robust modal proxy: round to uint8 histogram mode
        vals, counts = np.unique(flat.astype(np.uint8), return_counts=True)
        modal = int(vals[int(np.argmax(counts))])
    lo = max(clip[0], modal - spread)
    hi = min(clip[1], modal + spread)
    if hi <= lo:
        return int(np.clip(modal, clip[0], clip[1]))
    return int(rng.integers(lo, hi + 1))


class CircleInjector:
    """Filled circle of fixed pixel radius (v1-v3: C1 = radius 1, C2 = radius 2)."""

    def __init__(self, seed: int = 42, radius: int = 1,
                 placement_margin: float = 0.20,
                 intensity_spread: int = 20,
                 clip_range: Tuple[int, int] = (20, 235)):
        self.seed = seed
        self.radius = radius
        self.margin = placement_margin
        self.intensity_spread = intensity_spread
        self.clip_range = clip_range

    def __call__(self, image, patch_size: int, num_patches: int = 1,
                 image_path: Optional[str] = None):
        rng = np.random.default_rng(
            deterministic_seed(self.seed, image_path) if image_path else self.seed
        )
        noisy = image.copy()
        h, w = image.shape[:2]
        patch_log = []
        for _ in range(num_patches):
            y, x = sample_patch_origin(rng, h, w, patch_size, self.margin)
            # Centre the circle inside the placement patch
            cy = y + patch_size // 2
            cx = x + patch_size // 2
            local = noisy[y:y + patch_size, x:x + patch_size]
            intensity = _adaptive_intensity(rng, local, self.intensity_spread,
                                            self.clip_range)
            if noisy.ndim == 3:
                cv2.circle(noisy, (cx, cy), self.radius,
                           (intensity, intensity, intensity), thickness=-1)
            else:
                cv2.circle(noisy, (cx, cy), self.radius, intensity, thickness=-1)
            patch_log.append({"y": y, "x": x, "size": patch_size,
                              "radius": self.radius})
        # BUGFIX v4.1: apply clip only within each modified patch.  The prior
        # global clip silently shifted every dark-border pixel 0->20 and every
        # white 255->235, which the probe then detected as a global 'was this
        # image clipped?' signal instead of the local artifact.
        for _loc in patch_log:
            _y, _x, _s = _loc['y'], _loc['x'], _loc['size']
            noisy[_y:_y + _s, _x:_x + _s] = np.clip(
                noisy[_y:_y + _s, _x:_x + _s], *self.clip_range)
        noisy = noisy.astype(np.uint8)
        return noisy, {
            "artefact":   "circle",
            "radius":     self.radius,
            "patch_size": patch_size,
            "patch_locations": patch_log,
        }


class SquareInjector:
    """
    v1-v3 S4/S8 square noise.  Each pixel inside the patch is assigned an
    independently sampled intensity near the local modal value, producing
    locally heterogeneous variation without directional structure.
    """

    def __init__(self, seed: int = 42,
                 placement_margin: float = 0.20,
                 intensity_spread: int = 20,
                 clip_range: Tuple[int, int] = (20, 235)):
        self.seed = seed
        self.margin = placement_margin
        self.intensity_spread = intensity_spread
        self.clip_range = clip_range

    def __call__(self, image, patch_size: int, num_patches: int = 1,
                 image_path: Optional[str] = None):
        rng = np.random.default_rng(
            deterministic_seed(self.seed, image_path) if image_path else self.seed
        )
        noisy = image.copy()
        h, w = image.shape[:2]
        patch_log = []
        for _ in range(num_patches):
            y, x = sample_patch_origin(rng, h, w, patch_size, self.margin)
            local = noisy[y:y + patch_size, x:x + patch_size]
            # Compute an intensity range around local mode
            flat = np.asarray(local).reshape(-1, 3)[:, 0] if local.ndim == 3 \
                else np.asarray(local).reshape(-1)
            vals, counts = np.unique(flat.astype(np.uint8), return_counts=True)
            modal = int(vals[int(np.argmax(counts))]) if vals.size else 128
            lo = max(self.clip_range[0], modal - self.intensity_spread)
            hi = min(self.clip_range[1], modal + self.intensity_spread)
            block = rng.integers(lo, hi + 1, size=(patch_size, patch_size),
                                 dtype=np.int32)
            if noisy.ndim == 3:
                noisy[y:y + patch_size, x:x + patch_size] = \
                    np.stack([block, block, block], axis=-1).astype(np.uint8)
            else:
                noisy[y:y + patch_size, x:x + patch_size] = block.astype(np.uint8)
            patch_log.append({"y": y, "x": x, "size": patch_size})
        # BUGFIX v4.1: apply clip only within each modified patch.  The prior
        # global clip silently shifted every dark-border pixel 0->20 and every
        # white 255->235, which the probe then detected as a global 'was this
        # image clipped?' signal instead of the local artifact.
        for _loc in patch_log:
            _y, _x, _s = _loc['y'], _loc['x'], _loc['size']
            noisy[_y:_y + _s, _x:_x + _s] = np.clip(
                noisy[_y:_y + _s, _x:_x + _s], *self.clip_range)
        noisy = noisy.astype(np.uint8)
        return noisy, {
            "artefact":   "square",
            "patch_size": patch_size,
            "patch_locations": patch_log,
        }


class DiagonalLineInjector:
    """
    v1-v3 L4/L8 diagonal-line noise.  Draws a 1-pixel diagonal stroke
    inside a tile of side `patch_size`, along either the main or the
    anti-diagonal (randomised per placement).
    """

    def __init__(self, seed: int = 42,
                 placement_margin: float = 0.20,
                 intensity_spread: int = 20,
                 clip_range: Tuple[int, int] = (20, 235)):
        self.seed = seed
        self.margin = placement_margin
        self.intensity_spread = intensity_spread
        self.clip_range = clip_range

    def __call__(self, image, patch_size: int, num_patches: int = 1,
                 image_path: Optional[str] = None):
        rng = np.random.default_rng(
            deterministic_seed(self.seed, image_path) if image_path else self.seed
        )
        noisy = image.copy()
        h, w = image.shape[:2]
        patch_log = []
        for _ in range(num_patches):
            y, x = sample_patch_origin(rng, h, w, patch_size, self.margin)
            local = noisy[y:y + patch_size, x:x + patch_size]
            intensity = _adaptive_intensity(rng, local, self.intensity_spread,
                                            self.clip_range)
            main = bool(rng.integers(0, 2))    # randomise main vs anti-diag
            for i in range(patch_size):
                j = i if main else (patch_size - 1 - i)
                ty, tx = y + i, x + j
                if 0 <= ty < h and 0 <= tx < w:
                    if noisy.ndim == 3:
                        noisy[ty, tx] = (intensity, intensity, intensity)
                    else:
                        noisy[ty, tx] = intensity
            patch_log.append({"y": y, "x": x, "size": patch_size,
                              "main_diag": main})
        # BUGFIX v4.1: apply clip only within each modified patch.  The prior
        # global clip silently shifted every dark-border pixel 0->20 and every
        # white 255->235, which the probe then detected as a global 'was this
        # image clipped?' signal instead of the local artifact.
        for _loc in patch_log:
            _y, _x, _s = _loc['y'], _loc['x'], _loc['size']
            noisy[_y:_y + _s, _x:_x + _s] = np.clip(
                noisy[_y:_y + _s, _x:_x + _s], *self.clip_range)
        noisy = noisy.astype(np.uint8)
        return noisy, {
            "artefact":   "diagonal_line",
            "patch_size": patch_size,
            "patch_locations": patch_log,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_injector(name: str, **kwargs):
    """Name-based factory used by all experiment notebooks."""
    name = name.lower()
    if name in ("iso_blur", "gaussian_blur", "localized_blur"):
        return LocalizedBlurInjector(**kwargs)
    if name in ("directional_blur", "dir_motion", "motion_blur_linear"):
        return DirectionalMotionBlurInjector(**kwargs)
    if name in ("reticular", "reticular_pattern"):
        return ReticularPatternInjector(**kwargs)
    if name in ("ground_glass", "gg"):
        return GroundGlassInjector(**kwargs)
    if name in ("circle", "c1", "c2"):
        return CircleInjector(**kwargs)
    if name in ("square", "s4", "s8"):
        return SquareInjector(**kwargs)
    if name in ("diagonal_line", "line", "l4", "l8"):
        return DiagonalLineInjector(**kwargs)
    raise ValueError(f"Unknown injector: {name!r}")
