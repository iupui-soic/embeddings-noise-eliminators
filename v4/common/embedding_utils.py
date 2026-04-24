"""
common/embedding_utils.py
=========================

Model loading and embedding extraction with THREE pooling modes:

    cls            - CLS token (v1-v3 baseline)
    patch_mean     - mean-pooled patch tokens (global, no-CLS)
    patch_local    - mean-pooled patch tokens intersecting the known
                     artefact location (local to perturbation)

`patch_local` is the experiment that arbitrates the patch-token-dilution
hypothesis requested by Reviewer 3 of the RYAI submission.
"""

from __future__ import annotations

import gc
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class EmbeddingExtractor:
    """Single-model extractor. Loads one model at a time (GPU-friendly)."""

    def __init__(self, model_name: str, hf_token: Optional[str] = None,
                 device: Optional[str] = None):
        from transformers import AutoImageProcessor, AutoModel
        self.model_name = model_name.lower()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.hf_token = hf_token
        self._loader = "hf_automodel"   # default; biomedclip overrides below

        if self.model_name == "raddino":
            self.processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
            self.model = AutoModel.from_pretrained("microsoft/rad-dino")
            self._patch_size = 14
        elif self.model_name == "dinov3":
            kwargs = {"token": hf_token} if hf_token else {}
            self.processor = AutoImageProcessor.from_pretrained(
                "facebook/dinov3-vit7b16-pretrain-lvd1689m", **kwargs)
            self.model = AutoModel.from_pretrained(
                "facebook/dinov3-vit7b16-pretrain-lvd1689m",
                torch_dtype=torch.float16, **kwargs)
            self._patch_size = 16
        elif self.model_name == "dinov3_vits":
            # Small DINOv3 variant — fp32 is fine since it's only 22M params.
            kwargs = {"token": hf_token} if hf_token else {}
            self.processor = AutoImageProcessor.from_pretrained(
                "facebook/dinov3-vits16-pretrain-lvd1689m", **kwargs)
            self.model = AutoModel.from_pretrained(
                "facebook/dinov3-vits16-pretrain-lvd1689m", **kwargs)
            self._patch_size = 16
        elif self.model_name == "dinov2":
            # Natural-image DINOv2 ViT-B/14 — architectural twin of RAD-DINO,
            # isolates the pretraining-domain effect from dimensionality.
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            self.model = AutoModel.from_pretrained("facebook/dinov2-base")
            self._patch_size = 14
        elif self.model_name == "biomedclip":
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            model.eval()
            self.model = model
            self.processor = preprocess   # a torchvision transform, not an HF processor
            self._patch_size = 16
            self._loader = "open_clip"
        elif self.model_name == "medsiglip":
            kwargs = {"token": hf_token} if hf_token else {}
            # MedSigLIP is a SigLIP-family VLM.  We only need the image side,
            # so we skip the text tokenizer (which requires sentencepiece) by
            # loading just the image processor.
            self.processor = AutoImageProcessor.from_pretrained(
                "google/medsiglip-448", **kwargs)
            self.model = AutoModel.from_pretrained(
                "google/medsiglip-448", **kwargs)
            self._patch_size = 16
            self._loader = "medsiglip"
        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.model.to(self.device).eval()

    def close(self):
        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # -------------------------------------------------------------------
    # Low-level helpers
    # -------------------------------------------------------------------
    def _to_pil(self, image):
        if image.ndim == 3:
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return Image.fromarray(image).convert("RGB")

    def _forward_tokens(self, images):
        """
        Return both CLS and patch-token tensors for a batch.

        Returns
        -------
        cls_tokens   : (B, D)
        patch_tokens : (B, N, D)  where N = grid_h * grid_w
        grid_hw      : (grid_h, grid_w) of patch tokens after the processor
        """
        pil = [self._to_pil(im) for im in images]
        if self._loader == "open_clip":
            return self._forward_tokens_open_clip(pil)
        if self._loader == "medsiglip":
            return self._forward_tokens_medsiglip(pil)
        with torch.no_grad():
            inputs = self.processor(pil, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs, output_hidden_states=False)
            last = outputs.last_hidden_state        # (B, 1 + R + N, D)

        # Derive the patch grid from the preprocessed input resolution.
        # DINOv2/v3 insert R register tokens between CLS and patches; raw token
        # count = 1 (CLS) + R (registers) + N (patches).
        try:
            th = inputs["pixel_values"].shape[-2] // self._patch_size
            tw = inputs["pixel_values"].shape[-1] // self._patch_size
            expected_patches = th * tw
            total = last.shape[1]
            num_registers = max(0, total - 1 - expected_patches)
            gh, gw = th, tw
        except Exception:
            # Fallback: assume no registers, square grid
            num_registers = 0
            n = last.shape[1] - 1
            gh = gw = int(round(np.sqrt(n)))

        cls_tokens = last[:, 0, :].float().cpu().numpy()
        patch_start = 1 + num_registers
        patch_tokens = last[:, patch_start:patch_start + gh * gw, :].float().cpu().numpy()
        return cls_tokens, patch_tokens, (gh, gw)

    def _forward_tokens_medsiglip(self, pil_images):
        """Google MedSigLIP-448 forward; returns (cls, patches, grid_hw).

        MedSigLIP is a SigLIP-family VLM.  The SigLIP vision tower has no
        standalone CLS token; `pooler_output` is a learned attention-pooled
        image embedding (dim 1152 for the Large variant used by MedSigLIP-448).
        We use `pooler_output` as the CLS analogue and return the full patch
        sequence for patch_local / patch_mean pooling.
        """
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vision_out = self.model.vision_model(
                pixel_values=inputs["pixel_values"],
                output_hidden_states=False,
            )
            pooled = vision_out.pooler_output              # (B, D)
            patch  = vision_out.last_hidden_state          # (B, N, D)
        cls_tokens = pooled.float().cpu().numpy()
        patch_tokens = patch.float().cpu().numpy()
        # Derive grid from actual patch count (MedSigLIP-448 uses a 32x32
        # grid — effective patch_size 14, not the nominal 16 reported in
        # some docs).
        n = patch_tokens.shape[1]
        gh = gw = int(round(n ** 0.5))
        return cls_tokens, patch_tokens, (gh, gw)

    def _forward_tokens_open_clip(self, pil_images):
        """open_clip vision-tower forward; returns (cls, patches, grid_hw).

        Handles both the older native `VisionTransformer` (`visual.conv1`
        attribute) and the `TimmModel` wrapper around a timm ViT used by
        BiomedCLIP in open_clip>=3.x (`visual.trunk` attribute).

        We take ALL three tensors from the pre-projection 768-d transformer
        output so CLS and patch tokens share a dim, which keeps the patch-
        local pooling math identical to the HF AutoModel path.
        """
        batch = torch.stack([self.processor(im) for im in pil_images]).to(self.device)
        visual = self.model.visual
        with torch.no_grad():
            if hasattr(visual, "trunk"):
                # timm-backed vision tower (BiomedCLIP path in open_clip 3.x).
                # timm's ViT forward_features returns (B, 1+N, D) with CLS first.
                feats = visual.trunk.forward_features(batch)       # (B, 1+N, D)
                cls_tokens = feats[:, 0, :].float().cpu().numpy()
                patch = feats[:, 1:, :]
                # Grid from input resolution / patch size (square assumed)
                in_h = batch.shape[-2]
                gh = gw = in_h // self._patch_size
            else:
                # Older open_clip VisionTransformer path
                x = visual.conv1(batch)                            # (B, D, H', W')
                gh, gw = x.shape[-2], x.shape[-1]
                x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
                cls_tok = visual.class_embedding.to(x.dtype) + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
                x = torch.cat([cls_tok, x], dim=1)
                x = x + visual.positional_embedding.to(x.dtype)
                x = visual.ln_pre(x)
                x = x.permute(1, 0, 2)
                x = visual.transformer(x)
                x = x.permute(1, 0, 2)
                cls_tokens = visual.ln_post(x[:, 0, :]).float().cpu().numpy()
                patch = visual.ln_post(x[:, 1:, :])
        patch_tokens = patch.float().cpu().numpy()
        return cls_tokens, patch_tokens, (gh, gw)

    # -------------------------------------------------------------------
    # Public API - three pooling modes
    # -------------------------------------------------------------------

    def extract_cls(self, images) -> np.ndarray:
        cls, _, _ = self._forward_tokens(images)
        return cls

    def extract_patch_mean(self, images) -> np.ndarray:
        _, patches, _ = self._forward_tokens(images)
        return patches.mean(axis=1)

    def extract_patch_local(self, images,
                            patch_locations: List[List[Dict]],
                            image_hw: Tuple[int, int]) -> np.ndarray:
        """
        Mean-pool ONLY the patch tokens whose receptive field intersects
        any injected artefact patch for each image.

        Parameters
        ----------
        images           : list of HxWxC numpy arrays (the noisy ones)
        patch_locations  : for each image, list of {'y','x','size'} dicts
        image_hw         : (H, W) of the input images before HF processor
                           (we use this to compute which token grid cells
                           intersect each artefact patch)
        """
        _, patches, (gh, gw) = self._forward_tokens(images)
        H, W = image_hw
        cell_h = H / gh
        cell_w = W / gw

        pooled = np.zeros((patches.shape[0], patches.shape[2]),
                          dtype=patches.dtype)
        for bi, locs in enumerate(patch_locations):
            if not locs:
                pooled[bi] = patches[bi].mean(axis=0)
                continue
            mask_rows = np.zeros(gh, dtype=bool)
            mask_cols = np.zeros(gw, dtype=bool)
            for loc in locs:
                y0, x0, s = loc["y"], loc["x"], loc["size"]
                y1, x1 = y0 + s, x0 + s
                r0 = max(0, int(np.floor(y0 / cell_h)))
                r1 = min(gh, int(np.ceil(y1 / cell_h)))
                c0 = max(0, int(np.floor(x0 / cell_w)))
                c1 = min(gw, int(np.ceil(x1 / cell_w)))
                mask_rows[r0:r1] = True
                mask_cols[c0:c1] = True
            mask = np.outer(mask_rows, mask_cols).reshape(-1)
            if mask.sum() == 0:
                pooled[bi] = patches[bi].mean(axis=0)
            else:
                pooled[bi] = patches[bi][mask].mean(axis=0)
        return pooled

    def extract_all(self, images, patch_locations, image_hw):
        """Return a dict with all three pooling modes in a single forward."""
        cls, patches, (gh, gw) = self._forward_tokens(images)
        H, W = image_hw
        cell_h = H / gh
        cell_w = W / gw

        patch_mean = patches.mean(axis=1)
        patch_local = np.zeros_like(patch_mean)

        for bi, locs in enumerate(patch_locations):
            if not locs:
                patch_local[bi] = patches[bi].mean(axis=0)
                continue
            mask_rows = np.zeros(gh, dtype=bool)
            mask_cols = np.zeros(gw, dtype=bool)
            for loc in locs:
                y0, x0, s = loc["y"], loc["x"], loc["size"]
                y1, x1 = y0 + s, x0 + s
                r0 = max(0, int(np.floor(y0 / cell_h)))
                r1 = min(gh, int(np.ceil(y1 / cell_h)))
                c0 = max(0, int(np.floor(x0 / cell_w)))
                c1 = min(gw, int(np.ceil(x1 / cell_w)))
                mask_rows[r0:r1] = True
                mask_cols[c0:c1] = True
            mask = np.outer(mask_rows, mask_cols).reshape(-1)
            patch_local[bi] = patches[bi][mask].mean(axis=0) if mask.sum() else \
                              patches[bi].mean(axis=0)

        return {
            "cls":         cls,
            "patch_mean":  patch_mean,
            "patch_local": patch_local,
            "grid_hw":     (gh, gw),
        }
