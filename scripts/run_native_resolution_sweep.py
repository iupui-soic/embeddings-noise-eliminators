#!/usr/bin/env python3
"""Driver for the matched-native-resolution control sweep (S15).

Runs ``common.native_resolution.run_native_resolution_sweep`` and writes the
canonical results parquet to ``outputs/v4_exp15_native_resolution/``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path("/home/saptpurk/embeddings-noise-eliminators")
sys.path.insert(0, str(REPO))

from common.native_resolution import run_native_resolution_sweep  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "raddino", "dinov2", "dinov3", "biomedclip", "medsiglip"])
    ap.add_argument("--n-clean", type=int, default=200)
    ap.add_argument("--dataset", default="nih")
    ap.add_argument("--out-dir", default=str(
        REPO / "outputs" / "v4_exp15_native_resolution"))
    args = ap.parse_args()

    perturbations = [
        ("iso_blur", 4),
        ("iso_blur", 16),
        ("reticular", 32),
    ]
    run_native_resolution_sweep(
        models=args.models,
        perturbations=perturbations,
        dataset=args.dataset,
        n_clean=args.n_clean,
        out_dir=Path(args.out_dir),
        hf_token=os.environ.get("HF_TOKEN"),
    )


if __name__ == "__main__":
    main()
