#!/usr/bin/env python3
"""Render the canonical Table~12 (main) and Table~S10 (supp) from
``outputs/v4_exp_chestxdet10/unified_bbox_results.parquet``.

Prints LaTeX rows for both tables; the manuscript files are then
patched manually (idempotent diff is small enough to read).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path("/home/saptpurk/embeddings-noise-eliminators")
PARQUET = REPO / "outputs" / "v4_exp_chestxdet10" / "unified_bbox_results.parquet"

PRETTY = {
    "raddino": "RAD-DINO",
    "dinov2": "DINOv2-B/14",
    "dinov3": "DINOv3 ViT-7B",
    "biomedclip": "BiomedCLIP",
    "medsiglip": "MedSigLIP",
}
MODELS = ["raddino", "dinov2", "dinov3", "biomedclip", "medsiglip"]
CLASSES = ["Calcification", "Nodule", "Mass"]


def main():
    df = pd.read_parquet(PARQUET)
    wide = df.pivot_table(
        index=["model", "class"], columns="pool", values=["auc", "auc_lo", "auc_hi"]
    )

    print("=" * 70)
    print("TABLE 12 (main manuscript): bbox-level AUC + Δ")
    print("=" * 70)
    for m in MODELS:
        print(f"\\multirow{{3}}{{*}}{{{PRETTY[m]}}}")
        for c in CLASSES:
            cls = wide.loc[(m, c), ("auc", "cls")]
            pm = wide.loc[(m, c), ("auc", "patch_mean")]
            pl = wide.loc[(m, c), ("auc", "patch_local")]
            d = pl - cls
            print(f" & {c:<13} & {cls:.3f} & {pm:.3f} & {pl:.3f} & ${d:+.3f}$ \\\\")
        if m != MODELS[-1]:
            print("\\midrule")

    deltas = df.pivot_table(index=["model", "class"], columns="pool", values="auc")
    deltas["delta"] = deltas["patch_local"] - deltas["cls"]
    delta_min = deltas["delta"].min()
    delta_max = deltas["delta"].max()
    pl_min = deltas["patch_local"].min()
    mass_pl_min = deltas.loc[
        deltas.index.get_level_values("class") == "Mass", "patch_local"].min()
    print(f"\nΔ range: {delta_min:+.3f} to {delta_max:+.3f}")
    print(f"min patch_local AUC across all 15 cells: {pl_min:.3f}")
    print(f"min patch_local AUC on Mass cells: {mass_pl_min:.3f}")
    print("\nPer-FM mean Δ (averaged across 3 classes):")
    per_fm = deltas.groupby(level="model")["delta"].mean()
    for m in MODELS:
        print(f"  {PRETTY[m]:<14} {per_fm[m]:+.3f}")

    print("\n" + "=" * 70)
    print("TABLE S10 (supplementary): AUC + 95% bootstrap CI")
    print("=" * 70)
    for m in MODELS:
        for c in CLASSES:
            for p in ("cls", "patch_mean", "patch_local"):
                auc = wide.loc[(m, c), ("auc", p)]
                lo = wide.loc[(m, c), ("auc_lo", p)]
                hi = wide.loc[(m, c), ("auc_hi", p)]
                npos_train = int(df.query(
                    "model == @m and `class` == @c and pool == @p")["n_pos_train"].iloc[0])
                npos_test = int(df.query(
                    "model == @m and `class` == @c and pool == @p")["n_pos_test"].iloc[0])
                p_tex = p.replace("_", r"\_")
                print(f"{PRETTY[m]:<13s} & {c:<13} & {p_tex:<11} & "
                      f"{auc:.3f} [{lo:.3f}, {hi:.3f}] & {npos_train} & {npos_test} \\\\")
            if c != CLASSES[-1]:
                pass  # no inter-class midrule needed in longtable
        if m != MODELS[-1]:
            print("\\midrule")


if __name__ == "__main__":
    main()
