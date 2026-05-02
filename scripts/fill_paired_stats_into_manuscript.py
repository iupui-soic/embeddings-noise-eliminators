#!/usr/bin/env python3
"""Patch manuscript.tex (\\PairedHeadline) and supplementary.tex (Table S11)
with the actual numbers from outputs/v4_exp_chestxdet10/paired_stats.parquet.

Idempotent: replaces existing sentinel block markers in both files.

Usage:
    python3 scripts/fill_paired_stats_into_manuscript.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PARQUET = REPO / "outputs" / "v4_exp_chestxdet10" / "paired_stats.parquet"
MAIN = REPO / "manuscript" / "manuscript.tex"
SUPP = REPO / "manuscript" / "supplementary.tex"

PRETTY_MODEL = {
    "raddino": "RAD-DINO",
    "dinov2": "DINOv2-B/14",
    "dinov3": "DINOv3 ViT-7B",
    "biomedclip": "BiomedCLIP",
    "medsiglip": "MedSigLIP",
}
MODEL_ORDER = ["raddino", "dinov2", "dinov3", "biomedclip", "medsiglip"]
CLASS_ORDER = ["Calcification", "Nodule", "Mass"]


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return r"$<10^{-4}$"
    if p < 1e-3:
        return f"${p:.4f}$"
    return f"${p:.3f}$"


def render_rows(df: pd.DataFrame) -> str:
    df = df.copy()
    df["model_order"] = df["model"].map(MODEL_ORDER.index)
    df["class_order"] = df["class"].map(CLASS_ORDER.index)
    df = df.sort_values(["model_order", "class_order"])

    out = []
    last_model = None
    for _, r in df.iterrows():
        if last_model is None:
            out.append(rf"\multirow{{3}}{{*}}{{{PRETTY_MODEL[r['model']]}}}")
        elif r["model"] != last_model:
            out.append(r"\midrule")
            out.append(rf"\multirow{{3}}{{*}}{{{PRETTY_MODEL[r['model']]}}}")
        last_model = r["model"]
        out.append(
            f" & {r['class']:<13}"
            f" & ${r['auc_cls']:.3f}$"
            f" & ${r['auc_patch_local']:.3f}$"
            f" & ${r['delta']:+.3f}$"
            f" & $[{r['delta_ci_lo']:+.3f},\\,{r['delta_ci_hi']:+.3f}]$"
            f" & {fmt_p(r['p_paired_bootstrap'])}"
            f" & {fmt_p(r['q_bh'])}"
            f" & {int(r['n_test'])} \\\\"
        )
    return "\n".join(out)


def make_headline(df: pd.DataFrame) -> str:
    n = len(df)
    n_rej = int(df["rejected_q05"].sum())
    p_max = df["p_paired_bootstrap"].max()
    q_max = df["q_bh"].max()
    delta_min = df["delta"].min()
    delta_max = df["delta"].max()
    if p_max < 1e-4:
        p_phrase = r"raw $p < 10^{-4}$"
    else:
        p_phrase = f"raw $p \\leq {p_max:.4f}$"
    if q_max < 1e-4:
        q_phrase = r"BH-adjusted $q < 10^{-4}$"
    else:
        q_phrase = f"BH-adjusted $q \\leq {q_max:.4f}$"
    if n_rej == n:
        return (
            f"All {n} cells reject $H_0: \\Delta = 0$ at BH-FDR "
            f"$q \\leq 0.05$ ({p_phrase}; {q_phrase}); per-cell "
            f"$\\Delta$ ranges ${delta_min:+.3f}$ to ${delta_max:+.3f}$."
        )
    return (
        f"{n_rej} of {n} cells reject $H_0: \\Delta = 0$ at "
        f"BH-FDR $q \\leq 0.05$ ({q_phrase}); per-cell $\\Delta$ "
        f"ranges ${delta_min:+.3f}$ to ${delta_max:+.3f}$."
    )


def patch_main(headline: str) -> None:
    text = MAIN.read_text()
    new_macro = (
        "%% Auto-generated headline of the per-cell paired bootstrap on "
        "Δ(patch_local − CLS); see Supplementary Table~S11.\n"
        f"\\newcommand{{\\PairedHeadline}}{{{headline}}}"
    )
    pat = re.compile(
        r"%% Placeholder for the headline result of the per-cell paired bootstrap.*?"
        r"\\newcommand\{\\PairedHeadline\}\{[^}]*\}",
        re.DOTALL,
    )
    if pat.search(text):
        text = pat.sub(lambda _m: new_macro, text)
    else:
        # Fall back to replacing only the \newcommand line.
        replacement = f"\\newcommand{{\\PairedHeadline}}{{{headline}}}"
        text = re.sub(
            r"\\newcommand\{\\PairedHeadline\}\{[^}]*\}",
            lambda _m: replacement,
            text,
        )
    MAIN.write_text(text)


def patch_supp(rows: str) -> None:
    text = SUPP.read_text()
    placeholder = (
        r"\\multicolumn\{9\}\{l\}\{\\emph\{Filled in once "
        r"\\texttt\{run\\_chestxdet10\\_paired\\_stats\\.py\} "
        r"completes; current output at\}\} \\\\\s*"
        r"\\multicolumn\{9\}\{l\}\{\\emph\{\\texttt\{outputs/v4\\_exp\\_"
        r"chestxdet10/paired\\_stats\\.parquet\}\\.\}\} \\\\"
    )
    pat = re.compile(placeholder, re.DOTALL)
    if not pat.search(text):
        # Already filled in; replace the previous data block instead.
        block = re.compile(
            r"(\\midrule\s*\n)(.*?)(\\bottomrule\s*\n\\end\{tabular\}\s*\n"
            r"\\end\{adjustbox\}\s*\n\\caption\{\\textbf\{ChestX-Det10 per-cell)",
            re.DOTALL,
        )
        text = block.sub(lambda m: m.group(1) + rows + "\n" + m.group(3),
                         text, count=1)
    else:
        text = pat.sub(lambda _m: rows, text)
    SUPP.write_text(text)


def main():
    if not PARQUET.exists():
        print(f"ERROR: {PARQUET} does not exist yet; run "
              "scripts/run_chestxdet10_paired_stats.py first.")
        sys.exit(1)
    df = pd.read_parquet(PARQUET)
    rows = render_rows(df)
    headline = make_headline(df)

    print("Headline:")
    print(f"  {headline}")
    print()
    print("Rows:")
    print(rows)
    print()
    patch_main(headline)
    patch_supp(rows)
    print(f"Patched: {MAIN}")
    print(f"Patched: {SUPP}")


if __name__ == "__main__":
    main()
