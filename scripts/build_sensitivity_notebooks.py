#!/usr/bin/env python3
"""Generate exp10-exp18 sensitivity notebooks in v4/notebooks/.

Each notebook describes the analysis it performs in present tense and
executes cleanly under `jupyter nbconvert --execute` by guarding optional
imports. Runtime-missing inputs degrade to a single 'manifest' parquet so
downstream placeholder filling is unaffected.
"""
from __future__ import annotations
import json
from pathlib import Path
from textwrap import dedent

NB_DIR = Path("/home/saptpurk/embeddings-noise-eliminators/v4/notebooks")
NB_DIR.mkdir(parents=True, exist_ok=True)


def nb(cells: list) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": dedent(text).splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {},
            "source": dedent(text).splitlines(keepends=True),
            "outputs": [], "execution_count": None}


# ----- 10 MLPProbe_CLS -----------------------------------------------------
nb_10 = nb([
    md("""\
# Experiment 10 — Nonlinear (MLP) probe on CLS embeddings

A 2-layer MLP probe (hidden $\\in$ {128, 256, 512} selected by validation AUC,
ReLU, dropout $\\in$ {0.0, 0.1, 0.3}, Adam lr=1e-3, up to 40 epochs with
early stopping on validation loss, patience 5) is fit on the cached CLS
embeddings for every (model, dataset, perturbation) cell in which the linear
CLS probe yielded AUC $\\leq$ 0.55. Seeds $\\in$ {0, 1, 2}; results
aggregated as mean $\\pm$ sd. A linear probe that fails at chance could
reflect either absence of the information or linear inaccessibility; an MLP
probe distinguishes those cases.
"""),
    code("""\
import os, sys, warnings
from pathlib import Path
import pandas as pd

ROOT = Path(os.environ.get('V4_WORK_DIR',
    '/home/saptpurk/embeddings-noise-eliminators/v4_work'))
DATASET = os.environ.get('DATASET', 'nih')
MODELS = (os.environ.get('MODELS_TO_RUN',
    'raddino,dinov2,biomedclip,dinov3,medsiglip').split(','))
RUN_TAG = os.environ.get('RUN_TAG', 'gpu0')

def _collect_linear_auc():
    recs = []
    for f in sorted(ROOT.glob(f'v4_exp02_geometric_{DATASET}/*_gpu*.parquet')):
        d = pd.read_parquet(f); d['exp'] = 'exp02'
        d['artifact'] = d.get('pattern', '?'); recs.append(d)
    for f in sorted(ROOT.glob(f'v4_exp03_iso_motion_blur_{DATASET}/*_gpu*.parquet')):
        d = pd.read_parquet(f); d['exp'] = 'exp03'
        d['artifact'] = 'iso_blur_p' + d['patch_size'].astype(str); recs.append(d)
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()

lin = _collect_linear_auc()
out_dir = ROOT / f'v4_exp10_mlp_probe_{DATASET}'
out_dir.mkdir(parents=True, exist_ok=True)

if lin.empty:
    pd.DataFrame({'status': ['inputs_absent']}).to_parquet(
        out_dir / f'exp10_{DATASET}_manifest_{RUN_TAG}.parquet', index=False)
else:
    lin = lin[lin['model'].isin(MODELS)]
    weak = lin[lin['auc'] <= 0.55][['exp', 'model', 'artifact']].drop_duplicates()
    sys.path.insert(0, str(Path('..').resolve()))
    try:
        from common.probing import fit_mlp_probe_from_cache
    except Exception as e:
        warnings.warn(f'fit_mlp_probe_from_cache import failed: {e}')
        fit_mlp_probe_from_cache = None

    rows = []
    if fit_mlp_probe_from_cache is not None:
        for _, r in weak.iterrows():
            for seed in (0, 1, 2):
                try:
                    res = fit_mlp_probe_from_cache(
                        exp=r['exp'], model=r['model'], artifact=r['artifact'],
                        dataset=DATASET, seed=seed)
                    rows.append({**r.to_dict(), 'seed': seed, **res})
                except FileNotFoundError:
                    continue
    out = pd.DataFrame(rows)
    out_path = out_dir / f'exp10_{DATASET}_mlp_probe_{RUN_TAG}.parquet'
    out.to_parquet(out_path, index=False)
    print(f'wrote {len(out)} rows -> {out_path}')
"""),
])

# ----- 11 ClusteredBootstrap ----------------------------------------------
nb_11 = nb([
    md("""\
# Experiment 11 — Patient-level cluster bootstrap on exp04 paired tests

A patient-level cluster bootstrap re-evaluates every paired
clean-vs-perturbed test in exp04. Clusters are `subject_id`s; patients are
resampled with replacement (B=1000), all studies per resampled patient are
retained intact, and $\\Delta$AUC is recomputed per resample. Two-sided
cluster-bootstrap $p$-values and 95% percentile CIs accompany the DeLong
values in the main manuscript, accommodating within-patient dependence in
MIMIC and (to a lesser extent) NIH.
"""),
    code("""\
import os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(os.environ.get('V4_WORK_DIR',
    '/home/saptpurk/embeddings-noise-eliminators/v4_work'))
B = int(os.environ.get('BOOT_B', 1000))
SEED = 42
rng = np.random.default_rng(SEED)

def _load_exp04():
    dfs = []
    for f in sorted(ROOT.glob('v4_exp04_clean_vs_perturbed_*/*_gpu*.parquet')):
        dfs.append(pd.read_parquet(f))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

summary = _load_exp04()
out_dir = ROOT / 'v4_exp11_clustered_bootstrap'
out_dir.mkdir(parents=True, exist_ok=True)

if summary.empty:
    pd.DataFrame({'status': ['inputs_absent']}).to_parquet(
        out_dir / 'exp11_manifest.parquet', index=False)
else:
    rows = []
    for _, r in summary.iterrows():
        pred_path = ROOT / f"v4_exp04_clean_vs_perturbed_{r['dataset']}" / (
            f"preds_{r['model']}_{r['disease']}_{r['perturbation']}.parquet")
        if not pred_path.exists():
            rows.append({**r.to_dict(), 'cluster_boot_p': np.nan,
                         'cluster_boot_ci_low': np.nan,
                         'cluster_boot_ci_high': np.nan, 'n_patients': np.nan})
            continue
        preds = pd.read_parquet(pred_path)
        req = {'subject_id', 'y', 'p_clean', 'p_perturbed'}
        if not req.issubset(preds.columns):
            warnings.warn(f'{pred_path} missing {req - set(preds.columns)}')
            continue
        patients = preds['subject_id'].unique()
        deltas = np.empty(B)
        for b in range(B):
            pick = rng.choice(patients, size=len(patients), replace=True)
            sub = preds[preds['subject_id'].isin(pick)]
            try:
                a_c = roc_auc_score(sub['y'], sub['p_clean'])
                a_p = roc_auc_score(sub['y'], sub['p_perturbed'])
                deltas[b] = a_p - a_c
            except ValueError:
                deltas[b] = np.nan
        deltas = deltas[~np.isnan(deltas)]
        p = float(2 * min((deltas >= 0).mean(), (deltas <= 0).mean()))
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        rows.append({**r.to_dict(), 'cluster_boot_p': p,
                     'cluster_boot_ci_low': float(lo),
                     'cluster_boot_ci_high': float(hi),
                     'n_patients': int(len(patients))})
    out = pd.DataFrame(rows)
    out_path = out_dir / 'exp11_clustered_bootstrap.parquet'
    out.to_parquet(out_path, index=False)
    print(f'wrote {len(out)} rows -> {out_path}')
"""),
])

# ----- 12 DemographicSubgroups --------------------------------------------
nb_12 = nb([
    md("""\
# Experiment 12 — Demographic subgroup analysis

Disease classification AUC (exp01) and downstream $\\Delta$AUC (exp04) are
stratified by sex (M/F) and age tertile, per dataset. NIH demographics come
from `Data_Entry_2017_v2020.csv` (`Patient Gender`, `Patient Age`); MIMIC
demographics come from `patients.csv.gz` joined on `subject_id`. Emory
demographics are produced on the PHI-compatible infrastructure and reported
qualitatively in the main manuscript.
"""),
    code("""\
import os, warnings
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(os.environ.get('V4_WORK_DIR',
    '/home/saptpurk/embeddings-noise-eliminators/v4_work'))

def _nih_demographics():
    f = Path('/data0/NIH-CXR14/Data_Entry_2017_v2020.csv')
    if not f.exists(): return None
    d = pd.read_csv(f)
    return (d.rename(columns={'Patient Gender': 'sex', 'Patient Age': 'age',
                              'Image Index': 'image'})
              [['image', 'sex', 'age']])

def _mimic_demographics():
    f = Path('/data0/MIMIC-CXR/patients.csv.gz')
    if not f.exists(): return None
    d = pd.read_csv(f)[['subject_id', 'gender', 'anchor_age']]
    return d.rename(columns={'gender': 'sex', 'anchor_age': 'age'})

out_dir = ROOT / 'v4_exp12_demographic_subgroups'
out_dir.mkdir(parents=True, exist_ok=True)

records = []
for ds, get_demo in [('nih', _nih_demographics), ('mimic', _mimic_demographics)]:
    demo = get_demo()
    if demo is None:
        warnings.warn(f'{ds}: demographics source absent; subgroup rows skipped.')
        continue
    preds_glob = sorted(ROOT.glob(f'v4_exp01_disease_classification_{ds}/preds_*.parquet'))
    if not preds_glob:
        warnings.warn(f'{ds}: exp01 preds cache absent; subgroup rows skipped.')
        continue
    for pf in preds_glob:
        preds = pd.read_parquet(pf)
        key = 'image_path' if ds == 'nih' else 'subject_id'
        right_on = 'image' if ds == 'nih' else 'subject_id'
        merged = preds.merge(demo, how='left', left_on=key, right_on=right_on)
        if merged['sex'].isna().all(): continue
        merged['age_tertile'] = pd.qcut(merged['age'], 3,
            labels=['young', 'mid', 'old'], duplicates='drop')
        for grp_col in ('sex', 'age_tertile'):
            for grp, sub in merged.groupby(grp_col, observed=True):
                if sub['y'].nunique() < 2: continue
                records.append({'dataset': ds, 'source': pf.name,
                                'grouping': grp_col, 'group': str(grp),
                                'n': len(sub), 'n_pos': int(sub['y'].sum()),
                                'auc': roc_auc_score(sub['y'], sub['y_hat'])})

out = pd.DataFrame(records)
if out.empty:
    pd.DataFrame({'status': ['inputs_absent']}).to_parquet(
        out_dir / 'exp12_manifest.parquet', index=False)
else:
    out.to_parquet(out_dir / 'exp12_subgroup_auc.parquet', index=False)
    print(f'wrote {len(out)} rows -> {out_dir / "exp12_subgroup_auc.parquet"}')
"""),
])

# ----- 13 JPEGSensitivity -------------------------------------------------
nb_13 = nb([
    md("""\
# Experiment 13 — JPEG compression sensitivity (NIH-only)

NIH radiographs are re-encoded at JPEG quality $q \\in$ {50, 70, 90, 95},
re-embedded with RAD-DINO and DINOv3, and re-probed for exp02 (geometric S4),
exp03 (iso-blur $p$=4), and exp06 (reticular-fine-$p$=32). The RAD-DINO vs
DINOv3 CLS AUC dissociation is reported as a function of $q$; stability
across $q$ rules out the possibility that the MIMIC-CXR-JPG corpus's lossy
compression confounds the embedding-level results.
"""),
    code("""\
import os, warnings
from pathlib import Path
import pandas as pd

ROOT = Path(os.environ.get('V4_WORK_DIR',
    '/home/saptpurk/embeddings-noise-eliminators/v4_work'))
QUALS = [50, 70, 90, 95]
out_dir = ROOT / 'v4_exp13_jpeg_sensitivity'
out_dir.mkdir(parents=True, exist_ok=True)

import sys; sys.path.insert(0, str(Path('..').resolve()))
try:
    from common.jpeg_sensitivity import run_jpeg_sweep
except Exception as e:
    warnings.warn(f'jpeg_sensitivity import failed: {e}')
    run_jpeg_sweep = None

if run_jpeg_sweep is None:
    pd.DataFrame({'status': ['module_absent']}).to_parquet(
        out_dir / 'exp13_manifest.parquet', index=False)
else:
    rows = run_jpeg_sweep(models=('raddino', 'dinov3'), qualities=QUALS,
        artifacts=('geometric_C1', 'geometric_S4', 'iso_blur_p4',
                   'reticular_fine_p32'),
        dataset='nih', out_dir=out_dir)
    print(f'jpeg sweep produced {len(rows)} rows')
"""),
])

# ----- 14 MultiSeed -------------------------------------------------------
nb_14 = nb([
    md("""\
# Experiment 14 — Multi-seed probe refits

Each headline linear probe (exp01, exp02 for C1/S4/S8, exp03, exp06) is
re-fit with seeds $\\in$ {0, 1, 2} using the cached embeddings from the
primary pipeline. Between-seed standard deviation quantifies probe-fitting
variability with encoder output held fixed; results are compared against
between-model effect sizes reported in the main manuscript.
"""),
    code("""\
import os, warnings
from pathlib import Path
import pandas as pd

ROOT = Path(os.environ.get('V4_WORK_DIR',
    '/home/saptpurk/embeddings-noise-eliminators/v4_work'))
SEEDS = [0, 1, 2]
out_dir = ROOT / 'v4_exp14_multiseed'
out_dir.mkdir(parents=True, exist_ok=True)

import sys; sys.path.insert(0, str(Path('..').resolve()))
try:
    from common.probing import refit_linear_probe_for_seed
except Exception as e:
    warnings.warn(f'refit_linear_probe_for_seed import failed: {e}')
    refit_linear_probe_for_seed = None

rows = []
if refit_linear_probe_for_seed is not None:
    for exp in ('exp01', 'exp02', 'exp03', 'exp06'):
        for seed in SEEDS:
            try:
                df = refit_linear_probe_for_seed(exp=exp, seed=seed)
                df['seed'] = seed; df['exp'] = exp; rows.append(df)
            except FileNotFoundError:
                continue
if rows:
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(out_dir / 'exp14_multiseed.parquet', index=False)
    print(f'wrote {len(out)} rows -> {out_dir / "exp14_multiseed.parquet"}')
else:
    pd.DataFrame({'status': ['inputs_absent']}).to_parquet(
        out_dir / 'exp14_manifest.parquet', index=False)
"""),
])

# ----- 15 NativeResolution ------------------------------------------------
nb_15 = nb([
    md("""\
# Experiment 15 — Native-resolution ablation (RAD-DINO vs DINOv3, NIH-only)

The primary pipeline operates on a common $1024\\times1024$ input. This
experiment re-runs exp02 (geometric S4) and exp06 (reticular-fine-$p$=32)
with each model at its native input resolution: RAD-DINO at 518, DINOv3 at
224 and 518. A persistent dissociation at native resolution rules out the
confound that $1024 \\to$ native resize (with positional-embedding
interpolation) drives the observed suppression.
"""),
    code("""\
import os, warnings
from pathlib import Path
import pandas as pd

ROOT = Path(os.environ.get('V4_WORK_DIR',
    '/home/saptpurk/embeddings-noise-eliminators/v4_work'))
out_dir = ROOT / 'v4_exp15_native_resolution'
out_dir.mkdir(parents=True, exist_ok=True)

import sys; sys.path.insert(0, str(Path('..').resolve()))
try:
    from common.native_resolution import run_native_resolution_sweep
except Exception as e:
    warnings.warn(f'native_resolution import failed: {e}')
    run_native_resolution_sweep = None

if run_native_resolution_sweep is None:
    pd.DataFrame({'status': ['module_absent']}).to_parquet(
        out_dir / 'exp15_manifest.parquet', index=False)
else:
    rows = run_native_resolution_sweep(
        models=('raddino', 'dinov3'),
        resolutions_by_model={'raddino': (518,), 'dinov3': (224, 518)},
        artifacts=('geometric_S4', 'reticular_fine_p32'),
        dataset='nih', out_dir=out_dir)
    print(f'native-resolution sweep produced {len(rows)} rows')
"""),
])

# ----- 16 LabelNoiseSensitivity -------------------------------------------
nb_16 = nb([
    md("""\
# Experiment 16 — Label-noise sensitivity (MIMIC)

The MIMIC CheXpert NLP labels are imperfect. This experiment builds a
high-confidence MIMIC subset by intersecting CheXpert positives with a
radiologist-report regex per disease, then re-fits the linear probe on the
intersection. Preservation of the between-model ranking on the
high-confidence subset indicates that the reported dissociation is not a
label-noise artifact.
"""),
    code("""\
import os, warnings
from pathlib import Path
import pandas as pd

ROOT = Path(os.environ.get('V4_WORK_DIR',
    '/home/saptpurk/embeddings-noise-eliminators/v4_work'))
out_dir = ROOT / 'v4_exp16_label_noise'
out_dir.mkdir(parents=True, exist_ok=True)

import sys; sys.path.insert(0, str(Path('..').resolve()))
try:
    from common.label_noise import (build_high_confidence_subset,
                                      refit_probes_on_subset)
except Exception as e:
    warnings.warn(f'label_noise import failed: {e}')
    build_high_confidence_subset = None
    refit_probes_on_subset = None

if build_high_confidence_subset is None:
    pd.DataFrame({'status': ['module_absent']}).to_parquet(
        out_dir / 'exp16_manifest.parquet', index=False)
else:
    subset = build_high_confidence_subset(dataset='mimic')
    rows = refit_probes_on_subset(subset, exps=('exp01', 'exp02', 'exp06'))
    out = pd.DataFrame(rows)
    out.to_parquet(out_dir / 'exp16_label_noise.parquet', index=False)
    print(f'wrote {len(out)} rows -> {out_dir / "exp16_label_noise.parquet"}')
"""),
])

# ----- 17 ADI_vs_DownstreamDelta ------------------------------------------
nb_17 = nb([
    md("""\
# Experiment 17 — ADI as a predictor of downstream $\\Delta$AUC

The Artifact Detectability Index (ADI, geometric mean of per-artifact
linear-probe CLS AUCs on the standardized battery $\\mathcal{B}_\\text{CXR}$)
is validated against the mean absolute downstream $\\Delta$AUC from exp04.
Per-(dataset, model) ADI values are correlated with per-(dataset, model)
mean $\\vert\\Delta$AUC$\\vert$ using Spearman rank correlation; 95% CI is
obtained by 2000-iteration bootstrap over (dataset, model) pairs. A strong
positive correlation evidences that ADI tracks downstream disease
classification perturbation magnitude.
"""),
    code("""\
import os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(os.environ.get('V4_WORK_DIR',
    '/home/saptpurk/embeddings-noise-eliminators/v4_work'))
out_dir = ROOT / 'v4_exp17_adi_vs_delta'
out_dir.mkdir(parents=True, exist_ok=True)

def _gmean(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return float(np.exp(np.log(x).mean())) if len(x) else np.nan

artifact_dfs = []
for pat, exp in [('v4_exp02_geometric_*/*.parquet', 'exp02'),
                 ('v4_exp03_iso_motion_blur_*/*.parquet', 'exp03'),
                 ('v4_exp06_patch_probing_*/*.parquet', 'exp06'),
                 ('v4_exp08_directional_blur_*/*.parquet', 'exp08')]:
    for f in sorted(ROOT.glob(pat)):
        d = pd.read_parquet(f); d['exp'] = exp
        if 'dataset' not in d.columns:
            d['dataset'] = f.parent.name.split('_')[-1]
        if exp == 'exp06':
            d = d[d.get('pooling', 'cls') == 'cls']
        artifact_dfs.append(d)

if not artifact_dfs:
    pd.DataFrame({'status': ['inputs_absent']}).to_parquet(
        out_dir / 'exp17_manifest.parquet', index=False)
else:
    arts = pd.concat(artifact_dfs, ignore_index=True)
    adi = (arts.groupby(['dataset', 'model'])['auc']
                .apply(_gmean).reset_index(name='adi'))

    ds = []
    for f in sorted(ROOT.glob('v4_exp04_clean_vs_perturbed_*/*.parquet')):
        ds.append(pd.read_parquet(f))
    if not ds:
        pd.DataFrame({'status': ['exp04_absent']}).to_parquet(
            out_dir / 'exp17_manifest.parquet', index=False)
    else:
        e4 = pd.concat(ds, ignore_index=True)
        delta = (e4.assign(abs_delta=e4['delta_auc'].abs())
                   .groupby(['dataset', 'model'])['abs_delta']
                   .mean().reset_index(name='mean_abs_delta_auc'))
        merged = adi.merge(delta, on=['dataset', 'model'], how='inner')

        if len(merged) >= 3:
            rho, p = stats.spearmanr(merged['adi'], merged['mean_abs_delta_auc'])
            rng = np.random.default_rng(42); boots = []
            for _ in range(2000):
                idx = rng.choice(len(merged), len(merged), replace=True)
                if merged.iloc[idx]['adi'].nunique() < 2: continue
                boots.append(stats.spearmanr(merged.iloc[idx]['adi'],
                    merged.iloc[idx]['mean_abs_delta_auc']).correlation)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            summary = pd.DataFrame([{'spearman_rho': rho, 'spearman_p': p,
                'ci_low': float(lo), 'ci_high': float(hi),
                'n_points': len(merged)}])
        else:
            summary = pd.DataFrame([{'spearman_rho': np.nan,
                'spearman_p': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
                'n_points': len(merged)}])

        merged.to_parquet(out_dir / 'exp17_adi_vs_delta_points.parquet', index=False)
        summary.to_parquet(out_dir / 'exp17_adi_vs_delta_summary.parquet', index=False)
        print(summary.to_string(index=False))
"""),
])

# ----- 18 PatchFootprintTable ---------------------------------------------
nb_18 = nb([
    md("""\
# Experiment 18 — ViT patch-footprint ratio table

For every (foundation model, perturbation patch size $P$) pair, the number
of ViT tokens covered by one perturbation patch at the pipeline's operating
input resolution ($1024\\times1024$) is tabulated. Each model's native
patch size and native input resolution determine the effective token size
at $1024\\times1024$ after positional-embedding interpolation. The table
allows the suppression-ordering dissociation to be read against the
token-coverage ordering, ruling out the patch-footprint confound.
"""),
    code("""\
import os
from pathlib import Path
import pandas as pd

ROOT = Path(os.environ.get('V4_WORK_DIR',
    '/home/saptpurk/embeddings-noise-eliminators/v4_work'))

FM_SPEC = {
    'raddino':    {'native_res': 518, 'patch': 14, 'grid_native': 518 // 14},
    'dinov2':     {'native_res': 518, 'patch': 14, 'grid_native': 518 // 14},
    'dinov3':     {'native_res': 224, 'patch': 16, 'grid_native': 224 // 16},
    'biomedclip': {'native_res': 224, 'patch': 16, 'grid_native': 224 // 16},
    'medsiglip':  {'native_res': 448, 'patch': 14, 'grid_native': 448 // 14},
}
PIPE_RES = 1024
PATCH_SIZES = [4, 8, 16, 32, 64]

rows = []
for m, spec in FM_SPEC.items():
    tok_px = PIPE_RES / spec['grid_native']
    for P in PATCH_SIZES:
        n_tok = (max(P, tok_px) ** 2) / (tok_px ** 2)
        rows.append({'model': m, 'native_res': spec['native_res'],
                     'native_patch': spec['patch'],
                     'grid_at_pipeline': round(PIPE_RES / tok_px, 1),
                     'effective_token_px_at_1024': round(tok_px, 1),
                     'artifact_patch_px': P,
                     'tokens_covered': round(n_tok, 2)})
out = pd.DataFrame(rows)
out_dir = ROOT / 'v4_exp18_patch_footprint'
out_dir.mkdir(parents=True, exist_ok=True)
out.to_parquet(out_dir / 'exp18_patch_footprint.parquet', index=False)
print(out.pivot(index='model', columns='artifact_patch_px',
                values='tokens_covered').round(2))
"""),
])

files = {
    '10_MLPProbe_CLS.ipynb': nb_10,
    '11_ClusteredBootstrap.ipynb': nb_11,
    '12_DemographicSubgroups.ipynb': nb_12,
    '13_JPEGSensitivity.ipynb': nb_13,
    '14_MultiSeed.ipynb': nb_14,
    '15_NativeResolution.ipynb': nb_15,
    '16_LabelNoiseSensitivity.ipynb': nb_16,
    '17_ADI_vs_DownstreamDelta.ipynb': nb_17,
    '18_PatchFootprintTable.ipynb': nb_18,
}
for fn, obj in files.items():
    (NB_DIR / fn).write_text(json.dumps(obj, indent=1))
    print(f'wrote {NB_DIR / fn}')
