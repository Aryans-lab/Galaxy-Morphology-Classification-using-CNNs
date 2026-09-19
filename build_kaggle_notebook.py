"""Builds kaggle_run.ipynb (the all-in-one Kaggle notebook). Run with plain python."""
import json


def py(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


cells = []

cells.append(md("""# Galaxy Morphology v2 — full pipeline on Kaggle

Runs the **v2 (leak-free)** pipeline from `scripts/`.

**Before running**, create one Kaggle dataset named **galaxy-morph-v2**
(private) containing:

| file | what it is |
|---|---|
| `repo.zip` | the whole project folder (made by `make_upload_zips.py`) |
| `images_a.zip`, `images_b.zip` | your `{asset_id}.jpg` files in two halves |

Then attach that dataset to this notebook (Add Input) and set the
accelerator to **GPU (T4)**. See `KAGGLE_RUNBOOK.md` in the repo for the
full walkthrough, including how to split the run if the 1-hour GPU limit
hits.

> Your local `galaxy_dataset_balanced_100x100.npz` is a **v1 artifact** —
> v2 rebuilds everything from the raw images, so it is not uploaded.
"""))

cells.append(py("""# ---- Stage 1: locate inputs, set up a writable project copy ----
import glob, os, shutil, subprocess, sys

os.chdir('/kaggle/working')

# 1) find the project folder (any input folder containing scripts/train_model.py)
root = None
for d in glob.glob('/kaggle/input/*/*'):
    if os.path.isfile(os.path.join(d, 'scripts', 'train_model.py')):
        root = d
        break
assert root, 'Could not find the project folder under /kaggle/input (is repo.zip uploaded?).'
print('project found at:', root)

if not os.path.exists('project'):
    shutil.copytree(root, 'project', symlinks=True)
os.chdir('project')

# 2) collect the galaxy images into data/processed/images/ (hardlinks: instant)
os.makedirs('data/processed/images', exist_ok=True)
jpgs = (glob.glob('/kaggle/input/*/*.jpg')
        + glob.glob('/kaggle/input/*/*/*.jpg'))
seen = set()
srcs = []
for p in jpgs:
    if p not in seen:
        seen.add(p)
        srcs.append(p)
print('found', len(srcs), 'images in the input datasets')
for p in srcs:
    dst = os.path.join('data/processed/images', os.path.basename(p))
    if os.path.exists(dst):
        continue
    try:
        os.link(p, dst)          # instant, no disk duplication
    except OSError:
        shutil.copy2(p, dst)
n_img = len(glob.glob('data/processed/images/*.jpg'))
print('images in place:', n_img, '(expect 156,219)')
assert n_img >= 150000, 'Too few images - check that images_a.zip + images_b.zip were both uploaded.'
"""))

cells.append(py("""# ---- Stage 2: dependencies (Kaggle already ships TF/numpy/sklearn/xgboost) ----
import importlib.util
missing = []
for pkg, mod in [('imbalanced-learn', 'imblearn'), ('scikit-image', 'skimage'),
                 ('photutils', 'photutils'), ('astropy', 'astropy'), ('tqdm', 'tqdm')]:
    if importlib.util.find_spec(mod) is None:
        missing.append(pkg)
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing], check=True)
import tensorflow as tf, numpy as np
print('TF', tf.__version__, '| numpy', np.__version__, '| GPU:', tf.config.list_physical_devices('GPU'))
assert tf.config.list_physical_devices('GPU'), 'No GPU visible - set the accelerator to GPU (T4) in the notebook menu.'
"""))

cells.append(md("""---
## Stage 3 — Data prep (CPU steps, ~35–50 min)

`process_images` is the long one (opens 156k JPEGs). Every step is
idempotent — if the session dies, restart from the top.

> **Split-run tip:** the recommended plan is to STOP after the next three
> code cells, *Save Version → Save all output files to a dataset*
> (name it `galaxy-morph-v2-prep`), and then do Stages 4–6 in a fresh GPU
> notebook (cells 10+ below). The cell right after the manifest print is
> the resume step.
"""))

cells.append(py("""!python scripts/process_images.py   # filtered_labels.csv + images -> galaxy_dataset.npz (128x128)
"""))

cells.append(py("""!python scripts/make_splits.py       # resize 128->100 + seeded 70/15/15 split + manifest
"""))

cells.append(py("""!python scripts/balance_dataset.py   # oversample the TRAIN split only (no-leakage check inside)
"""))

cells.append(py("""import json
print(open('data/processed/splits_manifest.json').read())
"""))

cells.append(md("""---
## ⏱ Checkpoint (recommended split point)

If you are continuing **in this same session**: just run the next cell and
keep going — it will print "no prep dataset attached" and do nothing.

If you are in a **fresh GPU notebook** after saving Stage 3 as
`galaxy-morph-v2-prep`: attach that dataset too, skip the three Stage-3
code cells above, and run the next cell — it restores the `.npz` files.
"""))

cells.append(py("""# ---- Resume helper: restore .npz from a saved 'prep' dataset (no-op otherwise) ----
import glob, os, shutil
saved = glob.glob('/kaggle/input/*-prep/project/data/processed/*.npz')
if saved:
    for f in saved:
        dst = os.path.join('data/processed', os.path.basename(f))
        shutil.copy2(f, dst)
    print('restored prepared arrays:')
    for f in glob.glob('data/processed/*.npz'):
        print('  ', f)
else:
    print('no prep dataset attached - continuing in the same session (nothing to do)')
"""))

cells.append(md("""---
## Stage 4 — Baseline A: custom CNN (GPU, ~15–30 min)
"""))

cells.append(py("""!python scripts/train_model.py
"""))

cells.append(py("""!python scripts/evaluate_model.py --model-path models/galaxy_classifier.keras --output evaluation/metrics_cnn_custom.json
"""))

cells.append(md("""---
## Stage 5 — Baseline B: ResNet50 transfer learning (GPU, ~30–50 min)

If the session clock is already ~30 min before this stage, restart this
stage with the faster fallback: `--backbone resnet18` (saves
`galaxy_classifier_resnet18.keras` + `metrics_cnn_resnet18.json`).
"""))

cells.append(py("""!python scripts/train_transfer_baseline.py
"""))

cells.append(py("""!python scripts/evaluate_model.py --model-path models/galaxy_classifier_resnet50.keras --output evaluation/metrics_cnn_resnet50.json
"""))

cells.append(md("""---
## Stage 6 — Baseline C: classical photometric descriptors + XGBoost (CPU, ~15–25 min)

Gini / M20 (Lotz 2008), asymmetry / concentration (Conselice 2000),
ellipticity + XGBoost. Add `--with-sersic` for Sersic-fit features
(adds ~30–60 min).
"""))

cells.append(py("""!python scripts/classical_baseline.py
"""))

cells.append(py("""!python scripts/compare_baselines.py   # reads every evaluation/*metrics*.json
!python scripts/visualise_result.py    # publication figures
"""))

cells.append(py("""# ---- Final: show the result table + everything to keep ----
print(open('evaluation/baseline_comparison.md').read())
import glob as g
print('\\n--- files to keep (Save Version, then download) ---')
for f in sorted(g.glob('evaluation/*') + g.glob('results/figures/*')
                + g.glob('models/*.keras') + g.glob('logs/training_history_*.json')):
    print(f)
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open("kaggle_run.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote kaggle_run.ipynb with", len(cells), "cells")
