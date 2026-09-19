# 🌌 Galaxy Morphology Classification

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange.svg)](https://tensorflow.org/)
[![Tests](https://img.shields.io/badge/tests-py-green.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> A deep-learning pipeline for the automated morphological classification of
> galaxies from the **Galaxy Zoo 2 / DR7** release: *smooth* (elliptical /
> lenticular) vs. *disk / feature-rich* (spiral) — benchmarked against a
> classical photometric-morphology baseline and evaluated with confidence
> intervals on a strictly held-out test set.

---

## 📖 Abstract

Galaxy morphology encodes key information about galaxy formation and
evolutionary history. Automated classification is essential for modern and
upcoming large sky surveys (e.g. the Legacy Surveys, LSST/Rubin), where
manual labelling is infeasible.

This repository provides a complete, reproducible pipeline for binary
morphological classification on Galaxy Zoo 2:

1. **Data ingestion & filtering** — high-confidence labels from the
   *debiased* crowdsourced vote fractions (Hart et al. 2016).
2. **Rigorous experiment design** — a single seeded train/val/test split
   created *before* any resampling, so the test set is never touched until
   final evaluation (see [Pipeline v2](#-pipeline-v2-what-changed-why)).
3. **Three baselines** — a from-scratch CNN, an ImageNet-pretrained
   ResNet50 (transfer learning), and a classical
   Gini/M20/asymmetry/concentration/ellipticity + XGBoost model — all
   scored with identical metrics on the identical test split.
4. **Honest statistics** — accuracy, balanced accuracy, macro-F1,
   ROC-AUC, PR-AUC, per-class P/R/F1, with non-parametric bootstrap 95%
   confidence intervals.

---

## 🏗️ Pipeline overview

```
data/raw (GZ DR7 CSVs) + GZ2 images
        │  merge_and_filter.py     (t01 debiased > 0.8)
        ▼
data/processed/filtered_labels.csv          (156,219 galaxies: 53,620 smooth / 102,599 disk)
        │  process_images.py         (128×128 RGB)
        ▼
galaxy_dataset.npz
        │  make_splits.py            (seeded, stratified 70/15/15)
        ▼
galaxy_dataset_splits_100x100.npz + splits_manifest.json
        │  balance_dataset.py        (RandomOverSampler on TRAIN ONLY)
        ▼
galaxy_dataset_train_balanced.npz
        │
        ├──► train_model.py              (Baseline A: custom CNN)
        ├──► train_transfer_baseline.py  (Baseline B: ResNet50 + fine-tune)
        │         │
        │    evaluate_model.py ──────────┴──► evaluation/metrics_cnn_*.json
        │                                        (test split, bootstrap CIs)
        └──► classical_baseline.py ──────────► evaluation/classical_baseline_metrics.json
                                                          │
                                              compare_baselines.py
                                                          ▼
                                        evaluation/baseline_comparison.{md,csv}
                                        results/figures/baseline_comparison.png
```

| Step | Script | Role |
|---|---|---|
| 1 | `merge_and_filter.py` | Merge GZ DR7 CSVs; keep galaxies with debiased t01 probability > 0.8 |
| 2 | `process_images.py` | Map labels to JPEGs → 128×128 RGB `.npz` |
| 3 | `make_splits.py` | **Single** seeded stratified train/val/test split (70/15/15) at 100×100 |
| 4 | `balance_dataset.py` | Oversample the **training split only** (class balance 2:1 → 1:1) |
| 5 | `train_model.py` | Baseline A: 3-block VGG-style CNN from scratch |
| 6 | `train_transfer_baseline.py` | Baseline B: ResNet50 (ImageNet), frozen → fine-tuned |
| 7 | `classical_baseline.py` | Baseline C: photometric descriptors + XGBoost (no deep learning) |
| 8 | `evaluate_model.py` | Score any Keras model on the untouched test split (+ bootstrap CIs) |
| 9 | `compare_baselines.py` | Side-by-side table / CSV / figure of all baselines |
| 10 | `visualise_result.py` | Publication plots: training history, ROC/PR/calibration, error analysis |

Run the whole chain with `make pipeline` (or the individual targets in
`Makefile`).

---

## 🔬 Pipeline v2 — what changed and why

v1 (the first iteration of this project) had a well-structured pipeline but
two methodological flaws that inflate reported accuracy. Both are fixed
here, and the fix is what makes the v2 numbers *trustworthy*:

| # | v1 problem | v2 fix |
|---|---|---|
| 1 | **Resampling leakage.** `RandomOverSampler` ran on the *whole* dataset *before* the 85/15 split, so duplicated (test) images could also be training data. | Split first (`make_splits.py`), then oversample the training split only (`balance_dataset.py`). A unit test asserts every balanced row is a duplicate of a *train* row and disjoint from the test set. |
| 2 | **Optimistic model selection.** The same 15% holdout was used for early stopping *and* final evaluation — the model was effectively selected on its test set. | Separate 15% validation split for early stopping; the 15% test split is used exactly once, at evaluation. |
| 3 | Class names implied a pure taxonomy ("Elliptical/Spiral"), but the t01 *smooth* class contains lenticulars (S0). | Classes are reported as **Smooth (E/S0)** and **Disk/Feature (spiral-dominated)**. |
| 4 | Point estimates only (no error bars), single metric emphasis. | Balanced accuracy + macro-F1 as headline metrics; ROC/PR-AUC; bootstrap 95% CIs on all five. |
| 5 | Single model, no comparison to the literature. | Three baselines scored identically + a literature comparison table. |
| 6 | Tests mocked numpy itself. | Real-array tests, including the no-leakage invariant. |

> **Consequence for old numbers:** the committed `evaluation/metrics.json`
> (92.91% accuracy) is a **v1 legacy result** from the leaky pipeline.
> Regenerate all results with `make pipeline` before citing any number.

---

## 🚀 Getting started

### Prerequisites
Python 3.9–3.11. A GPU (Colab/Kaggle) is strongly recommended for the CNN
baselines; the classical baseline and everything else run on CPU.

### Installation
```bash
git clone https://github.com/yourusername/Galaxy-Morphology-Classification-using-CNNs
cd Galaxy-Morphology-Classification-using-CNNs
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r scripts/requirements.txt
```

### Data setup
1. Galaxy Zoo DR7 tabular data (includes `gz2_hart16.csv` and
   `gz2_filename_mapping.csv`) → `data/raw/`. Obtainable from the Galaxy Zoo
   website / SAO (see Hart et al. 2016).
2. Galaxy Zoo 2 image cutouts (`{asset_id}.jpg`) → `data/processed/images/`.

### Run everything
```bash
make pipeline        # or run targets individually: make splits, make train, ...
make test            # unit tests (no data or TF required)
```
Outputs:
- `evaluation/*_metrics.json` — per-baseline metrics with bootstrap CIs
- `evaluation/baseline_comparison.md|csv` — the result table
- `results/figures/` — publication plots

---

## 📊 Results

### v2 baselines (fill after first full run)

| Model | N (test) | Acc. | Bal. Acc. | Macro F1 | ROC AUC | PR AUC |
|---|---|---|---|---|---|---|
| Classical descriptors + XGBoost | | _run `make classical`_ | | | | |
| CNN (custom, from scratch) | | _run `make train evaluate`_ | | | | |
| ResNet50 (transfer learning) | | _run `make train-resnet evaluate-resnet`_ | | | | |

*(All scores carry 95% bootstrap CIs in `evaluation/baseline_comparison.md`.)*

### v1 legacy result (superseded — see leakage caveats above)

| Model | N (test) | Acc. | Macro F1 |
|---|---|---|---|
| CNN (custom), balanced-data pipeline | 30,780 | 92.91% | 92.90% |

### Literature comparison (GZ2 / related tasks)

Different papers use different class sets and label thresholds, so treat
this as an orientation, not a head-to-head benchmark:

| Study | Data / task | Reported result |
|---|---|---|
| Cheng et al. (2020) | Galaxy Zoo, 2,800 galaxies, E vs S | CNN ≈ 99% (after label correction) |
| Domínguez Sánchez et al. (2018) [1807.10406](https://arxiv.org/abs/1807.10406) | GZ2, 5 classes (round smooth / in-between / cigar / edge-on / spiral) | 95.2% overall CNN accuracy |
| Kalvankar et al. (2020) | GZ DECaLS, 7 classes | EfficientNet avg. precision ≈ 88.9% |
| Deformable-conv survey (2024) | GZ DECaLS DR9, 7 classes | 94.5% accuracy, outperforms VGG16/InceptionV4/EfficientNet |
| CvT (2024) | GZ2, 5 classes | >98% acc/precision/recall/F1 |
| Walmsley et al. (2020) | Galaxy Zoo, Bayesian CNN | accuracy + uncertainty estimates |

**Interpretation:** for a 2-class, high-confidence (t01 > 0.8) GZ2 subset,
the published envelope is ≈ 95–99%. A from-scratch CNN at ~93% (v1) is
therefore *expected* to underperform transfer learning; the v2 design
quantifies exactly how much of the gap is methodology (leakage/selection)
vs. model capacity.

---

## 🔁 Reproducibility

- **One seed everywhere** (default 42): split, resampling, training.
- `data/processed/splits_manifest.json` records the seed, split sizes, and
  per-class counts of the exact split every downstream step consumes.
- `requirements.txt` is fully pinned; Python 3.9–3.11.
- `make test` runs the test suite (no data, no GPU, no TF needed).
- Training logs (JSON history + TensorBoard) are written to `logs/`.

## 🧪 Tests

```
tests/
├── test_splits_balance.py    # split stratification, reproducibility, NO-LEAKAGE invariant
├── test_balance_dataset.py   # balancing semantics, file I/O, val/test untouched
└── test_descriptors.py       # physical sanity of Gini/M20/asymmetry/concentration/ellipticity
```
Run with `make test` (17 tests).

---

## 📁 Directory structure

```
├── data/
│   ├── raw/                  # GZ DR7 CSVs (git-ignored; see Data setup)
│   └── processed/            # labels, images, .npz splits (large files git-ignored)
├── evaluation/               # per-baseline metrics JSON + comparison table
├── logs/                     # training history, TensorBoard
├── models/                   # saved .keras models
├── results/figures/          # publication plots
├── scripts/                  # pipeline (see table above) + requirements.txt
├── tests/                    # pytest suite
├── Makefile                  # make pipeline / make test
└── README.md
```

---

## ⚠️ Known limitations

- **Binary, low-resolution task.** 100×100 crops and a 2-class split throw
  away information (lenticulars vs. ellipticals, barred vs. unbarred,
  mergers, irregulars) that the GZ2 question set actually supports.
- **No foreground-star / multiple / artifact filtering** beyond the t01
  confidence cut. GZ2 has dedicated questions (stars, multiples,
  artificiality) that a survey-grade pipeline would use.
- **CNN baselines only, not yet re-trained under v2** in this commit — the
  v2 numbers table above is populated by running `make pipeline`.
- **No uncertainty quantification** on individual predictions yet (see
  Future work); only statistical CIs on aggregate metrics.

## 🔮 Future work (in order of leverage)

1. **Multi-class taxonomy** — round smooth / in-between / cigar / edge-on /
   spiral (the 5-class GZ2 task used by the literature above), plus merger
   and irregular splits from the remaining GZ2 questions.
2. **Uncertainty quantification** — MC-dropout or deep ensembles; report a
   *confidence-vs-precision* (a.k.a. trigger-efficiency) curve, so the
   classifier can be "abstained" at a user-chosen reliability level.
   (cf. Walmsley et al. 2020.)
3. **Calibration & error physics** — is predicted "smoothness" a smooth
   function of the GZ2 debiased fraction? Do errors correlate with size,
   surface brightness, axis ratio, or redshift? This is where the project
   becomes a *physics* result rather than a ML benchmark.
4. **Interpretability** — Grad-CAM / SHAP overlays to show *where* in the
   image each model looks.
5. **Sersic regression** — instead of (or alongside) classification,
   regress Sersic `n` and `R_e` directly and map the classifier's errors
   in profile space.
6. **Survey-scale validation** — apply the best model to an independent
   DR with no GZ2 overlap (e.g. GZ DECaLS) to test generalization beyond
   the SDSS stack.

---

## 📚 References

- Lintott et al. (2008), *MNRAS 389, 1179* — Galaxy Zoo: the morphology of
  300,000 galaxies.
- Hart et al. (2016), *MNRAS 462, 2847* — Galaxy Zoo: DR7 (the label
  source used here).
- Lotz, Primack & Madau (2008), *MNRAS 384, 117* — Gini coefficient and
  M20 for morphology (Baseline C descriptors).
- Conselice et al. (2000), *ApJ 529, 582* — asymmetry, concentration and
  position-ellipticity indices (Baseline C descriptors).
- Domínguez Sánchez et al. (2018), *arXiv:1807.10406* — CNN morphology on
  GZ2 (5-class).
- Walmsley, Bickel & Cohn (2020), *MNRAS 499, 2076* — Bayesian CNNs with
  uncertainty for galaxy morphology.
- Kalvankar et al. (2020) — EfficientNet on GZ DECaLS (7-class).
- "Galaxy Morphology Classification via Deep Semi-Supervised Learning"
  (2025), *arXiv:2504.00500* — recent GZ2 benchmark incl. calibration (ECE).

---

## 👨‍🔬 Author

**Aryan Bandyopadhyay**
Integrated MSc Physics
School of Physical Sciences
NISER Bhubaneswar
