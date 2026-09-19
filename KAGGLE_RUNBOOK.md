# Kaggle runbook — Galaxy Morphology v2

Exact steps to take the v2 pipeline from "repo on your laptop" to
"filled-in baseline comparison table". Target: **~15 min of clicks +
~2–3 h of Kaggle compute** (mostly unattended).

## 0. What you have, and what v2 needs

| Your local file | v2 needs it? |
|---|---|
| `{asset_id}.jpg` images (~156k) | ✅ **yes** — the only heavy thing to upload |
| `filtered_labels.csv` | ✅ already **in the repo** (`data/processed/`) — nothing to do |
| `galaxy_dataset_balanced_100x100.npz` | ❌ **v1 artifact — do NOT upload or use.** v2 rebuilds the balanced set from the images with the leak-free split |

> ⚠️ If your local `filtered_labels.csv` differs from the committed one,
> overwrite the repo's with yours and re-zip (they should be identical:
> 156,219 rows).

## 1. Make the upload (on your laptop, ~10 min)

Use `make_upload_zips.py` from the repo root (on this branch):

```bash
python make_upload_zips.py <path-to-repo-folder> <path-to-your-images-folder>
# e.g.  python make_upload_zips.py . .\images_on_my_drive
```

It produces `./upload/`:

- `repo.zip` — the project (skips `.git`, venvs, big `.npz` files)
- `images_a.zip`, `images_b.zip` — the JPEGs, split in two
  (Kaggle's per-file upload limit is ~2 GB)

(If you can't pull this branch yet: the script is committed at
`make_upload_zips.py` in the repo — copy it anywhere and run it there.)

## 2. Create the Kaggle dataset (5 min)

1. kaggle.com → **Datasets** → **New Dataset**.
2. Name: `galaxy-morph-v2` — description whatever — **Private**.
3. Upload `repo.zip`, `images_a.zip`, `images_b.zip` (three files).
4. Finish. (Each file must be < 2 GB — that's why the images are split.)

## 3. Create the notebook (5 min)

1. **Notebooks** → **New Notebook**.
2. Title `galaxy-morph-v2-run`.
3. **Add Input** → pick the `galaxy-morph-v2` dataset.
4. Settings → **GPU (T4)** accelerator. (The CPU stages don't need it, but
   the run is long; the whole thing must fit one 1-hour GPU session — see
   below. If you prefer, use a CPU notebook for Stage 3 only, then a GPU
   notebook for the rest — see "Splitting the run".)
5. Upload `kaggle_run.ipynb` from the repo root (Notebook → **Upload
   notebook**, or just copy-paste its cells — all 19 of them, in order).
6. Run top to bottom.

## 4. What runs, and expected durations

| Stage | Cell | What it does | Time |
|---|---|---|---|
| 1 | 2 | copy repo to writable `/kaggle/working/project`, hardlink 156k images into `data/processed/images/`, assert count = 156,219 | ~1 min |
| 2 | 3 | install the few missing packages (TF/numpy/sklearn/xgboost already ship on Kaggle), assert GPU visible | ~1 min |
| 3 | 5–8 | `process_images` → `make_splits` → `balance_dataset` + manifest | **~35–50 min** (mostly JPEG reading) |
| 4 | 10–11 | custom CNN (Baseline A) + evaluation with bootstrap CIs | ~15–30 min |
| 5 | 13–14 | ResNet50 transfer learning (Baseline B) + evaluation | ~30–50 min |
| 6 | 16–17 | classical descriptors + XGBoost (Baseline C), comparison table, publication figures | ~15–25 min |

Worst case total ≈ 2 h of compute, but **the Kaggle session clock is
1 hour for a GPU notebook** — read the next section before hitting Run.

## 5. The 1-hour GPU limit — plan for it

Kaggle free tier: **1 h continuous** per notebook run, CPU or GPU
(CPU-only notebooks get 10 h/day, but the data-prep stages + both CNNs
in one session is the plan above).

**Recommended split (safest, ~10 extra minutes of clicks):**

- **Notebook 1 (CPU):** Stages 1–3 only. Then *Save Version* → tick
  **Save all output files to a dataset** → name `galaxy-morph-v2-prep`.
- **Notebook 2 (GPU T4):** Stages 4–6, and in Stage 1 *before* the
  training cells, run the short "resume" copy block from the notebook's
  last cell (it pulls the `.npz` files from `galaxy-morph-v2-prep`
  instead of re-doing Stage 3). Attach **both** datasets.

If Notebook 2 still approaches 1 h before the ResNet stage: save &
restart with `--backbone resnet18` (faster; the comparison table will
then read `resnet18`, which is a perfectly fine benchmark — note it in
the README).

## 6. Harvest the results

After the final stage, in the notebook's last cell you'll see the
`baseline_comparison.md` table and the list of files to keep:

```
evaluation/metrics_cnn_custom.json
evaluation/metrics_cnn_resnet50.json
evaluation/classical_baseline_metrics.json
evaluation/baseline_comparison.{md,csv}
results/figures/baseline_comparison.png
results/figures/... (all publication plots)
models/galaxy_classifier.keras, models/galaxy_classifier_resnet50.keras
logs/training_history_*.json
```

1. *Save Version* (again, all outputs to dataset) → **download** the
   dataset zip (or grab individual files from the dataset page).
2. Copy those files into your local repo at the same paths
   (`evaluation/`, `results/figures/`, `models/`, `logs/`).
3. Fill the **v2 baselines** table in `README.md` from
   `evaluation/baseline_comparison.md`.
4. `git add -A && git commit && git push` — and bring the table back
   here; the error-physics analysis (stage 2 of the roadmap) is the next
   step and it's where the project becomes a physics result.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Too few images` assert in Stage 1 | only one images zip uploaded, or filenames aren't `{asset_id}.jpg` — check `!ls /kaggle/input/galaxy-morph-v2/ \| head` |
| `No GPU visible` in Stage 2 | Settings → Accelerator → GPU (T4), then restart the session |
| `MemoryError` in `balance_dataset` | Kaggle has 20 GB — shouldn't happen; if it does, re-run just that cell (OS may have killed a transient spike) |
| Notebook killed at ~1 h | follow §5: Save Version → outputs to dataset → resume in a new notebook |
| `process_images` logs many `Failed ...` warnings | normal if a few images are missing from your download; the pipeline proceeds with what exists (check the count line) |
