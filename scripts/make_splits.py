"""Create the stratified train/val/test split - the ONLY place data is split.

Why this exists (v2 change)
---------------------------
The v1 pipeline ran ``RandomOverSampler`` on the *whole* dataset and only
then split 85/15. Two problems:

1. **Label leakage** - oversampled duplicates of any image could end up on
   both sides of the split, so the model was tested on near-identical
   copies of training examples.
2. **Optimistic selection** - the same 15% held-out set was used for early
   stopping *and* final evaluation, i.e. the model was effectively selected
   on its own test set.

The v2 pipeline fixes both: this script performs a single, logged,
stratified 70/15/15 split (train/val/test). Balancing is then applied to
the training split *only* (``balance_dataset.py``), early stopping uses
the validation split, and the test split is touched exactly once - at
final evaluation time.

Inputs
------
``data/processed/galaxy_dataset.npz``  - keys: images (N,128,128,3) uint8,
labels (N,) int, optionally asset_ids (N,) int.

Outputs
-------
``data/processed/galaxy_dataset_splits_100x100.npz``
    X_train, y_train, X_val, y_val, X_test, y_test  (100x100, uint8)
``data/processed/splits_manifest.json``
    seed, split sizes, per-split class counts, input file, timestamp.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
from skimage.transform import resize
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import BASE_DIR, PROCESSED_DIR  # noqa: E402

CLASS_NAMES = ["Smooth", "Disk/Feature"]  # 0 = elliptical/S0, 1 = spiral/disk


# ---------------------------------------------------------------------------
# Core (pure, testable) functions
# ---------------------------------------------------------------------------
def resize_dataset(X, target_size=(100, 100), batch_size=500, progress=True):
    """Batch-resize an (N, H, W, 3) uint8 stack to ``target_size``.

    Anti-aliased downsampling (same settings as the v1 balancing step, so
    results stay comparable).
    """
    n = len(X)
    out = np.empty((n, *target_size, X.shape[-1]), dtype=np.uint8)
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        batch = X[i:end]
        for j in range(len(batch)):
            out[i + j] = resize(
                batch[j], target_size, preserve_range=True, anti_aliasing=True
            ).astype(np.uint8)
        if progress:
            logging.info(f"Resized {end}/{n} images")
    return out


def make_split_indices(y, seed=42, val_frac=0.15, test_frac=0.15):
    """Stratified train/val/test index arrays (fractions of the *total*)."""
    y = np.asarray(y)
    n = len(y)
    rest_frac = val_frac + test_frac
    if rest_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1")

    train_idx, rest_idx = train_test_split(
        np.arange(n), test_size=rest_frac, stratify=y, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=test_frac / rest_frac,
        stratify=y[rest_idx],
        random_state=seed,
    )
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# File I/O wrapper
# ---------------------------------------------------------------------------
def run(
    input_path=None,
    output_path=None,
    manifest_path=None,
    seed=42,
    val_frac=0.15,
    test_frac=0.15,
    target_size=100,
):
    input_path = input_path or os.path.join(PROCESSED_DIR, "galaxy_dataset.npz")
    output_path = output_path or os.path.join(
        PROCESSED_DIR, f"galaxy_dataset_splits_{target_size}x{target_size}.npz"
    )
    manifest_path = manifest_path or os.path.join(PROCESSED_DIR, "splits_manifest.json")

    logging.info(f"Loading dataset from {input_path}")
    with np.load(input_path) as data:
        X = data["images"]
        y = data["labels"]
        asset_ids = data["asset_ids"] if "asset_ids" in data else None
    logging.info(f"Dataset shape: {X.shape}, class counts: "
                 f"{dict(zip(*np.unique(y, return_counts=True)))}")

    X = resize_dataset(X, target_size=(target_size, target_size))

    train_idx, val_idx, test_idx = make_split_indices(
        y, seed=seed, val_frac=val_frac, test_frac=test_frac
    )

    payload = {
        "X_train": X[train_idx], "y_train": y[train_idx],
        "X_val": X[val_idx], "y_val": y[val_idx],
        "X_test": X[test_idx], "y_test": y[test_idx],
    }
    if asset_ids is not None:
        payload.update({
            "asset_train": asset_ids[train_idx],
            "asset_val": asset_ids[val_idx],
            "asset_test": asset_ids[test_idx],
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **payload)
    logging.info(f"Saved split dataset to {output_path}")

    manifest = {
        "input_file": os.path.basename(input_path),
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "target_size": target_size,
        "created_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "counts": {
            split: {int(k): int(v) for k, v in zip(*np.unique(payload[f"y_{split}"], return_counts=True))}
            for split in ("train", "val", "test")
        },
        "sizes": {
            split: int(len(payload[f"y_{split}"])) for split in ("train", "val", "test")
        },
        "class_names": CLASS_NAMES,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
    logging.info(f"Saved split manifest to {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Create stratified train/val/test splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run(
        input_path=args.input,
        output_path=args.output,
        seed=args.seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        target_size=args.size,
    )
    print("Splits created. See data/processed/splits_manifest.json for details.")


if __name__ == "__main__":
    main()
