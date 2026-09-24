"""
make_splits.py — Create the stratified train / val / test split.
================================================================
This is the ONLY place the dataset is divided, so that balancing,
augmentation, and early-stopping can NEVER contaminate the test set.

v1 pipeline flaw this fixes
----------------------------
v1 ran RandomOverSampler on the *whole* dataset and then did an 85/15
split.  Exact-duplicate oversampled images therefore appeared on both
sides of the split (leakage), and the same 15 % was used for both
early-stopping *and* final evaluation.

v2 flow
-------
process_images  →  make_splits  →  balance_dataset (train only)
→  train_model  →  evaluate_model

Outputs
-------
data/processed/galaxy_dataset_splits_100x100.npz
    keys: X_train, y_train, X_val, y_val, X_test, y_test
data/processed/splits_manifest.json
    seed, split sizes, per-class counts, timestamp
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import logging
from datetime import datetime

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import BASE_DIR, PROCESSED_DIR, LOG_DIR

# ---------------------------------------------------------------------------
CLASS_NAMES = ["Smooth", "Disk/Feature"]
DEFAULT_INPUT  = os.path.join(PROCESSED_DIR, "galaxy_dataset.npz")
DEFAULT_OUTPUT = os.path.join(PROCESSED_DIR, "galaxy_dataset_splits_100x100.npz")
DEFAULT_MANIFEST = os.path.join(PROCESSED_DIR, "splits_manifest.json")
# ---------------------------------------------------------------------------


def resize_dataset(X: np.ndarray, target: int = 100, batch: int = 500) -> np.ndarray:
    """Resize (N, H, W, 3) uint8 array to (N, target, target, 3) using PIL BICUBIC."""
    N = len(X)
    out = np.empty((N, target, target, 3), dtype=np.uint8)
    for i in tqdm(range(0, N, batch), desc=f"Resizing -> {target}x{target}"):
        chunk = X[i : i + batch]
        for j, img in enumerate(chunk):
            pil = Image.fromarray(img).resize((target, target), Image.BICUBIC)
            out[i + j] = np.asarray(pil, dtype=np.uint8)
    return out


def make_split_indices(
    y: np.ndarray,
    seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
):
    """Return (train_idx, val_idx, test_idx) — stratified, disjoint."""
    n = len(y)
    all_idx = np.arange(n)

    rest_frac = val_frac + test_frac
    train_idx, rest_idx = train_test_split(
        all_idx, test_size=rest_frac, stratify=y, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=test_frac / rest_frac,
        stratify=y[rest_idx],
        random_state=seed,
    )
    return train_idx, val_idx, test_idx


def run(
    input_path: str = DEFAULT_INPUT,
    output_path: str = DEFAULT_OUTPUT,
    manifest_path: str = DEFAULT_MANIFEST,
    target_size: int = 100,
    seed: int = 42,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> bool:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(LOG_DIR, f"make_splits_{datetime.now():%Y%m%d}.log")
            ),
        ],
    )

    logging.info(f"Loading dataset from {input_path}")
    data = np.load(input_path)
    X, y = data["images"], data["labels"]
    logging.info(f"  Loaded {len(X)} images, shape {X.shape}, labels {np.bincount(y)}")

    # ------------------------------------------------------------------
    # 1. Resize (if needed)
    # ------------------------------------------------------------------
    if X.shape[1] != target_size or X.shape[2] != target_size:
        logging.info(f"Resizing {X.shape[1]}x{X.shape[2]} -> {target_size}x{target_size}")
        X = resize_dataset(X, target=target_size)

    # ------------------------------------------------------------------
    # 2. Split
    # ------------------------------------------------------------------
    logging.info(f"Splitting  70 / {val_frac*100:.0f} / {test_frac*100:.0f}  (seed={seed})")
    train_idx, val_idx, test_idx = make_split_indices(
        y, seed=seed, val_frac=val_frac, test_frac=test_frac
    )

    splits = dict(
        X_train=X[train_idx], y_train=y[train_idx],
        X_val  =X[val_idx],   y_val  =y[val_idx],
        X_test =X[test_idx],  y_test =y[test_idx],
    )

    for name, arr in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        counts = np.bincount(y[arr], minlength=2)
        logging.info(
            f"  {name:5s}: {len(arr):6d} images  "
            f"[{CLASS_NAMES[0]}={counts[0]:5d}  {CLASS_NAMES[1]}={counts[1]:5d}]"
        )

    # ------------------------------------------------------------------
    # 3. Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logging.info(f"Saving splits -> {output_path}")
    np.savez_compressed(output_path, **splits)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "target_size": target_size,
        "n_total": int(len(X)),
        "sizes": {
            "train": int(len(train_idx)),
            "val":   int(len(val_idx)),
            "test":  int(len(test_idx)),
        },
        "class_counts": {
            split: {cn: int(np.bincount(y[idx], minlength=2)[c])
                    for c, cn in enumerate(CLASS_NAMES)}
            for split, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]
        },
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logging.info(f"Manifest   -> {manifest_path}")
    return True


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create stratified train/val/test splits.")
    parser.add_argument("--input",  default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--size",  type=int, default=100, help="Target image side length")
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--val-frac",  type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    args = parser.parse_args()

    ok = run(
        input_path=args.input,
        output_path=args.output,
        manifest_path=args.manifest,
        target_size=args.size,
        seed=args.seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )
    print("Done." if ok else "Failed — check logs.")
