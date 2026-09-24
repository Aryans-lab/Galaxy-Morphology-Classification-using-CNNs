"""
balance_dataset.py — Oversample the TRAINING split only.
=========================================================
v1 flaw fixed here
------------------
v1 balanced the *whole* dataset before splitting, so oversampled duplicates
could land in the test set.  v2 applies RandomOverSampler strictly to the
training split *after* make_splits.py has created the split.

Inputs
------
data/processed/galaxy_dataset_splits_100x100.npz   (from make_splits.py)

Outputs
-------
data/processed/galaxy_dataset_train_balanced.npz
    keys: images (N, 100, 100, 3) uint8,  labels (N,) int
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
from datetime import datetime
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from utils import BASE_DIR, PROCESSED_DIR, LOG_DIR

DEFAULT_SPLITS  = os.path.join(PROCESSED_DIR, "galaxy_dataset_splits_100x100.npz")
DEFAULT_OUTPUT  = os.path.join(PROCESSED_DIR, "galaxy_dataset_train_balanced.npz")


def balance_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
):
    """Oversample the minority class in X_train to match the majority.

    Returns
    -------
    X_bal : (N_balanced, H, W, C) uint8
    y_bal : (N_balanced,) int
    """
    unique = np.unique(y_train)
    if set(unique.tolist()) != {0, 1}:
        raise ValueError(
            f"Expected binary labels {{0, 1}}, got {set(unique.tolist())}"
        )

    shape = X_train.shape[1:]          # (H, W, C)
    n     = len(X_train)

    ros = RandomOverSampler(random_state=random_state)
    X_flat, y_bal = ros.fit_resample(X_train.reshape(n, -1), y_train)
    X_bal = X_flat.reshape(-1, *shape)
    return X_bal, y_bal


def run(
    splits_path: str = DEFAULT_SPLITS,
    output_path: str = DEFAULT_OUTPUT,
    random_state: int = 42,
) -> bool:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(LOG_DIR, f"balance_{datetime.now():%Y%m%d}.log")
            ),
        ],
    )

    logging.info(f"Loading splits from {splits_path}")
    data     = np.load(splits_path)
    X_train  = data["X_train"]
    y_train  = data["y_train"]
    logging.info(
        f"  Train before balancing: {len(y_train)} images  "
        f"[smooth={(y_train==0).sum()}  disk={(y_train==1).sum()}]"
    )

    X_bal, y_bal = balance_train(X_train, y_train, random_state=random_state)
    logging.info(
        f"  Train after  balancing: {len(y_bal)} images  "
        f"[smooth={(y_bal==0).sum()}  disk={(y_bal==1).sum()}]"
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, images=X_bal, labels=y_bal)
    logging.info(f"Saved -> {output_path}")
    return True


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--splits",  default=DEFAULT_SPLITS)
    p.add_argument("--output",  default=DEFAULT_OUTPUT)
    p.add_argument("--seed",    type=int, default=42)
    args = p.parse_args()
    ok = run(args.splits, args.output, args.seed)
    print("Done." if ok else "Failed.")
