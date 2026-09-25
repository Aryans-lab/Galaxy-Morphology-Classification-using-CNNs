"""Balance the TRAINING split only (v2).

Why this changed
----------------
v1 applied ``RandomOverSampler`` to the *entire* dataset before the
train/test split, so duplicate (oversampled) copies of test images could
appear in training - a form of label leakage that inflates test accuracy.

v2 flow: ``make_splits.py`` first carves out train/val/test, and this
script oversamples the training split only. Validation and test sets are
touched by nothing else.

Inputs
------
``data/processed/galaxy_dataset_splits_100x100.npz``

Outputs
-------
``data/processed/galaxy_dataset_train_balanced.npz``  - keys: images, labels
(the balanced training split; val/test remain in the splits file).
"""

import argparse
import hashlib
import logging
import os
import sys

import numpy as np
from imblearn.over_sampling import RandomOverSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import PROCESSED_DIR  # noqa: E402

CLASS_NAMES = ["Smooth", "Disk/Feature"]


# ---------------------------------------------------------------------------
# Core (pure, testable) functions
# ---------------------------------------------------------------------------
def balance_train(X_train, y_train, random_state=42):
    """Oversample the training split to equal class sizes.

    Returns (X_balanced, y_balanced). Raises ``ValueError`` for non-binary
    data so misuse fails loudly instead of silently training a broken model.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    classes = set(np.unique(y_train).tolist())
    if classes != {0, 1}:
        raise ValueError(
            f"Expected binary classification data, got classes {sorted(classes)}"
        )

    ros = RandomOverSampler(random_state=random_state)
    X_flat = X_train.reshape(len(X_train), -1)
    X_bal_flat, y_bal = ros.fit_resample(X_flat, y_train)
    return X_bal_flat.reshape(-1, *X_train.shape[1:]), y_bal


# ---------------------------------------------------------------------------
# File I/O wrapper
# ---------------------------------------------------------------------------
def balance_dataset(splits_path=None, output_path=None, random_state=42):
    splits_path = splits_path or os.path.join(
        PROCESSED_DIR, "galaxy_dataset_splits_100x100.npz"
    )
    output_path = output_path or os.path.join(
        PROCESSED_DIR, "galaxy_dataset_train_balanced.npz"
    )

    try:
        with np.load(splits_path) as data:
            X_train, y_train = data["X_train"], data["y_train"]

        before = dict(zip(*np.unique(y_train, return_counts=True)))
        logging.info(f"Training split class counts before balancing: {before}")

        X_bal, y_bal = balance_train(X_train, y_train, random_state=random_state)

        after = dict(zip(*np.unique(y_bal, return_counts=True)))
        logging.info(f"Training split class counts after balancing:  {after}")
        logging.info(f"Balanced training set shape: {X_bal.shape}")

        # Sanity check (hashed + sampled to keep memory low on big sets):
        # the balanced set must contain ONLY images that were in the
        # training split (oversampling duplicates, never new images).
        rng = np.random.default_rng(random_state)
        flat_train = X_train.reshape(len(X_train), -1)
        train_hashes = {
            hashlib.blake2b(r.tobytes(), digest_size=16).digest() for r in flat_train
        }
        n_sample = min(2000, len(X_bal))
        sample = X_bal[rng.choice(len(X_bal), size=n_sample, replace=False)]
        for r in sample.reshape(n_sample, -1):
            if hashlib.blake2b(r.tobytes(), digest_size=16).digest() not in train_hashes:
                raise RuntimeError(
                    "Balanced set contains images missing from the training split!"
                )
        logging.info(f"No-leakage check passed (hashed {len(train_hashes)} train rows, "
                     f"verified {n_sample} balanced rows)")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(output_path, images=X_bal, labels=y_bal)
        logging.info(f"Saved balanced training set to {output_path}")
        return True

    except Exception as e:
        logging.error(f"Balancing failed: {str(e)}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Oversample the training split only")
    parser.add_argument("--splits", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if balance_dataset(args.splits, args.output, args.seed):
        print("Balancing completed. Output: data/processed/galaxy_dataset_train_balanced.npz")
    else:
        print("Balancing failed - check logs")
        sys.exit(1)


if __name__ == "__main__":
    main()
