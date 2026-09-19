"""Train the custom VGG-style CNN (baseline A) on the v2 splits.

v2 wiring
---------
- Training data : ``data/processed/galaxy_dataset_train_balanced.npz``
  (oversampled TRAIN split only - see balance_dataset.py)
- Validation    : the untouched val split from ``galaxy_dataset_splits_100x100.npz``
  (used for early stopping; never resampled)
- Test          : the untouched test split, evaluated ONLY by
  ``evaluate_model.py``.

The architecture is intentionally the simple one from v1 - it exists as
the "naive baseline" the transfer-learning baseline (ResNet50) is compared
against.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, callbacks, layers, models

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import BASE_DIR, PROCESSED_DIR  # noqa: E402

CLASS_NAMES = ["Smooth", "Disk/Feature"]


def build_model(input_shape=(100, 100, 3)):
    """3x Conv-relu-pool blocks + Dense head (the v1 architecture)."""
    model = models.Sequential([
        Input(shape=input_shape),
        layers.Rescaling(1 / 255.0),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.summary(print_fn=lambda x: logging.info(x))
    return model


def train(
    balanced_path=None,
    splits_path=None,
    model_path=None,
    input_shape=(100, 100, 3),
    epochs=20,
    patience=5,
    batch_size=32,
    seed=42,
):
    balanced_path = balanced_path or os.path.join(
        PROCESSED_DIR, "galaxy_dataset_train_balanced.npz"
    )
    splits_path = splits_path or os.path.join(
        PROCESSED_DIR, "galaxy_dataset_splits_100x100.npz"
    )
    model_path = model_path or os.path.join(BASE_DIR, "models", "galaxy_classifier.keras")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    tb_dir = os.path.join(log_dir, "tensorboard", timestamp)

    with np.load(balanced_path) as data:
        X_train, y_train = data["images"], data["labels"]
    with np.load(splits_path) as data:
        X_val, y_val = data["X_val"], data["y_val"]

    counts = dict(zip(*np.unique(y_train, return_counts=True)))
    logging.info(f"Training on balanced set {X_train.shape} (classes {counts})")
    logging.info(f"Validating on untouched val split {X_val.shape}")

    tf.keras.utils.set_random_seed(seed)
    model = build_model(input_shape)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            callbacks.TensorBoard(log_dir=tb_dir, histogram_freq=1),
            callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True
            ),
            callbacks.ModelCheckpoint(model_path, save_best_only=True),
        ],
        verbose=1,
    )

    with open(os.path.join(log_dir, f"training_history_{timestamp}.json"), "w") as f:
        json.dump(history.history, f, indent=4)
    logging.info(f"Training complete. Best model at {model_path}")
    logging.info(f"TensorBoard logs at {tb_dir}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the custom CNN (baseline A)")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(BASE_DIR, "logs", f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            ),
            logging.StreamHandler(),
        ],
    )
    train(
        model_path=args.model_path,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print("Training complete. Evaluate with: python scripts/evaluate_model.py")


if __name__ == "__main__":
    main()
