"""
train_model.py — Train the custom VGG-style CNN (Baseline A).
=============================================================
v2 changes
----------
* Loads the leak-free split (make_splits.py) + balanced train (balance_dataset.py)
* Validation set = the *held-out* val split, not the same 15% used for testing
* Model saved to  models/galaxy_classifier.keras
* History JSON saved to logs/
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json, logging
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils import BASE_DIR, PROCESSED_DIR, LOG_DIR

# paths
SPLITS_PATH   = os.path.join(PROCESSED_DIR, "galaxy_dataset_splits_100x100.npz")
BALANCED_PATH = os.path.join(PROCESSED_DIR, "galaxy_dataset_train_balanced.npz")
MODEL_PATH    = os.path.join(BASE_DIR, "models", "galaxy_classifier.keras")

CONFIG = {
    "batch_size":   32,
    "epochs":       30,
    "learning_rate":1e-4,
    "patience":     5,
    "seed":         42,
}
CLASS_NAMES = ["Smooth", "Disk/Feature"]


def build_model(input_shape=(100, 100, 3)) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ], name="galaxy_cnn_v2")
    return model


def train() -> bool:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(LOG_DIR, f"train_{datetime.now():%Y%m%d_%H%M}.log")
            ),
        ],
    )

    tf.random.set_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    logging.info("Loading balanced training data ...")
    bal   = np.load(BALANCED_PATH)
    X_tr, y_tr = bal["images"], bal["labels"]

    logging.info("Loading val split ...")
    splits = np.load(SPLITS_PATH)
    X_val, y_val = splits["X_val"], splits["y_val"]

    logging.info(f"  Train: {len(X_tr)}  Val: {len(X_val)}")

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(CONFIG["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=logging.info)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=CONFIG["patience"],
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(BASE_DIR, "logs", "tensorboard"), histogram_freq=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
    ]

    history = model.fit(
        X_tr, y_tr,
        batch_size=CONFIG["batch_size"],
        epochs=CONFIG["epochs"],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )

    hist_path = os.path.join(
        LOG_DIR, f"training_history_{datetime.now():%Y%m%d_%H%M}.json"
    )
    with open(hist_path, "w") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in history.history.items()}, f
        )
    logging.info(f"History -> {hist_path}")
    logging.info(f"Model   -> {MODEL_PATH}")
    return True


if __name__ == "__main__":
    ok = train()
    print("Training complete." if ok else "Training failed.")
