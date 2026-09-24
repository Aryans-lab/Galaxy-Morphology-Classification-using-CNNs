"""
train_transfer_baseline.py — ResNet50 transfer-learning baseline (Baseline B).
===============================================================================
Uses ImageNet-pretrained ResNet50 as a frozen backbone, trains a small head,
then fine-tunes the full network.  Same data split as the custom CNN so
results are directly comparable.

Run on GPU (Kaggle T4 / Colab):  python scripts/train_transfer_baseline.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json, logging, argparse
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from utils import BASE_DIR, PROCESSED_DIR, LOG_DIR

SPLITS_PATH   = os.path.join(PROCESSED_DIR, "galaxy_dataset_splits_100x100.npz")
BALANCED_PATH = os.path.join(PROCESSED_DIR, "galaxy_dataset_train_balanced.npz")
MODEL_PATH    = os.path.join(BASE_DIR, "models", "galaxy_classifier_resnet50.keras")

CONFIG = {
    "input_shape":   (100, 100, 3),
    "batch_size":    32,
    "head_epochs":   5,
    "finetune_epochs": 20,
    "head_lr":       1e-3,
    "finetune_lr":   1e-4,
    "patience":      5,
    "seed":          42,
}


def build_model(input_shape=(100, 100, 3)) -> keras.Model:
    """ResNet50 backbone + custom classification head."""
    inputs = keras.Input(shape=input_shape)
    x = preprocess_input(inputs)              # ImageNet normalisation

    base = ResNet50(include_top=False, weights="imagenet", input_tensor=x)
    base.trainable = False                    # freeze for head training

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs, outputs, name="resnet50_galaxy")


def train(args) -> bool:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(LOG_DIR, f"train_resnet_{datetime.now():%Y%m%d_%H%M}.log")
            ),
        ],
    )

    tf.random.set_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    logging.info("Loading data ...")
    bal    = np.load(args.balanced)
    splits = np.load(args.splits)
    X_tr, y_tr   = bal["images"],     bal["labels"]
    X_val, y_val = splits["X_val"],   splits["y_val"]
    logging.info(f"  Train: {len(X_tr)}   Val: {len(X_val)}")

    model = build_model(tuple(CONFIG["input_shape"]))
    model.summary(print_fn=logging.info)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=CONFIG["patience"],
        restore_best_weights=True, verbose=1
    )
    checkpoint = keras.callbacks.ModelCheckpoint(
        args.model_path, monitor="val_accuracy", save_best_only=True, verbose=1
    )

    # Phase 1: train head only
    logging.info("Phase 1 — training head (backbone frozen) ...")
    model.compile(
        optimizer=keras.optimizers.Adam(CONFIG["head_lr"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    h1 = model.fit(
        X_tr, y_tr,
        batch_size=CONFIG["batch_size"],
        epochs=CONFIG["head_epochs"],
        validation_data=(X_val, y_val),
    )

    # Phase 2: fine-tune full network
    logging.info("Phase 2 — fine-tuning full network ...")
    model.layers[1].trainable = True          # unfreeze ResNet50
    model.compile(
        optimizer=keras.optimizers.Adam(CONFIG["finetune_lr"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    h2 = model.fit(
        X_tr, y_tr,
        batch_size=CONFIG["batch_size"],
        epochs=CONFIG["finetune_epochs"],
        validation_data=(X_val, y_val),
        callbacks=[early_stop, checkpoint],
    )

    history = {
        k: [float(v) for v in (h1.history.get(k, []) + h2.history.get(k, []))]
        for k in set(list(h1.history) + list(h2.history))
    }
    hist_path = os.path.join(LOG_DIR, f"resnet_history_{datetime.now():%Y%m%d_%H%M}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f)

    logging.info(f"History -> {hist_path}")
    logging.info(f"Model   -> {args.model_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits",     default=SPLITS_PATH)
    parser.add_argument("--balanced",   default=BALANCED_PATH)
    parser.add_argument("--model-path", default=MODEL_PATH)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    ok = train(args)
    print("Done." if ok else "Failed.")
