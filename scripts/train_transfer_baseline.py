"""Transfer-learning baseline (baseline B): ResNet50 (ImageNet weights).

The custom CNN in ``train_model.py`` learns from scratch on ~30k small
(100x100) galaxy images. Standard practice in the galaxy morphology
literature is to start from a backbone pretrained on ImageNet - this
script does exactly that on the *same* v2 splits, so the two baselines
differ only in initialization/features and are directly comparable.

Recipe
------
1. Freeze the ResNet50 trunk, train the classification head (3 epochs).
2. Unfreeze and fine-tune the whole network at a low LR with early
   stopping on the untouched val split.

Run on a GPU (Colab/Kaggle) for reasonable runtime; on CPU it works but
is slow.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import applications, callbacks, layers, models

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import BASE_DIR, PROCESSED_DIR  # noqa: E402


def build_model(input_shape=(100, 100, 3)):
    base = applications.ResNet50(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid", name="probability")(x)
    model = models.Model(inputs=base.input, outputs=out)
    model.summary(print_fn=lambda s: logging.info(s))
    return model


def train(
    balanced_path=None,
    splits_path=None,
    model_path=None,
    input_shape=(100, 100, 3),
    head_epochs=3,
    fine_tune_epochs=17,
    head_lr=1e-3,
    ft_lr=1e-4,
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
    model_path = model_path or os.path.join(
        BASE_DIR, "models", "galaxy_classifier_resnet50.keras"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    tb_dir = os.path.join(log_dir, "tensorboard", f"resnet50_{timestamp}")

    # ResNet50 ImageNet weights expect 0-255 uint8 input to
    # preprocess_input; feed that directly (saves a conversion pass).
    with np.load(balanced_path) as data:
        X_train, y_train = data["images"], data["labels"]
    with np.load(splits_path) as data:
        X_val, y_val = data["X_val"], data["y_val"]

    X_train = applications.resnet50.preprocess_input(X_train.astype("float32"))
    X_val = applications.resnet50.preprocess_input(X_val.astype("float32"))

    tf.keras.utils.set_random_seed(seed)
    model = build_model(input_shape)

    # Phase 1: head only
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=head_lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=head_epochs,
        batch_size=batch_size,
        callbacks=[callbacks.TensorBoard(log_dir=tb_dir, histogram_freq=1)],
        verbose=1,
    )

    # Phase 2: fine-tune everything at a low LR
    model.layers[0].trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=ft_lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=fine_tune_epochs,
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

    with open(os.path.join(log_dir, f"training_history_resnet50_{timestamp}.json"), "w") as f:
        json.dump(history.history, f, indent=4)
    logging.info(f"ResNet50 training complete. Best model at {model_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the ResNet50 transfer baseline")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--head-epochs", type=int, default=3)
    parser.add_argument("--fine-tune-epochs", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(BASE_DIR, "logs", f"train_resnet50_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            ),
            logging.StreamHandler(),
        ],
    )
    train(
        model_path=args.model_path,
        head_epochs=args.head_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print("ResNet50 training complete.")
    print("Evaluate with: python scripts/evaluate_model.py "
          "--model-path models/galaxy_classifier_resnet50.keras "
          "--output evaluation/metrics_cnn_resnet50.json")


if __name__ == "__main__":
    main()
