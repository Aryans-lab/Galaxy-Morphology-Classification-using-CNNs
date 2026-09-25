"""Evaluate a saved Keras model on the UNTOUCHED test split.

The test set comes from ``make_splits.py`` (seeded, logged in
``splits_manifest.json``). It was never used for early stopping, never
oversampled, and never augmented - which is what makes the reported
numbers trustworthy.

Usage
-----
    python scripts/evaluate_model.py                    # default model
    python scripts/evaluate_model.py \\
        --model-path models/galaxy_classifier_resnet50.keras \\
        --output evaluation/metrics_cnn_resnet50.json   # ResNet50 baseline
"""

import argparse
import json
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_utils import bootstrap_ci, compute_metrics  # noqa: E402
from utils import BASE_DIR, LOG_DIR, PROCESSED_DIR  # noqa: E402

CLASS_NAMES = ["Smooth", "Disk/Feature"]


def load_test_split(splits_path):
    with np.load(splits_path) as data:
        X_test, y_test = data["X_test"], data["y_test"]
    return X_test, y_test


def predict_probs(model, X_test):
    """Return calibrated-looking soft probabilities for the positive class."""
    prob = model.predict(X_test, verbose=0)
    return np.asarray(prob).reshape(-1).astype(float)


def save_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES).plot(
        cmap="Blues", ax=ax
    )
    plt.title("Test-Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)


def evaluate(model, X_test, y_test, n_boot=1000, seed=42):
    y_prob = predict_probs(model, X_test)
    metrics = compute_metrics(y_test, y_prob, CLASS_NAMES)
    metrics["n_boot"] = int(n_boot)
    metrics["bootstrap_ci"] = bootstrap_ci(y_test, y_prob, n_boot=n_boot, seed=seed)

    # Reference loss on the test set (continuity with the v1 metrics.json)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    metrics["test_loss"] = float(model.evaluate(X_test, y_test, verbose=0)[0])
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Keras model on the test split")
    parser.add_argument("--model-path", type=str,
                        default=os.path.join(BASE_DIR, "models", "galaxy_classifier.keras"))
    parser.add_argument("--splits", type=str,
                        default=os.path.join(PROCESSED_DIR, "galaxy_dataset_splits_100x100.npz"))
    parser.add_argument("--output", type=str,
                        default=os.path.join(BASE_DIR, "evaluation", "metrics.json"))
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(LOG_DIR, f"evaluate_{args.seed}.log"), filemode="a"
            ),
            logging.StreamHandler(),
        ],
    )

    model = tf.keras.models.load_model(args.model_path, compile=False)
    X_test, y_test = load_test_split(args.splits)
    logging.info(f"Loaded model {args.model_path}; test set size {len(X_test)}")

    metrics = evaluate(model, X_test, y_test, n_boot=args.n_boot, seed=args.seed)
    metrics["model"] = os.path.splitext(os.path.basename(args.model_path))[0]
    metrics["model_path"] = args.model_path
    metrics["class_names"] = CLASS_NAMES

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Saved metrics to {args.output}")

    cm_path = os.path.join(BASE_DIR, "evaluation", "confusion_matrix.png")
    save_confusion_matrix(np.array(metrics["confusion_matrix"]), cm_path)

    print(f"Model: {metrics['model']}")
    print(f"  accuracy           {metrics['test_accuracy']:.4f}")
    print(f"  balanced accuracy  {metrics['test_balanced_accuracy']:.4f}")
    print(f"  macro F1           {metrics['macro_f1']:.4f}")
    print(f"  ROC AUC            {metrics['roc_auc']:.4f}")
    print(f"  PR AUC             {metrics['pr_auc']:.4f}")
    ci = metrics["bootstrap_ci"]
    print(f"  95% CI (macro F1)  [{ci['macro_f1']['low']:.4f}, {ci['macro_f1']['high']:.4f}]")


if __name__ == "__main__":
    main()
