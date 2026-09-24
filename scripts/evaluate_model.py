"""
evaluate_model.py — Evaluate any saved Keras model on the held-out test split.
===============================================================================
v2 changes
----------
* Loads X_test / y_test from the split npz (NOT by re-splitting balanced data)
* Reports: accuracy, balanced accuracy, macro F1, ROC-AUC, PR-AUC
* Adds bootstrap 95% confidence intervals (1000 resamples)
* Saves per-model JSON: evaluation/<model_stem>_metrics.json
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json, logging, argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from utils import BASE_DIR, PROCESSED_DIR, LOG_DIR
from metrics_utils import compute_metrics, bootstrap_ci, CLASS_NAMES

DEFAULT_SPLITS = os.path.join(PROCESSED_DIR, "galaxy_dataset_splits_100x100.npz")
DEFAULT_MODEL  = os.path.join(BASE_DIR, "models", "galaxy_classifier.keras")
EVAL_DIR       = os.path.join(BASE_DIR, "evaluation")


def run(model_path: str, splits_path: str, output_path: str, n_boot: int = 1000) -> bool:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(LOG_DIR, f"evaluate_{datetime.now():%Y%m%d_%H%M}.log")
            ),
        ],
    )

    logging.info(f"Loading test split from {splits_path}")
    splits = np.load(splits_path)
    X_test, y_test = splits["X_test"], splits["y_test"]
    logging.info(
        f"  Test: {len(y_test)} images  "
        f"[smooth={(y_test==0).sum()}  disk={(y_test==1).sum()}]"
    )

    logging.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    logging.info("Running inference ...")
    y_prob = model.predict(X_test, batch_size=64, verbose=1).ravel()
    test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=64, verbose=0)

    logging.info("Computing metrics ...")
    metrics = compute_metrics(y_test, y_prob, class_names=CLASS_NAMES)
    metrics["test_loss"] = float(test_loss)
    metrics["model"] = Path(model_path).stem

    logging.info(f"  Accuracy         : {metrics['test_accuracy']:.4f}")
    logging.info(f"  Balanced accuracy: {metrics['test_balanced_accuracy']:.4f}")
    logging.info(f"  Macro F1         : {metrics['macro_f1']:.4f}")
    logging.info(f"  ROC-AUC          : {metrics['roc_auc']:.4f}")
    logging.info(f"  PR-AUC           : {metrics['pr_auc']:.4f}")

    logging.info(f"Bootstrap CI (n={n_boot}) ...")
    ci = bootstrap_ci(y_test, y_prob, n_boot=n_boot)
    metrics["bootstrap_ci"] = ci
    for k, v in ci.items():
        logging.info(f"  {k}: {v['point']:.4f}  [{v['low']:.4f}, {v['high']:.4f}]")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Results -> {output_path}")
    return True


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate a saved Keras model.")
    p.add_argument("--model-path", default=DEFAULT_MODEL)
    p.add_argument("--splits",     default=DEFAULT_SPLITS)
    p.add_argument("--output",     default=None,
                   help="Output JSON (default: evaluation/<model_stem>_metrics.json)")
    p.add_argument("--n-boot",     type=int, default=1000)
    args = p.parse_args()

    stem = Path(args.model_path).stem
    out  = args.output or os.path.join(EVAL_DIR, f"{stem}_metrics.json")
    ok   = run(args.model_path, args.splits, out, args.n_boot)
    print("Done." if ok else "Failed.")
