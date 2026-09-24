"""
metrics_utils.py — Shared evaluation metrics (no TensorFlow dependency).
=========================================================================
Used by evaluate_model.py, classical_baseline.py, and compare_baselines.py
so every model is judged on exactly the same metric suite.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

CLASS_NAMES = ["Smooth", "Disk/Feature"]


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names=CLASS_NAMES,
    threshold: float = 0.5,
) -> dict:
    """Full classification metric suite (no TF needed).

    Parameters
    ----------
    y_true  : (N,) int   ground-truth labels {0, 1}
    y_prob  : (N,) float predicted probability of class 1
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )

    return {
        "n_test":                int(len(y_true)),
        "threshold":             float(threshold),
        "test_accuracy":         float(accuracy_score(y_true, y_pred)),
        "test_balanced_accuracy":float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1":              float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc":               float(roc_auc_score(y_true, y_prob)),
        "pr_auc":                float(average_precision_score(y_true, y_prob)),
        "confusion_matrix":      confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": report,
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    threshold: float = 0.5,
) -> dict:
    """Nonparametric 95% bootstrap confidence intervals for the main metrics.

    Returns a dict mapping metric name -> {"low": ..., "high": ..., "point": ...}
    """
    rng  = np.random.default_rng(seed)
    n    = len(y_true)
    keys = ["test_accuracy", "test_balanced_accuracy", "macro_f1", "roc_auc", "pr_auc"]
    boot = {k: [] for k in keys}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            m = compute_metrics(y_true[idx], y_prob[idx], threshold=threshold)
            for k in keys:
                boot[k].append(m[k])
        except Exception:
            for k in keys:
                boot[k].append(np.nan)

    point = compute_metrics(y_true, y_prob, threshold=threshold)
    ci = {}
    for k in keys:
        vals = np.array(boot[k])
        lo, hi = np.nanpercentile(vals, [2.5, 97.5])
        ci[k] = {"low": float(lo), "high": float(hi), "point": float(point[k])}
    return ci
