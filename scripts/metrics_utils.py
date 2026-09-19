"""Shared evaluation metrics - no TensorFlow dependency.

Every baseline (custom CNN, transfer-learning CNN, classical descriptors)
is scored with the *same* functions on the *same* untouched test split,
so numbers in ``evaluation/`` are directly comparable.

Primary metrics
---------------
- ``test_balanced_accuracy`` - main headline number (robust to imbalance)
- ``macro_f1``               - second headline number
- ``roc_auc`` / ``pr_auc``   - threshold-free ranking quality
- ``test_accuracy``          - reported for continuity with v1
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_KEYS = (
    "test_accuracy",
    "test_balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "pr_auc",
)


def compute_metrics(y_true, y_prob, class_names=("Smooth", "Disk/Feature"), threshold=0.5):
    """Full metric set for a binary classifier given soft probabilities."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision_score(y_true, y_pred, pos_label=i, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, pos_label=i, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, pos_label=i, zero_division=0)),
            "support": int((y_true == i).sum()),
        }

    both_classes = len(np.unique(y_true)) == 2
    return {
        "n_test": int(len(y_true)),
        "threshold": float(threshold),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both_classes else None,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if both_classes else None,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=list(class_names), digits=4
        ),
    }


def bootstrap_ci(y_true, y_prob, n_boot=1000, seed=42, threshold=0.5):
    """Non-parametric 95% CIs by resampling the test set with replacement.

    Resamples that collapse to a single class yield ``nan`` for the
    AUC-based metrics and are ignored (``nanpercentile``).
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    n = len(y_true)
    rng = np.random.default_rng(seed)

    vals = {k: np.full(n_boot, np.nan) for k in METRIC_KEYS}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            m = compute_metrics(y_true[idx], y_prob[idx], threshold=threshold)
        except Exception:
            continue
        for k in METRIC_KEYS:
            vals[k][b] = m[k]

    ci = {}
    for k in METRIC_KEYS:
        v = vals[k]
        lo, hi = np.nanpercentile(v, [2.5, 97.5])
        ci[k] = {"low": round(float(lo), 4), "high": round(float(hi), 4)}
    return ci
