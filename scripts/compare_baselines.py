"""Compare all baseline results in ``evaluation/`` side by side.

Reads every ``*metrics*.json`` in ``evaluation/`` (written by
``evaluate_model.py`` and ``classical_baseline.py``) and produces:

- a printed table with point estimates + bootstrap 95% CIs
- ``evaluation/baseline_comparison.csv``
- ``evaluation/baseline_comparison.md``
- ``results/figures/baseline_comparison.png``
"""

import glob
import json
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import BASE_DIR  # noqa: E402

COLUMNS = [
    ("test_accuracy", "Accuracy"),
    ("test_balanced_accuracy", "Bal. Acc."),
    ("macro_f1", "Macro F1"),
    ("roc_auc", "ROC AUC"),
    ("pr_auc", "PR AUC"),
]


def load_results(evaluation_dir=None):
    evaluation_dir = evaluation_dir or os.path.join(BASE_DIR, "evaluation")
    files = sorted(
        glob.glob(os.path.join(evaluation_dir, "metrics*.json"))
        + glob.glob(os.path.join(evaluation_dir, "*_metrics.json"))
    )
    results = {}
    for path in files:
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logging.warning(f"Skipping {path}: {e}")
            continue
        if "model" not in data:
            logging.warning(f"Skipping {path}: no 'model' field")
            continue
        results[data["model"]] = {"path": path, **data}
    return results


def fmt(value, ci, key):
    if value is None:
        return "n/a"
    if ci and key in ci:
        c = ci[key]
        return f"{value:.4f} [{c['low']:.4f}, {c['high']:.4f}]"
    return f"{value:.4f}"


def build_table(results):
    # Sort by raw macro F1 (best first) before formatting for display.
    order = sorted(
        results.keys(),
        key=lambda n: results[n].get("macro_f1") or 0.0,
        reverse=True,
    )
    rows = []
    for name in order:
        data = results[name]
        ci = data.get("bootstrap_ci")
        row = {
            "model": name,
            "n_test": data.get("n_test", ""),
            **{key: fmt(data.get(key), ci, key) for key, _ in COLUMNS},
        }
        rows.append(row)
    return rows


def print_table(rows):
    headers = ["model", "n_test"] + [label for _, label in COLUMNS]
    table = [headers] + [
        [r["model"], str(r["n_test"])] + [r[k] for k, _ in COLUMNS] for r in rows
    ]
    widths = [max(len(str(row[i])) for row in table) for i in range(len(headers))]
    for row in table:
        print("  ".join(str(cell).ljust(w) for cell, w in zip(row, widths)))


def save_outputs(rows, results, evaluation_dir=None, figure_dir=None):
    evaluation_dir = evaluation_dir or os.path.join(BASE_DIR, "evaluation")
    figure_dir = figure_dir or os.path.join(BASE_DIR, "results", "figures")
    os.makedirs(evaluation_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    keys = ["model", "n_test"] + [k for k, _ in COLUMNS]
    with open(os.path.join(evaluation_dir, "baseline_comparison.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    md = ["| Model | N (test) | " + " | ".join(label for _, label in COLUMNS) + " |",
          "|---|---|" + "---|" * len(COLUMNS)]
    for r in rows:
        md.append(
            "| " + str(r["model"]) + " | " + str(r["n_test"]) + " | "
            + " | ".join(r[k] for k, _ in COLUMNS) + " |"
        )
    md.append("")
    md.append("CIs are non-parametric bootstrap 95% intervals "
              "(1000 resamples of the test set).")
    with open(os.path.join(evaluation_dir, "baseline_comparison.md"), "w") as f:
        f.write("\n".join(md))

    # Grouped bar chart of point estimates (CI as error bars)
    metric_keys = [k for k, _ in COLUMNS]
    names = list(results.keys())
    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.8 / max(len(names), 1)
    for i, name in enumerate(names):
        data = results[name]
        ci = data.get("bootstrap_ci") or {}
        vals = [data.get(k) or 0.0 for k in metric_keys]
        errs = [
            max(abs((data.get(k) or 0.0) - (ci.get(k) or {"low": 0.0, "high": 0.0})["low"]),
                abs((ci.get(k) or {"low": 0.0, "high": 0.0})["high"] - (data.get(k) or 0.0)))
            for k in metric_keys
        ]
        ax.bar(
            np.arange(len(metric_keys)) + i * width, vals, width,
            label=name, yerr=errs, capsize=3,
        )
    ax.set_xticks(np.arange(len(metric_keys)) + width * (len(names) - 1) / 2)
    ax.set_xticklabels([label for _, label in COLUMNS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Baseline comparison (test set, point est. + 95% bootstrap CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figure_dir, "baseline_comparison.png"), dpi=300)
    plt.close(fig)
    logging.info("Saved baseline comparison (csv/md/png) to evaluation/ and results/figures/")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    results = load_results()
    if not results:
        print("No baseline metrics found in evaluation/. Run the baselines first.")
        sys.exit(1)
    rows = build_table(results)
    print_table(rows)
    save_outputs(rows, results)


if __name__ == "__main__":
    main()
