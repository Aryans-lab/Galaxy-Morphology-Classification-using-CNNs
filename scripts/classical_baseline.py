"""Classical photometric-morphology baseline (no deep learning).

This is the standard "pre-CNN" approach in the galaxy morphology
literature and the baseline any CNN should be compared against:

- Gini coefficient & M20          (Lotz, Primack & Madau 2008)
- Asymmetry index R               (Conselice et al. 2000)
- Concentration index r90/r50     (Conselice et al. 2000)
- Ellipticity from 2nd moments
- [optional] Sersic fit n, R_e, SBC (photutils)

The descriptors are extracted from the *same* train/val/test splits as
the CNN baselines, trained with XGBoost (no resampling - the imbalance
is handled with ``scale_pos_weight``), and scored with the same
``metrics_utils`` functions. Output: ``evaluation/classical_baseline_metrics.json``.

Usage
-----
    python scripts/classical_baseline.py                  # descriptors only
    python scripts/classical_baseline.py --with-sersic    # + Sersic fits (slow)
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_utils import bootstrap_ci, compute_metrics  # noqa: E402
from utils import BASE_DIR, PROCESSED_DIR  # noqa: E402

CLASS_NAMES = ["Smooth", "Disk/Feature"]
FEATURE_NAMES = ["gini", "m20", "asymmetry", "concentration", "ellipticity"]
SERSIC_NAMES = ["sersic_n", "sersic_re", "sersic_sbc"]


# ---------------------------------------------------------------------------
# Descriptor extraction (vectorized, chunked)
# ---------------------------------------------------------------------------
def to_luminance(X, chunk=2000):
    """(N, H, W, 3) uint8 -> (N, H, W) float32 luminance, in chunks."""
    n = len(X)
    out = np.empty((n,) + X.shape[1:-1], dtype=np.float32)
    for i in range(0, n, chunk):
        b = np.asarray(X[i : i + chunk], dtype=np.float32)
        out[i : i + len(b)] = 0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]
    return out


def _sort_flux(f):
    """Sort per-image flux vectors ascending: f (B, N) -> (sorted_f,)."""
    order = np.argsort(f, axis=1, kind="stable")
    return np.take_along_axis(f, order, axis=1)


def gini_m20(f2, chunk=1000):
    """Lotz et al. (2008) Gini and M20, computed per image in chunks.

    Gini in [0, 1]: ~1 = mass concentrated in a few bright pixels
    (early-type), ~0 = diffuse. M20 = log10(frac of flux in the
    20% least-luminous pixels): very negative for point-like sources.
    """
    n = f2.shape[0]
    flat = f2.reshape(n, -1)
    N = flat.shape[1]
    gini = np.empty(n)
    m20 = np.empty(n)
    n20 = max(1, int(0.2 * N))
    idx = np.arange(1, N + 1, dtype=np.float64)

    for i in range(0, n, chunk):
        fs = _sort_flux(flat[i : i + chunk])
        s = fs.sum(axis=1, keepdims=True)
        gini[i : i + chunk] = (
            (2.0 * (idx * fs).sum(axis=1) - (N + 1) * s[:, 0]) / ((N - 1) * s[:, 0])
        )
        m20[i : i + chunk] = np.log10(
            np.maximum(fs[:, :n20].sum(axis=1), 1e-12) / s[:, 0]
        )
    return gini, m20


def asymmetry(f2):
    """Conselice et al. (2000) asymmetry R about the image centre:
    R = sum |I - I_rot180| / (2 sum I). 0 = perfectly symmetric."""
    fr = f2[:, ::-1, ::-1]
    num = np.abs(f2 - fr).sum(axis=(1, 2))
    den = 2.0 * f2.sum(axis=(1, 2))
    return num / np.maximum(den, 1e-12)


def concentration(f2, chunk=1000):
    """Concentration index r90/r50 from the radial profile about the
    centroid (larger = more extended/feature-rich)."""
    n = f2.shape[0]
    H, W = f2.shape[1:]
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    ys, xs = ys[None], xs[None]

    out = np.empty(n)
    for i in range(0, n, chunk):
        b = f2[i : i + chunk]
        tot = b.sum(axis=(1, 2), keepdims=True)
        tot_safe = np.maximum(tot, 1e-12)
        # keepdims keeps shapes (chunk, 1, 1) so broadcasting is safe for
        # any chunk size
        xc = (b * xs).sum(axis=(1, 2), keepdims=True) / tot_safe
        yc = (b * ys).sum(axis=(1, 2), keepdims=True) / tot_safe

        r = np.sqrt((xs - xc) ** 2 + (ys - yc) ** 2).reshape(len(b), -1)
        f = b.reshape(len(b), -1)
        order = np.argsort(r, axis=1, kind="stable")
        rf = np.take_along_axis(f, order, axis=1)
        rr = np.take_along_axis(r, order, axis=1)
        csum = np.cumsum(rf, axis=1)  # monotonic per row
        total = csum[:, -1:]  # (chunk, 1) so it broadcasts row-wise
        # first index where the cumulative profile crosses the threshold
        # (csum is monotonic, so argmax of the boolean mask == searchsorted)
        r50 = rr[np.arange(len(b)), np.argmax(csum >= 0.50 * total, axis=1)]
        r90 = rr[np.arange(len(b)), np.argmax(csum >= 0.90 * total, axis=1)]
        out[i : i + chunk] = r90 / np.maximum(r50, 1e-6)
    return out


def ellipticity(f2):
    """Position-ellipticity-like measure from flux-weighted 2nd moments:
    e = |Ixx - Iyy| / sqrt((Ixx - Iyy)^2 + 4 Ixy^2). 0 = circular."""
    H, W = f2.shape[1:]
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    ys, xs = ys[None], xs[None]

    tot = f2.sum(axis=(1, 2), keepdims=True)
    tot_safe = np.maximum(tot, 1e-12)
    xc = (f2 * xs).sum(axis=(1, 2), keepdims=True) / tot_safe
    yc = (f2 * ys).sum(axis=(1, 2), keepdims=True) / tot_safe

    Ixx = (f2 * (xs - xc) ** 2).sum(axis=(1, 2), keepdims=True) / tot_safe
    Iyy = (f2 * (ys - yc) ** 2).sum(axis=(1, 2), keepdims=True) / tot_safe
    Ixy = (f2 * (xs - xc) * (ys - yc)).sum(axis=(1, 2), keepdims=True) / tot_safe

    num = Ixx - Iyy
    den = np.sqrt(num ** 2 + 4.0 * Ixy ** 2)
    return (np.abs(num) / np.maximum(den, 1e-12)).reshape(-1)


def sersic_features(f2, max_iter=200, progress=True):
    """Optional 2D Sersic fit (photutils) per image. SLOW (~tens of ms per
    image) - keep behind the --with-sersic flag. Returns (n, R_e, SBC)."""
    from photutils.model import Sersic2D
    from photutils.modeling import fit_2d_model

    n = f2.shape[0]
    H, W = f2.shape[1:]
    cy, cx = H / 2.0, W / 2.0
    out_n = np.full(n, np.nan)
    out_re = np.full(n, np.nan)
    out_sbc = np.full(n, np.nan)

    for i in range(n):
        img = f2[i]
        if img.max() < 10:  # empty/black frame
            continue
        init = Sersic2D(
            amplitude=float(img.max()),
            x0=cx, y0=cy,
            ellip=0.2, phi=0.0,
            n=2.5,
            r_e=max(3.0, float(np.sqrt(img.sum() / (2.0 * np.pi * np.e ** 2.5)) + 1.0)),
        )
        try:
            fit = fit_2d_model(
                init, img,
                x_bounds=(0, W), y_bounds=(0, H),
                n_bounds=(0.3, 8.0), r_e_bounds=(2.0, W / 2.0),
                maxiter=max_iter,
            )
            out_n[i] = fit.n
            out_re[i] = fit.r_e
            out_sbc[i] = float(fit.SBC) if hasattr(fit, "SBC") else np.nan
        except Exception:
            continue
        if progress and (i + 1) % 500 == 0:
            logging.info(f"Sersic fit {i + 1}/{n}")
    return out_n, out_re, out_sbc


def extract_descriptors(X, with_sersic=False):
    """X (N, H, W, 3) uint8 -> (features NxK, feature_names)."""
    logging.info("Converting to luminance and extracting descriptors...")
    t0 = time.time()
    lum = to_luminance(X)

    gini, m20 = gini_m20(lum)
    asym = asymmetry(lum)
    conc = concentration(lum)
    ellip = ellipticity(lum)

    feats = [gini, m20, asym, conc, ellip]
    names = list(FEATURE_NAMES)

    if with_sersic:
        logging.info("Running Sersic fits (this is the slow part)...")
        sn, sre, ssbc = sersic_features(lum)
        feats += [sn, sre, ssbc]
        names += SERSIC_NAMES
        logging.info(f"Sersic fits done in {time.time() - t0:.1f}s")

    del lum
    return np.column_stack(feats).astype(np.float32), names


# ---------------------------------------------------------------------------
# Model + evaluation
# ---------------------------------------------------------------------------
def run(splits_path=None, output_path=None, with_sersic=False, seed=42,
        early_stopping_rounds=50, max_boost_round=1000):
    import xgboost as xgb

    splits_path = splits_path or os.path.join(
        PROCESSED_DIR, "galaxy_dataset_splits_100x100.npz"
    )
    output_path = output_path or os.path.join(
        BASE_DIR, "evaluation", "classical_baseline_metrics.json"
    )

    with np.load(splits_path) as data:
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        X_test, y_test = data["X_test"], data["y_test"]

    F_train, names = extract_descriptors(X_train, with_sersic=with_sersic)
    F_val, _ = extract_descriptors(X_val, with_sersic=with_sersic)
    F_test, _ = extract_descriptors(X_test, with_sersic=with_sersic)
    logging.info(f"Descriptors: {names}")

    dtrain = xgb.DMatrix(F_train, label=y_train)
    dval = xgb.DMatrix(F_val, label=y_val)
    dtest = xgb.DMatrix(F_test, label=y_test)

    n0, n1 = int((y_train == 0).sum()), int((y_train == 1).sum())
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 5,
        "scale_pos_weight": n0 / n1,  # handle imbalance WITHOUT resampling
        "seed": seed,
    }
    logging.info(f"XGBoost: {n0} smooth / {n1} disk, scale_pos_weight={n0 / n1:.3f}")

    clf = xgb.train(
        params, dtrain,
        num_boost_round=max_boost_round,
        evals=[(dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=50,
    )
    y_prob = clf.predict(dtest, iteration_range=(0, clf.best_iteration + 1))

    metrics = compute_metrics(y_test, y_prob, CLASS_NAMES)
    metrics["model"] = "classical_descriptors_xgboost"
    metrics["sersic"] = bool(with_sersic)
    metrics["best_iteration"] = int(clf.best_iteration)
    metrics["bootstrap_ci"] = bootstrap_ci(y_test, y_prob, n_boot=1000, seed=seed)
    importance = {n: float(v) for n, v in zip(names, clf.get_score().values())}
    metrics["feature_importance"] = importance

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Saved classical baseline metrics to {output_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Classical photometric-morphology baseline")
    parser.add_argument("--splits", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--with-sersic", action="store_true",
                        help="add Sersic n/R_e/SBC features (slow)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run(
        splits_path=args.splits,
        output_path=args.output,
        with_sersic=args.with_sersic,
        seed=args.seed,
    )
    print("Classical baseline complete. Output: evaluation/classical_baseline_metrics.json")


if __name__ == "__main__":
    main()
