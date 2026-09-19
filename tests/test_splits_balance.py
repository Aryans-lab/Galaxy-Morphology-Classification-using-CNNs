"""Real-data tests (no mocks) for the split + balance pipeline.

The central property under test: **the test set must never contain any
image the model trained on** - including oversampled duplicates.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from balance_dataset import balance_dataset, balance_train  # noqa: E402
from make_splits import make_split_indices, resize_dataset  # noqa: E402


@pytest.fixture()
def toy_dataset(tmp_path):
    """36 unique 12x12 images (2:1 imbalance), written as a .npz."""
    rng = np.random.default_rng(0)
    n = 36
    X = rng.integers(0, 256, size=(n, 12, 12, 3), dtype=np.uint8)
    y = np.array([0] * 24 + [1] * 12)
    asset_ids = np.arange(1000, 1000 + n, dtype=np.int64)

    src = tmp_path / "dataset.npz"
    np.savez_compressed(src, images=X, labels=y, asset_ids=asset_ids)
    splits = tmp_path / "splits.npz"
    balanced = tmp_path / "train_balanced.npz"
    return {"X": X, "y": y, "src": str(src), "splits": str(splits), "balanced": str(balanced)}


def test_split_indices_are_stratified_and_disjoint():
    y = np.array([0] * 24 + [1] * 12)
    train_idx, val_idx, test_idx = make_split_indices(y, seed=42)

    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    assert len(np.unique(all_idx)) == 36  # disjoint, covers all

    # 70/15/15 of 36 -> 25 / 5 / 6, stratified at 2:1
    assert len(train_idx) == 25
    assert len(val_idx) == 5
    assert len(test_idx) == 6
    for idx, (n0, n1) in zip(
        [train_idx, val_idx, test_idx], [(17, 8), (3, 2), (4, 2)]
    ):
        assert (y[idx] == 0).sum() == n0
        assert (y[idx] == 1).sum() == n1


def test_split_is_reproducible_for_a_given_seed():
    y = np.array([0] * 24 + [1] * 12)
    a = make_split_indices(y, seed=42)
    b = make_split_indices(y, seed=42)
    for x, z in zip(a, b):
        assert np.array_equal(x, z)
    c = make_split_indices(y, seed=7)
    assert any(not np.array_equal(x, z) for x, z in zip(a, c))


def test_resize_preserves_shape_and_dtype():
    X = np.random.default_rng(1).integers(0, 256, (4, 12, 12, 3), dtype=np.uint8)
    out = resize_dataset(X, target_size=(8, 8), progress=False)
    assert out.shape == (4, 8, 8, 3)
    assert out.dtype == np.uint8


def test_balance_train_oversamples_minority_only():
    rng = np.random.default_rng(2)
    X = rng.integers(0, 256, (10, 4, 4, 3), dtype=np.uint8)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 5/5 already balanced

    X_bal, y_bal = balance_train(X, y)
    assert len(X_bal) == 10
    assert (y_bal == 0).sum() == 5 and (y_bal == 1).sum() == 5

    # Imbalanced case: 4/1 -> 4 copies of minority
    y2 = np.array([0, 0, 0, 0, 1])
    X2 = X[:5]
    X_bal2, y_bal2 = balance_train(X2, y2)
    assert (y_bal2 == 0).sum() == 4 and (y_bal2 == 1).sum() == 4

    # Balanced output contains only rows from the input (duplicates, not new images)
    in_rows = {r.tobytes() for r in X2.reshape(len(X2), -1)}
    bal_rows = {r.tobytes() for r in X_bal2.reshape(len(X_bal2), -1)}
    assert bal_rows <= in_rows


def test_balance_train_rejects_non_binary():
    X = np.zeros((6, 4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="binary"):
        balance_train(X, np.array([0, 0, 1, 1, 2, 2]))


def test_end_to_end_no_leakage(tmp_path, toy_dataset):
    """Full file flow: splits -> balance. The balanced training set must be
    a pure superset-by-duplication of the TRAIN split, and every balanced
    row must be absent from the test split."""
    import make_splits

    manifest = make_splits.run(
        input_path=toy_dataset["src"],
        output_path=toy_dataset["splits"],
        manifest_path=str(tmp_path / "manifest.json"),
        seed=42,
        val_frac=0.15,
        test_frac=0.15,
        target_size=8,
    )
    assert manifest["sizes"] == {"train": 25, "val": 5, "test": 6}

    assert balance_dataset(
        splits_path=toy_dataset["splits"],
        output_path=toy_dataset["balanced"],
    )

    with np.load(toy_dataset["splits"]) as data:
        X_train, X_test = data["X_train"], data["X_test"]
    with np.load(toy_dataset["balanced"]) as data:
        X_bal, y_bal = data["images"], data["labels"]

    # balanced train = duplicates of train rows only
    train_rows = {r.tobytes() for r in X_train.reshape(len(X_train), -1)}
    bal_rows = {r.tobytes() for r in X_bal.reshape(len(X_bal), -1)}
    assert bal_rows <= train_rows

    # ...and therefore nothing from the test set is in training
    test_rows = {r.tobytes() for r in X_test.reshape(len(X_test), -1)}
    assert bal_rows.isdisjoint(test_rows)

    # val/test in the splits file are untouched by balancing
    with np.load(toy_dataset["splits"]) as data:
        assert dict(zip(*np.unique(data["y_test"], return_counts=True))) == {0: 4, 1: 2}
