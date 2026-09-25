"""Unit tests for balance_dataset with REAL arrays (no mocks)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from balance_dataset import balance_dataset, balance_train  # noqa: E402


def test_balance_train_balances_classes():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 256, (5, 4, 4, 3), dtype=np.uint8)
    y = np.array([0, 0, 0, 0, 1])  # 4 smooth, 1 disk

    X_bal, y_bal = balance_train(X, y)

    # minority class (disk) oversampled to match the majority (4) -> 4/4
    assert len(X_bal) == 8
    assert (y_bal == 0).sum() == 4
    assert (y_bal == 1).sum() == 4
    assert X_bal.shape == (8, 4, 4, 3)


def test_balance_train_is_reproducible():
    rng = np.random.default_rng(1)
    X = rng.integers(0, 256, (6, 4, 4, 3), dtype=np.uint8)
    y = np.array([0, 0, 0, 1, 1, 1])

    a = balance_train(X, y, random_state=42)
    b = balance_train(X, y, random_state=42)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_balance_train_rejects_non_binary():
    X = np.zeros((6, 4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="binary"):
        balance_train(X, np.array([0, 0, 1, 1, 2, 2]))


def test_balance_dataset_writes_balanced_file(tmp_path):
    """File-level flow: split npz -> balanced train npz; val/test untouched."""
    rng = np.random.default_rng(2)
    n = 40
    X = rng.integers(0, 256, (n, 6, 6, 3), dtype=np.uint8)
    # pattern 0,0,1 repeated: first 20 -> 14 smooth / 6 disk
    y = np.array(([0, 0, 1] * 13) + [0, 0, 1])
    assert (y[:20] == 0).sum() == 14 and (y[:20] == 1).sum() == 6

    # deterministic 20/10/10 split
    train_idx = np.arange(20)
    val_idx = np.arange(20, 30)
    test_idx = np.arange(30, 40)

    splits = tmp_path / "splits.npz"
    np.savez_compressed(
        splits,
        X_train=X[train_idx], y_train=y[train_idx],
        X_val=X[val_idx], y_val=y[val_idx],
        X_test=X[test_idx], y_test=y[test_idx],
    )
    out = tmp_path / "train_balanced.npz"

    assert balance_dataset(splits_path=str(splits), output_path=str(out))

    with np.load(out) as data:
        X_bal, y_bal = data["images"], data["labels"]

    # train split had 14 smooth / 6 disk -> balanced to 14/14
    assert (y_bal == 0).sum() == 14
    assert (y_bal == 1).sum() == 14

    # balanced rows are a pure duplication of train rows
    train_rows = {r.tobytes() for r in X[train_idx].reshape(len(train_idx), -1)}
    bal_rows = {r.tobytes() for r in X_bal.reshape(len(X_bal), -1)}
    assert bal_rows <= train_rows

    # val/test untouched
    with np.load(splits) as data:
        assert np.array_equal(data["X_test"], X[test_idx])
        assert np.array_equal(data["y_val"], y[val_idx])


def test_balance_dataset_handles_missing_file(tmp_path):
    assert balance_dataset(
        splits_path=str(tmp_path / "nope.npz"),
        output_path=str(tmp_path / "out.npz"),
    ) is False
