import pytest
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from training.splits import verify_no_group_leakage, get_grouped_train_test_splits


def test_leakage_detector_pass():
    tr_groups = np.array(["rec1", "rec2", "rec3"])
    val_groups = np.array(["rec4", "rec5"])
    te_groups = np.array(["rec6"])

    # Should pass silently
    verify_no_group_leakage(tr_groups, val_groups, te_groups)


def test_leakage_detector_fail():
    tr_groups = np.array(["rec1", "rec2", "rec3"])
    val_groups = np.array(["rec3", "rec4"])  # rec3 overlaps!

    with pytest.raises(ValueError, match="CRITICAL LEAKAGE DETECTED"):
        verify_no_group_leakage(tr_groups, val_groups)


def test_grouped_train_test_splits():
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 27, size=100)
    groups = np.array([f"session_{i // 10}" for i in range(100)])

    X_train, X_val, X_test, y_train, y_val, y_test = get_grouped_train_test_splits(X, y, groups, test_size=0.2)
    assert len(X_train) + len(X_val) + len(X_test) == 100
