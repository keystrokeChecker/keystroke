import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, StratifiedKFold, train_test_split


def verify_no_group_leakage(train_groups: np.ndarray, val_groups: np.ndarray, test_groups: Optional[np.ndarray] = None):
    """
    Asserts that no group ID appears in multiple split partitions.
    Fails automatically with ValueError if leakage is detected.
    """
    set_train = set(train_groups)
    set_val = set(val_groups)

    overlap_tr_val = set_train.intersection(set_val)
    if overlap_tr_val:
        raise ValueError(f"CRITICAL LEAKAGE DETECTED: Group overlap between Train and Val: {overlap_tr_val}")

    if test_groups is not None:
        set_test = set(test_groups)
        overlap_tr_te = set_train.intersection(set_test)
        if overlap_tr_te:
            raise ValueError(f"CRITICAL LEAKAGE DETECTED: Group overlap between Train and Test: {overlap_tr_te}")
        overlap_val_te = set_val.intersection(set_test)
        if overlap_val_te:
            raise ValueError(f"CRITICAL LEAKAGE DETECTED: Group overlap between Val and Test: {overlap_val_te}")

    print("[LEAKAGE CHECK PASSED] No group boundaries crossed across splits.")


def get_grouped_train_test_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs GroupShuffleSplit to split data into Train, Validation, and Test sets.
    """
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss1.split(X, y, groups))

    X_train_val, X_test = X[train_val_idx], X[test_idx]
    y_train_val, y_test = y[train_val_idx], y[test_idx]
    groups_train_val, groups_test = groups[train_val_idx], groups[test_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state + 1)
    train_idx, val_idx = next(gss2.split(X_train_val, y_train_val, groups_train_val))

    X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
    y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
    groups_train, groups_val = groups_train_val[train_idx], groups_train_val[val_idx]

    # Verify zero leakage
    verify_no_group_leakage(groups_train, groups_val, groups_test)

    return X_train, X_val, X_test, y_train, y_val, y_test
