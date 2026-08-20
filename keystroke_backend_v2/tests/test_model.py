import pytest
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.classical import get_classical_model
from models.calibration import calibrate_model, evaluate_calibration


def test_classical_model_fit_predict():
    X = np.random.randn(54, 20).astype(np.float32)
    y = np.concatenate([np.arange(27), np.arange(27)])

    model = get_classical_model("ExtraTrees")
    model.fit(X, y)

    probs = model.predict_proba(X)
    assert probs.shape == (54, 27)
    assert np.allclose(np.sum(probs, axis=1), 1.0)


def test_model_calibration():
    X_tr = np.random.randn(54, 20).astype(np.float32)
    y_tr = np.concatenate([np.arange(27), np.arange(27)])
    X_val = np.random.randn(54, 20).astype(np.float32)
    y_val = np.concatenate([np.arange(27), np.arange(27)])

    model = get_classical_model("LogisticRegression")
    model.fit(X_tr, y_tr)

    calib = calibrate_model(model, X_val, y_val, method="sigmoid")
    probs_val = calib.predict_proba(X_val)

    metrics = evaluate_calibration(probs_val, y_val)
    assert "log_loss" in metrics
    assert "brier_score" in metrics
    assert "ece" in metrics
