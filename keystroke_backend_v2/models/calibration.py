import numpy as np
from typing import Dict, Any, Tuple
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss


def calibrate_model(model: Any, X_val: np.ndarray, y_val: np.ndarray, method: str = "sigmoid") -> Any:
    """
    Calibrates model probabilities using Platt scaling ('sigmoid') or 'isotonic' regression on validation data.
    """
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(model), method=method)
        calibrated.fit(X_val, y_val)
        return calibrated
    except Exception:
        pass

    for cv_val in ["prefit", 3, 2]:
        try:
            calibrated = CalibratedClassifierCV(estimator=model, method=method, cv=cv_val)
            calibrated.fit(X_val, y_val)
            return calibrated
        except Exception:
            continue

    # Fallback to returning base model if calibration cannot be fit
    return model


def calculate_expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculates Expected Calibration Error (ECE) across confidence bins.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)


def evaluate_calibration(probs: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """
    Evaluates probability calibration using Brier Score, Log Loss, and Expected Calibration Error (ECE).
    """
    ll = float(log_loss(y_true, probs, labels=np.arange(probs.shape[1])))
    ece = calculate_expected_calibration_error(probs, y_true)

    # Multi-class Brier score = mean sum of squared errors per row
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    return {
        "log_loss": ll,
        "brier_score": brier,
        "ece": ece
    }
