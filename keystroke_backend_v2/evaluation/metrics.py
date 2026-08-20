import numpy as np
from typing import Dict, Any
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, log_loss


def calculate_top_k_accuracy(probs: np.ndarray, y_true: np.ndarray, k: int = 3) -> float:
    """Calculates top-K classification accuracy."""
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    correct = [y_true[i] in top_k_preds[i] for i in range(len(y_true))]
    return float(np.mean(correct))


def compute_comprehensive_metrics(probs: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """
    Computes Accuracy, Macro F1, Balanced Accuracy, Top-3 Accuracy, and Log Loss.
    """
    y_pred = np.argmax(probs, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    top3_acc = calculate_top_k_accuracy(probs, y_true, k=3)

    try:
        ll = float(log_loss(y_true, probs, labels=np.arange(probs.shape[1])))
    except Exception:
        ll = float('nan')

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "top3_accuracy": top3_acc,
        "log_loss": ll
    }


def compute_per_dataset_metrics(probs: np.ndarray, y_true: np.ndarray, dataset_names: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Breakdown metrics by dataset (e.g. SKAID, Kaggle, MultiPressure).
    """
    results = {}
    unique_ds = np.unique(dataset_names)

    for ds in unique_ds:
        mask = (dataset_names == ds)
        if np.sum(mask) > 0:
            m = compute_comprehensive_metrics(probs[mask], y_true[mask])
            results[str(ds)] = m

    return results
