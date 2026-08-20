import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.metrics import confusion_matrix
from datasets.canonical import CANONICAL_CLASSES


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Computes 27x27 confusion matrix."""
    return confusion_matrix(y_true, y_pred, labels=np.arange(27))


def find_top_confused_pairs(cm: np.ndarray, top_n: int = 10) -> List[Dict[str, Any]]:
    """
    Identifies top confused key pairs (excluding diagonal).
    """
    n_classes = cm.shape[0]
    confusions = []

    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                true_label = CANONICAL_CLASSES[i]
                pred_label = CANONICAL_CLASSES[j]
                count = int(cm[i, j])
                confusions.append({
                    "true_key": true_label,
                    "pred_key": pred_label,
                    "count": count
                })

    confusions.sort(key=lambda x: x["count"], reverse=True)
    return confusions[:top_n]
