import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import classification_report
from datasets.canonical import CANONICAL_CLASSES


def generate_per_class_report(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Generates a per-class DataFrame with Precision, Recall, F1-Score, and Support for all 27 canonical classes.
    """
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=np.arange(27),
        target_names=CANONICAL_CLASSES,
        output_dict=True,
        zero_division=0
    )
    
    rows = []
    for cls_name in CANONICAL_CLASSES:
        if cls_name in report_dict:
            item = report_dict[cls_name]
            rows.append({
                "Class": cls_name,
                "Precision": round(float(item["precision"]), 4),
                "Recall": round(float(item["recall"]), 4),
                "F1-Score": round(float(item["f1-score"]), 4),
                "Support": int(item["support"])
            })

    return pd.DataFrame(rows)
