import numpy as np
from typing import Dict, List, Any
from datasets.canonical import index_to_label


def format_top_k_predictions(
    probs: np.ndarray,
    k: int = 3,
    confident_threshold: float = 0.60,
    low_confidence_threshold: float = 0.35
) -> Dict[str, Any]:
    """
    Formats model output probabilities into top-K predictions and assigns confidence status:
    - CONFIDENT: top_prob >= confident_threshold
    - LOW_CONFIDENCE: low_confidence_threshold <= top_prob < confident_threshold
    - UNKNOWN: top_prob < low_confidence_threshold
    """
    top_indices = np.argsort(probs)[::-1][:k]
    top_k_list = []

    for idx in top_indices:
        top_k_list.append({
            "key": index_to_label(int(idx)),
            "probability": float(round(float(probs[idx]), 4))
        })

    top_prob = float(probs[top_indices[0]])
    predicted_key = index_to_label(int(top_indices[0]))

    if top_prob >= confident_threshold:
        status = "CONFIDENT"
    elif top_prob >= low_confidence_threshold:
        status = "LOW_CONFIDENCE"
    else:
        status = "UNKNOWN"

    return {
        "prediction": predicted_key,
        "confidence": float(round(top_prob, 4)),
        "status": status,
        "top_k": top_k_list
    }
