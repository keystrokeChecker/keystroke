import numpy as np
from typing import List, Any


class WeightedEnsembleClassifier:
    """
    Weighted probability ensemble combining predictions from multiple trained estimators.
    """

    def __init__(self, models: List[Any], weights: List[float]):
        if len(models) != len(weights):
            raise ValueError("Number of models must match number of weights.")
        self.models = models
        weights_arr = np.array(weights, dtype=np.float32)
        self.weights = weights_arr / np.sum(weights_arr)
        self.classes_ = getattr(models[0], "classes_", np.arange(27))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        prob_sum = None
        for model, w in zip(self.models, self.weights):
            probs = model.predict_proba(X)
            if prob_sum is None:
                prob_sum = w * probs
            else:
                prob_sum += w * probs
        return prob_sum

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
