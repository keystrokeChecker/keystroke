from typing import Any, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


def get_classical_model(model_name: str, random_state: int = 42) -> Any:
    """
    Returns an uninstantiated or configured classifier instance for tabular acoustic features.
    """
    name = model_name.lower().replace("-", "_").replace(" ", "_")

    if name in ["logistic_regression", "logisticregression", "lr"]:
        return LogisticRegression(max_iter=1000, random_state=random_state)

    elif name in ["linear_svm", "linearsvc"]:
        return SVC(kernel="linear", probability=True, random_state=random_state)

    elif name in ["rbf_svm", "svm"]:
        return SVC(kernel="rbf", probability=True, C=1.0, random_state=random_state)

    elif name in ["random_forest", "randomforest", "rf"]:
        return RandomForestClassifier(n_estimators=200, max_depth=15, random_state=random_state, n_jobs=-1)

    elif name in ["extratrees", "et"]:
        return ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=random_state, n_jobs=-1)

    elif name in ["hist_gradient_boosting", "histgradientboosting", "histgb"]:
        return HistGradientBoostingClassifier(random_state=random_state)

    elif name in ["xgboost", "xgb"]:
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=150, max_depth=6, random_state=random_state, n_jobs=-1, eval_metric="mlogloss")
        except ImportError:
            print("[WARN] XGBoost not installed, falling back to HistGradientBoosting")
            return HistGradientBoostingClassifier(random_state=random_state)

    elif name in ["lightgbm", "lgb"]:
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(n_estimators=150, max_depth=6, random_state=random_state, n_jobs=-1, verbose=-1)
        except ImportError:
            print("[WARN] LightGBM not installed, falling back to HistGradientBoosting")
            return HistGradientBoostingClassifier(random_state=random_state)

    elif name in ["knn", "kneighbors"]:
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

    else:
        raise ValueError(f"Unknown classical model '{model_name}'")
