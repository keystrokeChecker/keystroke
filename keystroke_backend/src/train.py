import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BACKEND_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BACKEND_DIR / "data" / "yamnet_dataset"
MODEL_DIR = BACKEND_DIR / "models"
MODEL_PATH = MODEL_DIR / "yamnet_keystroke_classifier.joblib"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    X_path = dataset_dir / "X.npy"
    y_path = dataset_dir / "y.npy"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Missing dataset files at {dataset_dir}. "
            "Run: python -m src.dataset_builder"
        )

    X = np.load(X_path)
    y = np.load(y_path)

    if len(X) < 10:
        raise ValueError("Not enough samples to train a classifier.")

    classes, counts = np.unique(y, return_counts=True)
    can_stratify = len(classes) > 1 and np.all(counts >= 2)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if can_stratify else None,
    )

    # RandomForest without probability calibration.
    # CalibratedClassifierCV with cv=3 collapses recall to ~0.25 on this
    # small dataset (~260 positive samples). Bare RF with balanced weights
    # achieves the best class-1 recall for keystroke detection.
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds, digits=3, zero_division=0))

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
