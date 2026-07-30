"""
Train a calibrated scikit-learn model using 1024-dimensional YAMNet embeddings.

Usage:
    python train_model.py
    python train_model.py --calibration isotonic --model-type rf
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    log_loss,
    mean_absolute_error,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURES_PATH = Path(__file__).resolve().parent / "data" / "yamnet_dataset" / "X.npy"
DEFAULT_LABELS_PATH = Path(__file__).resolve().parent / "data" / "yamnet_dataset" / "y.npy"
DEFAULT_OUTPUT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "yamnet_keystroke_classifier.joblib"


def build_model_pipeline(model_type: str = "rf", calibration_method: str = "sigmoid") -> Pipeline:
    if model_type == "gb":
        base_clf = GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42,
        )
    else:
        base_clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=4,
            class_weight="balanced",
            random_state=42,
        )
    
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_clf,
        method=calibration_method,
        cv=3,
    )
    
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", calibrated_clf),
    ])


def main():
    parser = argparse.ArgumentParser(description="Train calibrated model on 1024-D YAMNet embeddings.")
    parser.add_argument("--features", type=str, default=str(DEFAULT_FEATURES_PATH), help="Path to X.npy feature matrix")
    parser.add_argument("--labels", type=str, default=str(DEFAULT_LABELS_PATH), help="Path to y.npy ground truth array")
    parser.add_argument("--output-model", type=str, default=str(DEFAULT_OUTPUT_MODEL_PATH), help="Destination model path")
    parser.add_argument("--model-type", choices=["rf", "gb"], default="rf", help="Classifier type (rf=RandomForest, gb=GradientBoosting)")
    parser.add_argument("--calibration", choices=["sigmoid", "isotonic"], default="sigmoid", help="Calibration method for probability calibration")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for testing")
    args = parser.parse_args()

    features_path = Path(args.features)
    labels_path = Path(args.labels)
    output_path = Path(args.output_model)

    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(f"Feature or label file missing: {features_path}, {labels_path}")

    print(f"Loading features from {features_path}...")
    X = np.load(features_path)
    print(f"Loading labels from {labels_path}...")
    y = np.load(labels_path)

    print(f"Loaded dataset: X shape = {X.shape}, y shape = {y.shape}")
    if X.ndim != 2 or X.shape[1] != 1024:
        raise ValueError(f"Expected feature matrix with 1024 columns, got shape {X.shape}")

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    print(f"Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples")

    # Create & Fit Pipeline
    pipeline = build_model_pipeline(model_type=args.model_type, calibration_method=args.calibration)
    print(f"Fitting Pipeline (StandardScaler + Calibrated {args.model_type.upper()} [{args.calibration}])...")
    pipeline.fit(X_train, y_train)

    # Evaluate Predictions & Probabilities
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    brier = brier_score_loss(y_test, y_prob)
    loss = log_loss(y_test, y_prob)

    print("\n========================================================")
    print("                CALIBRATED EVALUATION RESULTS           ")
    print("========================================================")
    print(f"  Accuracy                : {acc * 100:.2f}%")
    print(f"  Mean Absolute Error     : {mae:.4f}")
    print(f"  Precision               : {precision:.4f}")
    print(f"  Recall                  : {recall:.4f}")
    print(f"  F1 Score                : {f1:.4f}")
    print(f"  Brier Score (Loss)      : {brier:.4f}")
    print(f"  Log Loss                : {loss:.4f}")
    print("--------------------------------------------------------")
    print(f"  Probability Min / Mean / Max: {y_prob.min():.4f} / {y_prob.mean():.4f} / {y_prob.max():.4f}")
    print("--------------------------------------------------------")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    print("========================================================\n")

    # Save trained pipeline
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"Successfully saved trained model pipeline to: {output_path}")


if __name__ == "__main__":
    main()
