import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from datasets.kaggle import KaggleDatasetLoader
from datasets.multipressure import MultiPressureDatasetLoader
from datasets.skaid import SkaidDatasetLoader
from training.prepare import prepare_dataset_matrix, balance_events
from training.splits import get_grouped_train_test_splits
from models.classical import get_classical_model
from models.calibration import calibrate_model, evaluate_calibration
from evaluation.metrics import compute_comprehensive_metrics


def train_production_model(
    data_dir: str,
    prod_dir: str,
    model_name: str = "SVM",
    feature_set: str = "C",
    normalization: str = "P2",
    sample_rate: int = 22050,
    n_mels: int = 80
) -> Dict[str, Any]:
    """
    Trains final production model on ALL allowed public datasets (Kaggle, SKAID, MultiPressure),
    calibrates probabilities on validation split, and exports production config artifacts.
    """
    os.makedirs(prod_dir, exist_ok=True)
    print("=== TRAINING FINAL PRODUCTION MODEL ===")

    def find_file(filename: str) -> str:
        p1 = os.path.join(data_dir, filename)
        if os.path.exists(p1): return p1
        p2 = os.path.join(os.path.dirname(data_dir), "raw", filename)
        if os.path.exists(p2): return p2
        return p1

    kaggle_path = find_file('archive.zip')
    mp_path = find_file('dataset.zip')
    mp_ks_path = find_file('Keystrokes_Dataset.zip')
    skaid_rec_path = find_file('Participant Recordings_zip.zip')
    skaid_log_path = find_file('Keystroke Logs_zip.zip')

    # Load all allowed public events
    kaggle_events = KaggleDatasetLoader(kaggle_path).load_events()
    mp_events = MultiPressureDatasetLoader(mp_path, mp_ks_path).load_events()
    skaid_events = SkaidDatasetLoader(skaid_rec_path, skaid_log_path).load_events(max_samples_per_class=500, window_duration_s=0.200)

    # Subsample SKAID for optimal dataset balance
    skaid_sub = balance_events(skaid_events, balance_strategy="A", max_skaid_per_class=1000)
    all_events = kaggle_events + mp_events + skaid_sub

    print(f"Total training events assembled: {len(all_events)}")

    # Feature matrix extraction
    X, y, groups, datasets = prepare_dataset_matrix(
        all_events, target_sr=sample_rate, feature_set=feature_set, normalization=normalization, n_mels=n_mels
    )

    # Grouped splits for training and calibration evaluation
    X_train, X_val, X_test, y_train, y_val, y_test = get_grouped_train_test_splits(X, y, groups, test_size=0.2)

    # Fit base model
    base_model = get_classical_model(model_name)
    base_model.fit(X_train, y_train)

    # Probability calibration on validation split
    calibrated_model = calibrate_model(base_model, X_val, y_val, method="sigmoid")

    # Evaluate on held-out test split
    test_probs = calibrated_model.predict_proba(X_test)
    metrics = compute_comprehensive_metrics(test_probs, y_test)
    calib_metrics = evaluate_calibration(test_probs, y_test)

    print("\n--- Production Model Validation Performance ---")
    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Test Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Test Log Loss: {metrics['log_loss']:.4f}")
    print(f"Test ECE:      {calib_metrics['ece']:.4f}")

    # Save trained model
    model_path = os.path.join(prod_dir, "model.joblib")
    joblib.dump(calibrated_model, model_path)
    print(f"Saved production model to {model_path}")

    # Save production configuration artifacts
    audio_cfg = {
        "sample_rate": sample_rate,
        "window_duration_s": 0.200,
        "n_fft": 1024,
        "hop_length": 256,
        "n_mels": n_mels
    }
    with open(os.path.join(prod_dir, "audio_config.json"), 'w') as f:
        json.dump(audio_cfg, f, indent=2)

    feat_cfg = {
        "feature_set": feature_set,
        "n_mfcc": 20,
        "include_deltas": True,
        "include_spectral": True,
        "include_temporal": True
    }
    with open(os.path.join(prod_dir, "feature_config.json"), 'w') as f:
        json.dump(feat_cfg, f, indent=2)

    prep_cfg = {
        "normalization": normalization,
        "highpass_cutoff_hz": 80.0,
        "preemphasis_coeff": 0.97
    }
    with open(os.path.join(prod_dir, "preprocessing_config.json"), 'w') as f:
        json.dump(prep_cfg, f, indent=2)

    thresh_cfg = {
        "confident_threshold": 0.60,
        "low_confidence_threshold": 0.35,
        "unknown_threshold": 0.20
    }
    with open(os.path.join(prod_dir, "thresholds.json"), 'w') as f:
        json.dump(thresh_cfg, f, indent=2)

    test_manifest = {
        "test_groups": [str(g) for g in sorted(list(set(groups[np.isin(y, y_test)])))],
        "test_samples_count": len(X_test),
        "target_classes_count": 27,
        "split_method": "GroupShuffleSplit"
    }
    with open(os.path.join(prod_dir, "final_test_manifest.json"), 'w') as f:
        json.dump(test_manifest, f, indent=2)

    metadata = {
        "model_name": model_name,
        "feature_set": feature_set,
        "normalization": normalization,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "validation_metrics": metrics,
        "calibration_metrics": calib_metrics,
        "classes_count": 27,
        "strictly_public_data_only": True
    }
    with open(os.path.join(prod_dir, "model_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata
