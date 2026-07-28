"""Train a logistic-regression filter on YAMNet onset embeddings.

Example:
    python train_yamnet_classifier.py --names session1 session2 session3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from onset_detector import detect_onsets, evaluate_against_ground_truth
from tune_and_evaluate import load_ground_truth
from yamnet_filter import extract_features, extract_yamnet_candidates

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "keystroke_classifier.joblib"


def label_candidates(onsets: np.ndarray, ground_truth: list[float], tolerance: float) -> np.ndarray:
    """Positive iff a rule-based candidate is within tolerance of any keylog time."""
    truth = np.asarray(ground_truth, dtype=float)
    if not len(truth):
        return np.zeros(len(onsets), dtype=np.int64)
    return np.asarray([np.any(np.abs(truth - onset) <= tolerance) for onset in onsets], dtype=np.int64)


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
        ]
    )


def prf(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return float(precision), float(recall), float(f1)


def validate_on_new_recording(wav_path: str, log_path: str, classifier_path: str, confidence_threshold: float = 0.3) -> dict:
    """
    Validate the trained classifier on a fresh recording from a different setup.
    """
    from yamnet_filter import filter_onsets_with_yamnet

    # 1. Load ground truth
    _, gt_times = load_ground_truth(log_path)
    gt_times = np.array(gt_times)
    
    # 2. Run raw detect_onsets
    raw_onsets, _, _ = detect_onsets(wav_path)
    raw_eval = evaluate_against_ground_truth(raw_onsets, gt_times)

    # 3. Apply YAMNet filter
    filtered_onsets = filter_onsets_with_yamnet(raw_onsets, wav_path, classifier_path, confidence_threshold=confidence_threshold)
    filtered_eval = evaluate_against_ground_truth(filtered_onsets, gt_times)

    print(f"\n========================================================")
    print(f"  VALIDATION ON FRESH RECORDING: {Path(wav_path).name}")
    print(f"  Filtering Confidence Threshold: {confidence_threshold:.2f}")
    print(f"========================================================")
    print(f"  Raw detector onsets: {len(raw_onsets)} | Ground truth keys: {len(gt_times)}")
    print(f"    Raw Precision : {raw_eval['precision']:.3f}")
    print(f"    Raw Recall    : {raw_eval['recall']:.3f}")
    print(f"    Raw F1        : {raw_eval['f1']:.3f}")
    print("-" * 56)
    print(f"  YAMNet filtered onsets: {len(filtered_onsets)}")
    print(f"    Filtered Precision : {filtered_eval['precision']:.3f}")
    print(f"    Filtered Recall    : {filtered_eval['recall']:.3f}")
    print(f"    Filtered F1        : {filtered_eval['f1']:.3f}")
    print(f"========================================================\n")
    return filtered_eval


def train_and_evaluate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, names: list[str], model_path: Path) -> None:
    """
    Train final classifier, perform cross-validation, and log outputs.
    """
    # Print generalization warning if small number of distinct recording setups / sessions
    unique_setups = set(groups)
    if len(unique_setups) < 3:
        print(f"\n⚠️  WARNING: The model is only being trained on {len(unique_setups)} distinct setup(s)/session(s).")
        print("   Real-world generalization to new devices/mics is UNVERIFIED until tested on external recordings.")
    
    print("\nLeave-one-session-out validation:")
    fold_scores = []
    for held_out in names:
        test_mask = groups == held_out
        train_mask = ~test_mask
        if not np.any(test_mask):
            print(f"  {held_out}: skipped (no candidates)")
            continue
        if len(np.unique(y[train_mask])) != 2:
            print(f"  {held_out}: skipped (training fold has only one class)")
            continue
        model = make_pipeline()
        model.fit(X[train_mask], y[train_mask])
        precision, recall, f1 = prf(y[test_mask], model.predict(X[test_mask]))
        fold_scores.append((precision, recall, f1))
        print(
            f"  held out {held_out}: P/R/F1 = {precision:.3f}/{recall:.3f}/{f1:.3f} "
            f"({int(y[test_mask].sum())} positive of {int(test_mask.sum())} candidates)"
        )
    if fold_scores:
        mean_scores = np.mean(fold_scores, axis=0)
        print(f"  mean held-out P/R/F1 = {mean_scores[0]:.3f}/{mean_scores[1]:.3f}/{mean_scores[2]:.3f}")

    final_model = make_pipeline()
    final_model.fit(X, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": final_model,
            "feature_type": "normalized_yamnet_features",
            "feature_dimension": int(X.shape[1]),
            "sessions": list(names),
        },
        model_path,
    )
    print(f"\nSaved final classifier trained on {len(y)} candidates to: {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YAMNet keystroke candidate filter.")
    parser.add_argument("--names", nargs="+", default=["session1", "session2", "session3"])
    parser.add_argument("--delta", type=float, default=0.07, help="Detector delta used to make candidates")
    parser.add_argument("--tolerance", type=float, default=0.08, help="Keylog match tolerance in seconds")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--validate-wav", type=str, default=None, help="Path to WAV file for validation")
    parser.add_argument("--validate-log", type=str, default=None, help="Path to CSV log file for validation")
    parser.add_argument("--threshold", type=float, default=0.3, help="Confidence threshold for YAMNet filter (default: 0.3)")
    args = parser.parse_args()

    # If validation arguments are provided, perform validation and exit
    if args.validate_wav and args.validate_log:
        validate_on_new_recording(args.validate_wav, args.validate_log, args.model_path, confidence_threshold=args.threshold)
        return

    if args.tolerance <= 0:
        parser.error("--tolerance must be greater than zero")

    feature_sets, label_sets, group_sets = [], [], []
    print("Building YAMNet training set from rule-based onset candidates...")
    for name in args.names:
        wav_path, log_path = DATA_DIR / f"{name}.wav", DATA_DIR / f"{name}_log.csv"
        if not wav_path.exists() or not log_path.exists():
            raise FileNotFoundError(f"Missing WAV or keylog for session '{name}' in {DATA_DIR}")
        _, ground_truth = load_ground_truth(str(log_path))
        onsets, _, _ = detect_onsets(str(wav_path), delta=args.delta)
        labels = label_candidates(onsets, ground_truth, args.tolerance)
        baseline = evaluate_against_ground_truth(onsets, ground_truth, tolerance=args.tolerance)
        print(
            f"  {name}: {len(onsets)} candidates, {int(labels.sum())} positive / "
            f"{int((labels == 0).sum())} negative; detector P/R/F1 = "
            f"{baseline['precision']:.3f}/{baseline['recall']:.3f}/{baseline['f1']:.3f}"
        )
        if len(onsets):
            candidates = extract_yamnet_candidates(str(wav_path), onsets)
            features = extract_features(candidates, str(wav_path))
            feature_sets.append(features)
            label_sets.append(labels)
            group_sets.append(np.full(len(labels), name, dtype=object))

    if not feature_sets:
        raise RuntimeError("No onset candidates were produced; lower --delta and retry")
    X, y, groups = np.vstack(feature_sets), np.concatenate(label_sets), np.concatenate(group_sets)
    if len(np.unique(y)) != 2:
        raise RuntimeError("Training data needs both positive and negative onset candidates")

    train_and_evaluate(X, y, groups, args.names, Path(args.model_path))


if __name__ == "__main__":
    main()
