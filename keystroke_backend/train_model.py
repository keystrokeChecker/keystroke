"""
Train a RandomForest model to predict per-word keystroke counts from YAMNet
per-onset embeddings, pooled per ground-truth word.

NOTE: trained on 6/18 sessions (rest had empty keylogs, no-detect, or
incomplete data). Treat accuracy as provisional until dataset is expanded.
keystroke_classifier.joblib was assumed canonical based on filename
convention, not confirmed by the team — verify before relying on this.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

from onset_detector import detect_onsets
from yamnet_filter import extract_features, extract_yamnet_candidates

DATA_DIR = Path(__file__).resolve().parent / "data"
MODELS_DIR = Path(__file__).resolve().parent / "models"
OUTPUT_MODEL = MODELS_DIR / "count_predictor_new.joblib"

VALID_SESSIONS = [
    "gain_check",
    "gain_test",
    "new1",
    "new2",
    "session1",
    "session4",
]


def parse_ground_truth(
    log_path: str,
) -> tuple[list[int], list[float], list[float]]:
    """
    Parse a keylog CSV and return per-word counts and the time range of each word.

    Returns
    -------
    word_counts : list[int]
        True keystroke count per word.
    word_starts : list[float]
        Onset timestamp of the first keypress in each word (seconds).
    word_ends   : list[float]
        Onset timestamp of the last  keypress in each word (seconds).
    """
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    timestamps = [float(r["timestamp_sec"]) for r in rows]
    word_boundaries = [
        r.get("is_word_boundary", "").strip().lower() in ("true", "1", "yes")
        for r in rows
    ]

    word_counts: list[int] = []
    word_starts: list[float] = []
    word_ends: list[float] = []

    current_keys = 0
    word_start = timestamps[0] if timestamps else 0.0

    for i, is_boundary in enumerate(word_boundaries):
        current_keys += 1
        if is_boundary:
            word_counts.append(current_keys)
            word_starts.append(word_start)
            word_ends.append(timestamps[i])
            current_keys = 0
            if i + 1 < len(timestamps):
                word_start = timestamps[i + 1]

    # Last word if the final boundary was missing
    if current_keys > 0:
        word_counts.append(current_keys)
        word_starts.append(word_start)
        word_ends.append(timestamps[-1] if timestamps else 0.0)

    return word_counts, word_starts, word_ends


def build_training_data(
    session_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build feature matrix *X* and target vector *y* from rule-based onset
    candidates paired with ground-truth word boundaries.

    For each word in each session:
    1. Detect all rule-based onsets for the full recording.
    2. Extract per-onset YAMNet features (1028-D) in one batch.
    3. Mean-pool the features of onsets whose timestamps fall inside the
       ground-truth word's time range → one 1028-D feature per word.
    4. Target = ground-truth keystroke count for that word.

    Returns
    -------
    X      : (n_words, 1028)
    y      : (n_words,)
    groups : (n_words,)  — session-name label for each word.
    """
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    groups_list: list[str] = []

    for session_name in session_names:
        wav_path = DATA_DIR / f"{session_name}.wav"
        log_path = DATA_DIR / f"{session_name}_log.csv"

        if not wav_path.exists() or not log_path.exists():
            print(f"  Skipping {session_name}: missing .wav or .csv")
            continue

        print(f"  Processing {session_name} ...")

        # ── Parse ground truth ────────────────────────────────────────────
        word_counts, word_starts, word_ends = parse_ground_truth(str(log_path))
        print(f"    {len(word_counts)} words, true counts = {word_counts}")

        # ── Rule-based onset detection ────────────────────────────────────
        onsets, _, _ = detect_onsets(str(wav_path), delta=0.07)
        print(f"    {len(onsets)} rule-based onsets")

        if len(onsets) == 0:
            continue

        # ── YAMNet features for all onsets at once ────────────────────────
        candidates = extract_yamnet_candidates(str(wav_path), onsets)
        if not candidates:
            continue
        all_features = extract_features(candidates, str(wav_path))  # (n_onsets, 1028)

        onset_times = np.array([c.onset_time for c in candidates])

        # ── Pool per word ─────────────────────────────────────────────────
        PADDING_SEC = 0.05  # small tolerance around the ground-truth window
        n_words_added = 0

        for wi in range(len(word_counts)):
            start_t = word_starts[wi] - PADDING_SEC
            end_t = word_ends[wi] + PADDING_SEC
            true_count = word_counts[wi]

            mask = (onset_times >= start_t) & (onset_times <= end_t)
            word_feats = all_features[mask]

            if len(word_feats) == 0:
                continue

            # Mean-pool per-onset features → single per-word feature vector
            X_list.append(word_feats.mean(axis=0))
            y_list.append(true_count)
            groups_list.append(session_name)
            n_words_added += 1

        print(f"    Added {n_words_added} word-level samples")

    if not X_list:
        raise RuntimeError("No training data produced — check session files.")

    return np.vstack(X_list), np.array(y_list), np.array(groups_list)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a RandomForest regressor that predicts per-word keystroke "
            "counts from mean-pooled YAMNet onset embeddings."
        )
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=VALID_SESSIONS,
        help="Session names to train on (default: all 6 valid sessions).",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_MODEL),
        help="Output model path (default: models/count_predictor_new.joblib).",
    )
    args = parser.parse_args()

    # ── Build dataset ─────────────────────────────────────────────────────
    print(f"Building training data from {len(args.names)} session(s)...")
    X, y, groups = build_training_data(args.names)
    print(f"\n  Total word samples : {len(X)}")
    print(f"  Feature dimension : {X.shape[1]}")
    print(f"  Count distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # ── Leave-One-Session-Out CV ──────────────────────────────────────────
    print("\nLeave-One-Session-Out Cross-Validation:")
    unique_sessions = np.unique(groups)
    fold_mae: list[float] = []
    fold_acc: list[float] = []

    for held_out in unique_sessions:
        test_mask = groups == held_out
        train_mask = ~test_mask

        if train_mask.sum() < 5:
            print(f"  Hold out {held_out}: skipped (only {train_mask.sum()} training samples)")
            continue

        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
        )
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)
        y_pred_int = np.round(y_pred).astype(int)
        mae = float(mean_absolute_error(y_te, y_pred))
        acc = float(accuracy_score(y_te, y_pred_int))

        fold_mae.append(mae)
        fold_acc.append(acc)
        print(
            f"  Hold out {held_out:15s}: "
            f"MAE={mae:.3f},  exact-match acc={acc:.3f}  "
            f"({len(X_te)} words)"
        )

    if fold_mae:
        print(
            f"\n  Mean held-out MAE  : {np.mean(fold_mae):.3f}  "
            f"(±{np.std(fold_mae):.3f})"
        )
        print(
            f"  Mean held-out acc  : {np.mean(fold_acc):.3f}  "
            f"(±{np.std(fold_acc):.3f})"
        )

    # ── Train final model on all data ─────────────────────────────────────
    print(f"\nTraining final model on all {len(X)} samples ...")
    final_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
    )
    final_model.fit(X, y)

    # ── Feature importance overview ───────────────────────────────────────
    importances = final_model.feature_importances_
    top5 = np.argsort(importances)[::-1][:5]
    print("\nTop 5 most important feature dimensions:")
    for rank, idx in enumerate(top5, 1):
        region = "YAMNet embedding" if idx < 1024 else "local feature"
        print(f"  {rank}. dim {idx:4d} ({region}):  {importances[idx]:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": final_model,
        "feature_type": "YAMNet_per_onset_embeddings_mean_pooled_per_word",
        "feature_dimension": int(X.shape[1]),
        "sessions": list(args.names),
        "n_word_samples": int(len(X)),
        "held_out_mae_mean": float(np.mean(fold_mae)) if fold_mae else None,
        "held_out_acc_mean": float(np.mean(fold_acc)) if fold_acc else None,
    }
    output_path = Path(args.output)
    joblib.dump(payload, output_path)
    print(f"\nSaved model to: {output_path}")


if __name__ == "__main__":
    main()
