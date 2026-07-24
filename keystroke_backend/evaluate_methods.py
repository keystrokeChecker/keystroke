"""
Compare rule-based vs ML-based keystroke count prediction on all 6 valid sessions.

MAE alignment method
--------------------
When the predicted and true count lists have different lengths, MAE is
computed over the **prefix** (first N words), where N = min(len(true),
len(pred)).  Extra words beyond the shorter list are IGNORED — they are
NOT padded with zeros or penalised.  This is the standard sklearn
mean_absolute_error convention (both arrays must be the same length).

A separate column **WcD** (word-count delta) reports len(pred) - len(true)
so the reader can see when over/under-splitting is hidden by truncation.

Usage:
    python evaluate_methods.py
    python evaluate_methods.py --delta 0.07 --gap-threshold 0.4
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error

from onset_detector import detect_onsets
from segmenter import segment_into_words
from yamnet_filter import extract_features, extract_yamnet_candidates
from src.yamnet_config import GAP_THRESHOLD_ML, GAP_THRESHOLD_RULE

DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "count_predictor_new.joblib"

VALID_SESSIONS = [
    "gain_check",
    "gain_test",
    "new1",
    "new2",
    "session1",
    "session4",
]


def load_ground_truth_counts(log_path: str) -> list[int]:
    """
    Read a keylog CSV and return the true per-word keystroke counts from the
    ``is_word_boundary`` field.
    """
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    word_boundaries = [
        r.get("is_word_boundary", "").strip().lower() in ("true", "1", "yes")
        for r in rows
    ]

    counts: list[int] = []
    current = 0
    for is_boundary in word_boundaries:
        current += 1
        if is_boundary:
            counts.append(current)
            current = 0
    if current > 0:
        counts.append(current)
    return counts


def predict_rule(wav_path: str, delta: float, gap_threshold: float, merge_gap: float) -> list[int]:
    """Rule-based path: detect_onsets -> segment_into_words."""
    onsets, _, _ = detect_onsets(wav_path, delta=delta, min_gap_seconds=merge_gap)
    if len(onsets) == 0:
        return []
    counts, _ = segment_into_words(onsets, gap_threshold=gap_threshold)
    return counts


def predict_ml(wav_path: str, model, delta: float, gap_threshold: float, merge_gap: float) -> list[int]:
    """ML path: detect_onsets -> segment_into_words -> per-word YAMNet features -> predict."""
    # Step 1: Detect onsets
    onsets, _, _ = detect_onsets(wav_path, delta=delta, min_gap_seconds=merge_gap)
    if len(onsets) == 0:
        return []

    # Step 2: Segment into words (this gives us the word groupings)
    _, word_groups = segment_into_words(onsets, gap_threshold=gap_threshold)

    # Step 3: Extract YAMNet candidates + features for ALL onsets at once
    candidates = extract_yamnet_candidates(wav_path, onsets)
    if not candidates:
        return [1] * len(word_groups)  # fallback: 1 per word

    all_features = extract_features(candidates, wav_path)  # (n_onsets, 1028)
    candidate_times = np.array([c.onset_time for c in candidates])

    # Step 4: Mean-pool per word and predict
    predicted_counts: list[int] = []
    for word_onsets in word_groups:
        # Find feature indices matching this word's onset times
        indices: list[int] = []
        for t in word_onsets:
            idx = int(np.argmin(np.abs(candidate_times - t)))
            if abs(candidate_times[idx] - t) < 0.002:
                indices.append(idx)

        if not indices:
            predicted_counts.append(1)
            continue

        word_feat = all_features[indices].mean(axis=0).reshape(1, -1)
        raw_pred = float(model.predict(word_feat)[0])
        predicted_counts.append(max(1, round(raw_pred)))

    return predicted_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare rule vs ML keystroke count prediction on valid sessions."
    )
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="YAMNet classifier threshold (default 0.5, unused by rule path)")
    parser.add_argument("--delta", type=float, default=0.07,
                        help="Onset detection sensitivity (default 0.07)")
    parser.add_argument("--gap-threshold", type=float, default=None,
                        help="Word boundary gap in seconds (default: None, uses 0.40s for Rule and 0.75s for ML)")
    parser.add_argument("--merge-gap", type=float, default=0.06,
                        help="Min gap between onsets in seconds (default 0.06)")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH),
                        help="Path to the trained ML model (default: models/count_predictor_new.joblib)")
    args = parser.parse_args()

    # Load the ML model
    model_path = Path(args.model)
    if model_path.exists():
        payload = joblib.load(model_path)
        model = payload["model"]
        print(f"ML model loaded from: {model_path}")
    else:
        print(f"ML model not found at {model_path} — ML path will be skipped.")
        model = None

    delta = args.delta
    rule_gap = args.gap_threshold if args.gap_threshold is not None else GAP_THRESHOLD_RULE
    ml_gap = args.gap_threshold if args.gap_threshold is not None else GAP_THRESHOLD_ML
    merge_gap = args.merge_gap

    print(f"Defaults in use — Rule gap_threshold: {rule_gap}s, ML gap_threshold: {ml_gap}s")

    # ── Comparison table ──────────────────────────────────────────────────
    print()
    sep = "=" * 135
    header = (f"{'Session':12s}  {'True':22s}  {'Rule':22s}  "
              f"{'Rule MAE':8s}  {'Rule Acc':8s}  "
              f"{'ML':22s}  {'ML MAE':8s}  {'ML Acc':8s}")
    print(sep)
    print(header)
    print(sep)

    all_rule_mae: list[float] = []
    all_rule_acc: list[float] = []
    all_ml_mae: list[float] = []
    all_ml_acc: list[float] = []

    for session in VALID_SESSIONS:
        wav_path = str(DATA_DIR / f"{session}.wav")
        log_path = str(DATA_DIR / f"{session}_log.csv")

        true_counts = load_ground_truth_counts(log_path)
        true_str = str(true_counts)

        # ── Rule ──────────────────────────────────────────────────────────
        rule_counts = predict_rule(wav_path, delta, rule_gap, merge_gap)
        rule_str = str(rule_counts)

        # Align: prefix-truncation to min(len(true), len(pred))
        prefix_len = min(len(true_counts), len(rule_counts))
        rule_mae = float(mean_absolute_error(true_counts[:prefix_len], rule_counts[:prefix_len]))
        rule_acc = float(accuracy_score(true_counts[:prefix_len], rule_counts[:prefix_len]))
        rule_wcd = len(rule_counts) - len(true_counts)

        all_rule_mae.append(rule_mae)
        all_rule_acc.append(rule_acc)

        # ── ML  ───────────────────────────────────────────────────────────
        ml_str = "SKIP (no model)"
        ml_mae = 0.0
        ml_acc = 0.0
        ml_wcd = 0

        if model is not None:
            ml_counts = predict_ml(wav_path, model, delta, ml_gap, merge_gap)
            ml_str = str(ml_counts)

            prefix_len = min(len(true_counts), len(ml_counts))
            ml_mae = float(mean_absolute_error(true_counts[:prefix_len], ml_counts[:prefix_len]))
            ml_acc = float(accuracy_score(true_counts[:prefix_len], ml_counts[:prefix_len]))
            ml_wcd = len(ml_counts) - len(true_counts)
            all_ml_mae.append(ml_mae)
            all_ml_acc.append(ml_acc)

        t_str = f"{true_str:22s}"
        r_str = f"{rule_str:22s}"
        m_str = f"{ml_str:22s}"
        print(f"{session:12s}  {t_str}  {r_str}  "
              f"{rule_mae:>7.3f}  {rule_acc:>7.3f}  {rule_wcd:>+4d}  "
              f"{m_str}  {ml_mae:>7.3f}  {ml_acc:>7.3f}  {ml_wcd:>+4d}")

    # ── Overall summary ───────────────────────────────────────────────────
    print(sep)
    print(f"\n{'Summary':12s}  {'':22s}  "
          f"{'Rule MAE':>8s}  {'Rule Acc':>8s}  {'Rule WcD':>8s}  "
          f"{'ML MAE':>8s}  {'ML Acc':>8s}  {'ML WcD':>8s}")
    print(f"{'Mean':12s}  {'':22s}  "
          f"{np.mean(all_rule_mae):>8.3f}  {np.mean(all_rule_acc):>8.3f}  "
          f"{np.mean([len(predict_rule(str(DATA_DIR / f'{s}.wav'), delta, rule_gap, merge_gap)) - len(load_ground_truth_counts(str(DATA_DIR / f'{s}_log.csv'))) for s in VALID_SESSIONS]):>+8.2f}  ",
          end="")
    if model is not None:
        ml_wcds = []
        for s in VALID_SESSIONS:
            w = str(DATA_DIR / f'{s}.wav')
            ml_wcds.append(len(predict_ml(w, model, delta, ml_gap, merge_gap)) - len(load_ground_truth_counts(str(DATA_DIR / f'{s}_log.csv'))))
        print(f"{np.mean(all_ml_mae):>8.3f}  {np.mean(all_ml_acc):>8.3f}  {np.mean(ml_wcds):>+8.2f}")
    else:
        print(f"{'N/A':>8s}  {'N/A':>8s}  {'N/A':>8s}")
    print()

    # Determine winner
    if model is not None:
        rule_mean = np.mean(all_rule_mae)
        ml_mean = np.mean(all_ml_mae)
        if rule_mean < ml_mean:
            print(f"Winner: RULE  (MAE {rule_mean:.3f} vs {ml_mean:.3f})")
        elif ml_mean < rule_mean:
            print(f"Winner: ML    (MAE {ml_mean:.3f} vs {rule_mean:.3f})")
        else:
            print("Tie between RULE and ML")


if __name__ == "__main__":
    main()
