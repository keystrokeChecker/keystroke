"""
Keystroke count prediction pipeline.

Provides three paths:
  1. predict_keystroke_counts        — YAMNet false-positive filter (default)
  2. predict_keystroke_counts_rule   — rule-based DSP-only (fallback)
  3. predict_keystroke_counts_ml     — RandomForest regressor on YAMNet embeddings

Pipeline flow:
    detect_onsets  →  [filter_onsets_with_yamnet | per-word pooling + predict]  →  segment_into_words
"""

from pathlib import Path

import joblib
import numpy as np

from onset_detector import detect_onsets
from segmenter import segment_into_words
from yamnet_filter import extract_features, extract_yamnet_candidates, filter_onsets_with_yamnet
from src.yamnet_config import GAP_THRESHOLD_ML, GAP_THRESHOLD_RULE

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_DEFAULT_CLASSIFIER = _MODELS_DIR / "keystroke_classifier.joblib"
_COUNT_PREDICTOR = _MODELS_DIR / "count_predictor_new.joblib"


def predict_keystroke_counts(
    wav_path: str,
    threshold: float = 0.5,
    delta: float = 0.07,
    gap_threshold: float = GAP_THRESHOLD_ML,
    merge_gap_seconds: float = 0.06,
) -> list[int]:
    """
    Predict per-word keystroke counts from a WAV audio file.

    Uses the YAMNet ML pipeline (onset detection → YAMNet false-positive
    filtering → word segmentation).  Falls back to rule-only if the trained
    classifier file is not present.

    Parameters
    ----------
    wav_path : str
        Path to the WAV audio file.
    threshold : float
        YAMNet classifier confidence threshold (0.0 to 1.0).
    delta : float
        Onset detection sensitivity (lower = more sensitive).
    gap_threshold : float
        Word boundary gap in seconds.
    merge_gap_seconds : float
        Minimum gap between consecutive onsets in seconds.

    Returns
    -------
    list[int]
        Per-word keystroke counts, e.g. [3, 7].
        Empty list if no keystrokes detected.
    """
    if gap_threshold is None:
        gap_threshold = GAP_THRESHOLD_ML

    # ── Step 1: Rule-based onset detection (DSP) ──────────────────────────
    onsets, _, _ = detect_onsets(
        wav_path,
        delta=delta,
        min_gap_seconds=merge_gap_seconds,
    )

    if len(onsets) == 0:
        return []

    # ── Step 2: YAMNet false-positive filtering ───────────────────────────
    if _DEFAULT_CLASSIFIER.exists():
        filtered = filter_onsets_with_yamnet(
            onsets,
            wav_path,
            str(_DEFAULT_CLASSIFIER),
            confidence_threshold=threshold,
        )
        if len(filtered) == 0:
            return []
        onset_times = filtered
    else:
        # Fall back to rule-based only (no ML model available)
        onset_times = onsets

    # ── Step 3: Segment onsets into words ─────────────────────────────────
    counts, _ = segment_into_words(onset_times, gap_threshold=gap_threshold)

    return counts


def predict_keystroke_counts_rule(
    wav_path: str,
    delta: float = 0.07,
    gap_threshold: float = GAP_THRESHOLD_RULE,
    merge_gap_seconds: float = 0.06,
) -> list[int]:
    """
    Predict per-word keystroke counts using rule-based DSP only (no ML).

    Pipeline: detect_onsets → segment_into_words

    Parameters
    ----------
    wav_path : str
        Path to the WAV audio file.
    delta : float
        Onset detection sensitivity (lower = more sensitive).
    gap_threshold : float
        Word boundary gap in seconds.
    merge_gap_seconds : float
        Minimum gap between consecutive onsets in seconds.

    Returns
    -------
    list[int]
        Per-word keystroke counts, e.g. [3, 7].
        Empty list if no keystrokes detected.
    """
    if gap_threshold is None:
        gap_threshold = GAP_THRESHOLD_RULE

    onsets, _, _ = detect_onsets(
        wav_path,
        delta=delta,
        min_gap_seconds=merge_gap_seconds,
    )

    if len(onsets) == 0:
        return []

    counts, _ = segment_into_words(onsets, gap_threshold=gap_threshold)
    return counts


def predict_keystroke_counts_ml(
    wav_path: str,
    delta: float = 0.07,
    gap_threshold: float = GAP_THRESHOLD_ML,
    merge_gap_seconds: float = 0.06,
) -> list[int]:
    """
    Predict per-word keystroke counts using the trained RandomForest regressor
    on per-word mean-pooled YAMNet embeddings.

    Pipeline: detect_onsets → segment_into_words → per-word YAMNet feature
    pooling → RandomForest predict → round

    Parameters
    ----------
    wav_path : str
        Path to the WAV audio file.
    delta : float
        Onset detection sensitivity (lower = more sensitive).
    gap_threshold : float
        Word boundary gap in seconds.
    merge_gap_seconds : float
        Minimum gap between consecutive onsets in seconds.

    Returns
    -------
    list[int]
        Per-word keystroke counts, e.g. [3, 7].
        Empty list if no keystrokes detected.
    """
    if gap_threshold is None:
        gap_threshold = GAP_THRESHOLD_ML

    if not _COUNT_PREDICTOR.exists():
        raise FileNotFoundError(
            f"Count predictor model not found: {_COUNT_PREDICTOR}. "
            f"Run train_model.py first."
        )

    # Load the model
    payload = joblib.load(str(_COUNT_PREDICTOR))
    model = payload["model"]

    # Step 1: Detect onsets
    onsets, _, _ = detect_onsets(wav_path, delta=delta, min_gap_seconds=merge_gap_seconds)
    if len(onsets) == 0:
        return []

    # Step 2: Segment into words
    _, word_groups = segment_into_words(onsets, gap_threshold=gap_threshold)

    # Step 3: Extract YAMNet features for all onsets at once
    candidates = extract_yamnet_candidates(wav_path, onsets)
    if not candidates:
        return [1] * len(word_groups)

    all_features = extract_features(candidates, wav_path)
    candidate_times = np.array([c.onset_time for c in candidates], dtype=float)

    # Step 4: Mean-pool per word and predict
    predicted_counts: list[int] = []
    for word_onsets in word_groups:
        indices: list[int] = []
        for t in word_onsets:
            idx = int(np.argmin(np.abs(candidate_times - t)))
            if np.abs(candidate_times[idx] - t) < 0.002:
                indices.append(idx)

        if not indices:
            predicted_counts.append(1)
            continue

        word_feat = all_features[indices].mean(axis=0).reshape(1, -1)
        raw_pred = float(model.predict(word_feat)[0])
        predicted_counts.append(max(1, round(raw_pred)))

    return predicted_counts
