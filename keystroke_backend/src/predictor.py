"""
Keystroke count prediction pipeline.

Provides three paths:
  1. predict_keystroke_counts        — YAMNet false-positive filter (default)
  2. predict_keystroke_counts_rule   — rule-based DSP-only (explicit alternative)
  3. predict_keystroke_counts_ml     — RandomForest regressor on YAMNet embeddings

Pipeline flow:
    detect_onsets  →  [filter_onsets_with_yamnet | per-word pooling + predict]  →  segment_into_words
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from onset_detector import detect_onsets
from segmenter import segment_into_words
from yamnet_filter import (
    extract_features,
    extract_yamnet_candidates,
    filter_onsets_with_yamnet,
    load_classifier,
    pool_word_features,
)
from src.yamnet_config import GAP_THRESHOLD_ML, GAP_THRESHOLD_RULE

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_DEFAULT_CLASSIFIER = _MODELS_DIR / "keystroke_classifier.joblib"
_COUNT_PREDICTOR = _MODELS_DIR / "count_predictor_new.joblib"


@dataclass(frozen=True)
class _LoadedCountPredictor:
    model: object
    feature_dimension: int | None


@dataclass(frozen=True)
class PredictionTrace:
    """Prediction output plus the serving-time onset evidence behind it.

    Tuples keep the frozen trace immutable while remaining directly encodable
    as JSON arrays. ``counted_onsets`` is the exact onset sequence passed to
    serving-time word segmentation, and ``word_groups`` is that segmenter's
    output. For the YAMNet path this means classifier-filtered onsets; the rule
    and count-regressor paths segment the detected onsets directly.
    """

    counts: tuple[int, ...]
    counted_onsets: tuple[float, ...]
    word_groups: tuple[tuple[float, ...], ...]


def _make_prediction_trace(
    counts,
    counted_onsets,
    word_groups,
) -> PredictionTrace:
    """Normalize NumPy scalars and mutable segmenter output for a trace."""
    return PredictionTrace(
        counts=tuple(int(count) for count in counts),
        counted_onsets=tuple(float(onset) for onset in counted_onsets),
        word_groups=tuple(
            tuple(float(onset) for onset in group) for group in word_groups
        ),
    )


def _optional_feature_dimension(value: object, *, source: str) -> int | None:
    """Validate a serialized/model feature count without requiring it."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{source} must be a positive integer")
    dimension = int(value)
    if dimension <= 0:
        raise ValueError(f"{source} must be a positive integer")
    return dimension


@lru_cache(maxsize=4)
def _load_count_predictor(model_path: str) -> _LoadedCountPredictor:
    """Load and validate a trusted count-predictor artifact once per path."""
    payload = joblib.load(model_path)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"{model_path} is not a supported count predictor file")

    model = payload["model"]
    if not callable(getattr(model, "predict", None)):
        raise ValueError(
            f"Count predictor in {model_path} must provide a callable predict method"
        )

    payload_dimension = _optional_feature_dimension(
        payload.get("feature_dimension"),
        source=f"feature_dimension in {model_path}",
    )
    model_dimension = _optional_feature_dimension(
        getattr(model, "n_features_in_", None),
        source=f"n_features_in_ on the count predictor in {model_path}",
    )
    if (
        payload_dimension is not None
        and model_dimension is not None
        and payload_dimension != model_dimension
    ):
        raise ValueError(
            f"Count predictor feature dimension mismatch in {model_path}: "
            f"payload declares {payload_dimension}, model expects {model_dimension}"
        )

    return _LoadedCountPredictor(
        model=model,
        feature_dimension=payload_dimension or model_dimension,
    )


def _model_file_status(path: object, loader) -> dict[str, object]:
    """Return safe readiness metadata for one local model artifact."""
    path_text = str(path)
    exists = bool(path.exists())
    status: dict[str, object] = {
        "path": path_text,
        "exists": exists,
        "ready": False,
        "error": None,
    }
    if not exists:
        status["error"] = f"Model file not found: {path_text}"
        return status

    try:
        loader(path_text)
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
    else:
        status["ready"] = True
    return status


def get_model_status() -> dict[str, object]:
    """Report local artifact readiness without loading YAMNet/TensorFlow Hub.

    This checks only the two trusted joblib artifacts and their estimator
    interfaces.  The successful loads are cached for subsequent predictions.
    """
    yamnet_classifier = _model_file_status(_DEFAULT_CLASSIFIER, load_classifier)
    count_predictor = _model_file_status(_COUNT_PREDICTOR, _load_count_predictor)
    all_present = bool(
        yamnet_classifier["exists"] and count_predictor["exists"]
    )
    all_ready = bool(yamnet_classifier["ready"] and count_predictor["ready"])
    return {
        "all_present": all_present,
        "all_ready": all_ready,
        "yamnet_classifier": yamnet_classifier,
        "count_predictor": count_predictor,
    }


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
    filtering → word segmentation). The trained classifier is required;
    callers that want DSP-only behavior must select the rule pipeline.

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
    return list(
        predict_keystroke_counts_with_trace(
            wav_path,
            threshold=threshold,
            delta=delta,
            gap_threshold=gap_threshold,
            merge_gap_seconds=merge_gap_seconds,
        ).counts
    )


def predict_keystroke_counts_with_trace(
    wav_path: str,
    threshold: float = 0.5,
    delta: float = 0.07,
    gap_threshold: float = GAP_THRESHOLD_ML,
    merge_gap_seconds: float = 0.06,
    classifier_path: str | Path | None = None,
) -> PredictionTrace:
    """Run the served YAMNet path and return its segmentation evidence."""
    if gap_threshold is None:
        gap_threshold = GAP_THRESHOLD_ML

    selected_classifier = (
        Path(classifier_path).resolve()
        if classifier_path is not None
        else _DEFAULT_CLASSIFIER
    )
    if not selected_classifier.exists():
        raise FileNotFoundError(
            f"YAMNet classifier model not found: {selected_classifier}. "
            "Run train_yamnet_classifier.py first or use method=rule."
        )

    # ── Step 1: Rule-based onset detection (DSP) ──────────────────────────
    onsets, _, _ = detect_onsets(
        wav_path,
        delta=delta,
        min_gap_seconds=merge_gap_seconds,
    )

    if len(onsets) == 0:
        return _make_prediction_trace([], [], [])

    # ── Step 2: YAMNet false-positive filtering ───────────────────────────
    filtered = filter_onsets_with_yamnet(
        onsets,
        wav_path,
        str(selected_classifier),
        confidence_threshold=threshold,
    )
    if len(filtered) == 0:
        return _make_prediction_trace([], [], [])
    onset_times = filtered

    # ── Step 3: Segment onsets into words ─────────────────────────────────
    counts, word_groups = segment_into_words(
        onset_times,
        gap_threshold=gap_threshold,
    )

    return _make_prediction_trace(counts, onset_times, word_groups)


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
    return list(
        predict_keystroke_counts_rule_with_trace(
            wav_path,
            delta=delta,
            gap_threshold=gap_threshold,
            merge_gap_seconds=merge_gap_seconds,
        ).counts
    )


def predict_keystroke_counts_rule_with_trace(
    wav_path: str,
    delta: float = 0.07,
    gap_threshold: float = GAP_THRESHOLD_RULE,
    merge_gap_seconds: float = 0.06,
) -> PredictionTrace:
    """Run the served DSP-only path and return its segmentation evidence."""
    if gap_threshold is None:
        gap_threshold = GAP_THRESHOLD_RULE

    onsets, _, _ = detect_onsets(
        wav_path,
        delta=delta,
        min_gap_seconds=merge_gap_seconds,
    )

    if len(onsets) == 0:
        return _make_prediction_trace([], [], [])

    counts, word_groups = segment_into_words(
        onsets,
        gap_threshold=gap_threshold,
    )
    return _make_prediction_trace(counts, onsets, word_groups)


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
    return list(
        predict_keystroke_counts_ml_with_trace(
            wav_path,
            delta=delta,
            gap_threshold=gap_threshold,
            merge_gap_seconds=merge_gap_seconds,
        ).counts
    )


def predict_keystroke_counts_ml_with_trace(
    wav_path: str,
    delta: float = 0.07,
    gap_threshold: float = GAP_THRESHOLD_ML,
    merge_gap_seconds: float = 0.06,
    count_model_path: str | Path | None = None,
) -> PredictionTrace:
    """Run the served count-regressor path and return its onset evidence."""
    if gap_threshold is None:
        gap_threshold = GAP_THRESHOLD_ML

    selected_count_model = (
        Path(count_model_path).resolve()
        if count_model_path is not None
        else _COUNT_PREDICTOR
    )
    if not selected_count_model.exists():
        raise FileNotFoundError(
            f"Count predictor model not found: {selected_count_model}. "
            f"Run train_model.py first."
        )

    # Load and validate the trusted local artifact once, then reuse it.
    artifact = _load_count_predictor(str(selected_count_model))
    model = artifact.model

    # Step 1: Detect onsets
    onsets, _, _ = detect_onsets(wav_path, delta=delta, min_gap_seconds=merge_gap_seconds)
    if len(onsets) == 0:
        return _make_prediction_trace([], [], [])

    # Step 2: Segment into words
    _, word_groups = segment_into_words(onsets, gap_threshold=gap_threshold)

    # Step 3: Extract YAMNet features for all onsets at once
    candidates = extract_yamnet_candidates(wav_path, onsets)
    if not candidates:
        return _make_prediction_trace(
            [1] * len(word_groups),
            onsets,
            word_groups,
        )

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

        word_feat = np.asarray(
            pool_word_features(all_features[indices], candidate_times[indices])
        ).reshape(1, -1)
        if (
            artifact.feature_dimension is not None
            and word_feat.shape[1] != artifact.feature_dimension
        ):
            raise ValueError(
                "Count predictor feature dimension mismatch: "
                f"model expects {artifact.feature_dimension}, "
                f"inference produced {word_feat.shape[1]}"
            )
        raw_pred = float(model.predict(word_feat)[0])
        predicted_counts.append(max(1, round(raw_pred)))

    return _make_prediction_trace(predicted_counts, onsets, word_groups)
