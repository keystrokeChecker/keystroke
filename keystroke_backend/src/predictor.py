from pathlib import Path

import numpy as np

from onset_detector import detect_onsets
from segmenter import segment_into_words
from yamnet_filter import filter_onsets_with_yamnet

DEFAULT_CLASSIFIER_PATH = Path(__file__).resolve().parent.parent / "models" / "yamnet_keystroke_classifier.joblib"


def predict_keystroke_counts(
    wav_path: str,
    threshold: float = 0.3,
    delta: float = 0.07,
    gap_threshold: float = 0.4,
    merge_gap_seconds: float = 0.07,
):
    """Predict keystroke counts per word from a recorded WAV file."""
    onsets, _, _ = detect_onsets(wav_path, delta=delta)
    if len(onsets) == 0:
        return []

    use_yamnet = DEFAULT_CLASSIFIER_PATH.exists()
    filtered_onsets = onsets
    if use_yamnet:
        try:
            filtered_onsets = filter_onsets_with_yamnet(
                onsets,
                wav_path,
                classifier_path=str(DEFAULT_CLASSIFIER_PATH),
                confidence_threshold=threshold,
            )
            if len(filtered_onsets) == 0:
                filtered_onsets = onsets
        except (ImportError, FileNotFoundError, RuntimeError):
            filtered_onsets = onsets

    counts, _ = segment_into_words(filtered_onsets, gap_threshold=gap_threshold)
    return counts
