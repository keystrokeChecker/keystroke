from pathlib import Path

import numpy as np

from onset_detector import detect_onsets
from segmenter import segment_into_words
from yamnet_filter import filter_onsets_with_yamnet

DEFAULT_CLASSIFIER_PATH = Path(__file__).resolve().parent.parent / "models" / "yamnet_keystroke_classifier.joblib"


from src.yamnet_config import CLASSIFIER_THRESHOLD


def predict_keystroke_counts(
    wav_path: str,
    threshold: float = CLASSIFIER_THRESHOLD,
    delta: float = 0.07,
    gap_threshold: float = 0.4,
    merge_gap_seconds: float = 0.07,
    noise_gate_factor: float = 4.0,
    min_centroid_hz: float = 1000.0,
    min_spectral_flatness: float = 0.08,
    max_decay_ratio: float = 1.0,   # effectively disables this gate for now
    min_gap_seconds: float = 0.06,
):
    """Predict keystroke counts per word from a recorded WAV file."""
    onsets, y, sr = detect_onsets(
        wav_path,
        delta=delta,
        noise_gate_factor=noise_gate_factor,
        min_centroid_hz=min_centroid_hz,
        min_spectral_flatness=min_spectral_flatness,
        max_decay_ratio=max_decay_ratio,
        min_gap_seconds=min_gap_seconds,
    )
    raw_peak = float(np.max(np.abs(y))) if len(y) else 0.0
    print(f"[predict] sr={sr} duration={len(y)/sr:.2f}s peak={raw_peak:.5f} raw_onsets={len(onsets)}")
    if len(onsets) == 0:
        return [0]

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

        except (ImportError, FileNotFoundError, RuntimeError) as e:
            import traceback
            traceback.print_exc()
            filtered_onsets = onsets

    counts, _ = segment_into_words(filtered_onsets, gap_threshold=gap_threshold)
    if not counts:
        counts = [0]
    print(f"[predict] filtered_onsets={len(filtered_onsets)} counts={counts}")
    return counts