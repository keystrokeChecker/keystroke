"""Production prediction path using the rule detector plus YAMNet filtering."""

from pathlib import Path

import numpy as np

from onset_detector import detect_onsets
from segmenter import segment_into_words
from yamnet_filter import filter_onsets_with_yamnet
from src.yamnet_config import CLASSIFIER_THRESHOLD


DEFAULT_CLASSIFIER_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "yamnet_keystroke_classifier.joblib"
)
def predict_keystroke_counts(
    wav_path: str,
    threshold: float = CLASSIFIER_THRESHOLD,
    typing_speed: str = "auto",
) -> dict:
    """Return YAMNet-filtered counts grouped by timing gaps."""
    onsets, y, sr = detect_onsets(
        wav_path,
        delta=0.07,
        noise_gate_factor=4.0,
        # Keep the corrected 400 Hz floor in the production path.  The old
        # 1000 Hz override rejected real, quieter phone keystrokes before
        # YAMNet could score them.
        min_centroid_hz=400.0,
        min_spectral_flatness=0.08,
        max_decay_ratio=6.0,
        min_gap_seconds=0.06,
        debug=False,
    )
    raw_peak = float(np.max(np.abs(y))) if len(y) else 0.0
    print(
        f"[predict] sr={sr} duration={len(y) / sr:.2f}s "
        f"peak={raw_peak:.5f} raw_onsets={len(onsets)}"
    )

    if len(onsets) == 0:
        return {
            "count": 0,
            "counts": [0],
            "formatted": "0",
            "raw_onsets": 0,
            "yamnet_onsets": 0,
        }

    filtered_onsets = filter_onsets_with_yamnet(
        onsets,
        wav_path,
        classifier_path=str(DEFAULT_CLASSIFIER_PATH),
        confidence_threshold=threshold,
    )
    counts, _ = segment_into_words(filtered_onsets, typing_speed=typing_speed)
    counts = counts or [0]
    total = int(sum(counts))
    formatted = "|".join(str(value) for value in counts)
    print(
        f"[predict] yamnet_onsets={len(filtered_onsets)} "
        f"counts={counts} total={total}"
    )
    return {
        "count": total,
        "counts": counts,
        "formatted": formatted,
        "raw_onsets": int(len(onsets)),
        "yamnet_onsets": int(len(filtered_onsets)),
    }
