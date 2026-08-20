import numpy as np
from typing import List, Dict, Any

from audio.onset import detect_onsets
from audio.segmentation import extract_event_window


def segment_audio_for_inference(
    audio: np.ndarray,
    sr: int,
    window_duration_s: float = 0.200
) -> List[Dict[str, Any]]:
    """
    Detects onsets in continuous audio and extracts centered 200ms windows for classification.
    """
    onset_times_s = detect_onsets(audio, sr=sr)
    window_samples = int(window_duration_s * sr)

    events = []
    for idx, onset_s in enumerate(onset_times_s):
        center_sample = int(onset_s * sr)
        clip = extract_event_window(audio, center_sample, window_samples)
        events.append({
            "event_index": idx,
            "timestamp_s": float(onset_s),
            "audio_clip": clip
        })

    return events
