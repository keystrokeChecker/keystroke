import numpy as np
from typing import List, Dict, Any, Tuple


def extract_event_window(
    audio: np.ndarray,
    center_sample: int,
    window_samples: int
) -> np.ndarray:
    """Extracts a fixed-length window centered at center_sample, zero-padding if necessary."""
    half_w = window_samples // 2
    start_idx = center_sample - half_w
    end_idx = start_idx + window_samples

    pad_left = 0
    pad_right = 0

    if start_idx < 0:
        pad_left = -start_idx
        start_idx = 0
    if end_idx > len(audio):
        pad_right = end_idx - len(audio)
        end_idx = len(audio)

    clip = audio[start_idx:end_idx]
    if pad_left > 0 or pad_right > 0:
        clip = np.pad(clip, (pad_left, pad_right), mode='constant')

    return clip.astype(np.float32)


def segment_continuous_audio(
    audio: np.ndarray,
    sr: int,
    onset_times_s: np.ndarray,
    window_duration_s: float = 0.200
) -> List[Dict[str, Any]]:
    """Segments continuous audio into event clips given detected onset times in seconds."""
    window_samples = int(window_duration_s * sr)
    segments = []

    for idx, onset_s in enumerate(onset_times_s):
        center_sample = int(onset_s * sr)
        clip = extract_event_window(audio, center_sample, window_samples)
        segments.append({
            "segment_idx": idx,
            "onset_s": float(onset_s),
            "clip": clip,
            "sample_rate": sr
        })

    return segments
