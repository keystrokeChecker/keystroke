import numpy as np
from typing import Tuple

from audio.io import load_audio_from_bytes, resample_audio
from audio.quality import check_audio_quality
from audio.filtering import highpass_filter, preemphasis
from audio.normalization import apply_normalization


def preprocess_audio_bytes(
    audio_bytes: bytes,
    target_sr: int = 22050,
    cutoff_hz: float = 0.0,
    apply_preemph: bool = False,
    normalization_strategy: str = "P2"
) -> Tuple[np.ndarray, int, dict]:
    """
    Decodes audio bytes, checks quality, resamples to target_sr, and applies exact training
    normalization strategy (P2 peak normalization).
    """
    audio, sr = load_audio_from_bytes(audio_bytes, target_sr=target_sr)

    # Quality check
    quality_info = check_audio_quality(audio, sr)
    if not quality_info["is_valid"]:
        print(f"[WARN] Preprocessing audio quality alert: {quality_info}")

    # Highpass filter (if configured)
    if cutoff_hz > 0:
        audio = highpass_filter(audio, sr=sr, cutoff_hz=cutoff_hz)

    # Preemphasis (if configured)
    if apply_preemph:
        audio = preemphasis(audio, coeff=0.97)

    # Normalization
    audio = apply_normalization(audio, strategy=normalization_strategy)

    return audio.astype(np.float32), sr, quality_info
