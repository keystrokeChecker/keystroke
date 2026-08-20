import numpy as np
import librosa
from typing import Dict, Any


def check_audio_quality(audio: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Evaluates comprehensive audio quality metrics:
    - RMS energy, Peak amplitude, Crest factor (Peak / RMS)
    - Background noise floor & Dynamic range ratio
    - Estimated SNR (dB)
    - Clipping & Saturation detection
    - Quality tier categorization & user recommendation
    """
    if len(audio) == 0:
        return {
            "is_valid": False,
            "reason": "Empty audio",
            "quality_tier": "unusable",
            "recommendation": "Recording was empty."
        }

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    crest = float(peak / (rms + 1e-6))

    # Clipping detection (> 0.99 amplitude threshold)
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.99))
    is_clipped = clipping_ratio > 0.01

    # Short-time frame RMS analysis
    frame_len = min(len(audio), 512)
    frame_hop = min(len(audio), 128)
    if len(audio) >= 256:
        f_rms = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=frame_hop)[0]
    else:
        f_rms = np.array([rms])

    noise_floor = float(np.percentile(f_rms, 15)) if len(f_rms) > 0 else rms
    peak_frame_rms = float(np.max(f_rms)) if len(f_rms) > 0 else rms
    dynamic_ratio = float(peak_frame_rms / (noise_floor + 1e-6))

    # Estimated Signal-to-Noise Ratio (dB)
    snr_db = float(20.0 * np.log10((peak_frame_rms + 1e-6) / (noise_floor + 1e-6)))

    # Silence & stationary noise check
    is_silent = peak < 0.025 or (dynamic_ratio < 2.0 and peak < 0.060)

    # Quality categorization & actionable recommendation
    if is_silent or peak < 0.015:
        quality_tier = "silent"
        recommendation = "No keyboard sounds detected. Move microphone closer or type clearly."
    elif is_clipped:
        quality_tier = "clipped"
        recommendation = "Audio is clipped/distorted. Lower microphone gain or move mic slightly away."
    elif snr_db < 6.0:
        quality_tier = "low_snr"
        recommendation = "High ambient room noise detected. Reduce background noise or move mic closer."
    else:
        quality_tier = "optimal"
        recommendation = "Audio quality is optimal for keystroke analysis."

    return {
        "is_valid": not (is_silent or is_clipped),
        "peak_amplitude": peak,
        "rms": rms,
        "crest_factor": crest,
        "noise_floor_rms": noise_floor,
        "dynamic_ratio": dynamic_ratio,
        "snr_db": snr_db,
        "clipping_ratio": clipping_ratio,
        "is_clipped": is_clipped,
        "is_silent": is_silent,
        "duration_s": float(len(audio) / sr),
        "quality_tier": quality_tier,
        "recommendation": recommendation
    }
