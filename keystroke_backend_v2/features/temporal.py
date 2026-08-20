import numpy as np
from scipy.stats import skew, kurtosis


def extract_temporal_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Extracts transient temporal features: RMS envelope quarter shapes, peak position, attack time, decay rate, skewness, kurtosis.
    """
    if len(audio) == 0:
        return np.zeros(12, dtype=np.float32)

    abs_audio = np.abs(audio)
    peak_idx = int(np.argmax(abs_audio))
    peak_val = float(abs_audio[peak_idx])

    peak_pos_ratio = peak_idx / len(audio)
    attack_time_ms = (peak_idx / sr) * 1000.0

    # Split audio into 4 quarter segments and compute RMS for each quarter
    q_len = len(audio) // 4
    q_rms = []
    for i in range(4):
        q = audio[i * q_len: (i + 1) * q_len] if i < 3 else audio[i * q_len:]
        rms = np.sqrt(np.mean(q ** 2)) if len(q) > 0 else 0.0
        q_rms.append(rms)

    total_rms = np.sqrt(np.mean(audio ** 2)) + 1e-8
    q_rms_normalized = [r / total_rms for r in q_rms]

    # Skewness and kurtosis of amplitude distribution
    sk = float(skew(audio)) if len(audio) > 2 else 0.0
    kt = float(kurtosis(audio)) if len(audio) > 2 else 0.0

    feats = [
        peak_val,
        peak_pos_ratio,
        attack_time_ms,
        sk,
        kt,
        *q_rms_normalized
    ]
    return np.array(feats, dtype=np.float32)
