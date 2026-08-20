import numpy as np
from typing import Tuple


def normalize_p1_none(audio: np.ndarray) -> np.ndarray:
    """P1: No normalization."""
    return audio.astype(np.float32)


def normalize_p2_peak(audio: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """P2: Peak normalization to max amplitude = 1.0."""
    peak = np.max(np.abs(audio))
    if peak > eps:
        return (audio / peak).astype(np.float32)
    return audio.astype(np.float32)


def normalize_p3_rms(audio: np.ndarray, target_rms: float = 0.1, eps: float = 1e-8) -> np.ndarray:
    """P3: Per-event RMS normalization."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > eps:
        return (audio * (target_rms / rms)).astype(np.float32)
    return audio.astype(np.float32)


def normalize_p4_robust_percentile(audio: np.ndarray, percentile: float = 95.0, eps: float = 1e-8) -> np.ndarray:
    """P4: Robust percentile normalization (scaling by 95th percentile amplitude)."""
    p_val = np.percentile(np.abs(audio), percentile)
    if p_val > eps:
        scaled = audio / p_val
        return np.clip(scaled, -1.0, 1.0).astype(np.float32)
    return audio.astype(np.float32)


def normalize_p5_recording_level(audio: np.ndarray, rec_std: float, eps: float = 1e-8) -> np.ndarray:
    """P5: Recording-level standard deviation normalization."""
    if rec_std > eps:
        return (audio / rec_std).astype(np.float32)
    return audio.astype(np.float32)


def apply_normalization(audio: np.ndarray, strategy: str = "P2") -> np.ndarray:
    """Applies chosen normalization strategy ('P1', 'P2', 'P3', 'P4', 'P5')."""
    strat = strategy.upper()
    if strat == "P1":
        return normalize_p1_none(audio)
    elif strat == "P2":
        return normalize_p2_peak(audio)
    elif strat == "P3":
        return normalize_p3_rms(audio)
    elif strat == "P4":
        return normalize_p4_robust_percentile(audio)
    elif strat == "P5":
        return normalize_p5_recording_level(audio, rec_std=np.std(audio) + 1e-8)
    else:
        raise ValueError(f"Unknown normalization strategy '{strategy}'. Choose P1..P5.")
