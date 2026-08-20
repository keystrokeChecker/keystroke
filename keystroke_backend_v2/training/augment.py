import numpy as np
from typing import Optional


def augment_gain(audio: np.ndarray, min_gain_db: float = -6.0, max_gain_db: float = 6.0) -> np.ndarray:
    """Random volume gain variation in dB."""
    gain_db = np.random.uniform(min_gain_db, max_gain_db)
    factor = 10.0 ** (gain_db / 20.0)
    return (audio * factor).astype(np.float32)


def augment_additive_noise(audio: np.ndarray, min_snr_db: float = 15.0, max_snr_db: float = 30.0) -> np.ndarray:
    """Adds low-level Gaussian white noise given a target SNR range in dB."""
    signal_power = np.mean(audio ** 2)
    if signal_power <= 1e-10:
        return audio

    snr_db = np.random.uniform(min_snr_db, max_snr_db)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear

    noise = np.random.normal(0, np.sqrt(noise_power), size=audio.shape)
    return (audio + noise).astype(np.float32)


def augment_time_shift(audio: np.ndarray, max_shift_samples: int = 128) -> np.ndarray:
    """Small temporal roll shift."""
    shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)
    return np.roll(audio, shift).astype(np.float32)


def apply_audio_augmentation(
    audio: np.ndarray,
    p_gain: float = 0.5,
    p_noise: float = 0.5,
    p_shift: float = 0.5
) -> np.ndarray:
    """Applies realistic audio augmentation pipeline for training data only."""
    augmented = audio.copy()
    if np.random.rand() < p_gain:
        augmented = augment_gain(augmented)
    if np.random.rand() < p_noise:
        augmented = augment_additive_noise(augmented)
    if np.random.rand() < p_shift:
        augmented = augment_time_shift(augmented)
    return augmented
