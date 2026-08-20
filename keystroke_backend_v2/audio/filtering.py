import numpy as np
from scipy.signal import butter, filtfilt, lfilter


def highpass_filter(audio: np.ndarray, sr: int, cutoff_hz: float = 80.0, order: int = 4) -> np.ndarray:
    """Applies a Butterworth highpass filter to remove low-frequency rumble."""
    nyquist = 0.5 * sr
    if cutoff_hz >= nyquist:
        return audio
    normal_cutoff = cutoff_hz / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered = filtfilt(b, a, audio)
    return filtered.astype(np.float32)


def preemphasis(audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Applies pre-emphasis filter to boost high frequencies."""
    return np.append(audio[0], audio[1:] - coeff * audio[:-1]).astype(np.float32)
