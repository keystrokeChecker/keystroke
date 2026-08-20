import numpy as np
import librosa
from typing import Tuple


def extract_spectral_features(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 256
) -> np.ndarray:
    """
    Extracts summary statistics for spectral contrast, centroid, bandwidth, flatness, rolloff, ZCR, RMS.
    """
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)), mode='constant')

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop_length)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=hop_length)
    rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)

    feats = []
    for mat in [centroid, bandwidth, contrast, flatness, rolloff, zcr, rms]:
        feats.append(np.mean(mat, axis=1))
        feats.append(np.std(mat, axis=1))

    return np.concatenate(feats).astype(np.float32)
