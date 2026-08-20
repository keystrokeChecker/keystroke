import numpy as np
import librosa
from typing import Tuple


def extract_mfcc_features(
    audio: np.ndarray,
    sr: int,
    n_mfcc: int = 20,
    n_fft: int = 1024,
    hop_length: int = 256,
    include_deltas: bool = True
) -> np.ndarray:
    """
    Extracts MFCC means, stds, deltas, and delta-deltas.
    """
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)), mode='constant')

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    feats = [mfcc_mean, mfcc_std]

    if include_deltas:
        delta1 = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        feats.extend([
            np.mean(delta1, axis=1), np.std(delta1, axis=1),
            np.mean(delta2, axis=1), np.std(delta2, axis=1)
        ])

    return np.concatenate(feats).astype(np.float32)
