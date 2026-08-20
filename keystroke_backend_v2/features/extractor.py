import numpy as np
from typing import Dict, Any, Union

from features.mfcc import extract_mfcc_features
from features.mel import extract_log_mel_spectrogram
from features.spectral import extract_spectral_features
from features.temporal import extract_temporal_features


def extract_features(
    audio: np.ndarray,
    sr: int,
    feature_set: str = "D",
    n_mels: int = 80,
    n_mfcc: int = 20
) -> np.ndarray:
    """
    Unified feature extractor supporting Feature Sets A through F.
    - Set A: MFCC (means & stds)
    - Set B: MFCC + delta + delta2
    - Set C: MFCC + spectral features
    - Set D: MFCC + spectral + temporal features
    - Set E: 2D Log-Mel Spectrogram (shape: n_mels, target_frames)
    - Set F: 2D Log-Mel + MFCC
    """
    fset = feature_set.upper()

    if fset == "A":
        return extract_mfcc_features(audio, sr, n_mfcc=n_mfcc, include_deltas=False)

    elif fset == "B":
        return extract_mfcc_features(audio, sr, n_mfcc=n_mfcc, include_deltas=True)

    elif fset == "C":
        mfcc = extract_mfcc_features(audio, sr, n_mfcc=n_mfcc, include_deltas=True)
        spec = extract_spectral_features(audio, sr)
        return np.concatenate([mfcc, spec]).astype(np.float32)

    elif fset == "D":
        mfcc = extract_mfcc_features(audio, sr, n_mfcc=n_mfcc, include_deltas=True)
        spec = extract_spectral_features(audio, sr)
        temp = extract_temporal_features(audio, sr)
        return np.concatenate([mfcc, spec, temp]).astype(np.float32)

    elif fset == "E":
        return extract_log_mel_spectrogram(audio, sr=sr, n_mels=n_mels)

    elif fset == "F":
        log_mel = extract_log_mel_spectrogram(audio, sr=sr, n_mels=n_mels)
        mfcc = extract_mfcc_features(audio, sr, n_mfcc=n_mfcc, include_deltas=True)
        # Flatten log_mel and concatenate with MFCCs
        return np.concatenate([log_mel.flatten(), mfcc]).astype(np.float32)

    else:
        raise ValueError(f"Unknown feature_set '{feature_set}'. Must be A, B, C, D, E, or F.")
