import numpy as np
import librosa
from typing import Tuple


def extract_log_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 22050,
    n_mels: int = 80,
    win_length_ms: float = 25.0,
    hop_length_ms: float = 10.0,
    target_frames: int = 32
) -> np.ndarray:
    """
    Extracts 2D Log-Mel Spectrogram preserving temporal structure.
    Output shape: (n_mels, target_frames) e.g., (80, 32).
    """
    n_fft = int((win_length_ms / 1000.0) * sr)
    # Ensure n_fft is power of 2 >= win_length
    n_fft = 2 ** int(np.ceil(np.log2(n_fft)))
    hop_length = int((hop_length_ms / 1000.0) * sr)

    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)), mode='constant')

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        win_length=int((win_length_ms / 1000.0) * sr),
        hop_length=hop_length,
        n_mels=n_mels
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize log_mel to range [-1, 1] or mean=0, std=1
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)

    # Pad or crop temporal frames to target_frames
    n_mels_actual, n_frames = log_mel.shape
    if n_frames < target_frames:
        pad_width = target_frames - n_frames
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
    elif n_frames > target_frames:
        log_mel = log_mel[:, :target_frames]

    return log_mel.astype(np.float32)
