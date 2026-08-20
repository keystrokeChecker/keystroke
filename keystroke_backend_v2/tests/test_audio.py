import pytest
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from audio.io import resample_audio
from audio.normalization import apply_normalization
from audio.quality import check_audio_quality


def test_resample_audio():
    sr_orig = 44100
    sr_target = 22050
    duration_s = 0.5
    t = np.linspace(0, duration_s, int(sr_orig * duration_s))
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    resampled = resample_audio(sine, sr_orig, sr_target)
    assert len(resampled) == int(sr_target * duration_s)
    assert resampled.dtype == np.float32


def test_normalization_strategies():
    audio = np.array([0.1, -0.5, 0.8, -0.2], dtype=np.float32)

    # P1 (None)
    p1 = apply_normalization(audio, "P1")
    assert np.allclose(p1, audio)

    # P2 (Peak)
    p2 = apply_normalization(audio, "P2")
    assert np.max(np.abs(p2)) == pytest.approx(1.0)

    # P3 (RMS)
    p3 = apply_normalization(audio, "P3")
    assert np.sqrt(np.mean(p3 ** 2)) == pytest.approx(0.1)


def test_audio_quality_check():
    silent = np.zeros(1000, dtype=np.float32)
    q_silent = check_audio_quality(silent, 22050)
    assert q_silent["is_valid"] is False
    assert q_silent["is_silent"] is True

    valid_audio = np.random.uniform(-0.5, 0.5, 22050).astype(np.float32)
    q_valid = check_audio_quality(valid_audio, 22050)
    assert q_valid["is_valid"] is True
