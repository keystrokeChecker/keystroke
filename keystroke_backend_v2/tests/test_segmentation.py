import pytest
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from audio.segmentation import extract_event_window, segment_continuous_audio
from audio.onset import evaluate_onsets


def test_extract_event_window_bounds():
    sr = 22050
    audio = np.ones(22050, dtype=np.float32)

    # Normal window
    win1 = extract_event_window(audio, center_sample=10000, window_samples=4410)
    assert len(win1) == 4410

    # Boundary window near start
    win2 = extract_event_window(audio, center_sample=100, window_samples=4410)
    assert len(win2) == 4410

    # Boundary window near end
    win3 = extract_event_window(audio, center_sample=22000, window_samples=4410)
    assert len(win3) == 4410


def test_onset_evaluation():
    gt = np.array([1.0, 2.0, 3.0])
    det = np.array([1.01, 2.02, 5.0])  # 2 correct within 50ms, 1 false positive

    eval_res = evaluate_onsets(det, gt, tolerance_ms=50.0)
    assert eval_res["tp"] == 2
    assert eval_res["fp"] == 1
    assert eval_res["fn"] == 1
    assert eval_res["precision"] == pytest.approx(2 / 3)
    assert eval_res["recall"] == pytest.approx(2 / 3)
