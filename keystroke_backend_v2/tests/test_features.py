import pytest
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from features.extractor import extract_features
from features.mel import extract_log_mel_spectrogram


def test_feature_sets_shape_and_validity():
    sr = 22050
    audio = np.random.uniform(-0.5, 0.5, int(sr * 0.200)).astype(np.float32)

    # Set A
    feat_a = extract_features(audio, sr, feature_set="A")
    assert feat_a.ndim == 1
    assert not np.isnan(feat_a).any()
    assert not np.isinf(feat_a).any()

    # Set B
    feat_b = extract_features(audio, sr, feature_set="B")
    assert len(feat_b) > len(feat_a)

    # Set C
    feat_c = extract_features(audio, sr, feature_set="C")
    assert len(feat_c) > len(feat_b)

    # Set D
    feat_d = extract_features(audio, sr, feature_set="D")
    assert len(feat_d) > len(feat_c)

    # Set E (Log-Mel 2D)
    feat_e = extract_features(audio, sr, feature_set="E", n_mels=80)
    assert feat_e.shape == (80, 32) or feat_e.ndim == 2
    assert not np.isnan(feat_e).any()
