import pytest
import numpy as np
import io
import soundfile as sf
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.inference.engine import InferenceEngine


def test_offline_inference_consistency():
    engine = InferenceEngine()
    if not engine.is_loaded:
        pytest.skip("Model not loaded yet.")

    # Create synthetic WAV bytes
    audio = np.random.uniform(-0.2, 0.2, 22050).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, 22050, format='WAV')
    wav_bytes = buf.getvalue()

    res1 = engine.predict_single_clip(wav_bytes)
    res2 = engine.predict_single_clip(wav_bytes)

    assert res1["prediction"] == res2["prediction"]
    assert res1["confidence"] == res2["confidence"]
    assert len(res1["top_k"]) == 3
