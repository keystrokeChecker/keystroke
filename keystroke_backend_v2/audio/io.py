import io
import os
import numpy as np
import soundfile as sf
from typing import Tuple, Optional


def load_audio_from_bytes(audio_bytes: bytes, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Decodes audio bytes (.wav, .m4a, etc.) into mono float32 numpy array and sample rate.
    Uses soundfile as primary, with PyAV (av) fallback for container formats like M4A/AAC.
    """
    # 1. Try soundfile
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if target_sr is not None and target_sr != sr:
            data = resample_audio(data, sr, target_sr)
            sr = target_sr
        return data, sr
    except Exception:
        pass

    # 2. Try PyAV (av)
    try:
        import av
        container = av.open(io.BytesIO(audio_bytes))
        stream = container.streams.audio[0]
        sr = stream.codec_context.sample_rate
        
        frames = []
        for frame in container.decode(stream):
            # Convert frame to numpy array
            arr = frame.to_ndarray()
            # If multi-channel, mean across channels
            if arr.ndim > 1:
                arr = np.mean(arr, axis=0)
            frames.append(arr)
            
        data = np.concatenate(frames, axis=0).astype(np.float32)
        # Normalize int16/int32 to float32 range [-1, 1] if needed
        if np.max(np.abs(data)) > 1.0:
            data = data / 32768.0

        if target_sr is not None and target_sr != sr:
            data = resample_audio(data, sr, target_sr)
            sr = target_sr
        return data, sr
    except Exception as e:
        raise ValueError(f"Failed to decode audio bytes: {e}")


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resamples 1D audio array from orig_sr to target_sr using scipy.signal.resample_poly."""
    if orig_sr == target_sr:
        return audio

    from scipy.signal import resample_poly
    import math

    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    resampled = resample_poly(audio, up, down).astype(np.float32)
    return resampled
