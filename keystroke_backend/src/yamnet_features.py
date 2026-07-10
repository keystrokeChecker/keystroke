from functools import lru_cache
import os

import numpy as np

try:
    from .yamnet_config import TARGET_SAMPLE_RATE
except ImportError:
    from yamnet_config import TARGET_SAMPLE_RATE

try:
    import tensorflow as tf
    import tensorflow_hub as hub
except ImportError as exc:
    tf = None
    hub = None
    _TF_IMPORT_ERROR = exc
else:
    _TF_IMPORT_ERROR = None


YAMNET_HANDLE = os.environ.get("YAMNET_HANDLE", "https://tfhub.dev/google/yamnet/1")


def _require_tensorflow():
    if tf is None or hub is None:
        raise ImportError(
            "YAMNet requires tensorflow and tensorflow-hub. "
            "Install them with: pip install tensorflow tensorflow-hub"
        ) from _TF_IMPORT_ERROR


@lru_cache(maxsize=1)
def load_yamnet():
    _require_tensorflow()
    return hub.load(YAMNET_HANDLE)


def normalize_audio(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    if audio.size == 0:
        return audio

    audio = audio - np.mean(audio)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return audio.astype(np.float32)


def get_embedding(audio, sr=TARGET_SAMPLE_RATE):
    """Return the 1024-d YAMNet embedding for a waveform segment."""
    if sr != TARGET_SAMPLE_RATE:
        raise ValueError("YAMNet expects audio sampled at 16 kHz.")

    audio = normalize_audio(audio)
    if audio.size == 0:
        raise ValueError("Audio segment is empty.")

    if len(audio) < TARGET_SAMPLE_RATE:
        audio = np.pad(audio, (0, TARGET_SAMPLE_RATE - len(audio)))

    yamnet = load_yamnet()
    waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
    _, embeddings, _ = yamnet(waveform)
    embeddings = embeddings.numpy()

    if embeddings.ndim == 1:
        return embeddings.astype(np.float32)

    return np.mean(embeddings, axis=0).astype(np.float32)
