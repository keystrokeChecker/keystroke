"""YAMNet embedding extraction and learned false-positive filtering."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import librosa
import numpy as np

YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
YAMNET_SAMPLE_RATE = 16_000
DEFAULT_WINDOW_SECONDS = 0.320
YAMNET_MIN_INPUT_SECONDS = 0.975

_TF = None
_YAMNET_MODEL = None
_CLASS_INDICES: dict[str, int] | None = None


@dataclass(frozen=True)
class YAMNetCandidate:
    onset_time: float
    embedding: np.ndarray
    typing_score: float
    computer_keyboard_score: float

    @property
    def audio_set_confidence(self) -> float:
        return float(max(self.typing_score, self.computer_keyboard_score))


def _backend_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_yamnet():
    """Lazily load YAMNet, so the original non-YAMNet pipeline is unaffected."""
    global _TF, _YAMNET_MODEL
    if _YAMNET_MODEL is None:
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
        except ImportError as exc:
            raise ImportError(
                "YAMNet filtering requires tensorflow and tensorflow-hub. "
                "Install the packages listed in requirements.txt."
            ) from exc
        _TF = tf
        print(f"Loading YAMNet from TensorFlow Hub: {YAMNET_HANDLE}")
        _YAMNET_MODEL = hub.load(YAMNET_HANDLE)
    return _TF, _YAMNET_MODEL


def _load_class_indices() -> dict[str, int]:
    """Load YAMNet's AudioSet class map from the model asset or a local cache."""
    global _CLASS_INDICES
    if _CLASS_INDICES is not None:
        return _CLASS_INDICES

    cache_path = _backend_dir() / "models" / "yamnet_class_map.csv"
    text: str | None = None

    # 1. Try local cache first
    if cache_path.exists():
        text = cache_path.read_text(encoding="utf-8")

    # 2. Extract from the loaded YAMNet model (bundled asset)
    if text is None:
        _, model = _load_yamnet()
        try:
            map_path = model.class_map_path().numpy().decode("utf-8")
            text = Path(map_path).read_text(encoding="utf-8")
            print(f"Loaded class map from YAMNet model asset: {map_path}")
            # Cache locally for faster subsequent runs
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(text, encoding="utf-8")
        except Exception as exc:
            raise RuntimeError(
                "Could not extract the class map from the YAMNet model. "
                "Ensure YAMNet loaded correctly from TensorFlow Hub."
            ) from exc

    class_map = {
        row["display_name"].strip().casefold(): int(row["index"])
        for row in csv.DictReader(text.splitlines())
    }
    required = ("typing", "computer keyboard")
    missing = [name for name in required if name not in class_map]
    if missing:
        raise RuntimeError("YAMNet class map missing: " + ", ".join(missing))
    _CLASS_INDICES = {name: class_map[name] for name in required}
    return _CLASS_INDICES


def _centered_yamnet_window(
    waveform: np.ndarray, onset_time: float, sample_rate: int, window_seconds: float
) -> np.ndarray:
    """Extract +/- window_seconds/2, padding both recording and model edges."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be greater than zero")
    context_samples = max(1, int(round(window_seconds * sample_rate)))
    model_samples = max(context_samples, int(round(YAMNET_MIN_INPUT_SECONDS * sample_rate)))
    context = np.zeros(context_samples, dtype=np.float32)
    centre = int(round(onset_time * sample_rate))
    source_start = centre - context_samples // 2
    source_end = source_start + context_samples
    copy_start, copy_end = max(0, source_start), min(len(waveform), source_end)
    if copy_end > copy_start:
        destination_start = copy_start - source_start
        context[destination_start : destination_start + copy_end - copy_start] = waveform[
            copy_start:copy_end
        ]
    padded = np.zeros(model_samples, dtype=np.float32)
    padded_start = (model_samples - context_samples) // 2
    padded[padded_start : padded_start + context_samples] = context
    return padded


def extract_yamnet_candidates(
    wav_path: str | os.PathLike[str],
    onsets: Iterable[float],
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> list[YAMNetCandidate]:
    """Return one mean 1024-D YAMNet embedding and AudioSet scores per onset."""
    onset_array = np.asarray(list(onsets), dtype=float)
    if onset_array.size == 0:
        return []
    tf, model = _load_yamnet()
    class_indices = _load_class_indices()
    waveform, sample_rate = librosa.load(str(wav_path), sr=YAMNET_SAMPLE_RATE, mono=True)
    
    # Preprocess: Peak normalization & Noise-floor estimation
    from onset_detector import normalize_recording
    waveform, rec_noise_floor = normalize_recording(waveform)

    result: list[YAMNetCandidate] = []
    for onset in onset_array:
        window = _centered_yamnet_window(waveform, float(onset), sample_rate, window_seconds)
        # Explicit CPU placement means no GPU is required.
        with tf.device("/CPU:0"):
            scores, embeddings, _ = model(tf.convert_to_tensor(window, dtype=tf.float32))
        scores, embeddings = np.asarray(scores.numpy()), np.asarray(embeddings.numpy())
        if scores.size == 0 or embeddings.size == 0:
            raise RuntimeError("YAMNet returned no frames for a padded candidate window")
        mean_scores = scores.mean(axis=0)
        result.append(
            YAMNetCandidate(
                onset_time=float(onset),
                embedding=embeddings.mean(axis=0).astype(np.float32),
                typing_score=float(mean_scores[class_indices["typing"]]),
                computer_keyboard_score=float(mean_scores[class_indices["computer keyboard"]]),
            )
        )
    return result


def score_onsets_with_yamnet(
    onsets: Iterable[float], wav_path: str | os.PathLike[str]
) -> list[dict[str, object]]:
    """Expose embeddings, both AudioSet scores, and raw YAMNet confidence."""
    return [
        {
            "onset_time": item.onset_time,
            "embedding": item.embedding,
            "typing_score": item.typing_score,
            "computer_keyboard_score": item.computer_keyboard_score,
            "confidence": item.audio_set_confidence,
        }
        for item in extract_yamnet_candidates(wav_path, onsets)
    ]


def candidate_embedding_matrix(candidates: Iterable[YAMNetCandidate]) -> np.ndarray:
    values = list(candidates)
    if not values:
        return np.empty((0, 1024), dtype=np.float32)
    return np.stack([item.embedding for item in values]).astype(np.float32, copy=False)


def extract_features(candidates: Iterable[YAMNetCandidate], wav_path: str | os.PathLike[str] = "") -> np.ndarray:
    """
    Extract raw 1024-D YAMNet embeddings for the classifier.
    """
    return candidate_embedding_matrix(candidates)


def load_classifier(classifier_path: str | os.PathLike[str]):
    path = Path(classifier_path)
    if not path.exists():
        raise FileNotFoundError(
            f"YAMNet classifier not found: {path}. Run train_yamnet_classifier.py first."
        )
    payload = joblib.load(path)
    if isinstance(payload, dict) and "pipeline" in payload:
        return payload["pipeline"]
    if hasattr(payload, "predict_proba"):
        return payload
    raise ValueError(f"{path} is not a supported YAMNet classifier file")


def filter_onsets_with_yamnet(
    onsets: Iterable[float],
    wav_path: str | os.PathLike[str],
    classifier_path: str | os.PathLike[str],
    confidence_threshold: float = 0.15,
) -> np.ndarray:
    """Return only raw onset candidates classified as valid keystrokes by the model."""
    candidates = extract_yamnet_candidates(wav_path, onsets)
    if not candidates:
        return np.array([], dtype=float)
    
    features = extract_features(candidates, wav_path)
    model = load_classifier(classifier_path)
    
    filtered_onsets = []
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[:, 1]
        print(f"\n[YAMNet Filter] Using threshold = {confidence_threshold:.2f}", flush=True)
        for candidate, score in zip(candidates, probabilities):
            status = "KEEP" if score >= confidence_threshold else "DROP"
            print(f"  -> Onset at {candidate.onset_time:.3f}s | ML Score: {score:.4f} | {status}", flush=True)
            if score >= confidence_threshold:
                filtered_onsets.append(candidate.onset_time)
    else:
        predictions = model.predict(features)
        print(f"\n[YAMNet Filter] Using direct model.predict()", flush=True)
        for candidate, pred in zip(candidates, predictions):
            is_keystroke = bool(pred > 0)
            status = "KEEP" if is_keystroke else "DROP"
            print(f"  -> Onset at {candidate.onset_time:.3f}s | Predicted Class: {pred} | {status}", flush=True)
            if is_keystroke:
                filtered_onsets.append(candidate.onset_time)
            
    return np.asarray(filtered_onsets, dtype=float)

