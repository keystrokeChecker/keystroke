import argparse
from pathlib import Path

import joblib
import librosa
import numpy as np
from scipy.signal import find_peaks

try:
    from .yamnet_config import (
        CLASSIFIER_THRESHOLD,
        GAP_THRESHOLD,
        HOP_SECONDS,
        MERGE_GAP_SECONDS,
        SENSITIVITY_DELTA,
        SMOOTHING_WINDOW,
        TARGET_SAMPLE_RATE,
        WINDOW_SECONDS,
    )
    from .yamnet_features import get_embedding
except ImportError:
    from yamnet_config import (
        CLASSIFIER_THRESHOLD,
        GAP_THRESHOLD,
        HOP_SECONDS,
        MERGE_GAP_SECONDS,
        SENSITIVITY_DELTA,
        SMOOTHING_WINDOW,
        TARGET_SAMPLE_RATE,
        WINDOW_SECONDS,
    )
    from yamnet_features import get_embedding


BACKEND_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BACKEND_DIR / "models" / "yamnet_keystroke_classifier.joblib"


def load_model(model_path=None):
    path = Path(model_path) if model_path else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"YAMNet classifier not found at {path}. "
            "Run: python -m src.run_yamnet_pipeline"
        )

    return joblib.load(path)


def _prepare_windows(audio, sr, window_seconds=WINDOW_SECONDS, hop_seconds=HOP_SECONDS):
    window_samples = max(1, int(round(window_seconds * sr)))
    hop_samples = max(1, int(round(hop_seconds * sr)))

    if len(audio) < window_samples:
        audio = np.pad(audio, (0, window_samples - len(audio)))

    last_start = max(0, len(audio) - window_samples)
    starts = list(range(0, last_start + 1, hop_samples))
    if starts and starts[-1] != last_start:
        starts.append(last_start)
    if not starts:
        starts = [0]

    segments = []
    centers = []
    for start in starts:
        end = start + window_samples
        segment = audio[start:end]
        if len(segment) < window_samples:
            segment = np.pad(segment, (0, window_samples - len(segment)))
        segments.append(segment)
        centers.append((start + window_samples / 2.0) / sr)

    return np.asarray(segments), np.asarray(centers, dtype=np.float32)


def _merge_close_timestamps(timestamps, min_gap=MERGE_GAP_SECONDS):
    if len(timestamps) == 0:
        return timestamps

    merged = [float(timestamps[0])]
    for t in timestamps[1:]:
        if t - merged[-1] <= min_gap:
            merged[-1] = (merged[-1] + float(t)) / 2.0
        else:
            merged.append(float(t))
    return np.asarray(merged, dtype=np.float32)


def predict_timestamps(
    wav_path,
    model_path=None,
    threshold=CLASSIFIER_THRESHOLD,
    delta=SENSITIVITY_DELTA,
    window_seconds=WINDOW_SECONDS,
    hop_seconds=HOP_SECONDS,
    merge_gap_seconds=MERGE_GAP_SECONDS,
):
    audio, sr = librosa.load(wav_path, sr=TARGET_SAMPLE_RATE, mono=True)
    audio = audio.astype(np.float32)

    model = load_model(model_path)
    segments, centers = _prepare_windows(audio, sr, window_seconds, hop_seconds)

    embeddings = np.vstack([get_embedding(segment, sr) for segment in segments])
    probs = model.predict_proba(embeddings)[:, 1]

    if len(probs) > 1:
        smooth_window = min(SMOOTHING_WINDOW, len(probs))
        kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
        probs = np.convolve(probs, kernel, mode="same")

    effective_threshold = float(np.clip(threshold + (delta - SENSITIVITY_DELTA) * 0.4, 0.05, 0.95))
    peak_distance = max(1, int(round(merge_gap_seconds / hop_seconds)))
    peaks, _ = find_peaks(probs, height=effective_threshold, distance=peak_distance)

    timestamps = centers[peaks]
    return _merge_close_timestamps(timestamps, min_gap=merge_gap_seconds), audio, sr


def detect_onsets(
    wav_path,
    delta=SENSITIVITY_DELTA,
    threshold=CLASSIFIER_THRESHOLD,
    window_seconds=WINDOW_SECONDS,
    hop_seconds=HOP_SECONDS,
    merge_gap_seconds=MERGE_GAP_SECONDS,
    model_path=None,
):
    return predict_timestamps(
        wav_path,
        model_path=model_path,
        threshold=threshold,
        delta=delta,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
        merge_gap_seconds=merge_gap_seconds,
    )


def _segment_into_words(onset_times, gap_threshold=0.4):
    if len(onset_times) == 0:
        return []

    word_counts = []
    current_count = 1

    for index in range(1, len(onset_times)):
        gap = float(onset_times[index] - onset_times[index - 1])
        if gap > gap_threshold:
            word_counts.append(current_count)
            current_count = 1
        else:
            current_count += 1

    word_counts.append(current_count)
    return word_counts


def predict_keystroke_counts(
    wav_path,
    threshold=CLASSIFIER_THRESHOLD,
    delta=SENSITIVITY_DELTA,
    window_seconds=WINDOW_SECONDS,
    hop_seconds=HOP_SECONDS,
    gap_threshold=GAP_THRESHOLD,
    merge_gap_seconds=MERGE_GAP_SECONDS,
    model_path=None,
):
    timestamps, _, _ = detect_onsets(
        wav_path,
        delta=delta,
        threshold=threshold,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
        merge_gap_seconds=merge_gap_seconds,
        model_path=model_path,
    )
    return _segment_into_words(timestamps, gap_threshold=gap_threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path")
    parser.add_argument("--threshold", type=float, default=CLASSIFIER_THRESHOLD)
    parser.add_argument("--delta", type=float, default=SENSITIVITY_DELTA)
    parser.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS)
    parser.add_argument("--hop-seconds", type=float, default=HOP_SECONDS)
    args = parser.parse_args()

    timestamps, _, _ = detect_onsets(
        args.wav_path,
        delta=args.delta,
        threshold=args.threshold,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
    )

    print(f"Detected {len(timestamps)} keystroke events")
    for index, timestamp in enumerate(timestamps, start=1):
        print(f"{index:03d}: {timestamp:.3f} sec")

