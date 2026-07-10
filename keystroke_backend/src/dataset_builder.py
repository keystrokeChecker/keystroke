import argparse
import csv
from pathlib import Path

import librosa
import numpy as np

try:
    from .yamnet_config import HOP_SECONDS, TARGET_SAMPLE_RATE, WINDOW_SECONDS, MODIFIER_KEYS
    from .yamnet_features import get_embedding
except ImportError:
    from yamnet_config import HOP_SECONDS, TARGET_SAMPLE_RATE, WINDOW_SECONDS, MODIFIER_KEYS
    from yamnet_features import get_embedding


BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"
OUTPUT_DIR = DATA_DIR / "yamnet_dataset"


def discover_session_names():
    return [log_path.stem.replace("_log", "") for log_path in sorted(DATA_DIR.glob("*_log.csv"))]


def load_positive_timestamps(log_path):
    """Load keystroke timestamps, excluding word boundaries and modifier keys."""
    timestamps = []
    with open(log_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("is_word_boundary") == "True":
                continue
            # Modifier keys (Shift, Ctrl, Alt, Cmd) produce no distinct
            # keystroke sound, so exclude them from positive training labels.
            if row.get("key", "") in MODIFIER_KEYS:
                continue
            timestamps.append(float(row["timestamp_sec"]))
    return np.asarray(timestamps, dtype=np.float32)


def iter_windows(audio, sr, window_seconds=WINDOW_SECONDS, hop_seconds=HOP_SECONDS):
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

    for start in starts:
        end = start + window_samples
        segment = audio[start:end]
        if len(segment) < window_samples:
            segment = np.pad(segment, (0, window_samples - len(segment)))
        center = (start + window_samples / 2.0) / sr
        yield center, segment


def build_from_session(wav_path, log_path, window_seconds=WINDOW_SECONDS, hop_seconds=HOP_SECONDS):
    audio, sr = librosa.load(wav_path, sr=TARGET_SAMPLE_RATE, mono=True)
    audio = audio.astype(np.float32)
    positive_times = load_positive_timestamps(log_path)

    X, y = [], []
    half_window = window_seconds / 2.0

    for center_time, segment in iter_windows(audio, sr, window_seconds, hop_seconds):
        label = int(np.any(np.abs(positive_times - center_time) <= half_window))
        embedding = get_embedding(segment, sr)
        X.append(embedding)
        y.append(label)

    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="*", help="Session names without extensions.")
    parser.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS)
    parser.add_argument("--hop-seconds", type=float, default=HOP_SECONDS)
    args = parser.parse_args()

    session_names = args.names or discover_session_names()
    if not session_names:
        raise FileNotFoundError("No sessions found in data/.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = [], []
    for name in session_names:
        wav_path = DATA_DIR / f"{name}.wav"
        log_path = DATA_DIR / f"{name}_log.csv"
        if not wav_path.exists():
            print(f"Skipping missing audio: {wav_path}")
            continue
        if not log_path.exists():
            print(f"Skipping missing log: {log_path}")
            continue

        session_X, session_y = build_from_session(
            wav_path,
            log_path,
            window_seconds=args.window_seconds,
            hop_seconds=args.hop_seconds,
        )
        X.extend(session_X)
        y.extend(session_y)
        print(f"{name}: {len(session_X)} windows")

    if not X:
        raise RuntimeError("No training windows were built.")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    np.save(OUTPUT_DIR / "X.npy", X)
    np.save(OUTPUT_DIR / "y.npy", y)

    print(f"Saved dataset to {OUTPUT_DIR}")
    print(f"Samples: {len(X)}")
    print(f"Positive ratio: {y.mean():.1%}")


if __name__ == "__main__":
    main()
