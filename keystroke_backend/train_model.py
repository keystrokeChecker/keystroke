"""
Step 5 (M5): Train an ML model that predicts the keystroke count of a word
directly from its audio segment's MFCC features (instead of relying purely
on rule-based onset counting).

This needs MULTIPLE recorded sessions to have enough training data.
Record several sessions first with record_dataset.py, using different phrases,
then run:

    python train_model.py --names session1 session2 session3

It will:
  1. Load each session's audio + ground truth word counts
  2. Split the audio into per-word segments (using ground-truth boundary timestamps)
  3. Extract MFCC features per segment
  4. Train a RandomForestRegressor to predict the count
  5. Save the trained model to models/count_predictor.pkl
"""

import argparse
import csv
import os
import pickle

import librosa
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_keylog_segments(log_path):
    """
    Returns a list of (start_time, end_time, true_count) for each word,
    based on the raw keylog (using word-boundary keys as separators).
    """
    rows = []
    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((float(row["timestamp_sec"]), row["is_word_boundary"] == "True"))

    segments = []
    word_start = None
    count = 0
    last_t = None

    for t, is_boundary in rows:
        if is_boundary:
            if count > 0 and word_start is not None:
                segments.append((word_start, last_t, count))
            word_start = None
            count = 0
        else:
            if word_start is None:
                word_start = t
            count += 1
            last_t = t

    if count > 0 and word_start is not None:
        segments.append((word_start, last_t, count))

    return segments


def extract_mfcc_features(y, sr, start_t, end_t, pad=0.15):
    """Extracts averaged MFCC features for the audio segment [start_t, end_t]."""
    start_sample = max(0, int((start_t - pad) * sr))
    end_sample = min(len(y), int((end_t + pad) * sr))
    segment = y[start_sample:end_sample]

    if len(segment) < 512:
        segment = np.pad(segment, (0, 512 - len(segment)))

    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
    # Aggregate over time: mean + std per coefficient -> fixed-length feature vector
    features = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    return features


def build_dataset(session_names):
    X, y_labels = [], []

    for name in session_names:
        wav_path = os.path.join(DATA_DIR, f"{name}.wav")
        log_path = os.path.join(DATA_DIR, f"{name}_log.csv")

        y_audio, sr = librosa.load(wav_path, sr=None)
        segments = load_keylog_segments(log_path)

        for start_t, end_t, true_count in segments:
            features = extract_mfcc_features(y_audio, sr, start_t, end_t)
            X.append(features)
            y_labels.append(true_count)

        print(f"  {name}: {len(segments)} word segments loaded")

    return np.array(X), np.array(y_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", required=True,
                         help="Session names to include in training, e.g. session1 session2")
    args = parser.parse_args()

    print("Loading dataset...")
    X, y = build_dataset(args.names)
    print(f"Total word samples: {len(X)}")

    if len(X) < 10:
        print("WARNING: very few samples. Record more sessions for a usable model.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rounded_preds = np.round(preds)
    exact_acc = np.mean(rounded_preds == y_test)

    print(f"Test MAE (avg error in # keystrokes): {mae:.2f}")
    print(f"Test exact-match accuracy: {exact_acc:.1%}")

    model_path = os.path.join(MODEL_DIR, "count_predictor.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
