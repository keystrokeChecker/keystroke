"""
Final pipeline (M6): Given a NEW audio file (not used in training), predict
the keystroke count per word, using either:
  --method rule   -> onset detection + gap segmentation (Step 2-3 pipeline)
  --method ml     -> trained RandomForest model (Step 5 pipeline), still uses
                      rule-based segmentation to find word boundaries, but ML
                      to predict the count within each segment

USAGE:
    python predict.py data/new_recording.wav --method rule
    python predict.py data/new_recording.wav --method ml
"""

import argparse
import pickle
import os

import librosa
import numpy as np

from onset_detector import detect_onsets
from segmenter import segment_into_words, format_output
from train_model import extract_mfcc_features

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def predict_rule_based(wav_path, threshold=0.4, delta=0.07):
    onsets, y, sr = detect_onsets(wav_path, delta=delta)
    counts, groups = segment_into_words(onsets, gap_threshold=threshold)
    return counts


def predict_ml_based(wav_path, threshold=0.4, delta=0.07):
    model_path = os.path.join(MODEL_DIR, "count_predictor.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("No trained model found. Run train_model.py first.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    onsets, y, sr = detect_onsets(wav_path, delta=delta)
    _, groups = segment_into_words(onsets, gap_threshold=threshold)

    counts = []
    for group in groups:
        start_t, end_t = group[0], group[-1]
        features = extract_mfcc_features(y, sr, start_t, end_t)
        pred = model.predict([features])[0]
        counts.append(int(round(pred)))

    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path")
    parser.add_argument("--method", choices=["rule", "ml"], default="rule")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--delta", type=float, default=0.07)
    args = parser.parse_args()

    if args.method == "rule":
        counts = predict_rule_based(args.wav_path, args.threshold, args.delta)
    else:
        counts = predict_ml_based(args.wav_path, args.threshold, args.delta)

    print(f"Method: {args.method}")
    print(f"Predicted counts: {counts}")
    print(f"Output: {format_output(counts)}")
