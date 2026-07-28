"""
Compare custom onset detector parameters against librosa.onset.onset_detect on the same session.
"""

import argparse
import os
import numpy as np
import librosa
from onset_detector import detect_onsets, evaluate_against_ground_truth
from tune_and_evaluate import load_ground_truth

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def main():
    parser = argparse.ArgumentParser(description="Compare onset detection methods.")
    parser.add_argument("--name", default="session_calibrated_test", help="Session name to run on.")
    args = parser.parse_args()

    wav_path = os.path.join(DATA_DIR, f"{args.name}.wav")
    log_path = os.path.join(DATA_DIR, f"{args.name}_log.csv")

    if not os.path.exists(wav_path) or not os.path.exists(log_path):
        print(f"Error: Missing WAV or log files for session {args.name} in {DATA_DIR}")
        return

    # Load ground truth
    _, gt_times = load_ground_truth(log_path)
    gt_times = np.array(gt_times)

    results = []

    # 1. Baseline: custom detector (smoothing_window=7)
    baseline_onsets, y, sr = detect_onsets(wav_path, smoothing_window=7)
    ev_baseline = evaluate_against_ground_truth(baseline_onsets, gt_times, tolerance=0.08)
    results.append({
        "method": "baseline (smoothing=7)",
        "detected": len(baseline_onsets),
        **ev_baseline
    })

    # 2. Reduced Smoothing: custom detector (smoothing_window=3)
    reduced_onsets, _, _ = detect_onsets(wav_path, smoothing_window=3)
    ev_reduced = evaluate_against_ground_truth(reduced_onsets, gt_times, tolerance=0.08)
    results.append({
        "method": "reduced smoothing (smoothing=3)",
        "detected": len(reduced_onsets),
        **ev_reduced
    })

    # 3. Librosa onset_detect with different deltas (0.02, 0.05, 0.07)
    # We load raw audio using librosa.load to get y and sr (or we can use the y, sr from detect_onsets)
    # The instruction says "on the same loaded/filtered audio (y, sr from detect_onsets(), or reload raw if simpler)"
    # Let's use y, sr returned from detect_onsets (which is mono, peak-normalized).
    for librosa_delta in [0.02, 0.05, 0.07]:
        librosa_onsets = librosa.onset.onset_detect(
            y=y,
            sr=sr,
            hop_length=256,
            backtrack=False,
            units='time',
            delta=librosa_delta
        )
        ev_librosa = evaluate_against_ground_truth(librosa_onsets, gt_times, tolerance=0.08)
        results.append({
            "method": f"librosa onset_detect (delta={librosa_delta})",
            "detected": len(librosa_onsets),
            **ev_librosa
        })

    # Sort results by F1 descending
    results.sort(key=lambda x: x["f1"], reverse=True)

    # Print clean comparison table
    print(f"\nMethod Comparison on '{args.name}' (Ground Truth Count: {len(gt_times)}):")
    print("-" * 92)
    print(f"{'Method':35} | {'Detected':8} | {'TP':4} | {'FP':4} | {'FN':4} | {'Precision':9} | {'Recall':6} | {'F1':5}")
    print("-" * 92)
    for r in results:
        print(f"{r['method']:35} | {r['detected']:8d} | {r['true_positives']:4d} | {r['false_positives']:4d} | {r['false_negatives']:4d} | {r['precision']:9.3f} | {r['recall']:.3f} | {r['f1']:.3f}")
    print("-" * 92)

if __name__ == "__main__":
    main()
