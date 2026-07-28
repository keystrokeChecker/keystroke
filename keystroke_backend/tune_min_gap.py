"""
Grid search script to tune min_gap_seconds for onset_detector.
"""

import argparse
import os
from onset_detector import detect_onsets, evaluate_against_ground_truth
from tune_and_evaluate import load_ground_truth

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def main():
    parser = argparse.ArgumentParser(description="Tune min_gap_seconds for a given session.")
    parser.add_argument("--name", default="session1", help="Session name to evaluate.")
    args = parser.parse_args()

    wav_path = os.path.join(DATA_DIR, f"{args.name}.wav")
    log_path = os.path.join(DATA_DIR, f"{args.name}_log.csv")

    if not os.path.exists(wav_path) or not os.path.exists(log_path):
        print(f"Error: Missing files for session {args.name}")
        return

    _, gt_times = load_ground_truth(log_path)

    min_gaps = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]
    
    print(f"Tuning min_gap_seconds for {args.name} (ground truth count: {len(gt_times)})...")
    print(f"{'min_gap_sec':12} | {'detected':8} | {'precision':9} | {'recall':6} | {'f1':5}")
    print("-" * 55)
    
    for gap in min_gaps:
        # Keeping delta fixed at current default (0.07)
        onsets, _, _ = detect_onsets(wav_path, delta=0.07, min_gap_seconds=gap)
        ev = evaluate_against_ground_truth(onsets, gt_times, tolerance=0.08)
        print(f"{gap:12.3f} | {len(onsets):8d} | {ev['precision']:9.3f} | {ev['recall']:.3f} | {ev['f1']:.3f}")

if __name__ == "__main__":
    main()
