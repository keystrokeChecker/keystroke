"""
Script to sweep timing offsets between audio-detected onsets and ground-truth keylogger timestamps.
Finds the constant time shift (in seconds) that maximizes precision, recall, and F1 score.

USAGE:
    python offset_sweep.py --names session1 session3
    python offset_sweep.py --name session1
"""

import argparse
import os
import sys
import numpy as np

from onset_detector import detect_onsets, load_ambient_rms, evaluate_against_ground_truth
from tune_and_evaluate import load_ground_truth

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def sweep_session(session_name, min_offset=-0.500, max_offset=0.500, step=0.010, tolerance=0.05):
    wav_path = os.path.join(DATA_DIR, f"{session_name}.wav")
    log_path = os.path.join(DATA_DIR, f"{session_name}_log.csv")
    meta_path = os.path.join(DATA_DIR, f"{session_name}_meta.json")

    if not os.path.exists(wav_path) or not os.path.exists(log_path):
        print(f"[Error] Audio file or ground truth log missing for session: {session_name}")
        return

    ambient = load_ambient_rms(meta_path)
    true_counts, gt_times = load_ground_truth(log_path)
    onsets, _, _ = detect_onsets(wav_path, ambient_rms=ambient)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  SWEEPING TIMING OFFSET FOR SESSION: {session_name}")
    print(f"  Detected Onsets: {len(onsets)} | Ground Truth Keystrokes: {len(gt_times)}")
    if ambient is not None:
        print(f"  Ambient RMS: {ambient:.2f}")
    print(sep)

    # Baseline evaluation at 0ms offset
    base_ev = evaluate_against_ground_truth(onsets, gt_times, tolerance=tolerance)
    print(f"  Baseline (0ms shift) -> P={base_ev['precision']:.3f}, R={base_ev['recall']:.3f}, F1={base_ev['f1']:.3f}")

    offsets = np.arange(min_offset, max_offset + step / 2.0, step)
    results = []

    best_f1 = -1.0
    best_offset = 0.0
    best_ev = base_ev

    for offset in offsets:
        shifted_onsets = onsets + offset
        ev = evaluate_against_ground_truth(shifted_onsets, gt_times, tolerance=tolerance)
        results.append((offset, ev['precision'], ev['recall'], ev['f1']))

        if ev['f1'] > best_f1:
            best_f1 = ev['f1']
            best_offset = offset
            best_ev = ev

    print(f"\n  [BEST OFFSET] {best_offset*1000:+.0f} ms -> Precision={best_ev['precision']:.3f}, Recall={best_ev['recall']:.3f}, F1={best_ev['f1']:.3f}")
    
    print("\n  Top 5 Offsets by F1 Score:")
    results.sort(key=lambda x: (x[3], x[1], x[2]), reverse=True)
    for off, p, r, f1 in results[:5]:
        print(f"    Offset: {off*1000:+.0f} ms  |  Precision: {p:.3f}  |  Recall: {r:.3f}  |  F1: {f1:.3f}")
    print(sep)

    return best_offset, base_ev, best_ev


def main():
    parser = argparse.ArgumentParser(description="Sweep timing offset between detected onsets and ground truth.")
    parser.add_argument("--name", help="Single session name (e.g. session1)")
    parser.add_argument("--names", nargs="+", help="Multiple session names (e.g. session1 session3)")
    parser.add_argument("--min-offset", type=float, default=-0.500, help="Minimum offset in seconds (default -0.500)")
    parser.add_argument("--max-offset", type=float, default=0.500, help="Maximum offset in seconds (default 0.500)")
    parser.add_argument("--step", type=float, default=0.010, help="Sweep step in seconds (default 0.010)")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Matching tolerance in seconds (default 0.05)")

    args = parser.parse_args()

    sessions = []
    if args.name:
        sessions.append(args.name)
    if args.names:
        sessions.extend(args.names)

    if not sessions:
        sessions = ["session1", "session3"]

    for session in sessions:
        sweep_session(session, min_offset=args.min_offset, max_offset=args.max_offset, step=args.step, tolerance=args.tolerance)


if __name__ == "__main__":
    main()
