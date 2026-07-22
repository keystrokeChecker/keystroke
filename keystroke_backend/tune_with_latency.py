"""
Grid search tuner for onset detector parameters with latency compensation enabled.

USAGE:
    python tune_with_latency.py --names session1 session3
"""

import argparse
import os
import numpy as np

from onset_detector import detect_onsets, load_ambient_rms, evaluate_against_ground_truth
from tune_and_evaluate import load_ground_truth

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

DEFAULT_OFFSETS = {
    "session1": 0.270,
    "session3": 0.160,
}


def tune_session(session_name, offset_sec=None):
    wav_path = os.path.join(DATA_DIR, f"{session_name}.wav")
    log_path = os.path.join(DATA_DIR, f"{session_name}_log.csv")
    meta_path = os.path.join(DATA_DIR, f"{session_name}_meta.json")

    if not os.path.exists(wav_path) or not os.path.exists(log_path):
        print(f"[Error] Missing files for session: {session_name}")
        return

    if offset_sec is None:
        offset_sec = DEFAULT_OFFSETS.get(session_name, 0.200)

    ambient = load_ambient_rms(meta_path)
    true_counts, gt_times = load_ground_truth(log_path)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  TUNING PARAMETERS WITH LATENCY COMPENSATION FOR: {session_name}")
    print(f"  Latency Shift: -{offset_sec*1000:.0f} ms | GT Keystrokes: {len(gt_times)}")
    print(sep)

    best_f1 = -1.0
    best_params = None
    best_ev = None

    for delta in [0.05, 0.07, 0.10, 0.15, 0.20]:
        for gate_factor in [1.5, 2.0, 3.0, 4.0]:
            for max_decay in [0.55, 0.70, 0.90]:
                onsets, _, _ = detect_onsets(
                    wav_path,
                    delta=delta,
                    ambient_rms=ambient,
                    noise_gate_factor=gate_factor,
                    max_decay_ratio=max_decay
                )
                comp_onsets = onsets - offset_sec
                ev = evaluate_against_ground_truth(comp_onsets, gt_times, tolerance=0.05)

                if ev['f1'] > best_f1:
                    best_f1 = ev['f1']
                    best_params = (delta, gate_factor, max_decay, len(onsets))
                    best_ev = ev

    d, g, dec, count = best_params
    print(f"  [BEST CONFIG] delta={d}, noise_gate_factor={g}, max_decay={dec} (Detections: {count})")
    print(f"  [RESULT] Precision={best_ev['precision']:.3f}, Recall={best_ev['recall']:.3f}, F1={best_ev['f1']:.3f}")
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Tune parameters with latency compensation.")
    parser.add_argument("--name", help="Single session name")
    parser.add_argument("--names", nargs="+", help="Multiple session names")
    args = parser.parse_args()

    sessions = []
    if args.name:
        sessions.append(args.name)
    if args.names:
        sessions.extend(args.names)
    if not sessions:
        sessions = ["session1", "session3"]

    for session in sessions:
        tune_session(session)


if __name__ == "__main__":
    main()
