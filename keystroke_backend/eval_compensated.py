"""
Evaluate detected onsets against ground truth after applying fixed hardware/driver latency compensation offsets.

USAGE:
    python eval_compensated.py --names session1 session3
"""

import argparse
import os
import numpy as np

from onset_detector import detect_onsets, load_ambient_rms, evaluate_against_ground_truth
from tune_and_evaluate import load_ground_truth

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

DEFAULT_OFFSETS = {
    "session1": 0.270,  # 270 ms hardware/driver latency
    "session3": 0.160,  # 160 ms hardware/driver latency
}


def evaluate_session(session_name, offset_sec=None, delta=0.07, noise_gate_factor=2.0, max_decay_ratio=0.90, tolerance=0.05):
    wav_path = os.path.join(DATA_DIR, f"{session_name}.wav")
    log_path = os.path.join(DATA_DIR, f"{session_name}_log.csv")
    meta_path = os.path.join(DATA_DIR, f"{session_name}_meta.json")

    if not os.path.exists(wav_path) or not os.path.exists(log_path):
        print(f"[Error] Audio file or ground truth log missing for session: {session_name}")
        return

    if offset_sec is None:
        offset_sec = DEFAULT_OFFSETS.get(session_name, 0.200)

    ambient = load_ambient_rms(meta_path)
    true_counts, gt_times = load_ground_truth(log_path)

    onsets, _, _ = detect_onsets(
        wav_path,
        delta=delta,
        ambient_rms=ambient,
        noise_gate_factor=noise_gate_factor,
        max_decay_ratio=max_decay_ratio
    )

    ev_uncomp = evaluate_against_ground_truth(onsets, gt_times, tolerance=tolerance)
    compensated_onsets = onsets - offset_sec
    ev_comp = evaluate_against_ground_truth(compensated_onsets, gt_times, tolerance=tolerance)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  LATENCY-COMPENSATED EVALUATION: {session_name}")
    print(f"  Applied Offset Shift : -{offset_sec*1000:.0f} ms")
    print(f"  Detected Onsets      : {len(onsets)}  (Ground Truth: {len(gt_times)})")
    if ambient is not None:
        print(f"  Ambient RMS          : {ambient:.2f}")
    print(f"------------------------------------------------------------")
    print(f"  BEFORE Compensation (0ms)  -> Precision={ev_uncomp['precision']:.3f}, Recall={ev_uncomp['recall']:.3f}, F1={ev_uncomp['f1']:.3f}")
    print(f"  AFTER Compensation (-{offset_sec*1000:.0f}ms) -> Precision={ev_comp['precision']:.3f}, Recall={ev_comp['recall']:.3f}, F1={ev_comp['f1']:.3f}")
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Evaluate onsets with latency compensation.")
    parser.add_argument("--name", help="Single session name")
    parser.add_argument("--names", nargs="+", help="Multiple session names")
    parser.add_argument("--offset", type=float, help="Explicit offset in seconds (e.g. 0.270)")
    parser.add_argument("--delta", type=float, default=0.07, help="Onset threshold delta")
    parser.add_argument("--noise-gate-factor", type=float, default=2.0, help="Noise gate factor")

    args = parser.parse_args()

    sessions = []
    if args.name:
        sessions.append(args.name)
    if args.names:
        sessions.extend(args.names)
    if not sessions:
        sessions = ["session1", "session3"]

    for session in sessions:
        evaluate_session(session, offset_sec=args.offset, delta=args.delta, noise_gate_factor=args.noise_gate_factor)


if __name__ == "__main__":
    main()
