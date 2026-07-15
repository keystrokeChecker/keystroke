"""
Step 4 (M4): Evaluate the rule-based pipeline against ground-truth keylogs
and help tune parameters.

USAGE:
    # Evaluate with defaults
    python tune_and_evaluate.py --name session1

    # Evaluate with specific parameters
    python tune_and_evaluate.py --name session1 --threshold 0.25 --delta 0.07

    # Auto-find the best parameters (recommended)
    python tune_and_evaluate.py --name session1 --auto

    # Auto-calibrate across multiple sessions
    python tune_and_evaluate.py --names session1 session2 session3 --auto
"""

import argparse
import csv
import os

import numpy as np

from onset_detector import detect_onsets, evaluate_against_ground_truth
from segmenter import segment_into_words, format_output

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Parameter search grid used by --auto
DELTA_GRID     = [0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
THRESHOLD_GRID = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]


# ─────────────────────────────────────────────────────────────────────────────
def load_ground_truth(log_path):
    """
    Read a keylog CSV and return:
        true_counts : list[int]   — keystrokes per word
        gt_times    : list[float] — timestamp of every non-boundary keypress
    """
    rows = list(csv.DictReader(open(log_path, "r")))

    true_counts: list[int]   = []
    gt_times:    list[float] = []
    current_count = 0

    for row in rows:
        if row["is_word_boundary"] == "True":
            if current_count > 0:
                true_counts.append(current_count)
            current_count = 0
        else:
            current_count += 1
            gt_times.append(float(row["timestamp_sec"]))

    if current_count > 0:
        true_counts.append(current_count)

    return true_counts, gt_times


def word_accuracy(true_counts, predicted_counts):
    n       = min(len(true_counts), len(predicted_counts))
    correct = sum(1 for i in range(n) if true_counts[i] == predicted_counts[i])
    total   = max(len(true_counts), len(predicted_counts))
    return (correct / total if total else 0.0), correct, total


# ─────────────────────────────────────────────────────────────────────────────
def auto_calibrate(wav_paths, gt_times_list, true_counts_list):
    """
    Grid-search (delta × threshold) to maximise mean word accuracy.
    Falls back to maximising F1 if word accuracy is 0 everywhere.
    """
    print("\nAuto-calibrating — sweeping parameters …")
    total = len(DELTA_GRID) * len(THRESHOLD_GRID)
    done  = 0

    best_acc   = -1.0
    best_f1    = -1.0
    best_delta, best_threshold = 0.07, 0.25

    for delta in DELTA_GRID:
        for threshold in THRESHOLD_GRID:
            accs, f1s = [], []
            for wav_path, gt_times, true_counts in zip(wav_paths, gt_times_list, true_counts_list):
                onsets, _, _ = detect_onsets(wav_path, delta=delta)
                pred_counts, _ = segment_into_words(onsets, gap_threshold=threshold)
                acc, _, _ = word_accuracy(true_counts, pred_counts)
                ev = evaluate_against_ground_truth(onsets, gt_times, tolerance=0.05)
                accs.append(acc)
                f1s.append(ev["f1"])

            mean_acc = float(np.mean(accs))
            mean_f1  = float(np.mean(f1s))
            done += 1
            print(f"  [{done:3d}/{total}]  delta={delta:.2f}  thresh={threshold:.2f}"
                  f"  word_acc={mean_acc:.1%}  F1={mean_f1:.3f}", end="\r")

            # Prefer better word accuracy; break ties with F1
            if mean_acc > best_acc or (mean_acc == best_acc and mean_f1 > best_f1):
                best_acc       = mean_acc
                best_f1        = mean_f1
                best_delta     = delta
                best_threshold = threshold

    print(f"\n  Best → delta={best_delta}  threshold={best_threshold}"
          f"  word_acc={best_acc:.1%}  F1={best_f1:.3f}")
    return best_delta, best_threshold


# ─────────────────────────────────────────────────────────────────────────────
def report_session(name, delta, threshold):
    wav_path = os.path.join(DATA_DIR, f"{name}.wav")
    log_path = os.path.join(DATA_DIR, f"{name}_log.csv")

    true_counts, gt_times = load_ground_truth(log_path)
    onsets, y, sr         = detect_onsets(wav_path, delta=delta)
    pred_counts, _        = segment_into_words(onsets, gap_threshold=threshold)
    acc, correct, total   = word_accuracy(true_counts, pred_counts)
    ev = evaluate_against_ground_truth(onsets, gt_times, tolerance=0.05)

    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  Session  : {name}")
    print(f"  delta={delta}   threshold={threshold}")
    print(sep)
    print(f"  True counts      : {true_counts}  →  {format_output(true_counts)}")
    print(f"  Predicted counts : {pred_counts}  →  {format_output(pred_counts)}")
    print(f"  Word accuracy    : {correct}/{total} = {acc:.1%}")
    print(f"  Onsets detected  : {len(onsets)}  (ground truth: {len(gt_times)})")
    print(f"  Precision        : {ev['precision']:.3f}")
    print(f"  Recall           : {ev['recall']:.3f}")
    print(f"  F1               : {ev['f1']:.3f}")
    print(sep)

    if acc < 0.85:
        print("  ⚠  Word accuracy below 85%. Suggestions:")
        if len(pred_counts) > len(true_counts):
            print(f"     → Too many words detected — try increasing --threshold (currently {threshold})")
        elif len(pred_counts) < len(true_counts):
            print(f"     → Too few words detected  — try decreasing --threshold (currently {threshold})")
        if ev["false_positives"] > ev["true_positives"]:
            print(f"     → Many false onsets        — try increasing --delta (currently {delta})")
        if ev["false_negatives"] > 2:
            print(f"     → Missed real keystrokes   — try decreasing --delta (currently {delta})")
        print("     → Run with --auto to let the script find the best parameters.")
    else:
        print("  ✅ Word accuracy at or above 85% target!")

    return acc, ev["f1"], true_counts


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and tune the keystroke detection pipeline."
    )
    parser.add_argument("--name",  help="Single session name, e.g. session1")
    parser.add_argument("--names", nargs="+", help="Multiple session names")
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="Word-boundary gap threshold in seconds (default 0.25)")
    parser.add_argument("--delta", type=float, default=0.07,
                        help="Onset sensitivity (default 0.07)")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-sweep parameters to maximise word accuracy")
    args = parser.parse_args()

    names = args.names if args.names else ([args.name] if args.name else None)
    if not names:
        parser.error("Provide --name or --names")

    delta, threshold = args.delta, args.threshold

    # ── Auto-calibration ──────────────────────────────────────────────────────
    if args.auto:
        wav_paths, gt_times_list, true_counts_list = [], [], []
        for n in names:
            wav_paths.append(os.path.join(DATA_DIR, f"{n}.wav"))
            tc, gt = load_ground_truth(os.path.join(DATA_DIR, f"{n}_log.csv"))
            true_counts_list.append(tc)
            gt_times_list.append(gt)
        delta, threshold = auto_calibrate(wav_paths, gt_times_list, true_counts_list)

    # ── Per-session report ────────────────────────────────────────────────────
    accs, f1s = [], []
    for name in names:
        acc, f1, _ = report_session(name, delta, threshold)
        accs.append(acc)
        f1s.append(f1)

    if len(names) > 1:
        print(f"\n{'=' * 52}")
        print(f"  Overall ({len(names)} sessions)")
        print(f"  Mean word accuracy : {np.mean(accs):.1%}")
        print(f"  Mean F1            : {np.mean(f1s):.3f}")
        print(f"{'=' * 52}")


if __name__ == "__main__":
    main()
