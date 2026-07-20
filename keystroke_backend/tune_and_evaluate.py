"""
Step 4 (M4): Evaluate the rule-based pipeline against ground-truth keylogs
and auto-calibrate parameters for best accuracy.

USAGE:
    # Evaluate with default parameters
    python tune_and_evaluate.py --name session1

    # Evaluate with specific parameters
    python tune_and_evaluate.py --name session1 --threshold 0.5 --delta 0.3

    # Auto-find best parameters (recommended)
    python tune_and_evaluate.py --name session1 --auto

    # Auto-calibrate across multiple sessions simultaneously
    python tune_and_evaluate.py --names session1 session2 session3 --auto

    # Evaluate with YAMNet false-positive filter (requires trained classifier)
    python tune_and_evaluate.py --names session1 session2 session3 --yamnet-filter

    # Combine auto-calibration with YAMNet filter
    python tune_and_evaluate.py --names session1 session2 session3 --auto --yamnet-filter

Accuracy improvements over the baseline:
    - --auto flag sweeps a grid of (delta, threshold) combinations and picks
      the pair with the best F1 score against ground-truth keystroke times,
      then re-evaluates word-level accuracy with those parameters.
    - Precision / Recall / F1 on individual keystroke detection are now shown
      in addition to the word-level accuracy, giving a clearer picture.
    - Keystroke-level evaluation uses evaluate_against_ground_truth() with
      a 80 ms tolerance window.
    - --yamnet-filter flag runs raw onset candidates through a trained YAMNet
      embedding classifier to remove false positives before segmentation.
"""

import argparse
import csv
import os

import numpy as np

from onset_detector import detect_onsets, evaluate_against_ground_truth
from segmenter import segment_into_words, auto_threshold, format_output

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_YAMNET_MODEL = os.path.join(os.path.dirname(__file__), "models", "keystroke_classifier.joblib")


def load_ground_truth(log_path):
    """
    Read a keylog CSV and return:
        true_counts  : list[int]   — keystrokes per word
        gt_times     : list[float] — timestamps of every non-boundary keypress
    """
    rows = []
    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    true_counts: list[int] = []
    gt_times:    list[float] = []
    current_count = 0

    for row in rows:
        is_boundary = row["is_word_boundary"] == "True"
        if is_boundary:
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
    """Positional word-level accuracy: fraction of words whose count matches exactly."""
    n = min(len(true_counts), len(predicted_counts))
    correct = sum(1 for i in range(n) if true_counts[i] == predicted_counts[i])
    total = max(len(true_counts), len(predicted_counts))
    return (correct / total if total else 0.0), correct, total


def _f1_for_params(wav_path, gt_times, delta, threshold):
    """Return F1 score for a given (delta, threshold) pair."""
    try:
        onsets, _, _ = detect_onsets(wav_path, delta=delta)
        ev = evaluate_against_ground_truth(onsets, gt_times, tolerance=0.08)
        return ev["f1"], onsets
    except Exception:
        return 0.0, np.array([])


# ──────────────────────────────────────────────────────────────────────────────
# Auto-calibration
# ──────────────────────────────────────────────────────────────────────────────

DELTA_GRID     = [0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80]
THRESHOLD_GRID = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]


def auto_calibrate(wav_paths, gt_times_list):
    """
    Grid-search over (delta, threshold) to maximise the average F1 across
    all provided sessions.

    Returns
    -------
    best_delta, best_threshold, best_f1
    """
    print("\nAuto-calibrating parameters …")
    best_f1        = -1.0
    best_delta     = 0.07
    best_threshold = 0.50
    total          = len(DELTA_GRID) * len(THRESHOLD_GRID)
    done           = 0

    for delta in DELTA_GRID:
        for threshold in THRESHOLD_GRID:
            f1_scores = []
            for wav_path, gt_times in zip(wav_paths, gt_times_list):
                onsets, _, _ = detect_onsets(wav_path, delta=delta)
                ev = evaluate_against_ground_truth(onsets, gt_times, tolerance=0.08)
                f1_scores.append(ev["f1"])
            mean_f1 = float(np.mean(f1_scores))
            done += 1
            print(f"  [{done:3d}/{total}] delta={delta:.2f}  threshold={threshold:.2f}  "
                  f"mean_F1={mean_f1:.3f}", end="\r")
            if mean_f1 > best_f1:
                best_f1        = mean_f1
                best_delta     = delta
                best_threshold = threshold

    print(f"\n  Best -> delta={best_delta}  threshold={best_threshold}  F1={best_f1:.3f}")
    return best_delta, best_threshold, best_f1


# ──────────────────────────────────────────────────────────────────────────────
# Per-session reporting
# ──────────────────────────────────────────────────────────────────────────────

def report_session(name, delta, threshold, use_auto_seg=False,
                   yamnet_filter=False, yamnet_model_path=None,
                   yamnet_confidence=0.5):
    wav_path = os.path.join(DATA_DIR, f"{name}.wav")
    log_path = os.path.join(DATA_DIR, f"{name}_log.csv")

    true_counts, gt_times = load_ground_truth(log_path)
    onsets, y, sr = detect_onsets(wav_path, delta=delta)
    raw_count = len(onsets)

    # ── Optional YAMNet false-positive filter ─────────────────────────────
    if yamnet_filter:
        from yamnet_filter import filter_onsets_with_yamnet
        classifier_path = yamnet_model_path or DEFAULT_YAMNET_MODEL
        print(f"  Filtering {raw_count} raw onsets with YAMNet classifier "
              f"(threshold={yamnet_confidence:.2f}) ...")
        onsets = filter_onsets_with_yamnet(
            onsets, wav_path, classifier_path,
            confidence_threshold=yamnet_confidence,
        )
        print(f"  -> {len(onsets)} onsets kept ({raw_count - len(onsets)} removed)")

    # Choose segmentation threshold
    if use_auto_seg:
        seg_threshold = auto_threshold(onsets)
    else:
        seg_threshold = threshold

    pred_counts, _ = segment_into_words(onsets, gap_threshold=seg_threshold)
    acc, correct, total = word_accuracy(true_counts, pred_counts)

    # Keystroke-level precision / recall / F1
    ev = evaluate_against_ground_truth(onsets, gt_times, tolerance=0.08)

    sep = "=" * 56
    filter_tag = " + YAMNet" if yamnet_filter else ""
    print(f"\n{sep}")
    print(f"  Session : {name}{filter_tag}")
    print(f"  Params  : delta={delta}  threshold={seg_threshold:.3f}")
    if yamnet_filter:
        print(f"  YAMNet  : confidence≥{yamnet_confidence:.2f}  "
              f"kept {len(onsets)}/{raw_count} onsets")
    print(sep)
    print(f"  True counts      : {true_counts}  ->  {format_output(true_counts)}")
    print(f"  Predicted counts : {pred_counts}  ->  {format_output(pred_counts)}")
    print(f"  Word accuracy    : {correct}/{total} = {acc:.1%}")
    print(f"  Keystroke detection:")
    print(f"    Total detected : {len(onsets)}  (ground truth: {len(gt_times)})")
    print(f"    True positives : {ev['true_positives']}")
    print(f"    False positives: {ev['false_positives']}")
    print(f"    False negatives: {ev['false_negatives']}")
    print(f"    Precision      : {ev['precision']:.3f}")
    print(f"    Recall         : {ev['recall']:.3f}")
    print(f"    F1             : {ev['f1']:.3f}")
    print(sep)

    if acc < 0.85:
        print("  [!] Word accuracy below 85%.  Suggestions:")
        if len(pred_counts) > len(true_counts):
            print("    -> Too many words detected: increase --threshold")
        elif len(pred_counts) < len(true_counts):
            print("    -> Too few words detected: decrease --threshold")
        if ev["false_positives"] > ev["true_positives"]:
            print("    -> Many false keystroke detections: increase --delta")
            print("      or try --auto to let the script find better parameters")
            if not yamnet_filter:
                print("      or try --yamnet-filter to apply the YAMNet classifier")
        if ev["false_negatives"] > 2:
            print("    -> Missing real keystrokes: decrease --delta")
    else:
        print("  [OK] Word accuracy at or above 85% target!")

    return acc, ev["f1"]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and auto-tune the keystroke detection pipeline."
    )
    # Single-session mode
    parser.add_argument("--name", help="Single session name, e.g. session1")
    # Multi-session mode (used for auto-calibration across sessions)
    parser.add_argument("--names", nargs="+", help="Multiple session names for joint calibration")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Word-boundary gap threshold (default: 0.50 s)")
    parser.add_argument("--delta", type=float, default=0.07,
                        help="Onset sensitivity (default: 0.07)")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-calibrate delta and threshold for best F1 score")
    parser.add_argument("--auto-seg", action="store_true",
                        help="Use auto_threshold() for word segmentation (no manual threshold needed)")
    parser.add_argument("--yamnet-filter", action="store_true",
                        help="Filter onsets through a trained YAMNet classifier to remove false positives")
    parser.add_argument("--yamnet-model", type=str, default=None,
                        help=f"Path to the YAMNet classifier model (default: {DEFAULT_YAMNET_MODEL})")
    parser.add_argument("--yamnet-confidence", type=float, default=0.5,
                        help="Confidence threshold for the YAMNet filter (default: 0.5)")
    args = parser.parse_args()

    # Resolve session list
    if args.names:
        names = args.names
    elif args.name:
        names = [args.name]
    else:
        parser.error("Provide --name or --names")

    delta     = args.delta
    threshold = args.threshold

    # ── Auto-calibration ──────────────────────────────────────────────────────
    if args.auto:
        wav_paths    = [os.path.join(DATA_DIR, f"{n}.wav") for n in names]
        gt_times_all = []
        for n in names:
            log_path = os.path.join(DATA_DIR, f"{n}_log.csv")
            _, gt_times = load_ground_truth(log_path)
            gt_times_all.append(gt_times)

        delta, threshold, _ = auto_calibrate(wav_paths, gt_times_all)

    # ── Per-session report ────────────────────────────────────────────────────
    if args.yamnet_filter:
        print("\n[YAMNET] YAMNet filter ENABLED")

    accs, f1s = [], []
    for name in names:
        acc, f1 = report_session(
            name, delta, threshold, use_auto_seg=args.auto_seg,
            yamnet_filter=args.yamnet_filter,
            yamnet_model_path=args.yamnet_model,
            yamnet_confidence=args.yamnet_confidence,
        )
        accs.append(acc)
        f1s.append(f1)

    # Summary when multiple sessions
    if len(names) > 1:
        filter_label = " (with YAMNet filter)" if args.yamnet_filter else ""
        print(f"\n{'=' * 56}")
        print(f"  Overall  —  {len(names)} sessions{filter_label}")
        print(f"  Mean word accuracy    : {np.mean(accs):.1%}")
        print(f"  Mean keystroke F1     : {np.mean(f1s):.3f}")
        print(f"{'=' * 56}")


if __name__ == "__main__":
    main()
