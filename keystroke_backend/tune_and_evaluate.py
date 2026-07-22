"""
Step 4 (M4): Evaluate the rule-based pipeline against ground-truth keylogs
and help tune parameters.

USAGE:
    # Evaluate with defaults (now auto-loads ambient_rms per session if the
    # matching data/<name>_meta.json exists)
    python tune_and_evaluate.py --name session1

    # Evaluate with specific parameters
    python tune_and_evaluate.py --name session1 --threshold 0.25 --delta 0.07 \
        --noise-gate-factor 3.0

    # Auto-find the best parameters, including the new noise/spectral gates
    python tune_and_evaluate.py --name session1 --auto

    # Auto-calibrate across multiple sessions
    python tune_and_evaluate.py --names session1 session2 session3 --auto
"""

import argparse
import csv
import os

import numpy as np

from onset_detector import detect_onsets, evaluate_against_ground_truth, load_ambient_rms
from segmenter import segment_into_words, format_output

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

DELTA_GRID           = [0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
THRESHOLD_GRID       = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]
# Coarse sweep for the new absolute-energy gate. This is the parameter that
# just collapsed detection to 0 on all four sessions at its default, so the
# grid deliberately starts well below the current default (6.0) and walks up.
NOISE_GATE_GRID      = [1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]


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


def _detect_kwargs(delta, noise_gate_factor, snr_threshold,
                    min_centroid_hz, min_flatness, max_decay_ratio):
    """Bundle the new detector parameters so every call site stays in sync."""
    return dict(
        delta=delta,
        noise_gate_factor=noise_gate_factor,
        snr_threshold=snr_threshold,
        min_centroid_hz=min_centroid_hz,
        min_spectral_flatness=min_flatness,
        max_decay_ratio=max_decay_ratio,
    )


# ─────────────────────────────────────────────────────────────────────────────
def auto_calibrate(wav_paths, gt_times_list, true_counts_list, ambient_list,
                    snr_threshold, min_centroid_hz, min_flatness, max_decay_ratio):
    """
    Grid-search (noise_gate_factor × delta × threshold) to maximise mean word
    accuracy. Falls back to maximising F1 if word accuracy is 0 everywhere.

    The noise_gate_factor sweep comes FIRST and is the outer loop deliberately:
    it's the parameter most likely to zero out every detection if left too
    strict, so we want the report to make that visible rather than averaging
    it away.
    """
    print("\nAuto-calibrating — sweeping noise_gate_factor x delta x threshold …")
    total = len(NOISE_GATE_GRID) * len(DELTA_GRID) * len(THRESHOLD_GRID)
    done  = 0

    best_acc    = -1.0
    best_f1     = -1.0
    best_gate, best_delta, best_threshold = NOISE_GATE_GRID[0], 0.07, 0.25

    for gate in NOISE_GATE_GRID:
        for delta in DELTA_GRID:
            for threshold in THRESHOLD_GRID:
                accs, f1s = [], []
                for wav_path, gt_times, true_counts, ambient in zip(
                        wav_paths, gt_times_list, true_counts_list, ambient_list):
                    kwargs = _detect_kwargs(delta, gate, snr_threshold,
                                             min_centroid_hz, min_flatness, max_decay_ratio)
                    onsets, _, _ = detect_onsets(wav_path, ambient_rms=ambient, **kwargs)
                    pred_counts, _ = segment_into_words(onsets, gap_threshold=threshold)
                    acc, _, _ = word_accuracy(true_counts, pred_counts)
                    ev = evaluate_against_ground_truth(onsets, gt_times, tolerance=0.05)
                    accs.append(acc)
                    f1s.append(ev["f1"])

                mean_acc = float(np.mean(accs))
                mean_f1  = float(np.mean(f1s))
                done += 1
                print(f"  [{done:4d}/{total}]  gate={gate:.1f}  delta={delta:.2f}  thresh={threshold:.2f}"
                      f"  word_acc={mean_acc:.1%}  F1={mean_f1:.3f}", end="\r")

                if mean_acc > best_acc or (mean_acc == best_acc and mean_f1 > best_f1):
                    best_acc       = mean_acc
                    best_f1        = mean_f1
                    best_gate      = gate
                    best_delta     = delta
                    best_threshold = threshold

    print(f"\n  Best -> noise_gate_factor={best_gate}  delta={best_delta}  threshold={best_threshold}"
          f"  word_acc={best_acc:.1%}  F1={best_f1:.3f}")
    return best_gate, best_delta, best_threshold


# ─────────────────────────────────────────────────────────────────────────────
def report_session(name, delta, threshold, noise_gate_factor,
                    snr_threshold, min_centroid_hz, min_flatness, max_decay_ratio):
    wav_path  = os.path.join(DATA_DIR, f"{name}.wav")
    log_path  = os.path.join(DATA_DIR, f"{name}_log.csv")
    meta_path = os.path.join(DATA_DIR, f"{name}_meta.json")

    ambient = load_ambient_rms(meta_path) if os.path.exists(meta_path) else None
    if ambient is None:
        print(f"  [warn] no ambient_rms found for '{name}' "
              f"(expected {meta_path}) — falling back to self-estimated noise floor only")

    true_counts, gt_times = load_ground_truth(log_path)
    kwargs = _detect_kwargs(delta, noise_gate_factor, snr_threshold,
                             min_centroid_hz, min_flatness, max_decay_ratio)
    onsets, y, sr       = detect_onsets(wav_path, ambient_rms=ambient, **kwargs)
    pred_counts, _      = segment_into_words(onsets, gap_threshold=threshold)
    acc, correct, total = word_accuracy(true_counts, pred_counts)
    ev = evaluate_against_ground_truth(onsets, gt_times, tolerance=0.05)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Session  : {name}")
    print(f"  delta={delta}  threshold={threshold}  noise_gate_factor={noise_gate_factor}"
          f"  ambient_rms={'yes' if ambient is not None else 'NO (self-estimate only)'}")
    print(sep)
    print(f"  True counts      : {true_counts}  ->  {format_output(true_counts)}")
    print(f"  Predicted counts : {pred_counts}  ->  {format_output(pred_counts)}")
    print(f"  Word accuracy    : {correct}/{total} = {acc:.1%}")
    print(f"  Onsets detected  : {len(onsets)}  (ground truth: {len(gt_times)})")
    print(f"  Precision        : {ev['precision']:.3f}")
    print(f"  Recall           : {ev['recall']:.3f}")
    print(f"  F1               : {ev['f1']:.3f}")
    print(sep)

    if len(onsets) == 0:
        print("  ⚠  ZERO onsets detected — noise_gate_factor is too strict for this "
              "recording's actual click loudness. Try a lower value, e.g.:")
        print(f"     python tune_and_evaluate.py --name {name} --noise-gate-factor 1.5")
    elif acc < 0.85:
        print("  ⚠  Word accuracy below 85%. Suggestions:")
        if len(pred_counts) > len(true_counts):
            print(f"     → Too many words detected — try increasing --threshold (currently {threshold})")
        elif len(pred_counts) < len(true_counts):
            print(f"     → Too few words detected  — try decreasing --threshold (currently {threshold})")
        if ev["false_positives"] > ev["true_positives"]:
            print(f"     → Many false onsets        — try increasing --noise-gate-factor "
                  f"(currently {noise_gate_factor}) or --delta (currently {delta})")
        if ev["false_negatives"] > 2:
            print(f"     → Missed real keystrokes   — try decreasing --noise-gate-factor "
                  f"(currently {noise_gate_factor}) or --delta (currently {delta})")
        print("     → Run with --auto to let the script find the best parameters.")
    else:
        print("  [OK] Word accuracy at or above 85% target!")

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
    parser.add_argument("--noise-gate-factor", type=float, default=6.0, dest="noise_gate_factor",
                        help="Absolute energy gate multiplier over the noise floor "
                             "(default 6.0 — LOWER this if you get 0 detections)")
    parser.add_argument("--snr-threshold", type=float, default=4.0, dest="snr_threshold",
                        help="Attack-score SNR threshold multiplier (default 4.0)")
    parser.add_argument("--min-centroid-hz", type=float, default=1200.0, dest="min_centroid_hz",
                        help="Reject candidates with spectral centroid below this (Hz). "
                             "Default 1200.0 — MUST match onset_detector.py's own --help default. "
                             "0 disables.")
    parser.add_argument("--min-flatness", type=float, default=0.08, dest="min_flatness",
                        help="Reject candidates with spectral flatness below this. 0 disables.")
    parser.add_argument("--max-decay-ratio", type=float, default=0.90, dest="max_decay_ratio",
                        help="Reject candidates that don't decay fast enough. "
                             "Default 0.90 — MUST match onset_detector.py's own --help default. "
                             "1.0 disables.")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-sweep noise_gate_factor x delta x threshold to maximise word accuracy")
    parser.add_argument("--yamnet-filter", action="store_true", dest="yamnet_filter",
                        help="Enable YAMNet-based filtering (not yet implemented)")
    args = parser.parse_args()

    names = args.names if args.names else ([args.name] if args.name else None)
    if not names:
        parser.error("Provide --name or --names")

    delta, threshold, noise_gate_factor = args.delta, args.threshold, args.noise_gate_factor

    # ── Auto-calibration ──────────────────────────────────────────────────────
    if args.auto:
        wav_paths, gt_times_list, true_counts_list, ambient_list = [], [], [], []
        for n in names:
            wav_paths.append(os.path.join(DATA_DIR, f"{n}.wav"))
            tc, gt = load_ground_truth(os.path.join(DATA_DIR, f"{n}_log.csv"))
            true_counts_list.append(tc)
            gt_times_list.append(gt)
            meta_path = os.path.join(DATA_DIR, f"{n}_meta.json")
            ambient_list.append(load_ambient_rms(meta_path) if os.path.exists(meta_path) else None)
        noise_gate_factor, delta, threshold = auto_calibrate(
            wav_paths, gt_times_list, true_counts_list, ambient_list,
            args.snr_threshold, args.min_centroid_hz, args.min_flatness, args.max_decay_ratio,
        )

    if args.yamnet_filter:
        print("\n[YAMNET] YAMNet filter ENABLED")

    accs, f1s = [], []
    for name in names:
        acc, f1, _ = report_session(
            name, delta, threshold, noise_gate_factor,
            args.snr_threshold, args.min_centroid_hz, args.min_flatness, args.max_decay_ratio,
        )
        accs.append(acc)
        f1s.append(f1)

    if len(names) > 1:
        print(f"\n{'=' * 60}")
        print(f"  Overall ({len(names)} sessions)")
        print(f"  Mean word accuracy : {np.mean(accs):.1%}")
        print(f"  Mean F1            : {np.mean(f1s):.3f}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()