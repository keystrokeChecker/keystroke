"""
Step 4 (M4): Evaluate the rule-based pipeline (onset detection + segmentation)
against your real keylog ground truth, and help you tune parameters.

USAGE:
    python tune_and_evaluate.py --name session1

This reads:
    data/session1.wav
    data/session1_log.csv   (produced by record_dataset.py)

And reports:
    - true word counts (from keylog) vs predicted word counts (from audio)
    - per-word match accuracy
    - suggestions if accuracy is low
"""

import argparse
import csv
import os

from onset_detector import detect_onsets
from segmenter import segment_into_words, format_output

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_ground_truth(log_path):
    """
    Reads the keylog CSV and returns the TRUE per-word counts,
    using word-boundary keys (space/enter/tab) as separators.
    """
    rows = []
    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    true_counts = []
    current_count = 0
    for row in rows:
        is_boundary = row["is_word_boundary"] == "True"
        if is_boundary:
            if current_count > 0:
                true_counts.append(current_count)
            current_count = 0
        else:
            current_count += 1
    if current_count > 0:
        true_counts.append(current_count)

    return true_counts


def evaluate(true_counts, predicted_counts):
    """Simple positional comparison: how many words match exactly."""
    n = min(len(true_counts), len(predicted_counts))
    correct = sum(1 for i in range(n) if true_counts[i] == predicted_counts[i])
    total = max(len(true_counts), len(predicted_counts))
    accuracy = correct / total if total else 0
    return accuracy, correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Session name used in record_dataset.py")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--delta", type=float, default=0.07)
    args = parser.parse_args()

    wav_path = os.path.join(DATA_DIR, f"{args.name}.wav")
    log_path = os.path.join(DATA_DIR, f"{args.name}_log.csv")

    true_counts = load_ground_truth(log_path)
    onsets, y, sr = detect_onsets(wav_path, delta=args.delta)
    pred_counts, _ = segment_into_words(onsets, gap_threshold=args.threshold)

    accuracy, correct, total = evaluate(true_counts, pred_counts)

    print("=" * 50)
    print(f"Session: {args.name}")
    print(f"True counts:      {true_counts}  ->  {format_output(true_counts)}")
    print(f"Predicted counts: {pred_counts}  ->  {format_output(pred_counts)}")
    print(f"Word-level accuracy: {correct}/{total} = {accuracy:.1%}")
    print("=" * 50)

    if accuracy < 0.85:
        print("Accuracy below target (85%). Try adjusting:")
        print(f"  --delta      (currently {args.delta}) "
              "-> lower if missing keystrokes, raise if detecting extra noise")
        print(f"  --threshold  (currently {args.threshold}) "
              "-> lower if words are merging together, raise if words are splitting apart")
        if len(pred_counts) > len(true_counts):
            print("  -> More predicted words than real words: threshold may be too LOW "
                  "(splitting single words apart). Try increasing --threshold.")
        elif len(pred_counts) < len(true_counts):
            print("  -> Fewer predicted words than real words: threshold may be too HIGH "
                  "(merging words together). Try decreasing --threshold.")


if __name__ == "__main__":
    main()
