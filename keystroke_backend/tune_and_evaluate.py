"""
YAMNet-based evaluation helper for a recorded session.

USAGE:
    python tune_and_evaluate.py --name session1
"""

import argparse
import csv
import os

from src.predictor import predict_keystroke_counts
from src.yamnet_config import CLASSIFIER_THRESHOLD, SENSITIVITY_DELTA, GAP_THRESHOLD, MERGE_GAP_SECONDS, MODIFIER_KEYS

from segmenter import format_output

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_ground_truth(log_path):
    """Load word keystroke counts from CSV, excluding modifier keys and boundaries."""
    true_counts = []
    current_count = 0

    with open(log_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            is_boundary = row["is_word_boundary"] == "True"
            if is_boundary:
                if current_count > 0:
                    true_counts.append(current_count)
                current_count = 0
            else:
                # Exclude modifier keys — they produce no distinct audio event
                if row.get("key", "") not in MODIFIER_KEYS:
                    current_count += 1

    if current_count > 0:
        true_counts.append(current_count)

    return true_counts


def evaluate(true_counts, predicted_counts):
    aligned = min(len(true_counts), len(predicted_counts))
    exact_matches = sum(1 for i in range(aligned) if true_counts[i] == predicted_counts[i])
    total_words = max(len(true_counts), len(predicted_counts))
    positional_accuracy = exact_matches / total_words if total_words else 0

    abs_word_errors = [abs(true_counts[i] - predicted_counts[i]) for i in range(aligned)]
    mean_abs_word_error = sum(abs_word_errors) / aligned if aligned else 0.0

    true_total_keystrokes = sum(true_counts)
    predicted_total_keystrokes = sum(predicted_counts)
    keystroke_total_error = abs(true_total_keystrokes - predicted_total_keystrokes)

    word_count_error = abs(len(true_counts) - len(predicted_counts))

    return {
        "positional_accuracy": positional_accuracy,
        "exact_matches": exact_matches,
        "total_words": total_words,
        "mean_abs_word_error": mean_abs_word_error,
        "word_count_error": word_count_error,
        "true_total_keystrokes": true_total_keystrokes,
        "predicted_total_keystrokes": predicted_total_keystrokes,
        "keystroke_total_error": keystroke_total_error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Session name used in record_dataset.py")
    parser.add_argument("--threshold", type=float, default=CLASSIFIER_THRESHOLD)
    parser.add_argument("--delta", type=float, default=SENSITIVITY_DELTA)
    parser.add_argument("--gap-threshold", type=float, default=GAP_THRESHOLD)
    parser.add_argument("--merge-gap-seconds", type=float, default=MERGE_GAP_SECONDS)
    args = parser.parse_args()

    wav_path = os.path.join(DATA_DIR, f"{args.name}.wav")
    log_path = os.path.join(DATA_DIR, f"{args.name}_log.csv")

    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Missing audio file: {wav_path}")
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Missing log file: {log_path}")

    true_counts = load_ground_truth(log_path)
    pred_counts = predict_keystroke_counts(
        wav_path,
        threshold=args.threshold,
        delta=args.delta,
        gap_threshold=args.gap_threshold,
        merge_gap_seconds=args.merge_gap_seconds,
    )

    metrics = evaluate(true_counts, pred_counts)

    print("=" * 50)
    print(f"Session: {args.name}")
    print(f"True counts:      {true_counts}  ->  {format_output(true_counts)}")
    print(f"Predicted counts: {pred_counts}  ->  {format_output(pred_counts)}")
    print(
        f"Word-level accuracy: {metrics['exact_matches']}/{metrics['total_words']} = "
        f"{metrics['positional_accuracy']:.1%}"
    )
    print(f"Mean abs word error: {metrics['mean_abs_word_error']:.2f}")
    print(f"Word count error:    {metrics['word_count_error']}")
    print(
        f"Total keystrokes:    true={metrics['true_total_keystrokes']} "
        f"pred={metrics['predicted_total_keystrokes']} "
        f"abs_error={metrics['keystroke_total_error']}"
    )
    print("=" * 50)

    if metrics["positional_accuracy"] < 0.85:
        print("Accuracy below target (85%). Try adjusting:")
        print(
            f"  --delta      (currently {args.delta}) -> lower if missing keystrokes, "
            "raise if detecting extra noise"
        )
        print(
            f"  --threshold  (currently {args.threshold}) -> lower if words are merging together, "
            "raise if words are splitting apart"
        )
        if len(pred_counts) > len(true_counts):
            print(
                "  -> More predicted words than real words: threshold may be too LOW "
                "(splitting single words apart). Try increasing --threshold."
            )
        elif len(pred_counts) < len(true_counts):
            print(
                "  -> Fewer predicted words than real words: threshold may be too HIGH "
                "(merging words together). Try decreasing --threshold."
            )


if __name__ == "__main__":
    main()

