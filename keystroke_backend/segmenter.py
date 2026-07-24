"""
Step 3 (M3): Segment a stream of detected onsets into words using pause gaps,
and output the keystroke count per word (e.g. "3|7" for "the project").

USAGE:
    python segmenter.py data/session1.wav --threshold 0.4
    python segmenter.py data/session1.wav --auto
"""

import argparse

import numpy as np

from onset_detector import detect_onsets


def auto_threshold(onset_times, min_threshold=0.15, max_threshold=2.0):
    """
    Automatically pick a word-boundary gap threshold from the onset timestamps.

    Finds the largest relative jump in the sorted gap distribution — that jump
    separates within-word gaps from between-word gaps.
    Falls back to 0.5 s if there are too few onsets to decide.
    """
    if len(onset_times) < 3:
        return 0.5

    gaps = np.diff(onset_times)
    if len(gaps) < 2:
        return 0.5

    sorted_gaps = np.sort(gaps)
    ratios = sorted_gaps[1:] / (sorted_gaps[:-1] + 1e-6)
    best_idx = int(np.argmax(ratios))
    threshold = float(np.sqrt(sorted_gaps[best_idx] * sorted_gaps[best_idx + 1]))
    return float(np.clip(threshold, min_threshold, max_threshold))


def segment_into_words(onset_times, gap_threshold=0.4):
    """
    Group onset timestamps into words by splitting on gaps > gap_threshold.

    gap_threshold : seconds between consecutive onsets that signals a word boundary.
                    Default 0.4 s — tune with tune_and_evaluate.py.

    Returns
    -------
    word_counts : list[int]         — keystroke count per word
    word_groups : list[list[float]] — onset timestamps grouped by word
    """
    if len(onset_times) == 0:
        return [], []

    word_counts = []
    word_groups = []
    current_group = [onset_times[0]]

    for i in range(1, len(onset_times)):
        gap = onset_times[i] - onset_times[i - 1]
        if gap > gap_threshold:
            word_counts.append(len(current_group))
            word_groups.append(current_group)
            current_group = [onset_times[i]]
        else:
            current_group.append(onset_times[i])

    word_counts.append(len(current_group))
    word_groups.append(current_group)

    return word_counts, word_groups


def format_output(word_counts):
    """Format counts as 'X|Y|Z' per the project spec."""
    return "|".join(str(c) for c in word_counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path", help="Path to the audio file")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Word-boundary gap in seconds. Omit to use --auto.")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-detect threshold from the gap distribution.")
    parser.add_argument("--delta", type=float, default=0.07,
                        help="Onset sensitivity (lower = more sensitive).")
    args = parser.parse_args()

    onsets, y, sr = detect_onsets(args.wav_path, delta=args.delta)

    if args.auto or args.threshold is None:
        threshold = auto_threshold(onsets)
        print(f"Auto-detected threshold: {threshold:.3f} s")
    else:
        threshold = args.threshold

    counts, groups = segment_into_words(onsets, gap_threshold=threshold)
    print(f"Detected {len(onsets)} total keystrokes across {len(counts)} word(s).")
    print("Per-word counts:", counts)
    print("Formatted output:", format_output(counts))
