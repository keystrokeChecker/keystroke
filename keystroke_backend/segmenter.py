"""
Step 3 (M3): Segment a stream of detected onsets into words using pause gaps,
and output the keystroke count per word (e.g. "3|7" for "the project").

USAGE:
    python segmenter.py data/session1.wav --threshold 0.4
"""

import argparse

from onset_detector import detect_onsets


def segment_into_words(onset_times, gap_threshold=0.4):
    """
    Groups onset timestamps into words based on time gaps between consecutive onsets.

    gap_threshold: seconds. If the gap between two consecutive clicks exceeds this,
                   it's treated as a word boundary (e.g. space bar pause).

    Returns:
        word_counts: list of ints, keystroke count per word, in order
        word_groups: list of lists, the actual onset timestamps per word (for debugging)
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
    """Formats counts as 'X|Y|Z' per the project spec."""
    return "|".join(str(c) for c in word_counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path", help="Path to the audio file")
    parser.add_argument("--threshold", type=float, default=0.4,
                         help="Word-boundary gap threshold in seconds (tune this!)")
    parser.add_argument("--delta", type=float, default=0.07,
                         help="Onset sensitivity (lower = more sensitive)")
    args = parser.parse_args()

    onsets, y, sr = detect_onsets(args.wav_path, delta=args.delta)
    counts, groups = segment_into_words(onsets, gap_threshold=args.threshold)

    print(f"Detected {len(onsets)} total keystrokes across {len(counts)} words.")
    print("Per-word counts:", counts)
    print("Formatted output:", format_output(counts))
