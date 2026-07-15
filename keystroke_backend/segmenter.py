"""
Step 3 (M3): Segment a stream of detected onsets into words using pause gaps,
and output the keystroke count per word (e.g. "3|7" for "the project").

USAGE:
    python segmenter.py data/session1.wav --threshold 0.4
    python segmenter.py data/session1.wav --auto   # auto-detect threshold

Accuracy improvement:
    Added auto_threshold() which analyses the distribution of inter-onset gaps
    and picks the split point between the "within-word" gap cluster and the
    "between-word" gap cluster using the valley between those two groups.
    This removes the need to manually tune --threshold for each session.
"""

import argparse

import numpy as np

from onset_detector import detect_onsets


# ──────────────────────────────────────────────────────────────────────────────
# Core segmentation
# ──────────────────────────────────────────────────────────────────────────────

def auto_threshold(onset_times, min_threshold=0.15, max_threshold=2.0):
    """
    Automatically estimate the word-boundary gap threshold from the onset
    timestamps themselves, without needing any ground-truth labels.

    Strategy
    --------
    Inter-onset gaps fall into two natural clusters:
      • Short gaps  (~0.1–0.5 s) — consecutive keystrokes within a word
      • Long gaps   (~0.4–2.0 s) — pauses between words (space-bar pause)

    We sort the gaps, look for the largest relative jump between consecutive
    sorted values, and use the midpoint of that jump as the threshold.
    If there is no clear jump (e.g. only one word, or all gaps similar),
    we fall back to 0.5 s.

    Parameters
    ----------
    onset_times   : array of onset timestamps in seconds
    min_threshold : never go below this (avoids splitting within a word)
    max_threshold : never go above this

    Returns
    -------
    threshold : float — recommended gap_threshold in seconds
    """
    if len(onset_times) < 3:
        return 0.5

    gaps = np.diff(onset_times)
    if len(gaps) < 2:
        return 0.5

    sorted_gaps = np.sort(gaps)

    # Find the largest relative jump in sorted gaps — that is the
    # boundary between within-word and between-word gap clusters.
    ratios = sorted_gaps[1:] / (sorted_gaps[:-1] + 1e-6)
    best_idx = int(np.argmax(ratios))

    # Threshold = geometric mean of the two gap values on either side of the jump
    threshold = float(np.sqrt(sorted_gaps[best_idx] * sorted_gaps[best_idx + 1]))
    threshold = float(np.clip(threshold, min_threshold, max_threshold))
    return threshold


def segment_into_words(onset_times, gap_threshold=0.5):
    """
    Group onset timestamps into words by splitting on inter-onset gaps that
    exceed gap_threshold.

    Parameters
    ----------
    onset_times   : array of onset timestamps in seconds
    gap_threshold : seconds — gaps larger than this are treated as word boundaries.
                    Default raised to 0.5 s (was 0.4 s) to better match the
                    typical space-bar pause measured from the session keylogs
                    (mean inter-word gap ≈ 0.65 s).

    Returns
    -------
    word_counts : list[int] — keystroke count per word
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


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path", help="Path to the audio file")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Word-boundary gap threshold in seconds. Omit to use --auto.",
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Auto-detect threshold from the gap distribution (recommended).",
    )
    parser.add_argument(
        "--delta", type=float, default=0.07,
        help="Onset sensitivity (lower = more sensitive).",
    )
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
