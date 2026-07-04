"""
Step 2 (M2): Detect keystroke "onset" events (clicks) in an audio file.

This module is reusable: import detect_onsets() from other scripts,
or run it directly to inspect onsets on one file:

    python onset_detector.py data/session1.wav
"""

import sys

import librosa
import numpy as np


def detect_onsets(wav_path, hop_length=256, delta=0.07, pre_max=3, post_max=3, backtrack=False):
    """
    Detects onset (keystroke click) timestamps in an audio file.

    Parameters you will likely need to TUNE against your own ground-truth log:
        delta      - higher = fewer, stronger onsets detected (reduces false positives)
        hop_length - smaller = finer time resolution, more compute
        pre_max/post_max - local-max window sizes for picking peaks

    Returns:
        onset_times: numpy array of timestamps (seconds) where a click was detected
    """
    y, sr = librosa.load(wav_path, sr=None)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        delta=delta,
        pre_max=pre_max,
        post_max=post_max,
        backtrack=backtrack,
    )

    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    return onset_times, y, sr


def evaluate_against_ground_truth(onset_times, ground_truth_times, tolerance=0.05):
    """
    Compares detected onsets to ground-truth keypress timestamps (from the keylog).
    tolerance: how close (seconds) a detected onset must be to a real keypress to count as a match.

    Returns: dict with counts of matches, false positives, false negatives.
    """
    matched_gt = set()
    matched_det = set()

    for i, det_t in enumerate(onset_times):
        for j, gt_t in enumerate(ground_truth_times):
            if j in matched_gt:
                continue
            if abs(det_t - gt_t) <= tolerance:
                matched_gt.add(j)
                matched_det.add(i)
                break

    true_positives = len(matched_det)
    false_positives = len(onset_times) - true_positives
    false_negatives = len(ground_truth_times) - len(matched_gt)

    precision = true_positives / len(onset_times) if len(onset_times) else 0
    recall = true_positives / len(ground_truth_times) if len(ground_truth_times) else 0

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python onset_detector.py <path_to_wav>")
        sys.exit(1)

    wav_path = sys.argv[1]
    onsets, y, sr = detect_onsets(wav_path)
    print(f"Detected {len(onsets)} onsets (keystroke clicks):")
    print(np.round(onsets, 3))
