"""
Step 2 (M2): Detect keystroke "onset" events (clicks) in an audio file.

This module is reusable: import detect_onsets() from other scripts,
or run it directly to inspect onsets on one file:

    python onset_detector.py data/session1.wav
"""

import sys

import librosa
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def detect_onsets(wav_path, hop_length=256, delta=0.07, pre_max=3, post_max=3, backtrack=False):
    """
    Detect keystroke click onsets from an audio file.

    The detector is tuned for percussive keyboard clicks rather than generic
    spectral changes. It first isolates the click band, builds a smoothed
    energy envelope, and then picks strong local peaks while suppressing
    duplicated detections from a single event.

    Parameters you may tune against your own ground-truth log:
        delta      - higher = fewer, stronger onsets detected (reduces false positives)
        hop_length - smaller = finer time resolution, more compute
        pre_max/post_max - local-neighborhood size used to avoid consecutive peaks

    Returns:
        onset_times: numpy array of timestamps (seconds) where a click was detected
    """
    y, sr = librosa.load(wav_path, sr=None)

    # Convert stereo audio to mono and normalize so the envelope is not biased by
    # recording amplitude differences across sessions.
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    y = y.astype(np.float32)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    # Apply a band-pass filter to emphasize the transient click energy and suppress
    # low-frequency rumble and high-frequency noise.
    low_cut = 800.0
    high_cut = 8000.0
    if sr < 2 * high_cut:
        high_cut = max(1000.0, sr / 2 - 100.0)

    b, a = butter(
        3,
        [low_cut / (sr / 2.0), high_cut / (sr / 2.0)],
        btype="bandpass",
    )
    filtered = filtfilt(b, a, y, method="pad")

    # Build two RMS envelopes: a short window for impulsive clicks and a longer
    # window for the local noise floor. The sharp attack of a keystroke produces a
    # rapid rise in the short-window envelope relative to the longer baseline.
    short_frame = max(64, hop_length // 2)
    long_frame = max(256, hop_length * 2)
    short_rms = librosa.feature.rms(y=filtered, frame_length=short_frame, hop_length=hop_length)[0]
    long_rms = librosa.feature.rms(y=filtered, frame_length=long_frame, hop_length=hop_length)[0]

    short_rms = np.maximum(short_rms, 1e-8)
    long_rms = np.maximum(long_rms, 1e-8)
    transient_score = short_rms / long_rms

    # Normalize the score so the thresholding behaves consistently across recordings.
    transient_score = transient_score / (np.percentile(transient_score, 95) + 1e-8)

    # Smooth the score slightly and then measure its first derivative. A real click
    # has a sudden rise; broad noise tends to rise and fall more gradually.
    smoothing_window = 3
    if len(transient_score) > 1:
        kernel = np.ones(smoothing_window, dtype=np.float32) / smoothing_window
        transient_score = np.convolve(transient_score, kernel, mode="same")
    attack_score = np.diff(transient_score)
    attack_score = np.clip(attack_score, 0.0, None)

    # Use an adaptive threshold so weak background noise is ignored while strong
    # clicks still stand out. Higher delta makes the detector more conservative.
    noise_floor = np.median(attack_score)
    dynamic_range = np.percentile(attack_score, 90) - noise_floor
    dynamic_range = max(dynamic_range, np.std(attack_score) + 1e-6)
    threshold = noise_floor + 0.7 * dynamic_range + 0.12 * max(0.0, delta)
    threshold = max(threshold, np.percentile(attack_score, 90) * 0.55)

    # Find peaks separated by a minimum spacing of ~70 ms to merge duplicates from
    # the same keypress while still allowing rapid successive presses.
    min_gap_seconds = 0.07
    min_distance = max(1, int(round(min_gap_seconds * sr / hop_length)))
    min_distance = max(min_distance, pre_max + post_max + 1)

    peak_indices, _ = find_peaks(
        attack_score,
        height=threshold,
        distance=min_distance,
        prominence=max(0.02, threshold * 0.25),
    )

    onset_frames = np.array(peak_indices + 1, dtype=int)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    # Merge any remaining peaks that are too close together, which can happen when
    # a single click produces a small cluster of local maxima.
    filtered_times = []
    minimum_gap = 0.07
    for t in onset_times:
        if not filtered_times or (t - filtered_times[-1]) > minimum_gap:
            filtered_times.append(float(t))

    onset_times = np.array(filtered_times, dtype=float)
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
