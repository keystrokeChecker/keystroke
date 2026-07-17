"""
Step 2 (M2): Detect keystroke "onset" events (clicks) in an audio file.

This module is reusable: import detect_onsets() from other scripts,
or run it directly to inspect onsets on one file:

    python onset_detector.py data/session1.wav

Accuracy improvements over the baseline:
    1. Re-normalise the bandpass-filtered signal before envelope extraction so
       that quiet / distant recordings are treated the same as loud ones.
    2. Use a longer baseline RMS window (hop*6) for a more stable noise floor.
    3. 99th-percentile score ceiling (was 95th) — less sensitive to noise bursts.
    4. Wider smoothing kernel (7 samples, was 3) — suppresses noise micro-spikes
       before computing the first derivative.
    5. 75th-percentile noise floor estimate (was median) — more conservative in
       the presence of sustained background noise.
    6. Stronger prominence guard (0.5x threshold, was 0.25x).
    7. Tunable minimum gap between detections (default 60 ms).
    8. min_gap_seconds and prominence_multiplier are now explicit parameters.
"""

import sys

import librosa
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def normalize_recording(y):
    """
    Peak-normalizes the recording and estimates a per-recording noise floor.

    Returns
    -------
    y_norm : np.ndarray — peak-normalized signal
    noise_floor : float — estimated noise floor (10th percentile RMS)
    """
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    y = y.astype(np.float32)
    
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
        
    # Compute 10th percentile of frame RMS values as noise floor
    frame_length = 1024
    hop_length = 256
    if len(y) > frame_length:
        rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        noise_floor = float(np.percentile(rms_frames, 10))
    else:
        noise_floor = float(np.mean(np.abs(y)))
        
    return y, noise_floor


def detect_onsets(wav_path, hop_length=256, delta=0.07, pre_max=3, post_max=3,
                  backtrack=False, min_gap_seconds=0.06, prominence_multiplier=0.5,
                  smoothing_window=7):
    """
    Detect keystroke click onsets from a WAV recording.

    Parameters
    ----------
    wav_path   : path to the WAV file
    hop_length : frame hop in samples (smaller = finer resolution)
    delta      : sensitivity knob — higher suppresses more false positives,
                 lower catches more (but noisier) events
    pre_max, post_max : neighbourhood size for local-max search (rarely need changing)
    min_gap_seconds : minimum time (seconds) between two detected onsets.
                      Lower values allow resolving fast-typed keystroke bursts;
                      higher values suppress double-detections of one click.
                      Default 0.06 s (60 ms).
    prominence_multiplier : find_peaks prominence = max(0.04, threshold * this).
                            Lower values accept weaker peaks; higher values
                            require sharper transients. Default 0.5.
    smoothing_window : width of the moving average smoothing kernel. Default 7.

    Returns
    -------
    onset_times : np.ndarray of float — click timestamps in seconds
    y           : np.ndarray — mono, peak-normalised waveform
    sr          : int — sample rate
    """
    y, sr = librosa.load(wav_path, sr=None)

    # ── Preprocessing: Peak normalisation & Noise-floor estimation ───────────
    y, rec_noise_floor = normalize_recording(y)

    # ── Band-pass filter: 800 Hz – 8 kHz (keyboard click band) ───────────────
    low_cut  = 800.0
    high_cut = 8000.0
    if sr < 2 * high_cut:
        high_cut = max(1000.0, sr / 2 - 100.0)

    b, a = butter(
        3,
        [low_cut / (sr / 2.0), high_cut / (sr / 2.0)],
        btype="bandpass",
    )
    filtered = filtfilt(b, a, y, method="pad")

    # ── FIX 1: re-normalise filtered signal ───────────────────────────────────
    # Quiet recordings (phone far from keyboard) have a very small filtered
    # amplitude. Without re-normalising, the transient ratio is dominated by
    # noise. Scaling to unit peak puts all recordings on the same footing.
    filtered_peak = np.max(np.abs(filtered))
    if filtered_peak > 1e-6:
        filtered = filtered / filtered_peak

    # ── FIX 2: dual RMS with longer baseline window ───────────────────────────
    short_frame = max(64,  hop_length // 2)
    long_frame  = max(512, hop_length * 6)      # was hop*2 — longer = more stable noise floor
    short_rms = librosa.feature.rms(y=filtered, frame_length=short_frame, hop_length=hop_length)[0]
    long_rms  = librosa.feature.rms(y=filtered, frame_length=long_frame,  hop_length=hop_length)[0]

    short_rms = np.maximum(short_rms, 1e-8)
    long_rms  = np.maximum(long_rms,  1e-8)
    transient_score = short_rms / long_rms

    # ── FIX 3: 99th-percentile ceiling ───────────────────────────────────────
    transient_score = transient_score / (np.percentile(transient_score, 99) + 1e-8)
    transient_score = np.clip(transient_score, 0.0, 2.0)   # hard cap at 2× to reduce outlier pull

    # ── FIX 4: wider smoothing kernel ────────────────────────────────────────
    if len(transient_score) > smoothing_window:
        kernel = np.ones(smoothing_window, dtype=np.float32) / smoothing_window
        transient_score = np.convolve(transient_score, kernel, mode="same")

    attack_score = np.diff(transient_score)
    attack_score = np.clip(attack_score, 0.0, None)

    # ── FIX 5: 75th-percentile noise floor ───────────────────────────────────
    noise_floor   = np.percentile(attack_score, 75)    # was: median (~50th)
    dynamic_range = np.percentile(attack_score, 97) - noise_floor
    dynamic_range = max(dynamic_range, np.std(attack_score) + 1e-6)

    # Scale factor 0.9 + delta-driven offset — higher delta = higher bar
    threshold = noise_floor + 0.9 * dynamic_range + 0.18 * max(0.0, delta)
    threshold = max(threshold, np.percentile(attack_score, 94) * 0.65)

    # ── FIX 7: tunable minimum gap between detections ─────────────────────────
    min_distance = max(1, int(round(min_gap_seconds * sr / hop_length)))
    min_distance = max(min_distance, pre_max + post_max + 1)

    # ── FIX 6: stronger prominence guard ─────────────────────────────────────
    peak_indices, _ = find_peaks(
        attack_score,
        height=threshold,
        distance=min_distance,
        prominence=max(0.04, threshold * prominence_multiplier),
    )

    onset_frames = np.array(peak_indices + 1, dtype=int)
    onset_times  = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    # ── Final merge pass (uses same min_gap_seconds) ──────────────────────────
    filtered_times: list[float] = []
    for t in onset_times:
        if not filtered_times or (t - filtered_times[-1]) > min_gap_seconds:
            filtered_times.append(float(t))

    return np.array(filtered_times, dtype=float), y, sr


def evaluate_against_ground_truth(onset_times, ground_truth_times, tolerance=0.08):
    """
    Compare detected onsets to ground-truth keypress timestamps.

    tolerance : maximum time difference (seconds) to count as a match.
                Default 0.08 s to account for audio-vs-keylog timing offset.

    Returns a dict with:
        true_positives, false_positives, false_negatives, precision, recall, f1
    """
    matched_gt  = set()
    matched_det = set()

    for i, det_t in enumerate(onset_times):
        for j, gt_t in enumerate(ground_truth_times):
            if j in matched_gt:
                continue
            if abs(det_t - gt_t) <= tolerance:
                matched_gt.add(j)
                matched_det.add(i)
                break

    tp = len(matched_det)
    fp = len(onset_times) - tp
    fn = len(ground_truth_times) - len(matched_gt)

    precision = tp / len(onset_times)        if len(onset_times)        else 0.0
    recall    = tp / len(ground_truth_times) if len(ground_truth_times) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "true_positives":  tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 3),
        "recall":    round(recall,    3),
        "f1":        round(f1,        3),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python onset_detector.py <path_to_wav>")
        sys.exit(1)

    wav_path = sys.argv[1]
    onsets, y, sr = detect_onsets(wav_path)
    print(f"Detected {len(onsets)} onsets (keystroke clicks):")
    print(np.round(onsets, 3))
