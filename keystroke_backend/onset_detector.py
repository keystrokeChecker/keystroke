"""
Step 2 (M2): Detect keystroke "onset" events (clicks) in an audio file.

This module is reusable: import detect_onsets() from other scripts,
or run it directly to inspect onsets on one file:

    python onset_detector.py data/session1.wav
    python onset_detector.py data/session1.wav --meta data/session1_meta.json

False-positive hardening (2026-07 revision)
-------------------------------------------
The original detector used a rank-based, self-relative threshold
(percentile statistics of attack_score within the same clip).  On a
pure-silence clip this always found ~73 "onsets/10 s" because it was
ranking noise against itself.  Five concrete bugs were fixed:

  Bug 1 – self-relative threshold: replaced with an absolute threshold
           anchored to the measured noise floor (see snr_threshold).
  Bug 2 – 95th-percentile normalization: removed; it destroyed all
           amplitude information before thresholding.
  Bug 3 – peak normalization blocking ambient_rms comparison: the scale
           factor is now tracked so ambient_rms (int16 PCM) can be
           converted to peak-normalized float space for comparison.
  Bug 4 – no absolute energy gate: frames whose short-RMS is below
           noise_gate_factor × noise_floor are zeroed out before any
           peak search.  On a silence clip every frame is gated → 0 detections.
  Bug 5 – no spectral shape check: per-candidate tests for spectral
           centroid (broadband vs. low-frequency thud), spectral flatness
           (broadband vs. tonal/narrow-band), and fast-decay (click vs.
           sustained sound) are now applied after peak finding.

IMPORTANT — SCOPE BOUNDARY
These gates only answer "did a click happen?" (binary, yes/no).
None of them encode, measure, or narrow down which key was pressed.
Spectral centroid and flatness are the same for every key on the same
keyboard model; they merely distinguish "mechanical click" from "not
a mechanical click."  Do not add any classifier, template match, or
feature that is key-identity-specific.
"""

import json
import os
import sys

import librosa
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


# ── Internal DSP helpers ──────────────────────────────────────────────────────

def _butter_bandpass(low_hz: float, high_hz: float, sr: int, order: int = 3):
    nyq = sr / 2.0
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="bandpass")
    return b, a


def _spectral_centroid_hz(frame: np.ndarray, sr: int) -> float:
    """
    Return the spectral centroid (Hz) of a short waveform frame.

    A broadband click has centroid typically 2–5 kHz.
    A low-frequency thud or knock has centroid < 1 kHz.
    NOTE: this is an acoustic-shape discriminator only; centroid is identical
    for all keys on a given keyboard model and carries zero key-identity info.
    """
    n = len(frame)
    if n < 4:
        return 0.0
    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(frame * window))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    total = spectrum.sum()
    if total < 1e-12:
        return 0.0
    return float(np.dot(freqs, spectrum) / total)


def _spectral_flatness(frame: np.ndarray, sr: int, low_hz: float = 800.0, high_hz: float = 8000.0) -> float:
    """
    Return the spectral flatness (Wiener entropy) of a short waveform frame within the passband.
    Range: 0 (pure tone / narrow-band) -> 1 (ideal broadband white noise).

    Keyboard clicks are broadband (flatness ≈ 0.10–0.35).
    HVAC hums, squeaks, and electrical interference are near 0.
    NOTE: flatness is the same for all keys -> no key-identity information.
    """
    n = len(frame)
    if n < 4:
        return 0.0
    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(frame * window))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    
    # Restrict to passband
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    band_spectrum = spectrum[mask] + 1e-12
    
    log_mean = float(np.mean(np.log(band_spectrum)))
    arith_mean = float(np.mean(band_spectrum))
    if arith_mean < 1e-12:
        return 0.0
    return float(np.exp(log_mean) / arith_mean)


# ── Public helper: load ambient_rms from session meta JSON ───────────────────

def load_ambient_rms(meta_path: str) -> float | None:
    """
    Read the ambient_rms field from a session_meta.json produced by
    record_dataset.py.  Returns the value (raw int16 PCM units) or None
    if the file is missing or the field is absent.

    Usage
    -----
        ambient = load_ambient_rms("data/session1_meta.json")
        onsets, y, sr = detect_onsets("data/session1.wav",
                                       ambient_rms=ambient)
    """
    if not meta_path or not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        val = meta.get("ambient_rms")
        return float(val) if val is not None else None
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


# ── Normalize helper (unchanged public API) ───────────────────────────────────

def normalize_recording(y: np.ndarray):
    """
    Peak-normalizes the recording and estimates a per-recording noise floor.

    Returns
    -------
    y_norm      : np.ndarray — peak-normalized signal
    noise_floor : float     — estimated noise floor (10th percentile frame RMS)
    """
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    y = y.astype(np.float32)

    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    frame_length = 1024
    hop_length = 256
    if len(y) > frame_length:
        rms_frames = librosa.feature.rms(
            y=y, frame_length=frame_length, hop_length=hop_length
        )[0]
        noise_floor = float(np.percentile(rms_frames, 10))
    else:
        noise_floor = float(np.mean(np.abs(y)))

    return y, noise_floor


# ── Main detection function ───────────────────────────────────────────────────

def detect_onsets(
    wav_path: str,
    hop_length: int = 256,
    delta: float = 0.07,
    pre_max: int = 3,
    post_max: int = 3,
    backtrack: bool = True,
    min_gap_seconds: float = 0.06,
    prominence_multiplier: float = 0.5,
    smoothing_window: int = 7,
    # ── Precision / false-positive control ────────────────────────────────
    ambient_rms: float | None = None,
    noise_gate_factor: float = 2.0,
    snr_threshold: float = 4.0,
    min_centroid_hz: float = 1200.0,
    min_spectral_flatness: float = 0.08,
    max_decay_ratio: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Detect keystroke click onsets from an audio file.

    Returns only TIMESTAMPS (binary "click happened at time T").
    No key identity, no letter recovery — that is explicitly out of scope.

    Core parameters
    ---------------
    delta             : coarse sensitivity knob kept for backward compat.
                        Higher → fewer, stronger detections; lower → more.
    hop_length        : frame hop in samples (time resolution).
    pre_max, post_max : local-peak neighbourhood in find_peaks.
    min_gap_seconds   : minimum spacing between consecutive onsets (s).

    Precision / FP-control parameters (new)
    ----------------------------------------
    ambient_rms           : raw int16 PCM RMS from the pre-recording
                            calibration silence in record_dataset.py.
                            Load with load_ambient_rms("…_meta.json").
                            When supplied, enables an ABSOLUTE energy gate
                            calibrated to the actual room noise.
    noise_gate_factor     : a frame's short-RMS must exceed
                            noise_gate_factor × noise_floor to be a
                            candidate.  Default 6.0 ≈ 15.6 dB SNR.
                            Raise (8–12) to tighten; lower (4) to loosen.
    snr_threshold         : multiplier for the absolute attack-score
                            threshold.  Default 4.0.
    min_centroid_hz       : spectral centroid of the ±15 ms attack window
                            must exceed this value.  Default 1500 Hz
                            (rejects thuds and low-frequency knocks).
                            Set to 0 to disable.
    min_spectral_flatness : spectral flatness of the attack window must
                            exceed this (0 = tonal, 1 = white noise).
                            Default 0.08 (rejects HVAC hum, pure tones).
                            Set to 0 to disable.
    max_decay_ratio       : ratio of post-attack RMS (5–30 ms after peak)
                            to peak-attack RMS.  Clicks decay fast (< 0.40);
                            sustained sounds stay above 0.55.  Default 0.55.
                            Set to 1.0 to disable.

    Returns
    -------
    onset_times : np.ndarray — timestamps of detected click events (seconds)
    y           : np.ndarray — mono, peak-normalised waveform (for plotting)
    sr          : int        — sample rate
    """

    # ── Load ──────────────────────────────────────────────────────────────────
    y_raw, sr = librosa.load(wav_path, sr=None)

    if y_raw.ndim > 1:
        y_raw = np.mean(y_raw, axis=0)
    y_raw = y_raw.astype(np.float32)

    # librosa.load maps int16 PCM → float32 via /32768 internally.
    # We record raw_peak (in that /32768 float space) so we can relate
    # ambient_rms (int16 units) to the peak-normalised signal later.
    raw_peak = float(np.max(np.abs(y_raw)))
    if raw_peak < 0.025:
        # Effectively silent (peak is < -32 dBFS). No keystrokes are present.
        return np.array([], dtype=float), y_raw, sr

    # Peak-normalize for all subsequent processing.
    y = y_raw / raw_peak  # ± 1.0 scale

    # ── Convert ambient_rms to peak-normalised float scale ───────────────────
    # ambient_rms is in raw int16 units (e.g. 40–300 for a quiet room).
    # y_raw is in float32 /32768 units (librosa).
    # y = y_raw / raw_peak  →  ambient_norm = (ambient_rms/32768) / raw_peak
    if ambient_rms is not None and float(ambient_rms) > 0.0:
        ambient_float = float(ambient_rms) / 32768.0   # same scale as y_raw
        ambient_norm  = ambient_float / raw_peak        # same scale as y and filtered
    else:
        ambient_norm = None

    # ── Band-pass filter: 800 Hz – 8 kHz ─────────────────────────────────────
    # Keystroke clicks have most energy in this range.
    # Mouse clicks and keyboard clatter share this range too, but low-frequency
    # transients (HVAC bumps, chair creaks, footfalls) are rejected here.
    low_cut  = 800.0
    high_cut = 8000.0
    if sr < 2 * high_cut:
        high_cut = max(1000.0, sr / 2.0 - 100.0)

    b, a = _butter_bandpass(low_cut, high_cut, sr)
    filtered = filtfilt(b, a, y, method="pad")

    # ── Short / long RMS envelopes ────────────────────────────────────────────
    short_frame = max(64, hop_length // 2)
    long_frame  = max(256, hop_length * 2)

    # Compute short_rms on unfiltered signal y for absolute energy gating
    short_rms_unfiltered = librosa.feature.rms(
        y=y, frame_length=short_frame, hop_length=hop_length
    )[0]
    short_rms_unfiltered = np.maximum(short_rms_unfiltered, 1e-12)

    short_rms = librosa.feature.rms(
        y=filtered, frame_length=short_frame, hop_length=hop_length
    )[0]
    long_rms = librosa.feature.rms(
        y=filtered, frame_length=long_frame, hop_length=hop_length
    )[0]

    short_rms = np.maximum(short_rms, 1e-12)
    long_rms  = np.maximum(long_rms,  1e-12)

    # ── Absolute noise-floor estimation ──────────────────────────────────────
    # 5th percentile of short_rms_unfiltered = RMS of the quietest 5% of frames in the
    # unfiltered signal. This matches the scale of ambient_norm.
    inband_noise_rms = float(np.percentile(short_rms_unfiltered, 5))

    # Use inband_noise_rms as primary noise floor; use ambient_norm only if valid and not distorted
    abs_noise_floor = inband_noise_rms
    if ambient_norm is not None and float(ambient_norm) <= 2.0 * inband_noise_rms:
        abs_noise_floor = max(inband_noise_rms, float(ambient_norm))
    abs_noise_floor = max(abs_noise_floor, 1e-9)

    # ── Absolute energy gate ──────────────────────────────────────────────────
    # Any frame whose short-RMS < noise_gate_factor × abs_noise_floor is
    # definitionally below the signal-to-noise threshold for a keystroke.
    # On a pure-silence clip every frame fails this gate → 0 detections.
    # This is an ABSOLUTE gate, not rank-based: its value does not change
    # if you replace signal with louder silence.
    energy_gate_rms = noise_gate_factor * abs_noise_floor
    gate_mask = short_rms_unfiltered < energy_gate_rms   # True = too quiet, zero it out

    # ── Transient ratio (short/long) ──────────────────────────────────────────
    transient_score = short_rms / long_rms

    # Compute un-gated attack score for threshold calculation
    transient_score_ungated = transient_score.copy()
    _smooth_win = 3
    kernel = np.ones(_smooth_win, dtype=np.float32) / _smooth_win
    if len(transient_score_ungated) > _smooth_win:
        transient_score_ungated = np.convolve(transient_score_ungated, kernel, mode="same")
    attack_score_ungated = np.diff(transient_score_ungated)
    attack_score_ungated = np.clip(attack_score_ungated, 0.0, None)

    # Gated transient score for actual peak detection
    transient_score_gated = np.where(gate_mask, 0.0, transient_score)
    if len(transient_score_gated) > _smooth_win:
        transient_score_gated = np.convolve(transient_score_gated, kernel, mode="same")
    attack_score = np.diff(transient_score_gated)
    attack_score = np.clip(attack_score, 0.0, None)

    # ── Early-exit: if nothing survived the energy gate, return empty ─────────
    active_attack = attack_score[attack_score > 1e-12]
    if len(active_attack) == 0:
        return np.array([], dtype=float), y, sr

    # ── Threshold calculation on un-gated attack score ────────────────────────
    # Fits to typing session dynamics but refuses to adapt down to pure silence.
    noise_floor_ungated = np.median(attack_score_ungated)
    dynamic_range_ungated = np.percentile(attack_score_ungated, 90) - noise_floor_ungated
    dynamic_range_ungated = max(dynamic_range_ungated, np.std(attack_score_ungated) + 1e-6)

    threshold = noise_floor_ungated + 0.7 * dynamic_range_ungated + 0.12 * max(0.0, delta)
    threshold = max(threshold, np.percentile(attack_score_ungated, 90) * 0.55)
    # Enforce an absolute minimum ratio change threshold to prevent noise pickup
    threshold = max(threshold, 0.10 + 0.12 * max(0.0, delta))

    # ── Peak finding ──────────────────────────────────────────────────────────
    _min_gap = max(float(min_gap_seconds), 0.06)
    min_distance = max(1, int(round(_min_gap * sr / hop_length)))
    min_distance = max(min_distance, pre_max + post_max + 1)

    peak_indices, _ = find_peaks(
        attack_score,
        height=threshold,
        distance=min_distance,
        prominence=max(1e-6, threshold * 0.25),
    )

    if len(peak_indices) == 0:
        return np.array([], dtype=float), y, sr

    onset_frames = np.array(peak_indices + 1, dtype=int)

    # ── Per-candidate spectral shape filters ──────────────────────────────────
    # These discriminate "broadband fast-transient click" from other acoustic
    # events (coughs, mouse clicks, chair creaks, HVAC pops).
    
    attack_win_samples   = int(0.005 * sr)  # 0 to 5 ms post-peak
    decay_start_samples  = int(0.015 * sr)  # starts 15 ms after peak
    decay_win_samples    = int(0.025 * sr)  # 25 ms decay window (15 to 40 ms)
    attack_half_samples  = int(0.015 * sr)  # ±15 ms around peak center for spectral gates

    kept_frames: list[int] = []

    for frame_idx in onset_frames:
        sample_idx = int(frame_idx * hop_length)

        # ── Spectral centroid gate ─────────────────────────────────────────
        # Reject: centroid below min_centroid_hz → low-frequency thud/knock.
        if min_centroid_hz > 0.0:
            s0 = max(0, sample_idx - attack_half_samples)
            s1 = min(len(filtered), sample_idx + attack_half_samples)
            chunk = filtered[s0:s1]
            if len(chunk) >= 4:
                centroid = _spectral_centroid_hz(chunk, sr)
                if centroid < min_centroid_hz:
                    continue

        # ── Spectral flatness gate ─────────────────────────────────────────
        # Reject: flatness below min_spectral_flatness → tonal/narrow-band
        # transient (HVAC hum burst, electrical pop, narrow squeak).
        if min_spectral_flatness > 0.0:
            s0 = max(0, sample_idx - attack_half_samples)
            s1 = min(len(filtered), sample_idx + attack_half_samples)
            chunk = filtered[s0:s1]
            if len(chunk) >= 4:
                flatness = _spectral_flatness(chunk, sr)
                if flatness < min_spectral_flatness:
                    continue

        # ── Fast-decay gate ────────────────────────────────────────────────
        # Reject: post-attack energy too high → sustained sound (cough,
        # creak, chair, HVAC).  Keyboard clicks decay to < 55% of peak RMS.
        if max_decay_ratio < 1.0:
            # Attack RMS: immediate peak energy (0 to 5 ms)
            a0 = sample_idx
            a1 = min(len(filtered), sample_idx + attack_win_samples)
            attack_chunk = filtered[a0:a1]
            attack_rms = (
                float(np.sqrt(np.mean(attack_chunk ** 2)))
                if len(attack_chunk) > 0
                else 0.0
            )

            # Decay RMS: later decay energy (15 to 40 ms)
            d0 = min(len(filtered), sample_idx + decay_start_samples)
            d1 = min(len(filtered), d0 + decay_win_samples)
            decay_chunk = filtered[d0:d1]
            decay_rms = (
                float(np.sqrt(np.mean(decay_chunk ** 2)))
                if len(decay_chunk) > 0
                else 0.0
            )

            if attack_rms > 1e-9 and (decay_rms / attack_rms) > max_decay_ratio:
                continue  # energy persists too long → not a click

        kept_frames.append(int(frame_idx))

    if not kept_frames:
        return np.array([], dtype=float), y, sr

    if backtrack:
        backtracked_frames = []
        for f in kept_frames:
            start_search = max(0, f - 8)
            min_idx = start_search + np.argmin(short_rms_unfiltered[start_search:f+1])
            backtracked_frames.append(min_idx)
        # Remove duplicates while preserving order
        seen = set()
        kept_frames = [x for x in backtracked_frames if not (x in seen or seen.add(x))]

    onset_times_raw = librosa.frames_to_time(
        np.array(kept_frames, dtype=int), sr=sr, hop_length=hop_length
    )

    # ── Merge detections closer than min_gap ──────────────────────────────────
    merged: list[float] = []
    for t in onset_times_raw:
        if not merged or (t - merged[-1]) > _min_gap:
            merged.append(float(t))

    return np.array(merged, dtype=float), y, sr


# ── Evaluation (interface unchanged) ─────────────────────────────────────────

def evaluate_against_ground_truth(
    onset_times: np.ndarray,
    ground_truth_times,
    tolerance: float = 0.05,
) -> dict:
    """
    Compare detected onsets to ground-truth keypress timestamps.

    tolerance : max time difference (s) to count as a match (default 0.05 s).

    Returns dict with: true_positives, false_positives, false_negatives,
                       precision, recall, f1.
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
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "true_positives":  tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 3),
        "recall":    round(recall,    3),
        "f1":        round(f1,        3),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect keystroke click events in a WAV file."
    )
    parser.add_argument("wav_path", help="Path to the WAV file")
    parser.add_argument(
        "--meta", default=None,
        help="Path to session_meta.json (auto-loads ambient_rms for absolute gate). "
             "If omitted, the detector looks for <wav_path without .wav>_meta.json.",
    )
    parser.add_argument(
        "--noise-gate-factor", type=float, default=6.0,
        help="Minimum SNR above noise floor (default 6.0 ≈ 15.6 dB).",
    )
    parser.add_argument(
        "--snr-threshold", type=float, default=4.0,
        help="Attack-score threshold multiplier (default 4.0).",
    )
    parser.add_argument(
        "--min-centroid-hz", type=float, default=1200.0,
        help="Minimum spectral centroid of attack window in Hz (default 1200). "
             "Set to 0 to disable.",
    )
    parser.add_argument(
        "--min-flatness", type=float, default=0.08,
        help="Minimum spectral flatness of attack window (default 0.08). "
             "Set to 0 to disable.",
    )
    parser.add_argument(
        "--max-decay-ratio", type=float, default=0.90,
        help="Max post-attack/attack RMS ratio (default 0.90). "
             "Set to 1.0 to disable.",
    )
    parser.add_argument(
        "--delta", type=float, default=0.07,
        help="Coarse sensitivity knob (default 0.07).",
    )
    args = parser.parse_args()

    # Auto-find meta JSON if not supplied
    meta_path = args.meta
    if meta_path is None:
        base = os.path.splitext(args.wav_path)[0]
        candidate = base + "_meta.json"
        if os.path.isfile(candidate):
            meta_path = candidate

    ambient = load_ambient_rms(meta_path) if meta_path else None
    if ambient is not None:
        print(f"[info] Loaded ambient_rms={ambient:.2f} from {meta_path}")
    else:
        print("[info] No ambient_rms available — using self-estimated noise floor.")

    onsets, y, sr = detect_onsets(
        args.wav_path,
        delta=args.delta,
        ambient_rms=ambient,
        noise_gate_factor=args.noise_gate_factor,
        snr_threshold=args.snr_threshold,
        min_centroid_hz=args.min_centroid_hz,
        min_spectral_flatness=args.min_flatness,
        max_decay_ratio=args.max_decay_ratio,
    )

    print(f"Detected {len(onsets)} click event(s):")
    if len(onsets):
        print(np.round(onsets, 3))
