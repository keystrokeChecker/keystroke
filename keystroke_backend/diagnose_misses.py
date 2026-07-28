"""
Diagnostic script to investigate why specific ground-truth keypress timestamps were missed.
Compares signal metrics against each gate in detect_onsets().

USAGE:
    python diagnose_misses.py data/latency_calib.wav --misses 2.807 10.097 12.557 13.857
"""

import argparse
import os
import json
import librosa
import numpy as np
from scipy.signal import find_peaks

from onset_detector import (
    load_ambient_rms,
    _butter_bandpass,
    filtfilt,
    _spectral_centroid_hz,
    _spectral_flatness,
)


def diagnose_recording(wav_path, missed_timestamps, delta=0.07, noise_gate_factor=2.0,
                       min_centroid_hz=1200.0, min_spectral_flatness=0.08, max_decay_ratio=0.90):
    meta_path = os.path.splitext(wav_path)[0] + "_meta.json"
    ambient_rms = load_ambient_rms(meta_path)

    y_raw, sr = librosa.load(wav_path, sr=None)
    if y_raw.ndim > 1:
        y_raw = np.mean(y_raw, axis=0)
    y_raw = y_raw.astype(np.float32)

    raw_peak = float(np.max(np.abs(y_raw)))
    y = y_raw / raw_peak

    if ambient_rms is not None and float(ambient_rms) > 0.0:
        ambient_float = float(ambient_rms) / 32768.0
        ambient_norm = ambient_float / raw_peak
    else:
        ambient_norm = None

    hop_length = 256
    short_frame = max(64, hop_length // 2)
    long_frame = max(256, hop_length * 2)

    low_cut, high_cut = 800.0, 8000.0
    if sr < 2 * high_cut:
        high_cut = max(1000.0, sr / 2.0 - 100.0)
    b, a = _butter_bandpass(low_cut, high_cut, sr)
    filtered = filtfilt(b, a, y, method="pad")

    short_rms_unfiltered = librosa.feature.rms(y=y, frame_length=short_frame, hop_length=hop_length)[0]
    short_rms_unfiltered = np.maximum(short_rms_unfiltered, 1e-12)

    short_rms = librosa.feature.rms(y=filtered, frame_length=short_frame, hop_length=hop_length)[0]
    long_rms = librosa.feature.rms(y=filtered, frame_length=long_frame, hop_length=hop_length)[0]
    short_rms = np.maximum(short_rms, 1e-12)
    long_rms = np.maximum(long_rms, 1e-12)

    inband_noise_rms = float(np.percentile(short_rms_unfiltered, 5))
    abs_noise_floor = inband_noise_rms
    if ambient_norm is not None and float(ambient_norm) <= 2.0 * inband_noise_rms:
        abs_noise_floor = max(inband_noise_rms, float(ambient_norm))
    abs_noise_floor = max(abs_noise_floor, 1e-9)

    energy_gate_rms = noise_gate_factor * abs_noise_floor
    gate_mask = short_rms_unfiltered < energy_gate_rms

    transient_score = short_rms / long_rms

    transient_score_ungated = transient_score.copy()
    _smooth_win = 3
    kernel = np.ones(_smooth_win, dtype=np.float32) / _smooth_win
    if len(transient_score_ungated) > _smooth_win:
        transient_score_ungated = np.convolve(transient_score_ungated, kernel, mode="same")
    attack_score_ungated = np.diff(transient_score_ungated)
    attack_score_ungated = np.clip(attack_score_ungated, 0.0, None)

    transient_score_gated = np.where(gate_mask, 0.0, transient_score)
    if len(transient_score_gated) > _smooth_win:
        transient_score_gated = np.convolve(transient_score_gated, kernel, mode="same")
    attack_score = np.diff(transient_score_gated)
    attack_score = np.clip(attack_score, 0.0, None)

    noise_floor_ungated = np.median(attack_score_ungated)
    dynamic_range_ungated = np.percentile(attack_score_ungated, 90) - noise_floor_ungated
    dynamic_range_ungated = max(dynamic_range_ungated, np.std(attack_score_ungated) + 1e-6)

    threshold = noise_floor_ungated + 0.7 * dynamic_range_ungated + 0.12 * max(0.0, delta)
    threshold = max(threshold, np.percentile(attack_score_ungated, 90) * 0.55)
    threshold = max(threshold, 0.10 + 0.12 * max(0.0, delta))

    print(f"=== GLOBAL PIPELINE PARAMETERS FOR: {wav_path} ===")
    print(f"  raw_peak (PCM float): {raw_peak:.6f}")
    print(f"  ambient_rms (int16) : {ambient_rms}")
    print(f"  ambient_norm        : {ambient_norm:.6f}" if ambient_norm else "  ambient_norm: None")
    print(f"  inband_noise_rms    : {inband_noise_rms:.6f}")
    print(f"  abs_noise_floor     : {abs_noise_floor:.6f}")
    print(f"  energy_gate_rms     : {energy_gate_rms:.6f} (factor={noise_gate_factor})")
    print(f"  attack_threshold    : {threshold:.6f} (delta={delta})")
    print("=========================================================\n")

    attack_win_samples = int(0.005 * sr)
    decay_start_samples = int(0.015 * sr)
    decay_win_samples = int(0.025 * sr)
    attack_half_samples = int(0.015 * sr)

    for t_gt in missed_timestamps:
        print(f"--- DIAGNOSING MISSED PRESS AT t = {t_gt:.3f}s ---")
        
        # Search window around ground truth timestamp (±0.4s)
        f_center = int(round(t_gt * sr / hop_length))
        f_start = max(0, f_center - int(0.4 * sr / hop_length))
        f_end = min(len(attack_score), f_center + int(0.4 * sr / hop_length))

        # Find the peak attack score in this neighborhood
        window_scores = attack_score[f_start:f_end]
        if len(window_scores) == 0:
            print("  [ERROR] Window out of bounds")
            continue
            
        local_peak_rel = np.argmax(window_scores)
        local_peak_frame = f_start + local_peak_rel
        peak_t = local_peak_frame * hop_length / sr
        peak_score = attack_score[local_peak_frame]
        peak_score_ungated = attack_score_ungated[local_peak_frame]

        # Also check short RMS at peak
        frame_rms_unfiltered = short_rms_unfiltered[min(local_peak_frame, len(short_rms_unfiltered)-1)]

        print(f"  Nearest candidate peak at t = {peak_t:.3f}s (diff: {peak_t - t_gt:+.3f}s)")
        
        # Gate 1: Energy Gate
        pass_energy = frame_rms_unfiltered >= energy_gate_rms
        print(f"  1. Energy Gate          : {'PASS' if pass_energy else 'FAIL'} | Value={frame_rms_unfiltered:.6f} vs Gate={energy_gate_rms:.6f}")

        # Gate 2: Attack Score Threshold
        pass_attack = peak_score >= threshold
        print(f"  2. Attack Score Threshold: {'PASS' if pass_attack else 'FAIL'} | GatedScore={peak_score:.6f} (Ungated={peak_score_ungated:.6f}) vs Thresh={threshold:.6f}")

        # Gates 3-5: Spectral and Decay Gates
        sample_idx = int((local_peak_frame + 1) * hop_length)
        s0 = max(0, sample_idx - attack_half_samples)
        s1 = min(len(filtered), sample_idx + attack_half_samples)
        chunk = filtered[s0:s1]

        centroid = _spectral_centroid_hz(chunk, sr) if len(chunk) >= 4 else 0.0
        flatness = _spectral_flatness(chunk, sr) if len(chunk) >= 4 else 0.0

        a0, a1 = sample_idx, min(len(filtered), sample_idx + attack_win_samples)
        attack_chunk = filtered[a0:a1]
        attack_rms = float(np.sqrt(np.mean(attack_chunk ** 2))) if len(attack_chunk) > 0 else 0.0

        d0 = min(len(filtered), sample_idx + decay_start_samples)
        d1 = min(len(filtered), d0 + decay_win_samples)
        decay_chunk = filtered[d0:d1]
        decay_rms = float(np.sqrt(np.mean(decay_chunk ** 2))) if len(decay_chunk) > 0 else 0.0
        decay_ratio = decay_rms / attack_rms if attack_rms > 1e-9 else 0.0

        pass_centroid = centroid >= min_centroid_hz
        print(f"  3. Spectral Centroid Gate: {'PASS' if pass_centroid else 'FAIL'} | Centroid={centroid:.1f} Hz vs Min={min_centroid_hz:.1f} Hz")

        pass_flatness = flatness >= min_spectral_flatness
        print(f"  4. Spectral Flatness Gate: {'PASS' if pass_flatness else 'FAIL'} | Flatness={flatness:.4f} vs Min={min_spectral_flatness:.4f}")

        pass_decay = decay_ratio <= max_decay_ratio
        print(f"  5. Decay Ratio Gate      : {'PASS' if pass_decay else 'FAIL'} | DecayRatio={decay_ratio:.4f} vs Max={max_decay_ratio:.4f}")

        failed_gates = []
        if not pass_energy: failed_gates.append("Energy Gate")
        if not pass_attack: failed_gates.append("Attack Threshold")
        if not pass_centroid: failed_gates.append("Spectral Centroid")
        if not pass_flatness: failed_gates.append("Spectral Flatness")
        if not pass_decay: failed_gates.append("Decay Ratio")

        print(f"  ==> FINAL DIAGNOSIS FOR {t_gt:.3f}s: {'ALL GATES PASSED' if not failed_gates else 'FAILED GATES: ' + ', '.join(failed_gates)}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Diagnose missed keypresses in onset detector.")
    parser.add_argument("wav_path", help="Path to WAV file (e.g. data/latency_calib.wav)")
    parser.add_argument("--misses", nargs="+", type=float, required=True, help="List of missed ground truth timestamps")
    args = parser.parse_args()

    diagnose_recording(args.wav_path, args.misses)


if __name__ == "__main__":
    main()
