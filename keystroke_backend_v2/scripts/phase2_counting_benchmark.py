import os
import sys
import io
import zipfile
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from audio.io import resample_audio
from audio.filtering import highpass_filter, preemphasis
from audio.normalization import apply_normalization
from audio.onset import detect_onsets, evaluate_onsets


def method1_energy_threshold(audio: np.ndarray, sr: int = 22050, threshold: float = 0.05, wait_ms: float = 60.0) -> np.ndarray:
    """Method 1: Simple RMS Energy Thresholding."""
    frame_len, hop_len = 512, 128
    f_rms = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=hop_len)[0]
    wait_frames = int((wait_ms / 1000.0) * sr / hop_len)
    
    peaks = []
    last_peak = -wait_frames
    for i, val in enumerate(f_rms):
        if val > threshold and (i - last_peak) >= wait_frames:
            peaks.append(i * hop_len / sr)
            last_peak = i
    return np.array(peaks, dtype=np.float32)


def method2_standard_spectral_flux(audio: np.ndarray, sr: int = 22050, delta: float = 0.18) -> np.ndarray:
    """Method 2: Standard Librosa Spectral Flux Onset Detection."""
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=256)
    frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=256, backtrack=True, delta=delta)
    return librosa.frames_to_time(frames, sr=sr, hop_length=256)


def method3_high_frequency_content(audio: np.ndarray, sr: int = 22050, threshold: float = 0.03) -> np.ndarray:
    """Method 3: High-Frequency Content (HFC) Weighted Energy Detector."""
    S = np.abs(librosa.stft(audio, n_fft=512, hop_length=128))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    # Weight bins by frequency (high frequencies heavily weighted)
    hfc = np.sum(S * freqs[:, np.newaxis], axis=0) / (np.sum(S, axis=0) + 1e-6)
    hfc_norm = hfc / (np.max(hfc) + 1e-8) if np.max(hfc) > 0 else hfc
    
    # Peak pick on HFC
    peaks = []
    wait_frames = int(0.060 * sr / 128)
    last_peak = -wait_frames
    for i in range(1, len(hfc_norm) - 1):
        if hfc_norm[i] > threshold and hfc_norm[i] > hfc_norm[i-1] and hfc_norm[i] > hfc_norm[i+1]:
            if (i - last_peak) >= wait_frames:
                peaks.append(i * 128 / sr)
                last_peak = i
    return np.array(peaks, dtype=np.float32)


def method4_adaptive_transient(audio: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Method 4: Optimized Adaptive Transient Filter + Physical Keystroke Validator."""
    return detect_onsets(audio, sr=sr)


def run_benchmark():
    print("=======================================================")
    print("PHASE 2: KEYSTROKE COUNTING & ONSET DETECTION BENCHMARK")
    print("=======================================================")

    # 1. Build Ground Truth Test Set:
    # 50 Clean Isolated Keystrokes (Kaggle), 30 Multi-Pressure (High/Med/Low), 20 Noise/Silence (0 keystrokes), 10 Continuous Scripted Multi-Keystroke Streams
    np.random.seed(42)
    sr = 22050

    ground_truth_test_cases = []

    # A. Kaggle Isolated Keystrokes (1 keystroke per file)
    with zipfile.ZipFile(os.path.join(BASE_DIR, "data", "archive.zip"), 'r') as zf:
        wavs = [n for n in zf.namelist() if n.endswith('.wav')][:50]
        for w in wavs:
            d, orig_sr = sf.read(io.BytesIO(zf.read(w)))
            if d.ndim > 1: d = np.mean(d, axis=1)
            d = resample_audio(d, orig_sr, sr)
            # True onset is located at transient peak
            peak_samp = np.argmax(np.abs(d))
            gt_time = float(peak_samp / sr)
            ground_truth_test_cases.append({
                "type": "isolated_clean",
                "audio": d,
                "gt_onsets": np.array([gt_time], dtype=np.float32)
            })

    # B. Multi-Pressure (High, Medium, Low pressure - 1 keystroke per file)
    with zipfile.ZipFile(os.path.join(BASE_DIR, "data", "dataset.zip"), 'r') as zf:
        wavs = [n for n in zf.namelist() if n.endswith('.wav')][:30]
        for w in wavs:
            d, orig_sr = sf.read(io.BytesIO(zf.read(w)))
            if d.ndim > 1: d = np.mean(d, axis=1)
            d = resample_audio(d, orig_sr, sr)
            peak_samp = np.argmax(np.abs(d))
            gt_time = float(peak_samp / sr)
            ground_truth_test_cases.append({
                "type": "multi_pressure",
                "audio": d,
                "gt_onsets": np.array([gt_time], dtype=np.float32)
            })

    # C. Pure Silence & Ambient Noise (0 keystrokes)
    for seed in range(20):
        np.random.seed(seed)
        noise = (np.random.randn(sr * 3) * (0.004 + seed * 0.002)).astype(np.float32)
        ground_truth_test_cases.append({
            "type": "noise_silence",
            "audio": noise,
            "gt_onsets": np.array([], dtype=np.float32)
        })

    # D. Continuous Multi-Keystroke Scripted Streams (3-5 keys spaced out)
    for stream_idx in range(10):
        np.random.seed(stream_idx + 100)
        stream_len = sr * 5
        stream_audio = (np.random.randn(stream_len) * 0.005).astype(np.float32)
        num_keys = 4
        gt_times = []
        for k in range(num_keys):
            t_s = 0.8 + k * 1.0 + np.random.uniform(-0.1, 0.1)
            samp_idx = int(t_s * sr)
            key_clip = ground_truth_test_cases[k]["audio"]
            end_idx = min(stream_len, samp_idx + len(key_clip))
            clip_portion = key_clip[:(end_idx - samp_idx)]
            stream_audio[samp_idx:end_idx] += clip_portion
            peak_offset = np.argmax(np.abs(clip_portion)) / sr
            gt_times.append(t_s + peak_offset)

        ground_truth_test_cases.append({
            "type": "continuous_stream",
            "audio": stream_audio,
            "gt_onsets": np.array(sorted(gt_times), dtype=np.float32)
        })

    print(f"Total Ground Truth Test Set: {len(ground_truth_test_cases)} audio scenarios")
    total_gt_events = sum([len(tc["gt_onsets"]) for tc in ground_truth_test_cases])
    print(f"Total True Keystroke Events: {total_gt_events}")

    # 2. Evaluate all 4 Methods
    methods = [
        ("Method 1: Fixed Energy Threshold", method1_energy_threshold),
        ("Method 2: Standard Spectral Flux", method2_standard_spectral_flux),
        ("Method 3: High-Frequency Content (HFC)", method3_high_frequency_content),
        ("Method 4: Adaptive Transient + Physical Validator", method4_adaptive_transient)
    ]

    method_metrics = {}
    for name, func in methods:
        total_tp, total_fp, total_fn = 0, 0, 0
        timing_errors = []
        missed_soft = 0
        doubled_peaks = 0
        false_noise = 0

        for tc in ground_truth_test_cases:
            detected = func(tc["audio"], sr=sr)
            gt = tc["gt_onsets"]
            eval_res = evaluate_onsets(detected, gt, tolerance_ms=50.0)

            total_tp += eval_res["tp"]
            total_fp += eval_res["fp"]
            total_fn += eval_res["fn"]
            if eval_res.get("timing_error_ms", 0) > 0:
                timing_errors.append(eval_res["timing_error_ms"])

            # Failure mode categorization
            if tc["type"] == "noise_silence" and len(detected) > 0:
                false_noise += len(detected)
            elif tc["type"] == "multi_pressure" and eval_res["fn"] > 0:
                missed_soft += eval_res["fn"]
            elif len(detected) > len(gt) and tc["type"] in ["isolated_clean", "continuous_stream"]:
                doubled_peaks += (len(detected) - len(gt))

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_timing_err = np.mean(timing_errors) if timing_errors else 0.0

        method_metrics[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "timing_error_ms": mean_timing_err,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "false_noise_triggers": false_noise,
            "missed_soft_keys": missed_soft,
            "doubled_release_peaks": doubled_peaks
        }

    print("\n=======================================================")
    print("METHOD COMPARISON RESULTS ON GROUND TRUTH TEST SET:")
    print("=======================================================")
    for name, m in method_metrics.items():
        print(f"\n[{name}]")
        print(f"  - Precision: {m['precision']*100:6.2f}% (TP={m['tp']}, FP={m['fp']})")
        print(f"  - Recall:    {m['recall']*100:6.2f}% (FN={m['fn']})")
        print(f"  - F1-Score:  {m['f1']*100:6.2f}%")
        print(f"  - Mean Timing Error: {m['timing_error_ms']:.2f} ms")
        print(f"  - Failure Breakdown: False Noise Triggers={m['false_noise_triggers']}, Missed Soft Keys={m['missed_soft_keys']}, Doubled Release Peaks={m['doubled_release_peaks']}")


if __name__ == "__main__":
    run_benchmark()
