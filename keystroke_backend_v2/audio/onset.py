import numpy as np
import librosa
from scipy.signal import butter, filtfilt
from typing import List, Tuple, Dict, Any, Optional


def detect_onsets(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 256,
    backtrack: bool = False,
    wait_ms: float = 65.0,
    min_peak: float = 0.022,
    min_rms: float = 0.002
) -> np.ndarray:
    """
    Detects true acoustic keystroke onset times (in seconds) in continuous audio.
    Filters out background hiss, continuous silence, stationary fan/AC noise, and boundary artifacts.
    """
    if len(audio) == 0:
        return np.array([], dtype=np.float32)

    total_peak = float(np.max(np.abs(audio)))
    total_rms = float(np.sqrt(np.mean(audio ** 2)))

    # 1. Global silence / ambient noise amplitude check
    if total_peak < min_peak or total_rms < min_rms:
        return np.array([], dtype=np.float32)

    # 2. Dynamic Energy / Noise Floor Analysis
    frame_len = 512
    frame_hop = 128
    f_rms = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=frame_hop)[0]
    if len(f_rms) == 0:
        return np.array([], dtype=np.float32)

    noise_floor = float(np.percentile(f_rms, 15))
    peak_frame_rms = float(np.max(f_rms))
    dynamic_ratio = peak_frame_rms / (noise_floor + 1e-6)

    # Reject stationary / flat background noise (fan/AC/hiss) that lacks impulsive dynamic transients
    if dynamic_ratio < 2.1 and total_peak < 0.070:
        return np.array([], dtype=np.float32)

    wait_frames = max(1, int((wait_ms / 1000.0) * sr / hop_length))

    # Focus onset envelope on mechanical keystroke frequencies (500Hz - 8000Hz)
    onset_env = librosa.onset.onset_strength(
        y=audio, sr=sr, hop_length=hop_length, fmin=500, fmax=8000
    )

    env_mean = float(np.mean(onset_env))
    env_max = float(np.max(onset_env))
    if env_max < 1e-4 or (env_max / (env_mean + 1e-6)) < 1.7:
        return np.array([], dtype=np.float32)

    # Normalize envelope to [0, 1] range for stable thresholding
    norm_env = onset_env / (env_max + 1e-8)

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=norm_env,
        sr=sr,
        hop_length=hop_length,
        backtrack=backtrack,
        wait=wait_frames,
        delta=0.14
    )

    if len(onset_frames) == 0:
        return np.array([], dtype=np.float32)

    candidate_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    # Highpass filter for mechanical impact transient check
    nyq = 0.5 * sr
    b, a = butter(2, 450.0 / nyq, btype='high')

    valid_onsets = []
    half_check = int(0.060 * sr)
    min_onset_sample = int(0.050 * sr)  # Reject STFT boundary ramp-up artifact

    for t in candidate_times:
        idx = int(t * sr)
        if idx < min_onset_sample:
            continue

        start = max(0, idx - int(0.010 * sr))
        end = min(len(audio), idx + half_check)
        local_clip = audio[start:end]
        if len(local_clip) < int(0.015 * sr):
            continue

        local_peak = float(np.max(np.abs(local_clip)))
        local_rms = float(np.sqrt(np.mean(local_clip ** 2)))
        crest = local_peak / (local_rms + 1e-6)

        # Candidate transient validation: must have peak above noise floor & impulse crest factor
        if local_peak < min_peak or crest < 2.3 or local_peak < (1.8 * noise_floor):
            continue

        # Check high frequency mechanical click content (>450Hz)
        try:
            hf = filtfilt(b, a, local_clip)
            hf_peak = float(np.max(np.abs(hf)))
            if (hf_peak / (local_peak + 1e-6)) < 0.14:
                continue
        except Exception:
            pass

        valid_onsets.append(t)

    return np.array(valid_onsets, dtype=np.float32)


def align_onset_local_neighborhood(
    audio: np.ndarray,
    sr: int,
    logged_center_idx: int,
    search_radius_ms: float = 150.0,
    window_duration_s: float = 0.200,
    method: str = "hybrid"
) -> Tuple[int, Dict[str, Any]]:
    """
    Searches a local neighborhood around logged_center_idx (+/- search_radius_ms)
    to find the true acoustic keystroke onset using candidate methods:
    1. 'rms_envelope': RMS energy peak
    2. 'spectral_flux': Librosa onset strength peak
    3. 'high_frequency_energy': High-frequency (4kHz-11kHz) band energy peak
    4. 'peak_amplitude': Max absolute amplitude peak
    5. 'hybrid': Weighted combination of spectral flux, high frequency energy, and peak amplitude
    
    Returns (best_center_idx, quality_info_dict).
    """
    search_samples = int((search_radius_ms / 1000.0) * sr)
    min_idx = max(0, logged_center_idx - search_samples)
    max_idx = min(len(audio), logged_center_idx + search_samples)

    if (max_idx - min_idx) < int(0.050 * sr):
        # Audio snippet too small
        return logged_center_idx, {
            "onset_confidence": 0.0,
            "method_used": method,
            "offset_ms": 0.0,
            "peak_amp": 0.0,
            "rms": 0.0,
            "snr_db": 0.0,
            "overlap_indicator": False,
            "quality_tier": "unusable"
        }

    sub_audio = audio[min_idx:max_idx]

    if method == "rms_envelope":
        # Frame-wise RMS
        frame_len = int(0.010 * sr)
        hop_len = int(0.002 * sr)
        rms = librosa.feature.rms(y=sub_audio, frame_length=frame_len, hop_length=hop_len)[0]
        peak_frame = np.argmax(rms)
        candidate_rel_sample = peak_frame * hop_len

    elif method == "spectral_flux":
        hop_len = int(0.002 * sr)
        onset_env = librosa.onset.onset_strength(y=sub_audio, sr=sr, hop_length=hop_len)
        peak_frame = np.argmax(onset_env)
        candidate_rel_sample = peak_frame * hop_len

    elif method == "high_frequency_energy":
        # Highpass filter above 3000 Hz for key strike click
        nyq = 0.5 * sr
        b, a = butter(4, 3000.0 / nyq, btype='high')
        filtered = filtfilt(b, a, sub_audio)
        candidate_rel_sample = np.argmax(np.abs(filtered))

    elif method == "peak_amplitude":
        candidate_rel_sample = np.argmax(np.abs(sub_audio))

    else:  # "hybrid" default
        hop_len = int(0.002 * sr)
        onset_env = librosa.onset.onset_strength(y=sub_audio, sr=sr, hop_length=hop_len)
        if len(onset_env) > 0 and np.max(onset_env) > 0:
            onset_env = onset_env / np.max(onset_env)
        else:
            onset_env = np.zeros_like(onset_env)

        # Resample envelope to match sample array size
        env_interp = np.interp(np.arange(len(sub_audio)), np.linspace(0, len(sub_audio), len(onset_env)), onset_env)
        peak_norm = np.abs(sub_audio) / (np.max(np.abs(sub_audio)) + 1e-8)

        score = 0.6 * env_interp + 0.4 * peak_norm
        candidate_rel_sample = np.argmax(score)

    best_center_idx = min_idx + candidate_rel_sample
    offset_ms = float((best_center_idx - logged_center_idx) / sr * 1000.0)

    # Extract centered clip to evaluate quality
    half_w = int((window_duration_s / 2.0) * sr)
    clip_s = max(0, best_center_idx - half_w)
    clip_e = min(len(audio), best_center_idx + half_w)
    clip = audio[clip_s:clip_e]

    peak_amp = float(np.max(np.abs(clip))) if len(clip) > 0 else 0.0
    rms_val = float(np.sqrt(np.mean(clip ** 2))) if len(clip) > 0 else 0.0

    # SNR estimate (ratio of transient peak window RMS to pre-transient background noise RMS)
    noise_s = max(0, clip_s - int(0.050 * sr))
    noise_e = clip_s
    noise_clip = audio[noise_s:noise_e] if noise_e > noise_s else np.array([])
    noise_rms = float(np.sqrt(np.mean(noise_clip ** 2))) if len(noise_clip) > 0 else 1e-5
    snr_db = float(20.0 * np.log10((rms_val + 1e-8) / (noise_rms + 1e-8)))

    # Overlap indicator: check if another strong peak exists within +/- 100ms
    surround_s = max(0, best_center_idx - int(0.120 * sr))
    surround_e = min(len(audio), best_center_idx + int(0.120 * sr))
    surround_audio = np.abs(audio[surround_s:surround_e])
    
    # Mask out the main peak region (+/- 15ms)
    mask_s = max(0, best_center_idx - min_idx - int(0.015 * sr))
    mask_e = min(len(surround_audio), best_center_idx - min_idx + int(0.015 * sr))
    surround_audio[mask_s:mask_e] = 0.0
    overlap_indicator = bool(np.max(surround_audio) > 0.6 * peak_amp) if len(surround_audio) > 0 else False

    # Quality categorization
    if peak_amp < 0.005 or rms_val < 0.001:
        quality_tier = "unusable"
        confidence = 0.1
    elif overlap_indicator or snr_db < 3.0:
        quality_tier = "low_quality" if not overlap_indicator else "overlapping"
        confidence = 0.5
    else:
        quality_tier = "valid"
        confidence = float(min(1.0, max(0.6, snr_db / 20.0)))

    quality_info = {
        "onset_confidence": confidence,
        "method_used": method,
        "offset_ms": offset_ms,
        "peak_amp": peak_amp,
        "rms": rms_val,
        "snr_db": snr_db,
        "overlap_indicator": overlap_indicator,
        "quality_tier": quality_tier
    }

    return best_center_idx, quality_info


def evaluate_onsets(
    detected_onsets_s: np.ndarray,
    ground_truth_onsets_s: np.ndarray,
    tolerance_ms: float = 50.0
) -> Dict[str, float]:
    """
    Evaluates onset detection precision, recall, F1, and mean absolute timing error (ms).
    """
    if len(ground_truth_onsets_s) == 0:
        fp_cnt = len(detected_onsets_s)
        prec = 1.0 if fp_cnt == 0 else 0.0
        return {"precision": prec, "recall": 1.0, "f1": prec, "timing_error_ms": 0.0, "tp": 0, "fp": fp_cnt, "fn": 0}
    if len(detected_onsets_s) == 0:
        fn_cnt = len(ground_truth_onsets_s)
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "timing_error_ms": 0.0, "tp": 0, "fp": 0, "fn": fn_cnt}

    tol_s = tolerance_ms / 1000.0
    matched_gt = set()
    matched_det = set()
    errors_s = []

    for i, det in enumerate(detected_onsets_s):
        diffs = np.abs(ground_truth_onsets_s - det)
        best_gt_idx = np.argmin(diffs)
        best_diff = diffs[best_gt_idx]

        if best_diff <= tol_s and best_gt_idx not in matched_gt:
            matched_gt.add(best_gt_idx)
            matched_det.add(i)
            errors_s.append(best_diff)

    tp = len(matched_det)
    fp = len(detected_onsets_s) - tp
    fn = len(ground_truth_onsets_s) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    timing_error_ms = float(np.mean(errors_s) * 1000.0) if errors_s else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "timing_error_ms": float(timing_error_ms),
        "tp": tp,
        "fp": fp,
        "fn": fn
    }
