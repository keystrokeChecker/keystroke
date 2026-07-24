"""
Candidate classifier implementation using hand-crafted features.
"""

import argparse
import os
import numpy as np
import librosa
import joblib
from onset_detector import evaluate_against_ground_truth
from tune_and_evaluate import load_ground_truth

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def normalize_recording(y):
    """
    Peak-normalizes the input signal and estimates the per-recording noise floor.
    """
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    y = y.astype(np.float32)
    
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
        
    # noise floor estimate: 10th percentile of RMS across the whole file
    frame_length = 1024
    hop_length = 256
    if len(y) > frame_length:
        rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        noise_floor = float(np.percentile(rms_frames, 10))
    else:
        noise_floor = float(np.mean(np.abs(y)))
        
    return y, noise_floor

def extract_candidates(wav_path, delta=0.03):
    """
    Extract onset candidates using librosa's built-in onset detector.
    """
    y, sr = librosa.load(wav_path, sr=None)
    y, noise_floor = normalize_recording(y)
    onsets = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        backtrack=False,
        units='time',
        hop_length=256,
        delta=delta
    )
    return onsets, y, sr

def extract_features(y, sr, onset_time, window_ms=60):
    """
    Extract normalized features for a given candidate onset window.
    """
    window_sec = window_ms / 1000.0
    start_time = onset_time - 0.010
    start_sample = int(start_time * sr)
    end_sample = int((start_time + window_sec) * sr)
    
    start_sample = max(0, start_sample)
    end_sample = min(len(y), end_sample)
    window_y = y[start_sample:end_sample]
    
    # Calculate recording-wide statistics
    abs_y = np.abs(y)
    rec_95th = float(np.percentile(abs_y, 95)) if len(abs_y) > 0 else 1.0
    if rec_95th == 0:
        rec_95th = 1e-6
        
    frame_length = min(len(y), 1024)
    hop_length = 256
    if len(y) > frame_length:
        rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        rec_noise_floor = float(np.percentile(rms_frames, 10))
    else:
        rec_noise_floor = float(np.mean(abs_y))
    if rec_noise_floor == 0:
        rec_noise_floor = 1e-6
        
    if len(window_y) == 0:
        return np.zeros(8, dtype=np.float32)
        
    win_peak = float(np.max(np.abs(window_y)))
    win_rms = float(np.sqrt(np.mean(window_y**2)))
    
    peak_ratio = win_peak / rec_95th
    rms_ratio = win_rms / rec_noise_floor
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=window_y)))
    
    n_fft = min(len(window_y), 512)
    if n_fft < 16:
        centroid = 0.0
        bandwidth = 0.0
        rolloff = 0.0
    else:
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=window_y, sr=sr, n_fft=n_fft)))
        bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=window_y, sr=sr, n_fft=n_fft)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=window_y, sr=sr, n_fft=n_fft)))
        
    half = len(window_y) // 2
    first_half = window_y[:half]
    second_half = window_y[half:]
    rms_first = float(np.sqrt(np.mean(first_half**2))) if len(first_half) > 0 else 0.0
    rms_second = float(np.sqrt(np.mean(second_half**2))) if len(second_half) > 0 else 0.0
    decay_ratio = rms_second / (rms_first + 1e-6)
    
    samples_10ms = int(0.010 * sr)
    attack_start = int(0.010 * sr)
    attack_segment = window_y[attack_start:attack_start + samples_10ms]
    if len(attack_segment) > 1:
        max_diff = float(np.max(np.abs(np.diff(attack_segment))))
    else:
        max_diff = 0.0
    attack_sharpness = max_diff / rec_95th
    
    return np.array([
        peak_ratio,
        rms_ratio,
        zcr,
        centroid,
        bandwidth,
        rolloff,
        decay_ratio,
        attack_sharpness
    ], dtype=np.float32)

def build_training_set(session_names, delta=0.03):
    """
    Build features and labels from training sessions.
    """
    X_list = []
    y_list = []
    groups_list = []
    
    for name in session_names:
        wav_path = os.path.join(DATA_DIR, f"{name}.wav")
        log_path = os.path.join(DATA_DIR, f"{name}_log.csv")
        
        if not os.path.exists(wav_path) or not os.path.exists(log_path):
            print(f"Skipping {name}: file(s) not found.")
            continue
            
        onsets, y, sr = extract_candidates(wav_path, delta=delta)
        _, gt_times = load_ground_truth(log_path)
        gt_times = np.array(gt_times)
        
        labels = []
        for det_t in onsets:
            is_pos = int(np.any(np.abs(gt_times - det_t) <= 0.08))
            labels.append(is_pos)
            
        session_features = []
        for det_t in onsets:
            feats = extract_features(y, sr, det_t)
            session_features.append(feats)
            
        if len(session_features) > 0:
            X_list.append(np.stack(session_features))
            y_list.extend(labels)
            groups_list.extend([name] * len(labels))
            
    if not X_list:
        return np.empty((0, 8)), np.array([]), np.array([])
        
    return np.vstack(X_list), np.array(y_list), np.array(groups_list)

def train_and_evaluate(session_names, delta=0.03):
    """
    Train and evaluate classifier using LeaveOneGroupOut CV.
    """
    X, y, groups = build_training_set(session_names, delta=delta)
    if len(X) == 0:
        print("No candidates extracted. Cannot train.")
        return
        
    unique_sessions = np.unique(groups)
    if len(unique_sessions) < 3:
        print(f"\n⚠️  WARNING: The model is only being trained on {len(unique_sessions)} distinct session(s).")
        print("   Real-world generalization to new devices/mics is UNVERIFIED until tested on external recordings.")
        
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import precision_recall_fscore_support
    
    logo = LeaveOneGroupOut()
    fold_scores = []
    
    print("\nLeave-One-Session-Out Cross-Validation:")
    for train_idx, test_idx in logo.split(X, y, groups):
        held_out = groups[test_idx[0]]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        fold_scores.append((p, r, f))
        print(f"  Held out {held_out}: Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}")
        
    if fold_scores:
        mean_scores = np.mean(fold_scores, axis=0)
        print(f"  Mean Cross-Validation Score: Precision={mean_scores[0]:.3f}, Recall={mean_scores[1]:.3f}, F1={mean_scores[2]:.3f}")
        
    # Feature Importance
    clf_full = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    clf_full.fit(X, y)
    
    feature_names = [
        "peak_amplitude_ratio",
        "rms_energy_ratio",
        "zero_crossing_rate",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff",
        "decay_ratio",
        "attack_sharpness"
    ]
    print("\nFeature Importances:")
    importances = clf_full.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in indices:
        print(f"  {feature_names[i]:25}: {importances[i]:.4f}")
        
    # Save the model
    os.makedirs(os.path.join(os.path.dirname(__file__), "models"), exist_ok=True)
    model_path = os.path.join(os.path.dirname(__file__), "models", "candidate_classifier.joblib")
    joblib.dump({
        "classifier": clf_full,
        "feature_names": feature_names,
        "delta": delta
    }, model_path)
    print(f"\nSaved model to {model_path}")

def filter_candidates(wav_path, classifier_path, delta=0.03, threshold=0.5):
    """
    Filter candidates using the trained Random Forest classifier.
    """
    payload = joblib.load(classifier_path)
    clf = payload["classifier"]
    
    onsets, y, sr = extract_candidates(wav_path, delta=delta)
    if len(onsets) == 0:
        return np.array([])
        
    features = []
    for det_t in onsets:
        feats = extract_features(y, sr, det_t)
        features.append(feats)
        
    X = np.stack(features)
    probs = clf.predict_proba(X)[:, 1]
    return onsets[probs >= threshold]

def validate_on_new_recording(wav_path, log_path, classifier_path, threshold=0.3):
    """
    Validate model on out-of-sample data.
    """
    _, gt_times = load_ground_truth(log_path)
    gt_times = np.array(gt_times)
    
    filtered_onsets = filter_candidates(wav_path, classifier_path, threshold=threshold)
    ev = evaluate_against_ground_truth(filtered_onsets, gt_times, tolerance=0.08)
    
    print(f"\n========================================================")
    print(f"  OUT-OF-SAMPLE VALIDATION ON NEW DEVICE/SETUP")
    print(f"  WAV File: {os.path.basename(wav_path)}")
    print(f"  Filtering Probability Threshold: {threshold:.2f}")
    print(f"========================================================")
    print(f"  Detected Filtered Onsets: {len(filtered_onsets)} (Ground Truth count: {len(gt_times)})")
    print(f"  Precision: {ev['precision']:.3f}")
    print(f"  Recall:    {ev['recall']:.3f}")
    print(f"  F1 Score:  {ev['f1']:.3f}")
    print(f"========================================================\n")
    return ev

def main():
    parser = argparse.ArgumentParser(description="Candidate classifier using hand-crafted features.")
    parser.add_argument("--train", action="store_true", help="Train and evaluate classifier.")
    parser.add_argument("--names", nargs="+", default=["session1", "session2", "session3"])
    parser.add_argument("--validate-wav", type=str, default=None, help="Path to WAV file for validation.")
    parser.add_argument("--validate-log", type=str, default=None, help="Path to CSV log file for validation.")
    parser.add_argument("--classifier", type=str, default=None, help="Path to classifier model.")
    parser.add_argument("--threshold", type=float, default=0.3, help="Probability threshold for classification (default: 0.3).")
    args = parser.parse_args()
    
    classifier_path = args.classifier or os.path.join(os.path.dirname(__file__), "models", "candidate_classifier.joblib")
    
    if args.train:
        train_and_evaluate(args.names)
    elif args.validate_wav and args.validate_log:
        validate_on_new_recording(args.validate_wav, args.validate_log, classifier_path, threshold=args.threshold)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
