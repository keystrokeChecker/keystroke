import os
import sys
import io
import json
import zipfile
import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, top_k_accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from datasets.canonical import CANONICAL_CLASSES, CANONICAL_TO_INDEX, sanitize_raw_label, is_forbidden_recording
from audio.io import resample_audio
from audio.filtering import highpass_filter, preemphasis
from audio.normalization import apply_normalization
from features.extractor import extract_features


def estimate_transient_duration_ms(audio: np.ndarray, sr: int, frame_length: int = 512, hop_length: int = 128) -> float:
    """
    Estimates “effective transient duration” from the RMS envelope by measuring how long RMS stays
    above a noise-floor-relative threshold around the peak.

    This is used to justify feature windowing (MFCC n_fft/hop) with measured data, not paper-defaults.
    """
    if len(audio) == 0:
        return 0.0

    audio = audio.astype(np.float32, copy=False)
    if len(audio) < frame_length:
        # Too short to compute a meaningful RMS envelope: fall back to clip length.
        return 1000.0 * len(audio) / sr

    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    if len(rms) == 0:
        return 0.0

    noise_floor = float(np.percentile(rms, 15))
    if noise_floor <= 0:
        noise_floor = 1e-8

    peak_idx = int(np.argmax(rms))
    peak_rms = float(rms[peak_idx])

    # Threshold relative to noise floor. We keep this conservative so we measure the “core transient”,
    # not the full ring-down tail.
    thr = noise_floor * 3.0

    # Find contiguous frames around the peak above threshold.
    above = rms >= thr
    if not np.any(above):
        return 0.0

    left = peak_idx
    while left > 0 and above[left]:
        left -= 1
    right = peak_idx
    while right < (len(above) - 1) and above[right]:
        right += 1

    # Convert frame indices to time span using hop_length.
    duration_s = (right - left) * hop_length / sr
    return float(duration_s * 1000.0)


def load_dataset_features(max_samples_per_class: int = 150):
    print("=======================================================")
    print("PHASE 3: FEATURE EXTRACTION & DATASET READINESS AUDIT")
    print("=======================================================")
    
    zip_path = os.path.join(BASE_DIR, "data", "archive.zip")
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Dataset archive not found: {zip_path}")

    X_list = []
    y_list = []
    groups_list = []
    class_counts = {c: 0 for c in CANONICAL_CLASSES}

    sr = 22050
    transient_durations_ms = []
    transient_duration_target_samples = 300  # keep runtime bounded

    with zipfile.ZipFile(zip_path, 'r') as zf:
        namelist = [n for n in zf.namelist() if n.lower().endswith('.wav') and not is_forbidden_recording(n)]
        
        for item in sorted(namelist):
            parts = item.split('/')
            if len(parts) < 2:
                continue
            raw_label = parts[1]
            label = sanitize_raw_label(raw_label)
            if label is None or label not in CANONICAL_TO_INDEX:
                continue

            if class_counts[label] >= max_samples_per_class:
                continue

            # Group ID = recording session identifier.
            # For Kaggle, each .wav file is a single keystroke event; using the full filename prevents
            # accidental leakage from overly-coarse grouping.
            fname = parts[-1]
            group_id = f"kaggle_{raw_label}_{fname}"

            wav_bytes = zf.read(item)
            d, orig_sr = sf.read(io.BytesIO(wav_bytes))
            if d.ndim > 1: d = np.mean(d, axis=1)
            d = resample_audio(d, orig_sr, sr)

            # Preprocessing
            d = d - np.mean(d)
            d = highpass_filter(d, sr=sr, cutoff_hz=150.0, order=4)
            d = preemphasis(d, coeff=0.97)
            d_norm = apply_normalization(d, strategy="P2")

            if len(transient_durations_ms) < transient_duration_target_samples:
                transient_durations_ms.append(estimate_transient_duration_ms(d_norm, sr))

            # Extract Feature Set D (98-dim MFCC + Spectral + Temporal)
            feat = extract_features(d_norm, sr=sr, feature_set="D")
            X_list.append(feat)
            y_list.append(CANONICAL_TO_INDEX[label])
            groups_list.append(group_id)
            class_counts[label] += 1

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    groups = np.array(groups_list)

    print(f"Total Samples Loaded: {len(X)}")
    print(f"Feature Dimension (Feature Set D): {X.shape[1]}")
    print(f"Number of Classes: {len(np.unique(y))} / {len(CANONICAL_CLASSES)}")
    print(f"Distinct Group Sessions: {len(np.unique(groups))}")
    if len(transient_durations_ms) > 0:
        td = np.array(transient_durations_ms, dtype=np.float32)
        print(
            "Measured transient duration (Phase-1 pipeline, RMS core): "
            f"mean={td.mean():.1f}ms, median={np.median(td):.1f}ms, p10={np.percentile(td,10):.1f}ms, p90={np.percentile(td,90):.1f}ms"
        )
    print("\nClass Balance Audit (Samples Per Key):")
    for c in CANONICAL_CLASSES:
        cnt = class_counts[c]
        flag = " [OK]" if cnt >= 50 else " [WARNING: LOW]"
        print(f"  - Key {c:5s}: {cnt:4d} samples{flag}")

    return X, y, groups


def benchmark_models(X, y, groups):
    print("\n=======================================================")
    print("TRAIN/TEST SPLIT & CANDIDATE MODEL BENCHMARK")
    print("=======================================================")

    # Group-disjoint split (no recording/session groups shared between train and test).
    # We also enforce class coverage so every class appears in both sets.
    from sklearn.model_selection import GroupShuffleSplit

    n_classes = len(np.unique(y))
    test_size = 0.25
    split_seeds = [42, 43, 44, 45, 46, 47]
    train_idx, test_idx = None, None
    for seed in split_seeds:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(gss.split(X, y, groups=groups))

        y_tr = y[tr]
        y_te = y[te]
        if len(np.unique(y_tr)) == n_classes and len(np.unique(y_te)) == n_classes:
            train_idx, test_idx = tr, te
            break

    if train_idx is None or test_idx is None:
        # Fall back: should not happen, but keeps script robust.
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))

    # Verify group disjointness explicitly.
    overlap = np.intersect1d(groups[train_idx], groups[test_idx])
    if len(overlap) > 0:
        print(f"[WARNING] Group overlap detected! overlap_groups={len(overlap)} (this violates split rule).")
    else:
        print("Group-disjoint split: OK (no shared session groups between train/test).")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"Training Set: {len(X_train)} samples across {len(np.unique(y_train))} classes")
    print(f"Test Set:     {len(X_test)} samples across {len(np.unique(y_test))} classes")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "ExtraTrees (n=200, depth=15)": ("tree", ExtraTreesClassifier(n_estimators=200, max_depth=15, random_state=42)),
        "RandomForest (n=200, depth=15)": ("tree", RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)),
        "Linear SVM (C=1.0)": ("svm", SVC(kernel='linear', C=1.0, probability=True, random_state=42)),
        "RBF SVM (C=5.0, gamma='scale')": ("svm", SVC(kernel='rbf', C=5.0, gamma='scale', probability=True, random_state=42)),
        "k-Nearest Neighbors (k=5)": ("knn", KNeighborsClassifier(n_neighbors=5, weights='distance'))
    }

    results = {}
    best_model_name = None
    best_acc = 0.0
    best_model_kind = None
    best_model_obj = None
    best_model_test_prob = None

    for name, (kind, clf) in models.items():
        # Tree models use raw features, SVM/KNN use scaled
        is_tree = kind == "tree"
        X_tr = X_train if is_tree else X_train_scaled
        X_te = X_test if is_tree else X_test_scaled

        clf.fit(X_tr, y_train)
        y_train_pred = clf.predict(X_tr)
        y_test_pred = clf.predict(X_te)
        y_test_prob = clf.predict_proba(X_te)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        bal_acc = balanced_accuracy_score(y_test, y_test_pred)
        macro_f1 = f1_score(y_test, y_test_pred, average='macro')
        top3_acc = top_k_accuracy_score(y_test, y_test_prob, k=3, labels=np.arange(27))
        overfitting_gap = train_acc - test_acc

        results[name] = {
            "train_acc": train_acc,
            "test_acc": test_acc,
            "balanced_acc": bal_acc,
            "macro_f1": macro_f1,
            "top3_acc": top3_acc,
            "gap": overfitting_gap,
            "y_test_pred": y_test_pred
        }

        print(f"\n[{name}]")
        print(f"  - Train Accuracy:    {train_acc*100:6.2f}%")
        print(f"  - Test Accuracy:     {test_acc*100:6.2f}% (Overfitting Gap: {overfitting_gap*100:5.2f}%)")
        print(f"  - Balanced Accuracy: {bal_acc*100:6.2f}%")
        print(f"  - Macro F1-Score:    {macro_f1*100:6.2f}%")
        print(f"  - Top-3 Accuracy:    {top3_acc*100:6.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_name = name
            best_model_kind = kind
            best_model_obj = clf
            best_model_test_prob = y_test_prob

    # Light “session CV”: repeat the same group-disjoint split logic a few more times
    # for the best model only, to detect sensitivity to which sessions land in train/test.
    cv_seeds = [123, 124]
    cv_metrics = []
    for seed in cv_seeds:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(gss.split(X, y, groups=groups))

        # Require all classes to be present; otherwise skip this fold.
        if len(np.unique(y[tr])) != n_classes or len(np.unique(y[te])) != n_classes:
            continue

        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        local_scaler = StandardScaler()
        X_tr_scaled = local_scaler.fit_transform(X_tr)
        X_te_scaled = local_scaler.transform(X_te)

        # Recreate the classifier with same hyperparameters.
        _, clf_proto = models[best_model_name]
        clf = clf_proto.__class__(**clf_proto.get_params())

        if best_model_kind == "tree":
            X_tr_use, X_te_use = X_tr, X_te
        else:
            X_tr_use, X_te_use = X_tr_scaled, X_te_scaled

        clf.fit(X_tr_use, y_tr)
        y_te_pred = clf.predict(X_te_use)
        y_te_prob = clf.predict_proba(X_te_use)

        test_acc = accuracy_score(y_te, y_te_pred)
        bal_acc = balanced_accuracy_score(y_te, y_te_pred)
        macro_f1 = f1_score(y_te, y_te_pred, average='macro')
        top3_acc = top_k_accuracy_score(y_te, y_te_prob, k=3, labels=np.arange(27))
        cv_metrics.append((test_acc, bal_acc, macro_f1, top3_acc))

    if cv_metrics:
        cv_metrics_np = np.array(cv_metrics, dtype=np.float32)
        print(f"\n[CV on best model: {best_model_name}]")
        print(f"  - test_acc  mean={cv_metrics_np[:,0].mean()*100:.2f}% std={cv_metrics_np[:,0].std()*100:.2f}%")
        print(f"  - bal_acc   mean={cv_metrics_np[:,1].mean()*100:.2f}% std={cv_metrics_np[:,1].std()*100:.2f}%")
        print(f"  - macro_f1  mean={cv_metrics_np[:,2].mean()*100:.2f}% std={cv_metrics_np[:,2].std()*100:.2f}%")
        print(f"  - top3_acc  mean={cv_metrics_np[:,3].mean()*100:.2f}% std={cv_metrics_np[:,3].std()*100:.2f}%")
    else:
        print(f"\n[CV on best model: {best_model_name}] skipped (could not find valid class-coverage splits).")

    # Generate Confusion Matrix Plot for Best Model
    best_res = results[best_model_name]
    y_pred_best = best_res["y_test_pred"]

    cm = confusion_matrix(y_test, y_pred_best, labels=np.arange(27))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)

    plt.figure(figsize=(13, 11))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Phase 3 Confusion Matrix — {best_model_name}\n(Test Accuracy: {best_res["test_acc"]*100:.2f}%, Macro F1: {best_res["macro_f1"]*100:.2f}%)', fontsize=12, fontweight='bold')
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(27)
    plt.xticks(tick_marks, CANONICAL_CLASSES, rotation=45, fontsize=9)
    plt.yticks(tick_marks, CANONICAL_CLASSES, fontsize=9)
    plt.xlabel('Predicted Key', fontsize=11, fontweight='bold')
    plt.ylabel('True Key', fontsize=11, fontweight='bold')
    plt.tight_layout()

    cm_path = os.path.join(BASE_DIR, "evaluation", "phase3_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"\nSaved Confusion Matrix Plot to: {cm_path}")

    # Per-key precision/recall breakdown
    report = classification_report(y_test, y_pred_best, target_names=CANONICAL_CLASSES, output_dict=True)
    print("\n=======================================================")
    print(f"PER-KEY PRECISION / RECALL BREAKDOWN ({best_model_name}):")
    print("=======================================================")
    for c in CANONICAL_CLASSES:
        if c in report:
            p = report[c]["precision"] * 100
            r = report[c]["recall"] * 100
            f = report[c]["f1-score"] * 100
            supp = int(report[c]["support"])
            print(f"  - Key {c:5s}: Precision={p:5.1f}%, Recall={r:5.1f}%, F1={f:5.1f}% (N={supp})")

    return results, best_model_name


if __name__ == "__main__":
    X, y, groups = load_dataset_features(max_samples_per_class=100)
    results, best_name = benchmark_models(X, y, groups)
