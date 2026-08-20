# Production Keystroke Sound Identification Model Report (v2)

## Executive Summary
This report documents the final production model and issue-fix resolution for `keystroke_backend_v2`. The system is built using **ONLY verified public datasets** (Kaggle Keyboard Sound, Multi-Pressure Keystroke Dataset, and SKAID). All local HP laptop recordings (`session1`, `session4`, `new1`, `new2`, `session_calibrated_test`, `gain_check`, `gain_test`, `session3`) are **strictly excluded** across training, validation, testing, feature extraction, calibration, and hyperparameter selection.

---

## 1. Public Dataset Audit & Event Composition

| Dataset Name | Source Archive | Audio Container | Event Count | Target Class Coverage | Recording Environment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Kaggle Keyboard Sound** | `archive.zip` | `.wav` | 8,631 | 27 / 27 (A-Z + SPACE) | Clean isolated keystrokes |
| **Multi-Pressure Keystroke** | `dataset.zip` | `.wav` | 162 | 27 / 27 (A-Z + SPACE) | Isolated (High/Medium/Low pressure) |
| **SKAID Dataset** | `Participant Recordings_zip.zip` | `.m4a` (PyAV decoded) | 82,740 | 27 / 27 (A-Z + SPACE) | Continuous typing (M4A audio) |
| **Total Public Corpus** | — | — | **91,533** | **27 / 27** | **Zero Local HP Laptop Data** |

---

## 2. Issue Resolution Summary

### Issue 1 & 2: SKAID Audio/Key Alignment & Low Standalone Accuracy Diagnosis
- **Multi-Method Onset Detection**: Implemented local neighborhood onset alignment searching $\pm 150\text{ ms}$ around CSV timestamps across candidate methods: RMS envelope, spectral flux, high-frequency energy, peak amplitude, and hybrid energy score.
- **Event Quality Breakdown**:
  - **Valid Events**: $1.97\%$
  - **Low Quality Events (SNR < 3dB)**: $2.24\%$
  - **Overlapping Keystrokes (<100ms separation)**: **91.34%**
  - **Unusable Events (Background silence/decay)**: $4.45\%$
- **Root Cause Verdict**: SKAID's low standalone performance ($\sim 9.77\%$) is caused by **rapid continuous typing keystroke overlap (91.34% of events)** and low-SNR laptop/phone microphone reverberation, contrasted with Kaggle's clean isolated keystrokes (**92.11% standalone accuracy**).

### Issue 3: Dataset Domination & Balancing
- Evaluated 5 corpus balancing strategies. Downsampling SKAID to 500 samples per class prevents SKAID from overwhelming Kaggle and Multi-Pressure while maintaining multi-dataset representation.

### Issue 4 & 5: Metric Reconciliation & EXP-17 Audit
- **EXP-08 (69.81%) vs Production (55.56%) Reconciliation**:
  - EXP-08 was evaluated on a 15,506-event corpus (300 SKAID samples/class).
  - Production model was evaluated on a 20,945-event corpus (500 SKAID samples/class).
  - Increasing SKAID continuous typing volume introduced more overlapping keystrokes, adjusting final test accuracy to **55.56%** (Macro F1 = 0.6011).
- **EXP-17 Leakage Audit**:
  - EXP-17 evaluated a tiny 486-event subset and contained duplicate feature clips across splits.
  - **EXP-17 IS OFFICIALLY MARKED INVALID** and is excluded from headline production claims.

### Issue 6 & 7: Feature Ablation & Normalization Validation
- Tested 8 feature sets and 5 normalization strategies.
- **Winning Feature Set**: **Feature Set D** (98 MFCC + spectral + transient + log-mel statistical features).
- **Winning Normalization**: **P2 Peak Normalization** ($x / (\max|x| + \epsilon)$).

---

## 3. Final Production Model Performance

- **Classifier Architecture**: Calibrated **ExtraTrees Classifier** ($200$ estimators, `max_depth=15`)
- **Training Corpus**: 20,945 public events across Kaggle, Multi-Pressure, and balanced SKAID
- **Grouped Split Method**: `GroupShuffleSplit` on `(participant_id, recording_id)` (Zero group leakage)
- **Official Locked Test Metrics**:
  - **Accuracy**: **55.56%** (over **15x higher** than 3.70% random baseline)
  - **Macro F1**: **0.6011**
  - **Balanced Accuracy**: **0.5960**
  - **Top-3 Accuracy**: **63.99%**
  - **Log Loss**: **1.8539**
  - **Expected Calibration Error (ECE)**: **0.2237**

---

## 4. Exported Production Artifacts (`production/`)

1. [model.joblib](file:///c:/Users/Tursi/Desktop/keystroke_v2/keystroke_backend_v2/production/model.joblib): Calibrated ExtraTrees classifier model object.
2. [feature_scaler.joblib](file:///c:/Users/Tursi/Desktop/keystroke_v2/keystroke_backend_v2/production/feature_scaler.joblib): Trained StandardScaler object.
3. [classes.json](file:///c:/Users/Tursi/Desktop/keystroke_v2/keystroke_backend_v2/production/classes.json): 27 canonical target class index mapping.
4. [model_metadata.json](file:///c:/Users/Tursi/Desktop/keystroke_v2/keystroke_backend_v2/production/model_metadata.json): Complete model training configuration and validation metrics.
5. [preprocessing_config.json](file:///c:/Users/Tursi/Desktop/keystroke_v2/keystroke_backend_v2/production/preprocessing_config.json): Audio sampling rate, window duration, and normalization configuration.
6. [thresholds.json](file:///c:/Users/Tursi/Desktop/keystroke_v2/keystroke_backend_v2/production/thresholds.json): Confidence thresholds (`confident_threshold=0.60`, `low_confidence_threshold=0.35`).
7. [final_test_manifest.json](file:///c:/Users/Tursi/Desktop/keystroke_v2/keystroke_backend_v2/production/final_test_manifest.json): Frozen test group manifest IDs.
