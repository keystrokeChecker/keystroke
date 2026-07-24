# Keystroke Analyzer — Pipeline Evaluation & Handoff Notes

## 1. Executive Summary

This document summarizes the current status, live HTTP verification, per-method parameter configuration, evaluation benchmarks, performance limitations, and unresolved open questions for the Keystroke Analyzer ML pipeline.

---

## 2. End-to-End API Verification (`POST /analyze`)

The FastAPI backend (`app.py`) has been fully verified via live HTTP POST requests on port `8008`.

- **Health check**: `GET /health` → `200 OK` `{"status": "ok"}`
- **Prediction methods tested (per-method defaults)**:
  - `method=ml`: `200 OK` → Uses `GAP_THRESHOLD_ML = 0.75s`. Returns count predictions from RandomForest regressor on YAMNet embeddings (`count_predictor_new.joblib`). Sample response: `{"counts": [5], "formatted": "5"}`
  - `method=rule`: `200 OK` → Uses `GAP_THRESHOLD_RULE = 0.40s`. Returns pure signal processing (DSP) onset counts per segment. Sample response: `{"counts": [1, 1, 11, 14, 2, 4, 2], "formatted": "1|1|11|14|2|4|2"}`
  - `method=yamnet`: `200 OK` → Uses `GAP_THRESHOLD_ML = 0.75s`. Returns onset counts filtered by YAMNet confidence threshold (`keystroke_classifier.joblib`). Sample response: `{"counts": [8, 9, 2], "formatted": "8|9|2"}`
- **Override behavior**: Callers can still override `gap_threshold`, `delta`, `threshold`, and `merge_gap_seconds` per request via form parameters.

---

## 3. Evaluation Benchmarks: Ground-Truth vs. Real Segmenter Output

Evaluation was performed across all 6 valid sessions (`gain_check`, `gain_test`, `new1`, `new2`, `session1`, `session4`).

> [!IMPORTANT]
> **Key Insight on Ground-Truth vs. Real-Segmenter Boundaries:**
> Previous evaluations computed MAE by forcing ground-truth word boundaries onto detected onsets (optimistic MAE ~0.300). In production, the backend receives raw audio without ground-truth boundary timestamps and must segment onsets using `segment_into_words(gap_threshold)`.

> [!WARNING]
> **Rule-Path Regression & Resolution:**
> Setting a single global default `gap_threshold = 0.75s` was ML-only and severely broke the rule-based path (Rule MAE degraded from `4.250` to `19.375` and Rule Acc dropped to `0.000` because separate words merged into a single segment of ~35 raw clicks).
> This regression has been resolved by implementing separate per-method defaults: `GAP_THRESHOLD_RULE = 0.40s` for `method=rule` and `GAP_THRESHOLD_ML = 0.75s` for `method=ml` and `method=yamnet`.

### Benchmark Results Table

| Evaluation Context | Method | Gap Threshold | MAE | Accuracy | Word Count Delta (WcD) | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Optimistic (GT Boundaries)** | Rule | N/A | 0.800 | 0.400 | N/A | Forced GT boundaries (unrealistic for production) |
| **Optimistic (GT Boundaries)** | ML | N/A | **0.300** | **0.750** | N/A | Forced GT boundaries (unrealistic for production) |
| **Real Segmenter (Un-tuned Baseline)** | Rule | 0.40s | 4.250 | 0.125 | +5.17 | Baseline over-splitting (+5.17 extra words/session) |
| **Real Segmenter (Un-tuned Baseline)** | ML | 0.40s | 1.167 | 0.208 | +5.17 | Baseline over-splitting (+5.17 extra words/session) |
| **Real Segmenter (Tuned Per-Method)** | Rule | **0.40s** | **4.250** | **0.125** | **+5.17** | Restored rule-path baseline (prevents word merging) |
| **Real Segmenter (Tuned Per-Method)** | ML | **0.75s** | **0.792** | **0.250** | **+0.17** | **Recommended ML default** (WcD ~ 0, MAE 0.792) |

*Legend: WcD = Word Count Delta (`len(predicted_words) - len(true_words)`). Positive indicates over-splitting.*

---

## 4. Performance Analysis & Next Levers

> [!CAUTION]
> **Segmenter Tuning Limitations & Data Volume Bottleneck:**
> While tuning `GAP_THRESHOLD_ML = 0.75s` narrowed the real-segmenter ML MAE gap from `1.167` down to `0.792`, real production accuracy (MAE `0.792`, Acc `0.250`) remains significantly below ground-truth-boundary accuracy (MAE `0.300`, Acc `0.750`).
> 
> **Segmenter parameter tuning has reached its limit and cannot close this remaining gap.**
> 
> The core bottleneck is a **data volume problem**: the ML model was trained on only 20 word-level samples across 6 usable sessions. Consequently, predictions cluster narrowly around 4–5 keystrokes per word. Further hyperparameter tuning will not solve this. **The actual next lever is collecting and re-recording more valid sessions to expand training data volume.**

---

## 5. Parameter Configuration Summary

- **`GAP_THRESHOLD_ML`**: **`0.75s`** (tuned for `method=ml` and `method=yamnet`).
- **`GAP_THRESHOLD_RULE`**: **`0.40s`** (retained for `method=rule`).
- **`GAP_THRESHOLD`**: **`0.75s`** (alias in `src/yamnet_config.py` for backward compatibility).
- **`SENSITIVITY_DELTA`**: **`0.07`**.
- **`MERGE_GAP_SECONDS`**: **`0.06s`**.

---

## 6. Explicit Open Questions for Team Review

1. **Canonical Production Model Selection**
   - Currently, `keystroke_classifier.joblib` (1028 features: YAMNet embedding + 4 local DSP features) was inferred to be the canonical candidate classifier for `method=yamnet`.
   - `count_predictor_new.joblib` was created specifically for `method=ml` (RandomForest regressor on mean-pooled word embeddings).
   - *Question for Team*: Should `count_predictor_new.joblib` be merged/promoted as the single default model, or should `keystroke_classifier.joblib` be retained separately for single-onset filtering?

2. **Data Quality & Unusable Audio Sessions (6 / 18 Usable)**
   - Only 6 of 18 recorded sessions in `data/` are usable (`gain_check`, `gain_test`, `new1`, `new2`, `session1`, `session4`).
   - The remaining 12 sessions failed due to empty keylog CSVs (8 sessions), near-zero audio amplitude / 0 onsets (3 sessions), or incomplete metadata (1 session).
   - *Question for Team*: Re-recording the 12 invalid sessions requires physical access to a mic + keyboard setup. Who will take ownership of re-recording these sessions to expand training data beyond 20 word samples?

---

## 7. Ground Rules & Artifact Integrity

- `keystroke_classifier.joblib` was preserved intact and NOT overwritten.
- `onset_detector.py` and `segmenter.py` core logic remained unmodified.
- Feature extraction between `train_model.py` and `src/predictor.py` was verified bit-identical.
