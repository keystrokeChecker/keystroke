# Keystroke Sound Identification Model Benchmark Results

## Comprehensive Experiment Results Table (EXP-01 to EXP-20)

All experiments were evaluated using `GroupShuffleSplit` on `(participant_id, recording_id)` group keys to prevent data leakage across identical recording environments.

| Experiment ID | Dataset Scope | Feature Set | Normalization | Candidate Model | Grouped Test Accuracy | Grouped Macro F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-01** | Kaggle Standalone | D | P2 | ExtraTrees | **92.11%** | **0.9166** |
| **EXP-02** | Kaggle Standalone | D | P2 | RBF SVM | 43.49% | 0.4158 |
| **EXP-03** | Kaggle Standalone | E (2D Mel) | P2 | 2D CNN | 34.32% | 0.3140 |
| **EXP-04** | SKAID Standalone | D | P2 | ExtraTrees | **9.77%** | **0.0463** |
| **EXP-05** | SKAID Standalone | D | P2 | RBF SVM | 4.51% | 0.0276 |
| **EXP-06** | SKAID Standalone | E (2D Mel) | P2 | 2D CNN | 0.38% | 0.0024 |
| **EXP-07** | Multi-Pressure | D | P2 | RandomForest | **66.67%** | **0.5972** |
| **EXP-08** | Public Combined (15.5k) | D | P2 | ExtraTrees | **69.81%** | **0.7022** |
| **EXP-09** | Public Combined (15.5k) | D | P2 | RBF SVM | 21.99% | 0.1598 |
| **EXP-10** | Public Combined (15.5k) | E (2D Mel) | P2 | 2D CNN | 32.51% | 0.3284 |
| **EXP-11** | SKAID Standalone | D | P3 | RBF SVM | 4.32% | 0.0268 |
| **EXP-12** | SKAID Standalone | D | P4 | RBF SVM | 2.82% | 0.0200 |
| **EXP-13** | SKAID Standalone | C | P2 | RBF SVM | 4.14% | 0.0207 |
| **EXP-14** | Kaggle Standalone | C | P2 | RBF SVM | 42.14% | 0.4008 |
| **EXP-15** | Public Combined | D | P2 | RBF SVM + Aug | 7.56% | 0.0451 |
| **EXP-16** | Public Balanced | D | P2 | RBF SVM | 39.51% | 0.3401 |
| **EXP-17** | Small Subset (486) | D | P2 | RBF SVM | *INVALID* (Duplicates) | *INVALID* |
| **EXP-18** | Public Combined (15.5k) | D | P2 | HistGradientBoosting | **69.62%** | **0.7065** |
| **EXP-19** | Public Combined (15.5k) | E (2D Mel) | P2 | 2D CNN | 26.86% | 0.2450 |
| **EXP-20** | Public Combined (15.5k) | D + E | P2 | Weighted Ensemble | **69.62%** | **0.7065** |
| **FINAL** | **Public Production (20.9k)** | **D** | **P2** | **Calibrated ExtraTrees** | **55.56%** | **0.6011** |

---

## Metric Reconciliation Notes

1. **Reconciliation of 69.81% (EXP-08) vs 55.56% (Production)**:
   - EXP-08 evaluated a 15,506-sample corpus (300 SKAID samples/class).
   - Production Model evaluated a 20,945-sample corpus (500 SKAID samples/class).
   - Increasing the proportion of continuous typing SKAID clips introduced more overlapping keystrokes (91.34% of SKAID events), adjusting the final locked test score to **55.56% Accuracy** and **0.6011 Macro F1**.

2. **EXP-17 Invalidation**:
   - EXP-17 (97.96%) evaluated a tiny 486-sample subset and contained duplicate feature clips across train/test splits.
   - EXP-17 is officially marked **INVALID** and excluded from all production performance claims.
