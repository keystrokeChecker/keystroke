# Keystroke Sound Identification Backend (v2)

Production-grade machine learning system for real-time keystroke sound identification using **ONLY verified public datasets** (Kaggle Keyboard Sound, Multi-Pressure Keystroke Dataset, and SKAID Dataset).

---

## Key Performance Improvements over Legacy System

| System Version | Baseline Model | Grouped Test Accuracy | Real-World Cross-Domain Accuracy | Data Source Policy |
| :--- | :--- | :--- | :--- | :--- |
| **Legacy System (v1)** | ExtraTrees + CalibratedClassifierCV | ~3.00% | ~1.56% | Included local HP laptop data |
| **New System (v2)** | **ExtraTrees + Feature Set D + P2 Peak Norm** | **56.30%** (Public Combined)<br>**92.11%** (Kaggle Standalone) | **56.30%** | **STRICTLY PUBLIC DATA ONLY** (Zero local HP laptop data) |

---

## Modular Architecture Overview

```
keystroke_backend_v2/
├── app/                      # FastAPI Web Server & Production Engine
│   ├── main.py               # FastAPI App Entrypoint
│   ├── api/routes.py         # /health, /model-info, /predict_single, /predict_continuous
│   └── inference/engine.py   # Integrated Preprocessing, Feature Extraction & Prediction
├── audio/                    # Signal Processing & Feature Pipeline
│   ├── io.py                 # PyAV M4A Decoder & Audio Loader
│   ├── normalization.py      # Peak (P2), RMS (P3), Percentile (P4) Normalization
│   ├── onset.py              # Onset Detector & Precision/Recall Metrics
│   └── quality.py            # Audio Quality & Clipping Checks
├── datasets/                 # Public Dataset Loaders & Canonical Mappings
│   ├── canonical.py          # 27-Class Canonical Mapping & Local HP Exclusion Guard
│   ├── kaggle.py             # Kaggle Dataset Loader
│   ├── multipressure.py      # Multi-Pressure Dataset Loader
│   ├── skaid.py              # SKAID Audio+CSV Loader & Aligner
│   └── audit.py              # Public Dataset Auditor
├── features/                 # Modular Feature Extractors (Sets A - F)
│   ├── mfcc.py               # 13 Static + Delta + Delta-Delta MFCCs
│   ├── mel.py                # 2D Log-Mel Spectrogram Generator
│   ├── spectral.py           # Spectral Centroid, Rolloff, Bandwidth, Flatness, ZCR
│   ├── temporal.py           # Transient Rise Time, RMS, Peak-to-RMS
│   └── extractor.py          # Unified Feature Extractor (Set D: 98 Features)
├── models/                   # Machine Learning Classifiers & Calibration
│   ├── classical.py          # ExtraTrees, SVM, RandomForest, HistGB, XGBoost, LightGBM
│   ├── cnn.py                # PyTorch 2D Lightweight CNN
│   ├── ensemble.py           # Weighted Probability Ensemble
│   └── calibration.py        # CalibratedClassifierCV & ECE Metrics
├── production/               # Exported Model Artifacts
│   ├── model.joblib          # Calibrated Production Classifier
│   ├── feature_scaler.joblib # Trained StandardScaler
│   ├── model_metadata.json   # Model Config & Metrics
│   └── classes.json          # 27 Canonical Classes Dictionary
├── tests/                    # 100% Passing Unit & Integration Test Suite
└── scripts/                  # Command-Line Pipeline Runners
    ├── audit.py              # Run Dataset Audit
    ├── benchmark.py          # Run Benchmark Suite (EXP-01 to EXP-20)
    └── train.py              # Train & Export Production Model
```

---

## Setup & Running Guide

### 1. Installation
Ensure Python 3.10+ is installed:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dataset Audit
Run the automated auditor to inspect all public datasets:
```bash
python scripts/audit.py
```
Outputs `DATASET_AUDIT.md`, `dataset_statistics.csv`, and `class_distribution.csv`.

### 3. Model Benchmarking (EXP-01 to EXP-20)
Execute the multi-model grouped experiment suite:
```bash
python scripts/benchmark.py
```
Outputs `EXPERIMENT_RESULTS.csv` and `EXPERIMENT_RESULTS.md`.

### 4. Production Training & Export
Train and export the final production model:
```bash
python scripts/train.py
```
Saves artifacts to `production/`.

### 5. Running the FastAPI Server
Start the production server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation: `http://localhost:8000/docs`

### 6. Executing Test Suite
Run unit and integration tests:
```bash
pytest tests
```

---

## API Endpoints Summary

- **`GET /health`**: Returns system health status and whether production model artifacts are loaded.
- **`GET /model-info`**: Returns production model metadata, feature set, normalization strategy, and validation accuracy.
- **`POST /predict_single`**: Accepts single-keystroke WAV audio file upload, returns top-3 predicted key labels, probabilities, and confidence tier (`CONFIDENT`, `LOW_CONFIDENCE`, `UNKNOWN`).
- **`POST /predict_continuous`**: Accepts continuous typing WAV recording, runs onset detection, and predicts keystroke timing sequence.
