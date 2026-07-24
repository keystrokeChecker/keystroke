# Keystroke Analyzer

A cross-platform system that listens to the sound of your physical keyboard and counts keystrokes per word — in real time. It pairs a **Flutter mobile app** with a **FastAPI Python backend** to record, analyze, and report keystroke counts from WAV audio.

---

## How It Works

The system follows a six-stage audio ML pipeline:

```
Record → Detect → Segment → Evaluate → Train → Predict
```

1. **Record** — Capture keyboard audio and synchronize it with a ground-truth keylog
2. **Detect** — Find keystroke click timestamps using bandpass filtering and adaptive onset detection
3. **Segment** — Group detected clicks into words by looking for inter-word pause gaps
4. **Evaluate** — Compare predictions against the ground truth and tune parameters
5. **Train** — Build a RandomForest ML model on MFCC audio features
6. **Predict** — Serve predictions via REST API (rule-based or ML-based)

The Flutter app records audio through the phone microphone, sends it to the backend over your local Wi-Fi network, and displays the result as a pipe-delimited count (e.g., `3|7` for the phrase "the project").

---

## Project Structure

```
keystroke/
├── keystroke_backend/          # FastAPI Python backend
│   ├── app.py                  # REST API server (/health, /analyze)
│   ├── predict.py              # Rule-based and ML-based prediction
│   ├── onset_detector.py       # Audio onset detection (keystroke click finder)
│   ├── segmenter.py            # Groups clicks into words by pause gaps
│   ├── train_model.py          # Train a RandomForest model on MFCC features
│   ├── tune_and_evaluate.py    # Evaluate and tune pipeline parameters
│   ├── record_dataset.py       # Record audio + keylog training data
│   ├── requirements.txt        # Python dependencies
│   ├── models/                 # Saved trained models (count_predictor.pkl)
│   └── data/                   # Audio + keylog training sessions
└── keystroke_app/              # Flutter mobile app
    ├── lib/
    │   └── main.dart           # App UI and recording/upload logic
    └── pubspec.yaml            # Flutter dependencies
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Flutter SDK 3.12+
- A physical keyboard and a phone on the same local Wi-Fi network

---

### Backend Setup

```bash
cd keystroke_backend

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Start the server:**

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Verify it's running:**

```bash
# From your machine
curl http://127.0.0.1:8000/health

# From your phone on the same network (replace with your machine's IP)
curl http://192.168.X.Y:8000/health
```

To find your local IP on Linux/macOS: `hostname -I` or `ip addr show`. On Windows: `ipconfig`.

---

### Flutter App Setup

```bash
cd keystroke_app

# Install dependencies
flutter pub get

# Run on a connected device
flutter run
```

Once the app launches, open **Settings** (top-right gear icon) and set the **Backend server URL** to your machine's local IP address:

```
http://192.168.X.Y:8000
```

---

## API Reference

### `GET /health`

Health check endpoint.

**Response:**
```json
{ "status": "ok" }
```

---

### `POST /analyze`

Accepts a WAV audio recording and returns keystroke counts per word.

**Form fields:**

| Field       | Type   | Default | Description                                               |
|-------------|--------|---------|-----------------------------------------------------------|
| `file`      | file   | —       | WAV audio file (required)                                 |
| `method`    | string | `rule`  | Prediction method: `rule` or `ml`                         |
| `threshold` | float  | `0.4`   | Word boundary gap in seconds (higher = fewer word splits) |
| `delta`     | float  | `0.07`  | Onset sensitivity (lower = more keystrokes detected)      |

**Example request:**
```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -F "file=@data/session1.wav" \
  -F "method=rule" \
  -F "threshold=0.4" \
  -F "delta=0.07"
```

**Response:**
```json
{
  "counts": [3, 7],
  "formatted": "3|7"
}
```

---

## Training Your Own Model

The ML mode uses a RandomForest model trained on MFCC audio features. To train it on your own data:

### Step 1 — Record a training session

Type a phrase at your keyboard while this script records both audio and keystrokes:

```bash
python record_dataset.py --name session1 --duration 15
```

This saves:
- `data/session1.wav` — raw audio
- `data/session1_log.csv` — ground-truth keypress timestamps

> **Ethics note:** Only use this to record your own typing with your own knowledge and consent.

---

### Step 2 — Evaluate and tune parameters

Test how well the rule-based pipeline performs against your ground truth:

```bash
python tune_and_evaluate.py --name session1
```

The script reports per-word accuracy and suggests adjustments if below 85%:

```
Session: session1
True counts:      [3, 7]  ->  3|7
Predicted counts: [3, 7]  ->  3|7
Word-level accuracy: 2/2 = 100.0%
```

Tuning tips:
- **`--delta`** — Lower if keystrokes are being missed; raise if you get false positives
- **`--threshold`** — Lower if words are being merged; raise if single words are being split

---

### Step 3 — Train the ML model

Record several sessions first, then train:

```bash
python train_model.py --names session1 session2 session3
```

The trained model is saved to `models/count_predictor.pkl`. Training reports:
- Mean absolute error (MAE) on the test split
- Exact-match accuracy

---

### Step 4 — Run standalone predictions

```bash
# Rule-based
python predict.py data/new_recording.wav --method rule

# ML-based (requires trained model)
python predict.py data/new_recording.wav --method ml
```

---

## Flutter App Features

- **Record & analyze** — Tap to start/stop recording; audio is automatically uploaded to the backend
- **Live status messages** — Feedback at every step (recording, uploading, result received)
- **History** — All past recordings are saved locally with timestamps and playback
- **Audio playback** — Replay any previous recording from the history list
- **Swipe to delete** — Remove individual history entries (audio file is also deleted)
- **Persistent state** — History survives app restarts via `SharedPreferences`
- **Settings panel** — Configure backend URL and prediction method (rule or ML)

---

## Tech Stack

### Backend

| Library        | Version  | Purpose                                   |
|----------------|----------|-------------------------------------------|
| FastAPI        | ≥ 0.109  | REST API framework                        |
| Uvicorn        | ≥ 0.23   | ASGI server                               |
| Librosa        | ≥ 0.10   | Audio analysis, onset detection, MFCCs   |
| NumPy          | ≥ 1.26   | Numerical computing                       |
| scikit-learn   | ≥ 1.4    | RandomForest model                        |
| SoundFile      | ≥ 0.12   | WAV file I/O                              |
| PyAudio        | —        | Audio recording (training data only)      |
| pynput         | —        | Keyboard logging (training data only)     |

### Frontend

| Package            | Version | Purpose                          |
|--------------------|---------|----------------------------------|
| Flutter SDK        | ≥ 3.12  | Cross-platform mobile framework  |
| http               | ≥ 1.6   | HTTP client for backend calls    |
| record             | ≥ 7.1   | Microphone audio recording       |
| path_provider      | ≥ 2.1   | App-scoped file storage          |
| shared_preferences | ≥ 2.2   | Persistent history storage       |
| just_audio         | ≥ 0.9   | Audio playback                   |

---

## Algorithm Overview

The onset detector isolates keyboard click energy through a signal processing chain:

1. **Normalize** — Convert to mono and normalize amplitude
2. **Bandpass filter** — Isolate 800–8000 Hz to capture click transients
3. **Dual RMS envelopes** — Compute short-window (impulsive) and long-window (baseline) energy
4. **Transient score** — Ratio of short RMS to long RMS highlights sudden energy spikes
5. **Attack score** — First derivative of the smoothed score; real clicks have a sharp rise
6. **Adaptive threshold** — Dynamic threshold based on noise floor and dynamic range
7. **Peak picking** — `scipy.find_peaks` with a minimum 70 ms gap between detections to merge duplicates

The segmenter then groups the detected clicks into words by splitting on gaps larger than `threshold` seconds (default: 0.4s), corresponding to natural inter-word pauses.

---

## License

This project is provided for educational and personal use. Do not use the keystroke recording functionality to capture input from others without their explicit consent.
