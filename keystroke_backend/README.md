# Keystroke Backend

A local FastAPI backend for analyzing WAV recordings of physical keyboard typing.

## Setup

1. Open a terminal in `keystroke_backend`.
2. Create and activate a Python virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## YAMNet Pipeline

The backend now uses YAMNet embeddings plus a lightweight classifier for keystroke onset detection.

Train and test in one command:

```bash
python -m src.run_yamnet_pipeline
```

You can also target specific sessions:

```bash
python -m src.run_yamnet_pipeline --names session1 session2 session3
```

That creates:

- `data/yamnet_dataset/X.npy`
- `data/yamnet_dataset/y.npy`
- `models/yamnet_keystroke_classifier.joblib`

## Run the server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Test the server

From your machine:

```bash
curl http://127.0.0.1:8000/health
```

## API

### `GET /health`

Returns a basic connectivity response:

```json
{"status": "ok"}
```

### `POST /analyze`

Upload a WAV file using multipart form data:

- `file`: the WAV audio file
- `method`: `yamnet` only
- `threshold`: word-gap threshold in seconds
- `delta`: detector sensitivity adjustment

Response:

```json
{
  "counts": [3, 7],
  "formatted": "3|7"
}
```

## Notes

- The backend now uses YAMNet end-to-end.
- The YAMNet model is loaded from TensorFlow Hub, so the first run needs TensorFlow and TensorFlow Hub installed.
- Your phone must be on the same local network as your development machine.
