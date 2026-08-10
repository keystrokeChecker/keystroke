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

## Phase 1 production path

The API uses `onset_detector.py` for candidate clicks, then classifies each
candidate with a lightweight scikit-learn Random Forest over MFCC features.
The classes are `letter`, `space`, and `noise`. YAMNet files are retained for
offline experiments but are not imported or called by the server.

Train the classifier after changing labeled recordings:

```bash
python train_keystroke_type_classifier.py
```

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

Response:

```json
{
  "count": 10,
  "counts": [5, 5],
  "formatted": "5|5"
}
```

## Notes

- The backend returns one total count; it does not attempt pause-based word splitting.
- Your phone must be on the same local network as your development machine.
