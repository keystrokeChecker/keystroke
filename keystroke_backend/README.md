# Keystroke Backend

A local FastAPI backend for analyzing WAV recordings of physical keyboard typing.

## Setup

1. Open a terminal in `development/keystroke_backend`.
2. Create and activate a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
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

From your phone on the same Wi-Fi network, replace the host with your machine IP:

```bash
curl http://192.168.X.Y:8000/health
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
- `method`: `rule` or `ml` (default: `rule`)
- `threshold`: optional word gap threshold in seconds
- `delta`: optional onset sensitivity

Response:

```json
{
  "counts": [3, 7],
  "formatted": "3|7"
}
```

## Notes

- If you want to use `ml` mode, make sure `models/count_predictor.pkl` exists.
- Your phone must be on the same local network as your development machine.
- To find your local IP on Ubuntu, use `hostname -I` or `ip addr show`.
