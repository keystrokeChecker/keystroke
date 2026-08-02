# Keystroke Backend

Local FastAPI service for analyzing WAV recordings of physical-keyboard typing.

## Environment

Create an isolated Python environment, then install one dependency set:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

# API runtime
pip install -r requirements.txt

# Runtime plus automated tests
pip install -r requirements-dev.txt

# Runtime plus local keyboard/audio data collection tools
pip install -r requirements-training.txt
```

The pinned runtime versions are the versions used by the checked-in model
artifacts and the automated test suite.

## Run

For access from an Android device on the same trusted Wi-Fi network:

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

Running `python app.py` is loopback-only by default. It accepts these optional
environment variables:

- `KEYSTROKE_HOST` — bind address; default `127.0.0.1`.
- `KEYSTROKE_PORT` — port; default `8000`.
- `KEYSTROKE_RELOAD=1` — enable development reload; disabled by default.

## API

### `GET /health`

Lightweight process health check:

```json
{"status": "ok"}
```

### `GET /ready`

Validates the two local joblib artifacts. It returns HTTP 503 with
`status=degraded` if either artifact is absent or invalid. The underlying
TensorFlow Hub YAMNet model remains lazily loaded on the first `ml` or `yamnet`
analysis.

### `POST /analyze`

Multipart fields:

- `file` — required `.wav` or `.wave` upload containing a real WAV stream.
- `method` — `rule`, `ml`, or `yamnet`; default `yamnet`.
- `threshold` — YAMNet classifier threshold from 0 to 1.
- `delta` — onset sensitivity greater than 0 and at most 1.
- `gap_threshold` — optional word gap greater than 0 and at most 5 seconds.
- `merge_gap_seconds` — duplicate-onset gap greater than 0 and at most 1 second.

The service bounds the whole multipart request, individual upload size,
recording duration, inference concurrency, and queue wait. Temporary recordings
are removed after success, validation errors, prediction failures, timeouts, and
request cancellation.

Example:

```powershell
curl.exe -X POST http://127.0.0.1:8000/analyze `
  -F "file=@data/session1.wav" `
  -F "method=rule"
```

## Tests

The default suite does not perform real TensorFlow downloads or audio-model
inference:

```powershell
python -m pytest -q
```

Model loading, predictor routing, API validation, request limits, cancellation
cleanup, segmentation, and readiness behavior are covered with deterministic
fixtures.

## Evaluation

Release evaluation is manifest-driven and invokes the same trace-producing
rule, ML, and YAMNet functions used by the API. The manifest must assign whole
recordings to `train`, `validation`, `test`, or `diagnostic` and keep each
`setup_id` in only one leakage-controlled split.

```powershell
python evaluate_methods.py `
  --manifest data/evaluation_manifest.json `
  --split diagnostic `
  --methods rule `
  --output-json results/diagnostic.json
```

Fresh private datasets use the consent and split procedure in
`../docs/DATASET_COLLECTION.md`. Capture metadata includes every
leakage-relevant setup field, the intended split, and fixture type. Validate a
populated private manifest before training or evaluation:

```powershell
python validate_dataset.py data/evaluation_manifest.json `
  --output results/dataset_quality.json
```

The validator checks PCM format, duration, clipping, near-silence, synchronized
truth, structured metadata, and minimum train/validation/test negative-fixture
coverage. Existing recordings remain diagnostic-only.

The candidate-training, deterministic validation-selection, artifact
promotion, and untouched-test workflow is in `../docs/MODEL_RELEASE.md`.

Diagnostic and legacy-data runs are always marked `valid_for_release: false`.
`select_production_method.py` turns a multi-method validation result into one
artifact-bound configuration lock. Evaluate the untouched test split with
exactly that selected method and pass the lock as `--locked-config-sha256`.
Test evidence is release-valid only when the lock
matches, fixed 80 ms metrics are unchanged, required negative fixtures are
present, and learned artifacts carry complete, non-contaminating provenance.
The full contract is in `../docs/EVALUATION_PROTOCOL.md`.

## Model status

The checked-in artifacts are preserved for reproducibility, but no prediction
method is approved as the final version 1 pipeline yet. See `models/README.md`
and the root `PROJECT_STATUS.md`.
