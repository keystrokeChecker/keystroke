from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from segmenter import format_output
from src.predictor import (
    get_model_status,
    predict_keystroke_counts,
    predict_keystroke_counts_ml,
    predict_keystroke_counts_rule,
)
from src.yamnet_config import (
    CLASSIFIER_THRESHOLD,
    MERGE_GAP_SECONDS,
    SENSITIVITY_DELTA,
)

logger = logging.getLogger(__name__)

VALID_METHODS = {"yamnet", "rule", "ml"}
UPLOAD_CHUNK_BYTES = 1024 * 1024
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + (2 * 1024 * 1024)
MAX_AUDIO_DURATION_SECONDS = 30.0
MAX_CONCURRENT_ANALYSES = 2
ANALYSIS_QUEUE_TIMEOUT_SECONDS = 10.0
TEMP_FILE_PREFIX = "keystroke-upload-"
STALE_TEMP_FILE_AGE_SECONDS = 24 * 60 * 60


class ClientInputError(ValueError):
    """Base class for errors caused by an invalid client upload or parameter."""


class InvalidUploadError(ClientInputError):
    """Raised when multipart file metadata or bytes are invalid."""


class InvalidAudioError(ClientInputError):
    """Raised when uploaded bytes are not a supported WAV recording."""


class UploadTooLargeError(InvalidUploadError):
    """Raised when an uploaded recording exceeds the configured byte limit."""


class AnalysisBusyError(RuntimeError):
    """Raised when all inference slots remain busy beyond the queue timeout."""


class RequestBodyTooLargeError(OSError):
    """Raised by middleware before multipart parsing consumes an oversized body."""


class RequestBodyLimitMiddleware:
    """Bound `/analyze` request bytes before Starlette parses multipart data."""

    def __init__(self, application, max_bytes: int) -> None:
        self.application = application
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope.get("path") != "/analyze":
            await self.application(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError:
                declared_size = None
            if declared_size is not None and declared_size > self.max_bytes:
                await self._send_too_large(scope, receive, send)
                return

        received_bytes = 0
        response_started = False

        async def limited_receive():
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_bytes:
                    raise RequestBodyTooLargeError
            return message

        async def guarded_send(message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.application(scope, limited_receive, guarded_send)
        except RequestBodyTooLargeError:
            if response_started:
                raise
            await self._send_too_large(scope, receive, send)

    async def _send_too_large(self, scope, receive, send) -> None:
        response = JSONResponse(
            status_code=413,
            content={
                "detail": (
                    "Request body exceeds the "
                    f"{self.max_bytes // (1024 * 1024)} MB limit"
                )
            },
        )
        await response(scope, receive, send)


def _safe_unlink(path: str | os.PathLike[str] | None) -> bool:
    """Best-effort removal that never masks the primary request result."""
    if not path:
        return True

    target = Path(path)
    for attempt in range(3):
        try:
            target.unlink(missing_ok=True)
            return True
        except PermissionError:
            if attempt < 2:
                time.sleep(0.05)
                continue
            logger.exception("Could not remove temporary upload: %s", target)
        except OSError:
            logger.exception("Could not remove temporary upload: %s", target)
        return False
    return False


def _close_upload_file(upload_file: UploadFile) -> None:
    """Close Starlette's spooled file synchronously so cancellation cannot skip it."""
    try:
        upload_file.file.close()
    except Exception:
        logger.exception("Could not close the multipart upload spool")


def _cleanup_stale_temp_files() -> None:
    cutoff = time.time() - STALE_TEMP_FILE_AGE_SECONDS
    temp_dir = Path(tempfile.gettempdir())
    for candidate in temp_dir.glob(f"{TEMP_FILE_PREFIX}*.wav"):
        try:
            if candidate.stat().st_mtime < cutoff:
                _safe_unlink(candidate)
        except OSError:
            logger.exception("Could not inspect stale temporary upload: %s", candidate)


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.analysis_loop = asyncio.get_running_loop()
    application.state.analysis_slots = asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)
    await run_in_threadpool(_cleanup_stale_temp_files)
    yield


app = FastAPI(
    title="Keystroke Audio Analyzer",
    description=(
        "Analyze WAV audio recorded from physical keyboard typing and return "
        "keystroke counts per word."
    ),
    version="0.2.0",
    lifespan=lifespan,
)
app.add_middleware(RequestBodyLimitMiddleware, max_bytes=MAX_REQUEST_BYTES)


def _analysis_slots() -> asyncio.Semaphore:
    """Return a semaphore bound to the current server/test event loop."""
    current_loop = asyncio.get_running_loop()
    slots = getattr(app.state, "analysis_slots", None)
    slots_loop = getattr(app.state, "analysis_loop", None)
    if slots is None or slots_loop is not current_loop:
        slots = asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)
        app.state.analysis_slots = slots
        app.state.analysis_loop = current_loop
    return slots


@asynccontextmanager
async def _analysis_slot():
    slots = _analysis_slots()
    try:
        await asyncio.wait_for(
            slots.acquire(),
            timeout=ANALYSIS_QUEUE_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        raise AnalysisBusyError from exc
    try:
        yield
    finally:
        slots.release()


def _create_temp_wav_path() -> str:
    with tempfile.NamedTemporaryFile(
        delete=False,
        prefix=TEMP_FILE_PREFIX,
        suffix=".wav",
    ) as tmp_file:
        return tmp_file.name


async def _save_upload_file_tmp(
    upload_file: UploadFile,
    wav_path: str | None = None,
) -> str:
    """Copy one bounded upload into an owned temporary WAV path."""
    target_path = wav_path or _create_temp_wav_path()
    completed = False
    try:
        suffix = Path(upload_file.filename or "").suffix.lower()
        if suffix not in {".wav", ".wave"}:
            raise InvalidUploadError(
                "Only WAV uploads with a .wav or .wave extension are supported"
            )

        total_bytes = 0
        with Path(target_path).open("wb") as tmp_file:
            while chunk := await upload_file.read(UPLOAD_CHUNK_BYTES):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise UploadTooLargeError(
                        "Uploaded file exceeds the "
                        f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit"
                    )
                await run_in_threadpool(tmp_file.write, chunk)

        if total_bytes == 0:
            raise InvalidUploadError("Uploaded file is empty")

        completed = True
        return target_path
    finally:
        _close_upload_file(upload_file)
        if not completed:
            _safe_unlink(target_path)


def _validate_wav_file(wav_path: str) -> None:
    try:
        info = sf.info(wav_path)
    except (RuntimeError, sf.LibsndfileError) as exc:
        raise InvalidAudioError("Uploaded file is not a valid WAV recording") from exc

    if info.format != "WAV" or info.frames <= 0 or info.samplerate <= 0:
        raise InvalidAudioError("Uploaded file is not a valid WAV recording")

    duration_seconds = info.frames / info.samplerate
    if duration_seconds > MAX_AUDIO_DURATION_SECONDS:
        raise InvalidAudioError(
            f"Recording exceeds the {MAX_AUDIO_DURATION_SECONDS:g} second duration limit"
        )


def _validate_counts(counts: object) -> list[int]:
    if not isinstance(counts, list):
        raise TypeError("Predictor returned a non-list result")
    invalid = any(
        isinstance(count, bool) or not isinstance(count, int) or count < 1
        for count in counts
    )
    if invalid:
        raise TypeError("Predictor returned invalid keystroke counts")
    return counts


def _public_model_status(models: dict[str, object]) -> dict[str, object]:
    public: dict[str, object] = {
        "all_present": bool(models.get("all_present")),
        "all_ready": bool(models.get("all_ready")),
    }
    for model_name in ("yamnet_classifier", "count_predictor"):
        raw = models.get(model_name)
        entry = raw if isinstance(raw, dict) else {}
        raw_path = str(entry.get("path", model_name))
        ready = bool(entry.get("ready"))
        public[model_name] = {
            "name": Path(raw_path).name,
            "exists": bool(entry.get("exists")),
            "ready": ready,
            "error": None if ready else "model artifact unavailable",
        }
    return public


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    internal_models = await run_in_threadpool(get_model_status)
    public_models = _public_model_status(internal_models)
    is_ready = bool(public_models["all_ready"])
    payload = {
        "status": "ready" if is_ready else "degraded",
        "models": public_models,
        "yamnet_runtime": "loaded lazily on first ML/YAMNet analysis",
    }
    if not is_ready:
        logger.warning("Model readiness check failed: %r", internal_models)
        return JSONResponse(status_code=503, content=payload)
    return payload


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    method: str = Form("yamnet"),
    threshold: float = Form(CLASSIFIER_THRESHOLD, ge=0.0, le=1.0),
    delta: float = Form(SENSITIVITY_DELTA, gt=0.0, le=1.0),
    gap_threshold: float | None = Form(None, gt=0.0, le=5.0),
    merge_gap_seconds: float = Form(MERGE_GAP_SECONDS, gt=0.0, le=1.0),
):
    method = method.strip().lower()
    if method not in VALID_METHODS:
        _close_upload_file(file)
        raise HTTPException(
            status_code=400,
            detail=f"method must be one of {sorted(VALID_METHODS)}",
        )

    wav_path: str | None = None
    try:
        wav_path = _create_temp_wav_path()
        await _save_upload_file_tmp(file, wav_path)
        await run_in_threadpool(_validate_wav_file, wav_path)

        async with _analysis_slot():
            if method == "rule":
                counts = await run_in_threadpool(
                    predict_keystroke_counts_rule,
                    wav_path,
                    delta=delta,
                    gap_threshold=gap_threshold,
                    merge_gap_seconds=merge_gap_seconds,
                )
            elif method == "ml":
                counts = await run_in_threadpool(
                    predict_keystroke_counts_ml,
                    wav_path,
                    delta=delta,
                    gap_threshold=gap_threshold,
                    merge_gap_seconds=merge_gap_seconds,
                )
            else:
                counts = await run_in_threadpool(
                    predict_keystroke_counts,
                    wav_path,
                    threshold=threshold,
                    delta=delta,
                    gap_threshold=gap_threshold,
                    merge_gap_seconds=merge_gap_seconds,
                )

        counts = _validate_counts(counts)
        return {"counts": counts, "formatted": format_output(counts)}
    except FileNotFoundError as exc:
        logger.warning("Required prediction model is unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Required prediction model is unavailable",
        ) from exc
    except AnalysisBusyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Analyzer is busy; retry the request shortly",
        ) from exc
    except UploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except ClientInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected audio processing failure")
        raise HTTPException(
            status_code=500,
            detail="Audio processing failed. Check the server logs for details.",
        ) from exc
    finally:
        _safe_unlink(wav_path)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("KEYSTROKE_HOST", "127.0.0.1")
    port = int(os.getenv("KEYSTROKE_PORT", "8000"))
    reload_enabled = os.getenv("KEYSTROKE_RELOAD", "0") == "1"
    uvicorn.run("app:app", host=host, port=port, reload=reload_enabled)
