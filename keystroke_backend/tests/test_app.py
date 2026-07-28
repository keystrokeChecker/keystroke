from __future__ import annotations

import asyncio
import io
import os
import wave
from collections.abc import Callable

import httpx
import pytest

import app as api


def _request(method: str, path: str, **kwargs: object) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def _wav_bytes(*, frames: int = 80, sample_rate: int = 8_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frames)
    return buffer.getvalue()


def _audio_upload(
    contents: bytes | None = None,
    *,
    filename: str = "recording.wav",
) -> dict:
    if contents is None:
        contents = _wav_bytes()
    return {"file": (filename, contents, "audio/wav")}


def test_health() -> None:
    response = _request("GET", "/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_reports_model_status(monkeypatch: pytest.MonkeyPatch) -> None:
    model_status = {
        "all_present": True,
        "all_ready": True,
        "yamnet_classifier": {
            "path": "C:/private/models/classifier.joblib",
            "exists": True,
            "ready": True,
            "error": None,
        },
        "count_predictor": {
            "path": "C:/private/models/count.joblib",
            "exists": True,
            "ready": True,
            "error": None,
        },
    }
    monkeypatch.setattr(api, "get_model_status", lambda: model_status)

    response = _request("GET", "/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["models"]["all_ready"] is True
    assert body["models"]["yamnet_classifier"] == {
        "name": "classifier.joblib",
        "exists": True,
        "ready": True,
        "error": None,
    }
    assert "private" not in str(body)


def test_ready_returns_503_when_model_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api,
        "get_model_status",
        lambda: {
            "all_present": True,
            "all_ready": False,
            "yamnet_classifier": {
                "path": "C:/private/models/classifier.joblib",
                "exists": True,
                "ready": True,
                "error": None,
            },
            "count_predictor": {
                "path": "C:/private/models/count.joblib",
                "exists": True,
                "ready": False,
                "error": "ValueError: internal model detail",
            },
        },
    )

    response = _request("GET", "/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["models"]["count_predictor"]["error"] == "model artifact unavailable"
    assert "internal model detail" not in str(body)


@pytest.mark.parametrize(
    ("method", "attribute", "expected_kwargs"),
    [
        (
            "rule",
            "predict_keystroke_counts_rule",
            {
                "delta": 0.11,
                "gap_threshold": 0.61,
                "merge_gap_seconds": 0.08,
            },
        ),
        (
            "ml",
            "predict_keystroke_counts_ml",
            {
                "delta": 0.11,
                "gap_threshold": 0.61,
                "merge_gap_seconds": 0.08,
            },
        ),
        (
            "yamnet",
            "predict_keystroke_counts",
            {
                "threshold": 0.73,
                "delta": 0.11,
                "gap_threshold": 0.61,
                "merge_gap_seconds": 0.08,
            },
        ),
    ],
)
def test_analyze_routes_to_requested_method_and_cleans_temp_file(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    attribute: str,
    expected_kwargs: dict[str, float],
) -> None:
    observed: dict[str, object] = {}

    def fake_predictor(wav_path: str, **kwargs: float) -> list[int]:
        observed["path"] = wav_path
        observed["exists_during_prediction"] = os.path.exists(wav_path)
        observed["kwargs"] = kwargs
        return [3, 7]

    monkeypatch.setattr(api, attribute, fake_predictor)

    response = _request(
        "POST",
        "/analyze",
        files=_audio_upload(),
        data={
            "method": method,
            "threshold": "0.73",
            "delta": "0.11",
            "gap_threshold": "0.61",
            "merge_gap_seconds": "0.08",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"counts": [3, 7], "formatted": "3|7"}
    assert observed["exists_during_prediction"] is True
    assert observed["kwargs"] == expected_kwargs
    assert not os.path.exists(str(observed["path"]))


def test_analyze_uses_yamnet_when_method_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fake_predictor(wav_path: str, **kwargs: float) -> list[int]:
        nonlocal called
        called = True
        return [4]

    monkeypatch.setattr(api, "predict_keystroke_counts", fake_predictor)

    response = _request("POST", "/analyze", files=_audio_upload())

    assert response.status_code == 200
    assert response.json() == {"counts": [4], "formatted": "4"}
    assert called is True


def test_analyze_rejects_unknown_method() -> None:
    response = _request(
        "POST",
        "/analyze",
        files=_audio_upload(),
        data={"method": "unknown"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "method must be one of ['ml', 'rule', 'yamnet']"


def test_analyze_rejects_empty_upload() -> None:
    response = _request(
        "POST",
        "/analyze",
        files=_audio_upload(b""),
        data={"method": "rule"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty"


@pytest.mark.parametrize("error_type", [FileNotFoundError, ValueError])
def test_analyze_hides_predictor_and_model_error_details(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    def fail_predictor(wav_path: str, **kwargs: float) -> list[int]:
        raise error_type("known failure")

    monkeypatch.setattr(api, "predict_keystroke_counts_rule", fail_predictor)

    response = _request(
        "POST",
        "/analyze",
        files=_audio_upload(),
        data={"method": "rule"},
    )

    if error_type is FileNotFoundError:
        assert response.status_code == 503
        assert response.json() == {
            "detail": "Required prediction model is unavailable"
        }
    else:
        assert response.status_code == 500
        assert response.json() == {
            "detail": "Audio processing failed. Check the server logs for details."
        }
    assert "known failure" not in response.text


def test_analyze_hides_unexpected_exception_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_predictor(wav_path: str, **kwargs: float) -> list[int]:
        raise RuntimeError("sensitive implementation detail")

    monkeypatch.setattr(api, "predict_keystroke_counts_rule", fail_predictor)

    response = _request(
        "POST",
        "/analyze",
        files=_audio_upload(),
        data={"method": "rule"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Audio processing failed. Check the server logs for details."
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("threshold", "1.1"),
        ("threshold", "-0.1"),
        ("delta", "0"),
        ("gap_threshold", "-1"),
        ("merge_gap_seconds", "0"),
    ],
)
def test_analyze_rejects_out_of_range_parameters(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    predictor: Callable[..., list[int]] = lambda wav_path, **kwargs: [1]
    monkeypatch.setattr(api, "predict_keystroke_counts_rule", predictor)
    data = {"method": "rule", field: value}

    response = _request("POST", "/analyze", files=_audio_upload(), data=data)

    assert response.status_code == 422


def test_analyze_rejects_non_wav_extension() -> None:
    response = _request(
        "POST",
        "/analyze",
        files=_audio_upload(filename="recording.mp3"),
        data={"method": "rule"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Only WAV uploads with a .wav or .wave extension are supported"
    }


def test_analyze_rejects_oversized_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 32)

    response = _request(
        "POST",
        "/analyze",
        files=_audio_upload(),
        data={"method": "rule"},
    )

    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"]


def test_analyze_rejects_corrupt_wav_and_removes_temp_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, str] = {}
    original_save = api._save_upload_file_tmp

    async def capture_path(upload_file, wav_path=None) -> str:
        path = await original_save(upload_file, wav_path)
        observed["path"] = path
        return path

    monkeypatch.setattr(api, "_save_upload_file_tmp", capture_path)

    response = _request(
        "POST",
        "/analyze",
        files=_audio_upload(b"not a wave file"),
        data={"method": "rule"},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Uploaded file is not a valid WAV recording"}
    assert not os.path.exists(observed["path"])


def test_analyze_rejects_recording_over_duration_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "MAX_AUDIO_DURATION_SECONDS", 0.001)

    response = _request(
        "POST",
        "/analyze",
        files=_audio_upload(),
        data={"method": "rule"},
    )

    assert response.status_code == 400
    assert "duration limit" in response.json()["detail"]


@pytest.mark.parametrize(
    "invalid_counts",
    [None, "3|7", [0], [-1], [True], [2.5]],
)
def test_analyze_rejects_invalid_predictor_results(
    monkeypatch: pytest.MonkeyPatch,
    invalid_counts: object,
) -> None:
    monkeypatch.setattr(
        api,
        "predict_keystroke_counts_rule",
        lambda wav_path, **kwargs: invalid_counts,
    )

    response = _request(
        "POST",
        "/analyze",
        files=_audio_upload(),
        data={"method": "rule"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Audio processing failed. Check the server logs for details."
    }


def test_request_body_limit_rejects_declared_oversized_body() -> None:
    response = _request(
        "POST",
        "/analyze",
        headers={"content-length": str(api.MAX_REQUEST_BYTES + 1)},
        content=b"",
    )

    assert response.status_code == 413
    assert "Request body exceeds" in response.json()["detail"]


def test_request_body_limit_counts_streamed_bytes_without_header() -> None:
    sent: list[dict] = []
    incoming = iter(
        [
            {"type": "http.request", "body": b"12", "more_body": True},
            {"type": "http.request", "body": b"34", "more_body": False},
        ]
    )

    async def receive():
        return next(incoming)

    async def send(message) -> None:
        sent.append(message)

    async def downstream(scope, receive, send) -> None:
        while True:
            message = await receive()
            if not message.get("more_body", False):
                break

    middleware = api.RequestBodyLimitMiddleware(downstream, max_bytes=3)
    scope = {"type": "http", "path": "/analyze", "headers": []}

    asyncio.run(middleware(scope, receive, send))

    assert sent[0]["status"] == 413


def test_cancelled_upload_removes_partial_temp_file() -> None:
    class CancellingUpload:
        filename = "recording.wav"

        def __init__(self) -> None:
            self.file = io.BytesIO()
            self.read_count = 0

        async def read(self, size: int) -> bytes:
            self.read_count += 1
            if self.read_count == 1:
                return b"partial"
            raise asyncio.CancelledError

    upload = CancellingUpload()
    wav_path = api._create_temp_wav_path()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(api._save_upload_file_tmp(upload, wav_path))

    assert not os.path.exists(wav_path)
    assert upload.file.closed is True


def test_upload_close_failure_does_not_mask_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CloseFailingFile:
        close_attempts = 0

        def close(self) -> None:
            self.close_attempts += 1
            if self.close_attempts == 1:
                raise OSError("close failed")

    class Upload:
        filename = "recording.wav"

        def __init__(self) -> None:
            self.file = CloseFailingFile()
            self.contents = iter([_wav_bytes(), b""])

        async def read(self, size: int) -> bytes:
            return next(self.contents)

    wav_path = api._create_temp_wav_path()
    upload = Upload()

    result = asyncio.run(api._save_upload_file_tmp(upload, wav_path))

    assert result == wav_path
    assert os.path.exists(wav_path)
    api._safe_unlink(wav_path)


def test_safe_unlink_permission_error_never_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        api.Path,
        "unlink",
        lambda self, missing_ok=True: (_ for _ in ()).throw(PermissionError("locked")),
    )

    assert api._safe_unlink("C:/tmp/locked.wav") is False


def test_busy_analysis_queue_returns_503_and_cleans_temp_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, str] = {}
    original_save = api._save_upload_file_tmp

    async def capture_path(upload_file, wav_path=None) -> str:
        path = await original_save(upload_file, wav_path)
        observed["path"] = path
        return path

    class BusySlots:
        async def acquire(self) -> None:
            await asyncio.Future()

        def release(self) -> None:
            pytest.fail("an unacquired slot must not be released")

    monkeypatch.setattr(api, "_save_upload_file_tmp", capture_path)
    monkeypatch.setattr(api, "_analysis_slots", lambda: BusySlots())
    monkeypatch.setattr(api, "ANALYSIS_QUEUE_TIMEOUT_SECONDS", 0.001)

    response = _request(
        "POST",
        "/analyze",
        files=_audio_upload(),
        data={"method": "rule"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Analyzer is busy; retry the request shortly"
    }
    assert not os.path.exists(observed["path"])
