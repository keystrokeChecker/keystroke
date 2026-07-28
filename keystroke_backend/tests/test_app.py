from __future__ import annotations

import asyncio
import os
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


def _audio_upload(contents: bytes = b"placeholder audio") -> dict:
    return {"file": ("recording.wav", contents, "audio/wav")}


def test_health() -> None:
    response = _request("GET", "/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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
def test_analyze_maps_known_predictor_errors_to_bad_request(
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

    assert response.status_code == 400
    assert response.json() == {"detail": "known failure"}


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


@pytest.mark.xfail(
    strict=True,
    reason="Step 4 will add numeric request bounds.",
)
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
