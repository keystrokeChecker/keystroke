from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import src.predictor as predictor


class _ModelPath:
    def __init__(self, exists: bool, name: str = "model.joblib") -> None:
        self._exists = exists
        self._name = name

    def exists(self) -> bool:
        return self._exists

    def __str__(self) -> str:
        return self._name


def _fail(message: str):
    def inner(*args, **kwargs):
        pytest.fail(message)

    return inner


def test_rule_pipeline_routes_detection_to_word_segmentation(monkeypatch) -> None:
    calls = {}
    onsets = np.array([0.1, 0.2, 1.1])

    def fake_detect(wav_path, **kwargs):
        calls["detect"] = (wav_path, kwargs)
        return onsets, np.array([]), 16_000

    def fake_segment(values, gap_threshold):
        calls["segment"] = (values, gap_threshold)
        return [2, 1], [[0.1, 0.2], [1.1]]

    monkeypatch.setattr(predictor, "detect_onsets", fake_detect)
    monkeypatch.setattr(predictor, "segment_into_words", fake_segment)

    result = predictor.predict_keystroke_counts_rule(
        "typing.wav",
        delta=0.12,
        gap_threshold=0.8,
        merge_gap_seconds=0.03,
    )

    assert result == [2, 1]
    assert calls["detect"] == (
        "typing.wav",
        {"delta": 0.12, "min_gap_seconds": 0.03},
    )
    assert calls["segment"][0] is onsets
    assert calls["segment"][1] == pytest.approx(0.8)


def test_rule_pipeline_short_circuits_when_no_onsets_are_detected(monkeypatch) -> None:
    monkeypatch.setattr(
        predictor,
        "detect_onsets",
        lambda *args, **kwargs: (np.array([]), np.array([]), 16_000),
    )
    monkeypatch.setattr(
        predictor,
        "segment_into_words",
        _fail("empty detection must not be segmented"),
    )

    assert predictor.predict_keystroke_counts_rule("silence.wav") == []


def test_yamnet_pipeline_filters_onsets_then_segments_them(monkeypatch) -> None:
    calls = {}
    raw_onsets = np.array([0.1, 0.2, 0.3, 1.2])
    filtered_onsets = np.array([0.1, 0.3, 1.2])
    classifier_path = _ModelPath(True, "classifier.joblib")

    def fake_detect(wav_path, **kwargs):
        calls["detect"] = (wav_path, kwargs)
        return raw_onsets, np.array([]), 16_000

    def fake_filter(onsets, wav_path, model_path, confidence_threshold):
        calls["filter"] = (
            onsets,
            wav_path,
            model_path,
            confidence_threshold,
        )
        return filtered_onsets

    def fake_segment(onsets, gap_threshold):
        calls["segment"] = (onsets, gap_threshold)
        return [2, 1], [[0.1, 0.3], [1.2]]

    monkeypatch.setattr(predictor, "_DEFAULT_CLASSIFIER", classifier_path)
    monkeypatch.setattr(predictor, "detect_onsets", fake_detect)
    monkeypatch.setattr(predictor, "filter_onsets_with_yamnet", fake_filter)
    monkeypatch.setattr(predictor, "segment_into_words", fake_segment)

    result = predictor.predict_keystroke_counts(
        "typing.wav",
        threshold=0.82,
        delta=0.11,
        gap_threshold=0.9,
        merge_gap_seconds=0.04,
    )

    assert result == [2, 1]
    assert calls["detect"] == (
        "typing.wav",
        {"delta": 0.11, "min_gap_seconds": 0.04},
    )
    assert calls["filter"][0] is raw_onsets
    assert calls["filter"][1:] == (
        "typing.wav",
        "classifier.joblib",
        0.82,
    )
    assert calls["segment"][0] is filtered_onsets
    assert calls["segment"][1] == pytest.approx(0.9)


def test_yamnet_pipeline_falls_back_to_raw_onsets_without_classifier(
    monkeypatch,
) -> None:
    calls = {}
    raw_onsets = np.array([0.1, 0.2, 1.1])

    monkeypatch.setattr(predictor, "_DEFAULT_CLASSIFIER", _ModelPath(False))
    monkeypatch.setattr(
        predictor,
        "detect_onsets",
        lambda *args, **kwargs: (raw_onsets, np.array([]), 16_000),
    )
    monkeypatch.setattr(
        predictor,
        "filter_onsets_with_yamnet",
        _fail("YAMNet filtering must not run without a classifier"),
    )

    def fake_segment(onsets, gap_threshold):
        calls["segment"] = (onsets, gap_threshold)
        return [2, 1], []

    monkeypatch.setattr(predictor, "segment_into_words", fake_segment)

    result = predictor.predict_keystroke_counts(
        "typing.wav",
        gap_threshold=None,
    )

    assert result == [2, 1]
    assert calls["segment"][0] is raw_onsets
    assert calls["segment"][1] == pytest.approx(predictor.GAP_THRESHOLD_ML)


def test_yamnet_pipeline_short_circuits_when_filter_rejects_everything(
    monkeypatch,
) -> None:
    monkeypatch.setattr(predictor, "_DEFAULT_CLASSIFIER", _ModelPath(True))
    monkeypatch.setattr(
        predictor,
        "detect_onsets",
        lambda *args, **kwargs: (np.array([0.1]), np.array([]), 16_000),
    )
    monkeypatch.setattr(
        predictor,
        "filter_onsets_with_yamnet",
        lambda *args, **kwargs: np.array([]),
    )
    monkeypatch.setattr(
        predictor,
        "segment_into_words",
        _fail("rejected onsets must not be segmented"),
    )

    assert predictor.predict_keystroke_counts("noise.wav") == []


def test_ml_pipeline_raises_before_audio_work_when_model_is_missing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(predictor, "_COUNT_PREDICTOR", _ModelPath(False, "missing.joblib"))
    monkeypatch.setattr(
        predictor.joblib,
        "load",
        _fail("a missing model must not be loaded"),
    )
    monkeypatch.setattr(
        predictor,
        "detect_onsets",
        _fail("audio detection must not run without a model"),
    )

    with pytest.raises(FileNotFoundError, match="missing.joblib"):
        predictor.predict_keystroke_counts_ml("typing.wav")


def test_ml_pipeline_routes_features_to_model_and_rounds_predictions(
    monkeypatch,
) -> None:
    calls = {"pool": [], "predict": []}
    onsets = np.array([0.1, 0.2, 1.1])
    word_groups = [[0.1, 0.2], [1.1]]
    candidates = [
        SimpleNamespace(onset_time=0.1),
        SimpleNamespace(onset_time=0.2),
        SimpleNamespace(onset_time=1.1),
    ]
    features = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

    class FakeModel:
        def __init__(self) -> None:
            self._predictions = iter([2.6, 0.2])

        def predict(self, values):
            calls["predict"].append(values.copy())
            return np.array([next(self._predictions)])

    model = FakeModel()

    monkeypatch.setattr(
        predictor,
        "_COUNT_PREDICTOR",
        _ModelPath(True, "count-predictor.joblib"),
    )

    def fake_load(path):
        calls["loaded_path"] = path
        return {"model": model}

    monkeypatch.setattr(predictor.joblib, "load", fake_load)

    def fake_detect(wav_path, **kwargs):
        calls["detect"] = (wav_path, kwargs)
        return onsets, np.array([]), 16_000

    monkeypatch.setattr(predictor, "detect_onsets", fake_detect)

    def fake_segment(values, gap_threshold):
        calls["segment"] = (values, gap_threshold)
        return [2, 1], word_groups

    monkeypatch.setattr(predictor, "segment_into_words", fake_segment)

    def fake_candidates(wav_path, values):
        calls["candidates"] = (wav_path, values)
        return candidates

    monkeypatch.setattr(predictor, "extract_yamnet_candidates", fake_candidates)

    def fake_features(values, wav_path):
        calls["features"] = (values, wav_path)
        return features

    monkeypatch.setattr(predictor, "extract_features", fake_features)

    def fake_pool(values, times):
        calls["pool"].append((values.copy(), times.copy()))
        return values.mean(axis=0)

    monkeypatch.setattr(predictor, "pool_word_features", fake_pool)

    result = predictor.predict_keystroke_counts_ml(
        "typing.wav",
        delta=0.1,
        gap_threshold=0.85,
        merge_gap_seconds=0.05,
    )

    assert result == [3, 1]
    assert calls["loaded_path"] == "count-predictor.joblib"
    assert calls["detect"] == (
        "typing.wav",
        {"delta": 0.1, "min_gap_seconds": 0.05},
    )
    assert calls["segment"][0] is onsets
    assert calls["segment"][1] == pytest.approx(0.85)
    assert calls["candidates"] == ("typing.wav", onsets)
    assert calls["features"] == (candidates, "typing.wav")
    assert len(calls["pool"]) == 2
    np.testing.assert_array_equal(calls["pool"][0][0], features[:2])
    np.testing.assert_array_equal(calls["pool"][1][0], features[2:])
    assert all(values.shape == (1, 2) for values in calls["predict"])


def test_ml_pipeline_falls_back_to_one_per_word_without_yamnet_candidates(
    monkeypatch,
) -> None:
    model = SimpleNamespace(predict=_fail("model must not run without features"))

    monkeypatch.setattr(predictor, "_COUNT_PREDICTOR", _ModelPath(True))
    monkeypatch.setattr(predictor.joblib, "load", lambda path: {"model": model})
    monkeypatch.setattr(
        predictor,
        "detect_onsets",
        lambda *args, **kwargs: (
            np.array([0.1, 0.2, 1.1]),
            np.array([]),
            16_000,
        ),
    )
    monkeypatch.setattr(
        predictor,
        "segment_into_words",
        lambda *args, **kwargs: ([2, 1], [[0.1, 0.2], [1.1]]),
    )
    monkeypatch.setattr(predictor, "extract_yamnet_candidates", lambda *args: [])
    monkeypatch.setattr(
        predictor,
        "extract_features",
        _fail("feature extraction must not run without candidates"),
    )

    assert predictor.predict_keystroke_counts_ml("typing.wav") == [1, 1]
