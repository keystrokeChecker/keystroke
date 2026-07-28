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
import yamnet_filter


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


@pytest.fixture(autouse=True)
def _clear_joblib_model_caches():
    predictor._load_count_predictor.cache_clear()
    yamnet_filter._load_classifier_artifact.cache_clear()
    yield
    predictor._load_count_predictor.cache_clear()
    yamnet_filter._load_classifier_artifact.cache_clear()


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


def test_yamnet_pipeline_raises_before_audio_work_without_classifier(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        predictor,
        "_DEFAULT_CLASSIFIER",
        _ModelPath(False, "missing-classifier.joblib"),
    )
    monkeypatch.setattr(
        predictor,
        "detect_onsets",
        _fail("audio detection must not run without the YAMNet classifier"),
    )
    monkeypatch.setattr(
        predictor,
        "filter_onsets_with_yamnet",
        _fail("YAMNet filtering must not run without a classifier"),
    )

    with pytest.raises(FileNotFoundError, match="missing-classifier.joblib"):
        predictor.predict_keystroke_counts("typing.wav", gap_threshold=None)


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


def test_count_predictor_artifact_is_loaded_once_per_path(monkeypatch) -> None:
    load_calls = []
    model = SimpleNamespace(predict=lambda values: np.array([1.0]))

    monkeypatch.setattr(
        predictor,
        "_COUNT_PREDICTOR",
        _ModelPath(True, "cached-count-predictor.joblib"),
    )

    def fake_load(path):
        load_calls.append(path)
        return {"model": model, "feature_dimension": 4}

    monkeypatch.setattr(predictor.joblib, "load", fake_load)
    monkeypatch.setattr(
        predictor,
        "detect_onsets",
        lambda *args, **kwargs: (np.array([]), np.array([]), 16_000),
    )

    assert predictor.predict_keystroke_counts_ml("first.wav") == []
    assert predictor.predict_keystroke_counts_ml("second.wav") == []
    assert load_calls == ["cached-count-predictor.joblib"]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (None, "supported count predictor"),
        ({}, "supported count predictor"),
        ({"model": object()}, "callable predict"),
    ],
)
def test_count_predictor_rejects_invalid_payloads_before_audio_work(
    monkeypatch,
    payload,
    message,
) -> None:
    monkeypatch.setattr(
        predictor,
        "_COUNT_PREDICTOR",
        _ModelPath(True, "invalid-count-predictor.joblib"),
    )
    monkeypatch.setattr(predictor.joblib, "load", lambda path: payload)
    monkeypatch.setattr(
        predictor,
        "detect_onsets",
        _fail("invalid model artifacts must fail before audio work"),
    )

    with pytest.raises(ValueError, match=message):
        predictor.predict_keystroke_counts_ml("typing.wav")


def test_count_predictor_rejects_conflicting_feature_metadata(monkeypatch) -> None:
    model = SimpleNamespace(
        n_features_in_=5,
        predict=lambda values: np.array([1.0]),
    )
    monkeypatch.setattr(
        predictor,
        "_COUNT_PREDICTOR",
        _ModelPath(True, "conflicting-count-predictor.joblib"),
    )
    monkeypatch.setattr(
        predictor.joblib,
        "load",
        lambda path: {"model": model, "feature_dimension": 4},
    )
    monkeypatch.setattr(
        predictor,
        "detect_onsets",
        _fail("invalid model artifacts must fail before audio work"),
    )

    with pytest.raises(ValueError, match="payload declares 4, model expects 5"):
        predictor.predict_keystroke_counts_ml("typing.wav")


def test_ml_pipeline_rejects_inference_feature_dimension_mismatch(
    monkeypatch,
) -> None:
    model = SimpleNamespace(
        n_features_in_=3,
        predict=_fail("model must not receive an incompatible feature matrix"),
    )
    monkeypatch.setattr(
        predictor,
        "_COUNT_PREDICTOR",
        _ModelPath(True, "three-feature-count-predictor.joblib"),
    )
    monkeypatch.setattr(
        predictor.joblib,
        "load",
        lambda path: {"model": model, "feature_dimension": 3},
    )
    monkeypatch.setattr(
        predictor,
        "detect_onsets",
        lambda *args, **kwargs: (np.array([0.1]), np.array([]), 16_000),
    )
    monkeypatch.setattr(
        predictor,
        "segment_into_words",
        lambda *args, **kwargs: ([1], [[0.1]]),
    )
    monkeypatch.setattr(
        predictor,
        "extract_yamnet_candidates",
        lambda *args: [SimpleNamespace(onset_time=0.1)],
    )
    monkeypatch.setattr(
        predictor,
        "extract_features",
        lambda *args: np.array([[1.0, 2.0]]),
    )
    monkeypatch.setattr(
        predictor,
        "pool_word_features",
        lambda values, times: values.mean(axis=0),
    )

    with pytest.raises(ValueError, match="model expects 3, inference produced 2"):
        predictor.predict_keystroke_counts_ml("typing.wav")


def test_get_model_status_validates_joblib_models_without_loading_tfhub(
    monkeypatch,
    tmp_path,
) -> None:
    classifier_path = tmp_path / "classifier.joblib"
    count_path = tmp_path / "count.joblib"
    classifier_path.touch()
    count_path.touch()

    classifier = SimpleNamespace(
        n_features_in_=4,
        predict_proba=lambda values: np.column_stack(
            [np.zeros(len(values)), np.ones(len(values))]
        ),
    )
    count_model = SimpleNamespace(
        n_features_in_=6,
        predict=lambda values: np.ones(len(values)),
    )

    def fake_load(path):
        if Path(path).name == classifier_path.name:
            return {
                "pipeline": classifier,
                "feature_dimension": 4,
            }
        if Path(path).name == count_path.name:
            return {
                "model": count_model,
                "feature_dimension": 6,
            }
        pytest.fail(f"unexpected model path: {path}")

    monkeypatch.setattr(predictor, "_DEFAULT_CLASSIFIER", classifier_path)
    monkeypatch.setattr(predictor, "_COUNT_PREDICTOR", count_path)
    monkeypatch.setattr(predictor.joblib, "load", fake_load)
    monkeypatch.setattr(
        yamnet_filter,
        "_load_yamnet",
        _fail("model readiness must not load TensorFlow Hub"),
    )

    status = predictor.get_model_status()

    assert status["all_present"] is True
    assert status["all_ready"] is True
    assert status["yamnet_classifier"] == {
        "path": str(classifier_path),
        "exists": True,
        "ready": True,
        "error": None,
    }
    assert status["count_predictor"] == {
        "path": str(count_path),
        "exists": True,
        "ready": True,
        "error": None,
    }


def test_get_model_status_distinguishes_presence_from_load_readiness(
    monkeypatch,
    tmp_path,
) -> None:
    classifier_path = tmp_path / "classifier.joblib"
    count_path = tmp_path / "count.joblib"
    classifier_path.touch()
    count_path.touch()
    classifier = SimpleNamespace(predict_proba=lambda values: np.empty((0, 2)))

    def fake_load(path):
        if Path(path).name == classifier_path.name:
            return {"pipeline": classifier}
        return {"unexpected": object()}

    monkeypatch.setattr(predictor, "_DEFAULT_CLASSIFIER", classifier_path)
    monkeypatch.setattr(predictor, "_COUNT_PREDICTOR", count_path)
    monkeypatch.setattr(predictor.joblib, "load", fake_load)

    status = predictor.get_model_status()

    assert status["all_present"] is True
    assert status["all_ready"] is False
    assert status["yamnet_classifier"]["ready"] is True
    assert status["count_predictor"]["ready"] is False
    assert "ValueError" in status["count_predictor"]["error"]


def test_get_model_status_does_not_load_missing_artifacts(monkeypatch) -> None:
    monkeypatch.setattr(
        predictor,
        "_DEFAULT_CLASSIFIER",
        _ModelPath(False, "missing-classifier.joblib"),
    )
    monkeypatch.setattr(
        predictor,
        "_COUNT_PREDICTOR",
        _ModelPath(False, "missing-count.joblib"),
    )
    monkeypatch.setattr(
        predictor,
        "load_classifier",
        _fail("a missing classifier must not be loaded"),
    )
    monkeypatch.setattr(
        predictor.joblib,
        "load",
        _fail("a missing count model must not be loaded"),
    )

    status = predictor.get_model_status()

    assert status["all_present"] is False
    assert status["all_ready"] is False
    assert status["yamnet_classifier"]["exists"] is False
    assert status["count_predictor"]["exists"] is False
