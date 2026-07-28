from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import yamnet_filter


def _fail(message: str):
    def inner(*args, **kwargs):
        pytest.fail(message)

    return inner


@pytest.fixture(autouse=True)
def _clear_classifier_cache():
    yamnet_filter._load_classifier_artifact.cache_clear()
    yield
    yamnet_filter._load_classifier_artifact.cache_clear()


def test_load_classifier_caches_a_validated_artifact(monkeypatch, tmp_path) -> None:
    model_path = tmp_path / "classifier.joblib"
    model_path.touch()
    load_calls = []
    pipeline = SimpleNamespace(
        n_features_in_=4,
        predict_proba=lambda values: np.empty((len(values), 2)),
    )

    def fake_load(path):
        load_calls.append(path)
        return {
            "pipeline": pipeline,
            "feature_dimension": 4,
        }

    monkeypatch.setattr(yamnet_filter.joblib, "load", fake_load)

    first = yamnet_filter.load_classifier(model_path)
    second = yamnet_filter.load_classifier(str(model_path))

    assert first is pipeline
    assert second is pipeline
    assert load_calls == [str(model_path.resolve())]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (None, "supported YAMNet classifier"),
        ({}, "supported YAMNet classifier"),
        ({"pipeline": object()}, "callable predict_proba"),
        (
            {
                "pipeline": SimpleNamespace(predict_proba=lambda values: values),
                "feature_dimension": "1028",
            },
            "positive integer",
        ),
    ],
)
def test_load_classifier_rejects_invalid_payloads(
    monkeypatch,
    tmp_path,
    payload,
    message,
) -> None:
    model_path = tmp_path / "invalid.joblib"
    model_path.touch()
    monkeypatch.setattr(yamnet_filter.joblib, "load", lambda path: payload)

    with pytest.raises(ValueError, match=message):
        yamnet_filter.load_classifier(model_path)


def test_load_classifier_rejects_conflicting_feature_metadata(
    monkeypatch,
    tmp_path,
) -> None:
    model_path = tmp_path / "conflicting.joblib"
    model_path.touch()
    pipeline = SimpleNamespace(
        n_features_in_=1028,
        predict_proba=lambda values: np.empty((len(values), 2)),
    )
    monkeypatch.setattr(
        yamnet_filter.joblib,
        "load",
        lambda path: {
            "pipeline": pipeline,
            "feature_dimension": 1024,
        },
    )

    with pytest.raises(ValueError, match="payload declares 1024, model expects 1028"):
        yamnet_filter.load_classifier(model_path)


def test_filter_rejects_wrong_feature_dimension_before_prediction(
    monkeypatch,
    tmp_path,
) -> None:
    model_path = tmp_path / "classifier.joblib"
    model_path.touch()
    pipeline = SimpleNamespace(
        n_features_in_=4,
        predict_proba=_fail("classifier must not receive incompatible features"),
    )
    monkeypatch.setattr(
        yamnet_filter.joblib,
        "load",
        lambda path: {"pipeline": pipeline, "feature_dimension": 4},
    )
    monkeypatch.setattr(
        yamnet_filter,
        "extract_yamnet_candidates",
        lambda *args: [SimpleNamespace(onset_time=0.1)],
    )
    monkeypatch.setattr(
        yamnet_filter,
        "extract_features",
        lambda *args: np.array([[1.0, 2.0, 3.0]]),
    )

    with pytest.raises(ValueError, match="model expects 4, inference produced 3"):
        yamnet_filter.filter_onsets_with_yamnet(
            [0.1],
            "typing.wav",
            model_path,
        )


def test_filter_rejects_malformed_probability_output(monkeypatch, tmp_path) -> None:
    model_path = tmp_path / "classifier.joblib"
    model_path.touch()
    pipeline = SimpleNamespace(
        n_features_in_=2,
        predict_proba=lambda values: np.ones((len(values), 1)),
    )
    monkeypatch.setattr(
        yamnet_filter.joblib,
        "load",
        lambda path: {"pipeline": pipeline, "feature_dimension": 2},
    )
    monkeypatch.setattr(
        yamnet_filter,
        "extract_yamnet_candidates",
        lambda *args: [SimpleNamespace(onset_time=0.1)],
    )
    monkeypatch.setattr(
        yamnet_filter,
        "extract_features",
        lambda *args: np.array([[1.0, 2.0]]),
    )

    with pytest.raises(ValueError, match="at least two class probabilities"):
        yamnet_filter.filter_onsets_with_yamnet(
            [0.1],
            "typing.wav",
            model_path,
        )
