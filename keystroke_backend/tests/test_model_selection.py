from __future__ import annotations

import hashlib
import json

import pytest

from src.model_selection import ModelSelectionError, select_production_method


def _aggregate(*, accuracy: float, silence_false_onsets: int = 0) -> dict:
    return {
        "count_metrics": {
            "micro": {
                "recording_count": 2,
                "aligned_exact_accuracy": accuracy,
                "full_sequence_exact_rate": accuracy - 0.1,
                "gap_penalized_mae": 1.0 - accuracy,
            }
        },
        "event_metrics": {"micro": {"f1": accuracy - 0.05}},
        "boundary_metrics": {"micro": {"f1": accuracy - 0.02}},
        "negative_fixture_metrics": {
            "silence": {
                "micro": {
                    "recording_count": 1,
                    "false_onset_count": silence_false_onsets,
                }
            },
            "non_keyboard": {
                "micro": {
                    "recording_count": 1,
                    "false_onsets_per_minute": 0.5,
                }
            },
        },
    }


def _result() -> dict:
    return {
        "schema_version": 1,
        "evaluator": "production-parity-manifest",
        "execution": {
            "canonical_predictors": True,
            "canonical_artifact_loader": True,
            "canonical_duration_reader": True,
            "custom_artifact_methods": ["ml", "yamnet"],
        },
        "manifest": {"selected_split": "validation"},
        "config": {
            "split": "validation",
            "methods": ["rule", "ml", "yamnet"],
            "event_tolerance_seconds": 0.08,
            "boundary_slack_seconds": 0.08,
            "serving": {
                "rule": {"delta": 0.07},
                "ml": {"delta": 0.07},
                "yamnet": {"delta": 0.07, "threshold": 0.5},
            },
        },
        "hashes": {
            "manifest_sha256": "a" * 64,
            "dataset_snapshot_sha256": "b" * 64,
            "config_sha256": "c" * 64,
        },
        "artifacts": {
            "ml": {"sha256": "d" * 64, "provenance": {"complete": True}},
            "yamnet": {"sha256": "e" * 64, "provenance": {"complete": True}},
        },
        "aggregates": {
            "rule": _aggregate(accuracy=0.80),
            "ml": _aggregate(accuracy=0.95, silence_false_onsets=1),
            "yamnet": _aggregate(accuracy=0.90),
        },
    }


def test_selects_best_safe_method_and_builds_single_method_lock() -> None:
    selection = select_production_method(_result())

    assert selection["selected_method"] == "yamnet"
    assert selection["status"] == "validation_locked"
    locked = selection["locked_configuration"]
    assert locked["methods"] == ["yamnet"]
    assert locked["serving"] == {"yamnet": {"delta": 0.07, "threshold": 0.5}}
    assert locked["artifact_sha256"] == {"yamnet": "e" * 64}
    assert "split" not in locked
    expected = hashlib.sha256(
        json.dumps(
            locked, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    assert selection["configuration_lock_sha256"] == expected
    ml = next(item for item in selection["candidates"] if item["method"] == "ml")
    assert not ml["eligible"]
    assert "produced detections on validation silence" in ml["reasons"]


def test_rejects_test_results_and_requires_multiple_methods() -> None:
    result = _result()
    result["manifest"]["selected_split"] = "test"
    with pytest.raises(ModelSelectionError, match="validation-split"):
        select_production_method(result)

    result = _result()
    result["config"]["methods"] = ["rule"]
    with pytest.raises(ModelSelectionError, match="at least two"):
        select_production_method(result)


def test_rejects_when_every_candidate_fails_safety_gate() -> None:
    result = _result()
    for aggregate in result["aggregates"].values():
        aggregate["negative_fixture_metrics"]["silence"]["micro"][
            "false_onset_count"
        ] = 1

    with pytest.raises(ModelSelectionError, match="no method"):
        select_production_method(result)
