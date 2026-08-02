from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evaluate_methods import (
    EvaluationConfigurationError,
    run_evaluation,
)
from src.evaluation_manifest import TestContaminationError
from src.predictor import PredictionTrace


def _write_typing_fixture(directory: Path, recording_id: str) -> dict[str, str]:
    wav_path = directory / f"{recording_id}.wav"
    log_path = directory / f"{recording_id}_log.csv"
    meta_path = directory / f"{recording_id}_meta.json"
    wav_path.write_bytes(
        f"not-read-by-mocked-predictor:{recording_id}".encode("utf-8")
    )
    log_path.write_text(
        "timestamp_sec,key,is_word_boundary\n"
        f"0.10,{recording_id}-a,false\n"
        f"0.20,{recording_id}-b,false\n"
        "0.30,space,true\n"
        f"0.60,{recording_id}-c,false\n",
        encoding="utf-8",
    )
    meta_path.write_text(
        json.dumps(
            {
                "recording_id": recording_id,
                "sync_offset_ms": 100.0,
                "duration_seconds": 10.0,
            }
        ),
        encoding="utf-8",
    )
    return {
        "wav": wav_path.name,
        "log": log_path.name,
        "meta": meta_path.name,
    }


def _write_manifest(directory: Path, recordings: list[dict[str, object]]) -> Path:
    path = directory / "manifest.json"
    path.write_text(
        json.dumps({"schema_version": 1, "recordings": recordings}),
        encoding="utf-8",
    )
    return path


def _exact_typing_trace() -> PredictionTrace:
    return PredictionTrace(
        counts=(2, 1),
        counted_onsets=(0.20, 0.30, 0.70),
        word_groups=((0.20, 0.30), (0.70,)),
    )


def test_diagnostic_run_scores_typing_and_negative_fixtures_without_audio(
    tmp_path: Path,
) -> None:
    typing_files = _write_typing_fixture(tmp_path, "typing-diagnostic")
    negative_wav = tmp_path / "room-noise.wav"
    negative_wav.write_bytes(b"duration-is-injected:room-noise")
    negative_meta = tmp_path / "room-noise-meta.json"
    negative_meta.write_text(
        json.dumps({"duration_seconds": 999.0}),
        encoding="utf-8",
    )
    silence_wav = tmp_path / "quiet-room.wav"
    silence_wav.write_bytes(b"duration-is-injected:quiet-room")
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "id": "typing-diagnostic",
                **typing_files,
                "split": "diagnostic",
                "setup_id": "setup-typing",
                "tags": ["mocked"],
            },
            {
                "id": "room-noise",
                "wav": negative_wav.name,
                "meta": negative_meta.name,
                "split": "diagnostic",
                "setup_id": "setup-noise",
                "non_keyboard": True,
            },
            {
                "id": "quiet-room",
                "wav": silence_wav.name,
                "split": "diagnostic",
                "setup_id": "setup-silence",
                "silence": True,
            },
        ],
    )

    calls: list[tuple[str, dict[str, float]]] = []

    def fake_rule(wav_path: str, **kwargs: float) -> PredictionTrace:
        stem = Path(wav_path).stem
        calls.append((stem, kwargs))
        if stem == "typing-diagnostic":
            # One extra onset in the second word exercises event FP and count
            # substitution metrics while preserving a correct boundary.
            return PredictionTrace(
                counts=(2, 2),
                counted_onsets=(0.20, 0.30, 0.70, 0.75),
                word_groups=((0.20, 0.30), (0.70, 0.75)),
            )
        if stem == "room-noise":
            return PredictionTrace(
                counts=(1,),
                counted_onsets=(0.50,),
                word_groups=((0.50,),),
            )
        return PredictionTrace(counts=(), counted_onsets=(), word_groups=())

    def artifact_loader_must_not_run(path: str) -> object:
        raise AssertionError(f"rule-only run loaded an artifact: {path}")

    result = run_evaluation(
        manifest_path,
        split="diagnostic",
        methods=("rule",),
        predictors={"rule": fake_rule},
        artifact_loader=artifact_loader_must_not_run,
        duration_reader=lambda path: 30.0 if path.stem == "room-noise" else 60.0,
        generated_at="2026-07-28T12:00:00Z",
    )

    assert result["schema_version"] == 1
    assert result["generated_at"] == "2026-07-28T12:00:00Z"
    assert result["valid_for_release"] is False
    assert "not the untouched test split" in result["release_validation"][
        "reasons"
    ][0]
    assert result["hashes"]["manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert len(result["hashes"]["config_sha256"]) == 64
    expected_recording_hashes = {
        "typing-diagnostic": {
            "wav_sha256": hashlib.sha256(
                (tmp_path / typing_files["wav"]).read_bytes()
            ).hexdigest(),
            "log_sha256": hashlib.sha256(
                (tmp_path / typing_files["log"]).read_bytes()
            ).hexdigest(),
            "meta_sha256": hashlib.sha256(
                (tmp_path / typing_files["meta"]).read_bytes()
            ).hexdigest(),
        },
        "room-noise": {
            "wav_sha256": hashlib.sha256(negative_wav.read_bytes()).hexdigest(),
            "log_sha256": None,
            "meta_sha256": hashlib.sha256(negative_meta.read_bytes()).hexdigest(),
        },
        "quiet-room": {
            "wav_sha256": hashlib.sha256(silence_wav.read_bytes()).hexdigest(),
            "log_sha256": None,
            "meta_sha256": None,
        },
    }
    assert result["hashes"]["recording_files"] == expected_recording_hashes
    snapshot_payload = json.dumps(
        {
            "schema_version": 1,
            "recording_files": expected_recording_hashes,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert result["hashes"]["dataset_snapshot_sha256"] == hashlib.sha256(
        snapshot_payload
    ).hexdigest()
    assert result["hashes"]["artifacts"] == {}

    typing = result["recordings"][0]
    assert typing["ground_truth"]["counts"] == [2, 1]
    assert typing["ground_truth"]["event_times_seconds"] == pytest.approx(
        [0.20, 0.30, 0.70]
    )
    assert typing["ground_truth"]["boundary_times_seconds"] == pytest.approx(
        [0.40]
    )
    assert typing["ground_truth"]["sync_offset_seconds"] == pytest.approx(0.1)
    rule_metrics = typing["methods"]["rule"]
    assert rule_metrics["count_metrics"]["aligned_exact_accuracy"] == 0.5
    assert rule_metrics["count_metrics"]["substitutions"] == 1
    assert rule_metrics["event_metrics"]["precision"] == pytest.approx(0.75)
    assert rule_metrics["event_metrics"]["recall"] == 1.0
    assert rule_metrics["boundary_metrics"]["f1"] == 1.0

    negative = result["recordings"][1]["methods"]["rule"][
        "negative_fixture_metrics"
    ]
    assert negative["duration_seconds"] == 30.0
    assert negative["false_onsets_per_minute"] == 2.0
    assert negative["false_words_per_minute"] == 2.0
    assert negative["any_output"] is True
    assert negative["empty_sequence_exact"] is False

    silence = result["recordings"][2]["methods"]["rule"][
        "negative_fixture_metrics"
    ]
    assert silence["duration_seconds"] == 60.0
    assert silence["false_onsets_per_minute"] == 0.0
    assert silence["empty_sequence_exact"] is True

    aggregate = result["aggregates"]["rule"]
    assert aggregate["count_metrics"]["micro"][
        "aligned_exact_accuracy"
    ] == 0.5
    assert aggregate["event_metrics"]["micro"]["false_positives"] == 1
    assert aggregate["negative_fixture_metrics"]["non_keyboard"]["micro"][
        "false_onsets_per_minute"
    ] == 2.0
    assert aggregate["negative_fixture_metrics"]["silence"]["micro"][
        "false_onsets_per_minute"
    ] == 0.0
    assert aggregate["negative_fixture_metrics"]["combined"]["micro"][
        "false_onsets_per_minute"
    ] == pytest.approx(2.0 / 3.0)
    assert [stem for stem, _kwargs in calls] == [
        "typing-diagnostic",
        "room-noise",
        "quiet-room",
    ]
    assert calls[0][1] == {
        "delta": 0.07,
        "gap_threshold": 0.4,
        "merge_gap_seconds": 0.06,
    }
    json.dumps(result)


def test_selected_model_artifact_rejects_explicit_test_contamination(
    tmp_path: Path,
) -> None:
    typing_files = _write_typing_fixture(tmp_path, "future-test")
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "id": "future-test",
                **typing_files,
                "split": "test",
                "setup_id": "new-unseen-setup",
            }
        ],
    )
    artifact_path = tmp_path / "count-model.joblib"
    artifact_path.write_bytes(b"mock-count-model")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    predictor_called = False

    def predictor_must_not_run(*_args: object, **_kwargs: object) -> object:
        nonlocal predictor_called
        predictor_called = True
        raise AssertionError("prediction must not run after contamination")

    with pytest.raises(TestContaminationError, match="training=future-test"):
        run_evaluation(
            manifest_path,
            split="test",
            methods=("ml",),
            predictors={"ml": predictor_must_not_run},
            artifact_paths={"ml": artifact_path},
            artifact_loader=lambda _path: {
                "training_recording_ids": ["future-test"],
                "tuning_recording_ids": [],
                "manifest_hash": manifest_hash,
            },
        )
    assert predictor_called is False


def test_missing_model_provenance_marks_test_run_invalid_and_unselected_artifact_is_ignored(
    tmp_path: Path,
) -> None:
    typing_files = _write_typing_fixture(tmp_path, "future-clean-test")
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "id": "future-clean-test",
                **typing_files,
                "split": "test",
                "setup_id": "new-clean-setup",
            }
        ],
    )
    count_artifact = tmp_path / "count-model.joblib"
    count_artifact.write_bytes(b"selected-artifact-bytes")
    missing_yamnet_artifact = tmp_path / "must-not-be-loaded.joblib"
    loaded: list[str] = []

    def inspect_count_model(path: str) -> object:
        loaded.append(path)
        return {
            "training_recording_ids": ["some-training-recording"],
            "tuning_recording_ids": [],
            # No manifest_hash: enough to run diagnostics, not release evidence.
        }

    result = run_evaluation(
        manifest_path,
        split="test",
        methods=("ml",),
        predictors={"ml": lambda _path, **_kwargs: _exact_typing_trace()},
        artifact_paths={
            "ml": count_artifact,
            "yamnet": missing_yamnet_artifact,
        },
        artifact_loader=inspect_count_model,
        generated_at="2026-07-28T12:00:00Z",
    )

    assert loaded == [str(count_artifact.resolve())]
    assert result["valid_for_release"] is False
    assert any(
        "lacks a manifest hash" in reason
        for reason in result["release_validation"]["reasons"]
    )
    assert any(
        "training IDs are absent from the manifest train split" in reason
        for reason in result["release_validation"]["reasons"]
    )
    assert any(
        "has no tuning recording IDs" in reason
        for reason in result["release_validation"]["reasons"]
    )
    assert any(
        "no silence-fixture coverage" in reason
        for reason in result["release_validation"]["reasons"]
    )
    assert any(
        "no non-silence non-keyboard-fixture coverage" in reason
        for reason in result["release_validation"]["reasons"]
    )
    assert result["hashes"]["artifacts"]["count_predictor"] == hashlib.sha256(
        count_artifact.read_bytes()
    ).hexdigest()
    assert result["artifacts"]["count_predictor"]["provenance"][
        "complete"
    ] is False
    assert result["artifacts"]["count_predictor"]["provenance"][
        "training_ids_outside_train_split"
    ] == ["some-training-recording"]


def test_split_aligned_model_provenance_is_complete_but_test_doubles_are_invalid(
    tmp_path: Path,
) -> None:
    train_files = _write_typing_fixture(tmp_path, "model-train")
    validation_files = _write_typing_fixture(tmp_path, "model-validation")
    test_files = _write_typing_fixture(tmp_path, "model-test")
    silence_wav = tmp_path / "model-test-silence.wav"
    non_keyboard_wav = tmp_path / "model-test-background.wav"
    silence_wav.write_bytes(b"mock-silence")
    non_keyboard_wav.write_bytes(b"mock-background")
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "id": "model-train",
                **train_files,
                "split": "train",
                "setup_id": "setup-train",
            },
            {
                "id": "model-validation",
                **validation_files,
                "split": "validation",
                "setup_id": "setup-validation",
            },
            {
                "id": "model-test",
                **test_files,
                "split": "test",
                "setup_id": "setup-test",
            },
            {
                "id": "model-test-silence",
                "wav": silence_wav.name,
                "split": "test",
                "setup_id": "setup-test-silence",
                "silence": True,
            },
            {
                "id": "model-test-background",
                "wav": non_keyboard_wav.name,
                "split": "test",
                "setup_id": "setup-test-background",
                "non_keyboard": True,
            },
        ],
    )
    artifact_path = tmp_path / "complete-count-model.joblib"
    artifact_path.write_bytes(b"complete-count-model")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    def fake_ml(wav_path: str, **_kwargs: float) -> PredictionTrace:
        if Path(wav_path).stem == "model-test":
            return _exact_typing_trace()
        return PredictionTrace(counts=(), counted_onsets=(), word_groups=())

    unlocked_result = run_evaluation(
        manifest_path,
        split="test",
        methods=("ml",),
        predictors={"ml": fake_ml},
        artifact_paths={"ml": artifact_path},
        artifact_loader=lambda _path: {
            "training_recording_ids": ["model-train"],
            "tuning_recording_ids": ["model-validation"],
            "manifest_hash": f"sha256:{manifest_hash}",
        },
        duration_reader=lambda _path: 60.0,
    )
    result = run_evaluation(
        manifest_path,
        split="test",
        methods=("ml",),
        predictors={"ml": fake_ml},
        artifact_paths={"ml": artifact_path},
        artifact_loader=lambda _path: {
            "training_recording_ids": ["model-train"],
            "tuning_recording_ids": ["model-validation"],
            "manifest_hash": f"sha256:{manifest_hash}",
        },
        duration_reader=lambda _path: 60.0,
        locked_config_sha256=unlocked_result["hashes"][
            "configuration_lock_sha256"
        ],
    )

    assert result["valid_for_release"] is False
    assert result["release_validation"]["configuration_lock"]["matches"] is True
    assert not any(
        "missing the validation-locked" in reason
        for reason in result["release_validation"]["reasons"]
    )
    assert any(
        "injected predictor" in reason
        for reason in result["release_validation"]["reasons"]
    )
    assert "acceptance thresholds are evaluated separately" in result[
        "release_validation"
    ]["meaning"]
    provenance = result["artifacts"]["count_predictor"]["provenance"]
    assert provenance["complete"] is True
    assert provenance["training_ids_outside_train_split"] == []
    assert provenance["tuning_ids_outside_validation_split"] == []


def test_rule_test_requires_locked_config_and_real_dependencies_and_rejects_legacy(
    tmp_path: Path,
) -> None:
    clean_files = _write_typing_fixture(tmp_path, "release-recording")
    silence_wav = tmp_path / "release-silence.wav"
    non_keyboard_wav = tmp_path / "release-non-keyboard.wav"
    silence_wav.write_bytes(b"mock-silence")
    non_keyboard_wav.write_bytes(b"mock-non-keyboard")
    clean_manifest = _write_manifest(
        tmp_path,
        [
            {
                "id": "release-recording",
                **clean_files,
                "split": "test",
                "setup_id": "untouched-setup",
            },
            {
                "id": "release-silence",
                "wav": silence_wav.name,
                "split": "test",
                "setup_id": "untouched-silence-setup",
                "silence": True,
            },
            {
                "id": "release-non-keyboard",
                "wav": non_keyboard_wav.name,
                "split": "test",
                "setup_id": "untouched-negative-setup",
                "non_keyboard": True,
            },
        ],
    )

    def clean_rule_predictor(wav_path: str, **_kwargs: float) -> PredictionTrace:
        if Path(wav_path).stem == "release-recording":
            return _exact_typing_trace()
        return PredictionTrace(counts=(), counted_onsets=(), word_groups=())

    unlocked_result = run_evaluation(
        clean_manifest,
        split="test",
        methods=("rule",),
        predictors={"rule": clean_rule_predictor},
        artifact_loader=lambda path: pytest.fail(
            f"rule run unexpectedly loaded {path}"
        ),
        duration_reader=lambda _path: 60.0,
    )
    assert any(
        "missing the validation-locked" in reason
        for reason in unlocked_result["release_validation"]["reasons"]
    )

    clean_result = run_evaluation(
        clean_manifest,
        split="test",
        methods=("rule",),
        predictors={"rule": clean_rule_predictor},
        artifact_loader=lambda path: pytest.fail(
            f"rule run unexpectedly loaded {path}"
        ),
        duration_reader=lambda _path: 60.0,
        locked_config_sha256=unlocked_result["hashes"][
            "configuration_lock_sha256"
        ],
    )
    assert clean_result["valid_for_release"] is False
    assert clean_result["release_validation"]["configuration_lock"]["matches"] is True
    assert any(
        "injected predictor" in reason
        for reason in clean_result["release_validation"]["reasons"]
    )

    changed_metric_result = run_evaluation(
        clean_manifest,
        split="test",
        methods=("rule",),
        predictors={"rule": clean_rule_predictor},
        artifact_loader=lambda path: pytest.fail(
            f"rule run unexpectedly loaded {path}"
        ),
        duration_reader=lambda _path: 60.0,
        event_tolerance_seconds=0.05,
        locked_config_sha256=unlocked_result["hashes"][
            "configuration_lock_sha256"
        ],
    )
    assert any(
        "locked 0.08-second event tolerance" in reason
        for reason in changed_metric_result["release_validation"]["reasons"]
    )
    assert any(
        "does not match the locked hash" in reason
        for reason in changed_metric_result["release_validation"]["reasons"]
    )

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    legacy_files = _write_typing_fixture(legacy_dir, "session1")
    legacy_manifest = _write_manifest(
        legacy_dir,
        [
            {
                "id": "session1",
                **legacy_files,
                "split": "test",
                "setup_id": "legacy-setup",
            }
        ],
    )
    with pytest.raises(
        EvaluationConfigurationError,
        match="may only use split='diagnostic': session1",
    ):
        run_evaluation(
            legacy_manifest,
            split="test",
            methods=("rule",),
            predictors={"rule": lambda _path, **_kwargs: _exact_typing_trace()},
        )


def test_ground_truth_predictor_and_trace_errors_have_evaluation_context(
    tmp_path: Path,
) -> None:
    bad_truth_dir = tmp_path / "bad-truth"
    bad_truth_dir.mkdir()
    bad_truth_files = _write_typing_fixture(bad_truth_dir, "bad-truth-recording")
    (bad_truth_dir / bad_truth_files["log"]).write_text(
        "timestamp_sec,key,is_word_boundary\n0.1,a,not-a-boolean\n",
        encoding="utf-8",
    )
    bad_truth_manifest = _write_manifest(
        bad_truth_dir,
        [
            {
                "id": "bad-truth-recording",
                **bad_truth_files,
                "split": "diagnostic",
                "setup_id": "bad-truth-setup",
            }
        ],
    )
    with pytest.raises(
        EvaluationConfigurationError,
        match=r"Ground-truth parsing failed for recording 'bad-truth-recording'.*KeylogSchemaError",
    ):
        run_evaluation(
            bad_truth_manifest,
            methods=("rule",),
            predictors={"rule": lambda _path, **_kwargs: _exact_typing_trace()},
        )

    predictor_dir = tmp_path / "bad-predictor"
    predictor_dir.mkdir()
    predictor_files = _write_typing_fixture(
        predictor_dir, "bad-predictor-recording"
    )
    predictor_manifest = _write_manifest(
        predictor_dir,
        [
            {
                "id": "bad-predictor-recording",
                **predictor_files,
                "split": "diagnostic",
                "setup_id": "bad-predictor-setup",
            }
        ],
    )

    def exploding_predictor(_path: str, **_kwargs: float) -> PredictionTrace:
        raise RuntimeError("mock inference failure")

    with pytest.raises(
        EvaluationConfigurationError,
        match=(
            r"recording 'bad-predictor-recording' with method 'rule': "
            r"RuntimeError: mock inference failure"
        ),
    ):
        run_evaluation(
            predictor_manifest,
            methods=("rule",),
            predictors={"rule": exploding_predictor},
        )

    malformed_trace = PredictionTrace(
        counts=(1,),
        counted_onsets=(),
        word_groups=(),
    )
    with pytest.raises(
        EvaluationConfigurationError,
        match=(
            r"recording 'bad-predictor-recording' with method 'rule'.*"
            r"has 1 counts but 0 word groups"
        ),
    ):
        run_evaluation(
            predictor_manifest,
            methods=("rule",),
            predictors={"rule": lambda _path, **_kwargs: malformed_trace},
        )


def test_duplicate_resolved_path_across_splits_is_rejected_before_prediction(
    tmp_path: Path,
) -> None:
    shared_wav = tmp_path / "shared.wav"
    shared_wav.write_bytes(b"one-physical-recording")
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "id": "training-copy",
                "wav": shared_wav.name,
                "split": "train",
                "setup_id": "training-setup",
                "silence": True,
            },
            {
                "id": "test-alias",
                "wav": str(tmp_path / "." / shared_wav.name),
                "split": "test",
                "setup_id": "test-setup",
                "silence": True,
            },
        ],
    )
    predictor_called = False

    def predictor_must_not_run(*_args: object, **_kwargs: object) -> object:
        nonlocal predictor_called
        predictor_called = True
        raise AssertionError("prediction must not run after preflight failure")

    with pytest.raises(
        EvaluationConfigurationError,
        match=(
            r"Duplicate resolved file path.*training-copy.*train.*wav.*"
            r"test-alias.*test.*wav"
        ),
    ):
        run_evaluation(
            manifest_path,
            split="test",
            methods=("rule",),
            predictors={"rule": predictor_must_not_run},
        )
    assert predictor_called is False


@pytest.mark.parametrize("duplicate_role", ["wav", "log", "meta"])
def test_duplicate_file_contents_across_splits_are_rejected_before_prediction(
    tmp_path: Path,
    duplicate_role: str,
) -> None:
    train_files = _write_typing_fixture(tmp_path, "content-train")
    test_files = _write_typing_fixture(tmp_path, "content-test")
    train_source = tmp_path / train_files[duplicate_role]
    copied_path = tmp_path / f"copied-test-{duplicate_role}{train_source.suffix}"
    copied_path.write_bytes(train_source.read_bytes())
    test_files[duplicate_role] = copied_path.name
    manifest_path = _write_manifest(
        tmp_path,
        [
            {
                "id": "content-train",
                **train_files,
                "split": "train",
                "setup_id": "content-training-setup",
            },
            {
                "id": "content-test",
                **test_files,
                "split": "test",
                "setup_id": "content-test-setup",
            },
        ],
    )
    predictor_called = False

    def predictor_must_not_run(*_args: object, **_kwargs: object) -> object:
        nonlocal predictor_called
        predictor_called = True
        raise AssertionError("prediction must not run after preflight failure")

    with pytest.raises(
        EvaluationConfigurationError,
        match=(
            rf"Duplicate {duplicate_role} file contents.*content-train.*train.*"
            rf"content-test.*test"
        ),
    ):
        run_evaluation(
            manifest_path,
            split="test",
            methods=("rule",),
            predictors={"rule": predictor_must_not_run},
        )
    assert predictor_called is False
