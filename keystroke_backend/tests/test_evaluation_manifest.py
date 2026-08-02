from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import pytest

from src.evaluation_manifest import (
    MANIFEST_SCHEMA_VERSION,
    EvaluationManifest,
    ManifestValidationError,
    ModelMetadata,
    TestContaminationError,
    load_evaluation_manifest,
    load_model_metadata,
)


def _create_typing_files(directory: Path, recording_id: str) -> dict[str, str]:
    paths = {
        "wav": directory / f"{recording_id}.wav",
        "log": directory / f"{recording_id}_log.csv",
        "meta": directory / f"{recording_id}_meta.json",
    }
    for path in paths.values():
        path.touch()
    return {name: path.name for name, path in paths.items()}


def _typing_recording(
    directory: Path,
    recording_id: str,
    split: str,
    setup_id: str,
    **extra: object,
) -> dict[str, object]:
    return {
        "id": recording_id,
        **_create_typing_files(directory, recording_id),
        "split": split,
        "setup_id": setup_id,
        **extra,
    }


def _write_manifest(directory: Path, recordings: list[dict[str, object]]) -> Path:
    path = directory / "evaluation_manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "recordings": recordings,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_load_resolves_paths_and_selects_splits_in_manifest_order(tmp_path) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        [
            _typing_recording(
                tmp_path,
                "train-a",
                "train",
                "setup-train",
                tags=["mechanical", "phone-mic"],
            ),
            _typing_recording(tmp_path, "test-a", "test", "setup-test-a"),
            _typing_recording(tmp_path, "test-b", "test", "setup-test-b"),
        ],
    )

    manifest = load_evaluation_manifest(manifest_path)

    assert manifest.schema_version == 1
    assert manifest.path == manifest_path.resolve()
    assert [entry.id for entry in manifest.select_split("test")] == [
        "test-a",
        "test-b",
    ]
    assert manifest.for_split("validation") == ()
    assert manifest.recording_ids("train") == frozenset({"train-a"})

    train = manifest.select_split("train")[0]
    assert train.wav_path == (tmp_path / "train-a.wav").resolve()
    assert train.log_path == (tmp_path / "train-a_log.csv").resolve()
    assert train.meta_path == (tmp_path / "train-a_meta.json").resolve()
    assert train.wav == train.wav_path
    assert train.log == train.log_path
    assert train.meta == train.meta_path
    assert train.tags == ("mechanical", "phone-mic")
    assert train.non_keyboard is False
    assert train.silence is False


@pytest.mark.parametrize("flag", ["silence", "non_keyboard"])
def test_negative_fixture_requires_only_wav_and_validates_supplied_paths(
    tmp_path,
    flag,
) -> None:
    wav_path = tmp_path / f"{flag}.wav"
    wav_path.touch()
    recording = {
        "id": flag,
        "wav": wav_path.name,
        "split": "diagnostic",
        "setup_id": f"setup-{flag}",
        flag: True,
    }

    manifest = EvaluationManifest.load(_write_manifest(tmp_path, [recording]))

    entry = manifest.recordings[0]
    assert entry.is_negative_fixture is True
    assert entry.log_path is None
    assert entry.meta_path is None

    recording["meta"] = "missing-meta.json"
    with pytest.raises(ManifestValidationError, match="meta does not exist"):
        EvaluationManifest.load(_write_manifest(tmp_path, [recording]))


def test_typing_recording_requires_log_and_meta(tmp_path) -> None:
    wav_path = tmp_path / "typing.wav"
    wav_path.touch()
    recording = {
        "id": "typing",
        "wav": wav_path.name,
        "split": "train",
        "setup_id": "setup-a",
    }

    with pytest.raises(ManifestValidationError, match=r"recordings\[0\]\.log is required"):
        EvaluationManifest.load(_write_manifest(tmp_path, [recording]))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "unsupported schema_version"),
        ("schema_version", "1", "schema_version must be an integer"),
        ("split", "holdout", "split must be one of"),
    ],
)
def test_rejects_unsupported_schema_or_split(tmp_path, field, value, message) -> None:
    recording = _typing_recording(tmp_path, "typing", "train", "setup-a")
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "recordings": [recording],
    }
    if field == "split":
        recording[field] = value
    else:
        payload[field] = value

    with pytest.raises(ManifestValidationError, match=message):
        EvaluationManifest.from_mapping(payload, base_dir=tmp_path)


def test_rejects_duplicate_recording_ids(tmp_path) -> None:
    first = _typing_recording(tmp_path, "same", "train", "setup-a")
    second = {
        **first,
        "split": "train",
        "setup_id": "setup-b",
    }

    with pytest.raises(ManifestValidationError, match="duplicate recording id 'same'"):
        EvaluationManifest.from_mapping(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "recordings": [first, second],
            },
            base_dir=tmp_path,
        )


def test_rejects_setup_crossing_leakage_controlled_splits(tmp_path) -> None:
    train = _typing_recording(tmp_path, "train-a", "train", "shared-setup")
    test = _typing_recording(tmp_path, "test-a", "test", "shared-setup")

    with pytest.raises(
        ManifestValidationError,
        match=r"'shared-setup': test, train",
    ):
        EvaluationManifest.from_mapping(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "recordings": [train, test],
            },
            base_dir=tmp_path,
        )


def test_diagnostic_recording_may_reuse_a_controlled_setup(tmp_path) -> None:
    train = _typing_recording(tmp_path, "train-a", "train", "shared-setup")
    diagnostic = _typing_recording(
        tmp_path,
        "diagnostic-a",
        "diagnostic",
        "shared-setup",
    )

    manifest = EvaluationManifest.from_mapping(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "recordings": [train, diagnostic],
        },
        base_dir=tmp_path,
    )

    assert manifest.recording_ids() == frozenset({"train-a", "diagnostic-a"})


def test_load_model_metadata_accepts_generic_and_legacy_mapping_shapes() -> None:
    payload = MappingProxyType(
        {
            "sessions": ["legacy-train"],
            "metadata": {
                "training_recording_ids": ["train-a", "train-b"],
                "manifest_hash": "sha256:abc123",
            },
            "tuning": {"session_ids": ("validation-a", "test-b")},
        }
    )

    metadata = load_model_metadata(payload)

    assert metadata.training_recording_ids == frozenset(
        {"legacy-train", "train-a", "train-b"}
    )
    assert metadata.training_session_ids == metadata.training_recording_ids
    assert metadata.tuning_recording_ids == frozenset({"validation-a", "test-b"})
    assert metadata.tuning_session_ids == metadata.tuning_recording_ids
    assert metadata.manifest_hash == "sha256:abc123"


def test_detects_training_and_tuning_contamination_against_test_only(tmp_path) -> None:
    recordings = [
        _typing_recording(tmp_path, "train-a", "train", "setup-train"),
        _typing_recording(tmp_path, "validation-a", "validation", "setup-validation"),
        _typing_recording(tmp_path, "test-a", "test", "setup-test-a"),
        _typing_recording(tmp_path, "test-b", "test", "setup-test-b"),
    ]
    manifest = EvaluationManifest.load(_write_manifest(tmp_path, recordings))

    report = manifest.detect_test_contamination(
        {
            "training_session_ids": ["train-a", "test-a"],
            "tuning_recording_ids": ["validation-a", "test-b"],
        }
    )

    assert report.training_test_ids == frozenset({"test-a"})
    assert report.tuning_test_ids == frozenset({"test-b"})
    assert report.contaminated_ids == frozenset({"test-a", "test-b"})
    assert report.is_contaminated is True
    assert bool(report) is True

    with pytest.raises(
        TestContaminationError,
        match=r"training=test-a; tuning=test-b",
    ):
        manifest.assert_no_test_contamination(
            ModelMetadata(
                training_recording_ids=frozenset({"train-a", "test-a"}),
                tuning_recording_ids=frozenset({"test-b"}),
            )
        )


def test_uncontaminated_metadata_passes_and_invalid_id_lists_fail(tmp_path) -> None:
    manifest = EvaluationManifest.load(
        _write_manifest(
            tmp_path,
            [_typing_recording(tmp_path, "test-a", "test", "setup-test")],
        )
    )

    metadata = {"training_recording_ids": ["train-a"], "tuning_sessions": []}
    manifest.assert_no_test_contamination(metadata)
    assert manifest.detect_contamination(metadata).is_contaminated is False

    with pytest.raises(ManifestValidationError, match="must be an array"):
        load_model_metadata({"training_recording_ids": "test-a"})


def test_file_validation_can_be_disabled_for_schema_only_checks(tmp_path) -> None:
    manifest = EvaluationManifest.from_mapping(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "recordings": [
                {
                    "id": "future-recording",
                    "wav": "private/future.wav",
                    "log": "private/future_log.csv",
                    "meta": "private/future_meta.json",
                    "split": "test",
                    "setup_id": "future-setup",
                }
            ],
        },
        base_dir=tmp_path,
        validate_files=False,
    )

    assert manifest.recordings[0].wav_path == (
        tmp_path / "private" / "future.wav"
    ).resolve()

