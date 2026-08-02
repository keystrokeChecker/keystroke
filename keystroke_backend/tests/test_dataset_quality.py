from __future__ import annotations

import json
import struct
import wave
from pathlib import Path

from record_dataset import derive_setup_id
from src.dataset_quality import validate_dataset, validate_recording
from src.evaluation_manifest import EvaluationManifest


CAPTURE = {
    "keyboard_id": "keyboard-a",
    "microphone_id": "microphone-a",
    "placement_id": "desk-left",
    "room_id": "room-a",
    "typist_id": "typist-a",
    "capture_batch_id": "batch-a",
}


def _write_wav(path: Path, *, amplitude: int = 1000, clipped: bool = False) -> None:
    sample_count = 4410
    value = 32767 if clipped else amplitude
    samples = [value if index % 2 else -value for index in range(sample_count)]
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(44100)
        wav_file.writeframes(struct.pack(f"<{sample_count}h", *samples))


def _recording(
    directory: Path,
    recording_id: str,
    split: str,
    fixture_type: str = "typing",
    *,
    amplitude: int = 1000,
    clipped: bool = False,
) -> dict[str, object]:
    setup_fields = {**CAPTURE, "capture_batch_id": f"batch-{recording_id}"}
    setup_id = derive_setup_id(setup_fields)
    wav_path = directory / f"{recording_id}.wav"
    meta_path = directory / f"{recording_id}_meta.json"
    _write_wav(wav_path, amplitude=amplitude, clipped=clipped)
    meta_path.write_text(
        json.dumps(
            {
                "recording_id": recording_id,
                "duration_seconds": 0.1,
                "sample_rate_hz": 44100,
                "sync_offset_ms": 0.0,
                "setup_id": setup_id,
                "intended_split": split,
                "fixture_type": fixture_type,
                **setup_fields,
            }
        ),
        encoding="utf-8",
    )
    value: dict[str, object] = {
        "id": recording_id,
        "wav": wav_path.name,
        "meta": meta_path.name,
        "split": split,
        "setup_id": setup_id,
        "silence": fixture_type == "silence",
        "non_keyboard": fixture_type == "non_keyboard",
    }
    if fixture_type == "typing":
        log_path = directory / f"{recording_id}_log.csv"
        log_path.write_text(
            "timestamp_sec,key,is_word_boundary\n"
            "0.01,a,false\n"
            "0.02,space,true\n"
            "0.04,b,false\n",
            encoding="utf-8",
        )
        value["log"] = log_path.name
    return value


def _manifest(directory: Path, recordings: list[dict[str, object]]) -> EvaluationManifest:
    path = directory / "manifest.json"
    path.write_text(
        json.dumps({"schema_version": 1, "recordings": recordings}),
        encoding="utf-8",
    )
    return EvaluationManifest.load(path)


def test_valid_dataset_reports_audio_truth_and_release_coverage(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        [
            _recording(tmp_path, "train-typing", "train"),
            _recording(tmp_path, "validation-typing", "validation"),
            _recording(tmp_path, "validation-silence", "validation", "silence", amplitude=0),
            _recording(tmp_path, "validation-background", "validation", "non_keyboard", amplitude=500),
            _recording(tmp_path, "test-typing", "test"),
            _recording(tmp_path, "test-silence", "test", "silence", amplitude=0),
            _recording(tmp_path, "test-background", "test", "non_keyboard", amplitude=500),
        ],
    )

    report = validate_dataset(manifest)

    assert report.valid
    assert report.valid_recording_count == 7
    assert report.split_counts == {"test": 3, "train": 1, "validation": 3}
    assert report.fixture_counts == {"non_keyboard": 2, "silence": 2, "typing": 3}
    assert report.recordings[0].event_count == 2
    assert report.recordings[0].word_count == 2
    assert report.recordings[2].audio is not None
    assert report.recordings[2].audio.near_silence
    assert report.to_dict()["valid"] is True


def test_recording_rejects_clipping_and_inconsistent_metadata(tmp_path: Path) -> None:
    raw = _recording(tmp_path, "clipped", "diagnostic", clipped=True)
    meta_path = tmp_path / "clipped_meta.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["duration_seconds"] = 9.0
    metadata["sample_rate_hz"] = 16000
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    recording = _manifest(tmp_path, [raw]).recordings[0]

    result = validate_recording(recording)
    codes = {issue.code for issue in result.issues}

    assert not result.valid
    assert {"wav_clipping", "metadata_duration_mismatch", "metadata_sample_rate"} <= codes


def test_dataset_distinguishes_diagnostic_quality_from_release_coverage(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [_recording(tmp_path, "diagnostic", "diagnostic")])

    report = validate_dataset(manifest)

    assert report.valid_recording_count == 1
    assert not report.valid
    assert {issue.code for issue in report.issues} == {
        "coverage_split_missing",
        "coverage_negative_missing",
    }


def test_setup_id_is_stable_and_changes_with_capture_batch() -> None:
    first = derive_setup_id(CAPTURE)
    reordered = derive_setup_id(dict(reversed(list(CAPTURE.items()))))
    changed = derive_setup_id({**CAPTURE, "capture_batch_id": "batch-b"})

    assert first == reordered
    assert first.startswith("setup-")
    assert first != changed
