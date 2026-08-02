"""Deterministic quality checks for manifest-managed recording datasets."""

from __future__ import annotations

import json
import math
import struct
import wave
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .evaluation import KeylogSchemaError, parse_keylog
from .evaluation_manifest import EvaluationManifest, RecordingEntry


REQUIRED_CAPTURE_FIELDS = (
    "keyboard_id",
    "microphone_id",
    "placement_id",
    "room_id",
    "typist_id",
    "capture_batch_id",
)


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    """Locked, unitless/default thresholds used by the quality CLI."""

    expected_channels: int = 1
    expected_sample_width_bytes: int = 2
    expected_sample_rate_hz: int = 44_100
    duration_tolerance_seconds: float = 0.25
    clipping_amplitude: int = 32_760
    maximum_clipping_ratio: float = 0.001
    near_silence_rms: float = 0.0001


@dataclass(frozen=True, slots=True)
class QualityIssue:
    code: str
    severity: str
    message: str


@dataclass(frozen=True, slots=True)
class AudioQuality:
    channels: int
    sample_width_bytes: int
    sample_rate_hz: int
    frame_count: int
    duration_seconds: float
    peak_normalized: float
    rms_normalized: float
    clipping_ratio: float
    near_silence: bool


@dataclass(frozen=True, slots=True)
class RecordingQuality:
    recording_id: str
    split: str
    setup_id: str
    fixture_type: str
    valid: bool
    audio: AudioQuality | None
    event_count: int | None
    word_count: int | None
    issues: tuple[QualityIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DatasetQualityReport:
    schema_version: int
    manifest: str | None
    valid: bool
    recording_count: int
    valid_recording_count: int
    split_counts: Mapping[str, int]
    fixture_counts: Mapping[str, int]
    setup_count: int
    issues: tuple[QualityIssue, ...]
    recordings: tuple[RecordingQuality, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _issue(code: str, message: str, *, warning: bool = False) -> QualityIssue:
    return QualityIssue(code, "warning" if warning else "error", message)


def _fixture_type(entry: RecordingEntry) -> str:
    if entry.silence and entry.non_keyboard:
        return "invalid_negative_flags"
    if entry.silence:
        return "silence"
    if entry.non_keyboard:
        return "non_keyboard"
    return "typing"


def _read_audio(path: Path, thresholds: QualityThresholds) -> AudioQuality:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getcomptype() != "NONE":
            raise ValueError("WAV must contain uncompressed PCM audio")
        channels = wav_file.getnchannels()
        width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)

    sample_count = len(raw) // 2
    if width == 1:
        sample_count = len(raw)
        samples: Iterable[int] = (value - 128 for value in raw)
        full_scale = 128
    elif width == 2:
        sample_count = len(raw) // 2
        samples = struct.unpack(f"<{sample_count}h", raw) if sample_count else ()
        full_scale = 32768
    elif width == 3:
        sample_count = len(raw) // 3

        def pcm24_values() -> Iterable[int]:
            for index in range(0, sample_count * 3, 3):
                value = int.from_bytes(raw[index : index + 3], "little", signed=False)
                yield value - (1 << 24) if value & (1 << 23) else value

        samples = pcm24_values()
        full_scale = 1 << 23
    elif width == 4:
        sample_count = len(raw) // 4
        samples = struct.unpack(f"<{sample_count}i", raw) if sample_count else ()
        full_scale = 1 << 31
    else:
        raise ValueError(f"unsupported PCM sample width: {width} bytes")
    absolute = [abs(sample) for sample in samples]
    peak = max(absolute, default=0)
    square_sum = sum(sample * sample for sample in absolute)
    clipping_count = sum(
        amplitude >= (thresholds.clipping_amplitude / 32768.0) * full_scale
        for amplitude in absolute
    )
    rms = math.sqrt(square_sum / sample_count) if sample_count else 0.0
    duration = frame_count / sample_rate if sample_rate else 0.0
    rms_normalized = rms / full_scale
    return AudioQuality(
        channels=channels,
        sample_width_bytes=width,
        sample_rate_hz=sample_rate,
        frame_count=frame_count,
        duration_seconds=duration,
        peak_normalized=peak / full_scale,
        rms_normalized=rms_normalized,
        clipping_ratio=clipping_count / sample_count if sample_count else 0.0,
        near_silence=rms_normalized <= thresholds.near_silence_rms,
    )


def _load_metadata(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"metadata is not valid JSON: {exc.msg}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("metadata root must be an object")
    return value


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def validate_recording(
    entry: RecordingEntry,
    *,
    thresholds: QualityThresholds = QualityThresholds(),
) -> RecordingQuality:
    """Validate one recording without modifying any source artifact."""

    issues: list[QualityIssue] = []
    fixture_type = _fixture_type(entry)
    if fixture_type == "invalid_negative_flags":
        issues.append(
            _issue(
                "fixture_flags_conflict",
                "a recording cannot be both silence and non_keyboard",
            )
        )

    audio: AudioQuality | None = None
    try:
        audio = _read_audio(entry.wav_path, thresholds)
    except (OSError, EOFError, ValueError, wave.Error) as exc:
        issues.append(_issue("wav_unreadable", str(exc)))

    if audio is not None:
        if audio.channels != thresholds.expected_channels:
            issues.append(
                _issue("wav_channels", f"expected mono audio; found {audio.channels} channels")
            )
        if audio.sample_width_bytes != thresholds.expected_sample_width_bytes:
            issues.append(
                _issue(
                    "wav_sample_width",
                    f"expected {thresholds.expected_sample_width_bytes * 8}-bit PCM; "
                    f"found {audio.sample_width_bytes * 8}-bit PCM",
                )
            )
        if audio.sample_rate_hz != thresholds.expected_sample_rate_hz:
            issues.append(
                _issue(
                    "wav_sample_rate",
                    f"expected {thresholds.expected_sample_rate_hz} Hz; "
                    f"found {audio.sample_rate_hz} Hz",
                )
            )
        if audio.frame_count == 0 or audio.duration_seconds <= 0:
            issues.append(_issue("wav_empty", "audio has no frames"))
        if audio.clipping_ratio > thresholds.maximum_clipping_ratio:
            issues.append(
                _issue(
                    "wav_clipping",
                    f"clipping ratio {audio.clipping_ratio:.6f} exceeds "
                    f"{thresholds.maximum_clipping_ratio:.6f}",
                )
            )
        if fixture_type == "typing" and audio.near_silence:
            issues.append(_issue("typing_near_silence", "typing audio is near-silent"))

    metadata: Mapping[str, object] | None = None
    if entry.meta_path is None:
        issues.append(_issue("metadata_missing", "capture metadata is required"))
    else:
        try:
            metadata = _load_metadata(entry.meta_path)
        except (OSError, ValueError) as exc:
            issues.append(_issue("metadata_invalid", str(exc)))

    sync_offset_ms: float | None = None
    if metadata is not None:
        recorded_id = metadata.get("recording_id", metadata.get("session_name"))
        if recorded_id != entry.id:
            issues.append(
                _issue(
                    "metadata_recording_id",
                    f"metadata recording ID {recorded_id!r} does not match {entry.id!r}",
                )
            )
        if metadata.get("setup_id") != entry.setup_id:
            issues.append(_issue("metadata_setup_id", "metadata setup_id does not match manifest"))
        if metadata.get("intended_split") != entry.split:
            issues.append(
                _issue("metadata_split", "metadata intended_split does not match manifest split")
            )
        if metadata.get("fixture_type") != fixture_type:
            issues.append(
                _issue("metadata_fixture_type", "metadata fixture_type does not match manifest flags")
            )
        for field in REQUIRED_CAPTURE_FIELDS:
            value = metadata.get(field)
            if not isinstance(value, str) or not value.strip():
                issues.append(_issue("metadata_capture_field", f"metadata {field} is required"))

        sync_offset_ms = _finite_number(metadata.get("sync_offset_ms"))
        if sync_offset_ms is None:
            issues.append(_issue("metadata_sync_offset", "sync_offset_ms must be finite"))
        metadata_rate = _finite_number(metadata.get("sample_rate_hz"))
        if audio is not None and metadata_rate != audio.sample_rate_hz:
            issues.append(
                _issue("metadata_sample_rate", "metadata sample_rate_hz does not match WAV")
            )
        metadata_duration = _finite_number(metadata.get("duration_seconds"))
        if metadata_duration is None or metadata_duration <= 0:
            issues.append(_issue("metadata_duration", "duration_seconds must be positive and finite"))
        elif audio is not None and abs(metadata_duration - audio.duration_seconds) > thresholds.duration_tolerance_seconds:
            issues.append(
                _issue(
                    "metadata_duration_mismatch",
                    f"metadata duration differs from WAV by "
                    f"{abs(metadata_duration - audio.duration_seconds):.3f}s",
                )
            )

    event_count: int | None = None
    word_count: int | None = None
    if entry.log_path is None:
        if fixture_type == "typing":
            issues.append(_issue("keylog_missing", "typing fixtures require a keylog"))
    else:
        try:
            truth = parse_keylog(entry.log_path, sync_offset_ms=sync_offset_ms or 0.0)
            event_count = len(truth.events)
            word_count = len(truth.words)
            if fixture_type == "typing" and (event_count == 0 or word_count == 0):
                issues.append(_issue("typing_truth_empty", "typing fixture has no countable key events"))
            if fixture_type != "typing" and event_count:
                issues.append(
                    _issue("negative_truth_events", "negative fixture contains countable key events")
                )
            if audio is not None and truth.events:
                if truth.events[0].timestamp_seconds < 0 or truth.events[-1].timestamp_seconds > audio.duration_seconds:
                    issues.append(_issue("keylog_out_of_bounds", "keylog events fall outside WAV duration"))
        except (OSError, KeylogSchemaError, ValueError) as exc:
            issues.append(_issue("keylog_invalid", str(exc)))

    return RecordingQuality(
        recording_id=entry.id,
        split=entry.split,
        setup_id=entry.setup_id,
        fixture_type=fixture_type,
        valid=not any(issue.severity == "error" for issue in issues),
        audio=audio,
        event_count=event_count,
        word_count=word_count,
        issues=tuple(issues),
    )


def validate_dataset(
    manifest: EvaluationManifest,
    *,
    thresholds: QualityThresholds = QualityThresholds(),
) -> DatasetQualityReport:
    """Validate recordings plus minimum release-evaluation fixture coverage."""

    recordings = tuple(
        validate_recording(entry, thresholds=thresholds)
        for entry in manifest.recordings
    )
    split_counts = Counter(recording.split for recording in recordings)
    fixture_counts = Counter(recording.fixture_type for recording in recordings)
    issues: list[QualityIssue] = []

    for split in ("train", "validation", "test"):
        selected = [recording for recording in recordings if recording.split == split]
        if not selected:
            issues.append(_issue("coverage_split_missing", f"{split} split is empty"))
        elif not any(recording.fixture_type == "typing" for recording in selected):
            issues.append(
                _issue("coverage_typing_missing", f"{split} split has no typing fixture")
            )

    for split in ("validation", "test"):
        split_types = {
            recording.fixture_type
            for recording in recordings
            if recording.split == split
        }
        for fixture_type in ("silence", "non_keyboard"):
            if fixture_type not in split_types:
                issues.append(
                    _issue(
                        "coverage_negative_missing",
                        f"{split} split has no {fixture_type} fixture",
                    )
                )

    if not recordings:
        issues.append(_issue("coverage_empty", "manifest contains no recordings"))

    valid_count = sum(recording.valid for recording in recordings)
    return DatasetQualityReport(
        schema_version=1,
        manifest=str(manifest.path) if manifest.path is not None else None,
        valid=valid_count == len(recordings)
        and not any(issue.severity == "error" for issue in issues),
        recording_count=len(recordings),
        valid_recording_count=valid_count,
        split_counts=dict(sorted(split_counts.items())),
        fixture_counts=dict(sorted(fixture_counts.items())),
        setup_count=len({recording.setup_id for recording in recordings}),
        issues=tuple(issues),
        recordings=recordings,
    )


__all__ = [
    "AudioQuality",
    "DatasetQualityReport",
    "QualityIssue",
    "QualityThresholds",
    "RecordingQuality",
    "REQUIRED_CAPTURE_FIELDS",
    "validate_dataset",
    "validate_recording",
]
