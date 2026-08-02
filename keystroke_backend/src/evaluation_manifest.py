"""Leakage-safe recording manifests for production evaluation.

The on-disk JSON schema is deliberately small and versioned::

    {
      "schema_version": 1,
      "recordings": [
        {
          "id": "typing_001",
          "wav": "data/typing_001.wav",
          "log": "data/typing_001_log.csv",
          "meta": "data/typing_001_meta.json",
          "split": "train",
          "setup_id": "keyboard-a__phone-a__room-a__batch-1",
          "tags": ["mechanical"],
          "non_keyboard": false,
          "silence": false
        }
      ]
    }

``wav`` is required for every recording.  ``log`` and ``meta`` are required
for typing recordings, but may be omitted for explicitly marked silence or
non-keyboard negative fixtures.  Every supplied path is still validated.
Relative paths are resolved from the directory containing the manifest.

This module uses only standard-library mappings and paths.  In particular,
model provenance validation does not load a joblib artifact; callers pass the
already-inspected metadata mapping to :func:`load_model_metadata`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


MANIFEST_SCHEMA_VERSION = 1
SUPPORTED_SPLITS = frozenset({"train", "validation", "test", "diagnostic"})
LEAKAGE_CONTROLLED_SPLITS = frozenset({"train", "validation", "test"})


class ManifestValidationError(ValueError):
    """Raised when an evaluation manifest does not satisfy its schema."""


class TestContaminationError(ManifestValidationError):
    """Raised when test recordings appear in model fit or tuning metadata."""

    __test__ = False


def _non_empty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ManifestValidationError(f"{field} must be a boolean")
    return value


def _tags(value: object, *, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ManifestValidationError(f"{field} must be an array of strings")

    parsed: list[str] = []
    seen: set[str] = set()
    for index, tag in enumerate(value):
        parsed_tag = _non_empty_string(tag, field=f"{field}[{index}]")
        if parsed_tag in seen:
            raise ManifestValidationError(
                f"{field} contains duplicate tag {parsed_tag!r}"
            )
        seen.add(parsed_tag)
        parsed.append(parsed_tag)
    return tuple(parsed)


def _resolve_path(
    value: object,
    *,
    field: str,
    base_dir: Path,
    required: bool,
    validate_files: bool,
) -> Path | None:
    if value is None:
        if required:
            raise ManifestValidationError(f"{field} is required")
        return None

    path_text = _non_empty_string(value, field=field)
    path = Path(path_text)
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()

    if validate_files:
        if not path.exists():
            raise ManifestValidationError(f"{field} does not exist: {path}")
        if not path.is_file():
            raise ManifestValidationError(f"{field} is not a file: {path}")
    return path


@dataclass(frozen=True, slots=True)
class RecordingEntry:
    """One whole-recording unit from an evaluation manifest."""

    id: str
    wav_path: Path
    log_path: Path | None
    meta_path: Path | None
    split: str
    setup_id: str
    tags: tuple[str, ...] = ()
    non_keyboard: bool = False
    silence: bool = False

    @property
    def wav(self) -> Path:
        """Alias matching the JSON field name."""
        return self.wav_path

    @property
    def log(self) -> Path | None:
        """Alias matching the JSON field name."""
        return self.log_path

    @property
    def meta(self) -> Path | None:
        """Alias matching the JSON field name."""
        return self.meta_path

    @property
    def is_negative_fixture(self) -> bool:
        return self.non_keyboard or self.silence

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        base_dir: Path,
        index: int,
        validate_files: bool = True,
    ) -> RecordingEntry:
        """Parse and validate one recording object."""
        if not isinstance(value, Mapping):
            raise ManifestValidationError(f"recordings[{index}] must be an object")

        prefix = f"recordings[{index}]"
        recording_id = _non_empty_string(value.get("id"), field=f"{prefix}.id")
        split = _non_empty_string(value.get("split"), field=f"{prefix}.split")
        if split not in SUPPORTED_SPLITS:
            supported = ", ".join(sorted(SUPPORTED_SPLITS))
            raise ManifestValidationError(
                f"{prefix}.split must be one of {supported}; got {split!r}"
            )

        setup_id = _non_empty_string(
            value.get("setup_id"), field=f"{prefix}.setup_id"
        )
        non_keyboard = _optional_boolean(
            value.get("non_keyboard", False), field=f"{prefix}.non_keyboard"
        )
        silence = _optional_boolean(
            value.get("silence", False), field=f"{prefix}.silence"
        )
        tags = _tags(value.get("tags", []), field=f"{prefix}.tags")

        negative_fixture = non_keyboard or silence
        wav_path = _resolve_path(
            value.get("wav"),
            field=f"{prefix}.wav",
            base_dir=base_dir,
            required=True,
            validate_files=validate_files,
        )
        log_path = _resolve_path(
            value.get("log"),
            field=f"{prefix}.log",
            base_dir=base_dir,
            required=not negative_fixture,
            validate_files=validate_files,
        )
        meta_path = _resolve_path(
            value.get("meta"),
            field=f"{prefix}.meta",
            base_dir=base_dir,
            required=not negative_fixture,
            validate_files=validate_files,
        )
        assert wav_path is not None  # required above; narrows the type checker

        return cls(
            id=recording_id,
            wav_path=wav_path,
            log_path=log_path,
            meta_path=meta_path,
            split=split,
            setup_id=setup_id,
            tags=tags,
            non_keyboard=non_keyboard,
            silence=silence,
        )


# A descriptive alias for callers that prefer the longer schema-oriented name.
RecordingManifestEntry = RecordingEntry


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Recording provenance extracted from a generic model metadata mapping."""

    training_recording_ids: frozenset[str] = frozenset()
    tuning_recording_ids: frozenset[str] = frozenset()
    manifest_hash: str | None = None

    @property
    def training_session_ids(self) -> frozenset[str]:
        """Backward-compatible terminology used by older model artifacts."""
        return self.training_recording_ids

    @property
    def tuning_session_ids(self) -> frozenset[str]:
        """Backward-compatible terminology used by older model artifacts."""
        return self.tuning_recording_ids

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ModelMetadata:
        if not isinstance(value, Mapping):
            raise ManifestValidationError("model metadata must be a mapping")

        training: set[str] = set()
        tuning: set[str] = set()
        manifest_hash: str | None = None

        # Artifacts in this repository historically stored ``sessions`` at the
        # payload root.  New artifacts should use the explicit recording-ID
        # names.  Other aliases make this reader independent of one trainer.
        training_keys = (
            "training_recording_ids",
            "train_recording_ids",
            "training_session_ids",
            "train_session_ids",
            "training_sessions",
            "train_sessions",
            "sessions",
        )
        tuning_keys = (
            "tuning_recording_ids",
            "tune_recording_ids",
            "validation_recording_ids",
            "tuning_session_ids",
            "tune_session_ids",
            "validation_session_ids",
            "tuning_sessions",
            "validation_sessions",
        )

        def add_ids(
            target: set[str], raw_ids: object, *, field: str
        ) -> None:
            if raw_ids is None:
                return
            if (
                isinstance(raw_ids, (str, bytes, Mapping))
                or not isinstance(raw_ids, Iterable)
            ):
                raise ManifestValidationError(
                    f"model metadata {field} must be an array of recording IDs"
                )
            for index, raw_id in enumerate(raw_ids):
                target.add(
                    _non_empty_string(
                        raw_id, field=f"model metadata {field}[{index}]"
                    )
                )

        def inspect(mapping: Mapping[str, object], *, location: str) -> None:
            nonlocal manifest_hash
            for key in training_keys:
                if key in mapping:
                    add_ids(training, mapping[key], field=f"{location}{key}")
            for key in tuning_keys:
                if key in mapping:
                    add_ids(tuning, mapping[key], field=f"{location}{key}")

            if "manifest_hash" in mapping and mapping["manifest_hash"] is not None:
                parsed_hash = _non_empty_string(
                    mapping["manifest_hash"],
                    field=f"model metadata {location}manifest_hash",
                )
                if manifest_hash is not None and parsed_hash != manifest_hash:
                    raise ManifestValidationError(
                        "model metadata contains conflicting manifest_hash values"
                    )
                manifest_hash = parsed_hash

            # Common generic wrappers are inspected without importing or
            # depending on a trainer-specific payload class.
            for wrapper in ("metadata", "provenance"):
                nested = mapping.get(wrapper)
                if nested is not None:
                    if not isinstance(nested, Mapping):
                        raise ManifestValidationError(
                            f"model metadata {location}{wrapper} must be a mapping"
                        )
                    inspect(nested, location=f"{location}{wrapper}.")

            for section, target in (("training", training), ("tuning", tuning)):
                nested = mapping.get(section)
                if nested is None:
                    continue
                if not isinstance(nested, Mapping):
                    raise ManifestValidationError(
                        f"model metadata {location}{section} must be a mapping"
                    )
                for key in ("recording_ids", "session_ids", "recordings", "sessions"):
                    if key in nested:
                        add_ids(
                            target,
                            nested[key],
                            field=f"{location}{section}.{key}",
                        )

        inspect(value, location="")
        return cls(
            training_recording_ids=frozenset(training),
            tuning_recording_ids=frozenset(tuning),
            manifest_hash=manifest_hash,
        )


def load_model_metadata(value: Mapping[str, object]) -> ModelMetadata:
    """Load model recording provenance from any mapping implementation."""
    return ModelMetadata.from_mapping(value)


@dataclass(frozen=True, slots=True)
class ContaminationReport:
    """Test recording IDs reused while fitting or tuning a model."""

    training_test_ids: frozenset[str] = frozenset()
    tuning_test_ids: frozenset[str] = frozenset()

    @property
    def contaminated_ids(self) -> frozenset[str]:
        return self.training_test_ids | self.tuning_test_ids

    @property
    def is_contaminated(self) -> bool:
        return bool(self.contaminated_ids)

    def __bool__(self) -> bool:
        return self.is_contaminated


@dataclass(frozen=True, slots=True)
class EvaluationManifest:
    """Validated evaluation recording manifest."""

    schema_version: int
    recordings: tuple[RecordingEntry, ...]
    path: Path | None = None

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        validate_files: bool = True,
    ) -> EvaluationManifest:
        manifest_path = Path(path).resolve()
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except FileNotFoundError as exc:
            raise ManifestValidationError(
                f"manifest does not exist: {manifest_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ManifestValidationError(
                f"manifest is not valid JSON: {manifest_path}: {exc.msg}"
            ) from exc
        except OSError as exc:
            raise ManifestValidationError(
                f"manifest could not be read: {manifest_path}: {exc}"
            ) from exc

        return cls.from_mapping(
            value,
            base_dir=manifest_path.parent,
            path=manifest_path,
            validate_files=validate_files,
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        base_dir: str | Path,
        path: str | Path | None = None,
        validate_files: bool = True,
    ) -> EvaluationManifest:
        if not isinstance(value, Mapping):
            raise ManifestValidationError("manifest root must be an object")

        schema_version = value.get("schema_version")
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise ManifestValidationError("schema_version must be an integer")
        if schema_version != MANIFEST_SCHEMA_VERSION:
            raise ManifestValidationError(
                "unsupported schema_version "
                f"{schema_version!r}; expected {MANIFEST_SCHEMA_VERSION}"
            )

        raw_recordings = value.get("recordings")
        if (
            isinstance(raw_recordings, (str, bytes, Mapping))
            or not isinstance(raw_recordings, Sequence)
        ):
            raise ManifestValidationError("recordings must be an array")

        resolved_base_dir = Path(base_dir).resolve()
        recordings: list[RecordingEntry] = []
        ids: set[str] = set()
        setup_splits: dict[str, set[str]] = {}

        for index, raw_recording in enumerate(raw_recordings):
            if not isinstance(raw_recording, Mapping):
                raise ManifestValidationError(
                    f"recordings[{index}] must be an object"
                )
            recording = RecordingEntry.from_mapping(
                raw_recording,
                base_dir=resolved_base_dir,
                index=index,
                validate_files=validate_files,
            )
            if recording.id in ids:
                raise ManifestValidationError(
                    f"duplicate recording id {recording.id!r}"
                )
            ids.add(recording.id)
            recordings.append(recording)

            if recording.split in LEAKAGE_CONTROLLED_SPLITS:
                setup_splits.setdefault(recording.setup_id, set()).add(
                    recording.split
                )

        crossing_setups = {
            setup_id: splits
            for setup_id, splits in setup_splits.items()
            if len(splits) > 1
        }
        if crossing_setups:
            details = "; ".join(
                f"{setup_id!r}: {', '.join(sorted(splits))}"
                for setup_id, splits in sorted(crossing_setups.items())
            )
            raise ManifestValidationError(
                "setup_id values may not cross train/validation/test splits: "
                f"{details}"
            )

        return cls(
            schema_version=schema_version,
            recordings=tuple(recordings),
            path=Path(path).resolve() if path is not None else None,
        )

    def select_split(self, split: str) -> tuple[RecordingEntry, ...]:
        """Return recordings in one supported split, preserving manifest order."""
        if split not in SUPPORTED_SPLITS:
            supported = ", ".join(sorted(SUPPORTED_SPLITS))
            raise ManifestValidationError(
                f"split must be one of {supported}; got {split!r}"
            )
        return tuple(recording for recording in self.recordings if recording.split == split)

    def for_split(self, split: str) -> tuple[RecordingEntry, ...]:
        """Alias for :meth:`select_split`."""
        return self.select_split(split)

    def recording_ids(self, split: str | None = None) -> frozenset[str]:
        selected = self.recordings if split is None else self.select_split(split)
        return frozenset(recording.id for recording in selected)

    def detect_test_contamination(
        self,
        metadata: ModelMetadata | Mapping[str, object],
    ) -> ContaminationReport:
        """Find test IDs present in model training or tuning provenance."""
        parsed = (
            metadata
            if isinstance(metadata, ModelMetadata)
            else load_model_metadata(metadata)
        )
        test_ids = self.recording_ids("test")
        return ContaminationReport(
            training_test_ids=frozenset(
                test_ids.intersection(parsed.training_recording_ids)
            ),
            tuning_test_ids=frozenset(
                test_ids.intersection(parsed.tuning_recording_ids)
            ),
        )

    def detect_contamination(
        self,
        metadata: ModelMetadata | Mapping[str, object],
    ) -> ContaminationReport:
        """Alias for :meth:`detect_test_contamination`."""
        return self.detect_test_contamination(metadata)

    def assert_no_test_contamination(
        self,
        metadata: ModelMetadata | Mapping[str, object],
    ) -> None:
        report = self.detect_test_contamination(metadata)
        if not report:
            return

        details: list[str] = []
        if report.training_test_ids:
            details.append(
                "training=" + ",".join(sorted(report.training_test_ids))
            )
        if report.tuning_test_ids:
            details.append("tuning=" + ",".join(sorted(report.tuning_test_ids)))
        raise TestContaminationError(
            "model metadata contaminates the test split (" + "; ".join(details) + ")"
        )


def load_evaluation_manifest(
    path: str | Path,
    *,
    validate_files: bool = True,
) -> EvaluationManifest:
    """Convenience wrapper around :meth:`EvaluationManifest.load`."""
    return EvaluationManifest.load(path, validate_files=validate_files)
