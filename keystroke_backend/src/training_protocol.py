"""Leakage-safe training plans and model-artifact provenance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

from .dataset_quality import DatasetQualityReport, validate_dataset
from .evaluation_manifest import EvaluationManifest, RecordingEntry, load_evaluation_manifest


class TrainingProtocolError(ValueError):
    """Raised when a manifest cannot safely be used for model fitting."""


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    manifest: EvaluationManifest
    manifest_sha256: str
    training_recordings: tuple[RecordingEntry, ...]
    training_negative_recordings: tuple[RecordingEntry, ...]
    validation_recording_ids: tuple[str, ...]

    @property
    def training_recording_ids(self) -> tuple[str, ...]:
        return tuple(entry.id for entry in self.training_recordings)

    @property
    def training_setup_ids(self) -> tuple[str, ...]:
        return tuple(sorted({entry.setup_id for entry in self.training_recordings}))


QualityValidator = Callable[[EvaluationManifest], DatasetQualityReport]


def load_training_plan(
    manifest_path: str | Path,
    *,
    quality_validator: QualityValidator = validate_dataset,
) -> TrainingPlan:
    """Load a quality-passing manifest without exposing validation/test truth."""

    path = Path(manifest_path).resolve()
    manifest = load_evaluation_manifest(path)
    quality = quality_validator(manifest)
    if not quality.valid:
        recording_errors = sum(not recording.valid for recording in quality.recordings)
        raise TrainingProtocolError(
            "dataset quality validation failed "
            f"({recording_errors} invalid recording(s), {len(quality.issues)} coverage issue(s))"
        )

    training = tuple(
        entry
        for entry in manifest.select_split("train")
        if not entry.is_negative_fixture
    )
    training_negatives = tuple(
        entry for entry in manifest.select_split("train") if entry.is_negative_fixture
    )
    if not training:
        raise TrainingProtocolError("train split contains no typing recordings")
    if any(entry.log_path is None or entry.meta_path is None for entry in training):
        raise TrainingProtocolError("every training recording needs keylog and metadata")

    validation_ids = tuple(entry.id for entry in manifest.select_split("validation"))
    if not validation_ids:
        raise TrainingProtocolError("validation split is empty")

    return TrainingPlan(
        manifest=manifest,
        manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        training_recordings=training,
        training_negative_recordings=training_negatives,
        validation_recording_ids=validation_ids,
    )


def read_sync_offset_ms(entry: RecordingEntry) -> float:
    if entry.meta_path is None:
        raise TrainingProtocolError(f"{entry.id}: capture metadata is missing")
    try:
        metadata = json.loads(entry.meta_path.read_text(encoding="utf-8"))
        value = float(metadata["sync_offset_ms"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TrainingProtocolError(
            f"{entry.id}: metadata has no valid sync_offset_ms"
        ) from exc
    return value


def artifact_provenance(
    plan: TrainingPlan,
    *,
    artifact_role: str,
    fit_config: Mapping[str, object],
    trained_at: str | None = None,
    training_recordings: tuple[RecordingEntry, ...] | None = None,
) -> dict[str, object]:
    """Return the common provenance fields embedded in every new artifact."""

    timestamp = trained_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    fitted_recordings = training_recordings or plan.training_recordings
    return {
        "artifact_schema_version": 1,
        "artifact_role": artifact_role,
        "trained_at": timestamp,
        "manifest_hash": plan.manifest_sha256,
        "training_recording_ids": [entry.id for entry in fitted_recordings],
        "tuning_recording_ids": list(plan.validation_recording_ids),
        "training_setup_ids": sorted({entry.setup_id for entry in fitted_recordings}),
        "fit_config": dict(fit_config),
    }


__all__ = [
    "TrainingPlan",
    "TrainingProtocolError",
    "artifact_provenance",
    "load_training_plan",
    "read_sync_offset_ms",
]
