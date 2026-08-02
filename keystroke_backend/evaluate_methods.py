"""Production-parity evaluation over a leakage-safe recording manifest.

This module intentionally contains orchestration only.  Predictions are made
by the exact trace-producing functions used by the API, while ground-truth
parsing and metric definitions live in :mod:`src.evaluation`.

Example::

    python evaluate_methods.py \
        --manifest data/evaluation_manifest.json \
        --split diagnostic \
        --methods rule ml yamnet \
        --output-json results/diagnostic.json

The historical hard-coded six-session evaluator was not release-safe.  It
truncated unequal sequences, counted separator rows, and reused model fitting
recordings.  This replacement never discovers recordings implicitly: a
versioned manifest is required for both the callable and CLI entry points.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import operator
import os
import sys
import wave
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import joblib

from src.evaluation import (
    AggregateSequenceMetrics,
    BoundaryMetrics,
    EventMetrics,
    GroundTruth,
    SequenceMetrics,
    aggregate_boundary_metrics,
    aggregate_event_metrics,
    aggregate_sequence_metrics,
    match_boundaries,
    match_events,
    parse_keylog,
    predicted_boundary_intervals,
    score_count_sequence,
)
from src.evaluation_manifest import (
    EvaluationManifest,
    ManifestValidationError,
    RecordingEntry,
    load_evaluation_manifest,
    load_model_metadata,
)
from src.predictor import (
    predict_keystroke_counts_ml_with_trace,
    predict_keystroke_counts_rule_with_trace,
    predict_keystroke_counts_with_trace,
)
from src.yamnet_config import (
    CLASSIFIER_THRESHOLD,
    GAP_THRESHOLD_ML,
    GAP_THRESHOLD_RULE,
    MERGE_GAP_SECONDS,
    SENSITIVITY_DELTA,
)


RESULT_SCHEMA_VERSION = 1
DATASET_SNAPSHOT_SCHEMA_VERSION = 1
SUPPORTED_METHODS = ("rule", "ml", "yamnet")
DEFAULT_EVENT_TOLERANCE_SECONDS = 0.08
DEFAULT_BOUNDARY_SLACK_SECONDS = 0.08

BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_PATHS: Mapping[str, Path] = {
    "ml": BACKEND_DIR / "models" / "count_predictor_new.joblib",
    "yamnet": BACKEND_DIR / "models" / "keystroke_classifier.joblib",
}
ARTIFACT_ROLES = {
    "ml": "count_predictor",
    "yamnet": "yamnet_keystroke_classifier",
}

# These are the recordings that existed before the leakage-safe evaluation
# protocol.  They remain useful for exercising the harness, but may not be
# relabelled as train/validation/test evidence.
LEGACY_DIAGNOSTIC_RECORDING_IDS = frozenset(
    {
        "calib_check1",
        "calib_check2",
        "calib_check3",
        "calib_check4",
        "gain_check",
        "gain_test",
        "latency_calib",
        "new1",
        "new2",
        "session1",
        "session2",
        "session3",
        "session4",
        "session_calibrated_test",
        "silence_10s",
        "silence_calib",
        "silence_check",
        "silence_new",
    }
)

Predictor = Callable[..., object]
ArtifactLoader = Callable[[str], object]
DurationReader = Callable[[Path], float]


class EvaluationConfigurationError(ValueError):
    """Raised when an evaluation run cannot produce auditable results."""


@dataclass(frozen=True, slots=True)
class _TraceView:
    counts: tuple[int, ...]
    counted_onsets: tuple[float, ...]
    word_groups: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class _NegativeMetrics:
    duration_seconds: float
    false_onset_count: int
    false_word_count: int
    false_onsets_per_minute: float
    false_words_per_minute: float
    any_output: bool
    empty_sequence_exact: bool


def _default_predictors(
    artifact_paths: Mapping[str, str | Path] = DEFAULT_ARTIFACT_PATHS,
) -> dict[str, Predictor]:
    """Resolve defaults at call time so tests can replace individual paths."""

    return {
        "rule": predict_keystroke_counts_rule_with_trace,
        "ml": partial(
            predict_keystroke_counts_ml_with_trace,
            count_model_path=artifact_paths.get("ml", DEFAULT_ARTIFACT_PATHS["ml"]),
        ),
        "yamnet": partial(
            predict_keystroke_counts_with_trace,
            classifier_path=artifact_paths.get(
                "yamnet", DEFAULT_ARTIFACT_PATHS["yamnet"]
            ),
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _recording_file_hashes(
    manifest: EvaluationManifest,
) -> dict[str, dict[str, str | None]]:
    """Hash manifest files and reject aliases/copies before inference.

    The audit covers the entire manifest, rather than only the requested
    split, so a copied test recording cannot evade detection during a test-only
    run.  Content duplication is checked independently for each file role;
    exact copies of WAV, keylog, or metadata files across recording IDs are
    all treated as evidence leakage.  Resolved-path aliases are checked across
    every role as an additional guard against malformed manifests.
    """

    path_owners: dict[str, tuple[RecordingEntry, str, Path]] = {}
    content_owners: dict[
        tuple[str, str], tuple[RecordingEntry, Path]
    ] = {}
    hashes: dict[str, dict[str, str | None]] = {}

    for entry in manifest.recordings:
        entry_hashes: dict[str, str | None] = {
            "wav_sha256": None,
            "log_sha256": None,
            "meta_sha256": None,
        }
        files = (
            ("wav", entry.wav_path),
            ("log", entry.log_path),
            ("meta", entry.meta_path),
        )
        for role, path in files:
            if path is None:
                continue

            resolved_path = path.resolve()
            # Honor the host filesystem's path-case convention: Windows path
            # aliases differ from POSIX paths that are distinct only by case.
            path_key = os.path.normcase(str(resolved_path))
            prior_path = path_owners.get(path_key)
            if prior_path is not None:
                prior_entry, prior_role, _prior_resolved_path = prior_path
                raise EvaluationConfigurationError(
                    "Duplicate resolved file path in evaluation manifest: "
                    f"recording {prior_entry.id!r} ({prior_entry.split}) "
                    f"{prior_role} and recording {entry.id!r} "
                    f"({entry.split}) {role} both resolve to {resolved_path}"
                )
            path_owners[path_key] = (entry, role, resolved_path)

            try:
                file_hash = _sha256_file(resolved_path)
            except OSError as exc:
                raise EvaluationConfigurationError(
                    f"Could not hash {role} file for recording {entry.id!r}: "
                    f"{resolved_path}: {exc}"
                ) from exc

            prior_content = content_owners.get((role, file_hash))
            if prior_content is not None:
                prior_entry, prior_content_path = prior_content
                raise EvaluationConfigurationError(
                    f"Duplicate {role} file contents in evaluation manifest "
                    f"(sha256={file_hash}): recording {prior_entry.id!r} "
                    f"({prior_entry.split}, {prior_content_path}) and recording "
                    f"{entry.id!r} ({entry.split}, {resolved_path})"
                )
            content_owners[(role, file_hash)] = (entry, resolved_path)
            entry_hashes[f"{role}_sha256"] = file_hash

        hashes[entry.id] = entry_hashes

    return hashes


def _dataset_snapshot_sha256(
    recording_file_hashes: Mapping[str, Mapping[str, str | None]],
) -> str:
    """Return a stable content/membership hash for one selected dataset."""

    return _sha256_json(
        {
            "schema_version": DATASET_SNAPSHOT_SCHEMA_VERSION,
            "recording_files": recording_file_hashes,
        }
    )


def _normalize_hash(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized.startswith("sha256:"):
        normalized = normalized[len("sha256:") :]
    return normalized


def _validated_sha256(value: str, *, field: str) -> str:
    normalized = _normalize_hash(value)
    if (
        normalized is None
        or len(normalized) != 64
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise EvaluationConfigurationError(
            f"{field} must be a 64-character SHA-256 hex digest"
        )
    return normalized


def _normalize_methods(methods: Sequence[str]) -> tuple[str, ...]:
    if isinstance(methods, (str, bytes)):
        methods = (str(methods),)
    normalized: list[str] = []
    seen: set[str] = set()
    for method in methods:
        name = str(method).strip().lower()
        if name not in SUPPORTED_METHODS:
            raise EvaluationConfigurationError(
                f"Unsupported method {method!r}; expected one of "
                + ", ".join(SUPPORTED_METHODS)
            )
        if name not in seen:
            normalized.append(name)
            seen.add(name)
    if not normalized:
        raise EvaluationConfigurationError("At least one method must be selected")
    return tuple(normalized)


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise EvaluationConfigurationError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationConfigurationError(
            f"{field} must be a finite number"
        ) from exc
    if not math.isfinite(result):
        raise EvaluationConfigurationError(f"{field} must be a finite number")
    return result


def _positive_number(value: object, *, field: str) -> float:
    result = _finite_number(value, field=field)
    if result <= 0:
        raise EvaluationConfigurationError(f"{field} must be greater than zero")
    return result


def _nonnegative_number(value: object, *, field: str) -> float:
    result = _finite_number(value, field=field)
    if result < 0:
        raise EvaluationConfigurationError(f"{field} must be non-negative")
    return result


def _load_metadata(entry: RecordingEntry) -> dict[str, object]:
    if entry.meta_path is None:
        return {}
    try:
        with entry.meta_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise EvaluationConfigurationError(
            f"Metadata for {entry.id!r} is not valid JSON: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise EvaluationConfigurationError(
            f"Metadata for {entry.id!r} could not be read: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise EvaluationConfigurationError(
            f"Metadata for {entry.id!r} must be a JSON object"
        )
    return value


def _sync_offset_ms(entry: RecordingEntry, metadata: Mapping[str, object]) -> float:
    if "sync_offset_ms" not in metadata:
        raise EvaluationConfigurationError(
            f"Metadata for typing recording {entry.id!r} is missing sync_offset_ms"
        )
    return _finite_number(
        metadata["sync_offset_ms"],
        field=f"metadata[{entry.id!r}].sync_offset_ms",
    )


def _wav_duration_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            if frame_rate <= 0:
                raise EvaluationConfigurationError(
                    f"WAV sample rate must be positive: {path}"
                )
            return wav_file.getnframes() / frame_rate
    except (OSError, EOFError, wave.Error) as exc:
        raise EvaluationConfigurationError(
            f"Could not determine WAV duration for {path}: {exc}"
        ) from exc


def _duration_seconds(
    entry: RecordingEntry,
    duration_reader: DurationReader,
) -> float:
    # Capture metadata stores the configured recording duration, which may be
    # rounded or differ from the file after an interrupted capture.  False
    # positive rates therefore always use the actual frame duration.
    try:
        duration = duration_reader(entry.wav_path)
    except EvaluationConfigurationError:
        raise
    except Exception as exc:
        raise EvaluationConfigurationError(
            f"Could not read actual WAV duration for {entry.id!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return _positive_number(duration, field=f"WAV duration for {entry.id!r}")


def _validate_legacy_recordings(manifest: EvaluationManifest) -> None:
    invalid = sorted(
        entry.id
        for entry in manifest.recordings
        if entry.split != "diagnostic"
        and (
            entry.id in LEGACY_DIAGNOSTIC_RECORDING_IDS
            or entry.wav_path.stem in LEGACY_DIAGNOSTIC_RECORDING_IDS
        )
    )
    if invalid:
        raise EvaluationConfigurationError(
            "Legacy development recordings may only use split='diagnostic': "
            + ", ".join(invalid)
        )


def _provenance_field_presence(payload: Mapping[str, object]) -> tuple[bool, bool]:
    """Detect explicit training and tuning ID fields without loading a model."""

    training_keys = {
        "training_recording_ids",
        "train_recording_ids",
        "training_session_ids",
        "train_session_ids",
        "training_sessions",
        "train_sessions",
        "sessions",
    }
    tuning_keys = {
        "tuning_recording_ids",
        "tune_recording_ids",
        "validation_recording_ids",
        "tuning_session_ids",
        "tune_session_ids",
        "validation_session_ids",
        "tuning_sessions",
        "validation_sessions",
    }
    training_present = any(key in payload for key in training_keys)
    tuning_present = any(key in payload for key in tuning_keys)

    for wrapper in ("metadata", "provenance"):
        nested = payload.get(wrapper)
        if isinstance(nested, Mapping):
            nested_training, nested_tuning = _provenance_field_presence(nested)
            training_present = training_present or nested_training
            tuning_present = tuning_present or nested_tuning

    training_section = payload.get("training")
    if isinstance(training_section, Mapping):
        training_present = training_present or any(
            key in training_section
            for key in ("recording_ids", "session_ids", "recordings", "sessions")
        )
    tuning_section = payload.get("tuning")
    if isinstance(tuning_section, Mapping):
        tuning_present = tuning_present or any(
            key in tuning_section
            for key in ("recording_ids", "session_ids", "recordings", "sessions")
        )
    return training_present, tuning_present


def _inspect_artifacts(
    *,
    manifest: EvaluationManifest,
    manifest_sha256: str,
    methods: tuple[str, ...],
    artifact_paths: Mapping[str, str | Path],
    artifact_loader: ArtifactLoader,
) -> tuple[dict[str, dict[str, object]], list[str]]:
    artifacts: dict[str, dict[str, object]] = {}
    warnings: list[str] = []

    # Rule evaluation has no learned artifact.  Crucially, unselected learned
    # methods are not loaded merely because their files happen to exist.
    for method in methods:
        if method not in DEFAULT_ARTIFACT_PATHS:
            continue
        raw_path = artifact_paths.get(method, DEFAULT_ARTIFACT_PATHS[method])
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise EvaluationConfigurationError(
                f"Selected method {method!r} requires artifact: {path}"
            )

        try:
            payload = artifact_loader(str(path))
        except Exception as exc:
            raise EvaluationConfigurationError(
                f"Could not inspect artifact for method {method!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        artifact: dict[str, object] = {
            "role": ARTIFACT_ROLES[method],
            "method": method,
            "path": str(path),
            "sha256": _sha256_file(path),
        }

        if isinstance(payload, Mapping):
            metadata = load_model_metadata(payload)
            # Explicit overlap is unsafe regardless of which split this run
            # selects; the manifest's test set must stay untouched.
            manifest.assert_no_test_contamination(metadata)
            training_present, tuning_present = _provenance_field_presence(payload)
            manifest_training_ids = manifest.recording_ids("train")
            manifest_tuning_ids = manifest.recording_ids("validation")
            training_ids_outside_train = sorted(
                metadata.training_recording_ids - manifest_training_ids
            )
            tuning_ids_outside_validation = sorted(
                metadata.tuning_recording_ids - manifest_tuning_ids
            )
            artifact_manifest_hash = _normalize_hash(metadata.manifest_hash)
            manifest_matches = artifact_manifest_hash == manifest_sha256
            provenance_complete = (
                training_present
                and tuning_present
                and bool(metadata.training_recording_ids)
                and bool(metadata.tuning_recording_ids)
                and not training_ids_outside_train
                and not tuning_ids_outside_validation
                and artifact_manifest_hash is not None
                and manifest_matches
            )
            artifact["provenance"] = {
                "training_recording_ids": sorted(
                    metadata.training_recording_ids
                ),
                "tuning_recording_ids": sorted(metadata.tuning_recording_ids),
                "training_ids_field_present": training_present,
                "tuning_ids_field_present": tuning_present,
                "training_ids_outside_train_split": training_ids_outside_train,
                "tuning_ids_outside_validation_split": (
                    tuning_ids_outside_validation
                ),
                "manifest_hash": metadata.manifest_hash,
                "manifest_hash_matches": manifest_matches,
                "complete": provenance_complete,
            }
            if not training_present or not tuning_present:
                warnings.append(
                    f"{method} artifact lacks explicit training/tuning recording-ID provenance"
                )
            if training_present and not metadata.training_recording_ids:
                warnings.append(f"{method} artifact has no training recording IDs")
            if tuning_present and not metadata.tuning_recording_ids:
                warnings.append(f"{method} artifact has no tuning recording IDs")
            if training_ids_outside_train:
                warnings.append(
                    f"{method} artifact training IDs are absent from the manifest "
                    "train split: " + ", ".join(training_ids_outside_train)
                )
            if tuning_ids_outside_validation:
                warnings.append(
                    f"{method} artifact tuning IDs are absent from the manifest "
                    "validation split: " + ", ".join(tuning_ids_outside_validation)
                )
            if artifact_manifest_hash is None:
                warnings.append(f"{method} artifact lacks a manifest hash")
            elif not manifest_matches:
                warnings.append(
                    f"{method} artifact manifest hash does not match this manifest"
                )
        else:
            artifact["provenance"] = {
                "training_recording_ids": [],
                "tuning_recording_ids": [],
                "training_ids_field_present": False,
                "tuning_ids_field_present": False,
                "training_ids_outside_train_split": [],
                "tuning_ids_outside_validation_split": [],
                "manifest_hash": None,
                "manifest_hash_matches": False,
                "complete": False,
            }
            warnings.append(
                f"{method} artifact is not a metadata-bearing mapping; provenance is unavailable"
            )

        artifacts[ARTIFACT_ROLES[method]] = artifact
    return artifacts, warnings


def _coerce_trace(value: object, *, method: str, recording_id: str) -> _TraceView:
    try:
        raw_counts = getattr(value, "counts")
        raw_onsets = getattr(value, "counted_onsets")
        raw_groups = getattr(value, "word_groups")
    except AttributeError as exc:
        raise EvaluationConfigurationError(
            f"{method} predictor for {recording_id!r} did not return a prediction trace"
        ) from exc

    counts: list[int] = []
    for index, count in enumerate(raw_counts):
        if isinstance(count, bool):
            raise EvaluationConfigurationError(
                f"{method} trace count {index} for {recording_id!r} is not an integer"
            )
        try:
            parsed = int(operator.index(count))
        except (TypeError, ValueError) as exc:
            raise EvaluationConfigurationError(
                f"{method} trace count {index} for {recording_id!r} is not an integer"
            ) from exc
        if parsed <= 0:
            raise EvaluationConfigurationError(
                f"{method} trace count {index} for {recording_id!r} must be positive"
            )
        counts.append(parsed)

    onsets = tuple(
        _finite_number(onset, field=f"{method} trace onset for {recording_id!r}")
        for onset in raw_onsets
    )
    if any(current < previous for previous, current in zip(onsets, onsets[1:])):
        raise EvaluationConfigurationError(
            f"{method} trace onsets for {recording_id!r} are not ordered"
        )

    groups = tuple(
        tuple(
            _finite_number(
                onset,
                field=f"{method} trace word-group onset for {recording_id!r}",
            )
            for onset in group
        )
        for group in raw_groups
    )
    if len(groups) != len(counts):
        raise EvaluationConfigurationError(
            f"{method} trace for {recording_id!r} has {len(counts)} counts "
            f"but {len(groups)} word groups"
        )
    # This validates non-empty groups plus global temporal ordering.
    predicted_boundary_intervals(groups)
    if tuple(onset for group in groups for onset in group) != onsets:
        raise EvaluationConfigurationError(
            f"{method} trace word groups for {recording_id!r} do not exactly "
            "partition counted_onsets"
        )
    return _TraceView(tuple(counts), onsets, groups)


def _predict(
    method: str,
    predictor: Predictor,
    wav_path: Path,
    serving_config: Mapping[str, Mapping[str, float]],
) -> object:
    return predictor(str(wav_path), **serving_config[method])


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_optional(values: Sequence[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return sum(available) / len(available) if available else None


def _sequence_metrics_json(metrics: SequenceMetrics) -> dict[str, object]:
    return {
        "aligned_exact_accuracy": metrics.aligned_accuracy,
        "gap_penalized_mae": metrics.gap_penalized_mae,
        "full_sequence_exact_match": metrics.full_sequence_exact,
        "matches": metrics.matches,
        "substitutions": metrics.substitutions,
        "deletions": metrics.deletions,
        "insertions": metrics.insertions,
        "aligned_items": metrics.aligned_items,
        "true_word_count": metrics.true_word_count,
        "predicted_word_count": metrics.predicted_word_count,
        "signed_word_count_delta": (
            metrics.predicted_word_count - metrics.true_word_count
        ),
        "absolute_total_count_error": metrics.total_count_error,
        "signed_total_count_delta": metrics.signed_total_count_delta,
        "gap_absolute_error": metrics.gap_absolute_error,
        "alignment": [
            {
                "operation": operation.kind.value,
                "true_index": operation.true_index,
                "predicted_index": operation.predicted_index,
                "true_count": operation.true_count,
                "predicted_count": operation.predicted_count,
                "absolute_count_error": operation.absolute_count_error,
            }
            for operation in metrics.alignment.operations
        ],
    }


def _event_metrics_json(metrics: EventMetrics) -> dict[str, object]:
    return {
        "true_count": metrics.true_count,
        "predicted_count": metrics.predicted_count,
        "true_positives": metrics.true_positives,
        "false_positives": metrics.false_positives,
        "false_negatives": metrics.false_negatives,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "timing_mae_seconds": metrics.mean_absolute_timing_error_seconds,
        "timing_p95_seconds": metrics.p95_absolute_timing_error_seconds,
        "timing_max_seconds": metrics.max_absolute_timing_error_seconds,
        "matches": [
            {
                "true_index": pair.true_index,
                "predicted_index": pair.predicted_index,
                "true_time_seconds": pair.true_time_seconds,
                "predicted_time_seconds": pair.predicted_time_seconds,
                "timing_error_seconds": pair.timing_error_seconds,
            }
            for pair in metrics.matched_pairs
        ],
    }


def _boundary_metrics_json(metrics: BoundaryMetrics) -> dict[str, object]:
    distances = [pair.distance_seconds for pair in metrics.matched_pairs]
    return {
        "true_count": metrics.true_count,
        "predicted_count": metrics.predicted_count,
        "true_positives": metrics.true_positives,
        "false_positives": metrics.false_positives,
        "false_negatives": metrics.false_negatives,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "distance_mae_seconds": metrics.mean_distance_seconds,
        "distance_p95_seconds": _percentile(distances, 0.95),
        "distance_max_seconds": metrics.max_distance_seconds,
        "matches": [
            {
                "true_index": pair.true_index,
                "predicted_index": pair.predicted_index,
                "true_time_seconds": pair.true_time_seconds,
                "predicted_interval": {
                    "start_time_seconds": (
                        pair.predicted_interval.start_time_seconds
                    ),
                    "end_time_seconds": pair.predicted_interval.end_time_seconds,
                },
                "distance_seconds": pair.distance_seconds,
            }
            for pair in metrics.matched_pairs
        ],
    }


def _negative_metrics_json(metrics: _NegativeMetrics) -> dict[str, object]:
    return {
        "duration_seconds": metrics.duration_seconds,
        "false_onset_count": metrics.false_onset_count,
        "false_word_count": metrics.false_word_count,
        "false_onsets_per_minute": metrics.false_onsets_per_minute,
        "false_words_per_minute": metrics.false_words_per_minute,
        "any_output": metrics.any_output,
        "empty_sequence_exact": metrics.empty_sequence_exact,
    }


def _aggregate_sequences(
    sessions: Sequence[SequenceMetrics],
) -> dict[str, object]:
    micro: AggregateSequenceMetrics = aggregate_sequence_metrics(sessions)
    return {
        "micro": {
            "recording_count": micro.session_count,
            "aligned_exact_accuracy": micro.aligned_accuracy,
            "gap_penalized_mae": micro.gap_penalized_mae,
            "full_sequence_exact_rate": micro.full_sequence_exact_rate,
            "matches": micro.matches,
            "substitutions": micro.substitutions,
            "deletions": micro.deletions,
            "insertions": micro.insertions,
            "aligned_items": micro.aligned_items,
            "true_word_count": micro.true_word_count,
            "predicted_word_count": micro.predicted_word_count,
            "signed_word_count_delta": (
                micro.predicted_word_count - micro.true_word_count
            ),
            "absolute_total_count_error_sum": micro.total_count_error,
            "signed_total_count_delta": micro.signed_total_count_delta,
            "gap_absolute_error": micro.gap_absolute_error,
        },
        "macro": {
            "recording_count": len(sessions),
            "aligned_exact_accuracy_mean": _mean(
                [session.aligned_accuracy for session in sessions]
            ),
            "gap_penalized_mae_mean": _mean(
                [session.gap_penalized_mae for session in sessions]
            ),
            "full_sequence_exact_rate": _mean(
                [float(session.full_sequence_exact) for session in sessions]
            ),
            "absolute_total_count_error_mean": _mean(
                [float(session.total_count_error) for session in sessions]
            ),
            "absolute_word_count_delta_mean": _mean(
                [
                    float(
                        abs(
                            session.predicted_word_count
                            - session.true_word_count
                        )
                    )
                    for session in sessions
                ]
            ),
        },
    }


def _aggregate_events(sessions: Sequence[EventMetrics]) -> dict[str, object]:
    micro = aggregate_event_metrics(sessions)
    return {
        "micro": {
            "recording_count": micro.session_count,
            "true_count": micro.true_count,
            "predicted_count": micro.predicted_count,
            "true_positives": micro.true_positives,
            "false_positives": micro.false_positives,
            "false_negatives": micro.false_negatives,
            "precision": micro.precision,
            "recall": micro.recall,
            "f1": micro.f1,
            "timing_mae_seconds": micro.mean_absolute_timing_error_seconds,
            "timing_p95_seconds": micro.p95_absolute_timing_error_seconds,
        },
        "macro": {
            "recording_count": len(sessions),
            "precision_mean": _mean([session.precision for session in sessions]),
            "recall_mean": _mean([session.recall for session in sessions]),
            "f1_mean": _mean([session.f1 for session in sessions]),
            "timing_mae_seconds_mean": _mean_optional(
                [session.mean_absolute_timing_error_seconds for session in sessions]
            ),
        },
    }


def _aggregate_boundaries(
    sessions: Sequence[BoundaryMetrics],
) -> dict[str, object]:
    micro = aggregate_boundary_metrics(sessions)
    distances = [
        pair.distance_seconds
        for session in sessions
        for pair in session.matched_pairs
    ]
    return {
        "micro": {
            "recording_count": micro.session_count,
            "true_count": micro.true_count,
            "predicted_count": micro.predicted_count,
            "true_positives": micro.true_positives,
            "false_positives": micro.false_positives,
            "false_negatives": micro.false_negatives,
            "precision": micro.precision,
            "recall": micro.recall,
            "f1": micro.f1,
            "distance_mae_seconds": micro.mean_distance_seconds,
            "distance_p95_seconds": _percentile(distances, 0.95),
        },
        "macro": {
            "recording_count": len(sessions),
            "precision_mean": _mean([session.precision for session in sessions]),
            "recall_mean": _mean([session.recall for session in sessions]),
            "f1_mean": _mean([session.f1 for session in sessions]),
            "distance_mae_seconds_mean": _mean(
                [session.mean_distance_seconds for session in sessions]
            ),
        },
    }


def _aggregate_negatives(
    sessions: Sequence[_NegativeMetrics],
) -> dict[str, object]:
    duration_seconds = sum(session.duration_seconds for session in sessions)
    false_onsets = sum(session.false_onset_count for session in sessions)
    false_words = sum(session.false_word_count for session in sessions)
    total_minutes = duration_seconds / 60.0
    return {
        "micro": {
            "recording_count": len(sessions),
            "duration_seconds": duration_seconds,
            "false_onset_count": false_onsets,
            "false_word_count": false_words,
            "false_onsets_per_minute": (
                false_onsets / total_minutes if total_minutes else 0.0
            ),
            "false_words_per_minute": (
                false_words / total_minutes if total_minutes else 0.0
            ),
            "any_output_rate": _mean(
                [float(session.any_output) for session in sessions]
            ),
            "empty_sequence_exact_rate": _mean(
                [float(session.empty_sequence_exact) for session in sessions]
            ),
        },
        "macro": {
            "recording_count": len(sessions),
            "false_onsets_per_minute_mean": _mean(
                [session.false_onsets_per_minute for session in sessions]
            ),
            "false_words_per_minute_mean": _mean(
                [session.false_words_per_minute for session in sessions]
            ),
            "any_output_rate": _mean(
                [float(session.any_output) for session in sessions]
            ),
            "empty_sequence_exact_rate": _mean(
                [float(session.empty_sequence_exact) for session in sessions]
            ),
        },
    }


def _ground_truth_json(truth: GroundTruth) -> dict[str, object]:
    return {
        "counts": list(truth.counts),
        "event_times_seconds": list(truth.event_times),
        "boundary_times_seconds": list(truth.boundary_times),
        "separator_times_seconds": list(truth.separator_times),
        "sync_offset_seconds": truth.sync_offset_seconds,
    }


def _fixture_type(entry: RecordingEntry) -> str:
    if entry.silence and entry.non_keyboard:
        return "silence_non_keyboard"
    if entry.silence:
        return "silence"
    if entry.non_keyboard:
        return "non_keyboard"
    return "typing"


def evaluate_manifest(
    manifest: EvaluationManifest,
    *,
    split: str = "diagnostic",
    methods: Sequence[str] = SUPPORTED_METHODS,
    threshold: float = CLASSIFIER_THRESHOLD,
    delta: float = SENSITIVITY_DELTA,
    rule_gap_threshold: float = GAP_THRESHOLD_RULE,
    ml_gap_threshold: float = GAP_THRESHOLD_ML,
    yamnet_gap_threshold: float = GAP_THRESHOLD_ML,
    merge_gap_seconds: float = MERGE_GAP_SECONDS,
    event_tolerance_seconds: float = DEFAULT_EVENT_TOLERANCE_SECONDS,
    boundary_slack_seconds: float = DEFAULT_BOUNDARY_SLACK_SECONDS,
    predictors: Mapping[str, Predictor] | None = None,
    artifact_paths: Mapping[str, str | Path] | None = None,
    artifact_loader: ArtifactLoader = joblib.load,
    duration_reader: DurationReader = _wav_duration_seconds,
    locked_config_sha256: str | None = None,
    generated_at: str | None = None,
) -> dict[str, object]:
    """Evaluate one manifest split and return a JSON-serializable result.

    Dependency injection is limited to predictors, artifact inspection, and
    duration reading so integration tests can remain deterministic and avoid
    loading audio or ML runtimes.  Production callers should use the defaults.
    """

    selected_methods = _normalize_methods(methods)
    _validate_legacy_recordings(manifest)
    selected_recordings = manifest.select_split(split)
    if manifest.path is None or not manifest.path.is_file():
        raise EvaluationConfigurationError(
            "EvaluationManifest must be loaded from an on-disk manifest so its "
            "exact bytes can be hashed"
        )

    threshold = _nonnegative_number(threshold, field="threshold")
    if threshold > 1:
        raise EvaluationConfigurationError("threshold must not exceed 1")
    delta = _positive_number(delta, field="delta")
    rule_gap_threshold = _positive_number(
        rule_gap_threshold, field="rule_gap_threshold"
    )
    ml_gap_threshold = _positive_number(
        ml_gap_threshold, field="ml_gap_threshold"
    )
    yamnet_gap_threshold = _positive_number(
        yamnet_gap_threshold, field="yamnet_gap_threshold"
    )
    merge_gap_seconds = _nonnegative_number(
        merge_gap_seconds, field="merge_gap_seconds"
    )
    event_tolerance_seconds = _nonnegative_number(
        event_tolerance_seconds, field="event_tolerance_seconds"
    )
    boundary_slack_seconds = _nonnegative_number(
        boundary_slack_seconds, field="boundary_slack_seconds"
    )

    resolved_artifact_paths = artifact_paths or DEFAULT_ARTIFACT_PATHS
    available_predictors = _default_predictors(resolved_artifact_paths)
    if predictors is not None:
        available_predictors.update(predictors)
    missing_predictors = [
        method
        for method in selected_methods
        if not callable(available_predictors.get(method))
    ]
    if missing_predictors:
        raise EvaluationConfigurationError(
            "No callable predictor supplied for: " + ", ".join(missing_predictors)
        )

    serving_config: dict[str, dict[str, float]] = {
        "rule": {
            "delta": delta,
            "gap_threshold": rule_gap_threshold,
            "merge_gap_seconds": merge_gap_seconds,
        },
        "ml": {
            "delta": delta,
            "gap_threshold": ml_gap_threshold,
            "merge_gap_seconds": merge_gap_seconds,
        },
        "yamnet": {
            "threshold": threshold,
            "delta": delta,
            "gap_threshold": yamnet_gap_threshold,
            "merge_gap_seconds": merge_gap_seconds,
        },
    }
    config: dict[str, object] = {
        "split": split,
        "methods": list(selected_methods),
        "event_tolerance_seconds": event_tolerance_seconds,
        "boundary_slack_seconds": boundary_slack_seconds,
        "serving": {
            method: serving_config[method] for method in selected_methods
        },
    }
    supplied_configuration_lock = (
        _validated_sha256(
            locked_config_sha256,
            field="locked_config_sha256",
        )
        if locked_config_sha256 is not None
        else None
    )
    manifest_sha256 = _sha256_file(manifest.path)
    # Hash and audit every manifest asset before loading predictors or running
    # inference.  The output is narrowed to the selected split below, but the
    # duplicate checks deliberately span train/validation/test/diagnostic.
    all_recording_file_hashes = _recording_file_hashes(manifest)
    selected_recording_file_hashes = {
        entry.id: all_recording_file_hashes[entry.id]
        for entry in selected_recordings
    }
    dataset_snapshot_sha256 = _dataset_snapshot_sha256(
        selected_recording_file_hashes
    )
    custom_artifact_methods = sorted(
        method
        for method in selected_methods
        if method in DEFAULT_ARTIFACT_PATHS
        and Path(
            resolved_artifact_paths.get(method, DEFAULT_ARTIFACT_PATHS[method])
        ).resolve()
        != DEFAULT_ARTIFACT_PATHS[method].resolve()
    )
    artifacts, artifact_warnings = _inspect_artifacts(
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        methods=selected_methods,
        artifact_paths=resolved_artifact_paths,
        artifact_loader=artifact_loader,
    )
    lockable_config = {
        key: value for key, value in config.items() if key != "split"
    }
    lockable_config["artifact_sha256"] = {
        method: artifacts[method]["sha256"]
        for method in selected_methods
        if method in artifacts
    }
    configuration_lock_sha256 = _sha256_json(lockable_config)

    accumulators: dict[str, dict[str, list[Any]]] = {
        method: {
            "sequences": [],
            "events": [],
            "boundaries": [],
            "negatives": [],
            "silence_negatives": [],
            "non_keyboard_negatives": [],
        }
        for method in selected_methods
    }
    recording_results: list[dict[str, object]] = []

    for entry in selected_recordings:
        metadata = _load_metadata(entry)
        negative_fixture = entry.is_negative_fixture
        duration: float | None = None
        if negative_fixture:
            truth = GroundTruth(
                words=(),
                events=(),
                boundary_times=(),
                separator_times=(),
                sync_offset_seconds=0.0,
            )
            duration = _duration_seconds(entry, duration_reader)
        else:
            if entry.log_path is None:
                raise EvaluationConfigurationError(
                    f"Typing recording {entry.id!r} has no keylog"
                )
            sync_offset_ms = _sync_offset_ms(entry, metadata)
            try:
                truth = parse_keylog(
                    entry.log_path,
                    sync_offset_ms=sync_offset_ms,
                )
            except Exception as exc:
                raise EvaluationConfigurationError(
                    f"Ground-truth parsing failed for recording {entry.id!r}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        recording_result: dict[str, object] = {
            "id": entry.id,
            "split": entry.split,
            "setup_id": entry.setup_id,
            "tags": list(entry.tags),
            "fixture_type": _fixture_type(entry),
            "ground_truth": _ground_truth_json(truth),
            "methods": {},
        }
        if duration is not None:
            recording_result["duration_seconds"] = duration

        per_method = recording_result["methods"]
        assert isinstance(per_method, dict)
        for method in selected_methods:
            try:
                raw_trace = _predict(
                    method,
                    available_predictors[method],
                    entry.wav_path,
                    serving_config,
                )
                trace = _coerce_trace(
                    raw_trace, method=method, recording_id=entry.id
                )
                boundary_intervals = predicted_boundary_intervals(
                    trace.word_groups
                )
                sequence_metrics = score_count_sequence(
                    truth.counts, trace.counts
                )
                event_metrics = match_events(
                    truth.event_times,
                    trace.counted_onsets,
                    tolerance_seconds=event_tolerance_seconds,
                )
                boundary_metrics = match_boundaries(
                    truth.boundary_times,
                    boundary_intervals,
                    slack_seconds=boundary_slack_seconds,
                )
            except Exception as exc:
                raise EvaluationConfigurationError(
                    f"Prediction/scoring failed for recording {entry.id!r} "
                    f"with method {method!r}: {type(exc).__name__}: {exc}"
                ) from exc

            method_result: dict[str, object] = {
                "prediction": {
                    "counts": list(trace.counts),
                    "counted_onsets_seconds": list(trace.counted_onsets),
                    "word_groups_seconds": [
                        list(group) for group in trace.word_groups
                    ],
                    "boundary_intervals_seconds": [
                        [interval.start_time_seconds, interval.end_time_seconds]
                        for interval in boundary_intervals
                    ],
                },
                "count_metrics": _sequence_metrics_json(sequence_metrics),
                "event_metrics": _event_metrics_json(event_metrics),
                "boundary_metrics": _boundary_metrics_json(boundary_metrics),
            }

            if negative_fixture:
                assert duration is not None
                minutes = duration / 60.0
                negative_metrics = _NegativeMetrics(
                    duration_seconds=duration,
                    false_onset_count=len(trace.counted_onsets),
                    false_word_count=len(trace.counts),
                    false_onsets_per_minute=(
                        len(trace.counted_onsets) / minutes
                    ),
                    false_words_per_minute=len(trace.counts) / minutes,
                    any_output=bool(trace.counts),
                    empty_sequence_exact=not trace.counts,
                )
                method_result["negative_fixture_metrics"] = (
                    _negative_metrics_json(negative_metrics)
                )
                accumulators[method]["negatives"].append(negative_metrics)
                if entry.silence:
                    accumulators[method]["silence_negatives"].append(
                        negative_metrics
                    )
                if entry.non_keyboard:
                    accumulators[method]["non_keyboard_negatives"].append(
                        negative_metrics
                    )
            else:
                accumulators[method]["sequences"].append(sequence_metrics)
                accumulators[method]["events"].append(event_metrics)
                accumulators[method]["boundaries"].append(boundary_metrics)

            per_method[method] = method_result
        recording_results.append(recording_result)

    aggregates: dict[str, object] = {}
    for method in selected_methods:
        accumulator = accumulators[method]
        aggregates[method] = {
            "count_metrics": _aggregate_sequences(accumulator["sequences"]),
            "event_metrics": _aggregate_events(accumulator["events"]),
            "boundary_metrics": _aggregate_boundaries(
                accumulator["boundaries"]
            ),
            "negative_fixture_metrics": {
                "combined": _aggregate_negatives(accumulator["negatives"]),
                "silence": _aggregate_negatives(
                    accumulator["silence_negatives"]
                ),
                "non_keyboard": _aggregate_negatives(
                    accumulator["non_keyboard_negatives"]
                ),
            },
        }

    legacy_selected = sorted(
        entry.id
        for entry in selected_recordings
        if entry.id in LEGACY_DIAGNOSTIC_RECORDING_IDS
        or entry.wav_path.stem in LEGACY_DIAGNOSTIC_RECORDING_IDS
    )
    invalid_reasons: list[str] = []
    if split != "test":
        invalid_reasons.append(
            f"split {split!r} is not the untouched test split"
        )
    if not selected_recordings:
        invalid_reasons.append(f"split {split!r} contains no recordings")
    if legacy_selected:
        invalid_reasons.append(
            "run contains legacy development recordings: "
            + ", ".join(legacy_selected)
        )
    if split == "test":
        if len(selected_methods) != 1:
            invalid_reasons.append(
                "test evaluation must contain exactly one pre-selected method"
            )
        if not math.isclose(
            event_tolerance_seconds,
            DEFAULT_EVENT_TOLERANCE_SECONDS,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            invalid_reasons.append(
                "test evaluation must use the locked 0.08-second event tolerance"
            )
        if not math.isclose(
            boundary_slack_seconds,
            DEFAULT_BOUNDARY_SLACK_SECONDS,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            invalid_reasons.append(
                "test evaluation must use the locked 0.08-second boundary slack"
            )
        if supplied_configuration_lock is None:
            invalid_reasons.append(
                "test evaluation is missing the validation-locked configuration hash"
            )
        elif supplied_configuration_lock != configuration_lock_sha256:
            invalid_reasons.append(
                "test evaluation configuration does not match the locked hash"
            )
        if predictors is not None:
            invalid_reasons.append(
                "test evaluation uses injected predictor dependencies"
            )
        if artifact_loader is not joblib.load:
            invalid_reasons.append(
                "test evaluation uses an injected artifact loader"
            )
        if duration_reader is not _wav_duration_seconds:
            invalid_reasons.append(
                "test evaluation uses an injected duration reader"
            )
        if custom_artifact_methods:
            invalid_reasons.append(
                "test evaluation uses non-canonical artifact paths: "
                + ", ".join(custom_artifact_methods)
            )
        if not any(not entry.is_negative_fixture for entry in selected_recordings):
            invalid_reasons.append("test split has no typing-recording coverage")
        if not any(entry.silence for entry in selected_recordings):
            invalid_reasons.append("test split has no silence-fixture coverage")
        if not any(
            entry.non_keyboard and not entry.silence
            for entry in selected_recordings
        ):
            invalid_reasons.append(
                "test split has no non-silence non-keyboard-fixture coverage"
            )
        invalid_reasons.extend(artifact_warnings)

    timestamp = generated_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    result: dict[str, object] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "generated_at": timestamp,
        "evaluator": "production-parity-manifest",
        "execution": {
            "canonical_predictors": predictors is None,
            "canonical_artifact_loader": artifact_loader is joblib.load,
            "canonical_duration_reader": duration_reader is _wav_duration_seconds,
            "custom_artifact_methods": sorted(custom_artifact_methods),
        },
        "valid_for_release": not invalid_reasons,
        "release_validation": {
            "valid_for_release": not invalid_reasons,
            "meaning": (
                "Evidence-integrity validation only; product acceptance "
                "thresholds are evaluated separately."
            ),
            "reasons": invalid_reasons,
            "artifact_warnings": artifact_warnings,
            "legacy_diagnostic_recording_ids": legacy_selected,
            "configuration_lock": {
                "supplied_sha256": supplied_configuration_lock,
                "actual_sha256": configuration_lock_sha256,
                "matches": supplied_configuration_lock
                == configuration_lock_sha256,
            },
        },
        "config": config,
        "hashes": {
            "config_sha256": _sha256_json(config),
            "configuration_lock_sha256": configuration_lock_sha256,
            "manifest_sha256": manifest_sha256,
            "dataset_snapshot_sha256": dataset_snapshot_sha256,
            "recording_files": selected_recording_file_hashes,
            "artifacts": {
                name: artifact["sha256"] for name, artifact in artifacts.items()
            },
        },
        "manifest": {
            "path": str(manifest.path),
            "schema_version": manifest.schema_version,
            "selected_split": split,
            "selected_recording_ids": [
                entry.id for entry in selected_recordings
            ],
            "selected_setup_ids": sorted(
                {entry.setup_id for entry in selected_recordings}
            ),
        },
        "artifacts": artifacts,
        "recordings": recording_results,
        "aggregates": aggregates,
    }
    return result


def run_evaluation(
    manifest_path: str | Path,
    **kwargs: object,
) -> dict[str, object]:
    """Load a manifest from disk and evaluate it without invoking the CLI."""

    manifest = load_evaluation_manifest(manifest_path)
    return evaluate_manifest(manifest, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate production predictor traces using a leakage-safe recording manifest."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to a schema-versioned evaluation manifest (required)",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test", "diagnostic"),
        default="diagnostic",
        help="Manifest split to evaluate (default: diagnostic)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=SUPPORTED_METHODS,
        default=list(SUPPORTED_METHODS),
        help="Production methods to evaluate (default: all)",
    )
    parser.add_argument("--threshold", type=float, default=CLASSIFIER_THRESHOLD)
    parser.add_argument("--delta", type=float, default=SENSITIVITY_DELTA)
    parser.add_argument(
        "--rule-gap-threshold", type=float, default=GAP_THRESHOLD_RULE
    )
    parser.add_argument(
        "--ml-gap-threshold", type=float, default=GAP_THRESHOLD_ML
    )
    parser.add_argument(
        "--yamnet-gap-threshold", type=float, default=GAP_THRESHOLD_ML
    )
    parser.add_argument(
        "--merge-gap-seconds", type=float, default=MERGE_GAP_SECONDS
    )
    parser.add_argument(
        "--event-tolerance-seconds",
        type=float,
        default=DEFAULT_EVENT_TOLERANCE_SECONDS,
    )
    parser.add_argument(
        "--boundary-slack-seconds",
        type=float,
        default=DEFAULT_BOUNDARY_SLACK_SECONDS,
    )
    parser.add_argument(
        "--locked-config-sha256",
        help=(
            "Configuration-lock hash emitted by the finalized validation run; "
            "required for release-valid test evidence"
        ),
    )
    parser.add_argument(
        "--ml-artifact",
        type=Path,
        help="candidate count artifact for train/validation evaluation only",
    )
    parser.add_argument(
        "--yamnet-artifact",
        type=Path,
        help="candidate YAMNet classifier for train/validation evaluation only",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Write the result to this path instead of standard output",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    artifact_paths = {
        method: path
        for method, path in (
            ("ml", args.ml_artifact),
            ("yamnet", args.yamnet_artifact),
        )
        if path is not None
    }
    try:
        result = run_evaluation(
            args.manifest,
            split=args.split,
            methods=args.methods,
            threshold=args.threshold,
            delta=args.delta,
            rule_gap_threshold=args.rule_gap_threshold,
            ml_gap_threshold=args.ml_gap_threshold,
            yamnet_gap_threshold=args.yamnet_gap_threshold,
            merge_gap_seconds=args.merge_gap_seconds,
            event_tolerance_seconds=args.event_tolerance_seconds,
            boundary_slack_seconds=args.boundary_slack_seconds,
            locked_config_sha256=args.locked_config_sha256,
            artifact_paths=artifact_paths or None,
        )
    except (EvaluationConfigurationError, ManifestValidationError) as exc:
        parser.exit(2, f"error: {exc}\n")

    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output_json is None:
        sys.stdout.write(encoded)
    else:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded, encoding="utf-8")
        print(f"Results written to: {args.output_json}")
        if not result["valid_for_release"]:
            print("Result is diagnostic only and is not valid release evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
