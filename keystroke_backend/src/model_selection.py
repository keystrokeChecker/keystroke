"""Deterministic production-method selection from validation evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SUPPORTED_METHODS = frozenset({"rule", "ml", "yamnet"})


class ModelSelectionError(ValueError):
    """Raised when an evaluation result cannot safely select a method."""


@dataclass(frozen=True, slots=True)
class CandidateScore:
    method: str
    eligible: bool
    reasons: tuple[str, ...]
    aligned_exact_accuracy: float
    full_sequence_exact_rate: float
    gap_penalized_mae: float
    event_f1: float
    boundary_f1: float
    silence_false_onsets: int
    non_keyboard_false_onsets_per_minute: float


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelSelectionError(f"{field} must be an object")
    return value


def _number(mapping: Mapping[str, Any], key: str, field: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelSelectionError(f"{field}.{key} must be numeric")
    return float(value)


def _score_candidate(
    method: str,
    aggregate: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> CandidateScore:
    count = _mapping(_mapping(aggregate.get("count_metrics"), f"{method}.count_metrics").get("micro"), f"{method}.count_metrics.micro")
    event = _mapping(_mapping(aggregate.get("event_metrics"), f"{method}.event_metrics").get("micro"), f"{method}.event_metrics.micro")
    boundary = _mapping(_mapping(aggregate.get("boundary_metrics"), f"{method}.boundary_metrics").get("micro"), f"{method}.boundary_metrics.micro")
    negatives = _mapping(aggregate.get("negative_fixture_metrics"), f"{method}.negative_fixture_metrics")
    silence = _mapping(_mapping(negatives.get("silence"), f"{method}.silence").get("micro"), f"{method}.silence.micro")
    non_keyboard = _mapping(_mapping(negatives.get("non_keyboard"), f"{method}.non_keyboard").get("micro"), f"{method}.non_keyboard.micro")

    reasons: list[str] = []
    if _number(count, "recording_count", f"{method}.count") <= 0:
        reasons.append("no validation typing recordings")
    if _number(silence, "recording_count", f"{method}.silence") <= 0:
        reasons.append("no validation silence fixtures")
    if _number(non_keyboard, "recording_count", f"{method}.non_keyboard") <= 0:
        reasons.append("no validation non-keyboard fixtures")
    silence_false_onsets = int(_number(silence, "false_onset_count", f"{method}.silence"))
    if silence_false_onsets:
        reasons.append("produced detections on validation silence")

    if method in {"ml", "yamnet"}:
        artifact = _mapping(artifacts.get(method), f"artifacts.{method}")
        provenance = _mapping(artifact.get("provenance"), f"artifacts.{method}.provenance")
        if provenance.get("complete") is not True:
            reasons.append("model artifact provenance is incomplete")

    return CandidateScore(
        method=method,
        eligible=not reasons,
        reasons=tuple(reasons),
        aligned_exact_accuracy=_number(count, "aligned_exact_accuracy", f"{method}.count"),
        full_sequence_exact_rate=_number(count, "full_sequence_exact_rate", f"{method}.count"),
        gap_penalized_mae=_number(count, "gap_penalized_mae", f"{method}.count"),
        event_f1=_number(event, "f1", f"{method}.event"),
        boundary_f1=_number(boundary, "f1", f"{method}.boundary"),
        silence_false_onsets=silence_false_onsets,
        non_keyboard_false_onsets_per_minute=_number(non_keyboard, "false_onsets_per_minute", f"{method}.non_keyboard"),
    )


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_production_method(result: Mapping[str, Any]) -> dict[str, Any]:
    """Select one method without consulting the untouched test split."""

    if result.get("schema_version") != 1:
        raise ModelSelectionError("unsupported evaluation result schema_version")
    if result.get("evaluator") != "production-parity-manifest":
        raise ModelSelectionError("result was not produced by the production-parity evaluator")
    execution = _mapping(result.get("execution"), "execution")
    if any(
        execution.get(field) is not True
        for field in (
            "canonical_predictors",
            "canonical_artifact_loader",
            "canonical_duration_reader",
        )
    ):
        raise ModelSelectionError("validation result used noncanonical execution dependencies")
    manifest = _mapping(result.get("manifest"), "manifest")
    if manifest.get("selected_split") != "validation":
        raise ModelSelectionError("method selection requires validation-split evidence")
    config = _mapping(result.get("config"), "config")
    if config.get("split") != "validation":
        raise ModelSelectionError("config split must be validation")
    methods = config.get("methods")
    if isinstance(methods, (str, bytes)) or not isinstance(methods, Sequence):
        raise ModelSelectionError("config.methods must be an array")
    parsed_methods = tuple(str(method) for method in methods)
    if len(parsed_methods) < 2 or len(set(parsed_methods)) != len(parsed_methods):
        raise ModelSelectionError("validation must compare at least two distinct methods")
    if not set(parsed_methods) <= SUPPORTED_METHODS:
        raise ModelSelectionError("validation contains an unsupported method")

    aggregates = _mapping(result.get("aggregates"), "aggregates")
    artifacts = _mapping(result.get("artifacts", {}), "artifacts")
    candidates = tuple(
        _score_candidate(method, _mapping(aggregates.get(method), f"aggregates.{method}"), artifacts)
        for method in parsed_methods
    )
    eligible = [candidate for candidate in candidates if candidate.eligible]
    if not eligible:
        raise ModelSelectionError("no method satisfies validation safety requirements")
    eligible.sort(
        key=lambda candidate: (
            -candidate.aligned_exact_accuracy,
            -candidate.full_sequence_exact_rate,
            -candidate.boundary_f1,
            -candidate.event_f1,
            candidate.non_keyboard_false_onsets_per_minute,
            candidate.gap_penalized_mae,
            candidate.method,
        )
    )
    selected = eligible[0]

    serving = _mapping(config.get("serving"), "config.serving")
    selected_serving = _mapping(serving.get(selected.method), f"config.serving.{selected.method}")
    single_config = dict(config)
    single_config["methods"] = [selected.method]
    single_config["serving"] = {selected.method: dict(selected_serving)}
    single_config.pop("split", None)
    single_config["artifact_sha256"] = (
        {selected.method: artifacts[selected.method]["sha256"]}
        if selected.method in artifacts
        else {}
    )
    lock_sha256 = _sha256_json(single_config)

    hashes = _mapping(result.get("hashes"), "hashes")
    for field in ("manifest_sha256", "dataset_snapshot_sha256", "config_sha256"):
        value = hashes.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ModelSelectionError(f"hashes.{field} must be a lowercase SHA-256")
    return {
        "schema_version": 1,
        "status": "validation_locked",
        "selected_method": selected.method,
        "selected_artifact": (
            {
                key: artifacts[selected.method].get(key)
                for key in ("role", "method", "sha256", "provenance")
            }
            if selected.method in artifacts
            else None
        ),
        "configuration_lock_sha256": lock_sha256,
        "locked_configuration": single_config,
        "validation_evidence": {
            "manifest_sha256": hashes.get("manifest_sha256"),
            "dataset_snapshot_sha256": hashes.get("dataset_snapshot_sha256"),
            "source_config_sha256": hashes.get("config_sha256"),
        },
        "selection_policy": [
            "zero validation-silence detections and complete artifact provenance",
            "aligned exact-word accuracy descending",
            "full-sequence exact rate descending",
            "boundary F1 descending",
            "event F1 descending",
            "non-keyboard false onsets per minute ascending",
            "gap-penalized MAE ascending",
            "method name ascending",
        ],
        "candidates": [asdict(candidate) for candidate in candidates],
    }


def load_and_select(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelSelectionError(f"unable to read validation result: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ModelSelectionError("evaluation result root must be an object")
    return select_production_method(value)


__all__ = ["CandidateScore", "ModelSelectionError", "load_and_select", "select_production_method"]
