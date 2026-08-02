from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.training_protocol import (
    TrainingProtocolError,
    artifact_provenance,
    load_training_plan,
)


def _write_manifest(directory: Path) -> Path:
    recordings = []
    for split in ("train", "validation", "test"):
        recording_id = f"{split}-typing"
        wav = directory / f"{recording_id}.wav"
        log = directory / f"{recording_id}_log.csv"
        meta = directory / f"{recording_id}_meta.json"
        wav.write_bytes(b"fixture")
        log.write_text("timestamp_sec,key,is_word_boundary\n", encoding="utf-8")
        meta.write_text('{"sync_offset_ms": 0}', encoding="utf-8")
        recordings.append(
            {
                "id": recording_id,
                "wav": wav.name,
                "log": log.name,
                "meta": meta.name,
                "split": split,
                "setup_id": f"setup-{split}",
            }
        )
    path = directory / "manifest.json"
    path.write_text(
        json.dumps({"schema_version": 1, "recordings": recordings}),
        encoding="utf-8",
    )
    return path


def test_training_plan_uses_only_train_and_embeds_complete_provenance(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path)
    quality_ok = lambda manifest: SimpleNamespace(valid=True, recordings=(), issues=())

    plan = load_training_plan(path, quality_validator=quality_ok)
    provenance = artifact_provenance(
        plan,
        artifact_role="count_predictor",
        fit_config={"random_state": 42},
        trained_at="2026-07-29T12:00:00Z",
    )

    assert plan.training_recording_ids == ("train-typing",)
    assert plan.validation_recording_ids == ("validation-typing",)
    assert "test-typing" not in json.dumps(provenance)
    assert provenance["manifest_hash"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert provenance["training_setup_ids"] == ["setup-train"]
    assert provenance["fit_config"] == {"random_state": 42}


def test_training_plan_refuses_failed_dataset_quality(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path)
    quality_bad = lambda manifest: SimpleNamespace(
        valid=False,
        recordings=(SimpleNamespace(valid=False),),
        issues=(object(), object()),
    )

    with pytest.raises(TrainingProtocolError, match="quality validation failed"):
        load_training_plan(path, quality_validator=quality_bad)
