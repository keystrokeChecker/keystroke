from __future__ import annotations

import json

import numpy as np
import pytest

from onset_detector import evaluate_against_ground_truth
from train_model import parse_ground_truth
from tune_and_evaluate import load_ground_truth


def _write_keylog(tmp_path):
    log_path = tmp_path / "fixture_log.csv"
    log_path.write_text(
        "timestamp_sec,key,is_word_boundary\n"
        "0.100,a,False\n"
        "0.200,Key.space,True\n"
        "0.300,b,False\n"
        "0.400,Key.space,True\n"
        "0.500,Key.space,True\n"
        "0.600,c,False\n",
        encoding="utf-8",
    )
    return log_path


def test_legacy_parsers_exclude_separator_rows_and_apply_capture_offset(tmp_path):
    log_path = _write_keylog(tmp_path)
    # The conventional sibling metadata name is fixture_meta.json.
    (tmp_path / "fixture_meta.json").write_text(
        json.dumps({"sync_offset_ms": -50.0}), encoding="utf-8"
    )

    counts, starts, ends = parse_ground_truth(str(log_path))
    assert counts == [1, 1, 1]
    assert starts == pytest.approx([0.05, 0.25, 0.55])
    assert ends == pytest.approx([0.05, 0.25, 0.55])

    tune_counts, event_times = load_ground_truth(str(log_path))
    assert tune_counts == counts
    assert event_times == pytest.approx([0.05, 0.25, 0.55])


def test_legacy_parser_explicit_offset_overrides_metadata(tmp_path):
    log_path = _write_keylog(tmp_path)
    (tmp_path / "fixture_meta.json").write_text(
        json.dumps({"sync_offset_ms": -50.0}), encoding="utf-8"
    )

    counts, starts, ends = parse_ground_truth(str(log_path), sync_offset_ms=100.0)
    assert counts == [1, 1, 1]
    assert starts == pytest.approx([0.2, 0.4, 0.7])
    assert ends == pytest.approx([0.2, 0.4, 0.7])

    tune_counts, event_times = load_ground_truth(str(log_path), 100.0)
    assert tune_counts == counts
    assert event_times == pytest.approx([0.2, 0.4, 0.7])


def test_onset_compatibility_wrapper_keeps_legacy_result_shape_and_rounding():
    result = evaluate_against_ground_truth(
        np.array([0.00, 0.11, 0.40]),
        [0.02, 0.10, 0.30],
        tolerance=0.05,
    )

    assert set(result) == {
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1",
    }
    assert result["true_positives"] == 2
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["precision"] == round(2 / 3, 3)
    assert result["recall"] == round(2 / 3, 3)
    assert result["f1"] == round(2 / 3, 3)
