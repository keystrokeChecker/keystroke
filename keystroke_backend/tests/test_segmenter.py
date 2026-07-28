from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from segmenter import auto_threshold, format_output, segment_into_words


def test_segment_into_words_returns_empty_results_for_no_onsets() -> None:
    counts, groups = segment_into_words(np.array([], dtype=float))

    assert counts == []
    assert groups == []


def test_segment_into_words_splits_only_when_gap_exceeds_threshold() -> None:
    counts, groups = segment_into_words(
        [0.0, 1.0, 2.01, 2.5, 4.0],
        gap_threshold=1.0,
    )

    assert counts == [2, 2, 1]
    assert groups == [[0.0, 1.0], [2.01, 2.5], [4.0]]


@pytest.mark.parametrize("onsets", [[], [0.25], [0.25, 1.25]])
def test_auto_threshold_falls_back_when_there_are_too_few_onsets(onsets) -> None:
    assert auto_threshold(onsets) == pytest.approx(0.5)


def test_auto_threshold_uses_largest_relative_gap_jump() -> None:
    threshold = auto_threshold([0.0, 0.1, 0.2, 1.2])

    assert threshold == pytest.approx(np.sqrt(0.1 * 1.0))


@pytest.mark.parametrize(
    ("onsets", "expected"),
    [
        ([0.0, 0.001, 0.003], 0.15),
        ([0.0, 1.0, 10.0], 2.0),
    ],
)
def test_auto_threshold_clips_to_configured_bounds(onsets, expected) -> None:
    assert auto_threshold(onsets) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ([], ""),
        ([3], "3"),
        ([3, 7, 1], "3|7|1"),
        (np.array([2, 4], dtype=np.int64), "2|4"),
    ],
)
def test_format_output_joins_counts_with_pipe(counts, expected) -> None:
    assert format_output(counts) == expected
