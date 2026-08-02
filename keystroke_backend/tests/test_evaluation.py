from __future__ import annotations

from io import StringIO

import pytest

from src.evaluation import (
    AlignmentKind,
    BoundaryInterval,
    KeylogColumns,
    KeylogSchemaError,
    aggregate_boundary_metrics,
    aggregate_event_metrics,
    aggregate_sequence_metrics,
    align_count_sequences,
    match_boundaries,
    match_events,
    parse_keylog,
    predicted_boundary_intervals,
    score_count_sequence,
)


def test_parse_keylog_excludes_separators_offsets_times_and_preserves_final_word():
    source = StringIO(
        "time,key,separator,ignored\n"
        "0.100,a,false,x\n"
        "0.200,b,0,x\n"
        "0.300,Key.space,true,x\n"
        "0.400,Key.space,yes,x\n"
        "0.500,c,false,x\n"
        "0.600,d,no,x\n"
    )

    truth = parse_keylog(
        source,
        sync_offset_ms=25.0,
        columns=KeylogColumns(
            timestamp="time", key="key", boundary="separator"
        ),
    )

    assert truth.counts == (2, 2)
    assert tuple(event.key for event in truth.events) == ("a", "b", "c", "d")
    assert truth.event_times == pytest.approx((0.125, 0.225, 0.525, 0.625))
    assert tuple(event.word_index for event in truth.events) == (0, 0, 1, 1)
    assert truth.separator_times == pytest.approx((0.325, 0.425))
    # Repeated separators represent one internal word boundary. The first
    # separator after the preceding word is retained as its timestamp.
    assert truth.boundary_times == pytest.approx((0.325,))
    assert truth.words[0].separator_time_seconds == pytest.approx(0.325)
    assert truth.words[1].separator_time_seconds is None
    assert truth.sync_offset_seconds == pytest.approx(0.025)


def test_parse_keylog_ignores_leading_and_terminal_separators_as_boundaries():
    truth = parse_keylog(
        StringIO(
            "timestamp_sec,key,is_word_boundary\n"
            "0.1,Key.space,true\n"
            "0.2,a,false\n"
            "0.3,Key.space,true\n"
        )
    )

    assert truth.counts == (1,)
    assert truth.event_times == (0.2,)
    assert truth.boundary_times == ()
    assert truth.separator_times == (0.1, 0.3)


@pytest.mark.parametrize(
    ("csv_text", "message"),
    [
        (
            "timestamp_sec,key\n0.1,a\n",
            "missing required column",
        ),
        (
            "timestamp_sec,key,is_word_boundary\n0.1,a,maybe\n",
            "invalid boolean",
        ),
        (
            "timestamp_sec,key,is_word_boundary\nnot-a-time,a,false\n",
            "non-numeric timestamp",
        ),
        (
            "timestamp_sec,key,is_word_boundary\n0.2,a,false\n0.1,b,false\n",
            "earlier than the preceding row",
        ),
    ],
)
def test_parse_keylog_reports_explicit_schema_errors(csv_text: str, message: str):
    with pytest.raises(KeylogSchemaError, match=message):
        parse_keylog(StringIO(csv_text))


def test_parse_keylog_rejects_conflicting_column_configuration():
    with pytest.raises(ValueError, match="either columns"):
        parse_keylog(
            StringIO("time,key,boundary\n"),
            timestamp_column="time",
            columns=KeylogColumns("time", "key", "boundary"),
        )


def test_alignment_and_sequence_score_penalize_all_edit_operations():
    alignment = align_count_sequences([3, 7], [3, 5, 2])

    assert [operation.kind for operation in alignment.operations] == [
        AlignmentKind.MATCH,
        AlignmentKind.SUBSTITUTION,
        AlignmentKind.INSERTION,
    ]
    assert alignment.aligned_true_counts == (3, 7, None)
    assert alignment.aligned_predicted_counts == (3, 5, 2)
    assert alignment.edit_distance == 2
    assert alignment.gap_count == 1
    assert alignment.total_absolute_count_error == 4

    metrics = score_count_sequence([3, 7], [3, 5, 2])
    assert metrics.matches == 1
    assert metrics.substitutions == 1
    assert metrics.deletions == 0
    assert metrics.insertions == 1
    assert metrics.aligned_items == 3
    assert metrics.aligned_accuracy == pytest.approx(1 / 3)
    assert metrics.gap_penalized_mae == pytest.approx(4 / 3)
    assert metrics.full_sequence_exact is False
    # Total count is equal even though aligned per-word counts are wrong.
    assert metrics.total_count_error == 0
    assert metrics.total_absolute_count_error == 0
    assert metrics.aligned_absolute_count_error == 4
    assert metrics.signed_total_count_delta == 0
    assert metrics.gap_absolute_error == 4


def test_alignment_prefers_fewer_gaps_before_absolute_error():
    alignment = align_count_sequences([1, 2], [2, 1])

    assert [operation.kind for operation in alignment.operations] == [
        AlignmentKind.SUBSTITUTION,
        AlignmentKind.SUBSTITUTION,
    ]
    assert alignment.gap_count == 0


def test_alignment_uses_lower_gap_penalized_absolute_error_after_gap_count():
    alignment = align_count_sequences([100, 1], [50])

    # Both possible paths have two edits and one gap. Substituting 100 -> 50
    # and deleting 1 has error 51, versus deleting 100 and substituting 1 ->
    # 50 with error 149.
    assert [operation.kind for operation in alignment.operations] == [
        AlignmentKind.SUBSTITUTION,
        AlignmentKind.DELETION,
    ]
    assert alignment.total_absolute_count_error == 51


def test_alignment_uses_diagonal_then_delete_then_insert_as_final_tie_break():
    alignment = align_count_sequences([1, 1], [2])

    # The two paths have identical edits, gaps, and absolute error. Choosing a
    # diagonal at the final cell means the deletion appears first.
    assert [operation.kind for operation in alignment.operations] == [
        AlignmentKind.DELETION,
        AlignmentKind.SUBSTITUTION,
    ]


def test_sequence_metrics_define_empty_and_missing_sequence_behavior():
    empty = score_count_sequence([], [])
    assert empty.aligned_items == 0
    assert empty.aligned_accuracy == 1.0
    assert empty.gap_penalized_mae == 0.0
    assert empty.full_sequence_exact is True

    missing = score_count_sequence([4], [])
    assert missing.deletions == 1
    assert missing.aligned_accuracy == 0.0
    assert missing.gap_penalized_mae == 4.0
    assert missing.total_count_error == 4
    assert missing.signed_total_count_delta == -4


def test_alignment_requires_integer_tokens():
    with pytest.raises(TypeError, match="integer token"):
        align_count_sequences([1, 2.5], [1, 2])
    with pytest.raises(TypeError, match="not bool"):
        align_count_sequences([True], [1])


def test_event_matching_maximizes_matches_then_minimizes_timing_error():
    metrics = match_events([0.0, 0.09], [0.08], tolerance_seconds=0.08)

    assert metrics.matches == 1
    assert metrics.matched_pairs[0].true_index == 1
    assert metrics.matched_pairs[0].predicted_index == 0
    assert metrics.matched_pairs[0].absolute_timing_error_seconds == pytest.approx(
        0.01
    )
    assert metrics.true_positives == 1
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 1
    assert metrics.precision == 1.0
    assert metrics.recall == 0.5
    assert metrics.f1 == pytest.approx(2 / 3)


def test_event_matching_is_one_to_one_ordered_and_inclusive_at_tolerance():
    metrics = match_events(
        [0.0, 0.10], [0.06, 0.11], tolerance_seconds=0.06
    )

    assert [(pair.true_index, pair.predicted_index) for pair in metrics.matched_pairs] == [
        (0, 0),
        (1, 1),
    ]
    assert metrics.matches == 2
    assert metrics.total_absolute_timing_error_seconds == pytest.approx(0.07)
    assert metrics.p95_absolute_timing_error_seconds == pytest.approx(0.0575)
    assert metrics.precision == metrics.recall == metrics.f1 == 1.0


def test_event_matching_handles_empty_sequences_and_rejects_unordered_input():
    empty = match_events([], [])
    assert empty.matches == 0
    assert empty.precision == empty.recall == empty.f1 == 1.0
    assert empty.mean_absolute_timing_error_seconds is None
    assert empty.p95_absolute_timing_error_seconds is None
    assert empty.max_absolute_timing_error_seconds is None

    with pytest.raises(ValueError, match="ordered"):
        match_events([0.2, 0.1], [0.1])


def test_event_and_boundary_matching_include_decimal_values_at_exact_limit():
    event_metrics = match_events([0.06], [0.14], tolerance_seconds=0.08)
    assert event_metrics.true_positives == 1

    boundary_metrics = match_boundaries(
        [0.14],
        [BoundaryInterval(0.20, 0.30)],
        slack_seconds=0.06,
    )
    assert boundary_metrics.true_positives == 1


def test_event_metrics_micro_aggregation_includes_empty_sessions_and_p95():
    metrics = aggregate_event_metrics(
        [
            match_events([], []),
            match_events([0.0, 0.1], [0.01, 0.12, 0.4], 0.05),
        ]
    )

    assert metrics.session_count == 2
    assert metrics.true_count == 2
    assert metrics.predicted_count == 3
    assert metrics.true_positives == 2
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 0
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == 1.0
    assert metrics.f1 == pytest.approx(0.8)
    assert metrics.mean_absolute_timing_error_seconds == pytest.approx(0.015)
    assert metrics.p95_absolute_timing_error_seconds == pytest.approx(0.0195)

    no_matches = aggregate_event_metrics([match_events([0.0], [1.0], 0.01)])
    assert no_matches.mean_absolute_timing_error_seconds is None
    assert no_matches.p95_absolute_timing_error_seconds is None
    assert no_matches.max_absolute_timing_error_seconds is None


def test_predicted_boundary_intervals_use_adjacent_word_event_ranges():
    intervals = predicted_boundary_intervals(
        [[0.1, 0.2], [0.6, 0.7], [1.1]]
    )

    assert intervals == (
        BoundaryInterval(0.2, 0.6, 0, 1),
        BoundaryInterval(0.7, 1.1, 1, 2),
    )


def test_predicted_boundary_intervals_reject_empty_or_overlapping_groups():
    with pytest.raises(ValueError, match="must not be empty"):
        predicted_boundary_intervals([[0.1], []])
    with pytest.raises(ValueError, match="globally ordered"):
        predicted_boundary_intervals([[0.2, 0.4], [0.3, 0.5]])


def test_boundary_matching_reports_precision_recall_f1_and_slack_distance():
    intervals = predicted_boundary_intervals(
        [[0.1, 0.2], [0.6, 0.7], [1.1]]
    )
    metrics = match_boundaries(
        [0.4, 0.68, 2.0], intervals, slack_seconds=0.03
    )

    assert metrics.matches == 2
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 1
    assert metrics.precision == 1.0
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(0.8)
    assert metrics.total_distance_seconds == pytest.approx(0.02)
    assert metrics.matched_pairs[0].distance_seconds == 0.0
    assert metrics.matched_pairs[1].distance_seconds == pytest.approx(0.02)


def test_boundary_matching_is_one_to_one_and_minimizes_interval_distance():
    metrics = match_boundaries(
        [0.19, 0.21], [BoundaryInterval(0.2, 0.3)], slack_seconds=0.02
    )

    assert metrics.matches == 1
    assert metrics.matched_pairs[0].true_index == 1
    assert metrics.matched_pairs[0].distance_seconds == 0.0
    assert metrics.precision == 1.0
    assert metrics.recall == 0.5


def test_boundary_matching_treats_two_empty_inputs_as_exact_detection():
    metrics = match_boundaries([], [])
    assert metrics.matches == 0
    assert metrics.precision == metrics.recall == metrics.f1 == 1.0


def test_boundary_metrics_micro_aggregation_includes_empty_sessions():
    metrics = aggregate_boundary_metrics(
        [
            match_boundaries([], []),
            match_boundaries(
                [0.25, 0.8],
                [BoundaryInterval(0.2, 0.3)],
                slack_seconds=0.0,
            ),
        ]
    )

    assert metrics.session_count == 2
    assert metrics.true_count == 2
    assert metrics.predicted_count == 1
    assert metrics.true_positives == 1
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 1
    assert metrics.precision == 1.0
    assert metrics.recall == 0.5
    assert metrics.f1 == pytest.approx(2 / 3)


def test_micro_aggregation_includes_empty_sequence_sessions():
    metrics = aggregate_sequence_metrics(
        [
            score_count_sequence([], []),
            score_count_sequence([3], [4, 2]),
            score_count_sequence([5], [5]),
        ]
    )

    assert metrics.session_count == 3
    assert metrics.exact_sequence_count == 2
    assert metrics.full_sequence_exact_rate == pytest.approx(2 / 3)
    assert metrics.true_word_count == 2
    assert metrics.predicted_word_count == 3
    assert metrics.matches == 1
    assert metrics.substitutions == 1
    assert metrics.insertions == 1
    assert metrics.deletions == 0
    assert metrics.aligned_items == 3
    assert metrics.aligned_accuracy == pytest.approx(1 / 3)
    assert metrics.gap_absolute_error == 3
    assert metrics.gap_penalized_mae == 1.0
    assert metrics.total_count_error == 3
    assert metrics.signed_total_count_delta == 3


def test_micro_aggregation_keeps_all_empty_sessions_in_exact_sequence_rate():
    metrics = aggregate_sequence_metrics(
        [score_count_sequence([], []), score_count_sequence([], [])]
    )

    assert metrics.session_count == 2
    assert metrics.exact_sequence_count == 2
    assert metrics.full_sequence_exact_rate == 1.0
    assert metrics.aligned_accuracy == 1.0
    assert metrics.gap_penalized_mae == 0.0


def test_micro_total_count_error_does_not_cancel_between_sessions():
    metrics = aggregate_sequence_metrics(
        [score_count_sequence([3], [5]), score_count_sequence([4], [2])]
    )

    assert metrics.total_count_error == 4
    assert metrics.signed_total_count_delta == 0


def test_micro_aggregation_of_no_sessions_is_explicitly_empty():
    metrics = aggregate_sequence_metrics([])
    assert metrics.session_count == 0
    assert metrics.full_sequence_exact_rate == 0.0
    assert metrics.aligned_accuracy == 0.0
    assert metrics.gap_penalized_mae == 0.0
