"""Deterministic, production-aligned evaluation primitives.

This module deliberately depends only on the Python standard library.  It is
shared by evaluation scripts so count, onset, and word-boundary results use
one definition instead of each script implementing a subtly different metric.
"""

from __future__ import annotations

import csv
import math
import operator
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Callable, Iterable, Sequence, TextIO, TypeVar


class KeylogSchemaError(ValueError):
    """Raised when a keylog does not conform to the configured CSV schema."""


@dataclass(frozen=True, slots=True)
class KeylogColumns:
    """Column names used by :func:`parse_keylog`."""

    timestamp: str = "timestamp_sec"
    key: str = "key"
    boundary: str = "is_word_boundary"


@dataclass(frozen=True, slots=True)
class TruthEvent:
    """One non-separator ground-truth keystroke in audio time."""

    timestamp_seconds: float
    key: str
    row_number: int
    word_index: int

    @property
    def timestamp(self) -> float:
        """Compatibility alias for ``timestamp_seconds``."""

        return self.timestamp_seconds

    @property
    def audio_time_seconds(self) -> float:
        return self.timestamp_seconds


@dataclass(frozen=True, slots=True)
class TruthWord:
    """A non-empty word assembled from consecutive non-separator events."""

    index: int
    events: tuple[TruthEvent, ...]
    separator_time_seconds: float | None = None

    @property
    def count(self) -> int:
        return len(self.events)

    @property
    def start_time_seconds(self) -> float:
        if not self.events:
            raise ValueError("A truth word has no events")
        return self.events[0].timestamp_seconds

    @property
    def end_time_seconds(self) -> float:
        if not self.events:
            raise ValueError("A truth word has no events")
        return self.events[-1].timestamp_seconds

    @property
    def event_times(self) -> tuple[float, ...]:
        return tuple(event.timestamp_seconds for event in self.events)


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """Parsed keylog truth in the audio recording's time coordinate system.

    ``separator_times`` retains every separator row. ``boundary_times`` only
    contains separators that occur between two non-empty words; leading,
    trailing, and repeated separators therefore cannot create phantom word
    boundaries.
    """

    words: tuple[TruthWord, ...]
    events: tuple[TruthEvent, ...]
    boundary_times: tuple[float, ...]
    separator_times: tuple[float, ...]
    sync_offset_seconds: float

    @property
    def counts(self) -> tuple[int, ...]:
        return tuple(word.count for word in self.words)

    @property
    def word_counts(self) -> tuple[int, ...]:
        return self.counts

    @property
    def event_times(self) -> tuple[float, ...]:
        return tuple(event.timestamp_seconds for event in self.events)


_TRUE_VALUES = frozenset({"1", "true", "t", "yes", "y"})
_FALSE_VALUES = frozenset({"0", "false", "f", "no", "n"})


def _parse_boundary(raw_value: object, *, row_number: int, column: str) -> bool:
    if raw_value is None:
        raise KeylogSchemaError(
            f"Row {row_number} is missing a value for boundary column {column!r}"
        )
    normalized = str(raw_value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise KeylogSchemaError(
        f"Row {row_number} has invalid boolean {raw_value!r} in column "
        f"{column!r}; expected one of true/false, 1/0, or yes/no"
    )


def _parse_timestamp(raw_value: object, *, row_number: int, column: str) -> float:
    if raw_value is None or not str(raw_value).strip():
        raise KeylogSchemaError(
            f"Row {row_number} is missing a timestamp in column {column!r}"
        )
    try:
        timestamp = float(str(raw_value).strip())
    except ValueError as exc:
        raise KeylogSchemaError(
            f"Row {row_number} has a non-numeric timestamp {raw_value!r} "
            f"in column {column!r}"
        ) from exc
    if not math.isfinite(timestamp):
        raise KeylogSchemaError(
            f"Row {row_number} has a non-finite timestamp in column {column!r}"
        )
    return timestamp


def parse_keylog(
    source: str | PathLike[str] | TextIO,
    *,
    sync_offset_ms: float = 0.0,
    timestamp_column: str = "timestamp_sec",
    key_column: str = "key",
    boundary_column: str = "is_word_boundary",
    columns: KeylogColumns | None = None,
) -> GroundTruth:
    """Parse a keylog CSV into word and event ground truth.

    Boundary/separator rows split words but are never counted as keystrokes.
    The final non-empty word is retained even when the file has no trailing
    separator. Every timestamp is converted from keylogger-relative time to
    audio time using ``timestamp + sync_offset_ms / 1000``.

    ``source`` may be a filesystem path or an already-open text stream. Column
    names can be configured either individually or with ``KeylogColumns``.
    Malformed headers, values, or timestamp order raise ``KeylogSchemaError``.
    """

    if columns is not None:
        defaults = KeylogColumns()
        if (
            timestamp_column != defaults.timestamp
            or key_column != defaults.key
            or boundary_column != defaults.boundary
        ):
            raise ValueError(
                "Pass either columns=KeylogColumns(...) or individual column "
                "names, not both"
            )
        timestamp_column = columns.timestamp
        key_column = columns.key
        boundary_column = columns.boundary

    configured_columns = (timestamp_column, key_column, boundary_column)
    if any(not isinstance(name, str) or not name for name in configured_columns):
        raise ValueError("Configured keylog column names must be non-empty strings")
    if len(set(configured_columns)) != len(configured_columns):
        raise ValueError("Configured keylog column names must be distinct")

    try:
        offset_seconds = float(sync_offset_ms) / 1000.0
    except (TypeError, ValueError) as exc:
        raise ValueError("sync_offset_ms must be a finite number") from exc
    if not math.isfinite(offset_seconds):
        raise ValueError("sync_offset_ms must be a finite number")

    stream: TextIO
    should_close = False
    if isinstance(source, (str, PathLike)):
        stream = Path(source).open("r", encoding="utf-8-sig", newline="")
        should_close = True
    elif hasattr(source, "read"):
        stream = source
    else:
        raise TypeError("source must be a path or an open text stream")

    try:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise KeylogSchemaError("Keylog CSV is missing a header row")
        if len(fieldnames) != len(set(fieldnames)):
            duplicates = sorted(
                name for name in set(fieldnames) if fieldnames.count(name) > 1
            )
            raise KeylogSchemaError(
                f"Keylog CSV contains duplicate columns: {', '.join(duplicates)}"
            )
        missing = [name for name in configured_columns if name not in fieldnames]
        if missing:
            raise KeylogSchemaError(
                "Keylog CSV is missing required column(s): " + ", ".join(missing)
            )

        words: list[TruthWord] = []
        events: list[TruthEvent] = []
        current_events: list[TruthEvent] = []
        separator_times: list[float] = []
        boundary_times: list[float] = []
        pending_boundary_time: float | None = None
        previous_time: float | None = None

        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise KeylogSchemaError(
                    f"Row {row_number} has more fields than the CSV header"
                )
            source_timestamp = _parse_timestamp(
                row.get(timestamp_column),
                row_number=row_number,
                column=timestamp_column,
            )
            audio_timestamp = source_timestamp + offset_seconds
            if previous_time is not None and audio_timestamp < previous_time:
                raise KeylogSchemaError(
                    f"Row {row_number} timestamp is earlier than the preceding row"
                )
            previous_time = audio_timestamp

            is_boundary = _parse_boundary(
                row.get(boundary_column),
                row_number=row_number,
                column=boundary_column,
            )
            raw_key = row.get(key_column)
            if raw_key is None:
                raise KeylogSchemaError(
                    f"Row {row_number} is missing a value for key column "
                    f"{key_column!r}"
                )

            if is_boundary:
                separator_times.append(audio_timestamp)
                if current_events:
                    words.append(
                        TruthWord(
                            index=len(words),
                            events=tuple(current_events),
                            separator_time_seconds=audio_timestamp,
                        )
                    )
                    current_events = []
                    # The first separator after a word is the representative
                    # boundary if another non-empty word follows.
                    pending_boundary_time = audio_timestamp
                continue

            if pending_boundary_time is not None and words:
                boundary_times.append(pending_boundary_time)
                pending_boundary_time = None

            event = TruthEvent(
                timestamp_seconds=audio_timestamp,
                key=str(raw_key),
                row_number=row_number,
                word_index=len(words),
            )
            current_events.append(event)
            events.append(event)

        if current_events:
            words.append(
                TruthWord(
                    index=len(words),
                    events=tuple(current_events),
                    separator_time_seconds=None,
                )
            )

        return GroundTruth(
            words=tuple(words),
            events=tuple(events),
            boundary_times=tuple(boundary_times),
            separator_times=tuple(separator_times),
            sync_offset_seconds=offset_seconds,
        )
    finally:
        if should_close:
            stream.close()


class AlignmentKind(str, Enum):
    MATCH = "match"
    SUBSTITUTION = "substitution"
    DELETION = "deletion"
    INSERTION = "insertion"


@dataclass(frozen=True, slots=True)
class AlignmentOperation:
    """One operation in an aligned true/predicted count sequence."""

    kind: AlignmentKind
    true_index: int | None
    predicted_index: int | None
    true_count: int | None
    predicted_count: int | None

    @property
    def operation(self) -> str:
        return self.kind.value

    @property
    def true_value(self) -> int | None:
        return self.true_count

    @property
    def predicted_value(self) -> int | None:
        return self.predicted_count

    @property
    def edit_cost(self) -> int:
        return 0 if self.kind is AlignmentKind.MATCH else 1

    @property
    def is_gap(self) -> bool:
        return self.kind in (AlignmentKind.DELETION, AlignmentKind.INSERTION)

    @property
    def absolute_count_error(self) -> int:
        return abs((self.true_count or 0) - (self.predicted_count or 0))


@dataclass(frozen=True, slots=True)
class SequenceAlignment:
    """A deterministic Levenshtein alignment over integer count tokens."""

    true_counts: tuple[int, ...]
    predicted_counts: tuple[int, ...]
    operations: tuple[AlignmentOperation, ...]
    edit_distance: int
    gap_count: int
    total_absolute_count_error: int

    @property
    def matches(self) -> int:
        return sum(op.kind is AlignmentKind.MATCH for op in self.operations)

    @property
    def substitutions(self) -> int:
        return sum(op.kind is AlignmentKind.SUBSTITUTION for op in self.operations)

    @property
    def deletions(self) -> int:
        return sum(op.kind is AlignmentKind.DELETION for op in self.operations)

    @property
    def insertions(self) -> int:
        return sum(op.kind is AlignmentKind.INSERTION for op in self.operations)

    @property
    def aligned_length(self) -> int:
        return len(self.operations)

    @property
    def distance(self) -> int:
        return self.edit_distance

    @property
    def aligned_true_counts(self) -> tuple[int | None, ...]:
        return tuple(operation.true_count for operation in self.operations)

    @property
    def aligned_predicted_counts(self) -> tuple[int | None, ...]:
        return tuple(operation.predicted_count for operation in self.operations)


def _coerce_integer_tokens(values: Iterable[int], *, name: str) -> tuple[int, ...]:
    coerced: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool):
            raise TypeError(f"{name}[{index}] must be an integer token, not bool")
        try:
            coerced.append(int(operator.index(value)))
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name}[{index}] must be an integer token") from exc
    return tuple(coerced)


def align_count_sequences(
    true_counts: Iterable[int], predicted_counts: Iterable[int]
) -> SequenceAlignment:
    """Align integer sequences with deterministic Levenshtein tie-breaking.

    Candidate paths minimize, in order: edit cost, number of gaps, total
    gap-to-zero/count absolute error, and current operation preference
    diagonal (match/substitution), deletion, then insertion.
    """

    truth = _coerce_integer_tokens(true_counts, name="true_counts")
    predictions = _coerce_integer_tokens(
        predicted_counts, name="predicted_counts"
    )
    true_length = len(truth)
    predicted_length = len(predictions)

    # Every cell stores (edit cost, gap count, absolute count error).
    scores: list[list[tuple[int, int, int] | None]] = [
        [None] * (predicted_length + 1) for _ in range(true_length + 1)
    ]
    choices: list[list[AlignmentKind | None]] = [
        [None] * (predicted_length + 1) for _ in range(true_length + 1)
    ]
    scores[0][0] = (0, 0, 0)

    for true_index in range(true_length + 1):
        for predicted_index in range(predicted_length + 1):
            if true_index == 0 and predicted_index == 0:
                continue
            candidates: list[
                tuple[tuple[int, int, int, int], tuple[int, int, int], AlignmentKind]
            ] = []

            if true_index > 0 and predicted_index > 0:
                previous = scores[true_index - 1][predicted_index - 1]
                assert previous is not None
                is_match = truth[true_index - 1] == predictions[predicted_index - 1]
                kind = (
                    AlignmentKind.MATCH
                    if is_match
                    else AlignmentKind.SUBSTITUTION
                )
                score = (
                    previous[0] + (0 if is_match else 1),
                    previous[1],
                    previous[2]
                    + abs(truth[true_index - 1] - predictions[predicted_index - 1]),
                )
                candidates.append((score + (0,), score, kind))

            if true_index > 0:
                previous = scores[true_index - 1][predicted_index]
                assert previous is not None
                score = (
                    previous[0] + 1,
                    previous[1] + 1,
                    previous[2] + abs(truth[true_index - 1]),
                )
                candidates.append((score + (1,), score, AlignmentKind.DELETION))

            if predicted_index > 0:
                previous = scores[true_index][predicted_index - 1]
                assert previous is not None
                score = (
                    previous[0] + 1,
                    previous[1] + 1,
                    previous[2] + abs(predictions[predicted_index - 1]),
                )
                candidates.append((score + (2,), score, AlignmentKind.INSERTION))

            _, selected_score, selected_kind = min(
                candidates, key=lambda candidate: candidate[0]
            )
            scores[true_index][predicted_index] = selected_score
            choices[true_index][predicted_index] = selected_kind

    operations: list[AlignmentOperation] = []
    true_index = true_length
    predicted_index = predicted_length
    while true_index or predicted_index:
        kind = choices[true_index][predicted_index]
        assert kind is not None
        if kind in (AlignmentKind.MATCH, AlignmentKind.SUBSTITUTION):
            operations.append(
                AlignmentOperation(
                    kind=kind,
                    true_index=true_index - 1,
                    predicted_index=predicted_index - 1,
                    true_count=truth[true_index - 1],
                    predicted_count=predictions[predicted_index - 1],
                )
            )
            true_index -= 1
            predicted_index -= 1
        elif kind is AlignmentKind.DELETION:
            operations.append(
                AlignmentOperation(
                    kind=kind,
                    true_index=true_index - 1,
                    predicted_index=None,
                    true_count=truth[true_index - 1],
                    predicted_count=None,
                )
            )
            true_index -= 1
        else:
            operations.append(
                AlignmentOperation(
                    kind=kind,
                    true_index=None,
                    predicted_index=predicted_index - 1,
                    true_count=None,
                    predicted_count=predictions[predicted_index - 1],
                )
            )
            predicted_index -= 1

    operations.reverse()
    final_score = scores[true_length][predicted_length]
    assert final_score is not None
    return SequenceAlignment(
        true_counts=truth,
        predicted_counts=predictions,
        operations=tuple(operations),
        edit_distance=final_score[0],
        gap_count=final_score[1],
        total_absolute_count_error=final_score[2],
    )


@dataclass(frozen=True, slots=True)
class SequenceMetrics:
    alignment: SequenceAlignment
    matches: int
    substitutions: int
    deletions: int
    insertions: int
    aligned_items: int
    aligned_accuracy: float
    gap_penalized_mae: float
    full_sequence_exact: bool
    total_count_error: int
    signed_total_count_delta: int
    gap_absolute_error: int
    true_word_count: int
    predicted_word_count: int

    @property
    def exact_aligned_accuracy(self) -> float:
        return self.aligned_accuracy

    @property
    def aligned_exact_accuracy(self) -> float:
        return self.aligned_accuracy

    @property
    def full_sequence_exact_match(self) -> bool:
        return self.full_sequence_exact

    @property
    def total_absolute_count_error(self) -> int:
        """Absolute error in total keystrokes, irrespective of alignment."""

        return self.total_count_error

    @property
    def absolute_total_keystroke_error(self) -> int:
        return self.total_count_error

    @property
    def aligned_absolute_count_error(self) -> int:
        """Absolute-error numerator used by ``gap_penalized_mae``."""

        return self.gap_absolute_error

    @property
    def word_count_delta(self) -> int:
        return self.predicted_word_count - self.true_word_count


def score_count_sequence(
    true_counts: Iterable[int], predicted_counts: Iterable[int]
) -> SequenceMetrics:
    """Score one count sequence, penalizing every insertion and deletion."""

    alignment = align_count_sequences(true_counts, predicted_counts)
    matches = alignment.matches
    substitutions = alignment.substitutions
    deletions = alignment.deletions
    insertions = alignment.insertions
    aligned_items = len(alignment.operations)
    aligned_accuracy = matches / aligned_items if aligned_items else 1.0
    gap_absolute_error = alignment.total_absolute_count_error
    gap_penalized_mae = (
        gap_absolute_error / aligned_items if aligned_items else 0.0
    )
    signed_delta = sum(alignment.predicted_counts) - sum(alignment.true_counts)

    return SequenceMetrics(
        alignment=alignment,
        matches=matches,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        aligned_items=aligned_items,
        aligned_accuracy=aligned_accuracy,
        gap_penalized_mae=gap_penalized_mae,
        full_sequence_exact=alignment.true_counts == alignment.predicted_counts,
        total_count_error=abs(signed_delta),
        signed_total_count_delta=signed_delta,
        gap_absolute_error=gap_absolute_error,
        true_word_count=len(alignment.true_counts),
        predicted_word_count=len(alignment.predicted_counts),
    )


@dataclass(frozen=True, slots=True)
class EventMatch:
    true_index: int
    predicted_index: int
    true_time_seconds: float
    predicted_time_seconds: float

    @property
    def timing_error_seconds(self) -> float:
        return self.predicted_time_seconds - self.true_time_seconds

    @property
    def absolute_timing_error_seconds(self) -> float:
        return abs(self.timing_error_seconds)


@dataclass(frozen=True, slots=True)
class EventMetrics:
    matched_pairs: tuple[EventMatch, ...]
    true_count: int
    predicted_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    total_absolute_timing_error_seconds: float
    mean_absolute_timing_error_seconds: float | None
    p95_absolute_timing_error_seconds: float | None
    max_absolute_timing_error_seconds: float | None

    @property
    def matches(self) -> int:
        return self.true_positives

    @property
    def tp(self) -> int:
        return self.true_positives

    @property
    def fp(self) -> int:
        return self.false_positives

    @property
    def fn(self) -> int:
        return self.false_negatives

    @property
    def total_timing_error_seconds(self) -> float:
        return self.total_absolute_timing_error_seconds

    @property
    def mean_timing_error_seconds(self) -> float | None:
        return self.mean_absolute_timing_error_seconds

    @property
    def timing_p95_seconds(self) -> float | None:
        return self.p95_absolute_timing_error_seconds


def _coerce_ordered_times(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    result: list[float] = []
    previous: float | None = None
    for index, value in enumerate(values):
        try:
            timestamp = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name}[{index}] must be a finite number") from exc
        if not math.isfinite(timestamp):
            raise ValueError(f"{name}[{index}] must be finite")
        if previous is not None and timestamp < previous:
            raise ValueError(f"{name} must be ordered from earliest to latest")
        result.append(timestamp)
        previous = timestamp
    return tuple(result)


def _nonnegative_finite(value: float, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _ordered_min_error_pairs(
    left_count: int,
    right_count: int,
    pair_error: Callable[[int, int], float | None],
) -> tuple[tuple[int, int], ...]:
    """Return ordered pairs maximizing matches, then minimizing total error."""

    match_counts = [[0] * (right_count + 1) for _ in range(left_count + 1)]
    errors = [[0.0] * (right_count + 1) for _ in range(left_count + 1)]
    choices: list[list[str | None]] = [
        [None] * (right_count + 1) for _ in range(left_count + 1)
    ]

    for left_index in range(left_count, -1, -1):
        for right_index in range(right_count, -1, -1):
            if left_index == left_count and right_index == right_count:
                continue
            candidates: list[tuple[tuple[int, float, int], int, float, str]] = []

            if left_index < left_count and right_index < right_count:
                error = pair_error(left_index, right_index)
                if error is not None:
                    matches = 1 + match_counts[left_index + 1][right_index + 1]
                    total_error = error + errors[left_index + 1][right_index + 1]
                    candidates.append(
                        ((-matches, total_error, 0), matches, total_error, "match")
                    )

            if left_index < left_count:
                matches = match_counts[left_index + 1][right_index]
                total_error = errors[left_index + 1][right_index]
                candidates.append(
                    ((-matches, total_error, 1), matches, total_error, "delete")
                )

            if right_index < right_count:
                matches = match_counts[left_index][right_index + 1]
                total_error = errors[left_index][right_index + 1]
                candidates.append(
                    ((-matches, total_error, 2), matches, total_error, "insert")
                )

            _, matches, total_error, choice = min(
                candidates, key=lambda candidate: candidate[0]
            )
            match_counts[left_index][right_index] = matches
            errors[left_index][right_index] = total_error
            choices[left_index][right_index] = choice

    pairs: list[tuple[int, int]] = []
    left_index = 0
    right_index = 0
    while left_index < left_count or right_index < right_count:
        choice = choices[left_index][right_index]
        assert choice is not None
        if choice == "match":
            pairs.append((left_index, right_index))
            left_index += 1
            right_index += 1
        elif choice == "delete":
            left_index += 1
        else:
            right_index += 1
    return tuple(pairs)


def _precision_recall_f1(
    true_positives: int, true_count: int, predicted_count: int
) -> tuple[float, float, float]:
    if true_count == 0 and predicted_count == 0:
        return 1.0, 1.0, 1.0
    precision = true_positives / predicted_count if predicted_count else 0.0
    recall = true_positives / true_count if true_count else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def _percentile(values: Iterable[float], quantile: float) -> float:
    """Return a deterministic linearly interpolated percentile."""

    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * quantile
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = rank - lower_index
    return ordered[lower_index] + fraction * (
        ordered[upper_index] - ordered[lower_index]
    )


def _within_inclusive_limit(value: float, limit: float) -> bool:
    """Compare finite seconds robustly at a documented inclusive boundary."""

    return value <= limit or math.isclose(
        value,
        limit,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def match_events(
    true_times: Iterable[float],
    predicted_times: Iterable[float],
    tolerance_seconds: float = 0.08,
) -> EventMetrics:
    """Match events one-to-one in order within a fixed inclusive tolerance.

    Dynamic programming first maximizes the number of matches and then
    minimizes total absolute timing error, avoiding greedy assignment errors.
    """

    truth = _coerce_ordered_times(true_times, name="true_times")
    predictions = _coerce_ordered_times(
        predicted_times, name="predicted_times"
    )
    tolerance = _nonnegative_finite(
        tolerance_seconds, name="tolerance_seconds"
    )

    def pair_error(true_index: int, predicted_index: int) -> float | None:
        error = abs(truth[true_index] - predictions[predicted_index])
        return error if _within_inclusive_limit(error, tolerance) else None

    indices = _ordered_min_error_pairs(len(truth), len(predictions), pair_error)
    matched_pairs = tuple(
        EventMatch(
            true_index=true_index,
            predicted_index=predicted_index,
            true_time_seconds=truth[true_index],
            predicted_time_seconds=predictions[predicted_index],
        )
        for true_index, predicted_index in indices
    )
    absolute_errors = [
        pair.absolute_timing_error_seconds for pair in matched_pairs
    ]
    true_positives = len(matched_pairs)
    false_positives = len(predictions) - true_positives
    false_negatives = len(truth) - true_positives
    precision, recall, f1 = _precision_recall_f1(
        true_positives, len(truth), len(predictions)
    )
    total_error = sum(absolute_errors)
    return EventMetrics(
        matched_pairs=matched_pairs,
        true_count=len(truth),
        predicted_count=len(predictions),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        total_absolute_timing_error_seconds=total_error,
        mean_absolute_timing_error_seconds=(
            total_error / true_positives if true_positives else None
        ),
        p95_absolute_timing_error_seconds=(
            _percentile(absolute_errors, 0.95) if absolute_errors else None
        ),
        max_absolute_timing_error_seconds=(
            max(absolute_errors) if absolute_errors else None
        ),
    )


@dataclass(frozen=True, slots=True)
class AggregateEventMetrics:
    session_count: int
    true_count: int
    predicted_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    total_absolute_timing_error_seconds: float
    mean_absolute_timing_error_seconds: float | None
    p95_absolute_timing_error_seconds: float | None
    max_absolute_timing_error_seconds: float | None

    @property
    def matches(self) -> int:
        return self.true_positives

    @property
    def tp(self) -> int:
        return self.true_positives

    @property
    def fp(self) -> int:
        return self.false_positives

    @property
    def fn(self) -> int:
        return self.false_negatives

    @property
    def timing_p95_seconds(self) -> float | None:
        return self.p95_absolute_timing_error_seconds


def aggregate_event_metrics(
    metrics: Iterable[EventMetrics],
) -> AggregateEventMetrics:
    """Micro-aggregate event detection counts and matched timing errors."""

    sessions = tuple(metrics)
    true_count = sum(metric.true_count for metric in sessions)
    predicted_count = sum(metric.predicted_count for metric in sessions)
    true_positives = sum(metric.true_positives for metric in sessions)
    false_positives = sum(metric.false_positives for metric in sessions)
    false_negatives = sum(metric.false_negatives for metric in sessions)
    absolute_errors = [
        pair.absolute_timing_error_seconds
        for metric in sessions
        for pair in metric.matched_pairs
    ]
    total_error = sum(absolute_errors)
    precision, recall, f1 = _precision_recall_f1(
        true_positives, true_count, predicted_count
    )
    return AggregateEventMetrics(
        session_count=len(sessions),
        true_count=true_count,
        predicted_count=predicted_count,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        total_absolute_timing_error_seconds=total_error,
        mean_absolute_timing_error_seconds=(
            total_error / true_positives if true_positives else None
        ),
        p95_absolute_timing_error_seconds=(
            _percentile(absolute_errors, 0.95) if absolute_errors else None
        ),
        max_absolute_timing_error_seconds=(
            max(absolute_errors) if absolute_errors else None
        ),
    )


@dataclass(frozen=True, slots=True)
class BoundaryInterval:
    """The time gap between the last event of one word and the next word."""

    start_time_seconds: float
    end_time_seconds: float
    left_word_index: int | None = None
    right_word_index: int | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_time_seconds) or not math.isfinite(
            self.end_time_seconds
        ):
            raise ValueError("Boundary interval endpoints must be finite")
        if self.end_time_seconds < self.start_time_seconds:
            raise ValueError("Boundary interval end must not precede its start")

    @property
    def start(self) -> float:
        return self.start_time_seconds

    @property
    def end(self) -> float:
        return self.end_time_seconds

    def distance_to(self, timestamp_seconds: float) -> float:
        if timestamp_seconds < self.start_time_seconds:
            return self.start_time_seconds - timestamp_seconds
        if timestamp_seconds > self.end_time_seconds:
            return timestamp_seconds - self.end_time_seconds
        return 0.0


def predicted_boundary_intervals(
    word_groups: Iterable[Iterable[float] | TruthWord],
) -> tuple[BoundaryInterval, ...]:
    """Build predicted boundary gaps from ordered, non-empty word groups."""

    groups: list[tuple[float, ...]] = []
    for index, group in enumerate(word_groups):
        values: Iterable[float]
        if isinstance(group, TruthWord):
            values = group.event_times
        else:
            values = group
        times = _coerce_ordered_times(values, name=f"word_groups[{index}]")
        if not times:
            raise ValueError(f"word_groups[{index}] must not be empty")
        groups.append(times)

    intervals: list[BoundaryInterval] = []
    for index in range(len(groups) - 1):
        left_end = groups[index][-1]
        right_start = groups[index + 1][0]
        if right_start < left_end:
            raise ValueError("word_groups must be globally ordered and non-overlapping")
        intervals.append(
            BoundaryInterval(
                start_time_seconds=left_end,
                end_time_seconds=right_start,
                left_word_index=index,
                right_word_index=index + 1,
            )
        )
    return tuple(intervals)


@dataclass(frozen=True, slots=True)
class BoundaryMatch:
    true_index: int
    predicted_index: int
    true_time_seconds: float
    predicted_interval: BoundaryInterval
    distance_seconds: float


@dataclass(frozen=True, slots=True)
class BoundaryMetrics:
    matched_pairs: tuple[BoundaryMatch, ...]
    true_count: int
    predicted_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    total_distance_seconds: float
    mean_distance_seconds: float
    max_distance_seconds: float

    @property
    def matches(self) -> int:
        return self.true_positives

    @property
    def tp(self) -> int:
        return self.true_positives

    @property
    def fp(self) -> int:
        return self.false_positives

    @property
    def fn(self) -> int:
        return self.false_negatives


@dataclass(frozen=True, slots=True)
class AggregateBoundaryMetrics:
    session_count: int
    true_count: int
    predicted_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    total_distance_seconds: float
    mean_distance_seconds: float
    max_distance_seconds: float

    @property
    def matches(self) -> int:
        return self.true_positives

    @property
    def tp(self) -> int:
        return self.true_positives

    @property
    def fp(self) -> int:
        return self.false_positives

    @property
    def fn(self) -> int:
        return self.false_negatives


def _coerce_boundary_interval(
    value: BoundaryInterval | Sequence[float], *, index: int
) -> BoundaryInterval:
    if isinstance(value, BoundaryInterval):
        return value
    try:
        start, end = value
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"predicted_intervals[{index}] must be a BoundaryInterval or "
            "a two-item (start, end) sequence"
        ) from exc
    try:
        return BoundaryInterval(float(start), float(end))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"predicted_intervals[{index}] has invalid endpoints"
        ) from exc


def match_boundaries(
    true_boundary_times: Iterable[float] | GroundTruth,
    predicted_intervals: Iterable[BoundaryInterval | Sequence[float]],
    slack_seconds: float = 0.08,
) -> BoundaryMetrics:
    """Match true separator timestamps to predicted inter-word intervals.

    A timestamp inside an interval has zero error. ``slack_seconds`` expands
    each interval on both sides; matches in this slack region are ranked by
    their distance to the original interval.
    """

    if isinstance(true_boundary_times, GroundTruth):
        true_values: Iterable[float] = true_boundary_times.boundary_times
    else:
        true_values = true_boundary_times
    truth = _coerce_ordered_times(true_values, name="true_boundary_times")
    slack = _nonnegative_finite(slack_seconds, name="slack_seconds")

    intervals = tuple(
        _coerce_boundary_interval(value, index=index)
        for index, value in enumerate(predicted_intervals)
    )
    previous_start: float | None = None
    for interval in intervals:
        if (
            previous_start is not None
            and interval.start_time_seconds < previous_start
        ):
            raise ValueError(
                "predicted_intervals must be ordered from earliest to latest"
            )
        previous_start = interval.start_time_seconds

    def pair_error(true_index: int, predicted_index: int) -> float | None:
        distance = intervals[predicted_index].distance_to(truth[true_index])
        return distance if _within_inclusive_limit(distance, slack) else None

    indices = _ordered_min_error_pairs(len(truth), len(intervals), pair_error)
    matched_pairs = tuple(
        BoundaryMatch(
            true_index=true_index,
            predicted_index=predicted_index,
            true_time_seconds=truth[true_index],
            predicted_interval=intervals[predicted_index],
            distance_seconds=intervals[predicted_index].distance_to(
                truth[true_index]
            ),
        )
        for true_index, predicted_index in indices
    )
    distances = [pair.distance_seconds for pair in matched_pairs]
    true_positives = len(matched_pairs)
    false_positives = len(intervals) - true_positives
    false_negatives = len(truth) - true_positives
    precision, recall, f1 = _precision_recall_f1(
        true_positives, len(truth), len(intervals)
    )
    total_distance = sum(distances)
    return BoundaryMetrics(
        matched_pairs=matched_pairs,
        true_count=len(truth),
        predicted_count=len(intervals),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        total_distance_seconds=total_distance,
        mean_distance_seconds=(
            total_distance / true_positives if true_positives else 0.0
        ),
        max_distance_seconds=max(distances, default=0.0),
    )


def aggregate_boundary_metrics(
    metrics: Iterable[BoundaryMetrics],
) -> AggregateBoundaryMetrics:
    """Micro-aggregate boundary interval detection metrics."""

    sessions = tuple(metrics)
    true_count = sum(metric.true_count for metric in sessions)
    predicted_count = sum(metric.predicted_count for metric in sessions)
    true_positives = sum(metric.true_positives for metric in sessions)
    false_positives = sum(metric.false_positives for metric in sessions)
    false_negatives = sum(metric.false_negatives for metric in sessions)
    distances = [
        pair.distance_seconds
        for metric in sessions
        for pair in metric.matched_pairs
    ]
    total_distance = sum(distances)
    precision, recall, f1 = _precision_recall_f1(
        true_positives, true_count, predicted_count
    )
    return AggregateBoundaryMetrics(
        session_count=len(sessions),
        true_count=true_count,
        predicted_count=predicted_count,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        total_distance_seconds=total_distance,
        mean_distance_seconds=(
            total_distance / true_positives if true_positives else 0.0
        ),
        max_distance_seconds=max(distances, default=0.0),
    )


@dataclass(frozen=True, slots=True)
class AggregateSequenceMetrics:
    """Micro count-sequence totals across all supplied sessions.

    ``total_count_error`` sums each session's absolute total-keystroke error so
    an over-count in one recording cannot cancel an under-count in another.
    ``signed_total_count_delta`` separately preserves the pooled direction.
    """

    session_count: int
    exact_sequence_count: int
    true_word_count: int
    predicted_word_count: int
    matches: int
    substitutions: int
    deletions: int
    insertions: int
    aligned_items: int
    aligned_accuracy: float
    gap_penalized_mae: float
    full_sequence_exact_rate: float
    total_count_error: int
    signed_total_count_delta: int
    gap_absolute_error: int

    @property
    def exact_aligned_accuracy(self) -> float:
        return self.aligned_accuracy

    @property
    def full_sequence_exact_accuracy(self) -> float:
        return self.full_sequence_exact_rate

    @property
    def total_absolute_count_error(self) -> int:
        return self.total_count_error

    @property
    def absolute_total_keystroke_error(self) -> int:
        return self.total_count_error

    @property
    def aligned_absolute_count_error(self) -> int:
        return self.gap_absolute_error

    @property
    def word_count_delta(self) -> int:
        return self.predicted_word_count - self.true_word_count


def aggregate_sequence_metrics(
    metrics: Iterable[SequenceMetrics],
) -> AggregateSequenceMetrics:
    """Micro-aggregate sequence metrics without dropping empty sessions."""

    sessions = tuple(metrics)
    session_count = len(sessions)
    exact_sequence_count = sum(metric.full_sequence_exact for metric in sessions)
    matches = sum(metric.matches for metric in sessions)
    substitutions = sum(metric.substitutions for metric in sessions)
    deletions = sum(metric.deletions for metric in sessions)
    insertions = sum(metric.insertions for metric in sessions)
    aligned_items = matches + substitutions + deletions + insertions
    gap_absolute_error = sum(metric.gap_absolute_error for metric in sessions)

    return AggregateSequenceMetrics(
        session_count=session_count,
        exact_sequence_count=exact_sequence_count,
        true_word_count=sum(metric.true_word_count for metric in sessions),
        predicted_word_count=sum(
            metric.predicted_word_count for metric in sessions
        ),
        matches=matches,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        aligned_items=aligned_items,
        aligned_accuracy=(
            matches / aligned_items
            if aligned_items
            else (1.0 if session_count else 0.0)
        ),
        gap_penalized_mae=(
            gap_absolute_error / aligned_items if aligned_items else 0.0
        ),
        full_sequence_exact_rate=(
            exact_sequence_count / session_count if session_count else 0.0
        ),
        total_count_error=sum(metric.total_count_error for metric in sessions),
        signed_total_count_delta=sum(
            metric.signed_total_count_delta for metric in sessions
        ),
        gap_absolute_error=gap_absolute_error,
    )


# Clear aliases for callers migrating from older one-off evaluation scripts.
parse_keylog_csv = parse_keylog
align_sequences = align_count_sequences
score_sequence = score_count_sequence
aggregate_metrics = aggregate_sequence_metrics


__all__ = [
    "AggregateBoundaryMetrics",
    "AggregateEventMetrics",
    "AggregateSequenceMetrics",
    "AlignmentKind",
    "AlignmentOperation",
    "BoundaryInterval",
    "BoundaryMatch",
    "BoundaryMetrics",
    "EventMatch",
    "EventMetrics",
    "GroundTruth",
    "KeylogColumns",
    "KeylogSchemaError",
    "SequenceAlignment",
    "SequenceMetrics",
    "TruthEvent",
    "TruthWord",
    "aggregate_boundary_metrics",
    "aggregate_event_metrics",
    "aggregate_metrics",
    "aggregate_sequence_metrics",
    "align_count_sequences",
    "align_sequences",
    "match_boundaries",
    "match_events",
    "parse_keylog",
    "parse_keylog_csv",
    "predicted_boundary_intervals",
    "score_count_sequence",
    "score_sequence",
]
