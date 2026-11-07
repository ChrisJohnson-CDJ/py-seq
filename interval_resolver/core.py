"""Stateless edit planning for circular tape intervals."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from tape import IntervalInsert, TapeEdit, TapeSnapshot, VoiceInterval


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


class RecordMode(Enum):
    OVERDUB = auto()
    OVERWRITE = auto()


@dataclass(frozen=True)
class InsertCandidate:
    start_tick: int
    duration: int
    pitch: int
    velocity: int
    gesture_id: int

    def __post_init__(self) -> None:
        _integer(self.start_tick, "start_tick", minimum=0)
        _integer(self.duration, "duration", minimum=1)
        _integer(self.pitch, "pitch", minimum=0)
        _integer(self.velocity, "velocity", minimum=1)
        _integer(self.gesture_id, "gesture_id")
        if self.pitch > 127:
            raise ValueError("pitch must be between 0 and 127")
        if self.velocity > 127:
            raise ValueError("velocity must be between 1 and 127")


@dataclass(frozen=True)
class EraseRegion:
    start_tick: int
    duration: int

    def __post_init__(self) -> None:
        _integer(self.start_tick, "start_tick", minimum=0)
        _integer(self.duration, "duration", minimum=1)


class IntervalResolver:
    """Produce a declarative, circular interval edit without mutating Tape."""

    def resolve(
        self,
        snapshot: TapeSnapshot,
        operation: InsertCandidate | EraseRegion,
        mode: RecordMode,
        fragment_threshold_ticks: int,
    ) -> TapeEdit:
        if not isinstance(snapshot, TapeSnapshot):
            raise TypeError("snapshot must be a TapeSnapshot")
        if not isinstance(operation, (InsertCandidate, EraseRegion)):
            raise TypeError("operation must be an InsertCandidate or EraseRegion")
        if not isinstance(mode, RecordMode):
            raise TypeError("mode must be a RecordMode")
        threshold = _integer(
            fragment_threshold_ticks, "fragment_threshold_ticks", minimum=0
        )

        length = snapshot.length_ticks
        if length == 0:
            raise ValueError("cannot resolve an edit against a zero-length tape")
        start = operation.start_tick % length
        duration = min(operation.duration, length)
        coverage = _coverage_offsets(start, duration, length)
        removals: set[int] = set()
        insertions: list[IntervalInsert] = []

        for interval in snapshot.intervals:
            covered = _intersections(interval, coverage, length)
            if not covered:
                continue
            survivors = [
                (left, right)
                for left, right in _subtract_coverage(interval.duration, covered)
                if right - left > threshold
            ]
            removals.add(interval.id)
            if len(survivors) == 1:
                insertions.append(
                    _fragment(interval, survivors[0], length, preserve_id=interval.id)
                )
            elif len(survivors) == 2:
                insertions.append(
                    _fragment(interval, survivors[0], length, preserve_id=interval.id)
                )
                insertions.append(_fragment(interval, survivors[1], length))
            elif survivors:
                raise RuntimeError("circular subtraction produced too many fragments")

        if isinstance(operation, InsertCandidate):
            insertions.append(
                IntervalInsert(
                    start=start,
                    duration=duration,
                    pitch=operation.pitch,
                    velocity=operation.velocity,
                    gesture_id=operation.gesture_id,
                )
            )

        return TapeEdit(frozenset(removals), tuple(insertions))


def _coverage_offsets(
    start: int, duration: int, length: int
) -> tuple[tuple[int, int], ...]:
    if duration == length:
        return ((0, length),)
    end = start + duration
    if end <= length:
        return ((start, end),)
    return ((start, length), (0, end - length))


def _intersections(
    interval: VoiceInterval,
    coverage: tuple[tuple[int, int], ...],
    length: int,
) -> tuple[tuple[int, int], ...]:
    if coverage == ((0, length),):
        return ((0, interval.duration),)

    start = interval.start
    end = start + interval.duration
    intersections: list[tuple[int, int]] = []
    for offset in range(-2, 3):
        for covered_start, covered_end in coverage:
            shifted_start = covered_start + offset * length
            shifted_end = covered_end + offset * length
            left = max(start, shifted_start)
            right = min(end, shifted_end)
            if left < right:
                intersections.append((left - start, right - start))
    return _merge_ranges(intersections)


def _merge_ranges(ranges: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple((start, end) for start, end in merged)


def _subtract_coverage(
    duration: int, covered: tuple[tuple[int, int], ...]
) -> tuple[tuple[int, int], ...]:
    survivors: list[tuple[int, int]] = []
    cursor = 0
    for start, end in covered:
        if cursor < start:
            survivors.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        survivors.append((cursor, duration))
    return tuple(survivors)


def _fragment(
    interval: VoiceInterval,
    offsets: tuple[int, int],
    length: int,
    *,
    preserve_id: int | None = None,
) -> IntervalInsert:
    start_offset, end_offset = offsets
    return IntervalInsert(
        start=(interval.start + start_offset) % length,
        duration=end_offset - start_offset,
        pitch=interval.pitch,
        velocity=interval.velocity,
        gesture_id=interval.gesture_id,
        preserve_id=preserve_id,
    )
