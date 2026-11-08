"""Passive storage for intervals on a circular monophonic timeline."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


class TapeError(Exception):
    """Base class for tape operation errors."""


class InvalidIntervalError(TapeError):
    """An interval violates a model or tape constraint."""


class TapeOverlapError(TapeError):
    """An interval overlaps another interval already on the tape."""


class DuplicateIntervalError(TapeError):
    """An interval ID is already present or duplicated in an edit."""


class IntervalNotFoundError(TapeError):
    """A requested interval ID is not present on the tape."""


def _require_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


@dataclass(frozen=True)
class VoiceInterval:
    """Immutable voice activity stored as a single circular half-open range."""

    id: int
    start: int
    duration: int
    pitch: int
    velocity: int
    gesture_id: int

    def __post_init__(self) -> None:
        for name in ("id", "start", "duration", "pitch", "velocity", "gesture_id"):
            _require_integer(getattr(self, name), name)
        if self.start < 0:
            raise InvalidIntervalError("start must be nonnegative")
        if self.duration <= 0:
            raise InvalidIntervalError("duration must be positive")
        if not 0 <= self.pitch <= 127:
            raise InvalidIntervalError("pitch must be between 0 and 127")
        if not 1 <= self.velocity <= 127:
            raise InvalidIntervalError("velocity must be between 1 and 127")


@dataclass(frozen=True)
class IntervalInsert:
    """Interval data to insert; Tape assigns its storage ID."""

    start: int
    duration: int
    pitch: int
    velocity: int
    gesture_id: int
    preserve_id: int | None = None

    def __post_init__(self) -> None:
        for name in ("start", "duration", "pitch", "velocity", "gesture_id"):
            _require_integer(getattr(self, name), name)
        if self.preserve_id is not None:
            _require_integer(self.preserve_id, "preserve_id")
        if self.start < 0:
            raise InvalidIntervalError("start must be nonnegative")
        if self.duration <= 0:
            raise InvalidIntervalError("duration must be positive")
        if not 0 <= self.pitch <= 127:
            raise InvalidIntervalError("pitch must be between 0 and 127")
        if not 1 <= self.velocity <= 127:
            raise InvalidIntervalError("velocity must be between 1 and 127")


@dataclass(frozen=True)
class TapeEdit:
    """One immutable, atomic edit against a Tape."""

    remove_ids: frozenset[int]
    insert: tuple[IntervalInsert, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.remove_ids, frozenset):
            raise TypeError("remove_ids must be a frozenset of interval IDs")
        if not isinstance(self.insert, tuple):
            raise TypeError("insert must be a tuple of IntervalInsert objects")
        for interval_id in self.remove_ids:
            _require_integer(interval_id, "interval ID")
        if not all(isinstance(item, IntervalInsert) for item in self.insert):
            raise TypeError("insert must contain only IntervalInsert objects")


@dataclass(frozen=True)
class TapeSnapshot:
    """Immutable view of a Tape's current intervals and length."""

    length_ticks: int
    intervals: tuple[VoiceInterval, ...]

    def __post_init__(self) -> None:
        _require_integer(self.length_ticks, "length_ticks")
        if self.length_ticks < 0:
            raise ValueError("length_ticks must be nonnegative")
        if not isinstance(self.intervals, tuple):
            raise TypeError("intervals must be a tuple")
        if not all(isinstance(item, VoiceInterval) for item in self.intervals):
            raise TypeError("intervals must contain only VoiceInterval objects")
        if self.length_ticks == 0 and self.intervals:
            raise ValueError("a zero-length tape cannot contain intervals")


class Tape:
    """Store non-overlapping intervals on a circular timeline.

    A zero-length tape is an empty loop awaiting its first recording pass.
    Query positions wrap modulo ``length_ticks``; stored intervals retain their
    original start and may cross the tape boundary without being split.
    """

    __slots__ = (
        "_length_ticks",
        "_intervals",
        "_next_id",
        "_revision",
        "_length_provisional",
    )

    def __init__(self, length_ticks: int) -> None:
        _require_integer(length_ticks, "length_ticks")
        if length_ticks < 0:
            raise ValueError("length_ticks must be nonnegative")
        self._length_ticks = length_ticks
        self._length_provisional = length_ticks == 0
        self._intervals: dict[int, VoiceInterval] = {}
        self._next_id = 1
        self._revision = 0

    @property
    def length_ticks(self) -> int:
        return self._length_ticks

    @property
    def revision(self) -> int:
        """Return the monotonically increasing content revision."""
        return self._revision

    def intervals(self) -> tuple[VoiceInterval, ...]:
        """Return stored intervals in deterministic ``(start, id)`` order."""
        return tuple(sorted(self._intervals.values(), key=lambda item: (item.start, item.id)))

    def snapshot(self) -> TapeSnapshot:
        """Return an immutable point-in-time view for edit planning."""
        return TapeSnapshot(self._length_ticks, self.intervals())

    def at(self, tick: int) -> VoiceInterval | None:
        """Return the interval occupying a circular tick, or ``None`` for silence."""
        tick = _require_integer(tick, "tick")
        if self._length_ticks == 0:
            return None
        tick %= self._length_ticks
        for interval in self._intervals.values():
            if any(start <= tick < end for start, end in _linear_ranges(
                interval.start, interval.duration, self._length_ticks
            )):
                return interval
        return None

    def overlaps(self, start: int, duration: int) -> tuple[VoiceInterval, ...]:
        """Return intervals intersecting a circular half-open query range."""
        start = _require_integer(start, "start")
        duration = _require_integer(duration, "duration")
        if self._length_ticks == 0:
            raise ValueError("a zero-length tape has no queryable range")
        start %= self._length_ticks
        if not 0 < duration <= self._length_ticks:
            raise ValueError(
                f"duration must be between 1 and {self._length_ticks} ticks"
            )
        query_ranges = _linear_ranges(start, duration, self._length_ticks)
        matches = (
            interval
            for interval in self._intervals.values()
            if _ranges_overlap(
                _linear_ranges(interval.start, interval.duration, self._length_ticks),
                query_ranges,
            )
        )
        return tuple(sorted(matches, key=lambda item: (item.start, item.id)))

    def by_gesture(self, gesture_id: int) -> tuple[VoiceInterval, ...]:
        """Return intervals with the given provenance ID in deterministic order."""
        gesture_id = _require_integer(gesture_id, "gesture_id")
        return tuple(
            sorted(
                (
                    interval
                    for interval in self._intervals.values()
                    if interval.gesture_id == gesture_id
                ),
                key=lambda item: (item.start, item.id),
            )
        )

    def insert(self, interval: IntervalInsert) -> VoiceInterval:
        """Insert one interval and return the Tape-assigned stored value."""
        self._validate_insertion(interval)
        if interval.preserve_id is not None:
            raise ValueError("insert() cannot preserve an ID; use replace()")
        self.replace(TapeEdit(frozenset(), (interval,)))
        return self._intervals[self._next_id - 1]

    def remove(self, interval_id: int) -> VoiceInterval:
        """Remove and return an interval by storage ID."""
        interval_id = _require_integer(interval_id, "interval_id")
        try:
            interval = self._intervals.pop(interval_id)
        except KeyError:
            raise IntervalNotFoundError(
                f"interval ID {interval_id} does not exist"
            ) from None
        self._revision += 1
        return interval

    def replace(self, edit: TapeEdit) -> None:
        """Atomically apply an edit and allocate IDs only after full validation."""
        if not isinstance(edit, TapeEdit):
            raise TypeError("edit must be a TapeEdit")

        validated_remove_ids = edit.remove_ids
        missing_ids = validated_remove_ids.difference(self._intervals)
        if missing_ids:
            missing = ", ".join(str(item) for item in sorted(missing_ids))
            raise IntervalNotFoundError(f"interval ID(s) do not exist: {missing}")

        candidate = {
            interval_id: interval
            for interval_id, interval in self._intervals.items()
            if interval_id not in validated_remove_ids
        }
        preserved_ids: set[int] = set()
        next_id = self._next_id
        pending: list[tuple[int, VoiceInterval]] = []
        for insertion in edit.insert:
            self._validate_insertion(insertion)
            if insertion.preserve_id is not None:
                interval_id = insertion.preserve_id
                if interval_id not in validated_remove_ids:
                    raise DuplicateIntervalError(
                        f"preserved ID {interval_id} must be included in remove_ids"
                    )
                if interval_id in preserved_ids:
                    raise DuplicateIntervalError(
                        f"preserved ID {interval_id} is used more than once"
                    )
                preserved_ids.add(interval_id)
            else:
                interval_id = next_id
                next_id += 1
            interval = VoiceInterval(
                interval_id,
                insertion.start,
                insertion.duration,
                insertion.pitch,
                insertion.velocity,
                insertion.gesture_id,
            )
            self._validate_interval(interval)
            pending.append((interval_id, interval))

        for interval_id, interval in pending:
            if interval_id in candidate:
                raise DuplicateIntervalError(
                    f"interval ID {interval_id} already exists on the tape"
                )
            candidate[interval_id] = interval
        self._validate_non_overlapping(candidate.values())
        if candidate != self._intervals:
            self._revision += 1
        self._intervals = candidate
        self._next_id = next_id

    def clear(self) -> None:
        """Atomically empty the tape without reusing IDs during its lifetime."""
        if self._intervals:
            self._intervals = {}
            self._revision += 1

    def extend_length_ticks(self, length_ticks: int) -> None:
        """Grow an unfinalized loop without publishing a content revision."""
        _require_integer(length_ticks, "length_ticks")
        if not self._length_provisional:
            raise RuntimeError("only an unfinalized loop can grow")
        if length_ticks < self._length_ticks:
            raise ValueError("a provisional tape can only grow")
        if any(
            interval.start + interval.duration > length_ticks
            for interval in self._intervals.values()
        ):
            raise InvalidIntervalError(
                "provisional length cannot truncate a stored interval"
            )
        self._length_ticks = length_ticks

    def finalize_length(self) -> None:
        """Publish the current provisional loop length as a tape revision."""
        if not self._length_provisional:
            raise RuntimeError("tape length is already finalized")
        if self._length_ticks > 0:
            self._revision += 1
            self._length_provisional = False

    def truncate(self) -> None:
        """Reset an empty tape's loop length to zero."""
        if self._intervals:
            raise ValueError("cannot truncate a tape that contains intervals")
        if self._length_ticks:
            self._length_ticks = 0
            self._revision += 1
            self._length_provisional = True

    def set_empty_length_ticks(self, length_ticks: int) -> None:
        """Set an empty tape's loop length without changing its phase origin."""
        _require_integer(length_ticks, "length_ticks")
        if length_ticks < 0:
            raise ValueError("length_ticks must be nonnegative")
        if self._intervals:
            raise ValueError("cannot change the length of a tape containing intervals")
        if self._length_ticks == length_ticks:
            return
        self._length_ticks = length_ticks
        self._length_provisional = length_ticks == 0
        self._revision += 1

    def _validate_insertion(self, insertion: IntervalInsert) -> None:
        if not isinstance(insertion, IntervalInsert):
            raise TypeError("insertion must be an IntervalInsert")
        if self._length_ticks == 0:
            raise InvalidIntervalError("cannot insert into a zero-length tape")
        if insertion.start >= self._length_ticks:
            raise InvalidIntervalError(
                f"start {insertion.start} must be in [0, {self._length_ticks})"
            )
        if insertion.duration > self._length_ticks:
            raise InvalidIntervalError(
                f"duration {insertion.duration} must be in [1, {self._length_ticks}]"
            )

    def _validate_interval(self, interval: VoiceInterval) -> None:
        if not isinstance(interval, VoiceInterval):
            raise TypeError("interval must be a VoiceInterval")
        for name in ("id", "start", "duration", "pitch", "velocity", "gesture_id"):
            _require_integer(getattr(interval, name), name)
        if interval.start < 0 or interval.start >= self._length_ticks:
            raise InvalidIntervalError(
                f"interval ID {interval.id} start {interval.start} must be in "
                f"[0, {self._length_ticks})"
            )
        if interval.duration <= 0 or interval.duration > self._length_ticks:
            raise InvalidIntervalError(
                f"interval ID {interval.id} duration {interval.duration} must be "
                f"in [1, {self._length_ticks}]"
            )
        if not 0 <= interval.pitch <= 127:
            raise InvalidIntervalError(
                f"interval ID {interval.id} pitch {interval.pitch} must be in [0, 127]"
            )
        if not 1 <= interval.velocity <= 127:
            raise InvalidIntervalError(
                f"interval ID {interval.id} velocity {interval.velocity} must be in [1, 127]"
            )

    def _validate_non_overlapping(self, intervals: Iterable[VoiceInterval]) -> None:
        ordered = sorted(intervals, key=lambda item: (item.start, item.id))
        for index, first in enumerate(ordered):
            first_ranges = _linear_ranges(
                first.start, first.duration, self._length_ticks
            )
            for second in ordered[index + 1 :]:
                second_ranges = _linear_ranges(
                    second.start, second.duration, self._length_ticks
                )
                if _ranges_overlap(first_ranges, second_ranges):
                    raise TapeOverlapError(
                        f"interval ID {first.id} range {first_ranges} overlaps "
                        f"interval ID {second.id} range {second_ranges}"
                    )


def _linear_ranges(
    start: int, duration: int, length_ticks: int
) -> tuple[tuple[int, int], ...]:
    end = start + duration
    if end <= length_ticks:
        return ((start, end),)
    return ((start, length_ticks), (0, end - length_ticks))


def _ranges_overlap(
    first: tuple[tuple[int, int], ...],
    second: tuple[tuple[int, int], ...],
) -> bool:
    return any(
        first_start < second_end and second_start < first_end
        for first_start, first_end in first
        for second_start, second_end in second
    )
