"""Pure, tick-driven playback boundary scheduling for one Tape."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from tape import Tape, TapeSnapshot, VoiceInterval


class PlaybackEventKind(Enum):
    ONSET = auto()
    END = auto()
    ENTER_ACTIVE = auto()
    EXIT_ACTIVE = auto()


@dataclass(frozen=True)
class PlaybackEvent:
    kind: PlaybackEventKind
    tick: int
    interval: VoiceInterval

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PlaybackEventKind):
            raise TypeError("kind must be a PlaybackEventKind")
        _require_tick(self.tick, "tick")
        if not isinstance(self.interval, VoiceInterval):
            raise TypeError("interval must be a VoiceInterval")


class PlaybackEventLimitError(RuntimeError):
    """A single scheduler advance would emit more events than its limit."""


def _require_tick(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _at(snapshot: TapeSnapshot, tick: int) -> VoiceInterval | None:
    if snapshot.length_ticks == 0:
        return None
    position = tick % snapshot.length_ticks
    for interval in snapshot.intervals:
        end = interval.start + interval.duration
        if end <= snapshot.length_ticks:
            if interval.start <= position < end:
                return interval
        elif position >= interval.start or position < end - snapshot.length_ticks:
            return interval
    return None


def _same_voice(first: VoiceInterval, second: VoiceInterval) -> bool:
    return (
        first.start,
        first.duration,
        first.pitch,
        first.velocity,
    ) == (
        second.start,
        second.duration,
        second.pitch,
        second.velocity,
    )


class PlaybackScheduler:
    """Report tape occupancy and onset boundaries without emitting MIDI.

    An advance processes every boundary in ``(previous_tick, tick]``. A
    maximum of ``MAX_EVENTS_PER_ADVANCE`` events is permitted; larger jumps
    raise ``PlaybackEventLimitError`` without advancing scheduler state.
    """

    MAX_EVENTS_PER_ADVANCE = 100_000

    def __init__(self, tape: Tape, *, phase_origin_tick: int = 0) -> None:
        if not isinstance(tape, Tape):
            raise TypeError("tape must be a Tape")
        self._phase_origin_tick = _require_tick(
            phase_origin_tick, "phase_origin_tick"
        )
        self.tape = tape
        self._snapshot = tape.snapshot()
        self._tick: int | None = None
        self._active: VoiceInterval | None = None
        self._running = False

    @property
    def current_tick(self) -> int | None:
        return self._tick

    @property
    def active_interval(self) -> VoiceInterval | None:
        return self._active

    @property
    def running(self) -> bool:
        return self._running

    @property
    def phase_origin_tick(self) -> int:
        return self._phase_origin_tick

    def set_phase_origin_tick(self, tick: int) -> None:
        tick = _require_tick(tick, "phase_origin_tick")
        if self._running:
            raise RuntimeError("phase origin can only change while stopped")
        self._phase_origin_tick = tick
        self._snapshot = self.tape.snapshot()

    def start(self, tick: int) -> tuple[PlaybackEvent, ...]:
        tick = _require_tick(tick, "tick")
        if self._running:
            raise RuntimeError("scheduler is already running")
        self._snapshot = self.tape.snapshot()
        self._tick = tick
        self._running = True
        self._active = _at(self._snapshot, tick - self._phase_origin_tick)
        if self._active is None:
            return ()
        return (self._occupancy_event(tick, self._active),)

    def advance(self, tick: int) -> tuple[PlaybackEvent, ...]:
        tick = _require_tick(tick, "tick")
        if not self._running or self._tick is None:
            raise RuntimeError("scheduler is not running")
        if tick < self._tick:
            raise ValueError("tick must be nondecreasing; use seek() to move backward")
        if tick == self._tick:
            return ()

        boundaries = self._boundaries_between(self._tick, tick)
        if len(boundaries) > self.MAX_EVENTS_PER_ADVANCE:
            raise PlaybackEventLimitError(
                f"advance would emit {len(boundaries)} events; limit is "
                f"{self.MAX_EVENTS_PER_ADVANCE}"
            )
        self._tick = tick
        self._active = _at(self._snapshot, tick - self._phase_origin_tick)
        return tuple(boundary[2] for boundary in boundaries)

    def advance_before(self, tick: int) -> tuple[PlaybackEvent, ...]:
        """Advance to a reset tick while excluding events exactly at that tick."""
        tick = _require_tick(tick, "tick")
        if not self._running or self._tick is None:
            raise RuntimeError("scheduler is not running")
        if tick < self._tick:
            raise ValueError("tick must be nondecreasing; use seek() to move backward")
        if tick == self._tick:
            return ()

        boundaries = self._boundaries_between(self._tick, tick - 1)
        if len(boundaries) > self.MAX_EVENTS_PER_ADVANCE:
            raise PlaybackEventLimitError(
                f"advance would emit {len(boundaries)} events; limit is "
                f"{self.MAX_EVENTS_PER_ADVANCE}"
            )
        self._tick = tick
        self._active = _at(self._snapshot, tick - 1 - self._phase_origin_tick)
        return tuple(boundary[2] for boundary in boundaries)

    def stop(self, tick: int) -> tuple[PlaybackEvent, ...]:
        tick = _require_tick(tick, "tick")
        events = list(self.advance(tick)) if self._running else []
        if self._active is not None:
            events.append(
                PlaybackEvent(PlaybackEventKind.EXIT_ACTIVE, tick, self._active)
            )
        self._active = None
        self._tick = None
        self._running = False
        return tuple(events)

    def seek(self, tick: int) -> tuple[PlaybackEvent, ...]:
        tick = _require_tick(tick, "tick")
        events: list[PlaybackEvent] = []
        if self._active is not None:
            events.append(
                PlaybackEvent(
                    PlaybackEventKind.EXIT_ACTIVE,
                    self._tick if self._tick is not None else tick,
                    self._active,
                )
            )
        self._snapshot = self.tape.snapshot()
        self._tick = tick
        self._active = _at(self._snapshot, tick - self._phase_origin_tick)
        if self._active is not None:
            events.append(self._occupancy_event(tick, self._active))
        return tuple(events)

    def reset_phase(self, tick: int) -> None:
        """Rebase a running or stopped scheduler to phase zero at ``tick``."""
        tick = _require_tick(tick, "tick")
        self._snapshot = self.tape.snapshot()
        self._phase_origin_tick = tick
        self._tick = tick if self._running else None
        self._active = None

    def reconcile(self, tick: int) -> tuple[PlaybackEvent, ...]:
        tick = _require_tick(tick, "tick")
        if self._tick is None:
            self._snapshot = self.tape.snapshot()
            return ()
        if tick != self._tick:
            raise ValueError("reconcile tick must equal the current scheduler tick")

        latest = self.tape.snapshot()
        if not self._running:
            self._snapshot = latest
            self._active = None
            return ()

        current = _at(latest, tick - self._phase_origin_tick)
        previous = self._active
        self._snapshot = latest
        if previous is not None and current is not None and _same_voice(previous, current):
            self._active = current
            return ()

        events: list[PlaybackEvent] = []
        if previous is not None:
            events.append(PlaybackEvent(PlaybackEventKind.EXIT_ACTIVE, tick, previous))
        if current is not None:
            events.append(self._occupancy_event(tick, current))
        self._active = current
        return tuple(events)

    def _occupancy_event(
        self, tick: int, interval: VoiceInterval
    ) -> PlaybackEvent:
        kind = (
            PlaybackEventKind.ONSET
            if (tick - self._phase_origin_tick) % self._snapshot.length_ticks
            == interval.start
            else PlaybackEventKind.ENTER_ACTIVE
        )
        return PlaybackEvent(kind, tick, interval)

    def _boundaries_between(
        self, previous_tick: int, tick: int
    ) -> list[tuple[int, int, PlaybackEvent]]:
        length = self._snapshot.length_ticks
        if length == 0:
            return []
        previous_tick -= self._phase_origin_tick
        tick -= self._phase_origin_tick
        count = 0
        ranges: list[tuple[VoiceInterval, int, int, int]] = []
        for interval in self._snapshot.intervals:
            for priority, offset in (
                (1, interval.start),
                (0, interval.start + interval.duration),
            ):
                first_cycle = (previous_tick - offset) // length + 1
                last_cycle = (tick - offset) // length
                occurrence_count = max(0, last_cycle - first_cycle + 1)
                count += occurrence_count
                ranges.append((interval, priority, offset, first_cycle))

        if count > self.MAX_EVENTS_PER_ADVANCE:
            raise PlaybackEventLimitError(
                f"advance would emit {count} events; limit is "
                f"{self.MAX_EVENTS_PER_ADVANCE}"
            )

        boundaries: list[tuple[int, int, PlaybackEvent]] = []
        for interval, priority, offset, first_cycle in ranges:
            last_cycle = (tick - offset) // length
            for cycle in range(first_cycle, last_cycle + 1):
                boundary_tick = (
                    self._phase_origin_tick + offset + cycle * length
                )
                kind = PlaybackEventKind.ONSET if priority else PlaybackEventKind.END
                boundaries.append(
                    (
                        boundary_tick,
                        priority,
                        PlaybackEvent(kind, boundary_tick, interval),
                    )
                )
        boundaries.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2].interval.start,
                item[2].interval.id,
            )
        )
        return boundaries
