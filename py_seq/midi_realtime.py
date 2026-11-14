from __future__ import annotations

import math
from collections.abc import Callable
from enum import Enum

import mido

from midi_output import MidiMessageSink

_NANOSECONDS_PER_MINUTE = 60_000_000_000
_MIDI_CLOCKS_PER_BEAT = 24
_TICKS_PER_MIDI_CLOCK = 40


class MidiOutputMode(Enum):
    OFF = "off"
    ON = "on"
    AUTO = "auto"


class _PulseGrid:
    def __init__(self, now_ns: int, bpm: float, phase: float) -> None:
        self._origin_ns = now_ns
        self._origin_phase = phase
        self._bpm = bpm
        self._next_index = math.floor(phase) + 1

    def phase_at(self, now_ns: int) -> float:
        elapsed_ns = max(0, now_ns - self._origin_ns)
        return (
            self._origin_phase
            + elapsed_ns * self._bpm * _MIDI_CLOCKS_PER_BEAT
            / _NANOSECONDS_PER_MINUTE
        )

    def next_index(self, now_ns: int) -> int:
        return max(self._next_index, math.floor(self.phase_at(now_ns)) + 1)

    def deadline_ns(self, index: int) -> int:
        elapsed_pulses = index - self._origin_phase
        return self._origin_ns + round(
            elapsed_pulses
            * _NANOSECONDS_PER_MINUTE
            / (_MIDI_CLOCKS_PER_BEAT * self._bpm)
        )

    def reanchor(self, now_ns: int, phase: float) -> None:
        self._origin_ns = now_ns
        self._origin_phase = phase
        self._next_index = math.floor(phase) + 1

    def set_tempo(self, now_ns: int, bpm: float) -> None:
        phase = self.phase_at(now_ns)
        self._origin_ns = now_ns
        self._origin_phase = phase
        self._bpm = bpm

    def consume_due(self, through_ns: int) -> int | None:
        latest_index = math.floor(self.phase_at(through_ns))
        while self.deadline_ns(latest_index) > through_ns:
            latest_index -= 1
        while self.deadline_ns(latest_index + 1) <= through_ns:
            latest_index += 1
        if latest_index < self._next_index:
            return None
        self._next_index = latest_index + 1
        return latest_index


class MidiRealtimeOutput:
    """Serialize outgoing transport and clock messages on the host worker."""

    def __init__(
        self,
        sinks: tuple[MidiMessageSink, ...],
        *,
        now_ns: int,
        bpm: float,
        current_tick: int,
    ) -> None:
        self._sinks = sinks
        self._grid = _PulseGrid(now_ns, bpm, current_tick / _TICKS_PER_MIDI_CLOCK)
        self._transport_mode = MidiOutputMode.OFF
        self._clock_mode = MidiOutputMode.OFF
        self._transport_running = False
        self._transport_active = False
        self._clock_active = False
        self._pending_transport_index: int | None = None

    @property
    def transport_mode(self) -> MidiOutputMode:
        return self._transport_mode

    @property
    def clock_mode(self) -> MidiOutputMode:
        return self._clock_mode

    @property
    def transport_active(self) -> bool:
        return self._transport_active

    @property
    def transport_start_pending(self) -> bool:
        return self._pending_transport_index is not None

    @property
    def clock_active(self) -> bool:
        return self._clock_active

    def set_tempo(self, now_ns: int, bpm: float) -> None:
        self._grid.set_tempo(now_ns, bpm)

    def set_transport_mode(
        self,
        mode: MidiOutputMode,
        *,
        now_ns: int,
        transport_running: bool,
    ) -> None:
        if not isinstance(mode, MidiOutputMode):
            raise TypeError("mode must be a MidiOutputMode")
        self._transport_mode = mode
        self._pending_transport_index = None
        self._transport_running = transport_running
        if mode is MidiOutputMode.OFF:
            self._send_transport(False)
        elif mode is MidiOutputMode.ON:
            self._send_transport(True)
        elif transport_running:
            self._pending_transport_index = self._grid.next_index(now_ns)
        else:
            self._send_transport(False)

    def set_clock_mode(
        self,
        mode: MidiOutputMode,
        *,
        now_ns: int,
        transport_running: bool,
        current_tick: int,
    ) -> None:
        if not isinstance(mode, MidiOutputMode):
            raise TypeError("mode must be a MidiOutputMode")
        self._clock_mode = mode
        self._transport_running = transport_running
        if mode is MidiOutputMode.AUTO and transport_running:
            self._grid.reanchor(now_ns, current_tick / _TICKS_PER_MIDI_CLOCK)
            if self._pending_transport_index is not None:
                self._pending_transport_index = self._grid.next_index(now_ns)
        self._clock_active = mode is MidiOutputMode.ON or (
            mode is MidiOutputMode.AUTO and transport_running
        )

    def application_transport_changed(
        self,
        running: bool,
        *,
        now_ns: int,
        current_tick: int,
    ) -> None:
        if not isinstance(running, bool):
            raise TypeError("running must be a bool")
        self._transport_running = running
        if self._clock_mode is not MidiOutputMode.ON:
            self._grid.reanchor(now_ns, current_tick / _TICKS_PER_MIDI_CLOCK)
            if self._pending_transport_index is not None:
                self._pending_transport_index = self._grid.next_index(now_ns)
        self._clock_active = self._clock_mode is MidiOutputMode.ON or (
            self._clock_mode is MidiOutputMode.AUTO and running
        )
        self._pending_transport_index = None
        if self._transport_mode is MidiOutputMode.AUTO:
            self._send_transport(running)

    def reset(self, *, transport_running: bool) -> None:
        if (
            transport_running
            and self._transport_mode is MidiOutputMode.AUTO
        ):
            self._pending_transport_index = None
            self._send_transport(True)

    def process_due(
        self,
        *,
        now_ns: int,
        through_ns: int,
        allow_clock: bool = True,
        advance_before: Callable[[int], None],
    ) -> bool:
        eligible_through_ns = min(now_ns, through_ns)
        due_clock_index = self._grid.consume_due(eligible_through_ns)
        pending_index = self._pending_transport_index
        due_transport_start = (
            pending_index is not None
            and self._grid.deadline_ns(pending_index) <= eligible_through_ns
            and self._transport_mode is MidiOutputMode.AUTO
            and self._transport_running
        )

        clock_due = due_clock_index is not None and self._clock_active
        clock_sent = False
        advanced_index: int | None = None

        if (
            due_transport_start
            and due_clock_index is not None
            and pending_index is not None
            and pending_index < due_clock_index
        ):
            assert pending_index is not None
            advance_before(self._grid.deadline_ns(pending_index))
            advanced_index = pending_index
            self._send_transport(True)
            self._pending_transport_index = None
            due_transport_start = False

        if clock_due and allow_clock:
            assert due_clock_index is not None
            clock_deadline = self._grid.deadline_ns(due_clock_index)
            advance_before(clock_deadline)
            advanced_index = due_clock_index
            self._send("clock")
            clock_sent = True

        if due_transport_start:
            assert pending_index is not None
            if advanced_index != pending_index:
                advance_before(self._grid.deadline_ns(pending_index))
            self._send_transport(True)
            self._pending_transport_index = None

        return clock_sent

    def _send_transport(self, running: bool) -> None:
        self._send("start" if running else "stop")
        self._transport_active = running

    def _send(self, message_type: str) -> None:
        message = mido.Message(message_type)
        for sink in self._sinks:
            try:
                sink.send(message)
            except Exception as error:
                raise RuntimeError(
                    f"failed to send MIDI {message_type} message"
                ) from error


__all__ = ["MidiOutputMode", "MidiRealtimeOutput"]
