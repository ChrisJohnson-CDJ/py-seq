"""Transport-tick recording state machine."""

from __future__ import annotations

from dataclasses import dataclass

from interval_resolver import (
    EraseRegion,
    InsertCandidate,
    IntervalResolver,
    RecordMode,
)
from pre_processor import VoiceEvent, VoiceEventType
from tape import Tape, TapeEdit


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _pitch(value: object, name: str, *, minimum: int, maximum: int) -> int:
    result = _integer(value, name, minimum=minimum)
    if result > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


@dataclass(frozen=True)
class SelectedVoice:
    pitch: int
    velocity: int
    gesture_id: int

    def __post_init__(self) -> None:
        _pitch(self.pitch, "pitch", minimum=0, maximum=127)
        _pitch(self.velocity, "velocity", minimum=1, maximum=127)
        _integer(self.gesture_id, "gesture_id")


@dataclass(frozen=True)
class PendingSegment:
    start_tick: int
    voice: SelectedVoice
    mode: RecordMode

    def __post_init__(self) -> None:
        _integer(self.start_tick, "start_tick", minimum=0)
        if not isinstance(self.voice, SelectedVoice):
            raise TypeError("voice must be a SelectedVoice")
        if not isinstance(self.mode, RecordMode):
            raise TypeError("mode must be a RecordMode")


class Recorder:
    """Coordinate voice selection, transport, recording controls, and edits."""

    def __init__(
        self,
        tape: Tape,
        resolver: IntervalResolver,
        *,
        fragment_threshold_ticks: int,
        mode: RecordMode = RecordMode.OVERDUB,
    ) -> None:
        if not isinstance(tape, Tape):
            raise TypeError("tape must be a Tape")
        if not isinstance(mode, RecordMode):
            raise TypeError("mode must be a RecordMode")
        self._fragment_threshold_ticks = _integer(
            fragment_threshold_ticks, "fragment_threshold_ticks", minimum=0
        )
        self.tape = tape
        self.resolver = resolver
        self._mode = mode
        self._record_armed = False
        self._transport_running = False
        self._selected_voice: SelectedVoice | None = None
        self._pending_segment: PendingSegment | None = None
        self._current_tick = 0
        self._coverage_cursor: int | None = None
        self._phase_origin_tick = 0

    @property
    def mode(self) -> RecordMode:
        return self._mode

    @property
    def record_armed(self) -> bool:
        return self._record_armed

    @property
    def transport_running(self) -> bool:
        return self._transport_running

    @property
    def overwriting_silence(self) -> bool:
        return (
            self.record_armed
            and self.transport_running
            and self.mode is RecordMode.OVERWRITE
            and self.selected_voice is None
        )

    @property
    def selected_voice(self) -> SelectedVoice | None:
        return self._selected_voice

    @property
    def pending_segment(self) -> PendingSegment | None:
        return self._pending_segment

    @property
    def current_tick(self) -> int:
        return self._current_tick

    @property
    def phase_origin_tick(self) -> int:
        return self._phase_origin_tick

    def set_phase_origin_tick(self, tick: int) -> None:
        tick = _integer(tick, "phase_origin_tick", minimum=0)
        if tick > self._current_tick:
            raise ValueError("phase origin cannot be in the future")
        if (
            self._pending_segment is not None
            and (
                self._pending_segment.start_tick != self._current_tick
                or tick > self._pending_segment.start_tick
            )
        ):
            raise RuntimeError("phase origin cannot change during a recording segment")
        self._phase_origin_tick = tick
        self._coverage_cursor = None

    def accept_voice(self, event: VoiceEvent, tick: int) -> None:
        self._validate_voice_event(event)
        self._validate_event_order(event)
        self._advance_to(tick)
        next_voice = (
            self._selected_voice_from_event(event)
            if event.kind is not VoiceEventType.VOICE_END
            else None
        )

        if event.kind is VoiceEventType.VOICE_START:
            self._selected_voice = next_voice
        elif event.kind is VoiceEventType.VOICE_CHANGE:
            self._close_pending(tick)
            self._selected_voice = next_voice
        else:
            self._close_pending(tick)
            self._selected_voice = None

        self._open_pending(tick)
        self._refresh_coverage_cursor(tick)

    def _selected_voice_from_event(self, event: VoiceEvent) -> SelectedVoice:
        if event.pitch is None or event.velocity is None:
            raise ValueError("voice event must include pitch and velocity")
        return SelectedVoice(event.pitch, event.velocity, event.gesture_id)

    def advance(self, tick: int) -> None:
        self._advance_to(tick)

    def set_record_armed(self, armed: bool, tick: int) -> None:
        if not isinstance(armed, bool):
            raise TypeError("armed must be a bool")
        self._advance_to(tick)
        if armed == self.record_armed:
            return
        if not armed:
            self._close_pending(tick)
        self._record_armed = armed
        self._open_pending(tick)
        self._refresh_coverage_cursor(tick)

    def set_mode(self, mode: RecordMode, tick: int) -> None:
        if not isinstance(mode, RecordMode):
            raise TypeError("mode must be a RecordMode")
        self._advance_to(tick)
        if mode is self.mode:
            return
        self._close_pending(tick)
        self._mode = mode
        self._open_pending(tick)
        self._refresh_coverage_cursor(tick)

    def set_transport_running(self, running: bool, tick: int) -> None:
        if not isinstance(running, bool):
            raise TypeError("running must be a bool")
        self._advance_to(tick)
        if running == self.transport_running:
            return
        if not running:
            self._close_pending(tick)
        self._transport_running = running
        self._open_pending(tick)
        self._refresh_coverage_cursor(tick)

    def seek(self, new_tick: int, *, old_tick: int) -> None:
        new_tick = _integer(new_tick, "new_tick", minimum=0)
        old_tick = _integer(old_tick, "old_tick", minimum=0)
        self._advance_to(old_tick)
        if new_tick == old_tick:
            return
        self._close_pending(old_tick)
        self._current_tick = new_tick
        self._open_pending(new_tick)
        self._refresh_coverage_cursor(new_tick)

    def reset_input(self, tick: int) -> None:
        self._advance_to(tick)
        self._close_pending(tick)
        self._selected_voice = None
        self._refresh_coverage_cursor(tick)

    def clear(self, tick: int) -> None:
        tick = _integer(tick, "tick", minimum=0)
        if tick < self._current_tick:
            raise ValueError("tick must be nondecreasing; use seek() for discontinuities")
        self.tape.clear()
        self._pending_segment = None
        self._current_tick = tick
        self._coverage_cursor = None
        self._open_pending(tick)
        self._refresh_coverage_cursor(tick)

    def _advance_to(self, tick: int) -> None:
        tick = _integer(tick, "tick", minimum=0)
        if tick < self._current_tick:
            raise ValueError("tick must be nondecreasing; use seek() for discontinuities")

        if self.overwriting_silence:
            if self._coverage_cursor is None:
                self._coverage_cursor = self._current_tick
            elapsed = tick - self._coverage_cursor
            if elapsed > 0:
                edit = self.resolver.resolve(
                    self.tape.snapshot(),
                    EraseRegion(
                        self._coverage_cursor - self._phase_origin_tick,
                        min(elapsed, self.tape.length_ticks),
                    ),
                    RecordMode.OVERWRITE,
                    self._fragment_threshold_ticks,
                )
                self._commit(edit)
                self._coverage_cursor = tick
        self._current_tick = tick

    def _open_pending(self, tick: int) -> None:
        if (
            self.pending_segment is None
            and self.record_armed
            and self.transport_running
            and self.selected_voice is not None
        ):
            self._pending_segment = PendingSegment(tick, self.selected_voice, self.mode)

    def _close_pending(self, tick: int) -> None:
        pending = self.pending_segment
        if pending is None:
            return
        duration = min(tick - pending.start_tick, self.tape.length_ticks)
        if duration > 0:
            edit = self.resolver.resolve(
                self.tape.snapshot(),
                InsertCandidate(
                    pending.start_tick - self._phase_origin_tick,
                    duration,
                    pending.voice.pitch,
                    pending.voice.velocity,
                    pending.voice.gesture_id,
                ),
                pending.mode,
                self._fragment_threshold_ticks,
            )
            self._commit(edit)
        self._pending_segment = None
        if pending.mode is RecordMode.OVERWRITE:
            self._coverage_cursor = tick

    def _refresh_coverage_cursor(self, tick: int) -> None:
        if self.overwriting_silence:
            self._coverage_cursor = tick
        else:
            self._coverage_cursor = None

    def _commit(self, edit: TapeEdit) -> None:
        if edit.remove_ids or edit.insert:
            self.tape.replace(edit)

    def _validate_voice_event(self, event: VoiceEvent) -> None:
        if not isinstance(event, VoiceEvent):
            raise TypeError("event must be a VoiceEvent")
        if not isinstance(event.kind, VoiceEventType):
            raise TypeError("event.kind must be a VoiceEventType")
        _integer(event.gesture_id, "gesture_id")

        if event.kind is VoiceEventType.VOICE_START:
            if event.previous_pitch is not None:
                raise ValueError("VOICE_START previous_pitch must be None")
            _pitch(event.pitch, "pitch", minimum=0, maximum=127)
            _pitch(event.velocity, "velocity", minimum=1, maximum=127)
        elif event.kind is VoiceEventType.VOICE_CHANGE:
            _pitch(event.previous_pitch, "previous_pitch", minimum=0, maximum=127)
            _pitch(event.pitch, "pitch", minimum=0, maximum=127)
            _pitch(event.velocity, "velocity", minimum=1, maximum=127)
            if event.pitch == event.previous_pitch:
                raise ValueError("VOICE_CHANGE pitch must differ from previous_pitch")
        else:
            _pitch(event.previous_pitch, "previous_pitch", minimum=0, maximum=127)
            if event.pitch is not None or event.velocity is not None:
                raise ValueError("VOICE_END pitch and velocity must be None")

    def _validate_event_order(self, event: VoiceEvent) -> None:
        if event.kind is VoiceEventType.VOICE_START:
            if self.selected_voice is not None:
                raise ValueError("VOICE_START requires no currently selected voice")
        else:
            selected = self.selected_voice
            if selected is None:
                raise ValueError(f"{event.kind.name} requires a selected voice")
            if event.gesture_id != selected.gesture_id:
                raise ValueError("event gesture_id does not match the selected voice")
            if event.previous_pitch != selected.pitch:
                raise ValueError("event previous_pitch does not match the selected voice")
