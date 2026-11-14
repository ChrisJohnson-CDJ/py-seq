"""Multi-channel orchestration for recording, scheduling, and live input."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

from articulator import Articulator, LiveVoice, NoteAction
from interval_resolver import RecordMode
from midi_output import MidiOutputAdapter, MidiOutputError
from pre_processor import VoiceEvent, VoiceEventType
from playback_scheduler import PlaybackEvent, PlaybackScheduler
from recorder import Recorder


def _tick(value: object, name: str = "tick") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


@dataclass
class PlaybackChannel:
    recorder: Recorder
    scheduler: PlaybackScheduler
    articulator: Articulator
    output: MidiOutputAdapter

    def __post_init__(self) -> None:
        if not isinstance(self.recorder, Recorder):
            raise TypeError("recorder must be a Recorder")
        if not isinstance(self.scheduler, PlaybackScheduler):
            raise TypeError("scheduler must be a PlaybackScheduler")
        if self.scheduler.tape is not self.recorder.tape:
            raise ValueError("recorder and scheduler must share the same Tape")
        if not isinstance(self.articulator, Articulator):
            raise TypeError("articulator must be an Articulator")
        if not isinstance(self.output, MidiOutputAdapter):
            raise TypeError("output must be a MidiOutputAdapter")


class PlaybackCoordinator:
    """Coordinate independent tape voices at a shared tick."""

    def __init__(
        self,
        channels: tuple[PlaybackChannel, ...],
        *,
        focused_channel: int = 0,
    ) -> None:
        if not isinstance(channels, tuple) or not channels:
            raise ValueError("channels must be a non-empty tuple of PlaybackChannel values")
        if not all(isinstance(item, PlaybackChannel) for item in channels):
            raise TypeError("channels must contain PlaybackChannel values")
        self.channels = channels
        self._focused_channel = self._channel_index(focused_channel)
        self._current_tick = channels[0].recorder.current_tick
        if any(
            channel.recorder.current_tick != self._current_tick for channel in channels
        ):
            raise ValueError("all recorders must start at the same tick")
        self._input_gesture_id: int | None = None
        self._suppressed_recording_gesture_id: int | None = None

    @property
    def current_tick(self) -> int:
        return self._current_tick

    @property
    def focused_channel(self) -> int:
        return self._focused_channel

    def start(self, tick: int) -> None:
        tick = _tick(tick)
        if any(channel.scheduler.running for channel in self.channels):
            raise RuntimeError("coordinator is already running")
        self._advance_to(tick)
        for channel in self.channels:
            channel.recorder.set_transport_running(True, tick)
            self._apply_events(channel, channel.scheduler.start(tick))
        self._current_tick = tick

    def advance(self, tick: int) -> None:
        self._advance_to(_tick(tick))

    def stop(self, tick: int) -> None:
        tick = _tick(tick)
        self._advance_to(tick)
        errors: list[MidiOutputError] = []
        for channel in self.channels:
            actions = list(
                self._actions_for_events(channel, channel.scheduler.stop(tick))
            )
            channel.recorder.set_transport_running(False, tick)
            actions.extend(self._reconcile_actions(channel, tick))
            actions.extend(channel.articulator.silence())
            try:
                self._emit(channel, tuple(actions))
            except MidiOutputError as error:
                errors.append(error)
        self._current_tick = tick
        if errors:
            raise ExceptionGroup("failed to stop all MIDI outputs", errors)

    def seek(self, tick: int) -> None:
        tick = _tick(tick)
        old_tick = self._current_tick
        if tick == old_tick:
            return
        for channel in self.channels:
            channel.recorder.seek(tick, old_tick=old_tick)
            self._apply_events(channel, channel.scheduler.seek(tick))
        self._current_tick = tick

    def reset_phases(
        self,
        tick: int,
        *,
        disarm_channels: tuple[int, ...] = (),
        before_reset: Callable[[], None] | None = None,
    ) -> None:
        """Atomically rebase all loop phases without disturbing live input."""
        tick = _tick(tick)
        if tick < self._current_tick:
            raise ValueError("tick must be nondecreasing")
        for index in disarm_channels:
            self._channel(index)

        errors: list[MidiOutputError] = []
        for channel in self.channels:
            channel.recorder.advance(tick)
        for index in disarm_channels:
            self.channels[index].recorder.set_record_armed(False, tick)

        for channel in self.channels:
            if channel.scheduler.running:
                try:
                    self._apply_events(
                        channel,
                        channel.scheduler.advance_before(tick),
                    )
                except MidiOutputError as error:
                    errors.append(error)

        if before_reset is not None:
            before_reset()

        for channel in self.channels:
            actions = channel.articulator.reset_playback()
            try:
                self._emit(channel, actions)
            except MidiOutputError as error:
                errors.append(error)

        for channel in self.channels:
            channel.recorder.set_phase_origin_tick(tick)
            channel.scheduler.reset_phase(tick)

        for channel in self.channels:
            if channel.scheduler.running:
                try:
                    self._reconcile(channel, tick)
                except MidiOutputError as error:
                    errors.append(error)

        self._current_tick = tick
        if errors:
            raise ExceptionGroup("failed to reset all playback phases", errors)

    def set_record_armed(self, channel_index: int, armed: bool, tick: int) -> None:
        tick = _tick(tick)
        self._advance_to(tick)
        channel = self._channel(channel_index)
        channel.recorder.set_record_armed(armed, tick)
        self._reconcile(channel, tick)

    def set_record_mode(
        self, channel_index: int, mode: RecordMode, tick: int
    ) -> None:
        tick = _tick(tick)
        self._advance_to(tick)
        channel = self._channel(channel_index)
        channel.recorder.set_mode(mode, tick)
        self._reconcile(channel, tick)

    def set_phase_origin(self, channel_index: int, tick: int, origin_tick: int) -> None:
        tick = _tick(tick)
        origin_tick = _tick(origin_tick, "phase origin tick")
        self._advance_to(tick)
        channel = self._channel(channel_index)
        channel.recorder.set_phase_origin_tick(origin_tick)
        channel.scheduler.set_phase_origin_tick(origin_tick)
        self._reconcile(channel, tick)

    def suspend_playback(self, channel_index: int, tick: int) -> None:
        tick = _tick(tick)
        self._advance_to(tick)
        channel = self._channel(channel_index)
        if channel.scheduler.running:
            self._apply_events(channel, channel.scheduler.stop(tick))

    def resume_playback(self, channel_index: int, tick: int) -> None:
        tick = _tick(tick)
        self._advance_to(tick)
        channel = self._channel(channel_index)
        if (
            any(item.scheduler.running for item in self.channels)
            and not channel.scheduler.running
        ):
            self._apply_events(channel, channel.scheduler.start(tick))

    def clear_tape(self, channel_index: int, tick: int) -> None:
        tick = _tick(tick)
        self._advance_to(tick)
        channel = self._channel(channel_index)
        channel.recorder.clear(tick)
        self._reconcile(channel, tick)

    def set_focus(self, channel_index: int, tick: int) -> None:
        tick = _tick(tick)
        self._advance_to(tick)
        target = self._channel_index(channel_index)
        if target == self._focused_channel:
            return

        old_channel = self.channels[self._focused_channel]
        old_channel.recorder.reset_input(tick)
        self._reconcile(old_channel, tick)
        self._emit(old_channel, old_channel.articulator.set_live_voice(None))

        self._suppressed_recording_gesture_id = self._input_gesture_id
        self._focused_channel = target

    def handle_voice_event(self, event: VoiceEvent, tick: int) -> None:
        tick = _tick(tick)
        self._validate_voice_event(event)
        self._advance_to(tick)
        channel = self.channels[self._focused_channel]
        suppressed = event.gesture_id == self._suppressed_recording_gesture_id

        if event.kind is VoiceEventType.VOICE_START:
            channel.recorder.accept_voice(event, tick)
            self._input_gesture_id = event.gesture_id
            self._suppressed_recording_gesture_id = None
            self._reconcile(channel, tick)
            self._set_live(channel, self._live_voice_from_event(event))
            return

        if not suppressed:
            channel.recorder.accept_voice(event, tick)
            self._reconcile(channel, tick)

        if event.kind is VoiceEventType.VOICE_CHANGE:
            self._set_live(channel, self._live_voice_from_event(event))
        else:
            self._set_live(channel, None)
            self._input_gesture_id = None
            self._suppressed_recording_gesture_id = None

    def disconnect_input(self, tick: int) -> None:
        tick = _tick(tick)
        self._advance_to(tick)
        errors: list[MidiOutputError] = []
        for channel in self.channels:
            channel.recorder.reset_input(tick)
            actions = list(self._reconcile_actions(channel, tick))
            actions.extend(channel.articulator.silence())
            try:
                self._emit(channel, tuple(actions))
            except MidiOutputError as error:
                errors.append(error)
        self._input_gesture_id = None
        self._suppressed_recording_gesture_id = None
        if errors:
            raise ExceptionGroup("failed to silence all MIDI outputs", errors)

    def _advance_to(self, tick: int) -> None:
        if tick < self._current_tick:
            raise ValueError("tick must be nondecreasing; use seek() for discontinuities")
        if tick == self._current_tick:
            return

        for channel in self.channels:
            channel.recorder.advance(tick)
        overwriting_silence = tuple(
            channel.recorder.overwriting_silence for channel in self.channels
        )
        for channel in self.channels:
            if channel.scheduler.running:
                self._reconcile(channel, self._current_tick)
        for index, channel in enumerate(self.channels):
            if channel.scheduler.running:
                events = channel.scheduler.advance(tick)
                if not overwriting_silence[index]:
                    self._apply_events(channel, events)
        for channel in self.channels:
            self._reconcile(channel, tick)
        self._current_tick = tick

    def _reconcile(self, channel: PlaybackChannel, tick: int) -> None:
        self._apply_events(channel, channel.scheduler.reconcile(tick))

    def _apply_events(
        self, channel: PlaybackChannel, events: tuple[PlaybackEvent, ...]
    ) -> None:
        self._emit(channel, self._actions_for_events(channel, events))

    @staticmethod
    def _actions_for_events(
        channel: PlaybackChannel, events: tuple[PlaybackEvent, ...]
    ) -> tuple[NoteAction, ...]:
        actions: list[NoteAction] = []
        for event in events:
            actions.extend(channel.articulator.handle(event))
        return tuple(actions)

    def _reconcile_actions(
        self, channel: PlaybackChannel, tick: int
    ) -> tuple[NoteAction, ...]:
        return self._actions_for_events(channel, channel.scheduler.reconcile(tick))

    def _set_live(self, channel: PlaybackChannel, voice: LiveVoice | None) -> None:
        self._emit(channel, channel.articulator.set_live_voice(voice))

    @staticmethod
    def _emit(channel: PlaybackChannel, actions: tuple[NoteAction, ...]) -> None:
        try:
            channel.output.send_actions(actions)
        except MidiOutputError:
            channel.articulator.forget_output_state()
            raise

    def _channel(self, index: int) -> PlaybackChannel:
        return self.channels[self._channel_index(index)]

    def _channel_index(self, index: object) -> int:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("channel index must be an integer")
        if not 0 <= index < len(self.channels):
            raise ValueError(
                f"channel index must be between 0 and {len(self.channels) - 1}"
            )
        return index

    @staticmethod
    def _validate_voice_event(event: VoiceEvent) -> None:
        if not isinstance(event, VoiceEvent):
            raise TypeError("event must be a VoiceEvent")
        if not isinstance(event.kind, VoiceEventType):
            raise TypeError("event.kind must be a VoiceEventType")
        if isinstance(event.gesture_id, bool) or not isinstance(event.gesture_id, int):
            raise TypeError("event.gesture_id must be an integer")
        if event.kind in (VoiceEventType.VOICE_START, VoiceEventType.VOICE_CHANGE):
            PlaybackCoordinator._live_voice_from_event(event)
            if event.kind is VoiceEventType.VOICE_START and event.previous_pitch is not None:
                raise ValueError("VOICE_START previous_pitch must be None")
            if event.kind is VoiceEventType.VOICE_CHANGE:
                previous_pitch = event.previous_pitch
                if previous_pitch is None:
                    raise ValueError("VOICE_CHANGE previous_pitch is required")
                _tick(previous_pitch, "previous_pitch")
                if previous_pitch > 127:
                    raise ValueError("previous_pitch must be between 0 and 127")
        else:
            previous_pitch = event.previous_pitch
            if previous_pitch is None:
                raise ValueError("VOICE_END previous_pitch is required")
            _tick(previous_pitch, "previous_pitch")
            if previous_pitch > 127:
                raise ValueError("previous_pitch must be between 0 and 127")
            if event.pitch is not None or event.velocity is not None:
                raise ValueError("VOICE_END pitch and velocity must be None")

    @staticmethod
    def _live_voice_from_event(event: VoiceEvent) -> LiveVoice:
        if event.pitch is None or event.velocity is None:
            raise ValueError(f"{event.kind.name} pitch and velocity are required")
        return LiveVoice(event.pitch, event.velocity)
