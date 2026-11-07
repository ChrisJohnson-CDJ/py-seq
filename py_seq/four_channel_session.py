from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

import mido

from articulator import Articulator
from interval_resolver import IntervalResolver, RecordMode
from midi_output import MidiMessageSink, MidiOutputAdapter
from playback_coordinator import PlaybackChannel, PlaybackCoordinator
from playback_scheduler import PlaybackScheduler
from pre_processor import MidiVoicePreprocessor, VoiceEventType, adapt_mido_message
from recorder import Recorder
from tape import Tape

TICKS_PER_BEAT = 960
CHANNEL_COUNT = 4


class RecordingStartMode(Enum):
    IMMEDIATE = auto()
    NEXT_BEAT = auto()
    NEXT_REFERENCE_LOOP_START = auto()
    FIRST_NOTE_ONSET = auto()


class RecordingStopMode(Enum):
    IMMEDIATE = auto()
    NEXT_BEAT = auto()
    NEXT_REFERENCE_LOOP_START = auto()


class RecordingState(Enum):
    IDLE = auto()
    WAITING_START = auto()
    RECORDING = auto()
    WAITING_STOP = auto()


class ResetMode(Enum):
    OFF = auto()
    BEAT = auto()
    LOOP = auto()


def _validate_tick(tick: object) -> int:
    if isinstance(tick, bool) or not isinstance(tick, int):
        raise TypeError("tick must be an integer")
    if tick < 0:
        raise ValueError("tick must be nonnegative")
    return tick


@dataclass(frozen=True)
class MidiRoute:
    label: str
    midi_channel: int
    sink: MidiMessageSink
    port_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or len(self.label) > 8:
            raise ValueError("route label must be at most 8 characters")
        if isinstance(self.midi_channel, bool) or not isinstance(
            self.midi_channel, int
        ):
            raise TypeError("midi_channel must be an integer")
        if not 0 <= self.midi_channel <= 15:
            raise ValueError("midi_channel must be between 0 and 15")
        if not callable(getattr(self.sink, "send", None)):
            raise TypeError("sink must provide send(message)")
        if self.port_index is not None and (
            isinstance(self.port_index, bool)
            or not isinstance(self.port_index, int)
            or self.port_index < 0
        ):
            raise ValueError("port_index must be a nonnegative integer or None")


class FourChannelSession:
    """Four tape loops with one focused input, transport, and record mode."""

    def __init__(
        self,
        lengths_ticks: tuple[int, int, int, int],
        bpm: float,
        routes: tuple[MidiRoute, MidiRoute, MidiRoute, MidiRoute],
    ) -> None:
        if len(lengths_ticks) != CHANNEL_COUNT:
            raise ValueError("lengths_ticks must contain four loop lengths")
        if any(
            isinstance(length, bool) or not isinstance(length, int) or length < 0
            for length in lengths_ticks
        ):
            raise ValueError("loop lengths must be nonnegative integers")
        if isinstance(bpm, bool) or not isinstance(bpm, (int, float)):
            raise TypeError("bpm must be a number")
        if not math.isfinite(bpm) or bpm <= 0:
            raise ValueError("bpm must be a finite number greater than zero")
        if len(routes) != CHANNEL_COUNT or not all(
            isinstance(route, MidiRoute) for route in routes
        ):
            raise ValueError("routes must contain four MidiRoute values")

        self.bpm = float(bpm)
        self.routes = routes
        self.programs = [0] * CHANNEL_COUNT
        self.bank_msbs = [0] * CHANNEL_COUNT
        self.tapes = tuple(Tape(length) for length in lengths_ticks)
        self.recorders = tuple(
            Recorder(
                tape,
                IntervalResolver(),
                fragment_threshold_ticks=0,
            )
            for tape in self.tapes
        )
        self.schedulers = tuple(PlaybackScheduler(tape) for tape in self.tapes)
        self.articulators = tuple(Articulator() for _ in range(CHANNEL_COUNT))
        channels = tuple(
            PlaybackChannel(
                self.recorders[index],
                self.schedulers[index],
                self.articulators[index],
                MidiOutputAdapter(route.sink, route.midi_channel),
            )
            for index, route in enumerate(routes)
        )
        self.coordinator = PlaybackCoordinator(channels)
        self.preprocessor = MidiVoicePreprocessor()
        self._record_armed = False
        self._mode = RecordMode.OVERDUB
        self._definition_channel: int | None = None
        self._definition_start_tick: int | None = None
        self._recording_state = RecordingState.IDLE
        self._recording_channel: int | None = None
        self._start_mode = RecordingStartMode.IMMEDIATE
        self._stop_mode = RecordingStopMode.IMMEDIATE
        self._reference_channel: int | None = None
        self._pending_boundary_tick: int | None = None
        self._pending_reference_signature: tuple[int, int] | None = None
        self._position_origin_tick = 0
        self._reset_mode = ResetMode.OFF
        self._reset_interval_ticks: int | None = None
        self._next_reset_tick: int | None = None
        self._reset_reference_signature: tuple[int, int, int] | None = None
        self._last_automatic_reset_tick: int | None = None

    @property
    def current_tick(self) -> int:
        return self.coordinator.current_tick

    @property
    def musical_tick(self) -> int:
        return self.current_tick - self._position_origin_tick

    @property
    def position_origin_tick(self) -> int:
        return self._position_origin_tick

    @property
    def transport_running(self) -> bool:
        return any(scheduler.running for scheduler in self.schedulers)

    @property
    def active_channel(self) -> int:
        return self.coordinator.focused_channel

    @property
    def record_armed(self) -> bool:
        return self._record_armed

    @property
    def mode(self) -> RecordMode:
        return self._mode

    @property
    def recording_state(self) -> RecordingState:
        return self._recording_state

    @property
    def loop_definition_channel(self) -> int | None:
        return self._definition_channel

    @property
    def start_mode(self) -> RecordingStartMode:
        return self._start_mode

    @property
    def stop_mode(self) -> RecordingStopMode:
        return self._stop_mode

    @property
    def reference_channel(self) -> int | None:
        return self._reference_channel

    @property
    def reset_mode(self) -> ResetMode:
        return self._reset_mode

    @property
    def reset_interval_beats(self) -> int | None:
        return (
            None
            if self._reset_interval_ticks is None
            else self._reset_interval_ticks // TICKS_PER_BEAT
        )

    def tape(self, channel: int) -> Tape:
        return self.tapes[self._channel_index(channel)]

    def recorder(self, channel: int) -> Recorder:
        return self.recorders[self._channel_index(channel)]

    def phase_origin_tick(self, channel: int) -> int:
        return self.schedulers[self._channel_index(channel)].phase_origin_tick

    def advance(self, tick: int) -> None:
        tick = _validate_tick(tick)
        if tick < self.current_tick:
            raise ValueError("tick must be nondecreasing")
        self._process_boundaries(tick, inclusive=True)
        self._advance_transport_to(tick)

    def reset(self, tick: int) -> None:
        """Immediately reset musical position and all loop phases at ``tick``."""
        tick = _validate_tick(tick)
        if tick < self.current_tick:
            raise ValueError("tick must be nondecreasing")
        self._process_boundaries(tick, inclusive=False)
        if self._last_automatic_reset_tick == tick:
            return
        self._execute_reset(tick)

    def configure_reset(
        self,
        mode: ResetMode,
        *,
        interval_beats: int | None = None,
    ) -> None:
        if not isinstance(mode, ResetMode):
            raise TypeError("mode must be a ResetMode")
        if mode is ResetMode.BEAT:
            if (
                isinstance(interval_beats, bool)
                or not isinstance(interval_beats, int)
                or interval_beats < 1
            ):
                raise ValueError("beat reset interval must be a positive integer")
        elif interval_beats is not None:
            raise ValueError("only beat reset mode accepts an interval")
        if mode is ResetMode.LOOP:
            self._require_defined_reference()

        self._reset_mode = mode
        self._reset_interval_ticks = (
            interval_beats * TICKS_PER_BEAT
            if mode is ResetMode.BEAT and interval_beats is not None
            else None
        )
        self._schedule_next_reset(self.current_tick)

    def _process_boundaries(self, tick: int, *, inclusive: bool) -> None:
        limit = tick if inclusive else tick - 1
        while limit >= self.current_tick:
            self._validate_pending_reference()
            self._validate_reset_reference()
            recording_tick = self._pending_boundary_tick
            if recording_tick is not None and recording_tick > limit:
                recording_tick = None
            reset_tick = (
                self._next_reset_tick
                if self.transport_running
                and self._next_reset_tick is not None
                and self._next_reset_tick <= limit
                else None
            )
            if recording_tick is None and reset_tick is None:
                break
            if reset_tick is not None and (
                recording_tick is None or reset_tick <= recording_tick
            ):
                self._execute_reset(reset_tick, automatic=True)
                continue

            assert recording_tick is not None
            self._advance_transport_to(recording_tick)
            self._validate_pending_reference()
            if self._pending_boundary_tick != recording_tick:
                continue
            if self._recording_state is RecordingState.WAITING_START:
                self._begin_recording_window(recording_tick)
            elif self._recording_state is RecordingState.WAITING_STOP:
                self._finish_recording_window(recording_tick)
            else:
                raise RuntimeError("recording boundary has no pending transition")

    def _execute_reset(self, tick: int, *, automatic: bool = False) -> None:
        disarm_channels: tuple[int, ...] = ()
        if self._recording_state in (
            RecordingState.RECORDING,
            RecordingState.WAITING_STOP,
        ):
            channel = self._require_recording_channel()
            disarm_channels = (channel,)
            if self._definition_channel == channel:
                start_tick = self._definition_start_tick
                assert start_tick is not None
                self.tapes[channel].extend_length_ticks(tick - start_tick)
                self._finalize_loop_definition(resume_playback=False)

        self._position_origin_tick = tick
        self._clear_recording_request()
        if automatic:
            self._last_automatic_reset_tick = tick
        try:
            self.coordinator.reset_phases(tick, disarm_channels=disarm_channels)
        finally:
            self._schedule_next_reset(tick)

    def _schedule_next_reset(self, tick: int) -> None:
        self._next_reset_tick = None
        self._reset_reference_signature = None
        if self._reset_mode is ResetMode.BEAT:
            assert self._reset_interval_ticks is not None
            self._next_reset_tick = tick + self._reset_interval_ticks
        elif self._reset_mode is ResetMode.LOOP:
            channel, tape = self._require_defined_reference()
            origin = self.phase_origin_tick(channel)
            length = tape.length_ticks
            self._next_reset_tick = origin + ((tick - origin) // length + 1) * length
            self._reset_reference_signature = (channel, length, origin)

    def _validate_reset_reference(self) -> None:
        if self._reset_mode is not ResetMode.LOOP:
            return
        try:
            channel, tape = self._require_defined_reference()
        except ValueError:
            self._reset_mode = ResetMode.OFF
            self._reset_interval_ticks = None
            self._next_reset_tick = None
            self._reset_reference_signature = None
            return

        signature = (channel, tape.length_ticks, self.phase_origin_tick(channel))
        if signature != self._reset_reference_signature:
            self._schedule_next_reset(self.current_tick)

    def toggle_transport(self) -> None:
        if self.transport_running:
            self.advance(self.current_tick)
            try:
                self.coordinator.stop(self.current_tick)
            finally:
                if self._recording_state is not RecordingState.WAITING_STOP:
                    self._finalize_loop_definition(resume_playback=False)
        else:
            if (
                self._record_armed
                and self._recording_state is RecordingState.RECORDING
            ):
                self._prepare_loop_definition(
                    self._require_recording_channel(), self.current_tick
                )
            self.coordinator.start(self.current_tick)
            if (
                self._definition_channel is not None
                and self._recording_state is RecordingState.RECORDING
            ):
                self.coordinator.suspend_playback(
                    self._definition_channel,
                    self.current_tick,
                )
            if (
                self._record_armed
                and self._recording_state is RecordingState.WAITING_START
            ):
                if self._start_mode is RecordingStartMode.IMMEDIATE:
                    self._begin_recording_window(self.current_tick)
                elif (
                    self._pending_boundary_tick is None
                    and self._start_mode is not RecordingStartMode.FIRST_NOTE_ONSET
                ):
                    self._schedule_start_boundary(self.current_tick)
            elif (
                self._record_armed
                and self._recording_state is RecordingState.IDLE
            ):
                self._begin_recording_request(self.current_tick)

    def toggle_recording(self) -> None:
        if self._recording_state is RecordingState.IDLE:
            self.set_recording_armed(True)
        elif self._recording_state is RecordingState.WAITING_START:
            self.set_recording_armed(False)
        elif self._recording_state is RecordingState.RECORDING:
            self.request_recording_stop()
        else:
            self.cancel_pending_stop()

    def set_recording_armed(self, armed: bool) -> None:
        if not isinstance(armed, bool):
            raise TypeError("armed must be a bool")
        if armed:
            if self._record_armed:
                return
            self._validate_recording_reference(
                self.active_channel,
                self._start_mode,
                self._stop_mode,
                self._reference_channel,
            )
            self._record_armed = True
            self._recording_channel = self.active_channel
            try:
                if self.transport_running:
                    self._begin_recording_request(self.current_tick)
                elif self._start_mode is RecordingStartMode.IMMEDIATE:
                    self._recording_state = RecordingState.RECORDING
                    self.coordinator.set_record_armed(
                        self._recording_channel, True, self.current_tick
                    )
                else:
                    if (
                        self._start_mode
                        is RecordingStartMode.NEXT_REFERENCE_LOOP_START
                    ):
                        self._require_defined_reference()
                    self._recording_state = RecordingState.WAITING_START
            except Exception:
                self._clear_recording_request()
                raise
            return
        if not self._record_armed:
            return
        if self._recording_state in (
            RecordingState.RECORDING,
            RecordingState.WAITING_STOP,
        ):
            channel = self._require_recording_channel()
            self.coordinator.set_record_armed(channel, False, self.current_tick)
            if self._definition_channel == channel:
                self._finalize_loop_definition(resume_playback=True)
        self._clear_recording_request()

    def configure_recording_sync(
        self,
        *,
        start_mode: RecordingStartMode | None = None,
        stop_mode: RecordingStopMode | None = None,
        reference_channel: int | None = None,
    ) -> None:
        if start_mode is not None and not isinstance(start_mode, RecordingStartMode):
            raise TypeError("start_mode must be a RecordingStartMode")
        if stop_mode is not None and not isinstance(stop_mode, RecordingStopMode):
            raise TypeError("stop_mode must be a RecordingStopMode")
        if reference_channel is not None:
            reference_channel = self._channel_index(reference_channel)
        if self._recording_state in (
            RecordingState.WAITING_START,
            RecordingState.WAITING_STOP,
        ):
            raise RuntimeError("cancel the pending recording boundary before reconfiguring")
        next_start_mode = self._start_mode if start_mode is None else start_mode
        next_stop_mode = self._stop_mode if stop_mode is None else stop_mode
        next_reference_channel = (
            self._reference_channel
            if reference_channel is None
            else reference_channel
        )
        if (
            next_start_mode is RecordingStartMode.NEXT_REFERENCE_LOOP_START
            or next_stop_mode is RecordingStopMode.NEXT_REFERENCE_LOOP_START
        ) and next_reference_channel is None:
            raise ValueError("reference-loop sync requires a selected reference channel")
        if self._reset_mode is ResetMode.LOOP:
            if next_reference_channel is None or (
                self.tapes[next_reference_channel].length_ticks <= 0
            ):
                raise ValueError(
                    "loop reset requires a reference channel with a defined loop"
                )
        if self._record_armed:
            recording_channel = (
                self.active_channel
                if self._recording_channel is None
                else self._recording_channel
            )
            self._validate_recording_reference(
                recording_channel,
                next_start_mode,
                next_stop_mode,
                next_reference_channel,
            )
        self._start_mode = next_start_mode
        self._stop_mode = next_stop_mode
        self._reference_channel = next_reference_channel
        if self._reset_mode is ResetMode.LOOP:
            self._schedule_next_reset(self.current_tick)

    def request_recording_stop(self) -> None:
        if self._recording_state is RecordingState.WAITING_START:
            self.set_recording_armed(False)
            return
        if self._recording_state is not RecordingState.RECORDING:
            raise RuntimeError("recording is not active")
        if self._stop_mode is RecordingStopMode.IMMEDIATE:
            self._finish_recording_window(self.current_tick)
            return
        self._recording_state = RecordingState.WAITING_STOP
        try:
            if self._stop_mode is RecordingStopMode.NEXT_REFERENCE_LOOP_START:
                self._schedule_stop_boundary(self.current_tick)
            else:
                self._pending_boundary_tick = self._next_global_boundary(
                    self.current_tick, self._stop_mode
                )
        except Exception:
            self._recording_state = RecordingState.RECORDING
            self._pending_boundary_tick = None
            self._pending_reference_signature = None
            raise

    def cancel_pending_stop(self) -> None:
        if self._recording_state is not RecordingState.WAITING_STOP:
            raise RuntimeError("there is no pending recording stop")
        self._recording_state = RecordingState.RECORDING
        self._pending_boundary_tick = None
        self._pending_reference_signature = None

    def select_channel(self, channel: int) -> None:
        channel = self._channel_index(channel)
        previous = self.active_channel
        if channel == previous:
            return

        was_armed = self._record_armed
        if was_armed:
            self._validate_recording_reference(
                channel,
                self._start_mode,
                self._stop_mode,
                self._reference_channel,
            )
        old_recording_channel = self._recording_channel
        if was_armed and old_recording_channel == previous:
            if self.recorders[previous].record_armed:
                self.coordinator.set_record_armed(previous, False, self.current_tick)
            if self._definition_channel == previous:
                self._finalize_loop_definition(resume_playback=True)
            self._clear_recording_request()
        if self._mode is RecordMode.OVERWRITE:
            self._set_mode(RecordMode.OVERDUB)
        self.coordinator.set_focus(channel, self.current_tick)
        if was_armed:
            self._record_armed = True
            self._recording_channel = channel
            if self.transport_running:
                self._begin_recording_request(self.current_tick)
            elif self._start_mode is RecordingStartMode.IMMEDIATE:
                self._recording_state = RecordingState.RECORDING
                self.coordinator.set_record_armed(channel, True, self.current_tick)
            else:
                self._recording_state = RecordingState.WAITING_START

    @staticmethod
    def _validate_recording_reference(
        channel: int,
        start_mode: RecordingStartMode,
        stop_mode: RecordingStopMode,
        reference_channel: int | None,
    ) -> None:
        uses_reference_boundary = (
            start_mode is RecordingStartMode.NEXT_REFERENCE_LOOP_START
            or stop_mode is RecordingStopMode.NEXT_REFERENCE_LOOP_START
        )
        if uses_reference_boundary and reference_channel == channel:
            raise ValueError(
                "recording channel cannot use itself as the reference channel"
            )

    def toggle_mode(self) -> None:
        if self._mode is RecordMode.OVERWRITE:
            self._set_mode(RecordMode.OVERDUB)
            return
        if not self._record_armed:
            self.set_recording_armed(True)
        self._set_mode(RecordMode.OVERWRITE)

    def clear_active_tape(self) -> None:
        self.clear_tape(self.active_channel)

    def clear_tape(self, channel: int) -> None:
        channel = self._channel_index(channel)
        if (
            self._recording_state in (
                RecordingState.WAITING_START,
                RecordingState.WAITING_STOP,
            )
            and self._reference_channel == channel
        ):
            self._cancel_due_to_reference_change()
        tape = self.tapes[channel]
        had_intervals = bool(tape.intervals())
        was_defining = self._definition_channel == channel
        was_armed = channel == self.active_channel and self._record_armed
        was_transport_running = self.transport_running
        if was_defining:
            self._definition_channel = None
            self._definition_start_tick = None
        self.coordinator.clear_tape(channel, self.current_tick)
        if was_defining and had_intervals:
            tape.finalize_length()
        reset_length = not had_intervals and tape.length_ticks > 0
        if reset_length:
            if self.schedulers[channel].running:
                self.coordinator.suspend_playback(channel, self.current_tick)
        if reset_length:
            tape.truncate()
            self.coordinator.set_phase_origin(
                channel, self.current_tick, self.current_tick
            )
        elif was_defining and had_intervals and was_transport_running:
            self.coordinator.resume_playback(channel, self.current_tick)
        if (
            was_armed
            and was_transport_running
            and tape.length_ticks == 0
        ):
            self._prepare_loop_definition(channel, self.current_tick)
            self.coordinator.suspend_playback(channel, self.current_tick)
        elif reset_length and was_transport_running:
            self.coordinator.resume_playback(channel, self.current_tick)
        if (
            self._reset_mode is ResetMode.LOOP
            and self._reference_channel == channel
            and tape.length_ticks <= 0
        ):
            self._reset_mode = ResetMode.OFF
            self._reset_interval_ticks = None
            self._next_reset_tick = None
            self._reset_reference_signature = None

    def change_program(self, program: int) -> None:
        if isinstance(program, bool) or not isinstance(program, int):
            raise TypeError("program must be an integer")
        if not 0 <= program <= 127:
            raise ValueError("program must be between 0 and 127")
        route = self.routes[self.active_channel]
        route.sink.send(
            mido.Message(
                "program_change",
                channel=route.midi_channel,
                program=program,
            )
        )
        self.programs[self.active_channel] = program

    def change_bank_program(self, bank_msb: int, program: int) -> None:
        for value, name in ((bank_msb, "bank_msb"), (program, "program")):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not 0 <= value <= 127:
                raise ValueError(f"{name} must be between 0 and 127")
        route = self.routes[self.active_channel]
        route.sink.send(
            mido.Message(
                "control_change",
                channel=route.midi_channel,
                control=0,
                value=bank_msb,
            )
        )
        route.sink.send(
            mido.Message(
                "program_change",
                channel=route.midi_channel,
                program=program,
            )
        )
        self.bank_msbs[self.active_channel] = bank_msb
        self.programs[self.active_channel] = program

    def process_midi_message(self, message: mido.Message, tick: int) -> None:
        self.advance(tick)
        normalized_message = (
            message.copy(channel=0)
            if getattr(message, "type", None) in ("note_on", "note_off")
            else message
        )
        voice_event = self.preprocessor.process(
            adapt_mido_message(normalized_message, timestamp_ns=0)
        )
        if voice_event is not None:
            if (
                voice_event.kind is VoiceEventType.VOICE_START
                and self._recording_state is RecordingState.WAITING_START
                and self._start_mode is RecordingStartMode.FIRST_NOTE_ONSET
                and self.transport_running
            ):
                self._begin_recording_window(tick)
            self.coordinator.handle_voice_event(voice_event, tick)

    def close(self) -> None:
        errors: list[Exception] = []
        if self.transport_running:
            try:
                self.coordinator.stop(self.current_tick)
            except Exception as error:
                errors.append(error)
            finally:
                self._finalize_loop_definition(resume_playback=False)
        try:
            self.coordinator.disconnect_input(self.current_tick)
        except Exception as error:
            errors.append(error)
        self.preprocessor.reset()
        if errors:
            raise ExceptionGroup("failed to close four-channel session", errors)

    def panic(self) -> None:
        errors: list[Exception] = []
        for channel in self.coordinator.channels:
            try:
                channel.output.panic()
                channel.articulator.forget_output_state()
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("failed to panic all MIDI outputs", errors)

    def _set_mode(self, mode: RecordMode) -> None:
        for channel in range(CHANNEL_COUNT):
            self.coordinator.set_record_mode(channel, mode, self.current_tick)
        self._mode = mode

    def _prepare_loop_definition(self, channel: int, tick: int) -> None:
        if (
            self._definition_channel is not None
            or self.tapes[channel].length_ticks != 0
        ):
            return
        if self.schedulers[channel].running:
            self.coordinator.suspend_playback(channel, tick)
        self.coordinator.set_phase_origin(channel, tick, tick)
        self._definition_channel = channel
        self._definition_start_tick = tick

    def _begin_recording_request(self, tick: int) -> None:
        if self._start_mode is RecordingStartMode.IMMEDIATE:
            self._begin_recording_window(tick)
            return
        self._recording_state = RecordingState.WAITING_START
        if self._start_mode is not RecordingStartMode.FIRST_NOTE_ONSET:
            self._schedule_start_boundary(tick)

    def _begin_recording_window(self, tick: int) -> None:
        channel = self._require_recording_channel()
        self._pending_boundary_tick = None
        self._pending_reference_signature = None
        if self.transport_running:
            self._prepare_loop_definition(channel, tick)
            self.coordinator.set_record_armed(channel, True, tick)
            if self._definition_channel == channel:
                self.coordinator.suspend_playback(channel, tick)
        self._recording_state = RecordingState.RECORDING

    def _finish_recording_window(self, tick: int) -> None:
        channel = self._require_recording_channel()
        if self.recorders[channel].record_armed:
            self.coordinator.set_record_armed(channel, False, tick)
        if self._definition_channel == channel:
            self._finalize_loop_definition(resume_playback=True)
        self._clear_recording_request()

    def _clear_recording_request(self) -> None:
        self._record_armed = False
        self._recording_state = RecordingState.IDLE
        self._recording_channel = None
        self._pending_boundary_tick = None
        self._pending_reference_signature = None

    def _require_recording_channel(self) -> int:
        if self._recording_channel is None:
            raise RuntimeError("recording request has no target channel")
        return self._recording_channel

    def _advance_transport_to(self, tick: int) -> None:
        if self._definition_channel is not None:
            start_tick = self._definition_start_tick
            assert start_tick is not None
            elapsed = tick - start_tick
            if elapsed < 0:
                raise ValueError("tick must be nondecreasing")
            self.tapes[self._definition_channel].extend_length_ticks(elapsed)
        self.coordinator.advance(tick)

    def _schedule_start_boundary(self, tick: int) -> None:
        if self._start_mode is RecordingStartMode.NEXT_REFERENCE_LOOP_START:
            self._schedule_reference_boundary(tick)
        else:
            self._pending_boundary_tick = self._next_global_boundary(
                tick, self._start_mode
            )

    def _schedule_stop_boundary(self, tick: int) -> None:
        self._schedule_reference_boundary(tick)

    def _schedule_reference_boundary(self, tick: int) -> None:
        channel = self._reference_channel
        channel, tape = self._require_defined_reference()
        if (
            self.transport_running
            and channel != self.active_channel
            and not self.schedulers[channel].running
        ):
            raise ValueError(f"reference channel {channel + 1} is not running")
        origin = self.phase_origin_tick(channel)
        length = tape.length_ticks
        self._pending_boundary_tick = origin + ((tick - origin) // length + 1) * length
        self._pending_reference_signature = (length, origin)

    def _require_defined_reference(self) -> tuple[int, Tape]:
        channel = self._reference_channel
        if channel is None:
            raise ValueError("reference-loop sync requires a selected reference channel")
        tape = self.tapes[channel]
        if tape.length_ticks <= 0:
            raise ValueError(f"reference channel {channel + 1} has no defined loop")
        return channel, tape

    @staticmethod
    def _next_global_boundary(
        tick: int,
        mode: RecordingStartMode | RecordingStopMode,
    ) -> int:
        if mode is RecordingStartMode.NEXT_BEAT or mode is RecordingStopMode.NEXT_BEAT:
            width = TICKS_PER_BEAT
        else:
            raise ValueError("mode does not use a global transport boundary")
        return (tick // width + 1) * width

    def _validate_pending_reference(self) -> None:
        if self._pending_reference_signature is None:
            return
        channel = self._reference_channel
        if channel is None:
            self._cancel_due_to_reference_change()
            return
        tape = self.tapes[channel]
        current_signature = (tape.length_ticks, self.phase_origin_tick(channel))
        channel_stopped = (
            self.transport_running
            and channel != self.active_channel
            and not self.schedulers[channel].running
        )
        if current_signature != self._pending_reference_signature or channel_stopped:
            self._cancel_due_to_reference_change()

    def _cancel_due_to_reference_change(self) -> None:
        if self._recording_state is RecordingState.WAITING_START:
            self._clear_recording_request()
        elif self._recording_state is RecordingState.WAITING_STOP:
            self._recording_state = RecordingState.RECORDING
            self._pending_boundary_tick = None
            self._pending_reference_signature = None

    def _finalize_loop_definition(self, *, resume_playback: bool) -> None:
        channel = self._definition_channel
        if channel is None:
            return
        self.tapes[channel].finalize_length()
        self._definition_channel = None
        self._definition_start_tick = None
        if resume_playback and self.transport_running:
            self.coordinator.resume_playback(channel, self.current_tick)

    @staticmethod
    def _channel_index(channel: int) -> int:
        if isinstance(channel, bool) or not isinstance(channel, int):
            raise TypeError("channel must be an integer")
        if not 0 <= channel < CHANNEL_COUNT:
            raise ValueError("channel must be between 0 and 3")
        return channel


__all__ = [
    "CHANNEL_COUNT",
    "FourChannelSession",
    "MidiRoute",
    "RecordingStartMode",
    "RecordingState",
    "RecordingStopMode",
    "ResetMode",
    "TICKS_PER_BEAT",
]
