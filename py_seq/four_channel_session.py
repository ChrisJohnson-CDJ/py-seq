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
    NEXT_GESTURE = auto()
    NEXT_REFERENCE_LOOP_START = auto()
    MIDI_TRANSPORT = auto()


class RecordingStopMode(Enum):
    IMMEDIATE = auto()
    NEXT_BEAT = auto()
    AFTER_DURATION = auto()
    NEXT_GESTURE = auto()
    NEXT_REFERENCE_LOOP_START = auto()
    MIDI_TRANSPORT = auto()


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
        self._recording_start_tick: int | None = None
        self._start_mode = RecordingStartMode.IMMEDIATE
        self._stop_mode = RecordingStopMode.IMMEDIATE
        self._start_beat_count: int | None = None
        self._stop_after_ticks: int | None = None
        self._start_gesture_count: int | None = None
        self._stop_gesture_count: int | None = None
        self._stop_gesture_cycle = False
        self._start_gestures_remaining: int | None = None
        self._stop_gestures_remaining: int | None = None
        self._cycle_waiting_for_start = False
        self._reference_channel: int | None = None
        self._pending_boundary_tick: int | None = None
        self._pending_reference_signature: tuple[int, int] | None = None
        self._position_origin_tick = 0
        self._reset_mode = ResetMode.OFF
        self._reset_interval_ticks: int | None = None
        self._next_reset_tick: int | None = None
        self._reset_reference_signature: tuple[int, int, int] | None = None
        self._last_automatic_reset_tick: int | None = None
        self._last_processed_beat_boundary_tick: int | None = None

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
    def start_beat_count(self) -> int | None:
        return self._start_beat_count

    @property
    def start_gesture_count(self) -> int | None:
        return self._start_gesture_count

    @property
    def stop_mode(self) -> RecordingStopMode:
        return self._stop_mode

    @property
    def stop_after_ticks(self) -> int | None:
        return self._stop_after_ticks

    @property
    def stop_gesture_count(self) -> int | None:
        return self._stop_gesture_count

    @property
    def stop_gesture_cycle(self) -> bool:
        return self._stop_gesture_cycle

    @property
    def reference_channel(self) -> int | None:
        return self._reference_channel

    @property
    def reset_mode(self) -> ResetMode:
        return self._reset_mode

    @property
    def reset_interval_beats(self) -> float | None:
        return (
            None
            if self._reset_interval_ticks is None
            else self._reset_interval_ticks / TICKS_PER_BEAT
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
        if self.transport_running:
            self._mark_beat_boundaries_processed(tick)

    def reset(self, tick: int) -> None:
        """Immediately reset musical position and all loop phases at ``tick``."""
        tick = _validate_tick(tick)
        if tick < self.current_tick:
            raise ValueError("tick must be nondecreasing")
        self._process_boundaries(tick, inclusive=False)
        if self._last_automatic_reset_tick == tick:
            return
        coincides_with_reset = self._next_reset_tick == tick
        self._execute_reset(tick)
        if coincides_with_reset:
            self._last_automatic_reset_tick = tick
            self._schedule_next_reset(tick)

    def configure_reset(
        self,
        mode: ResetMode,
        *,
        interval_beats: int | None = None,
        interval_ticks: int | None = None,
    ) -> None:
        if not isinstance(mode, ResetMode):
            raise TypeError("mode must be a ResetMode")
        if mode is ResetMode.BEAT:
            if interval_beats is not None and interval_ticks is not None:
                raise ValueError("specify interval_beats or interval_ticks, not both")
            if interval_ticks is None:
                if (
                    isinstance(interval_beats, bool)
                    or not isinstance(interval_beats, int)
                    or interval_beats < 1
                ):
                    raise ValueError("beat reset interval must be positive")
                interval_ticks = interval_beats * TICKS_PER_BEAT
            else:
                interval_ticks = _validate_tick(interval_ticks)
                if interval_ticks == 0:
                    raise ValueError("beat reset interval must be positive")
        elif interval_beats is not None or interval_ticks is not None:
            raise ValueError("only beat reset mode accepts an interval")
        if mode is ResetMode.LOOP:
            self._require_defined_reference()

        self._reset_mode = mode
        self._reset_interval_ticks = interval_ticks if mode is ResetMode.BEAT else None
        self._schedule_next_reset(self.current_tick)

    def _process_boundaries(self, tick: int, *, inclusive: bool) -> None:
        limit = tick if inclusive else tick - 1
        while limit >= self.current_tick:
            self._validate_pending_reference()
            self._validate_reset_reference()
            recording_tick = (
                self._pending_boundary_tick if self.transport_running else None
            )
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
            elif (
                self._recording_state is RecordingState.RECORDING
                and self._stop_mode is RecordingStopMode.AFTER_DURATION
            ):
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
        if self.transport_running:
            self._mark_beat_boundaries_processed(tick)

    def _schedule_next_reset(self, tick: int) -> None:
        self._next_reset_tick = None
        self._reset_reference_signature = None
        if self._reset_mode is ResetMode.BEAT:
            assert self._reset_interval_ticks is not None
            elapsed = tick - self._position_origin_tick
            self._next_reset_tick = self._position_origin_tick + (
                elapsed // self._reset_interval_ticks + 1
            ) * self._reset_interval_ticks
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
                    and self._start_mode is not RecordingStartMode.NEXT_GESTURE
                ):
                    self._schedule_start_boundary(self.current_tick)
            elif (
                self._record_armed
                and self._recording_state is RecordingState.IDLE
            ):
                self._begin_recording_request(self.current_tick)
            self._process_boundaries(self.current_tick, inclusive=True)
            self._mark_beat_boundaries_processed(self.current_tick)

    def toggle_recording(self) -> None:
        if self._recording_state is RecordingState.IDLE:
            self.set_recording_armed(True)
        elif self._recording_state is RecordingState.WAITING_START:
            self.set_recording_armed(False)
        elif self._recording_state is RecordingState.RECORDING:
            self.request_recording_stop()
        else:
            if self._cycle_waiting_for_start:
                self.set_recording_armed(False)
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
                    self._begin_recording_window(self.current_tick)
                else:
                    if (
                        self._start_mode
                        is RecordingStartMode.NEXT_REFERENCE_LOOP_START
                    ):
                        self._require_defined_reference()
                    self._recording_state = RecordingState.WAITING_START
                    self._initialize_start_gesture_count()
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
        start_beat_count: int | None = None,
        start_gesture_count: int | None = None,
        stop_gesture_count: int | None = None,
        stop_gesture_cycle: bool | None = None,
        stop_after_ticks: int | None = None,
    ) -> None:
        if start_mode is not None and not isinstance(start_mode, RecordingStartMode):
            raise TypeError("start_mode must be a RecordingStartMode")
        if stop_mode is not None and not isinstance(stop_mode, RecordingStopMode):
            raise TypeError("stop_mode must be a RecordingStopMode")
        if reference_channel is not None:
            reference_channel = self._channel_index(reference_channel)
        self._validate_sync_count(start_beat_count, "start_beat_count")
        self._validate_sync_count(start_gesture_count, "start_gesture_count")
        self._validate_sync_count(stop_gesture_count, "stop_gesture_count")
        if stop_gesture_cycle is not None and not isinstance(
            stop_gesture_cycle, bool
        ):
            raise TypeError("stop_gesture_cycle must be a bool")
        self._validate_stop_duration(stop_after_ticks)
        if self._recording_state in (
            RecordingState.WAITING_START,
            RecordingState.WAITING_STOP,
        ):
            raise RuntimeError("cancel the pending recording boundary before reconfiguring")
        next_start_mode = self._start_mode if start_mode is None else start_mode
        next_stop_mode = self._stop_mode if stop_mode is None else stop_mode
        if stop_gesture_cycle and next_stop_mode is not RecordingStopMode.NEXT_GESTURE:
            raise ValueError(
                "cycle stop synchronization requires gesture synchronization"
            )
        next_stop_gesture_cycle = (
            (
                bool(stop_gesture_cycle)
                if stop_gesture_cycle is not None
                else False
            )
            if stop_mode is not None
            else (
                self._stop_gesture_cycle
                if stop_gesture_cycle is None
                else stop_gesture_cycle
            )
        )
        next_reference_channel = (
            self._reference_channel
            if reference_channel is None
            else reference_channel
        )
        next_start_beat_count = self._next_beat_count(
            self._start_mode,
            self._start_beat_count,
            start_mode,
            start_beat_count,
            next_start_mode,
        )
        next_stop_after_ticks = self._next_stop_duration(
            self._stop_mode,
            self._stop_after_ticks,
            stop_mode,
            stop_after_ticks,
            next_stop_mode,
        )
        next_start_gesture_count = self._next_gesture_count(
            self._start_mode,
            self._start_gesture_count,
            start_mode,
            start_gesture_count,
            next_start_mode,
        )
        next_stop_gesture_count = self._next_gesture_count(
            self._stop_mode,
            self._stop_gesture_count,
            stop_mode,
            stop_gesture_count,
            next_stop_mode,
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
        self._start_beat_count = next_start_beat_count
        self._stop_after_ticks = next_stop_after_ticks
        self._start_gesture_count = next_start_gesture_count
        self._stop_gesture_count = next_stop_gesture_count
        self._stop_gesture_cycle = next_stop_gesture_cycle
        self._reference_channel = next_reference_channel
        if self._recording_state is RecordingState.RECORDING:
            if self._stop_mode is not RecordingStopMode.NEXT_GESTURE:
                self._stop_gestures_remaining = None
            elif stop_mode is not None or stop_gesture_count is not None:
                self._stop_gestures_remaining = self._stop_gesture_count
            if self._stop_mode is RecordingStopMode.AFTER_DURATION:
                if self._recording_start_tick is None:
                    raise RuntimeError("active recording has no start tick")
                if self._stop_after_ticks is None:
                    raise RuntimeError("duration-synchronized stop has no duration")
                stop_tick = self._recording_start_tick + self._stop_after_ticks
                if stop_tick <= self.current_tick:
                    self._finish_recording_window(self.current_tick)
                else:
                    self._pending_boundary_tick = stop_tick
                    self._pending_reference_signature = None
            else:
                self._pending_boundary_tick = None
                self._pending_reference_signature = None
        if self._reset_mode is ResetMode.LOOP:
            self._schedule_next_reset(self.current_tick)

    def request_recording_stop(self) -> None:
        if self._recording_state is RecordingState.WAITING_START:
            self.set_recording_armed(False)
            return
        if self._recording_state is not RecordingState.RECORDING:
            raise RuntimeError("recording is not active")
        if self._stop_mode in (
            RecordingStopMode.IMMEDIATE,
            RecordingStopMode.AFTER_DURATION,
            RecordingStopMode.NEXT_GESTURE,
            RecordingStopMode.MIDI_TRANSPORT,
        ):
            self._finish_recording_window(self.current_tick)
            return
        self._recording_state = RecordingState.WAITING_STOP
        try:
            if self._stop_mode is RecordingStopMode.NEXT_REFERENCE_LOOP_START:
                self._schedule_stop_boundary(self.current_tick)
            else:
                self._pending_boundary_tick = self._next_global_boundary(
                    self.current_tick,
                    self._stop_mode,
                    None,
                )
        except Exception:
            self._recording_state = RecordingState.RECORDING
            self._pending_boundary_tick = None
            self._pending_reference_signature = None
            self._stop_gestures_remaining = None
            raise

    def cancel_pending_stop(self) -> None:
        if self._recording_state is not RecordingState.WAITING_STOP:
            raise RuntimeError("there is no pending recording stop")
        self._recording_state = RecordingState.RECORDING
        self._pending_boundary_tick = None
        self._pending_reference_signature = None
        self._stop_gestures_remaining = None

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
                self._begin_recording_window(self.current_tick)
            else:
                self._recording_state = RecordingState.WAITING_START
                self._initialize_start_gesture_count()

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
            (
                (
                    self._recording_state is RecordingState.WAITING_START
                    and self._start_mode
                    is not RecordingStartMode.MIDI_TRANSPORT
                )
                or (
                    self._recording_state is RecordingState.WAITING_STOP
                    and not self._cycle_waiting_for_start
                )
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

    def set_loop_length(self, channel: int, length_ticks: int) -> None:
        channel = self._channel_index(channel)
        length_ticks = _validate_tick(length_ticks)
        if length_ticks == 0:
            raise ValueError("loop length must be greater than zero")
        if self.transport_running:
            raise RuntimeError(
                "loop length can only be changed while transport is stopped"
            )

        tape = self.tapes[channel]
        if tape.intervals():
            raise ValueError("loop length can only be changed on an empty channel")
        recorder = self.recorders[channel]
        if (
            self._recording_state is not RecordingState.IDLE
            and self._recording_channel == channel
        ) or (
            recorder.record_armed
            or recorder.pending_segment is not None
            or self._definition_channel == channel
        ):
            raise RuntimeError(
                "loop length cannot be changed during an active or pending recording"
            )

        phase_origin = self.phase_origin_tick(channel)
        tape.set_empty_length_ticks(length_ticks)
        self.coordinator.set_phase_origin(channel, self.current_tick, phase_origin)

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
        message_type = getattr(message, "type", None)
        if (
            message_type == "start"
            and self._recording_state is RecordingState.WAITING_START
            and self._start_mode is RecordingStartMode.MIDI_TRANSPORT
        ):
            self._begin_recording_window(tick)
        elif (
            message_type == "stop"
            and self._recording_state is RecordingState.RECORDING
            and self._stop_mode is RecordingStopMode.MIDI_TRANSPORT
        ):
            self._finish_recording_window(tick)
        normalized_message = (
            message.copy(channel=0)
            if message_type in ("note_on", "note_off")
            else message
        )
        voice_event = self.preprocessor.process(
            adapt_mido_message(normalized_message, timestamp_ns=0)
        )
        if voice_event is not None:
            if (
                voice_event.kind is VoiceEventType.VOICE_START
                and self._cycle_waiting_for_start
            ):
                self._finish_recording_window(tick)
            if (
                voice_event.kind is VoiceEventType.VOICE_START
                and self.transport_running
                and self._recording_state is RecordingState.WAITING_START
                and self._start_mode is RecordingStartMode.NEXT_GESTURE
            ):
                self._count_start_gesture(tick)
            self.coordinator.handle_voice_event(voice_event, tick)
            if voice_event.kind is VoiceEventType.VOICE_END:
                self._complete_gesture(tick)

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
        self._initialize_start_gesture_count()
        if self._start_mode is not RecordingStartMode.NEXT_GESTURE:
            self._schedule_start_boundary(tick)

    def _begin_recording_window(self, tick: int) -> None:
        channel = self._require_recording_channel()
        self._recording_start_tick = tick
        self._pending_boundary_tick = None
        self._pending_reference_signature = None
        self._start_gestures_remaining = None
        self._cycle_waiting_for_start = False
        if self._stop_mode is RecordingStopMode.NEXT_GESTURE:
            self._stop_gestures_remaining = self._stop_gesture_count or 1
        else:
            self._stop_gestures_remaining = None
        if self._stop_mode is RecordingStopMode.AFTER_DURATION:
            if self._stop_after_ticks is None:
                raise RuntimeError("duration-synchronized stop has no duration")
            self._pending_boundary_tick = tick + self._stop_after_ticks
        if self.transport_running:
            self._prepare_loop_definition(channel, tick)
            self.coordinator.set_record_armed(channel, True, tick)
            if self._definition_channel == channel:
                self.coordinator.suspend_playback(channel, tick)
        else:
            self.coordinator.set_record_armed(channel, True, tick)
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
        self._recording_start_tick = None
        self._pending_boundary_tick = None
        self._pending_reference_signature = None
        self._start_gestures_remaining = None
        self._stop_gestures_remaining = None
        self._cycle_waiting_for_start = False

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
        elif self._start_mode is RecordingStartMode.NEXT_BEAT:
            self._pending_boundary_tick = self._next_global_boundary(
                tick,
                self._start_mode,
                self._start_beat_count,
            )
        elif self._start_mode not in (
            RecordingStartMode.NEXT_GESTURE,
            RecordingStartMode.MIDI_TRANSPORT,
        ):
            raise ValueError("start mode does not use a scheduled boundary")

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

    def _next_global_boundary(
        self,
        tick: int,
        mode: RecordingStartMode | RecordingStopMode,
        beat_count: int | None,
    ) -> int:
        if mode is RecordingStartMode.NEXT_BEAT or mode is RecordingStopMode.NEXT_BEAT:
            width = TICKS_PER_BEAT
        else:
            raise ValueError("mode does not use a global transport boundary")
        elapsed = tick - self._position_origin_tick
        remainder = elapsed % width
        boundary = (
            tick
            if remainder == 0
            and self._last_processed_beat_boundary_tick != tick
            else self._position_origin_tick + (elapsed // width + 1) * width
        )
        return boundary + (beat_count or 0) * width

    def _initialize_start_gesture_count(self) -> None:
        self._start_gestures_remaining = (
            (self._start_gesture_count or 1)
            if self._start_mode is RecordingStartMode.NEXT_GESTURE
            else None
        )

    def _count_start_gesture(self, tick: int) -> None:
        if self._start_gestures_remaining is None:
            raise RuntimeError("gesture-synchronized start has no remaining count")
        self._start_gestures_remaining -= 1
        if self._start_gestures_remaining == 0:
            self._begin_recording_window(tick)

    def _complete_gesture(self, tick: int) -> None:
        if self._recording_state is not RecordingState.RECORDING:
            return
        if self._stop_mode is not RecordingStopMode.NEXT_GESTURE:
            return
        if self._stop_gestures_remaining is None:
            raise RuntimeError("gesture-synchronized stop has no remaining count")
        self._stop_gestures_remaining -= 1
        if self._stop_gestures_remaining == 0:
            if self._stop_gesture_cycle:
                self._recording_state = RecordingState.WAITING_STOP
                self._cycle_waiting_for_start = True
            else:
                self._finish_recording_window(tick)

    @staticmethod
    def _validate_sync_count(count: int | None, name: str) -> None:
        if count is not None and (
            isinstance(count, bool) or not isinstance(count, int) or count < 1
        ):
            raise ValueError(f"{name} must be a positive integer")

    @staticmethod
    def _validate_stop_duration(duration_ticks: int | None) -> None:
        if duration_ticks is not None and (
            isinstance(duration_ticks, bool)
            or not isinstance(duration_ticks, int)
            or duration_ticks < 1
        ):
            raise ValueError("stop_after_ticks must be a positive integer")

    @staticmethod
    def _next_beat_count(
        current_mode: RecordingStartMode | RecordingStopMode,
        current_count: int | None,
        requested_mode: RecordingStartMode | RecordingStopMode | None,
        requested_count: int | None,
        next_mode: RecordingStartMode | RecordingStopMode,
    ) -> int | None:
        if next_mode not in (
            RecordingStartMode.NEXT_BEAT,
            RecordingStopMode.NEXT_BEAT,
        ):
            if requested_count is not None:
                raise ValueError("beat count requires beat synchronization")
            return None
        if requested_mode is not None:
            return requested_count
        if requested_count is not None:
            if current_mode is not next_mode:
                raise ValueError("beat count requires beat synchronization")
            return requested_count
        return current_count

    @staticmethod
    def _next_stop_duration(
        current_mode: RecordingStopMode,
        current_duration: int | None,
        requested_mode: RecordingStopMode | None,
        requested_duration: int | None,
        next_mode: RecordingStopMode,
    ) -> int | None:
        if next_mode is not RecordingStopMode.AFTER_DURATION:
            if requested_duration is not None:
                raise ValueError("stop duration requires duration synchronization")
            return None
        if requested_mode is not None:
            duration = requested_duration
        elif requested_duration is not None:
            if current_mode is not RecordingStopMode.AFTER_DURATION:
                raise ValueError("stop duration requires duration synchronization")
            duration = requested_duration
        else:
            duration = current_duration
        if duration is None:
            raise ValueError("duration-synchronized stop requires a duration")
        return duration

    @staticmethod
    def _next_gesture_count(
        current_mode: RecordingStartMode | RecordingStopMode,
        current_count: int | None,
        requested_mode: RecordingStartMode | RecordingStopMode | None,
        requested_count: int | None,
        next_mode: RecordingStartMode | RecordingStopMode,
    ) -> int | None:
        uses_gestures = next_mode in (
            RecordingStartMode.NEXT_GESTURE,
            RecordingStopMode.NEXT_GESTURE,
        )
        if not uses_gestures:
            if requested_count is not None:
                raise ValueError("gesture count requires gesture synchronization")
            return None
        if requested_mode is not None:
            return 1 if requested_count is None else requested_count
        if requested_count is not None:
            if current_mode is not next_mode:
                raise ValueError("gesture count requires gesture synchronization")
            return requested_count
        return 1 if current_count is None else current_count

    def _mark_beat_boundaries_processed(self, tick: int) -> None:
        elapsed = tick - self._position_origin_tick
        boundary = (
            self._position_origin_tick
            + elapsed // TICKS_PER_BEAT * TICKS_PER_BEAT
        )
        if (
            self._last_processed_beat_boundary_tick is None
            or boundary > self._last_processed_beat_boundary_tick
        ):
            self._last_processed_beat_boundary_tick = boundary

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
