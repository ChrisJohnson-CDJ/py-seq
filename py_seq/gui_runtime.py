"""Host-owned runtime and immutable handoff for the Pygame front end."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol

import mido

from interval_resolver import RecordMode
from .four_channel_session import (
    CHANNEL_COUNT,
    TICKS_PER_BEAT,
    FourChannelSession,
    ResetMode,
)
from .recording_commands import (
    parse_length_command,
    parse_recording_sync_command,
    parse_reset_command,
)
from tape import VoiceInterval

logger = logging.getLogger(__name__)

_CHANNEL_IDS = tuple(f"channel-{index + 1}" for index in range(CHANNEL_COUNT))


class GuiIntentKind(Enum):
    TRANSPORT = auto()
    SELECT_CHANNEL = auto()
    RECORDING_ARM = auto()
    RECORDING_TOGGLE = auto()
    RECORDING_SYNC = auto()
    RESET = auto()
    RESET_CONFIG = auto()
    LOOP_LENGTH = auto()
    OVERWRITE = auto()
    CLEAR_TAPE = auto()
    PROGRAM = auto()
    BANK_PROGRAM = auto()
    SHUTDOWN = auto()


@dataclass(frozen=True, slots=True)
class GuiIntent:
    request_id: int
    submitted_at_ns: int
    kind: GuiIntentKind
    channel_id: str | None = None
    value: bool | int | str | None = None
    bank_msb: int | None = None


@dataclass(frozen=True, slots=True)
class GuiTransportSnapshot:
    position_ticks: int
    tempo_bpm: float
    playing: bool
    sampled_at_ns: int
    epoch: int
    ticks_per_beat: int = TICKS_PER_BEAT


@dataclass(frozen=True, slots=True)
class GuiChannelSnapshot:
    channel_id: str
    name: str
    midi_channel: int
    bank_msb: int
    program: int
    loop_length_ticks: int
    phase_origin_tick: int
    tape_revision: int
    intervals: tuple[VoiceInterval, ...]
    live_pitch: int | None = None
    live_velocity: int | None = None
    live_start_tick: int | None = None
    output_connected: bool = False
    loop_definition_active: bool = False


@dataclass(frozen=True, slots=True)
class GuiSnapshot:
    revision: int
    transport: GuiTransportSnapshot
    selected_channel_id: str
    recording_armed: bool
    recording_state: str
    recording_start_mode: str
    recording_stop_mode: str
    recording_reference_channel_id: str | None
    overwrite_channel_id: str | None
    channels: tuple[GuiChannelSnapshot, ...]
    input_connected: bool
    lifecycle: str
    notification_overflow_count: int
    midi_input_overflow_count: int
    terminal_message: str | None = None
    reset_mode: str = "OFF"
    reset_interval_beats: float | None = None
    recording_start_beat_count: int | None = None
    recording_stop_beat_count: int | None = None
    recording_start_gesture_count: int | None = None
    recording_stop_gesture_count: int | None = None
    recording_stop_after_ticks: int | None = None


@dataclass(frozen=True, slots=True)
class GuiNotification:
    request_id: int | None
    kind: str
    message: str
    severity: str = "info"
    channel_id: str | None = None
    tick: int | None = None
    tape_revision: int | None = None
    changed_intervals: tuple[VoiceInterval, ...] = ()
    removed_interval_ids: tuple[int, ...] = ()


class _InputPort(Protocol):
    def close(self) -> None: ...


class GuiCommandQueueFull(RuntimeError):
    """The host cannot accept another GUI command without losing it."""


class GuiHostUnavailable(RuntimeError):
    """The host worker is not running or is shutting down."""


class FourChannelGuiRuntime:
    """Serialize four-channel looper work away from front-end redraws."""

    def __init__(
        self,
        session: FourChannelSession,
        *,
        input_port: _InputPort | None = None,
        output_sinks: tuple[object, ...] = (),
        clock_ns=time.monotonic_ns,
        tick_interval: float = 0.002,
        snapshot_interval: float = 0.05,
        command_capacity: int = 128,
        notification_capacity: int = 256,
        input_capacity: int = 2048,
        shutdown_timeout: float = 2.0,
    ) -> None:
        if tick_interval <= 0 or snapshot_interval <= 0:
            raise ValueError("runtime intervals must be positive")
        if min(command_capacity, notification_capacity, input_capacity) <= 0:
            raise ValueError("queue capacities must be positive")
        if shutdown_timeout <= 0:
            raise ValueError("shutdown_timeout must be positive")

        self.session = session
        self.input_port = input_port
        self.output_sinks = output_sinks
        self.clock_ns = clock_ns
        self.tick_interval = tick_interval
        self.snapshot_interval = snapshot_interval
        self.shutdown_timeout = shutdown_timeout
        self._commands: queue.Queue[GuiIntent] = queue.Queue(command_capacity)
        self._notifications: queue.Queue[GuiNotification] = queue.Queue(
            notification_capacity
        )
        self._midi_events: queue.Queue[tuple[mido.Message, int]] = queue.Queue(
            input_capacity
        )
        self._wake = threading.Event()
        self._stop_requested = threading.Event()
        self._input_overflow_lock = threading.Lock()
        self._midi_input_overflow_count = 0
        self._notification_overflow_count = 0
        self._request_id = 0
        self._revision = 0
        self._tape_revisions = [tape.revision for tape in session.tapes]
        self._intervals = tuple(tape.intervals() for tape in session.tapes)
        self._live_voices = tuple(
            channel.articulator.live_voice
            for channel in session.coordinator.channels
        )
        self._live_start_ticks: list[int | None] = [
            None if voice is None else session.current_tick
            for voice in self._live_voices
        ]
        self._clock_origin_ns = clock_ns()
        self._tick_origin = session.current_tick
        self._transport_epoch = 0
        self._last_snapshot_ns = 0
        self._lifecycle = "ready"
        self._terminal_message: str | None = None
        self._thread: threading.Thread | None = None
        self._snapshot = self._make_snapshot(self._clock_origin_ns)

    @property
    def lifecycle(self) -> str:
        return self._snapshot.lifecycle

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self._thread is not None:
            if self._thread.is_alive():
                return
            raise GuiHostUnavailable("host runtime cannot be restarted")
        self._lifecycle = "running"
        now_ns = self.clock_ns()
        self._snapshot = self._make_snapshot(now_ns)
        self._thread = threading.Thread(
            target=self._run,
            name="py-seq-four-channel-host",
            daemon=True,
        )
        self._thread.start()

    def attach_input_port(self, input_port: _InputPort) -> None:
        if self._thread is not None or self._lifecycle != "ready":
            raise RuntimeError("input port must be attached before runtime start")
        if not callable(getattr(input_port, "close", None)):
            raise TypeError("input_port must provide close()")
        self.input_port = input_port
        self._snapshot = self._make_snapshot(self.clock_ns())

    def submit(
        self,
        kind: GuiIntentKind,
        *,
        channel_id: str | None = None,
        value: bool | int | str | None = None,
        bank_msb: int | None = None,
    ) -> int:
        if (
            not self.is_alive
            or self._lifecycle != "running"
            or self._stop_requested.is_set()
        ):
            raise GuiHostUnavailable("host runtime is not accepting commands")
        self._request_id += 1
        intent = GuiIntent(
            self._request_id,
            self.clock_ns(),
            kind,
            channel_id,
            value,
            bank_msb,
        )
        self._validate_intent(intent)
        try:
            self._commands.put_nowait(intent)
        except queue.Full as error:
            raise GuiCommandQueueFull(
                "host command queue is full; command was not accepted"
            ) from error
        self._wake.set()
        return intent.request_id

    def get_snapshot(self) -> GuiSnapshot:
        return self._snapshot

    def drain_notifications(self, limit: int = 64) -> tuple[GuiNotification, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        notifications: list[GuiNotification] = []
        for _ in range(limit):
            try:
                notifications.append(self._notifications.get_nowait())
            except queue.Empty:
                break
        return tuple(notifications)

    def enqueue_midi_message(self, message: mido.Message) -> None:
        if not isinstance(message, mido.Message):
            raise TypeError("message must be a mido.Message")
        try:
            self._midi_events.put_nowait((message.copy(), self.clock_ns()))
        except queue.Full:
            with self._input_overflow_lock:
                self._midi_input_overflow_count += 1
        self._wake.set()

    def close(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is None:
            if self._lifecycle == "ready":
                self._lifecycle = "shutting_down"
                self._cleanup()
                self._lifecycle = (
                    "failed" if self._terminal_message else "stopped"
                )
                self._publish_snapshot(self.clock_ns())
            return
        if thread.is_alive():
            self._stop_requested.set()
            self._wake.set()
            thread.join(self.shutdown_timeout if timeout is None else timeout)
        if thread.is_alive():
            raise TimeoutError("host runtime did not shut down before timeout")

    def _run(self) -> None:
        failure: str | None = None
        try:
            self._publish_snapshot(self.clock_ns())
            while not self._stop_requested.is_set():
                self._process_pending()
                if self._stop_requested.is_set():
                    break
                now_ns = self.clock_ns()
                if self._midi_input_has_overflowed():
                    raise RuntimeError(
                        "MIDI input queue overflow; stopping to avoid stuck notes"
                    )
                self._advance_to(self._tick_at(now_ns))
                if now_ns - self._last_snapshot_ns >= int(
                    self.snapshot_interval * 1_000_000_000
                ):
                    self._publish_snapshot(now_ns)
                self._wake.wait(self.tick_interval)
                self._wake.clear()
        except Exception as error:
            failure = f"host worker failed: {error}"
            logger.exception("four-channel GUI host worker failed")
        finally:
            if failure is None:
                self._lifecycle = "shutting_down"
            self._terminal_message = failure
            self._cleanup()
            self._lifecycle = (
                "failed" if self._terminal_message else "stopped"
            )
            self._publish_snapshot(self.clock_ns())
            self._notify(
                GuiNotification(
                    request_id=None,
                    kind="host_terminated",
                    message=failure or "Looper stopped",
                    severity="error" if failure else "info",
                )
            )
            self._publish_snapshot(self.clock_ns())

    def _process_pending(self) -> None:
        pending: list[tuple[int, int, object]] = []
        for _ in range(128):
            try:
                message, timestamp_ns = self._midi_events.get_nowait()
            except queue.Empty:
                break
            pending.append((timestamp_ns, 0, message))
        for _ in range(128):
            try:
                intent = self._commands.get_nowait()
            except queue.Empty:
                break
            pending.append((intent.submitted_at_ns, 1, intent))

        pending.sort(key=lambda item: (item[0], item[1]))
        processed = False
        for timestamp_ns, _, item in pending:
            if self._stop_requested.is_set():
                break
            tick = self._tick_at(timestamp_ns)
            if isinstance(item, mido.Message):
                processed = True
                try:
                    self.session.process_midi_message(item, tick)
                except Exception as error:
                    self._record_tape_changes(tick, None)
                    self._notify(
                        GuiNotification(
                            request_id=None,
                            kind="midi_input_error",
                            message=f"MIDI input processing failed: {error}",
                            severity="error",
                        )
                    )
                    continue
                self._sync_live_voices(tick)
                self._record_tape_changes(tick, None)
            else:
                intent = item
                assert isinstance(intent, GuiIntent)
                processed = True
                try:
                    self._apply_intent(intent, tick)
                    self._record_tape_changes(tick, intent.request_id)
                except Exception as error:
                    self._record_tape_changes(tick, intent.request_id)
                    self._notify(
                        GuiNotification(
                            request_id=intent.request_id,
                            kind="command_rejected",
                            message=str(error),
                            severity="error",
                            channel_id=intent.channel_id,
                            tick=tick,
                        )
                    )
                else:
                    if intent.kind is not GuiIntentKind.SHUTDOWN:
                        self._notify(
                            GuiNotification(
                                request_id=intent.request_id,
                                kind="command_applied",
                                message=self._success_message(intent),
                                channel_id=intent.channel_id,
                                tick=tick,
                            )
                        )
        if processed and not self._stop_requested.is_set():
            now_ns = self.clock_ns()
            self._advance_to(self._tick_at(now_ns))
            self._publish_snapshot(now_ns)

    def _apply_intent(self, intent: GuiIntent, tick: int) -> None:
        if intent.kind is GuiIntentKind.RESET:
            previous_reset_mode = self.session.reset_mode
            self.session.reset(tick)
            self._report_invalidated_loop_reset(previous_reset_mode)
            return
        self._advance_to(tick)
        if intent.kind is GuiIntentKind.SHUTDOWN:
            self._stop_requested.set()
            return
        if intent.kind is GuiIntentKind.TRANSPORT:
            desired = bool(intent.value)
            if desired != self.session.transport_running:
                self.session.toggle_transport()
                self._clock_origin_ns = intent.submitted_at_ns
                self._tick_origin = self.session.current_tick
                self._transport_epoch += 1
            return
        if intent.kind is GuiIntentKind.RECORDING_ARM:
            self.session.set_recording_armed(bool(intent.value))
            return
        if intent.kind is GuiIntentKind.RECORDING_TOGGLE:
            self.session.toggle_recording()
            return
        if intent.kind is GuiIntentKind.RECORDING_SYNC:
            if not isinstance(intent.value, str):
                raise TypeError("recording sync command must be a string")
            command = parse_recording_sync_command(intent.value)
            self.session.configure_recording_sync(
                start_mode=command.start_mode,
                stop_mode=command.stop_mode,
                reference_channel=command.reference_channel,
                start_beat_count=command.start_beat_count,
                stop_beat_count=command.stop_beat_count,
                start_gesture_count=command.start_gesture_count,
                stop_gesture_count=command.stop_gesture_count,
                stop_after_ticks=command.stop_after_ticks,
            )
            return
        if intent.kind is GuiIntentKind.RESET_CONFIG:
            if not isinstance(intent.value, str):
                raise TypeError("reset configuration must be a string")
            command = parse_reset_command(intent.value)
            self.session.configure_reset(
                command.mode,
                interval_ticks=command.interval_ticks,
            )
            return
        if intent.kind is GuiIntentKind.LOOP_LENGTH:
            if not isinstance(intent.value, str):
                raise TypeError("loop length command must be a string")
            length_ticks = parse_length_command(intent.value)
            self.session.set_loop_length(
                self._channel_index(intent.channel_id),
                length_ticks,
            )
            return

        channel_index = self._channel_index(intent.channel_id)
        if intent.kind is GuiIntentKind.SELECT_CHANNEL:
            self.session.select_channel(channel_index)
            self._sync_live_voices(tick)
        elif channel_index != self.session.active_channel:
            raise ValueError("selected channel changed before command was applied")
        elif intent.kind is GuiIntentKind.OVERWRITE:
            should_overwrite = bool(intent.value)
            is_overwrite_armed = (
                self.session.mode is RecordMode.OVERWRITE
                and self.session.record_armed
            )
            if should_overwrite and not is_overwrite_armed:
                if self.session.mode is RecordMode.OVERWRITE:
                    self.session.set_recording_armed(True)
                else:
                    self.session.toggle_mode()
            elif not should_overwrite and is_overwrite_armed:
                self.session.toggle_mode()
        elif intent.kind is GuiIntentKind.CLEAR_TAPE:
            previous_reset_mode = self.session.reset_mode
            self.session.clear_tape(channel_index)
            self._report_invalidated_loop_reset(previous_reset_mode)
        elif intent.kind is GuiIntentKind.PROGRAM:
            if self.session.routes[channel_index].port_index is None:
                raise ValueError("selected channel has no MIDI output route")
            value = intent.value
            assert isinstance(value, int)
            self.session.change_program(value)
        elif intent.kind is GuiIntentKind.BANK_PROGRAM:
            if self.session.routes[channel_index].port_index is None:
                raise ValueError("selected channel has no MIDI output route")
            value = intent.value
            assert isinstance(value, int) and intent.bank_msb is not None
            self.session.change_bank_program(intent.bank_msb, value)
        else:
            raise ValueError(f"unsupported intent: {intent.kind}")

    def _tick_at(self, timestamp_ns: int) -> int:
        if not self.session.transport_running:
            return self.session.current_tick
        elapsed_ns = max(0, timestamp_ns - self._clock_origin_ns)
        ticks = elapsed_ns * self.session.bpm * TICKS_PER_BEAT / 60_000_000_000
        return max(self.session.current_tick, self._tick_origin + int(ticks))

    def _advance_to(self, tick: int) -> None:
        if tick > self.session.current_tick:
            previous_reset_mode = self.session.reset_mode
            self.session.advance(tick)
            self._report_invalidated_loop_reset(previous_reset_mode)
            self._record_tape_changes(tick, None)

    def _report_invalidated_loop_reset(self, previous_mode: ResetMode) -> None:
        if (
            previous_mode is ResetMode.LOOP
            and self.session.reset_mode is ResetMode.OFF
        ):
            self._notify(
                GuiNotification(
                    request_id=None,
                    kind="reset_disabled",
                    message=(
                        "Loop reset disabled: reference channel has no defined loop"
                    ),
                    severity="error",
                )
            )

    def _record_tape_changes(
        self,
        tick: int,
        request_id: int | None,
    ) -> None:
        for index, tape in enumerate(self.session.tapes):
            if tape.revision == self._tape_revisions[index]:
                continue
            old_intervals = self._intervals[index]
            new_intervals = tape.intervals()
            old_by_id = {interval.id: interval for interval in old_intervals}
            new_by_id = {interval.id: interval for interval in new_intervals}
            changed = tuple(
                interval
                for interval in new_intervals
                if old_by_id.get(interval.id) != interval
            )
            removed = tuple(sorted(old_by_id.keys() - new_by_id.keys()))
            self._intervals = tuple(
                new_intervals if channel == index else self._intervals[channel]
                for channel in range(CHANNEL_COUNT)
            )
            self._tape_revisions[index] = tape.revision
            self._notify(
                GuiNotification(
                    request_id=request_id,
                    kind="tape_changed",
                    message=f"Channel {index + 1} tape changed",
                    channel_id=_CHANNEL_IDS[index],
                    tick=tick - self.session.position_origin_tick,
                    tape_revision=self._tape_revisions[index],
                    changed_intervals=changed,
                    removed_interval_ids=removed,
                )
            )

    def _sync_live_voices(self, tick: int) -> None:
        for index, channel in enumerate(self.session.coordinator.channels):
            live_voice = channel.articulator.live_voice
            if live_voice != self._live_voices[index]:
                self._live_voices = tuple(
                    live_voice if position == index else previous
                    for position, previous in enumerate(self._live_voices)
                )
                self._live_start_ticks[index] = (
                    tick if live_voice is not None else None
                )

    def _make_snapshot(self, now_ns: int) -> GuiSnapshot:
        channels = tuple(self._channel_snapshot(index) for index in range(CHANNEL_COUNT))
        selected = self.session.active_channel
        overwrite = (
            _CHANNEL_IDS[selected]
            if self.session.mode is RecordMode.OVERWRITE
            and self.session.record_armed
            else None
        )
        transport = GuiTransportSnapshot(
            position_ticks=self.session.musical_tick,
            tempo_bpm=self.session.bpm,
            playing=self.session.transport_running,
            sampled_at_ns=now_ns,
            epoch=self._transport_epoch,
        )
        with self._input_overflow_lock:
            input_overflow_count = self._midi_input_overflow_count
        return GuiSnapshot(
            revision=self._revision,
            transport=transport,
            selected_channel_id=_CHANNEL_IDS[selected],
            recording_armed=self.session.record_armed,
            recording_state=self.session.recording_state.name,
            recording_start_mode=self.session.start_mode.name,
            recording_stop_mode=self.session.stop_mode.name,
            recording_reference_channel_id=(
                None
                if self.session.reference_channel is None
                else _CHANNEL_IDS[self.session.reference_channel]
            ),
            overwrite_channel_id=overwrite,
            channels=channels,
            input_connected=self.input_port is not None,
            lifecycle=self._lifecycle,
            notification_overflow_count=self._notification_overflow_count,
            midi_input_overflow_count=input_overflow_count,
            terminal_message=self._terminal_message,
            reset_mode=self.session.reset_mode.name,
            reset_interval_beats=self.session.reset_interval_beats,
            recording_start_beat_count=self.session.start_beat_count,
            recording_stop_beat_count=self.session.stop_beat_count,
            recording_start_gesture_count=self.session.start_gesture_count,
            recording_stop_gesture_count=self.session.stop_gesture_count,
            recording_stop_after_ticks=self.session.stop_after_ticks,
        )

    def _channel_snapshot(self, index: int) -> GuiChannelSnapshot:
        route = self.session.routes[index]
        live_voice = self.session.coordinator.channels[index].articulator.live_voice
        return GuiChannelSnapshot(
            channel_id=_CHANNEL_IDS[index],
            name=route.label,
            midi_channel=route.midi_channel,
            bank_msb=self.session.bank_msbs[index],
            program=self.session.programs[index],
            loop_length_ticks=self.session.tapes[index].length_ticks,
            phase_origin_tick=(
                self.session.phase_origin_tick(index)
                - self.session.position_origin_tick
            ),
            tape_revision=self._tape_revisions[index],
            intervals=self._intervals[index],
            live_pitch=None if live_voice is None else live_voice.pitch,
            live_velocity=None if live_voice is None else live_voice.velocity,
            live_start_tick=(
                None
                if self._live_start_ticks[index] is None
                else self._live_start_ticks[index]
                - self.session.position_origin_tick
            ),
            output_connected=route.port_index is not None,
            loop_definition_active=self.session.loop_definition_channel == index,
        )

    def _publish_snapshot(self, now_ns: int) -> None:
        self._revision += 1
        self._snapshot = self._make_snapshot(now_ns)
        self._last_snapshot_ns = now_ns

    def _notify(self, notification: GuiNotification) -> None:
        try:
            self._notifications.put_nowait(notification)
        except queue.Full:
            self._notification_overflow_count += 1

    def _midi_input_has_overflowed(self) -> bool:
        with self._input_overflow_lock:
            return self._midi_input_overflow_count > 0

    def _cleanup(self) -> None:
        cleanup_errors: list[str] = []
        for description, operation in (
            ("session close", self.session.close),
            ("MIDI panic", self.session.panic),
        ):
            try:
                operation()
            except Exception as error:
                cleanup_errors.append(f"{description}: {error}")
                logger.exception("host cleanup failed during %s", description)

        if self.input_port is not None:
            try:
                self.input_port.close()
            except Exception as error:
                cleanup_errors.append(f"MIDI input close: {error}")
                logger.exception("failed to close MIDI input")
        closed: set[int] = set()
        for sink in self.output_sinks:
            if id(sink) in closed:
                continue
            closed.add(id(sink))
            close = getattr(sink, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as error:
                    cleanup_errors.append(f"MIDI output close: {error}")
                    logger.exception("failed to close MIDI output")
        if cleanup_errors:
            detail = "; ".join(cleanup_errors)
            self._terminal_message = (
                f"{self._terminal_message}; " if self._terminal_message else ""
            ) + f"cleanup errors: {detail}"
            if self._lifecycle != "failed":
                self._lifecycle = "failed"

    def _success_message(self, intent: GuiIntent) -> str:
        if intent.kind is GuiIntentKind.TRANSPORT:
            return "Transport playing" if intent.value else "Transport stopped"
        if intent.kind is GuiIntentKind.SELECT_CHANNEL:
            return f"Selected channel {intent.channel_id}"
        if intent.kind is GuiIntentKind.RECORDING_ARM:
            return "Recording armed" if intent.value else "Recording off"
        if intent.kind is GuiIntentKind.RECORDING_TOGGLE:
            return self.session.recording_state.name.replace("_", " ").title()
        if intent.kind is GuiIntentKind.RECORDING_SYNC:
            start_mode = self.session.start_mode.name
            stop_mode = self.session.stop_mode.name
            if self.session.start_beat_count is not None:
                start_mode += f" {self.session.start_beat_count}"
            if self.session.stop_beat_count is not None:
                stop_mode += f" {self.session.stop_beat_count}"
            if self.session.stop_after_ticks is not None:
                stop_mode += (
                    f" {self.session.stop_after_ticks / TICKS_PER_BEAT:g}"
                )
            if self.session.start_gesture_count is not None:
                start_mode += f" {self.session.start_gesture_count}"
            if self.session.stop_gesture_count is not None:
                stop_mode += f" {self.session.stop_gesture_count}"
            return (
                f"Start {start_mode}; stop {stop_mode}"
            )
        if intent.kind is GuiIntentKind.RESET:
            return "Musical position reset"
        if intent.kind is GuiIntentKind.RESET_CONFIG:
            if self.session.reset_mode is ResetMode.BEAT:
                return (
                    "Automatic reset every "
                    f"{self.session.reset_interval_beats:g} beats"
                )
            if self.session.reset_mode is ResetMode.LOOP:
                return "Automatic reset at reference loop wraps"
            return "Automatic reset off"
        if intent.kind is GuiIntentKind.LOOP_LENGTH:
            length = self.session.tapes[
                self._channel_index(intent.channel_id)
            ].length_ticks
            return f"Loop length set to {length} ticks"
        if intent.kind is GuiIntentKind.OVERWRITE:
            return "Overwrite enabled" if intent.value else "Overwrite disabled"
        if intent.kind is GuiIntentKind.CLEAR_TAPE:
            return f"Cleared {intent.channel_id}"
        if intent.kind is GuiIntentKind.PROGRAM:
            value = intent.value
            if not isinstance(value, int) or isinstance(value, bool):
                raise RuntimeError("validated program intent has no program value")
            return f"Program {value:03d} sent"
        if intent.kind is GuiIntentKind.BANK_PROGRAM:
            value = intent.value
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or intent.bank_msb is None
            ):
                raise RuntimeError("validated bank/program intent has incomplete values")
            return f"Bank {intent.bank_msb:03d}, program {value:03d} sent"
        return "Host shutdown requested"

    @staticmethod
    def _channel_index(channel_id: str | None) -> int:
        if channel_id not in _CHANNEL_IDS:
            raise ValueError(f"unknown channel ID: {channel_id!r}")
        return _CHANNEL_IDS.index(channel_id)

    @staticmethod
    def _validate_intent(intent: GuiIntent) -> None:
        if not isinstance(intent.kind, GuiIntentKind):
            raise TypeError("kind must be a GuiIntentKind")
        if intent.kind in (
            GuiIntentKind.SELECT_CHANNEL,
            GuiIntentKind.OVERWRITE,
            GuiIntentKind.CLEAR_TAPE,
            GuiIntentKind.PROGRAM,
            GuiIntentKind.BANK_PROGRAM,
            GuiIntentKind.LOOP_LENGTH,
        ) and intent.channel_id is None:
            raise ValueError(f"{intent.kind.name} requires a channel ID")
        if intent.kind in (
            GuiIntentKind.TRANSPORT,
            GuiIntentKind.RECORDING_ARM,
            GuiIntentKind.OVERWRITE,
        ) and not isinstance(intent.value, bool):
            raise TypeError(f"{intent.kind.name} requires a bool value")
        if intent.kind is GuiIntentKind.RECORDING_SYNC and not isinstance(
            intent.value, str
        ):
            raise TypeError("RECORDING_SYNC requires a command string")
        if intent.kind is GuiIntentKind.RESET_CONFIG and not isinstance(
            intent.value, str
        ):
            raise TypeError("RESET_CONFIG requires a command string")
        if intent.kind is GuiIntentKind.LOOP_LENGTH and not isinstance(
            intent.value, str
        ):
            raise TypeError("LOOP_LENGTH requires a command string")
        if intent.kind in (GuiIntentKind.PROGRAM, GuiIntentKind.BANK_PROGRAM):
            if isinstance(intent.value, bool) or not isinstance(intent.value, int):
                raise TypeError(f"{intent.kind.name} requires an integer value")
            if not 0 <= intent.value <= 127:
                raise ValueError(f"{intent.kind.name} value must be between 0 and 127")
        if intent.kind is GuiIntentKind.BANK_PROGRAM:
            if isinstance(intent.bank_msb, bool) or not isinstance(
                intent.bank_msb, int
            ):
                raise TypeError("BANK_PROGRAM requires an integer bank_msb")
            if not 0 <= intent.bank_msb <= 127:
                raise ValueError("bank_msb must be between 0 and 127")


__all__ = [
    "FourChannelGuiRuntime",
    "GuiChannelSnapshot",
    "GuiCommandQueueFull",
    "GuiHostUnavailable",
    "GuiIntent",
    "GuiIntentKind",
    "GuiNotification",
    "GuiSnapshot",
    "GuiTransportSnapshot",
]
