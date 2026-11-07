from __future__ import annotations

import threading
import time

import mido
import pytest

from midi_output import CollectingMidiSink
from py_seq.four_channel_session import FourChannelSession, MidiRoute, ResetMode
from py_seq.gui_runtime import (
    FourChannelGuiRuntime,
    GuiCommandQueueFull,
    GuiIntentKind,
)
from tape import IntervalInsert


def make_session() -> tuple[
    FourChannelSession,
    tuple[
        CollectingMidiSink,
        CollectingMidiSink,
        CollectingMidiSink,
        CollectingMidiSink,
    ],
]:
    sinks = (
        CollectingMidiSink(),
        CollectingMidiSink(),
        CollectingMidiSink(),
        CollectingMidiSink(),
    )
    routes = (
        MidiRoute("OUT1", 0, sinks[0], 0),
        MidiRoute("OUT2", 1, sinks[1], 1),
        MidiRoute("OUT3", 2, sinks[2], 2),
        MidiRoute("OUT4", 3, sinks[3], 3),
    )
    return FourChannelSession((3840, 4800, 5760, 6720), 120, routes), sinks


def wait_for_notification(runtime: FourChannelGuiRuntime, kind: str):
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        for notification in runtime.drain_notifications():
            if notification.kind == kind:
                return notification
        time.sleep(0.002)
    raise AssertionError(f"timed out waiting for {kind}")


def test_runtime_publishes_all_four_lanes_and_transport_snapshot():
    session, _ = make_session()
    runtime = FourChannelGuiRuntime(session)
    runtime.start()
    try:
        assert len(runtime.get_snapshot().channels) == 4
        assert runtime.get_snapshot().channels[2].phase_origin_tick == 0

        request_id = runtime.submit(
            GuiIntentKind.TRANSPORT,
            value=True,
        )
        notification = wait_for_notification(runtime, "command_applied")
        assert notification.request_id == request_id
        snapshot = runtime.get_snapshot()
        assert snapshot.transport.playing
        assert snapshot.transport.sampled_at_ns > 0
        assert snapshot.transport.ticks_per_beat == 960
    finally:
        runtime.close()
    assert runtime.get_snapshot().lifecycle == "stopped"


def test_runtime_snapshot_represents_uninitialized_loops():
    sinks = tuple(CollectingMidiSink() for _ in range(4))
    routes = tuple(
        MidiRoute(f"OUT{index + 1}", index, sinks[index], index)
        for index in range(4)
    )
    session = FourChannelSession((0, 0, 0, 0), 120, routes)
    runtime = FourChannelGuiRuntime(session)

    assert tuple(
        channel.loop_length_ticks
        for channel in runtime.get_snapshot().channels
    ) == (0, 0, 0, 0)


def test_runtime_snapshot_marks_a_loop_until_its_definition_is_finalized():
    sinks = tuple(CollectingMidiSink() for _ in range(4))
    routes = tuple(
        MidiRoute(f"OUT{index + 1}", index, sinks[index], index)
        for index in range(4)
    )
    session = FourChannelSession((0, 0, 0, 0), 120, routes)
    session.toggle_transport()
    session.set_recording_armed(True)
    session.advance(120)

    snapshot = FourChannelGuiRuntime(session).get_snapshot()

    assert snapshot.channels[0].loop_length_ticks == 120
    assert snapshot.channels[0].loop_definition_active
    assert not any(
        channel.loop_definition_active for channel in snapshot.channels[1:]
    )

    session.request_recording_stop()
    finalized_snapshot = FourChannelGuiRuntime(session).get_snapshot()

    assert not any(
        channel.loop_definition_active for channel in finalized_snapshot.channels
    )
    session.toggle_transport()


def test_host_worker_schedules_tape_without_gui_frame_updates():
    class ManualClock:
        def __init__(self) -> None:
            self.value = 1_000_000_000
            self.lock = threading.Lock()

        def __call__(self) -> int:
            with self.lock:
                return self.value

        def advance(self, nanoseconds: int) -> None:
            with self.lock:
                self.value += nanoseconds

    session, sinks = make_session()
    session.tape(0).insert(IntervalInsert(10, 10, 64, 90, 1))
    clock = ManualClock()
    runtime = FourChannelGuiRuntime(
        session,
        clock_ns=clock,
        snapshot_interval=1,
    )
    runtime.start()
    try:
        runtime.submit(GuiIntentKind.TRANSPORT, value=True)
        wait_for_notification(runtime, "command_applied")
        clock.advance(6_000_000)
        deadline = time.monotonic() + 1
        while not any(
            getattr(message, "type") == "note_on" for message in sinks[0].messages
        ) and time.monotonic() < deadline:
            time.sleep(0.002)
        assert any(
            getattr(message, "type") == "note_on" for message in sinks[0].messages
        )

        clock.advance(5_000_000)
        deadline = time.monotonic() + 1
        while not any(
            getattr(message, "type") == "note_off" for message in sinks[0].messages
        ) and time.monotonic() < deadline:
            time.sleep(0.002)
        assert any(
            getattr(message, "type") == "note_off" for message in sinks[0].messages
        )
    finally:
        runtime.close()


def test_recorder_edits_publish_tape_changes_for_recent_note_feedback():
    class ManualClock:
        def __init__(self) -> None:
            self.value = 2_000_000_000
            self.lock = threading.Lock()

        def __call__(self) -> int:
            with self.lock:
                return self.value

        def advance(self, nanoseconds: int) -> None:
            with self.lock:
                self.value += nanoseconds

    session, _ = make_session()
    clock = ManualClock()
    runtime = FourChannelGuiRuntime(session, clock_ns=clock)
    runtime.start()
    try:
        runtime.submit(GuiIntentKind.RECORDING_ARM, value=True)
        wait_for_notification(runtime, "command_applied")
        runtime.submit(GuiIntentKind.TRANSPORT, value=True)
        wait_for_notification(runtime, "command_applied")

        clock.advance(50_000_000)
        runtime.enqueue_midi_message(
            mido.Message("note_on", channel=0, note=60, velocity=90)
        )
        time.sleep(0.02)
        clock.advance(50_000_000)
        runtime.enqueue_midi_message(
            mido.Message("note_off", channel=0, note=60, velocity=0)
        )
        notification = wait_for_notification(runtime, "tape_changed")

        assert notification.channel_id == "channel-1"
        assert len(notification.changed_intervals) == 1
        interval = notification.changed_intervals[0]
        assert interval.pitch == 60
        assert interval.duration > 0
        channel = runtime.get_snapshot().channels[0]
        assert channel.intervals == (interval,)
        assert channel.tape_revision == notification.tape_revision
    finally:
        runtime.close()


def test_runtime_serializes_recording_overwrite_selection_and_clear():
    session, _ = make_session()
    session.tape(0).insert(IntervalInsert(0, 120, 60, 90, 1))
    removed = session.tape(1).insert(IntervalInsert(240, 120, 62, 80, 2))
    runtime = FourChannelGuiRuntime(session)
    runtime.start()
    try:
        runtime.submit(
            GuiIntentKind.OVERWRITE,
            channel_id="channel-1",
            value=True,
        )
        wait_for_notification(runtime, "command_applied")
        assert runtime.get_snapshot().recording_armed
        assert runtime.get_snapshot().overwrite_channel_id == "channel-1"

        runtime.submit(
            GuiIntentKind.RECORDING_ARM,
            value=False,
        )
        wait_for_notification(runtime, "command_applied")
        assert not runtime.get_snapshot().recording_armed
        assert runtime.get_snapshot().overwrite_channel_id is None

        runtime.submit(
            GuiIntentKind.OVERWRITE,
            channel_id="channel-1",
            value=True,
        )
        wait_for_notification(runtime, "command_applied")
        assert runtime.get_snapshot().recording_armed
        assert runtime.get_snapshot().overwrite_channel_id == "channel-1"

        runtime.submit(
            GuiIntentKind.SELECT_CHANNEL,
            channel_id="channel-2",
        )
        wait_for_notification(runtime, "command_applied")
        snapshot = runtime.get_snapshot()
        assert snapshot.selected_channel_id == "channel-2"
        assert snapshot.recording_armed
        assert snapshot.overwrite_channel_id is None

        runtime.submit(
            GuiIntentKind.CLEAR_TAPE,
            channel_id="channel-2",
        )
        notifications = []
        deadline = time.monotonic() + 1
        while (
            {item.kind for item in notifications}
            < {"tape_changed", "command_applied"}
            and time.monotonic() < deadline
        ):
            notifications.extend(runtime.drain_notifications())
            time.sleep(0.002)
        tape_change = next(
            item for item in notifications if item.kind == "tape_changed"
        )
        assert any(item.kind == "command_applied" for item in notifications)
        assert tape_change.channel_id == "channel-2"
        assert tape_change.removed_interval_ids == (removed.id,)
        assert tape_change.tape_revision == 2
        assert not session.tape(1).intervals()
        assert session.tape(0).intervals()
    finally:
        runtime.close()


def test_runtime_applies_sync_configuration_and_reports_armed_state():
    session, _ = make_session()
    runtime = FourChannelGuiRuntime(session)
    runtime.start()
    try:
        runtime.submit(
            GuiIntentKind.RECORDING_SYNC,
            value="start beat",
        )
        wait_for_notification(runtime, "command_applied")
        runtime.submit(GuiIntentKind.RECORDING_TOGGLE)
        wait_for_notification(runtime, "command_applied")

        snapshot = runtime.get_snapshot()
        assert snapshot.recording_start_mode == "NEXT_BEAT"
        assert snapshot.recording_state == "WAITING_START"
        assert snapshot.recording_armed
        assert not session.recorder(0).record_armed
    finally:
        runtime.close()


def test_runtime_reports_self_reference_arm_as_an_error():
    session, _ = make_session()
    runtime = FourChannelGuiRuntime(session)
    runtime.start()
    try:
        runtime.submit(
            GuiIntentKind.RECORDING_SYNC,
            value="reference 1",
        )
        wait_for_notification(runtime, "command_applied")
        runtime.submit(
            GuiIntentKind.RECORDING_SYNC,
            value="start loop",
        )
        wait_for_notification(runtime, "command_applied")

        runtime.submit(GuiIntentKind.RECORDING_TOGGLE)
        notification = wait_for_notification(runtime, "command_rejected")

        assert notification.severity == "error"
        assert notification.message == (
            "recording channel cannot use itself as the reference channel"
        )
        snapshot = runtime.get_snapshot()
        assert not snapshot.recording_armed
        assert snapshot.recording_state == "IDLE"
        assert snapshot.recording_reference_channel_id == "channel-1"
        assert snapshot.recording_start_mode == "NEXT_REFERENCE_LOOP_START"
    finally:
        runtime.close()


def test_runtime_reset_intents_preserve_automatic_mode_and_publish_musical_position():
    session, _ = make_session()
    runtime = FourChannelGuiRuntime(session)
    runtime.start()
    try:
        runtime.submit(
            GuiIntentKind.RESET_CONFIG,
            value="reset beat 2",
        )
        wait_for_notification(runtime, "command_applied")
        runtime.submit(GuiIntentKind.TRANSPORT, value=True)
        wait_for_notification(runtime, "command_applied")

        runtime.submit(GuiIntentKind.RESET)
        notification = wait_for_notification(runtime, "command_applied")

        assert notification.message == "Musical position reset"
        snapshot = runtime.get_snapshot()
        assert snapshot.transport.position_ticks == 0
        assert snapshot.reset_mode == ResetMode.BEAT.name
        assert snapshot.reset_interval_beats == 2
        assert tuple(channel.phase_origin_tick for channel in snapshot.channels) == (
            0,
            0,
            0,
            0,
        )
        assert snapshot.transport.playing
    finally:
        runtime.close()


def test_runtime_displays_error_when_clearing_loop_reset_reference():
    session, _ = make_session()
    session.configure_recording_sync(reference_channel=0)
    session.configure_reset(ResetMode.LOOP)
    runtime = FourChannelGuiRuntime(session)
    runtime.start()
    try:
        runtime.submit(
            GuiIntentKind.CLEAR_TAPE,
            channel_id="channel-1",
        )
        notification = wait_for_notification(runtime, "reset_disabled")

        assert notification.severity == "error"
        assert session.reset_mode is ResetMode.OFF
        assert runtime.get_snapshot().reset_mode == ResetMode.OFF.name
    finally:
        runtime.close()


def test_bank_program_sends_cc0_then_program_change_without_cc32():
    session, sinks = make_session()
    session.select_channel(2)
    session.change_bank_program(7, 42)

    messages = sinks[2].messages
    assert [
        (
            getattr(message, "type"),
            getattr(message, "channel"),
            getattr(message, "control", None),
        )
        for message in messages
    ] == [
        ("control_change", 2, 0),
        ("program_change", 2, None),
    ]
    assert [getattr(message, "value", None) for message in messages] == [7, None]
    assert [getattr(message, "program", None) for message in messages] == [None, 42]
    assert session.bank_msbs[2] == 7
    assert session.programs[2] == 42
    session.close()


def test_bank_program_intent_is_acknowledged_and_program_only_retains_bank():
    session, sinks = make_session()
    runtime = FourChannelGuiRuntime(session)
    runtime.start()
    try:
        runtime.submit(
            GuiIntentKind.BANK_PROGRAM,
            channel_id="channel-1",
            value=42,
            bank_msb=7,
        )
        wait_for_notification(runtime, "command_applied")
        runtime.submit(
            GuiIntentKind.PROGRAM,
            channel_id="channel-1",
            value=43,
        )
        wait_for_notification(runtime, "command_applied")

        assert [
            getattr(message, "type") for message in sinks[0].messages
        ] == ["control_change", "program_change", "program_change"]
        assert [getattr(message, "control", None) for message in sinks[0].messages] == [
            0,
            None,
            None,
        ]
        snapshot = runtime.get_snapshot().channels[0]
        assert snapshot.bank_msb == 7
        assert snapshot.program == 43
    finally:
        runtime.close()


def test_program_intent_rejects_unrouted_output_instead_of_claiming_success():
    sink = CollectingMidiSink()
    routes = (
        MidiRoute("NO OUT", 0, sink),
        MidiRoute("NO OUT", 1, CollectingMidiSink()),
        MidiRoute("NO OUT", 2, CollectingMidiSink()),
        MidiRoute("NO OUT", 3, CollectingMidiSink()),
    )
    session = FourChannelSession((3840, 4800, 5760, 6720), 120, routes)
    runtime = FourChannelGuiRuntime(session)
    runtime.start()
    try:
        runtime.submit(
            GuiIntentKind.PROGRAM,
            channel_id="channel-1",
            value=42,
        )
        notification = wait_for_notification(runtime, "command_rejected")
        assert notification.message == "selected channel has no MIDI output route"
        assert session.programs[0] == 0
        assert not sink.messages
    finally:
        runtime.close()


class BlockingSink(CollectingMidiSink):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def send(self, message: mido.Message) -> None:
        if getattr(message, "type") == "program_change":
            self.entered.set()
            self.release.wait(1)
        super().send(message)


def test_full_command_queue_rejects_instead_of_dropping():
    sink = BlockingSink()
    other_sinks = (
        CollectingMidiSink(),
        CollectingMidiSink(),
        CollectingMidiSink(),
    )
    routes = (
        MidiRoute("OUT1", 0, sink, 0),
        MidiRoute("OUT2", 1, other_sinks[0], 1),
        MidiRoute("OUT3", 2, other_sinks[1], 2),
        MidiRoute("OUT4", 3, other_sinks[2], 3),
    )
    session = FourChannelSession((3840, 3840, 3840, 3840), 120, routes)
    runtime = FourChannelGuiRuntime(session, command_capacity=1)
    runtime.start()
    try:
        runtime.submit(
            GuiIntentKind.PROGRAM,
            channel_id="channel-1",
            value=12,
        )
        assert sink.entered.wait(1)
        runtime.submit(
            GuiIntentKind.PROGRAM,
            channel_id="channel-1",
            value=13,
        )
        with pytest.raises(GuiCommandQueueFull, match="was not accepted"):
            runtime.submit(
                GuiIntentKind.PROGRAM,
                channel_id="channel-1",
                value=14,
            )
    finally:
        sink.release.set()
        runtime.close()


def test_notification_overflow_remains_visible_in_latest_snapshot():
    session, _ = make_session()
    runtime = FourChannelGuiRuntime(session, notification_capacity=1)
    runtime.start()
    try:
        runtime.submit(
            GuiIntentKind.SELECT_CHANNEL,
            channel_id="channel-2",
        )
        runtime.submit(
            GuiIntentKind.SELECT_CHANNEL,
            channel_id="channel-3",
        )
        deadline = time.monotonic() + 1
        while (
            runtime.get_snapshot().notification_overflow_count == 0
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        assert runtime.get_snapshot().notification_overflow_count == 1
    finally:
        runtime.close()


def test_midi_input_queue_overflow_terminates_host_and_is_visible():
    session, _ = make_session()
    runtime = FourChannelGuiRuntime(session, input_capacity=1)
    runtime.enqueue_midi_message(
        mido.Message("note_on", channel=0, note=60, velocity=90)
    )
    runtime.enqueue_midi_message(
        mido.Message("note_off", channel=0, note=60, velocity=0)
    )
    runtime.start()
    try:
        deadline = time.monotonic() + 1
        while runtime.lifecycle != "failed" and time.monotonic() < deadline:
            time.sleep(0.002)
        snapshot = runtime.get_snapshot()
        assert snapshot.lifecycle == "failed"
        assert snapshot.midi_input_overflow_count == 1
        assert "queue overflow" in (snapshot.terminal_message or "")
    finally:
        runtime.close()


def test_shutdown_panics_and_closes_each_unique_sink():
    class ClosingSink(CollectingMidiSink):
        def __init__(self):
            super().__init__()
            self.closed = False

        def close(self):
            self.closed = True

    sink = ClosingSink()
    routes = (
        MidiRoute("R0", 0, sink, 0),
        MidiRoute("R1", 1, sink, 1),
        MidiRoute("R2", 2, sink, 2),
        MidiRoute("R3", 3, sink, 3),
    )
    session = FourChannelSession((3840, 3840, 3840, 3840), 120, routes)
    runtime = FourChannelGuiRuntime(session, output_sinks=(sink, sink))
    runtime.start()
    runtime.close()

    panic_messages = [
        message
        for message in sink.messages
        if getattr(message, "type") == "control_change"
        and getattr(message, "control") == 123
    ]
    assert len(panic_messages) == 4
    assert sink.closed
    assert runtime.get_snapshot().lifecycle == "stopped"


def test_shutdown_cleanup_failure_is_reported_in_terminal_snapshot():
    class FailingCloseSink(CollectingMidiSink):
        def close(self):
            raise RuntimeError("port close failed")

    sink = FailingCloseSink()
    routes = tuple(
        MidiRoute(f"OUT{index + 1}", index, sink, index)
        for index in range(4)
    )
    session = FourChannelSession((3840, 3840, 3840, 3840), 120, routes)
    runtime = FourChannelGuiRuntime(session, output_sinks=(sink,))
    runtime.start()
    runtime.close()

    snapshot = runtime.get_snapshot()
    assert snapshot.lifecycle == "failed"
    assert "MIDI output close: port close failed" in (
        snapshot.terminal_message or ""
    )
