from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import mido
import pytest

from midi_output import CollectingMidiSink
from py_seq.four_channel_session import (
    FourChannelSession,
    MidiRoute,
    RecordingStartMode,
    RecordingStopMode,
    ResetMode,
)
from py_seq.gui_runtime import (
    FourChannelGuiRuntime,
    GuiCommandQueueFull,
    GuiIntent,
    GuiIntentKind,
    _MidiClockTempo,
)
from py_seq.pygame_gui import _project_recording_toggle, _reconcile_projected_state
from py_seq import pygame_gui
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


def test_midi_backlog_is_processed_before_later_recording_stop():
    now = [0]
    session, _ = make_session()
    session.toggle_transport()
    session.toggle_recording()
    runtime = FourChannelGuiRuntime(session, clock_ns=lambda: now[0])
    now[0] = 10_000_000
    for _ in range(128):
        runtime.enqueue_midi_message(mido.Message("clock"))
    runtime.enqueue_midi_message(mido.Message("note_on", note=60, velocity=90))
    runtime._commands.put_nowait(
        GuiIntent(1, 20_000_000, GuiIntentKind.RECORDING_TOGGLE)
    )
    now[0] = 30_000_000
    runtime._process_pending()

    assert session.recording_state.name == "IDLE"
    (interval,) = session.tape(0).intervals()
    assert (interval.start, interval.duration, interval.pitch) == (19, 19, 60)
    runtime.close()


def test_snapshot_acknowledges_command_even_when_midi_advances_past_projected_state():
    now = [0]
    session, _ = make_session()
    runtime = FourChannelGuiRuntime(session, clock_ns=lambda: now[0])
    session.configure_recording_sync(
        start_mode=RecordingStartMode.MIDI_TRANSPORT,
        stop_mode=RecordingStopMode.MIDI_TRANSPORT,
    )
    runtime._commands.put_nowait(GuiIntent(1, 10, GuiIntentKind.RECORDING_TOGGLE))
    projected = {"recording_state": "WAITING_START", "recording_armed": True}
    pending = {"recording_state": 1, "recording_armed": 1}
    _reconcile_projected_state(projected, pending, runtime.get_snapshot())
    assert projected["recording_state"] == "WAITING_START"
    now[0] = 20
    runtime.enqueue_midi_message(mido.Message("start"))
    now[0] = 30
    runtime.enqueue_midi_message(mido.Message("stop"))
    runtime._process_pending()

    snapshot = runtime.get_snapshot()
    assert snapshot.recording_state == "IDLE"
    assert snapshot.last_processed_request_id == 1
    _reconcile_projected_state(projected, pending, snapshot)
    assert not projected
    assert (
        _project_recording_toggle(snapshot.recording_state, snapshot) == "WAITING_START"
    )
    runtime.close()


def test_snapshot_acknowledges_rejected_command_even_if_notification_overflows():
    now = [0]
    session, _ = make_session()
    runtime = FourChannelGuiRuntime(
        session, clock_ns=lambda: now[0], notification_capacity=1
    )
    runtime._commands.put_nowait(
        GuiIntent(1, 0, GuiIntentKind.RECORDING_SYNC, value="start beat")
    )
    runtime._commands.put_nowait(
        GuiIntent(2, 0, GuiIntentKind.RECORDING_SYNC, value="not a command")
    )
    runtime._process_pending()
    snapshot = runtime.get_snapshot()
    assert snapshot.last_processed_request_id == 2
    assert snapshot.notification_overflow_count == 1
    projected: dict[str, object] = {"recording_state": "WAITING_START"}
    pending = {"recording_state": 2}
    _reconcile_projected_state(projected, pending, snapshot)
    assert not projected and not pending
    runtime.close()


def test_projection_waits_for_latest_request_even_if_old_snapshot_matches():
    session, _ = make_session()
    runtime = FourChannelGuiRuntime(session, clock_ns=lambda: 0)
    snapshot = runtime.get_snapshot()
    projected = {
        "recording_state": "IDLE",
        "recording_armed": True,
        "transport_playing": True,
    }
    pending = {"recording_state": 2, "recording_armed": 2, "transport_playing": 1}
    _reconcile_projected_state(projected, pending, snapshot)
    assert projected["recording_state"] == "IDLE"
    runtime._commands.put_nowait(
        GuiIntent(1, 0, GuiIntentKind.TRANSPORT, value=True)
    )
    runtime._process_pending()
    _reconcile_projected_state(projected, pending, runtime.get_snapshot())
    assert projected == {"recording_state": "IDLE", "recording_armed": True}
    assert pending == {"recording_state": 2, "recording_armed": 2}
    runtime._commands.put_nowait(GuiIntent(2, 0, GuiIntentKind.RECORDING_TOGGLE))
    runtime._process_pending()
    _reconcile_projected_state(projected, pending, runtime.get_snapshot())
    assert not projected and not pending
    runtime.close()


@pytest.mark.parametrize(
    ("stop_mode", "expected"),
    [
        ("IMMEDIATE", "IDLE"),
        ("AFTER_DURATION", "IDLE"),
        ("NEXT_GESTURE", "IDLE"),
        ("MIDI_TRANSPORT", "IDLE"),
        ("NEXT_BEAT", "WAITING_STOP"),
        ("NEXT_REFERENCE_LOOP_START", "WAITING_STOP"),
    ],
)
def test_recording_projection_matches_manual_stop_semantics(stop_mode, expected):
    session, _ = make_session()
    runtime = FourChannelGuiRuntime(session, clock_ns=lambda: 0)
    snapshot = replace(runtime.get_snapshot(), recording_stop_mode=stop_mode)
    assert _project_recording_toggle("RECORDING", snapshot) == expected
    runtime.close()


def test_gui_discards_projection_when_automatic_transition_skips_waiting_state(
    monkeypatch,
):
    session, _ = make_session()
    session.configure_recording_sync(start_mode=RecordingStartMode.MIDI_TRANSPORT)
    real_runtime = FourChannelGuiRuntime(session, clock_ns=lambda: 0)
    initial = real_runtime.get_snapshot()
    after_command = replace(initial, last_processed_request_id=1)

    class FakeRuntime:
        lifecycle = "running"
        shutdown_timeout = 1.0
        frame = 0
        next_request_id = 0

        def start(self):
            pass

        def get_snapshot(self):
            self.frame += 1
            return initial if self.frame == 1 else after_command

        def drain_notifications(self):
            return ()

        def submit(self, kind, **kwargs):
            self.next_request_id += 1
            if kind is GuiIntentKind.SHUTDOWN:
                self.lifecycle = "stopped"
            return self.next_request_id

        def close(self, timeout=None):
            pass

    runtime = FakeRuntime()
    enter = SimpleNamespace(type=1, key=13)
    quit_event = SimpleNamespace(type=2)
    events = iter(((enter,), (enter,), (quit_event,)))
    fake_pygame = SimpleNamespace(
        QUIT=2,
        VIDEORESIZE=3,
        KEYDOWN=1,
        K_ESCAPE=27,
        K_q=113,
        K_r=114,
        K_SPACE=32,
        K_TAB=9,
        K_RETURN=13,
        K_KP_ENTER=271,
        K_BACKSPACE=8,
        K_DELETE=127,
        K_d=100,
        RESIZABLE=1,
        init=lambda: None,
        quit=lambda: None,
        display=SimpleNamespace(
            set_mode=lambda *args: object(),
            set_caption=lambda text: None,
            flip=lambda: None,
        ),
        font=SimpleNamespace(SysFont=lambda *args, **kwargs: object()),
        time=SimpleNamespace(Clock=lambda: SimpleNamespace(tick=lambda fps: None)),
        event=SimpleNamespace(get=lambda: next(events)),
    )
    states = []
    original = pygame_gui._project_recording_toggle

    def observe_toggle(state, snapshot):
        states.append(state)
        return original(state, snapshot)

    monkeypatch.setitem(sys.modules, "pygame", fake_pygame)
    monkeypatch.setattr(pygame_gui, "_draw_ui", lambda *args: None)
    monkeypatch.setattr(pygame_gui, "_project_recording_toggle", observe_toggle)
    pygame_gui.run_gui(runtime)  # pyright: ignore[reportArgumentType]

    assert states == ["IDLE", "IDLE"]
    real_runtime.close()


def test_command_backlog_is_processed_before_later_midi_start():
    now = [0]
    session, _ = make_session()
    runtime = FourChannelGuiRuntime(
        session, clock_ns=lambda: now[0], command_capacity=256
    )
    for request_id in range(1, 129):
        runtime._commands.put_nowait(
            GuiIntent(
                request_id,
                10_000_000,
                GuiIntentKind.RECORDING_SYNC,
                value="start transport",
            )
        )
    runtime._commands.put_nowait(
        GuiIntent(129, 10_000_000, GuiIntentKind.RECORDING_TOGGLE)
    )
    now[0] = 20_000_000
    runtime.enqueue_midi_message(mido.Message("start"))
    runtime._process_pending()

    assert session.recording_state.name == "RECORDING"
    runtime.close()


def test_batch_capture_waits_for_inflight_midi_timestamp_and_enqueue():
    timestamp_started = threading.Event()
    release_timestamp = threading.Event()
    processing_done = threading.Event()

    def clock():
        if threading.current_thread().name == "midi-producer":
            timestamp_started.set()
            if not release_timestamp.wait(1):
                raise TimeoutError("MIDI timestamp was not released")
            return 10_000_000
        return 30_000_000

    session, _ = make_session()
    session.toggle_transport()
    session.toggle_recording()
    runtime = FourChannelGuiRuntime(session, clock_ns=clock)
    runtime._clock_origin_ns = 0
    runtime._commands.put_nowait(
        GuiIntent(1, 20_000_000, GuiIntentKind.RECORDING_TOGGLE)
    )
    producer = threading.Thread(
        target=runtime.enqueue_midi_message,
        args=(mido.Message("note_on", note=60, velocity=90),),
        name="midi-producer",
    )

    def process():
        runtime._process_pending()
        processing_done.set()

    worker = threading.Thread(target=process)
    producer.start()
    try:
        assert timestamp_started.wait(1)
        worker.start()
        assert not processing_done.wait(0.05)
    finally:
        release_timestamp.set()
        producer.join(1)
        if worker.ident is not None:
            worker.join(1)
    assert not producer.is_alive()
    assert not worker.is_alive()
    assert processing_done.is_set()
    (interval,) = session.tape(0).intervals()
    assert (interval.start, interval.duration) == (19, 19)
    runtime.close()


def test_arrivals_during_batch_are_not_overtaken_by_clock_advancement(monkeypatch):
    now = [0]
    session, _ = make_session()
    session.toggle_transport()
    session.toggle_recording()
    runtime = FourChannelGuiRuntime(session, clock_ns=lambda: now[0])
    now[0] = 10_000_000
    runtime.enqueue_midi_message(mido.Message("clock"))
    original = session.process_midi_message

    def process_and_enqueue(message, tick, **kwargs):
        original(message, tick, **kwargs)
        if message.type == "clock":
            now[0] = 20_000_000
            runtime.enqueue_midi_message(
                mido.Message("note_on", note=60, velocity=90)
            )
            now[0] = 30_000_000

    monkeypatch.setattr(session, "process_midi_message", process_and_enqueue)
    runtime._process_pending()
    assert session.current_tick == 19
    runtime._process_pending()
    session.toggle_recording()
    (interval,) = session.tape(0).intervals()
    assert (interval.start, interval.duration) == (38, 19)
    runtime.close()


@pytest.mark.parametrize(("tempo", "expected"), [(119.4, 119), (120.6, 121)])
def test_midi_clock_tracks_whole_bpm_after_four_beats(tempo, expected):
    estimate = _MidiClockTempo()
    for index in range(96):
        estimate.observe(round(index * 60_000_000_000 / (24 * tempo)))
    assert estimate.bpm is None
    estimate.observe(round(96 * 60_000_000_000 / (24 * tempo)))
    assert estimate.bpm == expected
    assert isinstance(estimate.bpm, int)


def test_midi_clock_four_beat_average_suppresses_slow_callback_jitter():
    estimate = _MidiClockTempo()
    readings = []
    two_beat_readings = []
    timestamps = []
    for index in range(193):
        timestamp = round(
            index * 500_000_000 / 24
            + 12_000_000 * math.sin(2 * math.pi * index / 96)
        )
        timestamps.append(timestamp)
        estimate.observe(timestamp)
        if index >= 96:
            readings.append(estimate.bpm)
            two_beat_readings.append(
                round(120_000_000_000 / (timestamp - timestamps[index - 48]))
            )
    assert readings == [120] * 97
    assert min(two_beat_readings) < 120 < max(two_beat_readings)


def test_midi_clock_estimate_uses_callback_timestamps_without_changing_transport():
    now = [1_000_000_000]
    session, _ = make_session()
    runtime = FourChannelGuiRuntime(session, clock_ns=lambda: now[0])

    for index in range(97):
        # Alternate callback jitter while keeping the four-beat duration exact.
        now[0] = 1_000_000_000 + round(index * 500_000_000 / 24) + (
            1_000_000 if index % 2 else 0
        )
        runtime.enqueue_midi_message(mido.Message("clock"))
        if index == 95:
            runtime._process_pending()
            assert runtime.get_snapshot().midi_in_bpm is None
    runtime._process_pending()

    snapshot = runtime.get_snapshot()
    assert snapshot.midi_in_bpm == 120
    assert not snapshot.midi_in_no_clock
    assert snapshot.transport.tempo_bpm == 120
    assert not snapshot.transport.playing
    assert session.current_tick == 0
    runtime.close()


def test_midi_clock_loss_retains_lock_and_reacquires_without_counting_gap():
    now = [0]
    session, _ = make_session()
    runtime = FourChannelGuiRuntime(session, clock_ns=lambda: now[0])
    for index in range(97):
        now[0] = round(index * 500_000_000 / 24)
        runtime.enqueue_midi_message(mido.Message("clock"))
    runtime._process_pending()
    assert runtime.get_snapshot().midi_in_bpm == 120

    now[0] += 2_000_000_000
    stopped = runtime._make_snapshot(now[0])
    assert stopped.midi_in_bpm == 120
    assert stopped.midi_in_no_clock

    restart = now[0]
    for index in range(97):
        now[0] = restart + round(index * 400_000_000 / 24)
        runtime.enqueue_midi_message(mido.Message("clock"))
        if index == 0:
            runtime._process_pending()
            assert runtime.get_snapshot().midi_in_bpm == 120
            assert not runtime.get_snapshot().midi_in_no_clock
    runtime._process_pending()
    assert runtime.get_snapshot().midi_in_bpm == 150
    runtime.close()


def test_midi_clock_requires_consistent_beat_windows_before_lock():
    now = [0]
    session, _ = make_session()
    runtime = FourChannelGuiRuntime(session, clock_ns=lambda: now[0])
    for index in range(97):
        now[0] = (
            round(index * 500_000_000 / 24)
            if index <= 48
            else 1_000_000_000 + round((index - 48) * 400_000_000 / 24)
        )
        runtime.enqueue_midi_message(mido.Message("clock"))
    runtime._process_pending()
    assert runtime.get_snapshot().midi_in_bpm is None
    assert not runtime.get_snapshot().midi_in_no_clock
    runtime.close()


def test_non_clock_messages_and_simultaneous_clock_bursts_do_not_create_a_lock():
    now = [0]
    session, _ = make_session()
    runtime = FourChannelGuiRuntime(session, clock_ns=lambda: now[0])
    for _ in range(97):
        runtime.enqueue_midi_message(mido.Message("clock"))
    runtime.enqueue_midi_message(mido.Message("start"))
    runtime.enqueue_midi_message(mido.Message("continue"))
    runtime.enqueue_midi_message(mido.Message("stop"))
    runtime._process_pending()
    assert runtime.get_snapshot().midi_in_bpm is None
    assert not runtime.get_snapshot().midi_in_no_clock

    now[0] += 2_000_000_000
    runtime.enqueue_midi_message(mido.Message("clock"))
    runtime._process_pending()
    assert runtime.get_snapshot().midi_in_bpm is None
    assert not runtime.get_snapshot().midi_in_no_clock
    runtime.close()


def test_midi_clock_follows_tempo_changes_with_internal_transport_running():
    now = [0]
    session, _ = make_session()
    session.toggle_transport()
    runtime = FourChannelGuiRuntime(session, clock_ns=lambda: now[0])
    for beat_duration in (500_000_000, 400_000_000):
        start = now[0]
        for index in range(1, 97):
            now[0] = start + round(index * beat_duration / 24)
            runtime.enqueue_midi_message(mido.Message("clock"))
        runtime._process_pending()
        if beat_duration == 500_000_000:
            assert runtime.get_snapshot().midi_in_bpm is None
    snapshot = runtime.get_snapshot()
    assert snapshot.midi_in_bpm == 150
    assert snapshot.transport.playing
    assert snapshot.transport.tempo_bpm == 120
    assert session.current_tick == 6912
    runtime.close()


def test_midi_clock_loss_is_published_without_further_input():
    now = [0]
    session, _ = make_session()
    runtime = FourChannelGuiRuntime(
        session, clock_ns=lambda: now[0], snapshot_interval=0.05
    )
    for index in range(97):
        now[0] = round(index * 500_000_000 / 24)
        runtime.enqueue_midi_message(mido.Message("clock"))
    runtime.start()
    try:
        deadline = time.monotonic() + 1
        while runtime.get_snapshot().midi_in_bpm is None and time.monotonic() < deadline:
            time.sleep(0.002)
        assert runtime.get_snapshot().midi_in_bpm == 120
        now[0] += 2_000_000_000
        runtime._wake.set()
        deadline = time.monotonic() + 1
        while not runtime.get_snapshot().midi_in_no_clock and time.monotonic() < deadline:
            time.sleep(0.002)
        assert runtime.get_snapshot().midi_in_no_clock
        assert runtime.get_snapshot().midi_in_bpm == 120
    finally:
        runtime.close()


def wait_for_notification(runtime: FourChannelGuiRuntime, kind: str):
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        for notification in runtime.drain_notifications():
            if notification.kind == kind:
                return notification
        time.sleep(0.002)
    raise AssertionError(f"timed out waiting for {kind}")


def test_worker_orders_backlogged_midi_and_active_gesture_stop_command(monkeypatch):
    class ManualClock:
        def __init__(self):
            self.value = 0
            self.lock = threading.Lock()

        def __call__(self):
            with self.lock:
                return self.value

        def set(self, value):
            with self.lock:
                self.value = value

    session, _ = make_session()
    clock = ManualClock()
    entered = threading.Event()
    release = threading.Event()
    original = session.process_midi_message

    def blocked_message(message, tick, **kwargs):
        if not entered.is_set():
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test did not release MIDI worker")
        original(message, tick, **kwargs)

    monkeypatch.setattr(session, "process_midi_message", blocked_message)
    runtime = FourChannelGuiRuntime(session, clock_ns=clock)
    runtime.start()
    try:
        runtime.submit(GuiIntentKind.TRANSPORT, value=True)
        wait_for_notification(runtime, "command_applied")
        runtime.submit(GuiIntentKind.RECORDING_ARM, value=True)
        wait_for_notification(runtime, "command_applied")
        clock.set(1_000_000)
        runtime.enqueue_midi_message(mido.Message("clock"))
        assert entered.wait(1)

        clock.set(10_000_000)
        for _ in range(128):
            runtime.enqueue_midi_message(mido.Message("clock"))
        runtime.enqueue_midi_message(mido.Message("note_on", note=60, velocity=90))
        clock.set(20_000_000)
        request = runtime.submit(GuiIntentKind.RECORDING_SYNC, value="stop gesture")
        clock.set(30_000_000)
        runtime.enqueue_midi_message(mido.Message("note_off", note=60))
        release.set()

        notification = wait_for_notification(runtime, "tape_changed")
        (interval,) = notification.changed_intervals
        assert (interval.start, interval.duration, interval.pitch) == (19, 38, 60)
        deadline = time.monotonic() + 1
        while (
            runtime.get_snapshot().last_processed_request_id < request
            and time.monotonic() < deadline
        ):
            time.sleep(0.002)
        snapshot = runtime.get_snapshot()
        assert snapshot.last_processed_request_id == request
        assert snapshot.recording_state == "IDLE"
        assert snapshot.transport.playing
    finally:
        release.set()
        runtime.close()


def test_gui_reconciles_real_worker_automatic_midi_transitions(monkeypatch):
    session, _ = make_session()
    session.configure_recording_sync(
        start_mode=RecordingStartMode.MIDI_TRANSPORT,
        stop_mode=RecordingStopMode.MIDI_TRANSPORT,
    )
    runtime = FourChannelGuiRuntime(session, clock_ns=lambda: 0)
    enter = SimpleNamespace(type=1, key=13)
    events = iter(((enter,), (enter,), (SimpleNamespace(type=2),)))
    states = []
    original = pygame_gui._project_recording_toggle
    frames = [0]

    def wait_for_state(state):
        deadline = time.monotonic() + 1
        while (
            runtime.get_snapshot().recording_state != state
            and time.monotonic() < deadline
        ):
            time.sleep(0.002)
        assert runtime.get_snapshot().recording_state == state

    def tick(fps):
        frames[0] += 1
        if frames[0] == 1:
            wait_for_state("WAITING_START")
            runtime.enqueue_midi_message(mido.Message("start"))
            wait_for_state("RECORDING")
            runtime.enqueue_midi_message(mido.Message("stop"))
            wait_for_state("IDLE")

    def observe_toggle(state, snapshot):
        states.append(state)
        return original(state, snapshot)

    fake_pygame = SimpleNamespace(
        QUIT=2, VIDEORESIZE=3, KEYDOWN=1, K_ESCAPE=27, K_q=113,
        K_r=114, K_SPACE=32, K_TAB=9, K_RETURN=13, K_KP_ENTER=271,
        K_BACKSPACE=8, K_DELETE=127, K_d=100, RESIZABLE=1,
        init=lambda: None,
        quit=lambda: None,
        display=SimpleNamespace(
            set_mode=lambda *args: object(), set_caption=lambda text: None,
            flip=lambda: None,
        ),
        font=SimpleNamespace(SysFont=lambda *args, **kwargs: object()),
        time=SimpleNamespace(Clock=lambda: SimpleNamespace(tick=tick)),
        event=SimpleNamespace(get=lambda: next(events, ())),
    )
    monkeypatch.setitem(sys.modules, "pygame", fake_pygame)
    monkeypatch.setattr(pygame_gui, "_draw_ui", lambda *args: None)
    monkeypatch.setattr(pygame_gui, "_project_recording_toggle", observe_toggle)
    pygame_gui.run_gui(runtime)
    assert states == ["IDLE", "IDLE"]


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
    session = FourChannelSession(
        (0, 0, 0, 0), 120, (routes[0], routes[1], routes[2], routes[3])
    )
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
    session = FourChannelSession(
        (0, 0, 0, 0), 120, (routes[0], routes[1], routes[2], routes[3])
    )
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


def test_queued_midi_transport_messages_trigger_recording_with_internal_ticks():
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
        for kind, value in (
            (GuiIntentKind.RECORDING_SYNC, "start transport"),
            (GuiIntentKind.RECORDING_SYNC, "stop transport"),
            (GuiIntentKind.TRANSPORT, True),
            (GuiIntentKind.RECORDING_ARM, True),
        ):
            runtime.submit(kind, value=value)
            wait_for_notification(runtime, "command_applied")
        assert runtime.get_snapshot().recording_state == "WAITING_START"

        clock.advance(50_000_000)
        runtime.enqueue_midi_message(mido.Message("start"))
        deadline = time.monotonic() + 1
        while (
            runtime.get_snapshot().recording_state != "RECORDING"
            and time.monotonic() < deadline
        ):
            time.sleep(0.002)
        assert runtime.get_snapshot().recording_state == "RECORDING"

        clock.advance(10_000_000)
        runtime.enqueue_midi_message(mido.Message("note_on", note=60, velocity=90))
        deadline = time.monotonic() + 1
        while (
            runtime.get_snapshot().channels[0].live_pitch != 60
            and time.monotonic() < deadline
        ):
            time.sleep(0.002)
        assert runtime.get_snapshot().channels[0].live_pitch == 60
        clock.advance(40_000_000)
        runtime.enqueue_midi_message(mido.Message("stop"))
        notification = wait_for_notification(runtime, "tape_changed")
        (interval,) = notification.changed_intervals
        assert (interval.start, interval.duration) == (115, 77)
        assert runtime.get_snapshot().recording_state == "IDLE"
        assert runtime.get_snapshot().transport.playing
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


def test_runtime_applies_counted_recording_and_fractional_reset_configuration():
    session, _ = make_session()
    runtime = FourChannelGuiRuntime(session, clock_ns=lambda: 0)
    runtime.start()
    try:
        runtime.submit(
            GuiIntentKind.RECORDING_SYNC,
            value="start beat 3",
        )
        notification = wait_for_notification(runtime, "command_applied")
        assert notification.message == "Start NEXT_BEAT 3; stop IMMEDIATE"
        assert runtime.get_snapshot().recording_start_beat_count == 3

        runtime.submit(
            GuiIntentKind.RESET_CONFIG,
            value="reset beat 3+3/8",
        )
        wait_for_notification(runtime, "command_applied")
        snapshot = runtime.get_snapshot()
        assert snapshot.reset_mode == ResetMode.BEAT.name
        assert snapshot.reset_interval_beats == 3.375
        assert session.reset_interval_beats == 3.375
    finally:
        runtime.close()


def test_runtime_applies_stop_after_and_publishes_duration():
    session, _ = make_session()
    runtime = FourChannelGuiRuntime(session, clock_ns=lambda: 0)
    runtime.start()
    try:
        runtime.submit(
            GuiIntentKind.RECORDING_SYNC,
            value="stop after 4+3/8",
        )
        notification = wait_for_notification(runtime, "command_applied")

        assert notification.message == "Start IMMEDIATE; stop AFTER_DURATION 4.375"
        assert (
            runtime.get_snapshot().recording_stop_after_ticks
            == 4 * 960 + 3 * 960 // 8
        )
    finally:
        runtime.close()


def test_runtime_applies_counted_gesture_sync_and_publishes_snapshot():
    session, _ = make_session()
    runtime = FourChannelGuiRuntime(session, clock_ns=lambda: 0)
    runtime.start()
    try:
        runtime.submit(
            GuiIntentKind.RECORDING_SYNC,
            value="start gesture 3",
        )
        notification = wait_for_notification(runtime, "command_applied")

        assert notification.message == "Start NEXT_GESTURE 3; stop IMMEDIATE"
        assert runtime.get_snapshot().recording_start_gesture_count == 3

        runtime.submit(
            GuiIntentKind.RECORDING_SYNC,
            value="stop gesture",
        )
        notification = wait_for_notification(runtime, "command_applied")

        assert notification.message == "Start NEXT_GESTURE 3; stop NEXT_GESTURE 1"
        snapshot = runtime.get_snapshot()
        assert snapshot.recording_start_gesture_count == 3
        assert snapshot.recording_stop_gesture_count == 1
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
    runtime = FourChannelGuiRuntime(session, clock_ns=lambda: 0)
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


def test_runtime_snapshot_rezeros_beat_position_and_loop_phases_after_resets():
    session, _ = make_session()
    session.toggle_transport()
    session.advance(200)
    session.reset(200)
    snapshot = FourChannelGuiRuntime(session).get_snapshot()
    assert snapshot.transport.position_ticks == 0
    assert tuple(channel.phase_origin_tick for channel in snapshot.channels) == (
        0,
        0,
        0,
        0,
    )

    session.configure_reset(ResetMode.BEAT, interval_ticks=3240)
    session.advance(200 + 3240)
    snapshot = FourChannelGuiRuntime(session).get_snapshot()
    assert snapshot.transport.position_ticks == 0
    assert tuple(channel.phase_origin_tick for channel in snapshot.channels) == (
        0,
        0,
        0,
        0,
    )

    session.advance(200 + 2 * 3240)
    snapshot = FourChannelGuiRuntime(session).get_snapshot()
    assert snapshot.transport.position_ticks == 0
    assert tuple(channel.phase_origin_tick for channel in snapshot.channels) == (
        0,
        0,
        0,
        0,
    )
    session.close()


def test_runtime_loop_length_command_updates_only_its_channel():
    session, _ = make_session()
    runtime = FourChannelGuiRuntime(session, clock_ns=lambda: 0)
    runtime.start()
    try:
        runtime.submit(
            GuiIntentKind.LOOP_LENGTH,
            channel_id="channel-2",
            value="length 3+3/8",
        )
        notification = wait_for_notification(runtime, "command_applied")

        assert notification.message == "Loop length set to 3240 ticks"
        snapshot = runtime.get_snapshot()
        assert tuple(channel.loop_length_ticks for channel in snapshot.channels) == (
            3840,
            3240,
            5760,
            6720,
        )
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
    session = FourChannelSession(
        (3840, 3840, 3840, 3840), 120, (routes[0], routes[1], routes[2], routes[3])
    )
    runtime = FourChannelGuiRuntime(session, output_sinks=(sink,))
    runtime.start()
    runtime.close()

    snapshot = runtime.get_snapshot()
    assert snapshot.lifecycle == "failed"
    assert "MIDI output close: port close failed" in (
        snapshot.terminal_message or ""
    )


def test_runtime_midi_output_defaults_off_and_routes_realtime_once_per_sink():
    sink = CollectingMidiSink()
    routes = (
        MidiRoute("OUT1", 0, sink, 0),
        MidiRoute("OUT2", 1, sink, 0),
        MidiRoute("OUT3", 2, sink, 0),
        MidiRoute("OUT4", 3, sink, 0),
    )
    session = FourChannelSession((3840, 4800, 5760, 6720), 120, routes)
    runtime = FourChannelGuiRuntime(session, clock_ns=lambda: 0)

    snapshot = runtime.get_snapshot()
    assert snapshot.midi_transport_mode == "OFF"
    assert snapshot.midi_clock_mode == "OFF"
    assert not snapshot.midi_transport_active
    assert not snapshot.midi_clock_active

    runtime._apply_intent(
        GuiIntent(1, 0, GuiIntentKind.MIDI_OUTPUT, value="trans on"), 0
    )
    assert sink.messages == [mido.Message("start")]
    assert runtime._make_snapshot(0).midi_transport_mode == "ON"
    assert runtime._make_snapshot(0).midi_transport_active


def test_runtime_auto_transport_emits_starts_for_recurring_resets():
    session, sinks = make_session()
    session.configure_reset(ResetMode.BEAT, interval_ticks=40)
    runtime = FourChannelGuiRuntime(session, clock_ns=lambda: 0)

    runtime._apply_intent(
        GuiIntent(1, 0, GuiIntentKind.TRANSPORT, value=True), 0
    )
    runtime._apply_intent(
        GuiIntent(2, 0, GuiIntentKind.MIDI_OUTPUT, value="transport auto"), 0
    )
    assert runtime._make_snapshot(0).midi_transport_start_pending

    runtime._advance_to(120)
    assert sinks[0].messages == [mido.Message("start")] * 3
    assert not runtime._make_snapshot(0).midi_transport_start_pending

    runtime._advance_to(121)
    runtime._apply_intent(GuiIntent(3, 0, GuiIntentKind.RESET), 121)
    assert sinks[0].messages == [mido.Message("start")] * 4

    runtime._apply_intent(
        GuiIntent(4, 0, GuiIntentKind.TRANSPORT, value=False), 121
    )
    assert sinks[0].messages[-1] == mido.Message("stop")


def test_runtime_auto_transport_does_not_start_for_a_stopped_reset():
    session, sinks = make_session()
    runtime = FourChannelGuiRuntime(session, clock_ns=lambda: 0)
    runtime._apply_intent(
        GuiIntent(1, 0, GuiIntentKind.MIDI_OUTPUT, value="transport auto"), 0
    )
    runtime._apply_intent(GuiIntent(2, 0, GuiIntentKind.RESET), 0)

    assert sinks[0].messages == [mido.Message("stop")]


def test_runtime_sends_due_clock_before_note_on_at_shared_ppqn_boundary():
    now = [0]
    session, sinks = make_session()
    session.tape(0).insert(IntervalInsert(40, 20, 60, 90, 1))
    runtime = FourChannelGuiRuntime(session, clock_ns=lambda: now[0])

    runtime._apply_intent(
        GuiIntent(1, 0, GuiIntentKind.TRANSPORT, value=True), 0
    )
    runtime._apply_intent(
        GuiIntent(2, 0, GuiIntentKind.MIDI_OUTPUT, value="clock on"), 0
    )
    now[0] = round(60_000_000_000 / (24 * session.bpm))
    runtime._process_pending()

    assert sinks[0].messages == [
        mido.Message("clock"),
        mido.Message("note_on", channel=0, note=60, velocity=90),
    ]
