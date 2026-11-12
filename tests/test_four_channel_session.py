import mido
import pytest

from interval_resolver import RecordMode
from midi_output import CollectingMidiSink
from py_seq.four_channel_session import (
    FourChannelSession,
    MidiRoute,
    RecordingStartMode,
    RecordingState,
    RecordingStopMode,
    ResetMode,
    TICKS_PER_BEAT,
)
from tape import IntervalInsert


def make_session(
    lengths: tuple[int, int, int, int] | None = None,
) -> tuple[
    FourChannelSession,
    tuple[CollectingMidiSink, CollectingMidiSink, CollectingMidiSink, CollectingMidiSink],
]:
    sinks = (
        CollectingMidiSink(),
        CollectingMidiSink(),
        CollectingMidiSink(),
        CollectingMidiSink(),
    )
    routes = (
        MidiRoute("PORT1", 0, sinks[0], 0),
        MidiRoute("PORT2", 1, sinks[1], 1),
        MidiRoute("PORT3", 2, sinks[2], 2),
        MidiRoute("PORT4", 3, sinks[3], 3),
    )
    session = FourChannelSession(
        lengths
        if lengths is not None
        else (
            16 * TICKS_PER_BEAT,
            16 * TICKS_PER_BEAT,
            16 * TICKS_PER_BEAT,
            16 * TICKS_PER_BEAT,
        ),
        120,
        routes,
    )
    return session, sinks


def test_midi_transport_starts_and_commits_recording_at_message_ticks():
    session, _ = make_session((0, 960, 960, 960))
    session.configure_recording_sync(
        start_mode=RecordingStartMode.MIDI_TRANSPORT,
        stop_mode=RecordingStopMode.MIDI_TRANSPORT,
    )
    session.toggle_transport()
    session.advance(20)
    session.toggle_recording()
    session.process_midi_message(mido.Message("note_on", note=60, velocity=90), 30)
    session.process_midi_message(mido.Message("note_off", note=60), 40)
    session.process_midi_message(mido.Message.from_bytes([0xFA]), 100)
    assert session.recording_state is RecordingState.RECORDING
    assert session.phase_origin_tick(0) == 100
    session.process_midi_message(mido.Message("note_on", note=62, velocity=90), 120)
    session.process_midi_message(mido.Message("start"), 130)
    session.process_midi_message(mido.Message.from_bytes([0xFC]), 160)

    assert session.recording_state is RecordingState.IDLE
    assert session.transport_running
    assert session.tape(0).length_ticks == 60
    (interval,) = session.tape(0).intervals()
    assert (interval.start, interval.duration, interval.pitch) == (20, 40, 62)
    session.close()


def test_midi_start_works_while_playback_is_stopped_without_starting_playback():
    session, _ = make_session()
    session.configure_recording_sync(start_mode=RecordingStartMode.MIDI_TRANSPORT)
    session.toggle_recording()
    session.process_midi_message(mido.Message("start"), 0)
    assert session.recording_state is RecordingState.RECORDING
    assert not session.transport_running
    session.toggle_transport()
    assert session.recording_state is RecordingState.RECORDING
    session.close()


def test_midi_stop_while_playback_is_stopped_does_not_start_playback():
    session, _ = make_session()
    session.configure_recording_sync(stop_mode=RecordingStopMode.MIDI_TRANSPORT)
    session.toggle_recording()
    session.process_midi_message(mido.Message("stop"), 0)
    assert session.recording_state is RecordingState.IDLE
    assert not session.transport_running
    session.close()


def test_midi_start_and_beat_stop_are_independently_configurable():
    session, _ = make_session()
    session.configure_recording_sync(
        start_mode=RecordingStartMode.MIDI_TRANSPORT,
        stop_mode=RecordingStopMode.NEXT_BEAT,
    )
    session.toggle_transport()
    session.toggle_recording()
    session.process_midi_message(mido.Message("start"), 10)
    session.toggle_recording()
    session.process_midi_message(mido.Message("stop"), 20)
    assert session.recording_state is RecordingState.WAITING_STOP
    session.advance(TICKS_PER_BEAT)
    assert session.recording_state is RecordingState.IDLE
    session.close()


def test_beat_start_and_midi_stop_are_independently_configurable():
    session, _ = make_session()
    session.configure_recording_sync(
        start_mode=RecordingStartMode.NEXT_BEAT,
        stop_mode=RecordingStopMode.MIDI_TRANSPORT,
    )
    session.toggle_transport()
    session.toggle_recording()
    session.process_midi_message(mido.Message("start"), 10)
    assert session.recording_state is RecordingState.WAITING_START
    session.advance(TICKS_PER_BEAT)
    assert session.recording_state is RecordingState.RECORDING
    session.process_midi_message(mido.Message("stop"), TICKS_PER_BEAT + 10)
    assert session.recording_state is RecordingState.IDLE
    session.close()


def test_midi_start_waiting_across_playback_restart_and_enter_cancels():
    session, _ = make_session()
    session.configure_recording_sync(start_mode=RecordingStartMode.MIDI_TRANSPORT)
    session.toggle_recording()
    session.toggle_transport()
    assert session.recording_state is RecordingState.WAITING_START
    session.toggle_transport()
    session.toggle_transport()
    assert session.recording_state is RecordingState.WAITING_START
    session.toggle_recording()
    session.process_midi_message(mido.Message("start"), 0)
    assert session.recording_state is RecordingState.IDLE
    assert not session.record_armed
    session.close()


def test_enter_commits_midi_stop_recording_immediately():
    session, _ = make_session((0, 960, 960, 960))
    session.configure_recording_sync(stop_mode=RecordingStopMode.MIDI_TRANSPORT)
    session.toggle_transport()
    session.toggle_recording()
    session.process_midi_message(mido.Message("note_on", note=60, velocity=90), 10)
    session.advance(50)
    session.toggle_recording()
    assert session.recording_state is RecordingState.IDLE
    assert session.tape(0).length_ticks == 50
    (interval,) = session.tape(0).intervals()
    assert (interval.start, interval.duration) == (10, 40)
    session.close()


@pytest.mark.parametrize("status", [0xF8, 0xFB, 0xFC])
def test_other_realtime_messages_do_not_start_transport_recording(status):
    session, _ = make_session()
    session.configure_recording_sync(
        start_mode=RecordingStartMode.MIDI_TRANSPORT,
        stop_mode=RecordingStopMode.MIDI_TRANSPORT,
    )
    session.toggle_transport()
    session.toggle_recording()
    session.process_midi_message(mido.Message.from_bytes([status]), 10)
    assert session.recording_state is RecordingState.WAITING_START
    session.process_midi_message(mido.Message("start"), 20)
    if status != 0xFC:
        session.process_midi_message(mido.Message.from_bytes([status]), 30)
        assert session.recording_state is RecordingState.RECORDING
    session.close()


@pytest.mark.parametrize("stop_mode", [RecordingStopMode.IMMEDIATE, RecordingStopMode.NEXT_BEAT])
def test_midi_stop_does_not_override_other_stop_modes(stop_mode):
    session, _ = make_session()
    session.configure_recording_sync(stop_mode=stop_mode)
    session.toggle_transport()
    session.toggle_recording()
    session.process_midi_message(mido.Message("stop"), 10)
    assert session.recording_state is RecordingState.RECORDING
    if stop_mode is RecordingStopMode.NEXT_BEAT:
        session.toggle_recording()
        session.process_midi_message(mido.Message("stop"), 20)
        assert session.recording_state is RecordingState.WAITING_STOP
    assert session.transport_running
    session.close()


def test_midi_start_does_not_override_another_start_mode():
    session, _ = make_session()
    session.configure_recording_sync(start_mode=RecordingStartMode.NEXT_GESTURE)
    session.toggle_transport()
    session.toggle_recording()
    session.process_midi_message(mido.Message("start"), 10)
    assert session.recording_state is RecordingState.WAITING_START
    session.process_midi_message(mido.Message("note_on", note=60, velocity=90), 20)
    assert session.recording_state is RecordingState.RECORDING
    session.close()


def test_midi_stop_respects_stop_mode_changed_during_recording():
    session, _ = make_session()
    session.toggle_transport()
    session.toggle_recording()
    session.configure_recording_sync(stop_mode=RecordingStopMode.MIDI_TRANSPORT)
    session.process_midi_message(mido.Message("stop"), 10)
    assert session.recording_state is RecordingState.IDLE

    session.toggle_recording()
    session.configure_recording_sync(stop_mode=RecordingStopMode.NEXT_GESTURE)
    session.process_midi_message(mido.Message("stop"), 20)
    assert session.recording_state is RecordingState.RECORDING
    session.close()


def test_clearing_reference_does_not_cancel_transport_start():
    session, _ = make_session((960, 960, 960, 960))
    session.configure_recording_sync(
        start_mode=RecordingStartMode.MIDI_TRANSPORT, reference_channel=1
    )
    session.toggle_transport()
    session.toggle_recording()
    session.clear_tape(1)
    assert session.recording_state is RecordingState.WAITING_START
    session.process_midi_message(mido.Message("start"), 10)
    assert session.recording_state is RecordingState.RECORDING
    session.close()


def test_four_channel_coordinator_runs_all_loops_on_shared_transport():
    session, sinks = make_session((960, 1920, 2880, 3840))
    for index in range(4):
        session.tape(index).insert(IntervalInsert(0, 240, 60 + index, 90, index + 1))

    session.toggle_transport()
    session.advance(240)

    for index, sink in enumerate(sinks):
        assert [
            (getattr(message, "type"), getattr(message, "note"))
            for message in sink.messages
        ] == [
            ("note_on", 60 + index),
            ("note_off", 60 + index),
        ]
    assert tuple(scheduler.current_tick for scheduler in session.schedulers) == (
        240,
        240,
        240,
        240,
    )
    session.close()

def test_zero_length_loop_is_defined_by_first_recording_pass_relative_to_arm():
    session, _ = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.advance(100)
    session.set_recording_armed(True)
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=64, velocity=100), 110
    )
    session.process_midi_message(
        mido.Message("note_off", channel=0, note=64, velocity=0), 130
    )
    session.advance(160)
    session.set_recording_armed(False)

    interval, = session.tape(0).intervals()
    assert session.tape(0).length_ticks == 60
    assert (interval.start, interval.duration, interval.pitch) == (10, 20, 64)
    assert session.phase_origin_tick(0) == 100
    session.close()

def test_zero_length_recording_finalizes_on_stop_without_extending_after_restart():
    session, _ = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.set_recording_armed(True)
    session.advance(120)
    session.toggle_transport()

    assert session.tape(0).length_ticks == 120
    assert session.record_armed

    session.toggle_transport()
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=67, velocity=90), 125
    )
    session.process_midi_message(
        mido.Message("note_off", channel=0, note=67, velocity=0), 135
    )
    session.advance(200)
    assert session.tape(0).length_ticks == 120
    assert [
        (interval.start, interval.duration, interval.pitch)
        for interval in session.tape(0).intervals()
    ] == [(5, 10, 67)]
    session.close()

def test_zero_tick_recording_pass_does_not_define_loop():
    session, _ = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.set_recording_armed(True)
    session.set_recording_armed(False)

    assert session.tape(0).length_ticks == 0
    session.close()

def test_channel_change_finalizes_old_pass_and_starts_new_zero_loop_pass():
    session, _ = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.set_recording_armed(True)
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=60, velocity=90), 10
    )
    session.advance(100)
    session.select_channel(1)
    session.advance(150)

    assert session.tape(0).length_ticks == 100
    interval, = session.tape(0).intervals()
    assert (interval.start, interval.duration, interval.pitch) == (10, 90, 60)
    assert session.tape(1).length_ticks == 50
    assert session.phase_origin_tick(0) == 0
    assert session.phase_origin_tick(1) == 100
    session.close()

def test_clear_nonempty_loop_retains_length_then_empty_loop_resets_length():
    session, _ = make_session((100, 0, 0, 0))
    session.tape(0).insert(IntervalInsert(10, 20, 60, 90, 1))
    session.toggle_transport()

    session.clear_active_tape()
    assert session.tape(0).intervals() == ()
    assert session.tape(0).length_ticks == 100

    session.clear_tape(0)
    assert session.tape(0).length_ticks == 0
    assert session.schedulers[0].running
    session.close()


def test_set_loop_length_resizes_only_empty_channel_and_preserves_phase_origin():
    session, _ = make_session((100, 200, 300, 400))
    session.reset(50)
    session.advance(100)

    session.set_loop_length(0, 3900)

    assert session.tape(0).length_ticks == 3900
    assert session.tape(0).revision == 1
    assert tuple(session.tape(index).length_ticks for index in range(1, 4)) == (
        200,
        300,
        400,
    )
    assert session.current_tick == 100
    assert session.phase_origin_tick(0) == 50
    assert session.recorder(0).phase_origin_tick == 50
    assert session.schedulers[0].phase_origin_tick == 50
    assert tuple(session.phase_origin_tick(index) for index in range(1, 4)) == (
        50,
        50,
        50,
    )
    session.close()


def test_set_loop_length_rejects_unsupported_changes_without_mutation():
    occupied, _ = make_session((100, 0, 0, 0))
    occupied.tape(0).insert(IntervalInsert(10, 20, 60, 90, 1))
    with pytest.raises(ValueError, match="empty channel"):
        occupied.set_loop_length(0, 200)
    assert occupied.tape(0).length_ticks == 100
    assert occupied.phase_origin_tick(0) == 0
    occupied.close()

    running, _ = make_session((0, 0, 0, 0))
    running.toggle_transport()
    with pytest.raises(RuntimeError, match="transport is stopped"):
        running.set_loop_length(0, 200)
    assert running.tape(0).length_ticks == 0
    running.close()

    recording, _ = make_session((0, 0, 0, 0))
    recording.set_recording_armed(True)
    with pytest.raises(RuntimeError, match="recording"):
        recording.set_loop_length(0, 200)
    assert recording.tape(0).length_ticks == 0
    recording.set_recording_armed(False)
    recording.configure_recording_sync(start_mode=RecordingStartMode.NEXT_BEAT)
    recording.set_recording_armed(True)
    assert recording.recording_state is RecordingState.WAITING_START
    with pytest.raises(RuntimeError, match="recording"):
        recording.set_loop_length(0, 200)
    assert recording.tape(0).length_ticks == 0
    recording.close()


def test_synchronized_start_excludes_waiting_time_and_captures_held_voice_at_zero():
    session, sinks = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.advance(200)
    session.configure_recording_sync(start_mode=RecordingStartMode.NEXT_BEAT)
    session.set_recording_armed(True)
    assert session.recording_state is RecordingState.WAITING_START
    assert not session.recorder(0).record_armed

    session.process_midi_message(
        mido.Message("note_on", channel=0, note=64, velocity=100), 500
    )
    session.advance(TICKS_PER_BEAT)
    assert session.recording_state is RecordingState.RECORDING
    session.advance(TICKS_PER_BEAT + 40)
    session.process_midi_message(
        mido.Message("note_off", channel=0, note=64, velocity=0),
        TICKS_PER_BEAT + 40,
    )
    assert sum(getattr(message, "type") == "note_on" for message in sinks[0].messages) == 1
    session.set_recording_armed(False)

    interval, = session.tape(0).intervals()
    assert session.tape(0).length_ticks == 40
    assert (interval.start, interval.duration, interval.pitch) == (0, 40, 64)
    session.close()

def test_waiting_start_does_not_apply_overwrite_silence_early():
    session, _ = make_session((1000, 1000, 0, 0))
    original = session.tape(0).insert(IntervalInsert(0, 800, 60, 90, 1))
    session.toggle_transport()
    session.set_recording_armed(True)
    session.toggle_mode()
    session.set_recording_armed(False)
    session.advance(100)
    session.configure_recording_sync(start_mode=RecordingStartMode.NEXT_BEAT)
    session.set_recording_armed(True)
    session.advance(500)

    assert session.recording_state is RecordingState.WAITING_START
    assert session.tape(0).intervals() == (original,)

    session.toggle_recording()
    session.close()

def test_gesture_start_records_triggering_gesture_once():
    session, sinks = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.configure_recording_sync(
        start_mode=RecordingStartMode.NEXT_GESTURE
    )
    session.set_recording_armed(True)
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=67, velocity=90), 100
    )
    session.process_midi_message(
        mido.Message("note_off", channel=0, note=67, velocity=0), 140
    )
    assert sum(getattr(message, "type") == "note_on" for message in sinks[0].messages) == 1
    session.set_recording_armed(False)

    interval, = session.tape(0).intervals()
    assert session.tape(0).length_ticks == 40
    assert (interval.start, interval.duration, interval.pitch) == (0, 40, 67)
    session.close()


@pytest.mark.parametrize("gesture_count", (1, 2, 4))
def test_gesture_start_accepts_first_second_and_nth_gesture(gesture_count):
    session, _ = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.configure_recording_sync(
        start_mode=RecordingStartMode.NEXT_GESTURE,
        start_gesture_count=gesture_count,
    )
    session.set_recording_armed(True)

    for gesture_index in range(gesture_count):
        start_tick = 10 + gesture_index * 20
        session.process_midi_message(
            mido.Message(
                "note_on", channel=0, note=60 + gesture_index, velocity=90
            ),
            start_tick,
        )
        if gesture_index < gesture_count - 1:
            assert session.recording_state is RecordingState.WAITING_START
        else:
            assert session.recording_state is RecordingState.RECORDING
        session.process_midi_message(
            mido.Message(
                "note_off", channel=0, note=60 + gesture_index, velocity=0
            ),
            start_tick + 10,
        )

    session.set_recording_armed(False)
    interval, = session.tape(0).intervals()
    assert session.phase_origin_tick(0) == 10 + (gesture_count - 1) * 20
    assert (interval.start, interval.duration) == (0, 10)
    session.close()


def test_counted_gesture_start_includes_trigger_and_ignores_voice_changes():
    session, _ = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.configure_recording_sync(
        start_mode=RecordingStartMode.NEXT_GESTURE,
        start_gesture_count=2,
    )
    session.set_recording_armed(True)

    session.process_midi_message(
        mido.Message("note_on", channel=0, note=60, velocity=90), 10
    )
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=62, velocity=90), 15
    )
    assert session.recording_state is RecordingState.WAITING_START
    session.process_midi_message(
        mido.Message("note_off", channel=0, note=60, velocity=0), 20
    )
    session.process_midi_message(
        mido.Message("note_off", channel=0, note=62, velocity=0), 25
    )
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=64, velocity=90), 30
    )
    assert session.recording_state is RecordingState.RECORDING
    session.process_midi_message(
        mido.Message("note_off", channel=0, note=64, velocity=0), 50
    )
    session.set_recording_armed(False)

    interval, = session.tape(0).intervals()
    assert session.phase_origin_tick(0) == 30
    assert session.tape(0).length_ticks == 20
    assert (interval.start, interval.duration, interval.pitch) == (0, 20, 64)
    session.close()


def test_gesture_start_ignores_gesture_already_active_when_armed():
    session, _ = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=60, velocity=90), 10
    )
    session.configure_recording_sync(
        start_mode=RecordingStartMode.NEXT_GESTURE
    )
    session.set_recording_armed(True)
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=62, velocity=90), 15
    )
    assert session.recording_state is RecordingState.WAITING_START

    session.process_midi_message(
        mido.Message("note_off", channel=0, note=60, velocity=0), 20
    )
    session.process_midi_message(
        mido.Message("note_off", channel=0, note=62, velocity=0), 25
    )
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=64, velocity=90), 30
    )
    assert session.recording_state is RecordingState.RECORDING
    session.set_recording_armed(False)
    session.close()


def test_counted_gesture_stop_automatically_stops_on_nth_voice_end():
    session, sinks = make_session((0, 1000, 0, 0))
    session.toggle_transport()
    session.configure_recording_sync(
        stop_mode=RecordingStopMode.NEXT_GESTURE,
        stop_gesture_count=2,
    )
    session.set_recording_armed(True)
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=60, velocity=90), 10
    )
    session.process_midi_message(
        mido.Message("note_off", channel=0, note=60, velocity=0), 20
    )
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=62, velocity=90), 30
    )
    assert session.recording_state is RecordingState.RECORDING
    session.process_midi_message(
        mido.Message("note_off", channel=0, note=62, velocity=0), 40
    )

    assert session.recording_state is RecordingState.IDLE
    assert tuple(
        (interval.start, interval.duration, interval.pitch)
        for interval in session.tape(0).intervals()
    ) == ((10, 10, 60), (30, 10, 62))
    assert session.tape(0).length_ticks == 40
    assert not session.record_armed
    assert [
        (getattr(message, "type"), getattr(message, "note"))
        for message in sinks[0].messages
        if getattr(message, "type") in ("note_on", "note_off")
    ] == [
        ("note_on", 60),
        ("note_off", 60),
        ("note_on", 62),
        ("note_off", 62),
    ]
    session.close()


def test_configuring_gesture_stop_during_recording_counts_held_gesture_end():
    session, _ = make_session((0, 1000, 0, 0))
    session.toggle_transport()
    session.toggle_recording()
    session.process_midi_message(mido.Message("note_on", note=60, velocity=90), 10)
    session.configure_recording_sync(stop_mode=RecordingStopMode.NEXT_GESTURE)
    session.process_midi_message(mido.Message("note_off", note=60), 20)
    assert session.recording_state is RecordingState.IDLE
    (interval,) = session.tape(0).intervals()
    assert (interval.start, interval.duration) == (10, 10)
    session.close()


def test_reconfiguring_gesture_count_restarts_counter_but_unrelated_changes_do_not():
    session, _ = make_session()
    session.configure_recording_sync(
        stop_mode=RecordingStopMode.NEXT_GESTURE, stop_gesture_count=3
    )
    session.toggle_transport()
    session.toggle_recording()
    session.process_midi_message(mido.Message("note_on", note=60, velocity=90), 10)
    session.process_midi_message(mido.Message("note_off", note=60), 20)
    session.configure_recording_sync(reference_channel=1)
    session.process_midi_message(mido.Message("note_on", note=62, velocity=90), 30)
    session.process_midi_message(mido.Message("note_off", note=62), 40)
    assert session.recording_state is RecordingState.RECORDING
    session.configure_recording_sync(stop_gesture_count=2, stop_gesture_cycle=True)
    session.process_midi_message(mido.Message("note_on", note=64, velocity=90), 50)
    session.process_midi_message(mido.Message("note_off", note=64), 60)
    assert session.recording_state is RecordingState.RECORDING
    session.process_midi_message(mido.Message("note_on", note=65, velocity=90), 70)
    session.process_midi_message(mido.Message("note_off", note=65), 80)
    assert session.recording_state is RecordingState.WAITING_STOP
    session.process_midi_message(mido.Message("note_on", note=67, velocity=90), 90)
    assert session.recording_state is RecordingState.IDLE
    session.close()


def test_unrelated_reconfiguration_preserves_gesture_stop_progress():
    session, _ = make_session()
    session.configure_recording_sync(
        stop_mode=RecordingStopMode.NEXT_GESTURE, stop_gesture_count=2
    )
    session.toggle_transport()
    session.toggle_recording()
    session.process_midi_message(mido.Message("note_on", note=60, velocity=90), 10)
    session.process_midi_message(mido.Message("note_off", note=60), 20)
    session.configure_recording_sync(reference_channel=1, stop_gesture_cycle=True)
    session.process_midi_message(mido.Message("note_on", note=62, velocity=90), 30)
    session.process_midi_message(mido.Message("note_off", note=62), 40)
    assert session.recording_state is RecordingState.WAITING_STOP
    session.process_midi_message(mido.Message("note_on", note=64, velocity=90), 50)
    assert session.recording_state is RecordingState.IDLE
    session.close()


def test_switching_away_and_back_to_gesture_stop_starts_a_fresh_count():
    session, _ = make_session()
    session.configure_recording_sync(
        stop_mode=RecordingStopMode.NEXT_GESTURE, stop_gesture_count=2
    )
    session.toggle_transport()
    session.toggle_recording()
    session.process_midi_message(mido.Message("note_on", note=60, velocity=90), 10)
    session.process_midi_message(mido.Message("note_off", note=60), 20)
    session.configure_recording_sync(stop_mode=RecordingStopMode.IMMEDIATE)
    session.configure_recording_sync(
        stop_mode=RecordingStopMode.NEXT_GESTURE, stop_gesture_count=2
    )
    session.process_midi_message(mido.Message("note_on", note=62, velocity=90), 30)
    session.process_midi_message(mido.Message("note_off", note=62), 40)
    assert session.recording_state is RecordingState.RECORDING
    session.process_midi_message(mido.Message("note_on", note=64, velocity=90), 50)
    session.process_midi_message(mido.Message("note_off", note=64), 60)
    assert session.recording_state is RecordingState.IDLE
    session.close()


def test_cycle_gesture_stop_commits_at_next_start_and_keeps_trigger_live():
    session, sinks = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.configure_recording_sync(
        stop_mode=RecordingStopMode.NEXT_GESTURE,
        stop_gesture_count=1,
        stop_gesture_cycle=True,
    )
    session.set_recording_armed(True)
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=60, velocity=90), 10
    )
    session.process_midi_message(
        mido.Message("note_off", channel=0, note=60, velocity=0), 20
    )

    assert session.recording_state is RecordingState.WAITING_STOP
    assert session.record_armed
    session.advance(70)
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=67, velocity=100), 80
    )

    assert session.recording_state is RecordingState.IDLE
    assert not session.record_armed
    assert session.tape(0).length_ticks == 80
    assert tuple(
        (interval.start, interval.duration, interval.pitch)
        for interval in session.tape(0).intervals()
    ) == ((10, 10, 60),)
    session.process_midi_message(
        mido.Message("note_off", channel=0, note=67, velocity=0), 90
    )
    assert [
        (getattr(message, "type"), getattr(message, "note"))
        for message in sinks[0].messages
        if getattr(message, "type") in ("note_on", "note_off")
    ] == [
        ("note_on", 60),
        ("note_off", 60),
        ("note_on", 67),
        ("note_off", 67),
    ]
    assert tuple(
        (interval.start, interval.duration, interval.pitch)
        for interval in session.tape(0).intervals()
    ) == ((10, 10, 60),)
    session.close()


def test_enter_commits_cycle_wait_at_current_tick():
    session, _ = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.configure_recording_sync(
        stop_mode=RecordingStopMode.NEXT_GESTURE,
        stop_gesture_count=1,
        stop_gesture_cycle=True,
    )
    session.set_recording_armed(True)
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=60, velocity=90), 10
    )
    session.process_midi_message(
        mido.Message("note_off", channel=0, note=60, velocity=0), 20
    )
    session.advance(70)

    session.toggle_recording()

    assert session.recording_state is RecordingState.IDLE
    assert not session.record_armed
    assert session.tape(0).length_ticks == 70
    assert tuple(
        (interval.start, interval.duration, interval.pitch)
        for interval in session.tape(0).intervals()
    ) == ((10, 10, 60),)
    session.close()


def test_reference_tape_clear_does_not_cancel_cycle_wait():
    session, _ = make_session((0, 1000, 0, 0))
    session.toggle_transport()
    session.configure_recording_sync(
        stop_mode=RecordingStopMode.NEXT_GESTURE,
        stop_gesture_cycle=True,
        reference_channel=1,
    )
    session.set_recording_armed(True)
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=60, velocity=90), 10
    )
    session.process_midi_message(
        mido.Message("note_off", channel=0, note=60, velocity=0), 20
    )
    assert session.recording_state is RecordingState.WAITING_STOP

    session.clear_tape(1)

    assert session.recording_state is RecordingState.WAITING_STOP
    assert session.record_armed
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=67, velocity=90), 30
    )
    assert session.recording_state is RecordingState.IDLE
    session.close()


def test_enter_commits_cycle_recording_immediately_before_gesture_end():
    session, _ = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.configure_recording_sync(
        stop_mode=RecordingStopMode.NEXT_GESTURE,
        stop_gesture_count=2,
        stop_gesture_cycle=True,
    )
    session.set_recording_armed(True)
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=60, velocity=90), 10
    )

    session.advance(15)
    session.toggle_recording()

    assert session.recording_state is RecordingState.IDLE
    assert not session.record_armed
    assert session.tape(0).length_ticks == 15
    interval, = session.tape(0).intervals()
    assert (interval.start, interval.duration, interval.pitch) == (10, 5, 60)
    session.close()


@pytest.mark.parametrize("gesture_count", (1, 2, 4))
def test_gesture_stop_counts_complete_gestures(gesture_count):
    session, _ = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.configure_recording_sync(
        stop_mode=RecordingStopMode.NEXT_GESTURE,
        stop_gesture_count=gesture_count,
    )
    session.set_recording_armed(True)

    for gesture_index in range(gesture_count):
        start_tick = 10 + gesture_index * 20
        session.process_midi_message(
            mido.Message(
                "note_on", channel=0, note=60 + gesture_index, velocity=90
            ),
            start_tick,
        )
        session.process_midi_message(
            mido.Message(
                "note_off", channel=0, note=60 + gesture_index, velocity=0
            ),
            start_tick + 10,
        )
        expected_state = (
            RecordingState.IDLE
            if gesture_index == gesture_count - 1
            else RecordingState.RECORDING
        )
        assert session.recording_state is expected_state
    session.close()


def test_gesture_active_before_recording_activation_counts_on_voice_end():
    session, _ = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.configure_recording_sync(
        stop_mode=RecordingStopMode.NEXT_GESTURE
    )
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=60, velocity=90), 10
    )
    session.set_recording_armed(True)
    session.process_midi_message(
        mido.Message("note_off", channel=0, note=60, velocity=0), 20
    )

    assert session.recording_state is RecordingState.IDLE
    session.close()


def test_immediate_gesture_stop_counts_voice_ends_while_transport_is_stopped():
    session, _ = make_session((0, 0, 0, 0))
    session.configure_recording_sync(
        stop_mode=RecordingStopMode.NEXT_GESTURE,
        stop_gesture_count=2,
    )
    session.set_recording_armed(True)

    for gesture_index in range(2):
        start_tick = 10 + gesture_index * 20
        session.process_midi_message(
            mido.Message(
                "note_on", channel=0, note=60 + gesture_index, velocity=90
            ),
            start_tick,
        )
        session.process_midi_message(
            mido.Message(
                "note_off", channel=0, note=60 + gesture_index, velocity=0
            ),
            start_tick + 10,
        )

    assert session.recording_state is RecordingState.IDLE
    assert not session.record_armed
    session.close()


def test_start_trigger_gesture_counts_toward_gesture_stop():
    session, _ = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.configure_recording_sync(
        start_mode=RecordingStartMode.NEXT_GESTURE,
        stop_mode=RecordingStopMode.NEXT_GESTURE,
        stop_gesture_count=2,
    )
    session.set_recording_armed(True)
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=60, velocity=90), 10
    )
    assert session.recording_state is RecordingState.RECORDING

    session.process_midi_message(
        mido.Message("note_off", channel=0, note=60, velocity=0), 20
    )
    assert session.recording_state is RecordingState.RECORDING
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=62, velocity=90), 30
    )
    session.process_midi_message(
        mido.Message("note_off", channel=0, note=62, velocity=0), 40
    )

    assert session.recording_state is RecordingState.IDLE
    assert tuple(
        (interval.start, interval.duration, interval.pitch)
        for interval in session.tape(0).intervals()
    ) == ((0, 10, 60), (20, 10, 62))
    session.close()


def test_start_trigger_counts_toward_gesture_stop_without_manual_request():
    session, _ = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.configure_recording_sync(
        start_mode=RecordingStartMode.NEXT_GESTURE,
        stop_mode=RecordingStopMode.NEXT_GESTURE,
        stop_gesture_count=1,
    )
    session.set_recording_armed(True)
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=60, velocity=90), 10
    )
    session.process_midi_message(
        mido.Message("note_off", channel=0, note=60, velocity=0), 20
    )

    assert session.recording_state is RecordingState.IDLE
    session.close()


def test_gesture_stop_counts_voice_end_once_after_voice_changes():
    session, _ = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.configure_recording_sync(
        stop_mode=RecordingStopMode.NEXT_GESTURE
    )
    session.set_recording_armed(True)

    session.process_midi_message(
        mido.Message("note_on", channel=0, note=60, velocity=90), 10
    )
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=62, velocity=90), 15
    )
    session.process_midi_message(
        mido.Message("note_off", channel=0, note=62, velocity=0), 20
    )
    assert session.recording_state is RecordingState.RECORDING
    session.process_midi_message(
        mido.Message("note_off", channel=0, note=60, velocity=0), 30
    )

    assert session.recording_state is RecordingState.IDLE
    assert tuple(
        (interval.start, interval.duration, interval.pitch)
        for interval in session.tape(0).intervals()
    ) == ((10, 5, 60), (15, 5, 62), (20, 10, 60))
    session.close()


def test_beat_start_can_stop_after_a_complete_gesture():
    session, _ = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.configure_recording_sync(
        start_mode=RecordingStartMode.NEXT_BEAT,
        stop_mode=RecordingStopMode.NEXT_GESTURE,
    )
    session.set_recording_armed(True)
    session.advance(TICKS_PER_BEAT)

    session.process_midi_message(
        mido.Message("note_on", channel=0, note=60, velocity=90),
        TICKS_PER_BEAT + 10,
    )
    session.process_midi_message(
        mido.Message("note_off", channel=0, note=60, velocity=0),
        TICKS_PER_BEAT + 20,
    )

    assert session.recording_state is RecordingState.IDLE
    assert session.tape(0).length_ticks == 20
    session.close()


def test_gesture_end_started_before_beat_recording_activation_counts():
    session, _ = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.configure_recording_sync(
        start_mode=RecordingStartMode.NEXT_BEAT,
        stop_mode=RecordingStopMode.NEXT_GESTURE,
    )
    session.set_recording_armed(True)
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=60, velocity=90), 10
    )
    session.advance(TICKS_PER_BEAT)
    assert session.recording_state is RecordingState.RECORDING

    session.process_midi_message(
        mido.Message("note_off", channel=0, note=60, velocity=0),
        TICKS_PER_BEAT + 10,
    )

    assert session.recording_state is RecordingState.IDLE
    interval, = session.tape(0).intervals()
    assert (interval.start, interval.duration, interval.pitch) == (0, 10, 60)
    session.close()


def test_gesture_start_can_stop_at_reference_loop_boundary():
    session, _ = make_session((0, 960, 0, 0))
    session.toggle_transport()
    session.configure_recording_sync(
        start_mode=RecordingStartMode.NEXT_GESTURE,
        stop_mode=RecordingStopMode.NEXT_REFERENCE_LOOP_START,
        reference_channel=1,
    )
    session.set_recording_armed(True)
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=60, velocity=90), 10
    )
    session.process_midi_message(
        mido.Message("note_off", channel=0, note=60, velocity=0), 20
    )
    session.request_recording_stop()
    session.advance(959)
    assert session.recording_state is RecordingState.WAITING_STOP

    session.advance(960)

    assert session.recording_state is RecordingState.IDLE
    session.close()


def test_reference_loop_start_can_stop_after_a_complete_gesture():
    session, _ = make_session((0, 960, 0, 0))
    session.toggle_transport()
    session.configure_recording_sync(
        start_mode=RecordingStartMode.NEXT_REFERENCE_LOOP_START,
        stop_mode=RecordingStopMode.NEXT_GESTURE,
        reference_channel=1,
    )
    session.set_recording_armed(True)
    session.advance(960)
    assert session.recording_state is RecordingState.RECORDING

    session.process_midi_message(
        mido.Message("note_on", channel=0, note=60, velocity=90), 970
    )
    session.process_midi_message(
        mido.Message("note_off", channel=0, note=60, velocity=0), 980
    )

    assert session.recording_state is RecordingState.IDLE
    session.close()


def test_gesture_stop_completes_after_recorded_gesture_crosses_loop_boundary():
    session, _ = make_session((100, 960, 0, 0))
    session.toggle_transport()
    session.advance(80)
    session.configure_recording_sync(
        stop_mode=RecordingStopMode.NEXT_GESTURE
    )
    session.set_recording_armed(True)
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=60, velocity=90), 90
    )
    session.advance(109)
    assert session.recording_state is RecordingState.RECORDING
    session.process_midi_message(
        mido.Message("note_off", channel=0, note=60, velocity=0), 110
    )

    assert session.recording_state is RecordingState.IDLE
    assert sum(
        interval.duration for interval in session.tape(0).intervals()
    ) == 20
    session.close()


def test_playback_onsets_do_not_count_as_recording_gestures():
    session, _ = make_session((0, 1000, 0, 0))
    session.tape(1).insert(IntervalInsert(10, 10, 60, 90, 1))
    session.toggle_transport()
    session.configure_recording_sync(
        start_mode=RecordingStartMode.NEXT_GESTURE
    )
    session.set_recording_armed(True)

    session.advance(10)

    assert session.recording_state is RecordingState.WAITING_START
    assert session.tape(1).intervals()
    session.set_recording_armed(False)
    session.close()


def test_synchronized_stop_closes_held_recording_at_boundary_without_silencing_live_voice():
    session, sinks = make_session((0, 1000, 0, 0))
    session.toggle_transport()
    session.advance(200)
    session.configure_recording_sync(stop_mode=RecordingStopMode.NEXT_BEAT)
    session.set_recording_armed(True)
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=60, velocity=90), 300
    )
    session.toggle_recording()
    assert session.recording_state is RecordingState.WAITING_STOP

    session.advance(1000)
    assert session.recording_state is RecordingState.IDLE
    assert session.tape(0).length_ticks == 760
    interval, = session.tape(0).intervals()
    assert (interval.start, interval.duration, interval.pitch) == (100, 660, 60)
    assert session.articulators[0].live_voice is not None
    assert not any(
        getattr(message, "type") == "note_off" and getattr(message, "note") == 60
        for message in sinks[0].messages
    )
    session.process_midi_message(
        mido.Message("note_off", channel=0, note=60, velocity=0), 1010
    )
    session.close()

def test_note_on_at_stop_boundary_is_excluded_from_half_open_recording_window():
    session, _ = make_session((0, 1000, 0, 0))
    session.toggle_transport()
    session.advance(200)
    session.configure_recording_sync(stop_mode=RecordingStopMode.NEXT_BEAT)
    session.set_recording_armed(True)
    session.toggle_recording()

    session.process_midi_message(
        mido.Message("note_on", channel=0, note=62, velocity=100),
        TICKS_PER_BEAT,
    )

    assert session.recording_state is RecordingState.IDLE
    assert session.tape(0).length_ticks == TICKS_PER_BEAT - 200
    assert session.tape(0).intervals() == ()
    session.close()

def test_pending_start_cancel_and_pending_stop_cancel_preserve_state():
    session, _ = make_session((0, 1000, 0, 0))
    session.toggle_transport()
    session.configure_recording_sync(start_mode=RecordingStartMode.NEXT_BEAT)
    session.set_recording_armed(True)
    session.toggle_recording()
    assert session.recording_state is RecordingState.IDLE
    assert session.tape(0).length_ticks == 0

    session.configure_recording_sync(start_mode=RecordingStartMode.IMMEDIATE)
    session.set_recording_armed(True)
    session.configure_recording_sync(stop_mode=RecordingStopMode.NEXT_BEAT)
    session.toggle_recording()
    assert session.recording_state is RecordingState.WAITING_STOP
    session.toggle_recording()
    assert session.recording_state is RecordingState.RECORDING
    assert session.record_armed
    session.set_recording_armed(False)
    session.close()

def test_reference_loop_start_uses_actual_phase_wrap_and_rejects_undefined_loop():
    session, _ = make_session((0, 1200, 0, 0))
    session.toggle_transport()
    session.advance(150)
    session.configure_recording_sync(
        start_mode=RecordingStartMode.NEXT_REFERENCE_LOOP_START,
        reference_channel=1,
    )
    session.set_recording_armed(True)
    assert session.recording_state is RecordingState.WAITING_START
    session.advance(1200)
    assert session.recording_state is RecordingState.RECORDING
    assert session.phase_origin_tick(0) == 1200
    session.set_recording_armed(False)
    session.close()

    undefined, _ = make_session((0, 0, 0, 0))
    undefined.toggle_transport()
    undefined.configure_recording_sync(
        start_mode=RecordingStartMode.NEXT_REFERENCE_LOOP_START,
        reference_channel=1,
    )
    with pytest.raises(ValueError, match="no defined loop"):
        undefined.set_recording_armed(True)
    assert not undefined.record_armed
    undefined.close()

def test_reference_loop_stop_uses_selected_loop_wrap():
    session, _ = make_session((0, 1200, 0, 0))
    session.toggle_transport()
    session.advance(200)
    session.configure_recording_sync(
        stop_mode=RecordingStopMode.NEXT_REFERENCE_LOOP_START,
        reference_channel=1,
    )
    session.set_recording_armed(True)
    session.toggle_recording()
    assert session.recording_state is RecordingState.WAITING_STOP

    session.advance(1200)

    assert session.recording_state is RecordingState.IDLE
    assert session.tape(0).length_ticks == 1000
    session.close()

def test_request_on_exact_beat_waits_for_the_next_occurrence():
    session, _ = make_session((0, 1000, 0, 0))
    session.toggle_transport()
    session.advance(TICKS_PER_BEAT)
    session.configure_recording_sync(start_mode=RecordingStartMode.NEXT_BEAT)
    session.set_recording_armed(True)

    session.advance(TICKS_PER_BEAT)
    assert session.recording_state is RecordingState.WAITING_START
    session.advance(2 * TICKS_PER_BEAT)

    assert session.recording_state is RecordingState.RECORDING
    assert session.phase_origin_tick(0) == 2 * TICKS_PER_BEAT
    session.set_recording_armed(False)
    session.close()


def test_counted_start_beat_aligns_then_waits_complete_beats():
    session, _ = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.advance(2 * TICKS_PER_BEAT + 400)
    session.configure_recording_sync(
        start_mode=RecordingStartMode.NEXT_BEAT,
        start_beat_count=2,
    )
    session.set_recording_armed(True)

    trigger_tick = 5 * TICKS_PER_BEAT
    session.advance(trigger_tick - 1)
    assert session.recording_state is RecordingState.WAITING_START
    session.advance(trigger_tick)

    assert session.recording_state is RecordingState.RECORDING
    assert session.phase_origin_tick(0) == trigger_tick
    session.set_recording_armed(False)
    session.close()


def test_stop_beat_uses_next_eligible_boundary_without_count():
    session, _ = make_session((0, 1000, 0, 0))
    session.toggle_transport()
    session.advance(2 * TICKS_PER_BEAT + 400)
    session.configure_recording_sync(
        start_mode=RecordingStartMode.IMMEDIATE,
        stop_mode=RecordingStopMode.NEXT_BEAT,
    )
    session.set_recording_armed(True)
    session.request_recording_stop()

    trigger_tick = 3 * TICKS_PER_BEAT
    session.advance(trigger_tick - 1)
    assert session.recording_state is RecordingState.WAITING_STOP
    session.advance(trigger_tick)

    assert session.recording_state is RecordingState.IDLE
    assert session.tape(0).length_ticks == trigger_tick - (
        2 * TICKS_PER_BEAT + 400
    )
    session.close()


def test_stop_after_measures_duration_from_actual_recording_start_tick():
    session, _ = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.advance(137)
    session.configure_recording_sync(
        stop_mode=RecordingStopMode.AFTER_DURATION,
        stop_after_ticks=4 * TICKS_PER_BEAT,
    )
    session.set_recording_armed(True)
    assert session.current_tick == 137
    assert session.recording_state is RecordingState.RECORDING

    session.advance(137 + 4 * TICKS_PER_BEAT - 1)
    assert session.recording_state is RecordingState.RECORDING
    session.advance(137 + 4 * TICKS_PER_BEAT)

    assert session.recording_state is RecordingState.IDLE
    assert session.tape(0).length_ticks == 4 * TICKS_PER_BEAT
    session.close()


def test_stop_after_fractional_duration_uses_exact_ticks():
    session, _ = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.advance(137)
    duration_ticks = 4 * TICKS_PER_BEAT + 3 * TICKS_PER_BEAT // 8
    session.configure_recording_sync(
        stop_mode=RecordingStopMode.AFTER_DURATION,
        stop_after_ticks=duration_ticks,
    )
    session.set_recording_armed(True)

    session.advance(137 + duration_ticks - 1)
    assert session.recording_state is RecordingState.RECORDING
    session.advance(137 + duration_ticks)

    assert session.recording_state is RecordingState.IDLE
    assert session.tape(0).length_ticks == duration_ticks
    session.close()


def test_stop_after_reconfiguration_uses_original_recording_start_tick():
    session, _ = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.advance(100)
    session.set_recording_armed(True)
    session.advance(110)
    session.configure_recording_sync(
        stop_mode=RecordingStopMode.AFTER_DURATION,
        stop_after_ticks=20,
    )

    session.advance(119)
    assert session.recording_state is RecordingState.RECORDING
    session.advance(120)

    assert session.recording_state is RecordingState.IDLE
    assert session.tape(0).length_ticks == 20
    session.close()


def test_stop_after_timer_starts_when_synchronized_recording_begins():
    session, _ = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.advance(137)
    session.configure_recording_sync(
        start_mode=RecordingStartMode.NEXT_BEAT,
        stop_mode=RecordingStopMode.AFTER_DURATION,
        stop_after_ticks=TICKS_PER_BEAT // 4,
    )
    session.set_recording_armed(True)

    session.advance(TICKS_PER_BEAT - 1)
    assert session.recording_state is RecordingState.WAITING_START
    session.advance(TICKS_PER_BEAT)
    assert session.recording_state is RecordingState.RECORDING

    stop_tick = TICKS_PER_BEAT + TICKS_PER_BEAT // 4
    session.advance(stop_tick - 1)
    assert session.recording_state is RecordingState.RECORDING
    session.advance(stop_tick)

    assert session.recording_state is RecordingState.IDLE
    assert session.tape(0).length_ticks == TICKS_PER_BEAT // 4
    session.close()


def test_stop_after_from_gesture_start_ignores_waiting_time():
    session, _ = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.advance(137)
    session.configure_recording_sync(
        start_mode=RecordingStartMode.NEXT_GESTURE,
        stop_mode=RecordingStopMode.AFTER_DURATION,
        stop_after_ticks=25,
    )
    session.set_recording_armed(True)
    session.advance(200)
    assert session.recording_state is RecordingState.WAITING_START

    session.process_midi_message(
        mido.Message("note_on", channel=0, note=60, velocity=90), 300
    )
    assert session.recording_state is RecordingState.RECORDING
    session.advance(324)
    assert session.recording_state is RecordingState.RECORDING
    session.advance(325)

    assert session.recording_state is RecordingState.IDLE
    assert session.tape(0).length_ticks == 25
    session.close()


def test_stop_after_from_reference_loop_start_uses_boundary_tick():
    session, _ = make_session((0, 960, 0, 0))
    session.toggle_transport()
    session.advance(137)
    session.configure_recording_sync(
        start_mode=RecordingStartMode.NEXT_REFERENCE_LOOP_START,
        stop_mode=RecordingStopMode.AFTER_DURATION,
        reference_channel=1,
        stop_after_ticks=25,
    )
    session.set_recording_armed(True)
    session.advance(959)
    assert session.recording_state is RecordingState.WAITING_START
    session.advance(960)
    assert session.recording_state is RecordingState.RECORDING

    session.advance(984)
    assert session.recording_state is RecordingState.RECORDING
    session.advance(985)

    assert session.recording_state is RecordingState.IDLE
    assert session.tape(0).length_ticks == 25
    session.close()


def test_stop_after_automatically_disarms_without_silencing_live_note():
    session, sinks = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.advance(100)
    session.configure_recording_sync(
        stop_mode=RecordingStopMode.AFTER_DURATION,
        stop_after_ticks=20,
    )
    session.set_recording_armed(True)
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=60, velocity=90), 110
    )

    session.advance(120)

    assert session.recording_state is RecordingState.IDLE
    assert not session.record_armed
    assert session.articulators[0].live_voice is not None
    session.process_midi_message(
        mido.Message("note_off", channel=0, note=60, velocity=0), 130
    )
    assert session.articulators[0].live_voice is None
    assert [
        (getattr(message, "type"), getattr(message, "note"))
        for message in sinks[0].messages
        if getattr(message, "type") in ("note_on", "note_off")
    ] == [("note_on", 60), ("note_off", 60)]
    interval, = session.tape(0).intervals()
    assert (interval.start, interval.duration, interval.pitch) == (10, 10, 60)
    session.close()


@pytest.mark.parametrize(
    ("beat_count", "trigger_beat"),
    [(None, 8), (1, 9), (2, 10)],
)
def test_stopped_unprocessed_boundary_aligns_counted_start(
    beat_count, trigger_beat
):
    stopped, _ = make_session((0, 0, 0, 0))
    stopped.advance(8 * TICKS_PER_BEAT)
    stopped.configure_recording_sync(
        start_mode=RecordingStartMode.NEXT_BEAT,
        start_beat_count=beat_count,
    )
    stopped.set_recording_armed(True)
    stopped.toggle_transport()

    trigger_tick = trigger_beat * TICKS_PER_BEAT
    if beat_count is None:
        assert stopped.recording_state is RecordingState.RECORDING
    else:
        assert stopped.recording_state is RecordingState.WAITING_START
        stopped.advance(trigger_tick - 1)
        assert stopped.recording_state is RecordingState.WAITING_START
        stopped.advance(trigger_tick)
        assert stopped.recording_state is RecordingState.RECORDING
    assert stopped.phase_origin_tick(0) == trigger_tick
    stopped.set_recording_armed(False)
    stopped.close()


def test_stopped_half_beat_excludes_partial_beat_from_counted_start():
    session, _ = make_session((0, 0, 0, 0))
    session.advance(8 * TICKS_PER_BEAT + TICKS_PER_BEAT // 2)
    session.configure_recording_sync(
        start_mode=RecordingStartMode.NEXT_BEAT,
        start_beat_count=1,
    )
    session.set_recording_armed(True)
    session.toggle_transport()

    trigger_tick = 10 * TICKS_PER_BEAT
    session.advance(trigger_tick - 1)
    assert session.recording_state is RecordingState.WAITING_START
    session.advance(trigger_tick)
    assert session.recording_state is RecordingState.RECORDING
    assert session.phase_origin_tick(0) == trigger_tick
    session.set_recording_armed(False)
    session.close()


def test_counted_start_pauses_while_transport_is_stopped():
    session, _ = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.advance(200)
    session.configure_recording_sync(
        start_mode=RecordingStartMode.NEXT_BEAT,
        start_beat_count=2,
    )
    session.set_recording_armed(True)
    scheduled_tick = 3 * TICKS_PER_BEAT

    session.toggle_transport()
    assert session.recording_state is RecordingState.WAITING_START
    stopped_tick = session.current_tick
    session.toggle_transport()
    assert session.current_tick == stopped_tick
    session.advance(scheduled_tick - 1)
    assert session.recording_state is RecordingState.WAITING_START
    session.advance(scheduled_tick)
    assert session.recording_state is RecordingState.RECORDING
    assert session.phase_origin_tick(0) == scheduled_tick
    session.set_recording_armed(False)
    session.close()


def test_start_beat_does_not_repeat_a_boundary_processed_before_stop():
    processed, _ = make_session((0, 0, 0, 0))
    processed.toggle_transport()
    processed.advance(8 * TICKS_PER_BEAT)
    processed.toggle_transport()
    processed.configure_recording_sync(start_mode=RecordingStartMode.NEXT_BEAT)
    processed.set_recording_armed(True)
    processed.toggle_transport()

    assert processed.recording_state is RecordingState.WAITING_START
    processed.advance(9 * TICKS_PER_BEAT - 1)
    assert processed.recording_state is RecordingState.WAITING_START
    processed.advance(9 * TICKS_PER_BEAT)
    assert processed.recording_state is RecordingState.RECORDING
    processed.set_recording_armed(False)
    processed.close()

def test_next_beat_boundary_is_not_missed_by_large_tick_advance():
    session, _ = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.advance(100)
    session.configure_recording_sync(start_mode=RecordingStartMode.NEXT_BEAT)
    session.set_recording_armed(True)

    session.advance(TICKS_PER_BEAT + 25)

    assert session.recording_state is RecordingState.RECORDING
    assert session.phase_origin_tick(0) == TICKS_PER_BEAT
    assert session.tape(0).length_ticks == 25
    session.set_recording_armed(False)
    session.close()

def test_pending_start_freezes_across_transport_stop_and_uses_next_future_boundary():
    session, _ = make_session((0, 1000, 0, 0))
    session.toggle_transport()
    session.advance(200)
    session.configure_recording_sync(start_mode=RecordingStartMode.NEXT_BEAT)
    session.set_recording_armed(True)
    session.advance(500)
    session.toggle_transport()
    assert session.recording_state is RecordingState.WAITING_START

    session.toggle_transport()
    session.advance(TICKS_PER_BEAT)
    assert session.recording_state is RecordingState.RECORDING
    assert session.phase_origin_tick(0) == TICKS_PER_BEAT
    session.set_recording_armed(False)
    session.close()

def test_pending_stop_freezes_across_transport_stop_and_resumes_to_scheduled_boundary():
    session, _ = make_session((0, 1000, 0, 0))
    session.toggle_transport()
    session.advance(200)
    session.configure_recording_sync(stop_mode=RecordingStopMode.NEXT_BEAT)
    session.set_recording_armed(True)
    session.advance(500)
    session.toggle_recording()
    assert session.recording_state is RecordingState.WAITING_STOP

    session.toggle_transport()
    session.toggle_transport()
    session.advance(TICKS_PER_BEAT)

    assert session.recording_state is RecordingState.IDLE
    assert session.tape(0).length_ticks == TICKS_PER_BEAT - 200
    session.close()

def test_reference_loop_change_cancels_pending_start():
    session, _ = make_session((0, 1000, 0, 0))
    session.toggle_transport()
    session.advance(100)
    session.configure_recording_sync(
        start_mode=RecordingStartMode.NEXT_REFERENCE_LOOP_START,
        reference_channel=1,
    )
    session.set_recording_armed(True)
    session.clear_tape(1)

    assert session.recording_state is RecordingState.IDLE
    assert not session.record_armed
    session.close()

def test_recording_is_global_focused_and_overwrite_resets_before_channel_change():
    session, _ = make_session()
    session.toggle_mode()
    assert session.mode is RecordMode.OVERWRITE
    assert session.record_armed
    assert session.recorder(0).mode is RecordMode.OVERWRITE
    assert session.recorder(0).record_armed

    session.select_channel(1)
    assert session.active_channel == 1
    assert session.mode is RecordMode.OVERDUB
    assert session.recorder(0).mode is RecordMode.OVERDUB
    assert not session.recorder(0).record_armed
    assert session.recorder(1).record_armed
    assert not session.recorder(2).record_armed

    session.toggle_recording()
    assert not session.record_armed
    assert not session.recorder(1).record_armed
    session.close()

def test_channel_change_preserves_recording_sync_settings_and_reference():
    session, _ = make_session()
    session.select_channel(1)
    session.configure_recording_sync(
        stop_mode=RecordingStopMode.NEXT_REFERENCE_LOOP_START,
        reference_channel=0,
    )
    session.set_recording_armed(True)
    session.select_channel(2)

    assert session.active_channel == 2
    assert session.record_armed
    assert session.start_mode is RecordingStartMode.IMMEDIATE
    assert session.stop_mode is RecordingStopMode.NEXT_REFERENCE_LOOP_START
    assert session.reference_channel == 0
    assert session.recording_state is RecordingState.RECORDING

    session.close()


@pytest.mark.parametrize(
    ("start_mode", "stop_mode"),
    [
        (
            RecordingStartMode.NEXT_REFERENCE_LOOP_START,
            RecordingStopMode.IMMEDIATE,
        ),
        (
            RecordingStartMode.IMMEDIATE,
            RecordingStopMode.NEXT_REFERENCE_LOOP_START,
        ),
    ],
)
def test_arming_rejects_self_reference_boundaries_without_changing_state(
    start_mode, stop_mode
):
    session, _ = make_session()
    session.configure_recording_sync(
        start_mode=start_mode,
        stop_mode=stop_mode,
        reference_channel=0,
    )

    with pytest.raises(
        ValueError,
        match="recording channel cannot use itself as the reference channel",
    ):
        session.set_recording_armed(True)

    assert not session.record_armed
    assert session.recording_state is RecordingState.IDLE
    session.close()


def test_armed_channel_change_rejects_a_self_referencing_target():
    session, _ = make_session()
    session.configure_recording_sync(
        stop_mode=RecordingStopMode.NEXT_REFERENCE_LOOP_START,
        reference_channel=0,
    )
    session.select_channel(1)
    session.set_recording_armed(True)

    with pytest.raises(ValueError, match="cannot use itself as the reference channel"):
        session.select_channel(0)

    assert session.active_channel == 1
    assert session.record_armed
    assert session.recording_state is RecordingState.RECORDING
    session.close()


def test_sync_configuration_cannot_make_an_active_recording_self_referential():
    session, _ = make_session()
    session.set_recording_armed(True)

    with pytest.raises(ValueError, match="cannot use itself as the reference channel"):
        session.configure_recording_sync(
            stop_mode=RecordingStopMode.NEXT_REFERENCE_LOOP_START,
            reference_channel=0,
        )

    assert session.stop_mode is RecordingStopMode.IMMEDIATE
    assert session.reference_channel is None
    assert session.record_armed
    session.close()

def test_four_channel_focus_change_does_not_change_other_channel_playback():
    session, sinks = make_session()
    session.tape(2).insert(IntervalInsert(120, 240, 72, 90, 7))
    session.toggle_transport()
    session.advance(120)
    session.select_channel(3)
    session.advance(360)

    assert [
        (getattr(message, "type"), getattr(message, "note"))
        for message in sinks[2].messages
    ] == [
        ("note_on", 72),
        ("note_off", 72),
    ]
    assert session.active_channel == 3
    session.close()

def test_input_from_any_midi_channel_records_only_on_selected_loop():
    session, sinks = make_session()
    session.toggle_transport()
    session.toggle_recording()
    session.process_midi_message(
        mido.Message("note_on", channel=7, note=64, velocity=100), 0
    )
    session.process_midi_message(
        mido.Message("note_off", channel=7, note=64, velocity=0), 480
    )

    intervals = session.tape(0).intervals()
    assert [(item.start, item.duration, item.pitch) for item in intervals] == [
        (0, 480, 64)
    ]
    assert [
        (getattr(message, "type"), getattr(message, "channel"))
        for message in sinks[0].messages
    ] == [
        ("note_on", 0),
        ("note_off", 0),
    ]
    assert all(not sink.messages for sink in sinks[1:])
    session.close()


def test_reset_rephases_all_loops_and_only_retriggers_phase_zero_onsets():
    session, sinks = make_session((1000, 1200, 600, 800))
    onset = session.tape(0).insert(IntervalInsert(0, 800, 60, 90, 1))
    spanning = session.tape(1).insert(IntervalInsert(1100, 200, 64, 80, 2))
    session.toggle_transport()
    session.advance(500)

    session.reset(500)

    assert session.current_tick == 500
    assert session.musical_tick == 0
    assert tuple(session.phase_origin_tick(index) for index in range(4)) == (
        500,
        500,
        500,
        500,
    )
    assert tuple(session.tape(index).length_ticks for index in range(4)) == (
        1000,
        1200,
        600,
        800,
    )
    assert session.tape(0).intervals() == (onset,)
    assert session.tape(1).intervals() == (spanning,)
    assert session.transport_running
    assert [
        (getattr(message, "type"), getattr(message, "note"))
        for message in sinks[0].messages
    ] == [
        ("note_on", 60),
        ("note_off", 60),
        ("note_on", 60),
    ]
    assert not any(getattr(message, "type") == "note_on" for message in sinks[1].messages)
    session.close()


def test_reset_rephases_free_recorded_loop_and_beat_synchronization():
    session, _ = make_session((0, 960, 600, 800))
    session.tape(1).insert(IntervalInsert(25, 50, 64, 80, 1))
    session.toggle_transport()
    session.advance(TICKS_PER_BEAT // 2)
    session.set_recording_armed(True)
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=60, velocity=90),
        TICKS_PER_BEAT // 2 + 20,
    )
    session.process_midi_message(
        mido.Message("note_off", channel=0, note=60, velocity=0),
        TICKS_PER_BEAT // 2 + 70,
    )
    session.advance(1680)
    session.set_recording_armed(False)

    original_lengths = tuple(session.tape(i).length_ticks for i in range(4))
    original_intervals = tuple(session.tape(i).intervals() for i in range(4))
    assert session.phase_origin_tick(0) == TICKS_PER_BEAT // 2
    assert original_lengths[0] == 1200
    assert original_intervals[0][0].start == 20

    session.reset(2000)

    assert session.musical_tick == 0
    assert tuple(session.phase_origin_tick(i) for i in range(4)) == (2000,) * 4
    assert tuple(session.tape(i).length_ticks for i in range(4)) == original_lengths
    assert tuple(session.tape(i).intervals() for i in range(4)) == original_intervals

    session.configure_recording_sync(
        start_mode=RecordingStartMode.NEXT_BEAT,
        stop_mode=RecordingStopMode.NEXT_BEAT,
    )
    session.set_recording_armed(True)
    assert session.recording_state is RecordingState.WAITING_START
    session.advance(2000 + TICKS_PER_BEAT - 1)
    assert session.recording_state is RecordingState.WAITING_START
    session.advance(2000 + TICKS_PER_BEAT)
    assert session.recording_state is RecordingState.RECORDING
    session.request_recording_stop()
    assert session.recording_state is RecordingState.WAITING_STOP
    session.advance(2000 + 2 * TICKS_PER_BEAT)
    assert session.recording_state is RecordingState.IDLE
    assert session.current_tick == 2000 + 2 * TICKS_PER_BEAT
    session.close()


def test_reset_while_paused_keeps_transport_stopped_and_live_note_state():
    session, sinks = make_session()
    session.tape(0).insert(IntervalInsert(0, 100, 60, 90, 1))
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=67, velocity=88), 0
    )

    session.reset(0)

    assert not session.transport_running
    assert session.articulators[0].live_voice is not None
    assert [
        (getattr(message, "type"), getattr(message, "note"))
        for message in sinks[0].messages
    ] == [("note_on", 67)]
    session.process_midi_message(
        mido.Message("note_off", channel=0, note=67, velocity=0), 0
    )
    assert [
        (getattr(message, "type"), getattr(message, "note"))
        for message in sinks[0].messages
    ] == [("note_on", 67), ("note_off", 67)]
    session.close()


def test_immediate_reset_coincident_with_processed_automatic_reset_is_coalesced():
    session, sinks = make_session()
    session.tape(0).insert(IntervalInsert(0, 200, 60, 90, 1))
    session.configure_reset(ResetMode.BEAT, interval_beats=1)
    session.toggle_transport()
    session.advance(500)

    session.advance(TICKS_PER_BEAT)
    note_ons_before_manual_reset = sum(
        getattr(message, "type") == "note_on" for message in sinks[0].messages
    )
    session.reset(TICKS_PER_BEAT)

    assert session.musical_tick == 0
    assert sum(
        getattr(message, "type") == "note_on" for message in sinks[0].messages
    ) == note_ons_before_manual_reset
    session.close()


def test_reset_cancels_pending_start_and_commits_active_and_pending_stop_recording():
    pending, _ = make_session((0, 1000, 0, 0))
    pending.toggle_transport()
    pending.advance(100)
    pending.configure_recording_sync(start_mode=RecordingStartMode.NEXT_BEAT)
    pending.set_recording_armed(True)
    pending.reset(200)
    assert pending.recording_state is RecordingState.IDLE
    assert not pending.record_armed
    assert pending.tape(0).length_ticks == 0
    pending.close()

    for stop_mode in (RecordingStopMode.IMMEDIATE, RecordingStopMode.NEXT_BEAT):
        session, _ = make_session((0, 1000, 0, 0))
        session.toggle_transport()
        session.set_recording_armed(True)
        session.process_midi_message(
            mido.Message("note_on", channel=0, note=60, velocity=90), 100
        )
        if stop_mode is RecordingStopMode.NEXT_BEAT:
            session.configure_recording_sync(stop_mode=stop_mode)
            session.request_recording_stop()
            assert session.recording_state is RecordingState.WAITING_STOP

        session.reset(300)

        assert session.recording_state is RecordingState.IDLE
        assert not session.record_armed
        (interval,) = session.tape(0).intervals()
        assert (interval.start, interval.duration) == (100, 200)
        session.close()


def test_reset_commits_actual_duration_of_a_new_loop_definition():
    session, _ = make_session((0, 1000, 0, 0))
    session.toggle_transport()
    session.set_recording_armed(True)
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=62, velocity=80), 100
    )

    session.reset(300)

    assert session.tape(0).length_ticks == 300
    (interval,) = session.tape(0).intervals()
    assert (interval.start, interval.duration) == (100, 200)
    assert session.loop_definition_channel is None
    session.close()


def test_recurring_beat_reset_survives_large_advances_and_transport_pause():
    session, _ = make_session()
    session.configure_reset(ResetMode.BEAT, interval_beats=1)
    session.toggle_transport()
    session.advance(500)
    session.toggle_transport()
    session.advance(500)
    assert session.musical_tick == 500

    session.toggle_transport()
    session.advance(2500)

    assert session.current_tick == 2500
    assert session.musical_tick == 580
    assert session.phase_origin_tick(0) == 1920
    assert session.reset_mode is ResetMode.BEAT
    session.close()


def test_manual_reset_restarts_beat_reset_schedule_from_new_origin():
    session, _ = make_session()
    session.configure_reset(ResetMode.BEAT, interval_beats=1)
    session.toggle_transport()
    session.advance(500)
    session.reset(500)
    session.advance(500 + TICKS_PER_BEAT - 1)

    assert session.musical_tick == TICKS_PER_BEAT - 1
    assert session.reset_mode is ResetMode.BEAT
    session.advance(500 + TICKS_PER_BEAT)
    assert session.musical_tick == 0
    assert session.reset_mode is ResetMode.BEAT
    session.reset(1500)
    session.advance(1500 + TICKS_PER_BEAT - 1)
    assert session.musical_tick == TICKS_PER_BEAT - 1
    session.advance(1500 + TICKS_PER_BEAT)
    assert session.musical_tick == 0
    session.configure_reset(ResetMode.OFF)
    session.close()


def test_fractional_beat_reset_uses_multiples_from_its_origin():
    session, _ = make_session()
    interval_ticks = 3 * TICKS_PER_BEAT + 3 * TICKS_PER_BEAT // 8
    session.toggle_transport()
    session.configure_reset(ResetMode.BEAT, interval_ticks=interval_ticks)

    session.advance(interval_ticks - 1)
    assert session.musical_tick == interval_ticks - 1
    session.advance(interval_ticks)
    assert session.musical_tick == 0
    assert session.phase_origin_tick(0) == interval_ticks

    session.advance(2 * interval_ticks)
    assert session.musical_tick == 0
    assert session.phase_origin_tick(0) == 2 * interval_ticks
    session.close()


def test_fractional_reset_alignment_uses_the_next_origin_relative_multiple():
    session, _ = make_session()
    interval_ticks = 3 * TICKS_PER_BEAT + 3 * TICKS_PER_BEAT // 8
    session.toggle_transport()
    session.advance(10 * TICKS_PER_BEAT + TICKS_PER_BEAT // 2)
    session.configure_reset(ResetMode.BEAT, interval_ticks=interval_ticks)

    next_multiple = 4 * interval_ticks
    session.advance(next_multiple - 1)
    assert session.musical_tick == next_multiple - 1
    session.advance(next_multiple)
    assert session.musical_tick == 0
    assert session.phase_origin_tick(0) == next_multiple
    session.close()


def test_changing_reset_interval_selects_next_origin_relative_multiple():
    session, _ = make_session()
    session.toggle_transport()
    session.advance(10 * TICKS_PER_BEAT + TICKS_PER_BEAT // 2)
    session.configure_reset(ResetMode.BEAT, interval_ticks=4 * TICKS_PER_BEAT)
    session.advance(12 * TICKS_PER_BEAT - 1)
    assert session.musical_tick == 12 * TICKS_PER_BEAT - 1
    session.advance(12 * TICKS_PER_BEAT)
    assert session.musical_tick == 0
    session.close()


def test_manual_reset_coincident_with_automatic_boundary_runs_once():
    session, sinks = make_session()
    session.tape(0).insert(IntervalInsert(0, 200, 60, 90, 1))
    session.configure_reset(ResetMode.BEAT, interval_beats=1)
    session.toggle_transport()
    session.advance(500)

    session.reset(TICKS_PER_BEAT)
    session.advance(TICKS_PER_BEAT)

    assert sum(
        getattr(message, "type") == "note_on" for message in sinks[0].messages
    ) == 2
    assert sum(
        getattr(message, "type") == "note_off" for message in sinks[0].messages
    ) == 1
    assert session.musical_tick == 0
    session.close()


def test_loop_reset_repeats_at_reference_wraps_and_reference_changes_reschedule():
    session, _ = make_session((1000, 1200, 600, 800))
    session.configure_recording_sync(reference_channel=0)
    session.configure_reset(ResetMode.LOOP)
    session.toggle_transport()
    session.advance(400)
    session.configure_recording_sync(reference_channel=1)
    session.advance(1000)
    assert session.musical_tick == 1000
    assert session.phase_origin_tick(0) == 0

    session.advance(2500)
    assert session.current_tick == 2500
    assert session.musical_tick == 100
    assert session.phase_origin_tick(0) == 2400
    assert session.phase_origin_tick(1) == 2400
    assert session.reset_mode is ResetMode.LOOP
    session.close()


def test_loop_reset_waits_for_future_wrap_after_paused_transport_resumes():
    session, _ = make_session((1000, 1200, 600, 800))
    session.configure_recording_sync(reference_channel=0)
    session.configure_reset(ResetMode.LOOP)
    session.toggle_transport()
    session.advance(400)
    session.toggle_transport()
    session.toggle_transport()
    session.advance(999)

    assert session.musical_tick == 999
    session.advance(1000)

    assert session.musical_tick == 0
    assert session.phase_origin_tick(0) == 1000
    session.close()


def test_loop_reset_accepts_empty_defined_reference_and_disables_when_cleared():
    session, _ = make_session((1000, 0, 600, 800))
    session.configure_recording_sync(reference_channel=0)
    session.configure_reset(ResetMode.LOOP)
    assert session.reset_mode is ResetMode.LOOP

    session.clear_tape(0)

    assert session.reset_mode is ResetMode.OFF
    session.close()
