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

def test_first_note_start_records_triggering_gesture_once():
    session, sinks = make_session((0, 0, 0, 0))
    session.toggle_transport()
    session.configure_recording_sync(
        start_mode=RecordingStartMode.FIRST_NOTE_ONSET
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


def test_immediate_reset_restarts_beat_interval_without_replacing_its_mode():
    session, _ = make_session()
    session.configure_reset(ResetMode.BEAT, interval_beats=1)
    session.toggle_transport()
    session.advance(500)
    session.reset(500)
    session.advance(1459)

    assert session.musical_tick == 959
    assert session.reset_mode is ResetMode.BEAT
    session.advance(1460)
    assert session.musical_tick == 0
    assert session.reset_mode is ResetMode.BEAT
    session.configure_reset(ResetMode.OFF)
    session.advance(2000)
    assert session.musical_tick == 540
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
