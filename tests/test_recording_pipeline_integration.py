"""End-to-end tests for simulated MIDI recording through Tape."""

import mido

from interval_resolver import IntervalResolver, RecordMode
from pre_processor import MidiVoicePreprocessor, VoiceEventType, adapt_mido_message
from recorder import Recorder
from tape import IntervalInsert, Tape


def build_pipeline(
    *,
    length_ticks: int = 100,
    mode: RecordMode = RecordMode.OVERDUB,
    initial_intervals: tuple[IntervalInsert, ...] = (),
) -> tuple[MidiVoicePreprocessor, Recorder]:
    tape = Tape(length_ticks)
    for insertion in initial_intervals:
        tape.insert(insertion)

    recorder = Recorder(
        tape,
        IntervalResolver(),
        fragment_threshold_ticks=0,
        mode=mode,
    )
    recorder.set_transport_running(True, 0)
    recorder.set_record_armed(True, 0)
    return MidiVoicePreprocessor(), recorder


def send_midi(
    preprocessor: MidiVoicePreprocessor,
    recorder: Recorder,
    message: mido.Message,
    tick: int,
):
    normalized = adapt_mido_message(message, timestamp_ns=1_000_000 + tick * 1_000)
    voice_event = preprocessor.process(normalized)
    if voice_event is not None:
        recorder.accept_voice(voice_event, tick)
    return voice_event


def occupied(tape: Tape, tick: int):
    interval = tape.at(tick)
    assert interval is not None
    return interval


def test_midi_voice_changes_and_release_restore_record_as_separate_intervals():
    preprocessor, recorder = build_pipeline()

    start = send_midi(
        preprocessor, recorder, mido.Message("note_on", note=60, velocity=90), 10
    )
    change = send_midi(
        preprocessor, recorder, mido.Message("note_on", note=64, velocity=70), 30
    )
    restore = send_midi(
        preprocessor, recorder, mido.Message("note_off", note=64, velocity=0), 50
    )
    end = send_midi(
        preprocessor, recorder, mido.Message("note_off", note=60, velocity=0), 80
    )

    assert start is not None
    assert change is not None
    assert restore is not None
    assert end is not None
    assert [start.kind, change.kind, restore.kind, end.kind] == [
        VoiceEventType.VOICE_START,
        VoiceEventType.VOICE_CHANGE,
        VoiceEventType.VOICE_CHANGE,
        VoiceEventType.VOICE_END,
    ]
    assert len({start.gesture_id, change.gesture_id, restore.gesture_id, end.gesture_id}) == 1
    assert [
        (item.start, item.duration, item.pitch, item.velocity, item.gesture_id)
        for item in recorder.tape.intervals()
    ] == [
        (10, 20, 60, 90, start.gesture_id),
        (30, 20, 64, 70, start.gesture_id),
        (50, 30, 60, 90, start.gesture_id),
    ]


def test_zero_velocity_note_on_closes_wrapping_interval_using_transport_ticks():
    preprocessor, recorder = build_pipeline(length_ticks=100)

    start = send_midi(
        preprocessor,
        recorder,
        mido.Message("note_on", note=67, velocity=100),
        90,
    )
    end = send_midi(
        preprocessor,
        recorder,
        mido.Message("note_on", note=67, velocity=0),
        120,
    )

    assert start is not None
    assert end is not None
    item, = recorder.tape.intervals()
    assert start.kind is VoiceEventType.VOICE_START
    assert end.kind is VoiceEventType.VOICE_END
    assert (item.start, item.duration, item.pitch, item.gesture_id) == (
        90,
        30,
        67,
        start.gesture_id,
    )
    assert occupied(recorder.tape, 5) == item


def test_overwrite_midi_capture_erases_silence_and_replaces_note_coverage():
    preprocessor, recorder = build_pipeline(
        mode=RecordMode.OVERWRITE,
        initial_intervals=(IntervalInsert(0, 80, 48, 80, 9),),
    )

    on = send_midi(
        preprocessor,
        recorder,
        mido.Message("note_on", note=72, velocity=110),
        20,
    )
    off = send_midi(
        preprocessor,
        recorder,
        mido.Message("note_off", note=72, velocity=0),
        40,
    )
    recorder.advance(60)

    assert on is not None
    assert off is not None
    assert on.kind is VoiceEventType.VOICE_START
    assert off.kind is VoiceEventType.VOICE_END
    assert [
        (item.start, item.duration, item.pitch, item.gesture_id)
        for item in recorder.tape.intervals()
    ] == [
        (20, 20, 72, on.gesture_id),
        (60, 20, 48, 9),
    ]
    assert recorder.tape.at(10) is None
    assert occupied(recorder.tape, 30).pitch == 72
    assert recorder.tape.at(50) is None
