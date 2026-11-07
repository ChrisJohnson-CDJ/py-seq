import mido

from articulator import Articulator
from interval_resolver import IntervalResolver, RecordMode
from midi_output import CollectingMidiSink, MidiOutputAdapter
from playback_coordinator import PlaybackChannel, PlaybackCoordinator
from playback_scheduler import PlaybackScheduler
from pre_processor import (
    MidiVoicePreprocessor,
    VoiceEvent,
    VoiceEventType,
    adapt_mido_message,
)
from recorder import Recorder
from tape import IntervalInsert, Tape


def build_coordinator(
    *,
    lengths: tuple[int, int] = (100, 150),
    intervals: tuple[tuple[IntervalInsert, ...], tuple[IntervalInsert, ...]] = (
        (),
        (),
    ),
):
    channels = []
    sinks = []
    for index in range(2):
        tape = Tape(lengths[index])
        for interval in intervals[index]:
            tape.insert(interval)
        recorder = Recorder(
            tape,
            IntervalResolver(),
            fragment_threshold_ticks=0,
            mode=RecordMode.OVERDUB,
        )
        sink = CollectingMidiSink()
        sinks.append(sink)
        channels.append(
            PlaybackChannel(
                recorder,
                PlaybackScheduler(tape),
                Articulator(),
                MidiOutputAdapter(sink, index + 2),
            )
        )
    return PlaybackCoordinator(tuple(channels)), tuple(sinks)


def test_two_tapes_with_different_lengths_remain_independently_phased():
    coordinator, sinks = build_coordinator(
        intervals=(
            (IntervalInsert(10, 10, 60, 90, 1),),
            (IntervalInsert(30, 10, 65, 80, 2),),
        )
    )
    coordinator.start(0)
    coordinator.advance(110)

    assert [
        (message.type, message.note, message.channel)
        for message in sinks[0].messages
    ] == [
        ("note_on", 60, 2),
        ("note_off", 60, 2),
        ("note_on", 60, 2),
    ]
    assert [
        (message.type, message.note, message.channel)
        for message in sinks[1].messages
    ] == [("note_on", 65, 3), ("note_off", 65, 3)]


def test_live_input_wins_over_simultaneous_playback_onset_and_is_not_restored():
    coordinator, sinks = build_coordinator(
        intervals=((IntervalInsert(10, 30, 60, 90, 1),), ())
    )
    preprocessor = MidiVoicePreprocessor()
    coordinator.start(0)

    normalized = adapt_mido_message(
        mido.Message("note_on", note=64, velocity=100), timestamp_ns=1
    )
    voice = preprocessor.process(normalized)
    assert voice is not None
    coordinator.handle_voice_event(voice, 10)

    normalized_off = adapt_mido_message(
        mido.Message("note_off", note=64, velocity=0), timestamp_ns=2
    )
    end = preprocessor.process(normalized_off)
    assert end is not None
    coordinator.handle_voice_event(end, 20)
    coordinator.advance(40)

    assert [
        (message.type, message.note, message.velocity)
        for message in sinks[0].messages
    ] == [
        ("note_on", 60, 90),
        ("note_off", 60, 0),
        ("note_on", 64, 100),
        ("note_off", 64, 0),
    ]


def test_recorder_edits_tape_then_playback_reconciles_without_stale_note():
    coordinator, sinks = build_coordinator()
    coordinator.start(0)
    coordinator.set_record_armed(0, True, 5)

    start = VoiceEvent(VoiceEventType.VOICE_START, 10, None, 72, 100)
    end = VoiceEvent(VoiceEventType.VOICE_END, 10, 72, None, None)
    coordinator.handle_voice_event(start, 10)
    coordinator.handle_voice_event(end, 20)
    coordinator.seek(0)
    coordinator.advance(10)
    coordinator.advance(20)

    assert [
        (message.type, message.note)
        for message in sinks[0].messages
    ] == [
        ("note_on", 72),
        ("note_off", 72),
        ("note_on", 72),
        ("note_off", 72),
    ]
    assert [
        (interval.start, interval.duration, interval.pitch)
        for interval in coordinator.channels[0].recorder.tape.intervals()
    ] == [(10, 10, 72)]


def test_overwrite_silence_does_not_play_erased_note_boundaries():
    coordinator, sinks = build_coordinator(
        intervals=((IntervalInsert(40, 20, 67, 90, 1),), ())
    )
    coordinator.set_record_armed(0, True, 0)
    coordinator.start(0)
    coordinator.set_record_mode(0, RecordMode.OVERWRITE, 0)
    coordinator.advance(40)
    coordinator.advance(50)
    coordinator.advance(60)

    assert sinks[0].messages == []


def test_focus_handoff_releases_old_output_and_routes_next_change_to_new_focus():
    coordinator, sinks = build_coordinator()
    preprocessor = MidiVoicePreprocessor()

    start = preprocessor.process(
        adapt_mido_message(mido.Message("note_on", note=60, velocity=90))
    )
    assert start is not None
    coordinator.handle_voice_event(start, 0)
    coordinator.set_focus(1, 5)

    change = preprocessor.process(
        adapt_mido_message(mido.Message("note_on", note=64, velocity=100))
    )
    assert change is not None
    coordinator.handle_voice_event(change, 10)

    release_new = preprocessor.process(
        adapt_mido_message(mido.Message("note_off", note=64, velocity=0))
    )
    assert release_new is not None
    coordinator.handle_voice_event(release_new, 15)
    release_old = preprocessor.process(
        adapt_mido_message(mido.Message("note_off", note=60, velocity=0))
    )
    assert release_old is not None
    coordinator.handle_voice_event(release_old, 20)

    assert [(msg.type, msg.note) for msg in sinks[0].messages] == [
        ("note_on", 60),
        ("note_off", 60),
    ]
    assert [(msg.type, msg.note) for msg in sinks[1].messages] == [
        ("note_on", 64),
        ("note_off", 64),
        ("note_on", 60),
        ("note_off", 60),
    ]
    assert coordinator.channels[1].recorder.tape.intervals() == ()


def test_transport_stop_releases_sounding_note_and_closes_recording():
    coordinator, sinks = build_coordinator(
        intervals=((IntervalInsert(0, 50, 60, 90, 1),), ())
    )
    coordinator.start(0)
    coordinator.stop(10)

    assert [(msg.type, msg.note) for msg in sinks[0].messages] == [
        ("note_on", 60),
        ("note_off", 60),
    ]
    assert not coordinator.channels[0].scheduler.running


def test_input_disconnect_attempts_to_silence_both_outputs_after_one_fails():
    class FailOnNoteOff:
        def __init__(self):
            self.messages = []

        def send(self, message):
            self.messages.append(message)
            if message.type == "note_off":
                raise OSError("device disconnected")

    tapes = (
        Tape(100),
        Tape(150),
    )
    tapes[0].insert(IntervalInsert(0, 40, 60, 90, 1))
    tapes[1].insert(IntervalInsert(0, 40, 65, 80, 2))
    bad_sink = FailOnNoteOff()
    good_sink = CollectingMidiSink()
    channel0 = PlaybackChannel(
        Recorder(tapes[0], IntervalResolver(), fragment_threshold_ticks=0),
        PlaybackScheduler(tapes[0]),
        Articulator(),
        MidiOutputAdapter(bad_sink, 0),
    )
    channel1 = PlaybackChannel(
        Recorder(tapes[1], IntervalResolver(), fragment_threshold_ticks=0),
        PlaybackScheduler(tapes[1]),
        Articulator(),
        MidiOutputAdapter(good_sink, 1),
    )
    channels = (channel0, channel1)
    coordinator = PlaybackCoordinator(channels)
    coordinator.start(0)

    try:
        coordinator.disconnect_input(1)
    except ExceptionGroup as error:
        assert len(error.exceptions) == 1
    else:
        raise AssertionError("disconnect must report the failed MIDI send")

    assert [getattr(message, "type") for message in good_sink.messages] == [
        "note_on",
        "note_off",
    ]
    assert coordinator.channels[0].articulator.active_pitch is None
    assert coordinator.channels[1].articulator.active_pitch is None
    assert coordinator.channels[0].output.desynchronized
    assert coordinator.current_tick == 1
