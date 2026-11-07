from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import mido
import pytest

from pre_processor import (
    MidiInputEvent,
    MidiInputEventType,
    MidiVoicePreprocessor,
    VoiceEvent,
    VoiceEventType,
    adapt_mido_message,
)


def event(kind, pitch=None, velocity=None, *, channel=0, timestamp_ns=None):
    return MidiInputEvent(kind, channel, pitch, velocity, timestamp_ns)


def on(pitch, velocity=90, *, timestamp_ns=None, channel=0):
    return event(
        MidiInputEventType.NOTE_ON,
        pitch,
        velocity,
        channel=channel,
        timestamp_ns=timestamp_ns,
    )


def off(pitch, velocity=0, *, timestamp_ns=None, channel=0):
    return event(
        MidiInputEventType.NOTE_OFF,
        pitch,
        velocity,
        channel=channel,
        timestamp_ns=timestamp_ns,
    )


def voice(kind, gesture_id, previous_pitch, pitch, velocity, timestamp_ns=None):
    return VoiceEvent(kind, gesture_id, previous_pitch, pitch, velocity, timestamp_ns)


def assert_invariants(preprocessor):
    held = preprocessor._held_notes
    assert (preprocessor._active_gesture_id is not None) == bool(held)
    if held:
        assert preprocessor._selected_pitch() == next(reversed(held))
        assert preprocessor._active_gesture_id > 0


def test_starts_ends_and_allocates_increasing_gesture_ids():
    processor = MidiVoicePreprocessor()
    assert processor.process(on(60, 90)) == voice(VoiceEventType.VOICE_START, 1, None, 60, 90)
    assert processor.process(off(60)) == voice(VoiceEventType.VOICE_END, 1, 60, None, None)
    assert processor.process(on(67, 80)) == voice(VoiceEventType.VOICE_START, 2, None, 67, 80)


def test_overlapping_notes_restore_last_accepted_held_pitch_and_velocity():
    processor = MidiVoicePreprocessor()
    outputs = [
        processor.process(on(60, 90)),
        processor.process(on(64, 70)),
        processor.process(on(67, 60)),
        processor.process(off(64)),
        processor.process(off(67)),
    ]
    assert outputs == [
        voice(VoiceEventType.VOICE_START, 1, None, 60, 90),
        voice(VoiceEventType.VOICE_CHANGE, 1, 60, 64, 70),
        voice(VoiceEventType.VOICE_CHANGE, 1, 64, 67, 60),
        None,
        voice(VoiceEventType.VOICE_CHANGE, 1, 67, 60, 90),
    ]
    assert processor._held_notes == {60: 90}
    assert_invariants(processor)


def test_selected_release_restores_previous_note_with_original_velocity():
    processor = MidiVoicePreprocessor()
    assert processor.process(on(60, 90)) is not None
    assert processor.process(on(64, 70)) is not None
    assert processor.process(off(64)) == voice(VoiceEventType.VOICE_CHANGE, 1, 64, 60, 90)
    assert processor.process(off(60)) == voice(VoiceEventType.VOICE_END, 1, 60, None, None)


def test_nonselected_release_does_not_change_selected_voice():
    processor = MidiVoicePreprocessor()
    processor.process(on(60))
    processor.process(on(64))
    assert processor.process(off(60)) is None
    assert processor._held_notes == {64: 90}
    assert_invariants(processor)


def test_duplicate_note_on_is_ignored_without_changing_velocity_or_priority():
    processor = MidiVoicePreprocessor()
    processor.process(on(60, 90))
    processor.process(on(64, 70))
    assert processor.process(on(60, 30)) is None
    assert processor._held_notes == {60: 90, 64: 70}
    assert processor.process(off(64)) == voice(VoiceEventType.VOICE_CHANGE, 1, 64, 60, 90)
    assert processor.process(off(60)) == voice(VoiceEventType.VOICE_END, 1, 60, None, None)
    assert processor.process(off(60)) is None


def test_duplicate_selected_note_on_is_ignored_until_first_release():
    processor = MidiVoicePreprocessor()
    assert processor.process(on(60, 90)) == voice(VoiceEventType.VOICE_START, 1, None, 60, 90)
    assert processor.process(on(60, 30)) is None
    assert processor._held_notes == {60: 90}
    assert processor.process(off(60)) == voice(VoiceEventType.VOICE_END, 1, 60, None, None)
    assert processor.process(off(60)) is None


def test_unmatched_releases_do_not_allocate_ids_before_or_after_gesture():
    processor = MidiVoicePreprocessor()
    assert processor.process(off(55)) is None
    assert processor._next_gesture_id == 1
    assert processor.process(on(60)) is not None
    assert processor.process(off(60)) is not None
    assert [processor.process(off(60)) for _ in range(3)] == [None, None, None]
    assert processor.process(on(62)) == voice(VoiceEventType.VOICE_START, 2, None, 62, 90)


def test_other_and_unconfigured_channel_have_no_state_effects():
    processor = MidiVoicePreprocessor()
    other = MidiInputEvent(MidiInputEventType.OTHER)
    assert processor.process(other) is None
    processor.process(on(60))
    assert processor.process(event(MidiInputEventType.OTHER)) is None
    assert processor.process(on(64, channel=1)) is None
    assert processor._held_notes == {60: 90}
    assert processor._next_gesture_id == 2
    assert_invariants(processor)


def test_timestamps_are_copied_and_never_used_to_reorder():
    processor = MidiVoicePreprocessor()
    first = processor.process(on(60, timestamp_ns=400))
    second = processor.process(on(64, timestamp_ns=12))
    third = processor.process(off(64, timestamp_ns=None))
    assert [first.timestamp_ns, second.timestamp_ns, third.timestamp_ns] == [400, 12, None]
    assert [first.kind, second.kind, third.kind] == [
        VoiceEventType.VOICE_START,
        VoiceEventType.VOICE_CHANGE,
        VoiceEventType.VOICE_CHANGE,
    ]


def test_timestamp_accepts_arbitrarily_large_nonnegative_integers():
    timestamp = 10**100
    assert MidiVoicePreprocessor().process(on(60, timestamp_ns=timestamp)).timestamp_ns == timestamp


def test_reset_clears_state_without_event_and_preserves_id_counter():
    processor = MidiVoicePreprocessor()
    assert processor.process(on(60)) is not None
    assert processor.reset() is None
    assert processor._held_notes == {}
    assert processor._active_gesture_id is None
    assert processor.process(off(60)) is None
    assert processor.process(on(64)) == voice(VoiceEventType.VOICE_START, 2, None, 64, 90)
    assert_invariants(processor)


def test_velocity_zero_note_on_is_normalized_to_note_off():
    processor = MidiVoicePreprocessor()
    processor.process(on(60))
    normalized = adapt_mido_message(
        mido.Message("note_on", channel=0, note=60, velocity=0), timestamp_ns=7
    )
    assert normalized.kind is MidiInputEventType.NOTE_OFF
    assert processor.process(normalized) == voice(
        VoiceEventType.VOICE_END, 1, 60, None, None, 7
    )


def test_adapter_passes_other_messages_and_note_fields_to_core():
    control = adapt_mido_message(mido.Message("control_change", control=7, value=10))
    assert control == MidiInputEvent(MidiInputEventType.OTHER)
    processor = MidiVoicePreprocessor()
    assert processor.process(control) is None
    assert processor.process(adapt_mido_message(mido.Message("note_on", note=60, velocity=90))) == voice(
        VoiceEventType.VOICE_START, 1, None, 60, 90
    )


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        (SimpleNamespace(type="note_on", channel=True, note=60, velocity=1), TypeError),
        (SimpleNamespace(type="note_on", channel=16, note=60, velocity=1), ValueError),
        (SimpleNamespace(type="note_on", channel=0, note=-1, velocity=1), ValueError),
        (SimpleNamespace(type="note_on", channel=0, note=60, velocity=128), ValueError),
        (SimpleNamespace(type="note_off", channel=0, note=60, velocity=True), TypeError),
    ],
)
def test_adapter_rejects_malformed_raw_note_fields(raw, error):
    with pytest.raises(error):
        adapt_mido_message(raw)


@pytest.mark.parametrize(
    ("bad_event", "error"),
    [
        (object(), TypeError),
        (replace(on(60), kind="note_on"), TypeError),
        (MidiInputEvent(MidiInputEventType.NOTE_ON), ValueError),
        (MidiInputEvent(MidiInputEventType.NOTE_ON, pitch=60), ValueError),
        (event(MidiInputEventType.NOTE_ON, True, 90), TypeError),
        (event(MidiInputEventType.NOTE_ON, 128, 90), ValueError),
        (event(MidiInputEventType.NOTE_ON, 60, 0), ValueError),
        (event(MidiInputEventType.NOTE_OFF, 60, 128), ValueError),
        (event(MidiInputEventType.NOTE_OFF, 60, 0, channel=True), TypeError),
        (event(MidiInputEventType.NOTE_OFF, 60, 0, channel=16), ValueError),
        (event(MidiInputEventType.OTHER, timestamp_ns=-1), ValueError),
        (event(MidiInputEventType.OTHER, timestamp_ns=True), TypeError),
        (event(MidiInputEventType.OTHER, timestamp_ns=1.5), TypeError),
    ],
)
def test_invalid_events_raise_before_state_mutation(bad_event, error):
    processor = MidiVoicePreprocessor()
    before = (processor._held_notes.copy(), processor._active_gesture_id, processor._next_gesture_id)
    with pytest.raises(error):
        processor.process(bad_event)
    assert (processor._held_notes, processor._active_gesture_id, processor._next_gesture_id) == before


def test_other_channel_short_circuits_pitch_and_velocity_but_checks_timestamp():
    processor = MidiVoicePreprocessor(input_channel=3)
    assert processor.process(event(MidiInputEventType.NOTE_ON, channel=2)) is None
    with pytest.raises(ValueError, match="timestamp_ns"):
        processor.process(event(MidiInputEventType.NOTE_ON, channel=2, timestamp_ns=-1))
    assert processor._next_gesture_id == 1


def test_normalized_note_on_zero_is_invalid():
    with pytest.raises(ValueError, match="velocity"):
        MidiVoicePreprocessor().process(event(MidiInputEventType.NOTE_ON, 60, 0))


def test_constructor_rejects_invalid_channels_and_event_contracts_are_immutable():
    for channel, error in [(True, TypeError), (-1, ValueError), (16, ValueError)]:
        with pytest.raises(error):
            MidiVoicePreprocessor(input_channel=channel)
    instance = MidiInputEvent(MidiInputEventType.OTHER)
    with pytest.raises(FrozenInstanceError):
        instance.kind = MidiInputEventType.NOTE_ON


def test_transition_sequence_emits_zero_or_one_event_per_message_and_invariants_hold():
    processor = MidiVoicePreprocessor()
    messages = [on(60), on(64), on(60), off(64), off(60), off(64), on(67)]
    results = []
    for message in messages:
        result = processor.process(message)
        assert result is None or isinstance(result, VoiceEvent)
        results.append(result)
        assert_invariants(processor)
    assert results == [
        voice(VoiceEventType.VOICE_START, 1, None, 60, 90),
        voice(VoiceEventType.VOICE_CHANGE, 1, 60, 64, 90),
        None,
        voice(VoiceEventType.VOICE_CHANGE, 1, 64, 60, 90),
        voice(VoiceEventType.VOICE_END, 1, 60, None, None),
        None,
        voice(VoiceEventType.VOICE_START, 2, None, 67, 90),
    ]


def test_reset_does_not_emit_voice_end_or_assume_downstream_note_identity():
    processor = MidiVoicePreprocessor()
    downstream_resets = []

    def caller_reset():
        processor.reset()
        downstream_resets.append("RESET")

    processor.process(on(60))
    caller_reset()
    assert downstream_resets == ["RESET"]
    assert processor._held_notes == {}
    assert processor._active_gesture_id is None
