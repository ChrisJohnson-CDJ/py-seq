import pytest

from articulator import Articulator, LiveVoice, NoteAction, NoteActionKind
from playback_scheduler import PlaybackEvent, PlaybackEventKind
from tape import VoiceInterval


def event(kind: PlaybackEventKind, interval: VoiceInterval, tick: int = 0):
    return PlaybackEvent(kind, tick, interval)


def test_playback_onset_retriggers_and_end_releases_only_matching_voice():
    articulator = Articulator()
    first = VoiceInterval(1, 0, 10, 60, 90, 1)
    adjacent = VoiceInterval(2, 10, 10, 60, 90, 2)

    assert articulator.handle(event(PlaybackEventKind.ONSET, first)) == (
        NoteAction(NoteActionKind.NOTE_ON, 60, 90),
    )
    assert articulator.handle(event(PlaybackEventKind.END, first, 10)) == (
        NoteAction(NoteActionKind.NOTE_OFF, 60),
    )
    assert articulator.handle(event(PlaybackEventKind.ONSET, adjacent, 10)) == (
        NoteAction(NoteActionKind.NOTE_ON, 60, 90),
    )


def test_enter_active_is_silent_and_exit_does_not_duplicate_note_off():
    articulator = Articulator()
    interval = VoiceInterval(1, 0, 10, 60, 90, 1)

    assert articulator.handle(event(PlaybackEventKind.ENTER_ACTIVE, interval)) == ()
    assert articulator.active_pitch is None
    assert articulator.handle(event(PlaybackEventKind.EXIT_ACTIVE, interval)) == ()


def test_live_voice_interrupts_and_does_not_restore_playback_mid_gate():
    articulator = Articulator()
    interval = VoiceInterval(1, 0, 20, 60, 90, 1)
    assert articulator.handle(event(PlaybackEventKind.ONSET, interval)) == (
        NoteAction(NoteActionKind.NOTE_ON, 60, 90),
    )

    actions = articulator.set_live_voice(LiveVoice(64, 100))
    assert actions == (
        NoteAction(NoteActionKind.NOTE_OFF, 60),
        NoteAction(NoteActionKind.NOTE_ON, 64, 100),
    )
    assert articulator.set_live_voice(None) == (
        NoteAction(NoteActionKind.NOTE_OFF, 64),
    )
    assert articulator.active_pitch is None
    assert articulator.handle(event(PlaybackEventKind.END, interval, 20)) == ()


def test_replacing_same_pitch_live_voice_releases_before_retrigger():
    articulator = Articulator()
    articulator.set_live_voice(LiveVoice(60, 90))

    assert articulator.set_live_voice(LiveVoice(60, 110)) == (
        NoteAction(NoteActionKind.NOTE_OFF, 60),
        NoteAction(NoteActionKind.NOTE_ON, 60, 110),
    )


def test_silence_and_irrelevant_events_are_idempotent():
    articulator = Articulator()
    interval = VoiceInterval(1, 0, 10, 60, 90, 1)
    articulator.handle(event(PlaybackEventKind.ONSET, interval))

    assert articulator.silence() == (NoteAction(NoteActionKind.NOTE_OFF, 60),)
    assert articulator.silence() == ()
    assert articulator.handle(event(PlaybackEventKind.END, interval, 10)) == ()


def test_reset_playback_releases_playback_but_preserves_live_voice():
    articulator = Articulator()
    playback = VoiceInterval(1, 0, 10, 60, 90, 1)
    articulator.handle(event(PlaybackEventKind.ONSET, playback))
    articulator.set_live_voice(LiveVoice(64, 100))

    assert articulator.reset_playback() == ()
    assert articulator.live_voice == LiveVoice(64, 100)
    assert articulator.active_pitch == 64
    assert articulator.reset_playback() == ()

    articulator.set_live_voice(None)
    assert articulator.active_pitch is None


def test_note_actions_validate_midi_fields():
    with pytest.raises(ValueError):
        NoteAction(NoteActionKind.NOTE_ON, 128, 90)
    with pytest.raises(ValueError):
        NoteAction(NoteActionKind.NOTE_OFF, 60, 1)
    with pytest.raises(ValueError):
        LiveVoice(60, 0)
