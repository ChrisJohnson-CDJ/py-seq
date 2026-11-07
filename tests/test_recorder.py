import pytest

from interval_resolver import (
    EraseRegion,
    InsertCandidate,
    IntervalResolver,
    RecordMode,
)
from pre_processor import VoiceEvent, VoiceEventType
from recorder import Recorder
from tape import IntervalInsert, Tape, TapeEdit, TapeSnapshot


def voice(kind, gesture_id=1, previous=None, pitch=None, velocity=None):
    return VoiceEvent(kind, gesture_id, previous, pitch, velocity)


def start(pitch=60, velocity=90, gesture_id=1):
    return voice(VoiceEventType.VOICE_START, gesture_id, None, pitch, velocity)


def change(previous, pitch, velocity=90, gesture_id=1):
    return voice(
        VoiceEventType.VOICE_CHANGE, gesture_id, previous, pitch, velocity
    )


def end(pitch=60, gesture_id=1):
    return voice(VoiceEventType.VOICE_END, gesture_id, pitch, None, None)


def recorder(length=100, *, mode=RecordMode.OVERDUB):
    return Recorder(
        Tape(length),
        IntervalResolver(),
        mode=mode,
        fragment_threshold_ticks=0,
    )


def pending(subject):
    assert subject.pending_segment is not None
    return subject.pending_segment


def selected(subject):
    assert subject.selected_voice is not None
    return subject.selected_voice


def at(tape, tick):
    item = tape.at(tick)
    assert item is not None
    return item


class SpyResolver(IntervalResolver):
    def __init__(self, *, fail_first=False):
        self._resolver = IntervalResolver()
        self.operations = []
        self.fail_first = fail_first

    def resolve(
        self,
        snapshot: TapeSnapshot,
        operation: InsertCandidate | EraseRegion,
        mode: RecordMode,
        fragment_threshold_ticks: int,
    ) -> TapeEdit:
        self.operations.append(operation)
        if self.fail_first:
            self.fail_first = False
            raise RuntimeError("edit planning failed")
        return self._resolver.resolve(
            snapshot, operation, mode, fragment_threshold_ticks
        )


def test_capture_stores_unquantized_wrapping_voice_interval():
    subject = recorder(100)
    subject.set_transport_running(True, 0)
    subject.set_record_armed(True, 0)
    subject.accept_voice(start(60, 90, 12), 90)
    subject.advance(110)
    subject.accept_voice(end(60, 12), 110)

    item, = subject.tape.intervals()
    assert (item.start, item.duration, item.pitch, item.gesture_id) == (
        90,
        20,
        60,
        12,
    )


def test_arm_while_held_does_not_backfill_and_disarm_closes_capture():
    subject = recorder()
    subject.accept_voice(start(), 5)
    subject.set_transport_running(True, 8)
    subject.set_record_armed(True, 10)
    assert pending(subject).start_tick == 10

    subject.set_record_armed(False, 30)

    item, = subject.tape.intervals()
    assert (item.start, item.duration) == (10, 20)
    assert selected(subject).pitch == 60
    assert subject.pending_segment is None


def test_same_tick_change_discards_zero_duration_segment():
    subject = recorder(200)
    subject.set_transport_running(True, 0)
    subject.set_record_armed(True, 0)
    subject.accept_voice(start(60), 100)
    subject.accept_voice(change(60, 64, 75), 100)
    subject.accept_voice(end(64), 150)

    item, = subject.tape.intervals()
    assert (item.start, item.duration, item.pitch) == (100, 50, 64)


def test_stop_restart_closes_and_reopens_using_preserved_selection():
    subject = recorder()
    subject.set_transport_running(True, 0)
    subject.set_record_armed(True, 0)
    subject.accept_voice(start(67, 80, 4), 10)
    subject.set_transport_running(False, 20)
    assert selected(subject).gesture_id == 4
    assert subject.pending_segment is None

    subject.set_transport_running(True, 30)
    subject.accept_voice(end(67, 4), 40)

    assert [(item.start, item.duration) for item in subject.tape.intervals()] == [
        (10, 10),
        (30, 10),
    ]


def test_seek_closes_old_segment_and_reopens_at_discontinuous_position():
    subject = recorder()
    subject.set_transport_running(True, 0)
    subject.set_record_armed(True, 0)
    subject.accept_voice(start(), 10)
    subject.seek(5, old_tick=20)
    assert pending(subject).start_tick == 5
    subject.accept_voice(end(), 15)

    assert [(item.start, item.duration) for item in subject.tape.intervals()] == [
        (5, 10),
        (15, 5),
    ]


def test_clear_discards_pending_material_and_reopens_held_voice():
    subject = recorder()
    subject.set_transport_running(True, 0)
    subject.set_record_armed(True, 0)
    subject.accept_voice(start(), 10)
    subject.clear(20)
    assert pending(subject).start_tick == 20
    subject.accept_voice(end(), 30)

    item, = subject.tape.intervals()
    assert (item.start, item.duration) == (20, 10)


def test_input_reset_closes_current_segment_and_clears_selection():
    subject = recorder()
    subject.set_transport_running(True, 0)
    subject.set_record_armed(True, 0)
    subject.accept_voice(start(), 10)
    subject.reset_input(25)

    item, = subject.tape.intervals()
    assert (item.start, item.duration) == (10, 15)
    assert subject.selected_voice is None
    assert subject.pending_segment is None


def test_overwrite_erases_silence_but_replaces_occupied_voice_atomically():
    tape = Tape(100)
    original = tape.insert(IntervalInsert(0, 50, 60, 90, 1))
    subject = Recorder(
        tape,
        IntervalResolver(),
        mode=RecordMode.OVERWRITE,
        fragment_threshold_ticks=0,
    )
    subject.set_transport_running(True, 0)
    subject.set_record_armed(True, 0)
    subject.advance(10)

    assert tape.at(5) is None
    assert at(tape, 20).id == original.id

    subject.accept_voice(start(64, 80, 2), 10)
    subject.advance(20)
    assert at(tape, 15).id == original.id
    subject.accept_voice(end(64, 2), 30)
    subject.advance(40)

    assert tape.at(5) is None
    assert at(tape, 15).pitch == 64
    assert tape.at(35) is None
    assert at(tape, 45).pitch == 60


def test_overwrite_held_voice_is_deferred_and_capped_to_one_loop():
    tape = Tape(10)
    tape.insert(IntervalInsert(0, 10, 60, 90, 1))
    subject = Recorder(
        tape,
        IntervalResolver(),
        mode=RecordMode.OVERWRITE,
        fragment_threshold_ticks=0,
    )
    subject.set_transport_running(True, 0)
    subject.set_record_armed(True, 0)
    subject.accept_voice(start(64, 80, 2), 1)
    subject.advance(50)

    assert [(item.start, item.duration, item.pitch) for item in tape.intervals()] == [
        (1, 9, 60)
    ]
    assert pending(subject).start_tick == 1
    subject.accept_voice(end(64, 2), 50)
    item, = tape.intervals()
    assert (item.start, item.duration, item.pitch) == (1, 10, 64)


def test_consecutive_overwrite_silence_spans_are_last_write_wins():
    subject = recorder(960, mode=RecordMode.OVERWRITE)
    subject.tape.insert(IntervalInsert(100, 200, 60, 90, 1))
    subject.set_transport_running(True, 0)
    subject.set_record_armed(True, 0)
    subject.accept_voice(start(64, 80, 2), 100)
    subject.accept_voice(end(64, 2), 300)
    subject.advance(1060)
    assert at(subject.tape, 150).pitch == 64
    subject.advance(1260)
    assert subject.tape.at(150) is None


def test_duplicate_controls_do_not_close_a_pending_segment():
    subject = recorder()
    subject.set_transport_running(True, 0)
    subject.set_record_armed(True, 0)
    subject.accept_voice(start(), 10)

    subject.set_record_armed(True, 20)
    subject.set_transport_running(True, 30)
    assert pending(subject).start_tick == 10
    subject.accept_voice(end(), 40)
    assert [(item.start, item.duration) for item in subject.tape.intervals()] == [
        (10, 30)
    ]


def test_repeated_advance_does_not_finalize_the_same_silence_twice():
    resolver = SpyResolver()
    subject = Recorder(
        Tape(100),
        resolver,
        mode=RecordMode.OVERWRITE,
        fragment_threshold_ticks=0,
    )
    subject.set_transport_running(True, 0)
    subject.set_record_armed(True, 0)

    subject.advance(10)
    subject.advance(10)

    assert resolver.operations == [EraseRegion(0, 10)]


def test_failed_overwrite_planning_propagates_without_advancing_coverage():
    tape = Tape(100)
    original = tape.insert(IntervalInsert(0, 20, 60, 90, 1))
    resolver = SpyResolver(fail_first=True)
    subject = Recorder(
        tape,
        resolver,
        mode=RecordMode.OVERWRITE,
        fragment_threshold_ticks=0,
    )
    subject.set_transport_running(True, 0)
    subject.set_record_armed(True, 0)

    with pytest.raises(RuntimeError, match="planning failed"):
        subject.advance(10)
    assert tape.intervals() == (original,)
    assert subject.current_tick == 0

    subject.advance(10)
    assert resolver.operations == [EraseRegion(0, 10), EraseRegion(0, 10)]
    assert [(item.start, item.duration) for item in tape.intervals()] == [(10, 10)]


def test_invalid_or_out_of_order_voice_events_do_not_change_selection():
    subject = recorder()
    subject.set_transport_running(True, 0)
    subject.set_record_armed(True, 0)
    subject.accept_voice(start(), 10)

    with pytest.raises(ValueError, match="previous_pitch"):
        subject.accept_voice(change(64, 67), 20)
    with pytest.raises(ValueError, match="nondecreasing"):
        subject.advance(9)
    assert selected(subject).pitch == 60
    assert pending(subject).start_tick == 10
