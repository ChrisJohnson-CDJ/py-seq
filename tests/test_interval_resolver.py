import pytest

from interval_resolver import (
    EraseRegion,
    InsertCandidate,
    IntervalResolver,
    RecordMode,
)
from tape import IntervalInsert, Tape


def insert(
    tape: Tape,
    start: int,
    duration: int,
    *,
    pitch: int = 60,
    velocity: int = 90,
    gesture_id: int = 1,
):
    return tape.insert(
        IntervalInsert(start, duration, pitch, velocity, gesture_id)
    )


def apply(tape: Tape, operation, threshold: int = 20, mode=RecordMode.OVERDUB):
    edit = IntervalResolver().resolve(tape.snapshot(), operation, mode, threshold)
    tape.replace(edit)
    return edit


def test_overdub_splits_existing_interval_and_keeps_onset_id():
    tape = Tape(960)
    original = insert(tape, 100, 400, gesture_id=5)

    edit = apply(tape, InsertCandidate(200, 100, 64, 80, 9))

    assert edit.remove_ids == frozenset({original.id})
    onset, incoming, ending = tape.intervals()
    assert (onset.id, onset.start, onset.duration) == (original.id, 100, 100)
    assert (incoming.start, incoming.duration, incoming.pitch) == (200, 100, 64)
    assert (ending.start, ending.duration, ending.gesture_id) == (300, 200, 5)
    assert ending.id != original.id


def test_zero_length_tape_cannot_resolve_an_edit():
    with pytest.raises(ValueError, match="zero-length"):
        IntervalResolver().resolve(
            Tape(0).snapshot(),
            InsertCandidate(0, 1, 60, 90, 1),
            RecordMode.OVERDUB,
            0,
        )


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (InsertCandidate(110, 190, 64, 80, 9), (300, 200)),
        (InsertCandidate(200, 290, 64, 80, 9), (100, 100)),
    ],
)
def test_threshold_drops_each_short_old_fragment_without_extending_candidate(
    candidate, expected
):
    tape = Tape(960)
    original = insert(tape, 100, 400)

    apply(tape, candidate, threshold=20)

    old_material = [item for item in tape.intervals() if item.gesture_id == 1]
    assert [(item.start, item.duration) for item in old_material] == [expected]
    assert old_material[0].id == original.id
    incoming = next(item for item in tape.intervals() if item.gesture_id == 9)
    assert (incoming.start, incoming.duration) == (
        candidate.start_tick,
        candidate.duration,
    )


def test_threshold_does_not_suppress_short_incoming_candidate():
    tape = Tape(960)
    original = insert(tape, 100, 300)

    apply(tape, InsertCandidate(200, 10, 64, 80, 9), threshold=20)

    assert [(item.start, item.duration, item.gesture_id) for item in tape.intervals()] == [
        (100, 100, 1),
        (200, 10, 9),
        (210, 190, 1),
    ]
    assert tape.intervals()[0].id == original.id
    assert tape.intervals()[-1].id != original.id


def test_erasure_creates_silence_and_preserves_surviving_material():
    tape = Tape(960)
    original = insert(tape, 100, 400)

    edit = apply(tape, EraseRegion(150, 300), mode=RecordMode.OVERWRITE)

    onset, ending = tape.intervals()
    assert edit.insert[0].preserve_id == original.id
    assert (onset.id, onset.start, onset.duration) == (original.id, 100, 50)
    assert (ending.start, ending.duration) == (450, 50)
    assert ending.id != original.id
    assert tape.at(200) is None


def test_wraparound_candidate_and_full_length_onset_are_subtracted_circularly():
    tape = Tape(10)
    original = insert(tape, 8, 4)

    apply(tape, InsertCandidate(10, 2, 64, 80, 9), threshold=0)

    remaining = next(item for item in tape.intervals() if item.gesture_id == 1)
    assert (remaining.id, remaining.start, remaining.duration) == (
        original.id,
        8,
        2,
    )

    full = Tape(10)
    full_interval = insert(full, 3, 10)
    apply(full, InsertCandidate(4, 1, 64, 80, 9), threshold=0)
    fragments = [item for item in full.intervals() if item.gesture_id == 1]
    assert [(item.start, item.duration) for item in fragments] == [(3, 1), (5, 8)]
    assert fragments[0].id == full_interval.id
    assert fragments[1].id != full_interval.id


def test_long_coverage_is_capped_and_unaffected_intervals_are_retained():
    tape = Tape(10)
    first = insert(tape, 1, 2, gesture_id=1)
    second = insert(tape, 5, 2, gesture_id=2)

    edit = IntervalResolver().resolve(
        tape.snapshot(), EraseRegion(3, 100), RecordMode.OVERWRITE, 0
    )

    assert edit.remove_ids == frozenset({first.id, second.id})
    tape.replace(edit)
    assert tape.intervals() == ()

    candidate = InsertCandidate(12, 20, 70, 100, 3)
    edit = IntervalResolver().resolve(
        tape.snapshot(), candidate, RecordMode.OVERDUB, 0
    )
    assert (edit.insert[0].start, edit.insert[0].duration) == (2, 10)


def test_resolver_is_pure_and_validates_operation_parameters():
    tape = Tape(20)
    original = insert(tape, 2, 5)
    before = tape.intervals()
    edit = IntervalResolver().resolve(
        tape.snapshot(), InsertCandidate(3, 2, 64, 80, 2), RecordMode.OVERDUB, 0
    )
    assert tape.intervals() == before
    assert original.id in edit.remove_ids

    with pytest.raises(TypeError):
        IntervalResolver().resolve(
            tape.snapshot(), EraseRegion(0, 1), RecordMode.OVERDUB, True
        )
    with pytest.raises(ValueError):
        InsertCandidate(0, 0, 60, 90, 1)


def test_snapshot_is_immutable_and_remains_a_point_in_time_view():
    tape = Tape(20)
    original = insert(tape, 2, 5)
    snapshot = tape.snapshot()
    tape.clear()

    edit = IntervalResolver().resolve(
        snapshot, InsertCandidate(3, 2, 64, 80, 2), RecordMode.OVERDUB, 0
    )
    assert original.id in edit.remove_ids
    with pytest.raises(AttributeError):
        setattr(snapshot, "length_ticks", 30)
