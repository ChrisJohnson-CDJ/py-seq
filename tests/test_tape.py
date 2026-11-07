from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from tape import (
    DuplicateIntervalError,
    IntervalInsert,
    IntervalNotFoundError,
    InvalidIntervalError,
    Tape,
    TapeEdit,
    TapeOverlapError,
    VoiceInterval,
)


def insert(
    tape: Tape,
    start: int,
    duration: int,
    *,
    pitch: int = 60,
    velocity: int = 90,
    gesture_id: int = 1,
) -> VoiceInterval:
    return tape.insert(
        IntervalInsert(start, duration, pitch, velocity, gesture_id)
    )


def test_tape_and_interval_validation():
    for length in (-1,):
        with pytest.raises(ValueError):
            Tape(length)
    for length in (True, 1.5, "12"):
        with pytest.raises(TypeError):
            Tape(cast(int, length))
    empty = Tape(0)
    assert empty.snapshot().intervals == ()
    assert empty.at(0) is None
    with pytest.raises(ValueError):
        empty.overlaps(0, 1)
    with pytest.raises(InvalidIntervalError):
        insert(empty, 0, 1)

    with pytest.raises(TypeError):
        VoiceInterval(True, 0, 1, 60, 90, 1)
    with pytest.raises(InvalidIntervalError):
        VoiceInterval(1, 0, 0, 60, 90, 1)
    with pytest.raises(InvalidIntervalError):
        IntervalInsert(0, 1, 128, 90, 1)
    with pytest.raises(FrozenInstanceError):
        setattr(IntervalInsert(0, 1, 60, 90, 1), "start", 2)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("id", True, TypeError),
        ("start", True, TypeError),
        ("duration", True, TypeError),
        ("pitch", True, TypeError),
        ("velocity", True, TypeError),
        ("gesture_id", True, TypeError),
        ("start", -1, InvalidIntervalError),
        ("duration", -1, InvalidIntervalError),
        ("pitch", -1, InvalidIntervalError),
        ("velocity", 0, InvalidIntervalError),
    ],
)
def test_voice_interval_rejects_invalid_fields(field, value, error):
    values = dict(id=1, start=0, duration=1, pitch=60, velocity=90, gesture_id=4)
    values[field] = value
    with pytest.raises(error):
        VoiceInterval(**values)


def test_insert_allocates_ids_and_preserves_query_semantics():
    tape = Tape(10)
    assert tape.revision == 0
    tail = insert(tape, 8, 2, pitch=72, velocity=111, gesture_id=43)
    assert tape.revision == 1
    head = insert(tape, 0, 8, pitch=40, velocity=1, gesture_id=43)
    assert tape.revision == 2

    assert (tail.id, head.id) == (1, 2)
    assert tape.intervals() == (head, tail)
    assert tape.at(8) == tail
    assert tape.at(10) == head
    assert tape.at(-1) == tail
    assert tape.overlaps(9, 3) == (head, tail)
    assert tape.by_gesture(43) == (head, tail)


def test_tape_revision_changes_only_when_content_changes():
    tape = Tape(10)
    tape.clear()
    assert tape.revision == 0

    interval = insert(tape, 2, 3)
    assert tape.revision == 1
    tape.remove(interval.id)
    assert tape.revision == 2
    tape.clear()
    assert tape.revision == 2


def test_only_provisional_loops_can_grow_and_finalized_loops_cannot_resize():
    tape = Tape(0)
    tape.extend_length_ticks(10)
    tape.insert(IntervalInsert(2, 3, 60, 90, 1))
    tape.extend_length_ticks(12)
    tape.finalize_length()

    with pytest.raises(RuntimeError, match="unfinalized"):
        tape.extend_length_ticks(13)
    with pytest.raises(ValueError, match="contains intervals"):
        tape.truncate()


def test_circular_queries_are_half_open_unique_and_validate_arguments():
    tape = Tape(10)
    tail = insert(tape, 8, 2, gesture_id=2)
    head = insert(tape, 0, 2, gesture_id=1)
    middle = insert(tape, 4, 2, gesture_id=1)

    assert tape.at(2) is None
    assert tape.at(-1) == tail
    assert tape.at(14) == middle
    assert tape.overlaps(9, 3) == (head, tail)
    assert tape.overlaps(4, 2) == (middle,)
    assert tape.overlaps(6, 2) == ()
    assert tape.by_gesture(1) == (head, middle)
    assert tape.by_gesture(999) == ()

    for bad_tick in (True, 1.5, "1"):
        with pytest.raises(TypeError):
            tape.at(cast(int, bad_tick))
    with pytest.raises(TypeError):
        tape.overlaps(True, 1)
    with pytest.raises(ValueError):
        tape.overlaps(0, 11)
    with pytest.raises(TypeError):
        tape.by_gesture(cast(int, True))


def test_replace_splits_and_preserves_only_designated_ids():
    tape = Tape(20)
    original = insert(tape, 0, 10, gesture_id=77)
    edit = TapeEdit(
        frozenset({original.id}),
        (
            IntervalInsert(0, 4, 60, 90, 77, preserve_id=original.id),
            IntervalInsert(7, 3, 60, 90, 77),
            IntervalInsert(4, 3, 64, 80, 88),
        ),
    )
    tape.replace(edit)

    onset, incoming, ending = tape.intervals()
    assert onset.id == original.id
    assert (onset.start, onset.duration, onset.gesture_id) == (0, 4, 77)
    assert (incoming.start, incoming.duration, incoming.pitch) == (4, 3, 64)
    assert (ending.start, ending.duration, ending.gesture_id) == (7, 3, 77)
    assert ending.id != original.id


def test_replace_can_reuse_one_removed_id_or_remove_without_insertion():
    tape = Tape(20)
    original = insert(tape, 0, 6, gesture_id=77)
    tape.replace(
        TapeEdit(
            frozenset({original.id}),
            (
                IntervalInsert(
                    5, 6, 64, 80, 77, preserve_id=original.id
                ),
            ),
        )
    )
    replacement, = tape.intervals()
    assert replacement.id == original.id
    assert tape.by_gesture(77) == (replacement,)

    tape.replace(TapeEdit(frozenset({replacement.id}), ()))
    assert tape.intervals() == ()


def test_replace_is_atomic_and_does_not_consume_ids_on_failure():
    tape = Tape(12)
    existing = insert(tape, 2, 4)
    before = tape.intervals()

    with pytest.raises(TapeOverlapError):
        tape.replace(
            TapeEdit(
                frozenset(),
                (IntervalInsert(5, 2, 70, 90, 3),),
            )
        )
    assert tape.intervals() == before

    following = insert(tape, 8, 2)
    assert following.id == existing.id + 1


def test_replace_validates_ids_and_arguments_atomically():
    tape = Tape(20)
    original = insert(tape, 2, 4)
    before = tape.intervals()

    with pytest.raises(IntervalNotFoundError):
        tape.replace(
            TapeEdit(frozenset({404}), (IntervalInsert(10, 2, 60, 90, 1),))
        )
    with pytest.raises(DuplicateIntervalError):
        tape.replace(
            TapeEdit(
                frozenset({original.id}),
                (
                    IntervalInsert(2, 1, 60, 90, 1, preserve_id=original.id),
                    IntervalInsert(3, 1, 60, 90, 1, preserve_id=original.id),
                ),
            )
        )
    with pytest.raises(DuplicateIntervalError):
        tape.replace(
            TapeEdit(frozenset(), (IntervalInsert(10, 1, 60, 90, 1, 77),))
        )
    with pytest.raises(TypeError):
        TapeEdit(cast(frozenset[int], {original.id}), ())
    assert tape.intervals() == before


def test_circular_overlap_bounds_and_full_length_interval():
    tape = Tape(10)
    crossing = insert(tape, 8, 4)
    assert tape.overlaps(9, 3) == (crossing,)
    assert tape.at(1) == crossing
    assert tape.at(2) is None

    full = Tape(10)
    all_ticks = insert(full, 6, 10)
    assert full.overlaps(9, 10) == (all_ticks,)
    with pytest.raises(TapeOverlapError):
        insert(full, 0, 1)
    with pytest.raises(InvalidIntervalError):
        insert(Tape(10), 0, 11)


def test_failed_removal_and_clear_do_not_reuse_storage_ids():
    tape = Tape(8)
    first = insert(tape, 1, 2)
    with pytest.raises(IntervalNotFoundError):
        tape.remove(999)
    assert tape.remove(first.id) == first

    second = insert(tape, 4, 2)
    tape.clear()
    assert tape.intervals() == ()
    third = insert(tape, 0, 1)
    assert third.id > second.id
    tape.replace(TapeEdit(frozenset(), ()))
    assert tape.intervals() == (third,)
