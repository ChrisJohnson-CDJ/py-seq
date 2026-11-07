import pytest

from playback_scheduler import (
    PlaybackEventKind,
    PlaybackEventLimitError,
    PlaybackScheduler,
)
from tape import IntervalInsert, Tape, TapeEdit


def build_tape(*intervals: IntervalInsert, length: int = 100) -> Tape:
    tape = Tape(length)
    for interval in intervals:
        tape.insert(interval)
    return tape


def test_advance_reports_skipped_boundaries_in_order_and_end_before_onset():
    tape = build_tape(
        IntervalInsert(10, 10, 60, 90, 1),
        IntervalInsert(20, 10, 64, 80, 2),
    )
    scheduler = PlaybackScheduler(tape)

    assert scheduler.start(0) == ()
    events = scheduler.advance(30)

    assert [(event.kind, event.tick, event.interval.pitch) for event in events] == [
        (PlaybackEventKind.ONSET, 10, 60),
        (PlaybackEventKind.END, 20, 60),
        (PlaybackEventKind.ONSET, 20, 64),
        (PlaybackEventKind.END, 30, 64),
    ]


def test_wrap_crossing_interval_does_not_end_at_loop_wrap():
    tape = build_tape(IntervalInsert(90, 20, 67, 100, 1))
    scheduler = PlaybackScheduler(tape)

    scheduler.start(85)
    events = scheduler.advance(210)

    assert [(event.kind, event.tick) for event in events] == [
        (PlaybackEventKind.ONSET, 90),
        (PlaybackEventKind.END, 110),
        (PlaybackEventKind.ONSET, 190),
        (PlaybackEventKind.END, 210),
    ]


def test_full_loop_interval_has_no_wrap_transition_and_reonsets_at_designated_start():
    tape = build_tape(IntervalInsert(20, 100, 60, 90, 1))
    scheduler = PlaybackScheduler(tape)

    assert scheduler.start(0)[0].kind is PlaybackEventKind.ENTER_ACTIVE
    events = scheduler.advance(120)

    assert [(event.kind, event.tick) for event in events] == [
        (PlaybackEventKind.END, 20),
        (PlaybackEventKind.ONSET, 20),
        (PlaybackEventKind.END, 120),
        (PlaybackEventKind.ONSET, 120),
    ]


def test_start_and_seek_distinguish_onset_from_mid_interval_entry():
    tape = build_tape(IntervalInsert(10, 20, 60, 90, 1))
    scheduler = PlaybackScheduler(tape)

    assert scheduler.start(15)[0].kind is PlaybackEventKind.ENTER_ACTIVE
    events = scheduler.seek(10)

    assert [event.kind for event in events] == [
        PlaybackEventKind.EXIT_ACTIVE,
        PlaybackEventKind.ONSET,
    ]
    assert [event.tick for event in events] == [15, 10]


def test_reset_advance_excludes_events_at_reset_tick_and_rebases_phase():
    tape = build_tape(
        IntervalInsert(0, 10, 64, 80, 1),
        IntervalInsert(10, 10, 60, 90, 2),
    )
    scheduler = PlaybackScheduler(tape)
    scheduler.start(0)

    events = scheduler.advance_before(10)

    assert events == ()
    scheduler.reset_phase(10)
    phase_zero = scheduler.reconcile(10)
    assert [(event.kind, event.tick, event.interval.pitch) for event in phase_zero] == [
        (PlaybackEventKind.ONSET, 10, 64),
    ]
    assert scheduler.phase_origin_tick == 10
    assert scheduler.advance(11) == ()


def test_equal_ticks_and_backward_advances_are_handled_explicitly():
    scheduler = PlaybackScheduler(build_tape())
    scheduler.start(12)

    assert scheduler.advance(12) == ()
    with pytest.raises(ValueError, match="seek"):
        scheduler.advance(11)


def test_reconcile_ignores_storage_and_provenance_identity_but_exits_changed_content():
    tape = build_tape(IntervalInsert(10, 30, 60, 90, 1))
    scheduler = PlaybackScheduler(tape)
    scheduler.start(15)

    original = tape.intervals()[0]
    tape.replace(
        TapeEdit(frozenset({original.id}), (IntervalInsert(10, 30, 60, 90, 999),))
    )
    assert scheduler.reconcile(15) == ()

    replacement = tape.intervals()[0]
    tape.replace(
        TapeEdit(
            frozenset({replacement.id}),
            (IntervalInsert(10, 30, 61, 90, 999),),
        )
    )
    events = scheduler.reconcile(15)
    assert [event.kind for event in events] == [
        PlaybackEventKind.EXIT_ACTIVE,
        PlaybackEventKind.ENTER_ACTIVE,
    ]
    assert events[0].interval.pitch == 60
    assert events[1].interval.pitch == 61


def test_clear_reconcile_exits_active_interval_and_stop_resets_state():
    tape = build_tape(IntervalInsert(10, 30, 60, 90, 1))
    scheduler = PlaybackScheduler(tape)
    scheduler.start(15)
    tape.clear()

    event, = scheduler.reconcile(15)
    assert event.kind is PlaybackEventKind.EXIT_ACTIVE
    assert scheduler.active_interval is None

    scheduler.stop(15)
    assert not scheduler.running
    with pytest.raises(ValueError):
        scheduler.stop(-1)


def test_event_limit_error_does_not_move_scheduler_cursor():
    tape = build_tape(IntervalInsert(0, 1, 60, 90, 1), length=1)
    scheduler = PlaybackScheduler(tape)
    scheduler.start(0)

    with pytest.raises(PlaybackEventLimitError):
        scheduler.advance(50_001)
    assert scheduler.current_tick == 0
    assert scheduler.advance(1)[-1].tick == 1


def test_zero_length_scheduler_is_inert_and_phase_origin_shifts_loop_boundaries():
    empty_scheduler = PlaybackScheduler(Tape(0))
    assert empty_scheduler.start(0) == ()
    assert empty_scheduler.advance(100) == ()
    assert empty_scheduler.stop(100) == ()

    tape = build_tape(IntervalInsert(5, 10, 60, 90, 1), length=40)
    scheduler = PlaybackScheduler(tape, phase_origin_tick=100)
    assert scheduler.start(100) == ()
    assert [
        (event.kind, event.tick)
        for event in scheduler.advance(106)
    ] == [(PlaybackEventKind.ONSET, 105)]
    assert [
        (event.kind, event.tick)
        for event in scheduler.advance(116)
    ] == [(PlaybackEventKind.END, 115)]
