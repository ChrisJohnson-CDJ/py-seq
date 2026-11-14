import mido
import pytest

from midi_output import CollectingMidiSink
from py_seq.midi_realtime import MidiOutputMode, MidiRealtimeOutput

_PULSE_NS = round(60_000_000_000 / (24 * 120))


def make_output():
    sink = CollectingMidiSink()
    return MidiRealtimeOutput((sink,), now_ns=0, bpm=120, current_tick=0), sink


def process(output, now_ns, *, allow_clock=True):
    advanced = []
    sent_clock = output.process_due(
        now_ns=now_ns,
        through_ns=now_ns,
        allow_clock=allow_clock,
        advance_before=advanced.append,
    )
    return sent_clock, advanced


def test_transport_modes_send_immediate_messages_and_ignore_app_changes_when_manual():
    output, sink = make_output()

    assert output.transport_mode is MidiOutputMode.OFF
    assert not output.transport_active
    output.set_transport_mode(
        MidiOutputMode.ON,
        now_ns=0,
        transport_running=False,
    )
    output.application_transport_changed(True, now_ns=0, current_tick=0)
    output.reset(transport_running=True)
    assert sink.messages == [mido.Message("start")]
    assert output.transport_active

    output.set_transport_mode(
        MidiOutputMode.OFF,
        now_ns=1,
        transport_running=True,
    )
    output.application_transport_changed(False, now_ns=2, current_tick=0)
    output.reset(transport_running=True)
    assert sink.messages == [mido.Message("start"), mido.Message("stop")]
    assert not output.transport_active


def test_transport_auto_follows_app_state_and_sends_reset_starts():
    output, sink = make_output()
    output.set_transport_mode(
        MidiOutputMode.AUTO,
        now_ns=0,
        transport_running=False,
    )
    assert sink.messages == [mido.Message("stop")]

    output.application_transport_changed(True, now_ns=0, current_tick=0)
    output.reset(transport_running=True)
    output.reset(transport_running=True)
    output.application_transport_changed(False, now_ns=1, current_tick=0)
    output.reset(transport_running=False)
    assert sink.messages == [
        mido.Message("stop"),
        mido.Message("start"),
        mido.Message("start"),
        mido.Message("start"),
        mido.Message("stop"),
    ]


def test_transport_auto_selection_while_running_waits_for_next_ppqn_edge():
    output, sink = make_output()
    output.application_transport_changed(True, now_ns=0, current_tick=0)
    output.set_transport_mode(
        MidiOutputMode.AUTO,
        now_ns=5_000_000,
        transport_running=True,
    )
    assert sink.messages == []
    assert output.transport_start_pending

    assert process(output, _PULSE_NS - 1) == (False, [])
    assert process(output, _PULSE_NS) == (False, [_PULSE_NS])
    assert sink.messages == [mido.Message("start")]
    assert output.transport_active
    assert not output.transport_start_pending


def test_superseding_transport_state_cancels_pending_start():
    output, sink = make_output()
    output.application_transport_changed(True, now_ns=0, current_tick=0)
    output.set_transport_mode(
        MidiOutputMode.AUTO,
        now_ns=1_000_000,
        transport_running=True,
    )
    output.application_transport_changed(False, now_ns=2_000_000, current_tick=0)
    process(output, 2 * _PULSE_NS)

    assert sink.messages == [mido.Message("stop")]
    assert not output.transport_start_pending


@pytest.mark.parametrize(
    ("mode", "running", "active"),
    [
        (MidiOutputMode.OFF, False, False),
        (MidiOutputMode.OFF, True, False),
        (MidiOutputMode.ON, False, True),
        (MidiOutputMode.ON, True, True),
        (MidiOutputMode.AUTO, False, False),
        (MidiOutputMode.AUTO, True, True),
    ],
)
def test_clock_modes_report_effective_state(mode, running, active):
    output, _ = make_output()
    output.set_clock_mode(
        mode,
        now_ns=0,
        transport_running=running,
        current_tick=0,
    )
    assert output.clock_mode is mode
    assert output.clock_active is active


def test_clock_on_continues_while_stopped_and_preserves_phase_on_transport_changes():
    output, sink = make_output()
    output.set_clock_mode(
        MidiOutputMode.ON,
        now_ns=0,
        transport_running=False,
        current_tick=0,
    )
    assert output.clock_active
    assert process(output, _PULSE_NS)[0]
    output.application_transport_changed(
        True, now_ns=25_000_000, current_tick=0
    )
    output.application_transport_changed(
        False, now_ns=30_000_000, current_tick=0
    )
    assert process(
        output, round(2 * 60_000_000_000 / (24 * 120))
    )[0]
    assert sink.messages == [mido.Message("clock"), mido.Message("clock")]


def test_clock_auto_follows_transport_and_realigns_to_musical_tick_on_start():
    output, sink = make_output()
    output.set_clock_mode(
        MidiOutputMode.AUTO,
        now_ns=0,
        transport_running=False,
        current_tick=0,
    )
    assert not output.clock_active
    assert not process(output, _PULSE_NS)[0]
    output.application_transport_changed(
        True, now_ns=30_000_000, current_tick=80
    )
    assert process(output, 30_000_000 + _PULSE_NS)[0]
    output.application_transport_changed(
        False, now_ns=60_000_000, current_tick=80
    )
    assert not output.clock_active
    assert not process(output, 2 * _PULSE_NS)[0]
    assert sink.messages == [mido.Message("clock")]


def test_tempo_change_preserves_clock_phase_and_missed_pulses_do_not_burst():
    output, sink = make_output()
    output.set_clock_mode(
        MidiOutputMode.ON,
        now_ns=0,
        transport_running=False,
        current_tick=0,
    )
    change_ns = _PULSE_NS // 2
    output.set_tempo(change_ns, 60)
    phase_at_change = change_ns * 120 * 24 / 60_000_000_000
    next_deadline_ns = change_ns + round(
        (1 - phase_at_change) * 60_000_000_000 / (24 * 60)
    )
    assert not process(output, next_deadline_ns - 1)[0]
    assert process(output, next_deadline_ns)[0]

    output, sink = make_output()
    output.set_clock_mode(
        MidiOutputMode.ON,
        now_ns=0,
        transport_running=False,
        current_tick=0,
    )
    assert process(output, 60_000_000)[0]
    assert not process(output, 60_000_000)[0]
    assert len(sink.messages) == 1
    assert process(output, 65_000_000)[0]
    assert len(sink.messages) == 2


def test_clock_and_scheduled_transport_start_have_deterministic_order():
    output, sink = make_output()
    output.application_transport_changed(True, now_ns=0, current_tick=0)
    output.set_clock_mode(
        MidiOutputMode.AUTO,
        now_ns=0,
        transport_running=True,
        current_tick=0,
    )
    output.set_transport_mode(
        MidiOutputMode.AUTO,
        now_ns=0,
        transport_running=True,
    )
    advanced = []
    output.process_due(
        now_ns=_PULSE_NS,
        through_ns=_PULSE_NS,
        advance_before=advanced.append,
    )

    assert sink.messages == [mido.Message("clock"), mido.Message("start")]
    assert advanced == [_PULSE_NS]


def test_all_nine_transport_and_clock_mode_pairs_are_independent():
    for transport_mode in MidiOutputMode:
        for clock_mode in MidiOutputMode:
            output, _ = make_output()
            output.set_transport_mode(
                transport_mode,
                now_ns=0,
                transport_running=False,
            )
            output.set_clock_mode(
                clock_mode,
                now_ns=0,
                transport_running=False,
                current_tick=0,
            )
            assert output.transport_mode is transport_mode
            assert output.clock_mode is clock_mode
