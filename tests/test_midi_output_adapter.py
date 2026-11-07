from unittest.mock import MagicMock, patch

import mido
import pytest

from articulator import NoteAction, NoteActionKind
from midi_output import (
    CollectingMidiSink,
    MidiOutputAdapter,
    MidiOutputError,
    MidoOutputPortSink,
)


def test_adapter_maps_actions_in_order_to_zero_based_midi_channel():
    sink = CollectingMidiSink()
    adapter = MidiOutputAdapter(sink, 3)

    adapter.send_actions(
        (
            NoteAction(NoteActionKind.NOTE_OFF, 60),
            NoteAction(NoteActionKind.NOTE_ON, 64, 88),
        )
    )

    assert sink.messages == [
        mido.Message("note_off", channel=3, note=60, velocity=0),
        mido.Message("note_on", channel=3, note=64, velocity=88),
    ]


@pytest.mark.parametrize("channel", [-1, 16, True])
def test_adapter_rejects_invalid_channel(channel):
    with pytest.raises((TypeError, ValueError)):
        MidiOutputAdapter(CollectingMidiSink(), channel)


def test_collecting_sink_copies_messages_and_port_sink_closes_port():
    sink = CollectingMidiSink()
    message = mido.Message("note_on", note=60, velocity=80)
    sink.send(message)
    message.note = 62
    assert getattr(sink.messages[0], "note") == 60

    port = MagicMock()
    with patch("midi_output.core.mido.open_output", return_value=port):
        output = MidoOutputPortSink("test-port")
    output.send(mido.Message("note_on", note=60, velocity=80))
    output.close()
    port.send.assert_called_once()
    port.close.assert_called_once()


def test_partial_send_failure_desynchronizes_until_explicit_panic():
    sink = MagicMock()
    sink.send.side_effect = [None, OSError("device lost"), None]
    adapter = MidiOutputAdapter(sink, 0)

    with pytest.raises(MidiOutputError) as raised:
        adapter.send_actions(
            (
                NoteAction(NoteActionKind.NOTE_ON, 60, 90),
                NoteAction(NoteActionKind.NOTE_ON, 62, 90),
            )
        )
    assert raised.value.sent_count == 1
    assert adapter.desynchronized
    with pytest.raises(MidiOutputError, match="desynchronized"):
        adapter.send_actions((NoteAction(NoteActionKind.NOTE_OFF, 60),))

    adapter.panic()
    assert not adapter.desynchronized
    panic_message = sink.send.call_args_list[-1].args[0]
    assert panic_message.type == "control_change"
    assert panic_message.control == 123
