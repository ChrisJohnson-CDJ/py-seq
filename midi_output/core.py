"""Mido adapter and sinks for semantic note actions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, cast

import mido

from articulator import NoteAction, NoteActionKind


class MidiMessageSink(Protocol):
    def send(self, message: mido.Message) -> None: ...


class _OutputPort(Protocol):
    def send(self, message: mido.Message) -> None: ...

    def close(self) -> None: ...


class MidiOutputError(RuntimeError):
    """A MIDI message could not be sent; the adapter now requires recovery."""

    def __init__(self, message: str, *, sent_count: int = 0) -> None:
        super().__init__(message)
        self.sent_count = sent_count


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


class CollectingMidiSink:
    """In-memory MIDI sink useful for tests and diagnostics."""

    def __init__(self) -> None:
        self.messages: list[mido.Message] = []

    def send(self, message: mido.Message) -> None:
        if not isinstance(message, mido.Message):
            raise TypeError("message must be a mido.Message")
        self.messages.append(message.copy())


class MidoOutputPortSink:
    """Open and wrap a Mido output port."""

    def __init__(self, port_name: str) -> None:
        if not isinstance(port_name, str) or not port_name:
            raise ValueError("port_name must be a non-empty string")
        open_port = getattr(mido, "open_output", None)
        if not callable(open_port):
            raise RuntimeError("Mido does not provide an output-port factory")
        self.port = cast(_OutputPort, open_port(port_name))

    def send(self, message: mido.Message) -> None:
        self.port.send(message)

    def close(self) -> None:
        self.port.close()


class MidiOutputAdapter:
    """Convert ordered semantic note actions to channel-specific Mido messages."""

    def __init__(self, sink: MidiMessageSink, channel: int) -> None:
        if not callable(getattr(sink, "send", None)):
            raise TypeError("sink must provide send(message)")
        self.sink = sink
        self.channel = _integer(channel, "channel", 0, 15)
        self._desynchronized = False

    @property
    def desynchronized(self) -> bool:
        return self._desynchronized

    def send_actions(self, actions: Iterable[NoteAction]) -> None:
        validated = tuple(self._message_for(action) for action in actions)
        if not validated:
            return
        if self._desynchronized:
            raise MidiOutputError(
                "MIDI output is desynchronized; call panic() before sending more actions"
            )

        for index, message in enumerate(validated):
            try:
                self.sink.send(message)
            except Exception as error:
                self._desynchronized = True
                raise MidiOutputError(
                    f"MIDI send failed after {index} message(s)",
                    sent_count=index,
                ) from error

    def panic(self) -> None:
        message = mido.Message(
            "control_change",
            channel=self.channel,
            control=123,
            value=0,
        )
        try:
            self.sink.send(message)
        except Exception as error:
            self._desynchronized = True
            raise MidiOutputError("MIDI panic failed") from error
        self._desynchronized = False

    def _message_for(self, action: NoteAction) -> mido.Message:
        if not isinstance(action, NoteAction):
            raise TypeError("actions must contain only NoteAction values")
        note = _integer(action.pitch, "pitch", 0, 127)
        velocity = _integer(action.velocity, "velocity", 0, 127)
        if action.kind is NoteActionKind.NOTE_ON:
            if velocity == 0:
                raise ValueError("NOTE_ON velocity must be between 1 and 127")
            return mido.Message(
                "note_on",
                channel=self.channel,
                note=note,
                velocity=velocity,
            )
        if action.kind is NoteActionKind.NOTE_OFF:
            if velocity != 0:
                raise ValueError("NOTE_OFF velocity must be zero")
            return mido.Message(
                "note_off",
                channel=self.channel,
                note=note,
                velocity=0,
            )
        raise ValueError(f"unsupported note action kind: {action.kind!r}")
