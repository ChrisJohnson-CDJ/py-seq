"""Mido-compatible output adapters and sinks."""

from .core import (
    CollectingMidiSink,
    MidiMessageSink,
    MidiOutputAdapter,
    MidiOutputError,
    MidoOutputPortSink,
)

__all__ = [
    "CollectingMidiSink",
    "MidiMessageSink",
    "MidiOutputAdapter",
    "MidiOutputError",
    "MidoOutputPortSink",
]
