"""Mido message adapter for the pure MIDI preprocessor contract."""

from __future__ import annotations

from typing import Any

from .core import MidiInputEvent, MidiInputEventType


def _validate_raw_integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if value is None:
        raise ValueError(f"MIDI {name} is required")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"MIDI {name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"MIDI {name} must be between {minimum} and {maximum}")
    return value


def adapt_mido_message(
    message: Any, *, timestamp_ns: int | None = None
) -> MidiInputEvent:
    """Normalize a Mido message while preserving its supplied timestamp."""
    message_type = getattr(message, "type", None)
    if message_type not in ("note_on", "note_off"):
        return MidiInputEvent(MidiInputEventType.OTHER, timestamp_ns=timestamp_ns)

    channel = _validate_raw_integer(getattr(message, "channel", None), "channel", 0, 15)
    pitch = _validate_raw_integer(getattr(message, "note", None), "pitch", 0, 127)
    velocity = _validate_raw_integer(
        getattr(message, "velocity", None), "velocity", 0, 127
    )

    if message_type == "note_on" and velocity > 0:
        kind = MidiInputEventType.NOTE_ON
    else:
        kind = MidiInputEventType.NOTE_OFF

    return MidiInputEvent(kind, channel, pitch, velocity, timestamp_ns)
