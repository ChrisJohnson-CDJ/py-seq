"""Pure MIDI note normalization state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class MidiInputEventType(Enum):
    NOTE_ON = auto()
    NOTE_OFF = auto()
    OTHER = auto()


@dataclass(frozen=True)
class MidiInputEvent:
    kind: MidiInputEventType
    channel: int | None = None
    pitch: int | None = None
    velocity: int | None = None
    timestamp_ns: int | None = None


class VoiceEventType(Enum):
    VOICE_START = auto()
    VOICE_CHANGE = auto()
    VOICE_END = auto()


@dataclass(frozen=True)
class VoiceEvent:
    kind: VoiceEventType
    gesture_id: int
    previous_pitch: int | None
    pitch: int | None
    velocity: int | None
    timestamp_ns: int | None = None


def _validate_integer(
    value: object,
    name: str,
    *,
    minimum: int,
    maximum: int,
    missing_value: bool = False,
) -> int:
    if value is None and missing_value:
        raise ValueError(f"{name} is required")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _validate_timestamp(timestamp_ns: object) -> int | None:
    if timestamp_ns is None:
        return None
    if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int):
        raise TypeError("timestamp_ns must be an integer")
    if timestamp_ns < 0:
        raise ValueError("timestamp_ns must be nonnegative")
    return timestamp_ns


class MidiVoicePreprocessor:
    """Select one held pitch using last-distinct-note priority."""

    def __init__(self, input_channel: int = 0) -> None:
        self.input_channel = _validate_integer(
            input_channel, "input_channel", minimum=0, maximum=15
        )
        self._held_notes: dict[int, int] = {}
        self._active_gesture_id: int | None = None
        self._next_gesture_id = 1

    def process(self, event: MidiInputEvent) -> VoiceEvent | None:
        if not isinstance(event, MidiInputEvent):
            raise TypeError("event must be a MidiInputEvent")
        if not isinstance(event.kind, MidiInputEventType):
            raise TypeError("event.kind must be a MidiInputEventType")

        if event.kind is MidiInputEventType.OTHER:
            _validate_timestamp(event.timestamp_ns)
            return None

        channel = _validate_integer(
            event.channel, "channel", minimum=0, maximum=15, missing_value=True
        )
        is_configured_channel = channel == self.input_channel

        pitch: int | None = None
        velocity: int | None = None
        if is_configured_channel:
            pitch = _validate_integer(
                event.pitch, "pitch", minimum=0, maximum=127, missing_value=True
            )
            velocity = _validate_integer(
                event.velocity,
                "velocity",
                minimum=1 if event.kind is MidiInputEventType.NOTE_ON else 0,
                maximum=127,
                missing_value=True,
            )
        timestamp_ns = _validate_timestamp(event.timestamp_ns)
        if not is_configured_channel:
            return None

        assert pitch is not None
        assert velocity is not None
        if event.kind is MidiInputEventType.NOTE_ON:
            return self._note_on(pitch, velocity, timestamp_ns)
        return self._note_off(pitch, timestamp_ns)

    def _note_on(
        self, pitch: int, velocity: int, timestamp_ns: int | None
    ) -> VoiceEvent | None:
        if pitch in self._held_notes:
            return None

        previous_pitch = self._selected_pitch()
        if previous_pitch is None:
            gesture_id = self._next_gesture_id
            self._next_gesture_id += 1
            self._active_gesture_id = gesture_id
            self._held_notes[pitch] = velocity
            return VoiceEvent(
                VoiceEventType.VOICE_START,
                gesture_id,
                None,
                pitch,
                velocity,
                timestamp_ns,
            )

        self._held_notes[pitch] = velocity
        return VoiceEvent(
            VoiceEventType.VOICE_CHANGE,
            self._require_active_gesture_id(),
            previous_pitch,
            pitch,
            velocity,
            timestamp_ns,
        )

    def _note_off(self, pitch: int, timestamp_ns: int | None) -> VoiceEvent | None:
        if pitch not in self._held_notes:
            return None

        previous_pitch = self._selected_pitch()
        del self._held_notes[pitch]
        if pitch != previous_pitch:
            return None

        selected_pitch = self._selected_pitch()
        gesture_id = self._require_active_gesture_id()
        if selected_pitch is None:
            self._active_gesture_id = None
            return VoiceEvent(
                VoiceEventType.VOICE_END,
                gesture_id,
                previous_pitch,
                None,
                None,
                timestamp_ns,
            )

        return VoiceEvent(
            VoiceEventType.VOICE_CHANGE,
            gesture_id,
            previous_pitch,
            selected_pitch,
            self._held_notes[selected_pitch],
            timestamp_ns,
        )

    def _selected_pitch(self) -> int | None:
        return next(reversed(self._held_notes), None)

    def _require_active_gesture_id(self) -> int:
        if self._active_gesture_id is None:
            raise RuntimeError("held notes exist without an active gesture")
        return self._active_gesture_id

    def reset(self) -> None:
        """Clear local held-note state without emitting a voice event."""
        self._held_notes.clear()
        self._active_gesture_id = None
