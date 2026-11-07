"""Semantic note-action policy for one logical output voice."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from playback_scheduler import PlaybackEvent, PlaybackEventKind
from tape import VoiceInterval


class NoteActionKind(Enum):
    NOTE_ON = auto()
    NOTE_OFF = auto()


@dataclass(frozen=True)
class NoteAction:
    kind: NoteActionKind
    pitch: int
    velocity: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, NoteActionKind):
            raise TypeError("kind must be a NoteActionKind")
        _midi_integer(self.pitch, "pitch", 0, 127)
        _midi_integer(self.velocity, "velocity", 0, 127)
        if self.kind is NoteActionKind.NOTE_ON and self.velocity == 0:
            raise ValueError("NOTE_ON velocity must be between 1 and 127")
        if self.kind is NoteActionKind.NOTE_OFF and self.velocity != 0:
            raise ValueError("NOTE_OFF velocity must be zero")


@dataclass(frozen=True)
class LiveVoice:
    pitch: int
    velocity: int

    def __post_init__(self) -> None:
        _midi_integer(self.pitch, "pitch", 0, 127)
        _midi_integer(self.velocity, "velocity", 1, 127)


@dataclass(frozen=True)
class _SoundingVoice:
    pitch: int
    velocity: int
    source: str


def _midi_integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _same_interval(first: VoiceInterval, second: VoiceInterval) -> bool:
    return (
        first.start,
        first.duration,
        first.pitch,
        first.velocity,
    ) == (
        second.start,
        second.duration,
        second.pitch,
        second.velocity,
    )


class Articulator:
    """Translate playback occupancy events and live priority into note actions."""

    def __init__(self) -> None:
        self._playback_interval: VoiceInterval | None = None
        self._playback_eligible = False
        self._live_voice: LiveVoice | None = None
        self._sounding: _SoundingVoice | None = None

    @property
    def active_pitch(self) -> int | None:
        return None if self._sounding is None else self._sounding.pitch

    @property
    def active_source(self) -> str | None:
        return None if self._sounding is None else self._sounding.source

    @property
    def live_voice(self) -> LiveVoice | None:
        return self._live_voice

    @property
    def playback_interval(self) -> VoiceInterval | None:
        return self._playback_interval

    def handle(self, event: PlaybackEvent) -> tuple[NoteAction, ...]:
        if not isinstance(event, PlaybackEvent):
            raise TypeError("event must be a PlaybackEvent")

        if event.kind is PlaybackEventKind.ONSET:
            self._playback_interval = event.interval
            self._playback_eligible = True
            if self._live_voice is not None:
                return ()
            return tuple(self._sound_playback(retrigger=True))

        if event.kind is PlaybackEventKind.ENTER_ACTIVE:
            if self._playback_interval is not None and self._sounding is not None:
                if (
                    self._sounding.source == "playback"
                    and not _same_interval(self._playback_interval, event.interval)
                ):
                    actions = self._release_sounding()
                else:
                    actions = []
            else:
                actions = []
            self._playback_interval = event.interval
            self._playback_eligible = False
            return tuple(actions)

        if event.kind is PlaybackEventKind.END:
            if not self._matches_playback(event.interval):
                return ()
            self._playback_interval = None
            self._playback_eligible = False
            if self._sounding is not None and self._sounding.source == "playback":
                return tuple(self._release_sounding())
            return ()

        if not self._matches_playback(event.interval):
            return ()
        self._playback_interval = None
        self._playback_eligible = False
        if self._sounding is not None and self._sounding.source == "playback":
            return tuple(self._release_sounding())
        return ()

    def set_live_voice(
        self, voice: LiveVoice | None
    ) -> tuple[NoteAction, ...]:
        if voice is not None and not isinstance(voice, LiveVoice):
            raise TypeError("voice must be a LiveVoice or None")
        if voice == self._live_voice:
            return ()

        actions = self._release_sounding()
        self._live_voice = voice
        if voice is None:
            self._playback_eligible = False
            return tuple(actions)

        self._sounding = _SoundingVoice(voice.pitch, voice.velocity, "live")
        actions.append(NoteAction(NoteActionKind.NOTE_ON, voice.pitch, voice.velocity))
        return tuple(actions)

    def silence(self) -> tuple[NoteAction, ...]:
        actions = self._release_sounding()
        self._live_voice = None
        self._playback_interval = None
        self._playback_eligible = False
        return tuple(actions)

    def reset_playback(self) -> tuple[NoteAction, ...]:
        """Release playback articulation while preserving any live input voice."""
        actions = (
            self._release_sounding()
            if self._sounding is not None and self._sounding.source == "playback"
            else []
        )
        self._playback_interval = None
        self._playback_eligible = False
        return tuple(actions)

    def forget_output_state(self) -> None:
        """Invalidate local assumptions after a device send failure or panic."""
        self._sounding = None
        self._live_voice = None
        self._playback_interval = None
        self._playback_eligible = False

    def _matches_playback(self, interval: VoiceInterval) -> bool:
        return (
            self._playback_interval is not None
            and _same_interval(self._playback_interval, interval)
        )

    def _sound_playback(self, *, retrigger: bool) -> list[NoteAction]:
        interval = self._playback_interval
        if (
            interval is None
            or not self._playback_eligible
            or self._live_voice is not None
        ):
            return []

        actions: list[NoteAction] = []
        if self._sounding is not None:
            if not retrigger and (
                self._sounding.source == "playback"
                and self._sounding.pitch == interval.pitch
                and self._sounding.velocity == interval.velocity
            ):
                return []
            actions.extend(self._release_sounding())
        self._sounding = _SoundingVoice(interval.pitch, interval.velocity, "playback")
        actions.append(
            NoteAction(NoteActionKind.NOTE_ON, interval.pitch, interval.velocity)
        )
        return actions

    def _release_sounding(self) -> list[NoteAction]:
        sounding = self._sounding
        self._sounding = None
        if sounding is None:
            return []
        return [NoteAction(NoteActionKind.NOTE_OFF, sounding.pitch)]
