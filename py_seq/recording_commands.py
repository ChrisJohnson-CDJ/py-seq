from __future__ import annotations

import re
from dataclasses import dataclass

from .four_channel_session import (
    TICKS_PER_BEAT,
    RecordingStartMode,
    RecordingStopMode,
    ResetMode,
)

_MUSICAL_LENGTH = re.compile(r"([0-9]+)(?:\+([0-9]+)(?:/([0-9]+))?)?\Z")
_SUPPORTED_LENGTH_DENOMINATORS = {2, 4, 8, 16, 32, 64}


@dataclass(frozen=True)
class RecordingSyncCommand:
    start_mode: RecordingStartMode | None = None
    stop_mode: RecordingStopMode | None = None
    reference_channel: int | None = None
    start_beat_count: int | None = None
    stop_beat_count: int | None = None
    start_gesture_count: int | None = None
    stop_gesture_count: int | None = None
    stop_after_ticks: int | None = None


@dataclass(frozen=True)
class ResetCommand:
    mode: ResetMode
    interval_beats: float | None = None

    @property
    def interval_ticks(self) -> int | None:
        return (
            None
            if self.interval_beats is None
            else int(self.interval_beats * TICKS_PER_BEAT)
        )


def parse_musical_length(
    expression: str,
    *,
    allow_exact_binary_denominators: bool = False,
) -> int:
    if not isinstance(expression, str):
        raise TypeError("musical length must be a string")
    match = _MUSICAL_LENGTH.fullmatch(expression)
    if match is None:
        raise ValueError("use {beats}[+{n}[/{m}]] for a musical length")

    beats = int(match.group(1))
    numerator = int(match.group(2) or 0)
    denominator = int(match.group(3) or 16)
    if match.group(3) is not None and denominator not in _SUPPORTED_LENGTH_DENOMINATORS:
        if not allow_exact_binary_denominators:
            raise ValueError("denominator must be one of 2, 4, 8, 16, 32, or 64")
        if denominator < 2 or denominator & (denominator - 1):
            raise ValueError("denominator must be a power of two greater than one")
    if numerator >= denominator:
        raise ValueError("fraction numerator must be less than its denominator")

    fractional_ticks, remainder = divmod(numerator * TICKS_PER_BEAT, denominator)
    if remainder:
        raise ValueError("fractional beat is not exactly representable in ticks")
    return beats * TICKS_PER_BEAT + fractional_ticks


def parse_length_command(command: str) -> int:
    if not isinstance(command, str):
        raise TypeError("command must be a string")
    parts = command.strip().lower().split()
    if len(parts) != 2 or parts[0] != "length":
        raise ValueError("use 'length {beats}[+{n}[/{m}]]'")
    length_ticks = parse_musical_length(parts[1])
    if length_ticks <= 0:
        raise ValueError("loop length must be greater than zero")
    return length_ticks


def parse_reset_command(command: str) -> ResetCommand:
    if not isinstance(command, str):
        raise TypeError("command must be a string")
    parts = command.strip().lower().split()
    if parts == ["reset", "off"]:
        return ResetCommand(ResetMode.OFF)
    if parts == ["reset", "loop"]:
        return ResetCommand(ResetMode.LOOP)
    if parts[:2] == ["reset", "beat"] and len(parts) in (2, 3):
        interval = (
            TICKS_PER_BEAT
            if len(parts) == 2
            else parse_musical_length(
                parts[2], allow_exact_binary_denominators=True
            )
        )
        if interval <= 0:
            raise ValueError("reset beat interval must be greater than zero")
        return ResetCommand(ResetMode.BEAT, interval / TICKS_PER_BEAT)
    raise ValueError("use 'reset off', 'reset beat [length]', or 'reset loop'")


def parse_recording_sync_command(command: str) -> RecordingSyncCommand:
    if not isinstance(command, str):
        raise TypeError("command must be a string")
    parts = command.strip().lower().split()
    if len(parts) == 2 and parts[0] == "reference":
        try:
            channel = int(parts[1])
        except ValueError as error:
            raise ValueError("use 'reference 1' through 'reference 4'") from error
        if not 1 <= channel <= 4:
            raise ValueError("reference channel must be between 1 and 4")
        return RecordingSyncCommand(reference_channel=channel - 1)

    if parts[:2] == ["stop", "after"] and len(parts) == 3:
        duration_ticks = parse_musical_length(parts[2])
        if duration_ticks <= 0:
            raise ValueError("stop duration must be greater than zero")
        return RecordingSyncCommand(
            stop_mode=RecordingStopMode.AFTER_DURATION,
            stop_after_ticks=duration_ticks,
        )

    if parts[0:2] == ["stop", "beat"] and len(parts) == 3:
        raise ValueError("use 'stop beat' or 'stop after {beats}[+{n}[/{m}]]'")

    if (
        len(parts) not in (2, 3)
        or parts[0] not in ("start", "stop")
        or (len(parts) == 3 and parts[1] not in ("beat", "gesture"))
    ):
        raise ValueError(
            "use 'start immediate|beat [N]|gesture [N]|loop', "
            "'stop immediate|beat|after {beats}[+{n}[/{m}]]|gesture [N]|loop', "
            "or 'reference 1-4'"
        )

    start_modes = {
        "immediate": RecordingStartMode.IMMEDIATE,
        "beat": RecordingStartMode.NEXT_BEAT,
        "gesture": RecordingStartMode.NEXT_GESTURE,
        "loop": RecordingStartMode.NEXT_REFERENCE_LOOP_START,
    }
    stop_modes = {
        "immediate": RecordingStopMode.IMMEDIATE,
        "beat": RecordingStopMode.NEXT_BEAT,
        "gesture": RecordingStopMode.NEXT_GESTURE,
        "loop": RecordingStopMode.NEXT_REFERENCE_LOOP_START,
    }
    beat_count = None
    gesture_count = 1 if parts[1] == "gesture" else None
    if len(parts) == 3:
        if re.fullmatch(r"[0-9]+", parts[2]) is None or int(parts[2]) < 1:
            raise ValueError("synchronization count must be a positive whole number")
        count = int(parts[2])
        if parts[1] == "beat":
            beat_count = count
        else:
            gesture_count = count
    if parts[0] == "start" and parts[1] in start_modes:
        return RecordingSyncCommand(
            start_mode=start_modes[parts[1]],
            start_beat_count=beat_count,
            start_gesture_count=gesture_count,
        )
    if parts[0] == "stop" and parts[1] in stop_modes:
        return RecordingSyncCommand(
            stop_mode=stop_modes[parts[1]],
            stop_beat_count=beat_count if parts[1] == "beat" else None,
            stop_gesture_count=gesture_count,
        )
    choices = (
        "immediate|beat [N]|gesture [N]|loop"
        if parts[0] == "start"
        else "immediate|beat|after {beats}[+{n}[/{m}]]|gesture [N]|loop"
    )
    raise ValueError(f"invalid {parts[0]} mode; choose {choices}")


__all__ = [
    "parse_length_command",
    "parse_musical_length",
    "RecordingSyncCommand",
    "ResetCommand",
    "parse_recording_sync_command",
    "parse_reset_command",
]
