from __future__ import annotations

import re
from dataclasses import dataclass

from .four_channel_session import RecordingStartMode, RecordingStopMode, ResetMode


@dataclass(frozen=True)
class RecordingSyncCommand:
    start_mode: RecordingStartMode | None = None
    stop_mode: RecordingStopMode | None = None
    reference_channel: int | None = None


@dataclass(frozen=True)
class ResetCommand:
    mode: ResetMode
    interval_beats: int | None = None


def parse_reset_command(command: str) -> ResetCommand:
    if not isinstance(command, str):
        raise TypeError("command must be a string")
    parts = command.strip().lower().split()
    if parts == ["reset", "off"]:
        return ResetCommand(ResetMode.OFF)
    if parts == ["reset", "loop"]:
        return ResetCommand(ResetMode.LOOP)
    if len(parts) == 3 and parts[:2] == ["reset", "beat"]:
        if re.fullmatch(r"[0-9]+", parts[2]) is None:
            raise ValueError("use 'reset beat N' with a positive whole number")
        interval = int(parts[2])
        if interval < 1:
            raise ValueError("reset beat interval must be a positive integer")
        return ResetCommand(ResetMode.BEAT, interval)
    raise ValueError("use 'reset off', 'reset beat N', or 'reset loop'")


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

    if len(parts) != 2 or parts[0] not in ("start", "stop"):
        raise ValueError(
            "use 'start immediate|beat|loop|note', "
            "'stop immediate|beat|loop', or 'reference 1-4'"
        )

    start_modes = {
        "immediate": RecordingStartMode.IMMEDIATE,
        "beat": RecordingStartMode.NEXT_BEAT,
        "loop": RecordingStartMode.NEXT_REFERENCE_LOOP_START,
        "note": RecordingStartMode.FIRST_NOTE_ONSET,
    }
    stop_modes = {
        "immediate": RecordingStopMode.IMMEDIATE,
        "beat": RecordingStopMode.NEXT_BEAT,
        "loop": RecordingStopMode.NEXT_REFERENCE_LOOP_START,
    }
    if parts[0] == "start" and parts[1] in start_modes:
        return RecordingSyncCommand(start_mode=start_modes[parts[1]])
    if parts[0] == "stop" and parts[1] in stop_modes:
        return RecordingSyncCommand(stop_mode=stop_modes[parts[1]])
    choices = (
        "immediate|beat|loop|note"
        if parts[0] == "start"
        else "immediate|beat|loop"
    )
    raise ValueError(f"invalid {parts[0]} mode; choose {choices}")


__all__ = [
    "RecordingSyncCommand",
    "ResetCommand",
    "parse_recording_sync_command",
    "parse_reset_command",
]
