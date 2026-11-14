"""Optional Pygame front end for the four-channel host runtime."""

from __future__ import annotations

import argparse
import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import ceil, floor, isfinite
from typing import Any, cast

import mido

from midi_output import CollectingMidiSink, MidoOutputPortSink

from .four_channel_session import (
    CHANNEL_COUNT,
    MAX_BPM,
    TICKS_PER_BEAT,
    FourChannelSession,
    MidiRoute,
)
from .recording_commands import (
    parse_bpm_command,
    parse_length_command,
    parse_midi_output_command,
    parse_midi_config_command,
    parse_recording_sync_command,
    parse_reset_command,
    resolve_keyword,
)
from .midi_config import MidiConfig, UnassignedMidiSink, config_path, load_config
from .gui_runtime import (
    FourChannelGuiRuntime,
    GuiChannelSnapshot,
    GuiCommandQueueFull,
    GuiHostUnavailable,
    GuiIntentKind,
    GuiNotification,
    GuiSnapshot,
)

WINDOW_SIZE = (1024, 600)
VISIBLE_BEATS = 16.0
PLAYHEAD_POSITION = 0.35
FRAME_RATE = 15
BACKGROUND = (11, 15, 20)
PANEL = (18, 26, 33)
GRID = (32, 48, 60)
PRIMARY = (230, 238, 244)
SECONDARY = (140, 161, 178)
PLAY = (94, 201, 125)
PLAYHEAD = (246, 248, 250)
LOOP_BOUNDARY = (224, 108, 117)
CHANNEL_COLORS = (
    (109, 182, 230),
    (212, 161, 94),
    (143, 193, 126),
    (183, 154, 227),
)
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_PROGRAM_COMMAND = re.compile(r"[pP](\d{3})(?::(\d{3}))?\Z")
_RECORDING_MODE_LABELS = {
    "IMMEDIATE": "NOW",
    "NEXT_BEAT": "BEAT",
    "AFTER_DURATION": "AFTER",
    "NEXT_GESTURE": "GESTURE",
    "NEXT_REFERENCE_LOOP_START": "LOOP",
    "MIDI_TRANSPORT": "TRANSPORT",
}
logger = logging.getLogger(__name__)


class GuiCommandError(ValueError):
    """A local command failed syntax or range validation."""


@dataclass(frozen=True, slots=True)
class ParsedProgramCommand:
    bank_msb: int | None
    program: int


class CommandHistory:
    def __init__(self) -> None:
        self._commands: list[str] = []
        self._pending: dict[int, str] = {}
        self._position: int | None = None
        self._draft = ""

    @property
    def commands(self) -> tuple[str, ...]:
        return tuple(self._commands)

    def reset_navigation(self) -> None:
        self._position = None
        self._draft = ""

    def add(self, command: str) -> None:
        if not command.strip():
            return
        if self._commands and self._commands[-1] == command:
            return
        self._commands.append(command)
        if len(self._commands) > 50:
            del self._commands[0]

    def submitted(self, request_id: int, command: str) -> None:
        self._pending[request_id] = command

    def applied(self, request_id: int | None) -> None:
        if request_id is not None:
            command = self._pending.pop(request_id, None)
            if command is not None:
                self.add(command)

    def rejected(self, request_id: int | None) -> None:
        if request_id is not None:
            self._pending.pop(request_id, None)

    def previous(self, current_text: str) -> str:
        if not self._commands:
            return current_text
        if self._position is None:
            self._draft = current_text
            self._position = len(self._commands)
        self._position = max(0, self._position - 1)
        return self._commands[self._position]

    def next(self, current_text: str) -> str:
        if self._position is None:
            return current_text
        if self._position < len(self._commands) - 1:
            self._position += 1
            return self._commands[self._position]
        self._position = None
        return self._draft


class MidiDeviceCompletion:
    def __init__(self) -> None:
        self._last = ""
        self._names: list[str] = []
        self._index = -1

    def complete(self, text: str) -> tuple[str, str | None]:
        match = re.fullmatch(r"\s*(\S+)\s+(\S+)\s+(\d+)(?:\s+.*)?", text)
        if match is None:
            return text, None
        try:
            keyword = resolve_keyword(
                match.group(1).lower(),
                (
                    "reset", "length", "start", "stop", "reference",
                    "transport", "clock", "bpm", "tempo", "midi",
                ),
            )
            if keyword != "midi":
                return text, None
            target = resolve_keyword(match.group(2).lower(), ("in", "out", "label"))
        except ValueError:
            return text, None
        if target not in ("in", "out") or not 1 <= int(match.group(3)) <= 16:
            return text, None
        if text != self._last:
            self._names = _get_port_names(
                "get_input_names" if target == "in" else "get_output_names"
            )
            self._index = -1
        if not self._names:
            return text, f"No MIDI {target} devices available"
        self._index = (self._index + 1) % len(self._names)
        self._last = (
            f"{match.group(1)} {match.group(2)} {match.group(3)} "
            f"{self._names[self._index]}"
        )
        return self._last, None


def parse_program_command(command: str) -> ParsedProgramCommand:
    match = _PROGRAM_COMMAND.fullmatch(command.strip())
    if match is None:
        raise GuiCommandError("Invalid command; use pNNN or pNNN:NNN")
    values = tuple(int(value) for value in match.groups() if value is not None)
    if any(value > 127 for value in values):
        raise GuiCommandError("Bank and program must be between 000 and 127")
    if len(values) == 1:
        return ParsedProgramCommand(None, values[0])
    return ParsedProgramCommand(values[0], values[1])


def interpolated_position_beats(snapshot: GuiSnapshot, now_ns: int) -> float:
    transport = snapshot.transport
    position = transport.position_ticks / transport.ticks_per_beat
    if transport.playing:
        elapsed_ns = max(0, now_ns - transport.sampled_at_ns)
        position += elapsed_ns * transport.tempo_bpm / 60_000_000_000
    return position


def _note_name(pitch: int) -> str:
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def _format_beat_position(position_beats: float) -> str:
    subdivisions = floor(position_beats * 64 + 0.5)
    beats, fraction = divmod(subdivisions, 64)
    return f"{beats}:{fraction:02d}"


def _loop_label(position_beats: float, loop_beats: float) -> str:
    if loop_beats <= 0:
        return "--:-- ( --%)"
    phase = (position_beats % loop_beats) if loop_beats else 0.0
    percentage = min(99, max(0, int(phase / loop_beats * 100)))
    return f"{_format_beat_position(phase)} ({percentage:3d}%)"


def _footer_lines(
    snapshot: GuiSnapshot,
    display_beats: float,
    command_mode: bool,
    command_text: str,
    status_text: str,
) -> tuple[str, str]:
    reference = snapshot.recording_reference_channel_id or "-"
    start_mode = _RECORDING_MODE_LABELS.get(
        snapshot.recording_start_mode, snapshot.recording_start_mode
    )
    stop_mode = _RECORDING_MODE_LABELS.get(
        snapshot.recording_stop_mode, snapshot.recording_stop_mode
    )
    if snapshot.recording_start_beat_count is not None:
        start_mode = f"{start_mode} {snapshot.recording_start_beat_count}"
    if snapshot.recording_stop_after_ticks is not None:
        stop_mode = (
            f"{stop_mode} "
            f"{snapshot.recording_stop_after_ticks / TICKS_PER_BEAT:g}"
        )
    if snapshot.recording_start_gesture_count is not None:
        start_mode = f"{start_mode} {snapshot.recording_start_gesture_count}"
    if snapshot.recording_stop_gesture_count is not None:
        stop_mode = f"{stop_mode} {snapshot.recording_stop_gesture_count}"
    if snapshot.recording_stop_gesture_cycle:
        stop_mode = f"{stop_mode} CYCLE"
    configuration = (
        f"BEAT {_format_beat_position(display_beats)}  REC {snapshot.recording_state}  "
        f"START {start_mode}  STOP {stop_mode}  REF {reference}  "
        f"RESET {_reset_mode_label(snapshot)}  "
        f"MIDI TRN {snapshot.midi_transport_mode}/"
        f"{'PENDING' if snapshot.midi_transport_start_pending else 'ON' if snapshot.midi_transport_active else 'OFF'}  "
        f"CLK {snapshot.midi_clock_mode}/"
        f"{'ON' if snapshot.midi_clock_active else 'OFF'}"
    )
    status = f"> {command_text}" if command_mode else status_text
    return configuration, status


def _reset_mode_label(snapshot: GuiSnapshot) -> str:
    if snapshot.reset_mode == "BEAT":
        if snapshot.reset_interval_beats is None:
            return "BEAT"
        return f"BEAT {snapshot.reset_interval_beats:g}"
    return snapshot.reset_mode


def _nonnegative_index(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from error
    if result < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return result


def _positive_bpm(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    if result > MAX_BPM:
        raise argparse.ArgumentTypeError(f"must not exceed {MAX_BPM} BPM")
    return result


def _midi_channel(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer from 0 to 15") from error
    if not 0 <= result <= 15:
        raise argparse.ArgumentTypeError("must be an integer from 0 to 15")
    return result


def _loop_length_ticks(value: str) -> int:
    try:
        beats = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a positive whole number of beats"
        ) from error
    if beats <= 0:
        raise argparse.ArgumentTypeError("must be a positive whole number of beats")
    return beats * TICKS_PER_BEAT


def _route_spec(values: Sequence[str]) -> tuple[int, int, int, str]:
    channel_text, port_text, midi_channel_text, label = values
    try:
        channel = int(channel_text)
        port_index = int(port_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "loop channel and output index must be integers"
        ) from error
    if not 1 <= channel <= CHANNEL_COUNT:
        raise argparse.ArgumentTypeError("loop channel must be between 1 and 4")
    if port_index < 0:
        raise argparse.ArgumentTypeError("output index must be nonnegative")
    midi_channel = _midi_channel(midi_channel_text)
    if not label or len(label) > 8:
        raise argparse.ArgumentTypeError("label must be between 1 and 8 characters")
    return channel, port_index, midi_channel, label


class _MidiOutAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        if values is None or isinstance(values, str):
            parser.error(f"{option_string}: expected four route values")
        try:
            route = _route_spec([str(value) for value in values])
        except argparse.ArgumentTypeError as error:
            parser.error(f"{option_string}: {error}")
        routes = getattr(namespace, self.dest, None)
        if routes is None:
            routes = []
        elif any(existing[0] == route[0] for existing in routes):
            parser.error("--midi-out may specify each loop channel only once")
        routes.append(route)
        setattr(namespace, self.dest, routes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Four-channel MIDI tape looper with a Pygame interface."
    )
    parser.add_argument("--list-midi", action="store_true", help="list indexed MIDO ports")
    parser.add_argument("--midi-in", type=_nonnegative_index, help="MIDI input port index")
    parser.add_argument(
        "--midi-out",
        nargs=4,
        action=_MidiOutAction,
        metavar=("CHANNEL", "INDEX", "MIDI_CHANNEL", "LABEL"),
        help="route loop channel 1-4 to an output index, MIDI channel 0-15, and label",
    )
    parser.add_argument(
        "--length",
        nargs=CHANNEL_COUNT,
        type=_loop_length_ticks,
        default=[0] * CHANNEL_COUNT,
        metavar=("CH1", "CH2", "CH3", "CH4"),
        help="initial loop lengths in beats (default: define while recording)",
    )
    parser.add_argument(
        "--bpm", type=_positive_bpm, default=10.0,
        help=f"tempo in BPM (greater than 0 and at most {MAX_BPM})",
    )
    parser.add_argument(
        "--fps",
        type=positive_int,
        default=FRAME_RATE,
        help="maximum render frame rate (default: 15)",
    )
    return parser


def _get_port_names(factory_name: str) -> list[str]:
    factory = getattr(mido, factory_name, None)
    if not callable(factory):
        raise RuntimeError(f"Mido does not provide {factory_name}")
    return cast(Callable[[], list[str]], factory)()


def list_midi_ports() -> bool:
    succeeded = True
    for label, factory_name in (
        ("MIDI input ports", "get_input_names"),
        ("MIDI output ports", "get_output_names"),
    ):
        print(f"{label}:")
        try:
            names = _get_port_names(factory_name)
        except Exception:
            logger.exception("could not enumerate %s", label.lower())
            succeeded = False
            continue
        if not names:
            print("  (none)")
        for index, name in enumerate(names):
            print(f"  {index}: {name}")
    return succeeded


def _open_outputs(
    args: argparse.Namespace,
    output_sinks: list[MidoOutputPortSink],
) -> tuple[MidiRoute, MidiRoute, MidiRoute, MidiRoute]:
    names = _get_port_names("get_output_names")
    sinks_by_index: dict[int, MidoOutputPortSink] = {}
    routes: list[MidiRoute] = [
        MidiRoute("NO OUT", 0, CollectingMidiSink()) for _ in range(CHANNEL_COUNT)
    ]
    for channel, index, midi_channel, label in args.midi_out or []:
        if index >= len(names):
            raise ValueError(
                f"MIDI output index {index} is unavailable; "
                f"current output count is {len(names)}"
            )
        sink = sinks_by_index.get(index)
        if sink is None:
            sink = MidoOutputPortSink(names[index])
            sinks_by_index[index] = sink
            output_sinks.append(sink)
        routes[channel - 1] = MidiRoute(label, midi_channel, sink, index)
    return routes[0], routes[1], routes[2], routes[3]


def _draw_text(surface, text, font, color, position) -> None:
    surface.blit(font.render(text, True, color), position)


def _pitch_to_y(pitch: int, timeline) -> int:
    pitch_fraction = min(1.0, max(0.0, (pitch - 24) / 60))
    return timeline.bottom - 8 - round(pitch_fraction * (timeline.height - 16))


def _beat_to_x(beat: float, view_start: float, timeline) -> int:
    pixels_per_beat = timeline.width / VISIBLE_BEATS
    return timeline.x + round((beat - view_start) * pixels_per_beat)


def _draw_note_lane(
    pygame,
    surface,
    channel: GuiChannelSnapshot,
    color: tuple[int, int, int],
    timeline,
    display_beats: float,
    recent_occurrences: set[tuple[str, int, int]],
    note_font,
) -> None:
    loop_beats = channel.loop_length_ticks / TICKS_PER_BEAT
    phase_beats = channel.phase_origin_tick / TICKS_PER_BEAT
    view_start = display_beats - VISIBLE_BEATS * PLAYHEAD_POSITION
    pixels_per_beat = timeline.width / VISIBLE_BEATS
    previous_clip = surface.get_clip()
    try:
        first_grid = floor(view_start)
        last_grid = ceil(view_start + VISIBLE_BEATS)
        for beat in range(first_grid, last_grid + 1):
            x = _beat_to_x(beat, view_start, timeline)
            if timeline.left <= x < timeline.right:
                pygame.draw.line(
                    surface, GRID, (x, timeline.y), (x, timeline.bottom)
                )

        surface.set_clip(timeline)
        if loop_beats > 0:
            for interval in channel.intervals:
                onset_in_loop = interval.start / TICKS_PER_BEAT
                duration = interval.duration / TICKS_PER_BEAT
                first_cycle = floor(
                    (view_start - phase_beats - onset_in_loop - duration)
                    / loop_beats
                )
                last_cycle = ceil(
                    (view_start + VISIBLE_BEATS - phase_beats - onset_in_loop)
                    / loop_beats
                )
                for cycle in range(first_cycle, last_cycle + 1):
                    onset = phase_beats + onset_in_loop + cycle * loop_beats
                    x = _beat_to_x(onset, view_start, timeline)
                    width = max(2, round(duration * pixels_per_beat))
                    center_y = _pitch_to_y(interval.pitch, timeline)
                    note_rect = pygame.Rect(x, center_y - 9, width, 18)
                    pygame.draw.rect(surface, color, note_rect, border_radius=3)

                    occurrence_key = (channel.channel_id, interval.id, cycle)
                    is_recent = (
                        occurrence_key in recent_occurrences and onset < display_beats
                    )
                    if is_recent:
                        playhead_x = timeline.x + round(
                            timeline.width * PLAYHEAD_POSITION
                        )
                        surface.set_clip(
                            pygame.Rect(
                                timeline.left,
                                timeline.top,
                                max(0, playhead_x - timeline.left),
                                timeline.height,
                            )
                        )
                        dark = tuple(round(component * 0.55) for component in color)
                        pygame.draw.rect(surface, dark, note_rect, border_radius=3)
                        surface.set_clip(timeline)

                    label = _note_name(interval.pitch)
                    if width >= note_font.size(label)[0] + 8:
                        _draw_text(
                            surface,
                            label,
                            note_font,
                            PRIMARY if is_recent else BACKGROUND,
                            (x + 4, note_rect.y + 2),
                        )

        if channel.live_pitch is not None and channel.live_start_tick is not None:
            onset = channel.live_start_tick / TICKS_PER_BEAT
            duration = max(0.12, display_beats - onset)
            x = _beat_to_x(onset, view_start, timeline)
            width = max(3, round(duration * pixels_per_beat))
            center_y = _pitch_to_y(channel.live_pitch, timeline)
            pygame.draw.rect(
                surface,
                PRIMARY,
                pygame.Rect(x, center_y - 10, width, 20),
                width=1,
            )

        if channel.loop_definition_active:
            x = _beat_to_x(phase_beats, view_start, timeline)
            if timeline.left <= x < timeline.right:
                pygame.draw.line(
                    surface,
                    LOOP_BOUNDARY,
                    (x, timeline.top),
                    (x, timeline.bottom),
                    1,
                )
        elif loop_beats > 0:
            first_boundary = ceil((view_start - phase_beats) / loop_beats)
            last_boundary = floor(
                (view_start + VISIBLE_BEATS - phase_beats) / loop_beats
            )
            for cycle in range(first_boundary, last_boundary + 1):
                boundary = phase_beats + cycle * loop_beats
                x = _beat_to_x(boundary, view_start, timeline)
                if timeline.left <= x < timeline.right:
                    pygame.draw.line(
                        surface,
                        LOOP_BOUNDARY,
                        (x, timeline.top),
                        (x, timeline.bottom),
                        1,
                    )

        playhead_x = timeline.x + round(timeline.width * PLAYHEAD_POSITION)
        pygame.draw.line(
            surface,
            PLAYHEAD,
            (playhead_x, timeline.top),
            (playhead_x, timeline.bottom),
            2,
        )
    finally:
        surface.set_clip(previous_clip)


def _draw_ui(
    pygame,
    screen,
    fonts,
    snapshot: GuiSnapshot,
    display_beats: float,
    show_seconds: bool,
    command_mode: bool,
    command_text: str,
    status_text: str,
    recent_occurrences: set[tuple[str, int, int]],
) -> None:
    width, height = screen.get_size()
    screen.fill(BACKGROUND)
    _draw_text(screen, "PY-SEQ", fonts["title"], PRIMARY, (20, 12))
    state_text = "PLAYING" if snapshot.transport.playing else "PAUSED"
    _draw_text(
        screen,
        state_text,
        fonts["small_bold"],
        PLAY if snapshot.transport.playing else SECONDARY,
        (150, 21),
    )
    tempo_text = f"{snapshot.transport.tempo_bpm:g} BPM"
    _draw_text(
        screen,
        tempo_text,
        fonts["small"],
        PRIMARY,
        (245, 21),
    )
    midi_in_bpm = (
        "---" if snapshot.midi_in_bpm is None else str(snapshot.midi_in_bpm)
    )
    midi_in_text = f"MIDI IN BPM {midi_in_bpm}"
    if snapshot.midi_in_no_clock:
        midi_in_text += " (no clock)"
    midi_in_x = max(340, 245 + fonts["small"].size(tempo_text)[0] + 20)
    _draw_text(screen, midi_in_text, fonts["small"], PRIMARY, (midi_in_x, 21))
    connected_outputs = sum(channel.output_connected for channel in snapshot.channels)
    input_state = "IN OK" if snapshot.input_connected else "IN --"
    output_state = f"OUT {connected_outputs}/{CHANNEL_COUNT}"
    if snapshot.lifecycle == "failed":
        midi_status = "HOST ERROR"
        status_color = LOOP_BOUNDARY
    else:
        midi_status = f"MIDI {input_state}  {output_state}"
        status_color = PLAY
    status_x = max(
        midi_in_x + fonts["small"].size(midi_in_text)[0] + 20,
        width - fonts["small"].size(midi_status)[0] - 25,
    )
    midi_status = _truncate_text(midi_status, fonts["small"], width - status_x - 20)
    _draw_text(screen, midi_status, fonts["small"], status_color, (status_x, 21))
    pygame.draw.line(screen, GRID, (20, 50), (width - 20, 50))

    top = 58
    footer_height = 44
    lane_gap = 7
    lane_height = (height - top - footer_height - 18 - lane_gap * 3) // CHANNEL_COUNT
    for index, channel in enumerate(snapshot.channels):
        y = top + index * (lane_height + lane_gap)
        rect = pygame.Rect(18, y, width - 36, lane_height)
        color = CHANNEL_COLORS[index % len(CHANNEL_COLORS)]
        selected = channel.channel_id == snapshot.selected_channel_id
        pygame.draw.rect(screen, PANEL, rect, border_radius=5)
        pygame.draw.rect(screen, color if selected else GRID, rect, 2 if selected else 1, 5)

        loop_beats = channel.loop_length_ticks / TICKS_PER_BEAT
        relative_position = display_beats - channel.phase_origin_tick / TICKS_PER_BEAT
        phase = (
            (relative_position % loop_beats) / loop_beats
            if loop_beats and not channel.loop_definition_active
            else 0.0
        )
        if not loop_beats:
            length_label = "UNSET"
        elif (
            show_seconds
            and snapshot.transport.tempo_bpm > 0
            and isfinite(snapshot.transport.tempo_bpm)
        ):
            loop_length = loop_beats * 60 / snapshot.transport.tempo_bpm
            length_label = f"{loop_length:.1f} SEC"
        else:
            length_label = f"{loop_beats:g} BEATS"
        loop_label = (
            "--:-- ( --%)"
            if channel.loop_definition_active
            else _loop_label(relative_position, loop_beats)
        )
        metadata = (
            f"{chr(ord('A') + index)} {channel.name}   MIDI CH {channel.midi_channel + 1:02d}   "
            f"BANK {channel.bank_msb:03d}  PRG {channel.program:03d}   "
            f"LOOP {length_label}   {loop_label}"
        )
        _draw_text(
            screen,
            metadata,
            fonts["small_bold"] if selected else fonts["small"],
            color if selected else SECONDARY,
            (rect.x + 12, rect.y + 7),
        )
        state_label = ""
        if selected and snapshot.recording_state == "WAITING_START":
            state_label = (
                "ARMED / OVR"
                if channel.channel_id == snapshot.overwrite_channel_id
                else "ARMED"
            )
        elif selected and snapshot.recording_state == "WAITING_STOP":
            state_label = (
                "STOP PENDING / OVR"
                if channel.channel_id == snapshot.overwrite_channel_id
                else "STOP PENDING"
            )
        elif channel.channel_id == snapshot.overwrite_channel_id:
            state_label = "OVERWRITE"
        elif selected and snapshot.recording_state == "RECORDING":
            state_label = "RECORDING"
        if state_label:
            _draw_text(
                screen,
                state_label,
                fonts["small_bold"],
                LOOP_BOUNDARY,
                (rect.right - 160, rect.y + 7),
            )

        gauge = pygame.Rect(rect.x + 12, rect.y + 28, rect.width - 24, 3)
        pygame.draw.rect(screen, GRID, gauge)
        pygame.draw.rect(
            screen,
            color,
            pygame.Rect(gauge.x, gauge.y, round(gauge.width * phase), gauge.height),
        )
        timeline = pygame.Rect(
            rect.x + 12,
            rect.y + 38,
            rect.width - 24,
            max(24, rect.height - 45),
        )
        pygame.draw.rect(screen, BACKGROUND, timeline)
        _draw_note_lane(
            pygame,
            screen,
            channel,
            color,
            timeline,
            display_beats,
            recent_occurrences,
            fonts["note"],
        )

    footer_y = height - footer_height + 3
    configuration, status = _footer_lines(
        snapshot, display_beats, command_mode, command_text, status_text
    )
    _draw_text(
        screen,
        _truncate_text(configuration, fonts["small"], width - 40),
        fonts["small"],
        SECONDARY,
        (20, footer_y),
    )
    _draw_text(
        screen,
        _truncate_text(status, fonts["small_bold"], width - 40),
        fonts["small_bold"],
        PRIMARY,
        (20, footer_y + 17),
    )


def _truncate_text(text, font, available_width: int) -> str:
    if available_width <= 0 or font.size(text)[0] <= available_width:
        return text if available_width > 0 else ""
    ellipsis = "..."
    ellipsis_width = font.size(ellipsis)[0]
    if ellipsis_width > available_width:
        return ""
    prefix = text
    while prefix and font.size(prefix)[0] + ellipsis_width > available_width:
        prefix = prefix[:-1]
    return f"{prefix}{ellipsis}"


def _handle_notification(
    notification: GuiNotification,
    snapshot: GuiSnapshot,
    recent_occurrences: set[tuple[str, int, int]],
) -> tuple[str, float, str]:
    channel_id = notification.channel_id
    if notification.kind == "tape_changed" and channel_id is not None:
        channel = next(
            (item for item in snapshot.channels if item.channel_id == channel_id),
            None,
        )
        if channel is not None and notification.tick is not None:
            loop_length = channel.loop_length_ticks
            for interval in notification.changed_intervals if loop_length else ():
                cycle = (
                    notification.tick - channel.phase_origin_tick - interval.start
                ) // loop_length
                recent_occurrences.add((channel_id, interval.id, cycle))
            removed_interval_ids = frozenset(notification.removed_interval_ids)
            removed_occurrences = {
                key
                for key in recent_occurrences
                if key[0] == channel_id and key[1] in removed_interval_ids
            }
            recent_occurrences.difference_update(removed_occurrences)
        return "Tape updated", time.monotonic() + 2, "info"
    return (
        notification.message,
        time.monotonic() + (5 if notification.severity == "error" else 2),
        notification.severity,
    )


def _reconcile_projected_state(
    projected: dict[str, object],
    pending_request_ids: dict[str, int],
    snapshot: GuiSnapshot,
) -> None:
    for field, request_id in tuple(pending_request_ids.items()):
        if request_id <= snapshot.last_processed_request_id:
            projected.pop(field, None)
            pending_request_ids.pop(field)


def _project_recording_toggle(state: str, snapshot: GuiSnapshot) -> str:
    if state == "IDLE":
        return (
            "RECORDING"
            if snapshot.recording_start_mode == "IMMEDIATE"
            else "WAITING_START"
        )
    if state == "WAITING_START":
        return "IDLE"
    if state == "RECORDING":
        return (
            "IDLE"
            if snapshot.recording_stop_mode
            in ("IMMEDIATE", "AFTER_DURATION", "NEXT_GESTURE", "MIDI_TRANSPORT")
            else "WAITING_STOP"
        )
    if state == "WAITING_STOP":
        return "IDLE" if snapshot.recording_stop_gesture_cycle else "RECORDING"
    raise ValueError(f"unknown recording state: {state}")


def run_gui(runtime: FourChannelGuiRuntime, *, fps: int = FRAME_RATE) -> None:
    try:
        import pygame
    except ImportError as error:
        runtime.close()
        raise RuntimeError(
            "Pygame is not installed; install py-seq[gui] to run the GUI"
        ) from error

    try:
        runtime.start()
        pygame.init()
        screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
        pygame.display.set_caption("PY-SEQ")
        fonts = {
            "title": pygame.font.SysFont("dejavusansmono", 22, bold=True),
            "small": pygame.font.SysFont("dejavusansmono", 12),
            "small_bold": pygame.font.SysFont("dejavusansmono", 12, bold=True),
            "note": pygame.font.SysFont("dejavusansmono", 12, bold=True),
        }
        clock = pygame.time.Clock()
        command_mode = False
        command_text = ""
        command_history = CommandHistory()
        midi_completion = MidiDeviceCompletion()
        show_seconds = False
        closing = False
        status_text = "Ready"
        status_until = 0.0
        status_severity = "info"
        recent_occurrences: set[tuple[str, int, int]] = set()
        shutdown_sent = False
        projected: dict[str, object] = {}
        pending_request_ids: dict[str, int] = {}

        def submit_projected(
            kind: GuiIntentKind,
            *,
            state_changes: Mapping[str, object] | None = None,
            channel_id: str | None = None,
            value: bool | int | str | None = None,
            bank_msb: int | None = None,
        ) -> int:
            request_id = runtime.submit(
                kind,
                channel_id=channel_id,
                value=value,
                bank_msb=bank_msb,
            )
            for field, projected_value in (state_changes or {}).items():
                projected[field] = projected_value
                pending_request_ids[field] = request_id
            return request_id

        running = True
        while running:
            now_ns = time.monotonic_ns()
            snapshot = runtime.get_snapshot()
            for notification in runtime.drain_notifications():
                if notification.kind == "command_applied":
                    command_history.applied(notification.request_id)
                elif notification.kind == "command_rejected":
                    command_history.rejected(notification.request_id)
                candidate_text, candidate_until, candidate_severity = _handle_notification(
                    notification,
                    snapshot,
                    recent_occurrences,
                )
                severity_order = {"info": 0, "warning": 1, "error": 2}
                if (
                    time.monotonic() >= status_until
                    or severity_order[candidate_severity]
                    >= severity_order[status_severity]
                ):
                    status_text = candidate_text
                    status_until = candidate_until
                    status_severity = candidate_severity
            _reconcile_projected_state(projected, pending_request_ids, snapshot)
            display_beats = interpolated_position_beats(snapshot, now_ns)
            view_start = display_beats - VISIBLE_BEATS * PLAYHEAD_POSITION
            intervals_by_id = {
                (channel.channel_id, interval.id): interval
                for channel in snapshot.channels
                for interval in channel.intervals
            }
            visible_occurrences = {
                key
                for key in recent_occurrences
                if key[:2] in intervals_by_id
                and (
                    snapshot_channel := next(
                        item
                        for item in snapshot.channels
                        if item.channel_id == key[0]
                    )
                )
                and snapshot_channel.loop_length_ticks > 0
                and (
                    snapshot_channel.phase_origin_tick + key[2] * snapshot_channel.loop_length_ticks
                    + intervals_by_id[key[:2]].start
                    + intervals_by_id[key[:2]].duration
                )
                / TICKS_PER_BEAT
                >= view_start
            }
            recent_occurrences.intersection_update(visible_occurrences)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    closing = True
                elif event.type == pygame.VIDEORESIZE:
                    screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                elif event.type == pygame.KEYDOWN:
                    if command_mode:
                        if event.key == pygame.K_ESCAPE:
                            command_mode = False
                            command_text = ""
                            command_history.reset_navigation()
                            status_text = "Command cancelled"
                            status_until = time.monotonic() + 2
                        elif event.key == pygame.K_UP:
                            command_text = command_history.previous(command_text)
                        elif event.key == pygame.K_DOWN:
                            command_text = command_history.next(command_text)
                        elif event.key == pygame.K_BACKSPACE:
                            command_text = command_text[:-1]
                        elif event.key == pygame.K_TAB:
                            try:
                                command_text, completion_error = midi_completion.complete(
                                    command_text
                                )
                                if completion_error:
                                    status_text = completion_error
                                    status_until = time.monotonic() + 5
                                    status_severity = "error"
                            except Exception as error:
                                status_text = f"MIDI discovery failed: {error}"
                                status_until = time.monotonic() + 5
                                status_severity = "error"
                        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                            try:
                                command_parts = command_text.strip().lower().split(
                                    maxsplit=1
                                )
                                command_word = command_parts[0] if command_parts else ""
                                is_program_command = (
                                    command_word == "p"
                                    or (
                                        command_word.startswith("p")
                                        and len(command_word) > 1
                                        and command_word[1].isdigit()
                                    )
                                )
                                if command_word and not is_program_command:
                                    command_word = resolve_keyword(
                                        command_word,
                                        (
                                            "reset",
                                            "length",
                                            "start",
                                            "stop",
                                            "reference",
                                            "transport",
                                            "clock",
                                            "bpm",
                                            "tempo",
                                            "midi",
                                        ),
                                    )
                                if command_word == "reset":
                                    parse_reset_command(command_text)
                                    request_id = submit_projected(
                                        GuiIntentKind.RESET_CONFIG,
                                        value=command_text,
                                    )
                                elif command_word == "length":
                                    parse_length_command(command_text)
                                    selected = projected.get(
                                        "selected_channel_id",
                                        snapshot.selected_channel_id,
                                    )
                                    if not isinstance(selected, str):
                                        selected = snapshot.selected_channel_id
                                    request_id = submit_projected(
                                        GuiIntentKind.LOOP_LENGTH,
                                        channel_id=selected,
                                        value=command_text,
                                    )
                                elif command_word in ("start", "stop", "reference"):
                                    parse_recording_sync_command(command_text)
                                    request_id = submit_projected(
                                        GuiIntentKind.RECORDING_SYNC,
                                        value=command_text,
                                    )
                                elif command_word in ("transport", "clock"):
                                    parse_midi_output_command(command_text)
                                    request_id = submit_projected(
                                        GuiIntentKind.MIDI_OUTPUT,
                                        value=command_text,
                                    )
                                elif command_word in ("bpm", "tempo"):
                                    parse_bpm_command(command_text)
                                    request_id = submit_projected(
                                        GuiIntentKind.BPM_CHANGE,
                                        value=command_text,
                                    )
                                elif command_word == "midi":
                                    parse_midi_config_command(command_text)
                                    selected = projected.get(
                                        "selected_channel_id", snapshot.selected_channel_id
                                    )
                                    if not isinstance(selected, str):
                                        selected = snapshot.selected_channel_id
                                    request_id = submit_projected(
                                        GuiIntentKind.MIDI_CONFIG,
                                        channel_id=selected,
                                        value=command_text,
                                    )
                                else:
                                    parsed = parse_program_command(command_text)
                                    selected = projected.get(
                                        "selected_channel_id",
                                        snapshot.selected_channel_id,
                                    )
                                    if not isinstance(selected, str):
                                        selected = snapshot.selected_channel_id
                                    if parsed.bank_msb is None:
                                        request_id = submit_projected(
                                            GuiIntentKind.PROGRAM,
                                            channel_id=selected,
                                            value=parsed.program,
                                        )
                                    else:
                                        request_id = submit_projected(
                                            GuiIntentKind.BANK_PROGRAM,
                                            channel_id=selected,
                                            value=parsed.program,
                                            bank_msb=parsed.bank_msb,
                                        )
                                command_history.submitted(request_id, command_text)
                            except (
                                GuiCommandError,
                                GuiCommandQueueFull,
                                TypeError,
                                ValueError,
                            ) as error:
                                status_text = str(error)
                                status_until = time.monotonic() + 5
                                status_severity = "error"
                            except GuiHostUnavailable as error:
                                status_text = str(error)
                                closing = True
                            command_mode = False
                            command_text = ""
                            command_history.reset_navigation()
                        else:
                            character = getattr(event, "unicode", "")
                            if character.isascii() and character.isprintable():
                                command_text += character
                        continue

                    try:
                        if event.key == pygame.K_ESCAPE:
                            command_mode = True
                            command_text = ""
                            command_history.reset_navigation()
                            midi_completion = MidiDeviceCompletion()
                        elif event.key == pygame.K_q:
                            closing = True
                        elif event.key == pygame.K_r:
                            runtime.submit(GuiIntentKind.RESET)
                        elif event.key == pygame.K_SPACE:
                            desired_playing = not bool(
                                projected.get(
                                    "transport_playing",
                                    snapshot.transport.playing,
                                )
                            )
                            submit_projected(
                                GuiIntentKind.TRANSPORT,
                                state_changes={
                                    "transport_playing": desired_playing
                                },
                                value=desired_playing,
                            )
                        elif event.key == pygame.K_TAB:
                            selected_id = projected.get(
                                "selected_channel_id",
                                snapshot.selected_channel_id,
                            )
                            if not isinstance(selected_id, str):
                                selected_id = snapshot.selected_channel_id
                            selected_index = next(
                                index
                                for index, channel in enumerate(snapshot.channels)
                                if channel.channel_id == selected_id
                            )
                            next_channel = snapshot.channels[
                                (selected_index + 1) % len(snapshot.channels)
                            ]
                            changes: dict[str, object] = {
                                "selected_channel_id": next_channel.channel_id
                            }
                            effective_overwrite = projected.get(
                                "overwrite_channel_id",
                                snapshot.overwrite_channel_id,
                            )
                            if effective_overwrite is not None:
                                changes["overwrite_channel_id"] = None
                            submit_projected(
                                GuiIntentKind.SELECT_CHANNEL,
                                state_changes=changes,
                                channel_id=next_channel.channel_id,
                            )
                        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                            state = str(
                                projected.get(
                                    "recording_state",
                                    snapshot.recording_state,
                                )
                            )
                            next_state = _project_recording_toggle(state, snapshot)
                            changes: dict[str, object] = {
                                "recording_state": next_state,
                                "recording_armed": next_state != "IDLE",
                            }
                            if next_state == "IDLE" and (
                                projected.get(
                                    "overwrite_channel_id",
                                    snapshot.overwrite_channel_id,
                                )
                                is not None
                            ):
                                changes["overwrite_channel_id"] = None
                            submit_projected(
                                GuiIntentKind.RECORDING_TOGGLE,
                                state_changes=changes,
                            )
                        elif event.key == pygame.K_BACKSPACE:
                            selected = projected.get(
                                "selected_channel_id",
                                snapshot.selected_channel_id,
                            )
                            if not isinstance(selected, str):
                                selected = snapshot.selected_channel_id
                            active_overwrite = (
                                projected.get(
                                    "overwrite_channel_id",
                                    snapshot.overwrite_channel_id,
                                )
                                == selected
                            )
                            enable_overwrite = not active_overwrite
                            changes: dict[str, object] = {
                                "overwrite_channel_id": (
                                    selected if enable_overwrite else None
                                )
                            }
                            if enable_overwrite:
                                was_armed = bool(
                                    projected.get(
                                        "recording_armed", snapshot.recording_armed
                                    )
                                )
                                changes["recording_armed"] = True
                                if not was_armed:
                                    changes["recording_state"] = (
                                        _project_recording_toggle("IDLE", snapshot)
                                    )
                            submit_projected(
                                GuiIntentKind.OVERWRITE,
                                state_changes=changes,
                                channel_id=selected,
                                value=enable_overwrite,
                            )
                        elif event.key == pygame.K_DELETE:
                            selected = projected.get(
                                "selected_channel_id",
                                snapshot.selected_channel_id,
                            )
                            if not isinstance(selected, str):
                                selected = snapshot.selected_channel_id
                            submit_projected(
                                GuiIntentKind.CLEAR_TAPE,
                                channel_id=selected,
                            )
                        elif event.key == pygame.K_d:
                            show_seconds = not show_seconds
                    except GuiCommandQueueFull as error:
                        status_text = str(error)
                        status_until = time.monotonic() + 5
                        status_severity = "error"
                    except GuiHostUnavailable:
                        closing = True

            if closing and runtime.lifecycle == "running" and not shutdown_sent:
                try:
                    runtime.submit(GuiIntentKind.SHUTDOWN)
                    shutdown_sent = True
                except GuiCommandQueueFull:
                    runtime.close(timeout=runtime.shutdown_timeout)
                except GuiHostUnavailable:
                    shutdown_sent = True
            if runtime.lifecycle in ("stopped", "failed"):
                if snapshot.terminal_message:
                    status_text = snapshot.terminal_message
                    status_severity = "error"
                running = False

            if time.monotonic() > status_until and not command_mode:
                status_text = "Ready"
                status_severity = "info"
            if snapshot.notification_overflow_count:
                status_text = (
                    f"{snapshot.notification_overflow_count} host notifications lost"
                )
                status_severity = "error"
            _draw_ui(
                pygame,
                screen,
                fonts,
                snapshot,
                display_beats,
                show_seconds,
                command_mode,
                command_text,
                status_text,
                recent_occurrences,
            )
            if status_severity == "error":
                pygame.display.set_caption(f"PY-SEQ - {status_text[:80]}")
            else:
                pygame.display.set_caption("PY-SEQ")
            pygame.display.flip()
            clock.tick(fps)
    finally:
        runtime.close(timeout=runtime.shutdown_timeout)
        pygame.quit()


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("FPS must be a positive integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("FPS must be a positive integer")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.description = "Four-channel MIDI tape looper with a Pygame interface."
    args = parser.parse_args(argv)
    if args.list_midi:
        return 0 if list_midi_ports() else 1

    session: FourChannelSession | None = None
    runtime: FourChannelGuiRuntime | None = None
    try:
        path = config_path()
        config = load_config(path)
        if args.midi_in is not None:
            try:
                names = _get_port_names("get_input_names")
                if args.midi_in >= len(names):
                    raise ValueError(f"input index {args.midi_in} is unavailable")
                config.input.device = names[args.midi_in]
            except Exception as error:
                logger.warning("Cannot resolve MIDI input override: %s", error)
        if args.midi_out:
            try:
                names = _get_port_names("get_output_names")
            except Exception as error:
                logger.warning("Cannot enumerate MIDI output overrides: %s", error)
                names = []
            for channel, index, midi_channel, label in args.midi_out:
                if index >= len(names):
                    logger.warning("MIDI output index %s is unavailable", index)
                    continue
                setting = config.outputs[channel - 1]
                setting.device = names[index]
                setting.channel = midi_channel + 1
                setting.label = label
        routes = tuple(
            MidiRoute(item.label, item.channel - 1, UnassignedMidiSink())
            for item in config.outputs
        )
        session = FourChannelSession(
            tuple(args.length), args.bpm, (routes[0], routes[1], routes[2], routes[3])
        )
        runtime = FourChannelGuiRuntime(
            session,
            midi_config=config,
            midi_config_path=path,
        )
        runtime.connect_configured_midi()
        if args.midi_in is not None or args.midi_out:
            runtime.save_midi_config()
        session = None
        run_gui(runtime, fps=args.fps)
        runtime = None
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        parser.exit(1, f"py-seq-gui: {error}\n")
    finally:
        if runtime is not None:
            runtime.close(timeout=runtime.shutdown_timeout)
        else:
            if session is not None:
                session.close()
    return 0


__all__ = [
    "GuiCommandError",
    "ParsedProgramCommand",
    "interpolated_position_beats",
    "main",
    "parse_program_command",
    "run_gui",
]


if __name__ == "__main__":
    raise SystemExit(main())
