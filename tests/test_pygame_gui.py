from __future__ import annotations

from dataclasses import replace

import mido
import pytest

from midi_output import MidoOutputPortSink
from py_seq.gui_runtime import (
    GuiChannelSnapshot,
    GuiSnapshot,
    GuiTransportSnapshot,
)
from py_seq.pygame_gui import (
    GuiCommandError,
    LOOP_BOUNDARY,
    PLAYHEAD,
    _beat_to_x,
    _draw_note_lane,
    _draw_ui,
    _format_beat_position,
    _footer_lines,
    _handle_notification,
    _loop_label,
    _open_outputs,
    _pitch_to_y,
    _truncate_text,
    build_parser,
    interpolated_position_beats,
    list_midi_ports,
    parse_program_command,
)
from py_seq.four_channel_session import (
    RecordingStartMode,
    RecordingStopMode,
    ResetMode,
)
from py_seq.gui_runtime import GuiNotification
from py_seq.recording_commands import (
    parse_recording_sync_command,
    parse_reset_command,
)
from tape import VoiceInterval


def test_gui_parser_accepts_midi_routes_and_beat_lengths():
    args = build_parser().parse_args(
        [
            "--midi-in",
            "0",
            "--midi-out",
            "1",
            "0",
            "2",
            "CASIO",
            "--length",
            "4",
            "5",
            "6",
            "7",
            "--bpm",
            "120",
            "--fps",
            "30",
        ]
    )

    assert args.midi_in == 0
    assert args.midi_out == [(1, 0, 2, "CASIO")]
    assert args.length == [4 * 960, 5 * 960, 6 * 960, 7 * 960]
    assert args.bpm == 120
    assert args.fps == 30


def test_gui_parser_rejects_invalid_routes_and_lengths():
    parser = build_parser()

    for args in (
        ["--length", "0", "1", "1", "1"],
        ["--midi-out", "5", "0", "0", "BAD"],
        ["--midi-out", "1", "0", "16", "BAD"],
        [
            "--midi-out",
            "1",
            "0",
            "0",
            "ONE",
            "--midi-out",
            "1",
            "1",
            "0",
            "TWO",
        ],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(args)


def test_gui_opens_each_selected_output_once(monkeypatch):
    opened: list[str] = []

    class FakePort:
        messages: list[mido.Message] = []

        def send(self, message: mido.Message) -> None:
            self.messages.append(message)

        def close(self) -> None:
            pass

    monkeypatch.setattr(mido, "get_output_names", lambda: ["Keyboard Synth"])
    monkeypatch.setattr(
        mido, "open_output", lambda name: (opened.append(name) or FakePort())
    )
    args = build_parser().parse_args(
        [
            "--midi-out",
            "1",
            "0",
            "2",
            "CASIO",
            "--midi-out",
            "2",
            "0",
            "3",
            "CASIO",
        ]
    )
    output_sinks: list[MidoOutputPortSink] = []

    routes = _open_outputs(args, output_sinks)

    assert opened == ["Keyboard Synth"]
    assert routes[0].sink is routes[1].sink
    assert routes[0].midi_channel == 2
    assert routes[1].midi_channel == 3
    assert routes[2].label == "NO OUT"
    for sink in output_sinks:
        sink.close()


def test_gui_list_midi_prints_current_port_indices(monkeypatch, capsys):
    monkeypatch.setattr(mido, "get_input_names", lambda: ["Keyboard"])
    monkeypatch.setattr(mido, "get_output_names", lambda: ["Synth A", "Synth B"])

    assert list_midi_ports()
    output = capsys.readouterr().out

    assert "0: Keyboard" in output
    assert "0: Synth A" in output
    assert "1: Synth B" in output


def test_program_command_parser_supports_program_and_bank_msb_only():
    assert parse_program_command("p042") == parse_program_command(" P042 ")
    assert parse_program_command("p042").bank_msb is None
    assert parse_program_command("p007:042").bank_msb == 7
    assert parse_program_command("p007:042").program == 42

    for command in ("", "p12", "p128", "p000:128", "p001:1"):
        with pytest.raises(GuiCommandError):
            parse_program_command(command)


def test_recording_sync_command_parser_supports_modes_and_reference_channel():
    assert parse_recording_sync_command(" start beat ") == parse_recording_sync_command(
        "START BEAT"
    )
    assert (
        parse_recording_sync_command("start note").start_mode
        is RecordingStartMode.FIRST_NOTE_ONSET
    )
    assert (
        parse_recording_sync_command("stop loop").stop_mode
        is RecordingStopMode.NEXT_REFERENCE_LOOP_START
    )
    assert parse_recording_sync_command("reference 4").reference_channel == 3

    with pytest.raises(ValueError, match="between 1 and 4"):
        parse_recording_sync_command("reference 5")
    with pytest.raises(ValueError, match="invalid stop mode"):
        parse_recording_sync_command("stop note")
    with pytest.raises(ValueError, match="invalid start mode"):
        parse_recording_sync_command("start bar")
    with pytest.raises(ValueError, match="invalid stop mode"):
        parse_recording_sync_command("stop bar")


def test_reset_command_parser_accepts_only_supported_modes_and_positive_beats():
    assert parse_reset_command("reset off").mode is ResetMode.OFF
    assert parse_reset_command("reset loop").mode is ResetMode.LOOP
    beat_reset = parse_reset_command("reset beat 8")
    assert beat_reset.mode is ResetMode.BEAT
    assert beat_reset.interval_beats == 8

    for command in (
        "reset",
        "reset beat",
        "reset beat 0",
        "reset beat -1",
        "reset beat 1.5",
        "reset beat nope",
        "reset unknown",
        "reset loop extra",
    ):
        with pytest.raises(ValueError):
            parse_reset_command(command)


def test_transport_interpolation_is_visual_and_freezes_when_paused():
    channel = GuiChannelSnapshot(
        channel_id="channel-1",
        name="OUT",
        midi_channel=0,
        bank_msb=0,
        program=0,
        loop_length_ticks=3840,
        phase_origin_tick=0,
        tape_revision=0,
        intervals=(),
    )
    snapshot = GuiSnapshot(
        revision=1,
        transport=GuiTransportSnapshot(
            position_ticks=960,
            tempo_bpm=120,
            playing=True,
            sampled_at_ns=1_000_000_000,
            epoch=0,
        ),
        selected_channel_id="channel-1",
        recording_armed=False,
        recording_state="IDLE",
        recording_start_mode="IMMEDIATE",
        recording_stop_mode="IMMEDIATE",
        recording_reference_channel_id=None,
        overwrite_channel_id=None,
        channels=(channel,),
        input_connected=False,
        lifecycle="running",
        notification_overflow_count=0,
        midi_input_overflow_count=0,
    )
    assert interpolated_position_beats(snapshot, 1_500_000_000) == pytest.approx(2)
    paused = GuiSnapshot(
        revision=snapshot.revision,
        transport=GuiTransportSnapshot(
            position_ticks=960,
            tempo_bpm=120,
            playing=False,
            sampled_at_ns=1_000_000_000,
            epoch=1,
        ),
        selected_channel_id=snapshot.selected_channel_id,
        recording_armed=False,
        recording_state="IDLE",
        recording_start_mode="IMMEDIATE",
        recording_stop_mode="IMMEDIATE",
        recording_reference_channel_id=None,
        overwrite_channel_id=None,
        channels=snapshot.channels,
        input_connected=False,
        lifecycle="running",
        notification_overflow_count=0,
        midi_input_overflow_count=0,
    )
    assert interpolated_position_beats(paused, 1_500_000_000) == 1


def test_gui_footer_keeps_recording_configuration_separate_from_status():
    channel = GuiChannelSnapshot(
        channel_id="channel-1",
        name="OUT",
        midi_channel=0,
        bank_msb=0,
        program=0,
        loop_length_ticks=3840,
        phase_origin_tick=0,
        tape_revision=0,
        intervals=(),
    )
    snapshot = GuiSnapshot(
        revision=1,
        transport=GuiTransportSnapshot(
            position_ticks=960,
            tempo_bpm=120,
            playing=True,
            sampled_at_ns=1_000_000_000,
            epoch=0,
        ),
        selected_channel_id="channel-1",
        recording_armed=True,
        recording_state="WAITING_STOP",
        recording_start_mode="NEXT_BEAT",
        recording_stop_mode="NEXT_REFERENCE_LOOP_START",
        recording_reference_channel_id="channel-1",
        overwrite_channel_id=None,
        channels=(channel,),
        input_connected=False,
        lifecycle="running",
        notification_overflow_count=0,
        midi_input_overflow_count=0,
    )

    configuration, status = _footer_lines(
        snapshot, 1.0, False, "", "Channel cleared"
    )

    assert configuration == (
        "BEAT 1:00  REC WAITING_STOP  START BEAT  STOP LOOP  "
        "REF channel-1  RESET OFF"
    )
    assert status == "Channel cleared"
    assert not any(
        hint in configuration or hint in status
        for hint in ("SPACE", "ENTER", "BACKSPACE", "DELETE")
    )
    assert _footer_lines(snapshot, 1.0, True, "start bar", "ignored")[1] == (
        "> start bar"
    )


def test_recording_mode_labels_are_explicit_and_concise():
    snapshot = GuiSnapshot(
        revision=1,
        transport=GuiTransportSnapshot(
            position_ticks=0,
            tempo_bpm=120,
            playing=False,
            sampled_at_ns=0,
            epoch=0,
        ),
        selected_channel_id="channel-1",
        recording_armed=False,
        recording_state="IDLE",
        recording_start_mode="IMMEDIATE",
        recording_stop_mode="NEXT_REFERENCE_LOOP_START",
        recording_reference_channel_id=None,
        overwrite_channel_id=None,
        channels=(),
        input_connected=False,
        lifecycle="running",
        notification_overflow_count=0,
        midi_input_overflow_count=0,
    )

    configuration, _ = _footer_lines(snapshot, 0, False, "", "")

    assert "START NOW" in configuration
    assert "STOP LOOP" in configuration
    note_start, _ = _footer_lines(
        replace(snapshot, recording_start_mode="FIRST_NOTE_ONSET"),
        0,
        False,
        "",
        "",
    )
    assert "START NOTE" in note_start


def test_beat_position_format_rounds_to_nearest_64th_and_carries():
    assert _format_beat_position(127 + 32 / 64) == "127:32"
    assert _format_beat_position(127 + 63.5 / 64) == "128:00"


def test_pitch_and_beat_coordinates_preserve_existing_scale():
    class Timeline:
        x = 10
        width = 160
        height = 100
        bottom = 120

    timeline = Timeline()

    assert _pitch_to_y(24, timeline) == 112
    assert _pitch_to_y(54, timeline) == 70
    assert _pitch_to_y(84, timeline) == 28
    assert _pitch_to_y(0, timeline) == 112
    assert _pitch_to_y(127, timeline) == 28
    assert _beat_to_x(8.25, 4.25, timeline) == 50


def test_footer_text_truncation_uses_pixel_width_and_handles_narrow_width():
    class Font:
        def size(self, text):
            return (len(text) * 5, 10)

    font = Font()

    assert _truncate_text("short", font, 30) == "short"
    truncated = _truncate_text("a long footer line", font, 30)
    assert truncated.endswith("...")
    assert font.size(truncated)[0] <= 30
    assert _truncate_text("text", font, 15) == "..."
    assert _truncate_text("text", font, 10) == ""
    assert _truncate_text("text", font, 0) == ""


def test_uninitialized_loop_has_safe_display_label():
    assert _loop_label(1, 0) == "--:-- ( --%)"
    assert _loop_label(0, 4).endswith("(  0%)")
    assert _loop_label(3.99, 4).endswith("( 99%)")


def test_removed_intervals_clear_all_matching_recent_occurrences_safely():
    channel = GuiChannelSnapshot(
        channel_id="channel-1",
        name="OUT",
        midi_channel=0,
        bank_msb=0,
        program=0,
        loop_length_ticks=3840,
        phase_origin_tick=0,
        tape_revision=0,
        intervals=(),
    )
    snapshot = GuiSnapshot(
        revision=1,
        transport=GuiTransportSnapshot(
            position_ticks=960,
            tempo_bpm=120,
            playing=True,
            sampled_at_ns=1_000_000_000,
            epoch=0,
        ),
        selected_channel_id="channel-1",
        recording_armed=False,
        recording_state="IDLE",
        recording_start_mode="IMMEDIATE",
        recording_stop_mode="IMMEDIATE",
        recording_reference_channel_id=None,
        overwrite_channel_id=None,
        channels=(channel,),
        input_connected=False,
        lifecycle="running",
        notification_overflow_count=0,
        midi_input_overflow_count=0,
    )
    recent = {
        ("channel-1", 10, 0),
        ("channel-1", 10, 1),
        ("channel-1", 11, 0),
        ("channel-2", 10, 0),
    }

    _handle_notification(
        GuiNotification(
            request_id=1,
            kind="tape_changed",
            message="Channel 1 tape changed",
            channel_id="channel-1",
            tick=960,
            removed_interval_ids=(10, 11),
        ),
        snapshot,
        recent,
    )

    assert recent == {("channel-2", 10, 0)}


def test_uninitialized_pygame_lane_draws_grid_without_loop_arithmetic():
    class Draw:
        def __init__(self):
            self.lines = []

        def line(self, *args):
            self.lines.append(args)

    class FakePygame:
        def __init__(self):
            self.draw = Draw()

    class Surface:
        def __init__(self):
            self.clips = []
            self.clip = "parent"

        def get_clip(self):
            return self.clip

        def set_clip(self, clip):
            self.clips.append(clip)
            self.clip = clip

    class Timeline:
        x = left = y = 0
        width = 100
        right = 100
        top = 0
        bottom = 20

    pygame = FakePygame()
    channel = GuiChannelSnapshot(
        channel_id="channel-1",
        name="OUT",
        midi_channel=0,
        bank_msb=0,
        program=0,
        loop_length_ticks=0,
        phase_origin_tick=0,
        tape_revision=0,
        intervals=(),
    )
    surface = Surface()
    timeline = Timeline()

    _draw_note_lane(
        pygame,
        surface,
        channel,
        (100, 120, 140),
        timeline,
        0,
        set(),
        None,
    )

    assert surface.clips == [timeline, "parent"]
    assert surface.get_clip() == "parent"
    assert any(line[1] == PLAYHEAD and line[4] == 2 for line in pygame.draw.lines)
    assert not any(line[1] == LOOP_BOUNDARY for line in pygame.draw.lines)

    _draw_note_lane(
        pygame,
        surface,
        replace(channel, loop_definition_active=True),
        (100, 120, 140),
        timeline,
        0,
        set(),
        None,
    )

    assert sum(line[1] == LOOP_BOUNDARY for line in pygame.draw.lines) == 1


def test_valid_loop_stored_note_keeps_its_existing_screen_geometry():
    class Rect:
        def __init__(self, x, y, width, height):
            self.x = x
            self.y = y
            self.width = width
            self.height = height

        @property
        def left(self):
            return self.x

        @property
        def right(self):
            return self.x + self.width

        @property
        def top(self):
            return self.y

        @property
        def bottom(self):
            return self.y + self.height

    class Draw:
        def __init__(self):
            self.rectangles = []
            self.lines = []

        def line(self, *args):
            self.lines.append(args)

        def rect(self, surface, color, rect, **kwargs):
            self.rectangles.append((surface, color, rect, kwargs))

    class FakePygame:
        def __init__(self):
            self.Rect = Rect
            self.draw = Draw()

    class Surface:
        def __init__(self):
            self.clips = []
            self.clip = None

        def get_clip(self):
            return self.clip

        def set_clip(self, clip):
            self.clips.append(clip)
            self.clip = clip

    class NoteFont:
        def size(self, text):
            assert text
            return (100, 10)

    pygame = FakePygame()
    timeline = Rect(10, 10, 160, 100)
    channel = GuiChannelSnapshot(
        channel_id="channel-1",
        name="OUT",
        midi_channel=0,
        bank_msb=0,
        program=0,
        loop_length_ticks=3840,
        phase_origin_tick=0,
        tape_revision=0,
        intervals=(
            VoiceInterval(
                id=1,
                start=480,
                duration=960,
                pitch=54,
                velocity=100,
                gesture_id=1,
            ),
        ),
    )
    surface = Surface()

    _draw_note_lane(
        pygame,
        surface,
        channel,
        (100, 120, 140),
        timeline,
        1,
        {("channel-1", 1, 0)},
        NoteFont(),
    )

    assert any(
        drawn_surface is surface
        and color == (100, 120, 140)
        and (rect.x, rect.y, rect.width, rect.height) == (61, 51, 10, 18)
        and kwargs.get("border_radius") == 3
        for drawn_surface, color, rect, kwargs in pygame.draw.rectangles
    )
    assert surface.get_clip() is None
    assert len(surface.clips) == 4
    assert surface.clips[0] is timeline
    assert surface.clips[2] is timeline
    assert surface.clips[3] is None


def test_zero_tempo_uses_beats_when_seconds_are_requested():
    class Rect:
        def __init__(self, x, y, width, height):
            self.x = x
            self.y = y
            self.width = width
            self.height = height

        @property
        def left(self):
            return self.x

        @property
        def right(self):
            return self.x + self.width

        @property
        def top(self):
            return self.y

        @property
        def bottom(self):
            return self.y + self.height

    class Draw:
        def __init__(self):
            self.lines = []
            self.rectangles = []

        def line(self, *args):
            self.lines.append(args)

        def rect(self, *args, **kwargs):
            self.rectangles.append((args, kwargs))

    class FakePygame:
        def __init__(self):
            self.Rect = Rect
            self.draw = Draw()

    class Font:
        def render(self, text, *args):
            assert args
            return text

        def size(self, text):
            return (len(text) * 6, 12)

    class Screen:
        def __init__(self):
            self.texts = []
            self.clip = None

        def get_size(self):
            return (800, 600)

        def fill(self, color):
            self.fill_color = color

        def blit(self, text, position):
            assert position
            self.texts.append(text)

        def get_clip(self):
            return self.clip

        def set_clip(self, clip):
            self.clip = clip

    channel = GuiChannelSnapshot(
        channel_id="channel-1",
        name="OUT",
        midi_channel=0,
        bank_msb=0,
        program=0,
        loop_length_ticks=3840,
        phase_origin_tick=0,
        tape_revision=0,
        intervals=(),
    )
    snapshot = GuiSnapshot(
        revision=1,
        transport=GuiTransportSnapshot(
            position_ticks=0,
            tempo_bpm=0,
            playing=False,
            sampled_at_ns=0,
            epoch=0,
        ),
        selected_channel_id="channel-1",
        recording_armed=False,
        recording_state="RECORDING",
        recording_start_mode="IMMEDIATE",
        recording_stop_mode="IMMEDIATE",
        recording_reference_channel_id=None,
        overwrite_channel_id=None,
        channels=(
            replace(channel, loop_definition_active=True),
            replace(channel, channel_id="channel-2", loop_length_ticks=0),
            replace(channel, channel_id="channel-3", loop_length_ticks=0),
            replace(channel, channel_id="channel-4", loop_length_ticks=0),
        ),
        input_connected=False,
        lifecycle="running",
        notification_overflow_count=0,
        midi_input_overflow_count=0,
    )
    font = Font()
    screen = Screen()
    pygame = FakePygame()

    _draw_ui(
        pygame,
        screen,
        {"title": font, "small": font, "small_bold": font, "note": font},
        snapshot,
        0,
        True,
        False,
        "",
        "",
        set(),
    )

    assert any("LOOP 4 BEATS   --:-- ( --%)" in text for text in screen.texts)
    assert sum(line[1] == LOOP_BOUNDARY for line in pygame.draw.lines) == 1
    assert any(
        len(rectangle_call[0]) > 2
        and rectangle_call[0][0] is screen
        and rectangle_call[0][2].width == 0
        and rectangle_call[0][2].height == 3
        for rectangle_call in pygame.draw.rectangles
    )
    assert sum(
        len(line) > 4 and line[1] == PLAYHEAD and line[4] == 2
        for line in pygame.draw.lines
    ) == 4
