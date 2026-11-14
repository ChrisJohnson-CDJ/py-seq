from __future__ import annotations

from dataclasses import replace

import mido
import pytest

from midi_output import CollectingMidiSink, MidoOutputPortSink
from py_seq.gui_runtime import (
    FourChannelGuiRuntime,
    GuiChannelSnapshot,
    GuiSnapshot,
    GuiTransportSnapshot,
)
from py_seq.pygame_gui import (
    CommandHistory,
    GuiCommandError,
    GRID,
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
    FourChannelSession,
    MidiRoute,
    RecordingStartMode,
    RecordingStopMode,
    ResetMode,
)
from py_seq.gui_runtime import GuiNotification
from py_seq.recording_commands import (
    parse_length_command,
    parse_midi_output_command,
    parse_musical_length,
    parse_recording_sync_command,
    parse_reset_command,
    resolve_keyword,
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


def test_command_history_tracks_only_applied_commands_and_suppresses_adjacent_duplicates():
    history = CommandHistory()
    history.add("   ")
    assert history.commands == ()
    history.submitted(1, "start gesture")
    assert history.commands == ()
    history.rejected(1)
    assert history.commands == ()

    history.submitted(2, " start gesture ")
    history.applied(2)
    history.submitted(3, " start gesture ")
    history.applied(3)
    history.submitted(4, "stop gesture")
    history.applied(4)
    history.submitted(5, " start gesture ")
    history.applied(5)

    assert history.commands == (
        " start gesture ",
        "stop gesture",
        " start gesture ",
    )


def test_command_history_navigation_preserves_draft_and_edits_recalled_text():
    history = CommandHistory()
    history.add("first")
    history.add("second")

    assert history.previous("") == "second"
    assert history.previous("edited second") == "first"
    assert history.previous("edited first") == "first"
    assert history.next("edited first") == "second"
    assert history.next("edited second") == ""
    assert history.next("draft") == "draft"
    assert history.commands == ("first", "second")

    assert history.previous("new draft") == "second"
    history.reset_navigation()
    assert history.previous("fresh draft") == "second"
    assert history.next("edited second") == "fresh draft"


def test_command_history_empty_boundaries_capacity_and_session_reset():
    empty = CommandHistory()
    assert empty.previous("draft") == "draft"
    assert empty.next("draft") == "draft"

    history = CommandHistory()
    for index in range(51):
        history.add(f"command {index}")
    assert len(history.commands) == 50
    assert history.commands[0] == "command 1"
    assert history.commands[-1] == "command 50"

    restarted_application = CommandHistory()
    assert restarted_application.commands == ()


def test_command_keyword_resolution_is_exact_first_and_contextual():
    keywords = ("start", "stop")
    assert resolve_keyword("start", keywords) == "start"
    assert resolve_keyword("sta", keywords) == "start"
    assert resolve_keyword("cy", ("cycle",)) == "cycle"
    with pytest.raises(ValueError, match="Ambiguous keyword 'st'.*start, stop"):
        resolve_keyword("st", keywords)
    with pytest.raises(ValueError, match="Unknown keyword 'xyz'.*start, stop"):
        resolve_keyword("xyz", keywords)

    assert parse_reset_command("reset b").mode is ResetMode.BEAT
    assert parse_length_command("len 3+3/8") == parse_length_command("length 3+3/8")
    assert (
        parse_recording_sync_command("sta ges 6")
        == parse_recording_sync_command("start gesture 6")
    )
    assert (
        parse_recording_sync_command("sto ges 5 cyc")
        == parse_recording_sync_command("stop gesture 5 cyc")
    )
    assert (
        parse_recording_sync_command("sto a 4")
        == parse_recording_sync_command("stop after 4")
    )
    with pytest.raises(ValueError, match="Ambiguous keyword 'st'"):
        parse_recording_sync_command("st beat")
    with pytest.raises(ValueError, match="expected one of: immediate, beat, gesture"):
        parse_recording_sync_command("start xyz")
    with pytest.raises(ValueError, match="positive whole number"):
        parse_recording_sync_command("start gesture 0")
    with pytest.raises(ValueError, match="positive whole number"):
        parse_recording_sync_command("stop gesture 1.5")


def test_midi_output_command_parser_resolves_target_and_mode_prefixes():
    assert parse_midi_output_command("trans aut") == ("transport", "auto")
    assert parse_midi_output_command("cloc on") == ("clock", "on")
    assert parse_midi_output_command("clock off") == ("clock", "off")
    with pytest.raises(ValueError, match="Ambiguous keyword 'o'.*off, on"):
        parse_midi_output_command("clock o")
    with pytest.raises(ValueError, match="Unknown keyword 'missing'"):
        parse_midi_output_command("transport missing")
    with pytest.raises(ValueError, match="use 'transport off"):
        parse_midi_output_command("clock")


def test_recording_sync_command_parser_supports_modes_and_reference_channel():
    assert parse_recording_sync_command(" start beat ") == parse_recording_sync_command(
        "START BEAT"
    )
    with pytest.raises(ValueError, match="Unknown keyword 'note'"):
        parse_recording_sync_command("start note")
    assert (
        parse_recording_sync_command("stop loop").stop_mode
        is RecordingStopMode.NEXT_REFERENCE_LOOP_START
    )
    assert (
        parse_recording_sync_command("start transport").start_mode
        is RecordingStartMode.MIDI_TRANSPORT
    )
    assert (
        parse_recording_sync_command("stop transport").stop_mode
        is RecordingStopMode.MIDI_TRANSPORT
    )
    for command in ("start transport 2", "stop transport 2"):
        with pytest.raises(ValueError):
            parse_recording_sync_command(command)
    assert parse_recording_sync_command("start beat 2").start_beat_count == 2
    with pytest.raises(ValueError, match="stop beat"):
        parse_recording_sync_command("stop beat 4")
    assert (
        parse_recording_sync_command("stop beat").stop_mode
        is RecordingStopMode.NEXT_BEAT
    )
    assert (
        parse_recording_sync_command("stop after 4").stop_after_ticks == 3840
    )
    assert (
        parse_recording_sync_command("stop after 4+1").stop_after_ticks
        == 3900
    )
    assert (
        parse_recording_sync_command("stop after 4+0").stop_after_ticks
        == 4 * 960
    )
    assert (
        parse_recording_sync_command("stop after 4+3/8").stop_after_ticks
        == 4200
    )
    assert (
        parse_recording_sync_command("stop after 0+1/4").stop_after_ticks
        == 240
    )
    assert (
        parse_recording_sync_command("start gesture").start_gesture_count == 1
    )
    assert (
        parse_recording_sync_command("stop gesture 3").stop_gesture_count == 3
    )
    assert parse_recording_sync_command(
        "stop gesture cycle"
    ).stop_gesture_count == 1
    assert parse_recording_sync_command(
        "stop gesture cycle"
    ).stop_gesture_cycle
    assert parse_recording_sync_command(
        "stop gesture cyc"
    ).stop_gesture_cycle
    assert parse_recording_sync_command(
        "stop gesture cy"
    ).stop_gesture_cycle
    assert (
        parse_recording_sync_command("stop gesture 5 cy")
        == parse_recording_sync_command("stop gesture 5 cyc")
        == parse_recording_sync_command("stop gesture 5 cycle")
    )
    assert parse_recording_sync_command(
        "stop gesture 3 cyc"
    ).stop_gesture_count == 3
    assert parse_recording_sync_command(
        "stop gesture 3 cycle"
    ).stop_gesture_cycle
    assert parse_recording_sync_command("start beat").start_beat_count is None
    assert parse_recording_sync_command("reference 4").reference_channel == 3

    with pytest.raises(ValueError, match="between 1 and 4"):
        parse_recording_sync_command("reference 5")
    with pytest.raises(ValueError, match="Unknown keyword 'note'"):
        parse_recording_sync_command("stop note")
    with pytest.raises(ValueError, match="Unknown keyword 'bar'"):
        parse_recording_sync_command("start bar")
    with pytest.raises(ValueError, match="Unknown keyword 'bar'"):
        parse_recording_sync_command("stop bar")
    for command in (
        "start beat 0",
        "stop beat -1",
        "start beat 1.5",
        "stop loop 2",
        "stop after 0",
        "stop after 0+1/3",
        "stop after 4+16/16",
        "stop after 4+1/128",
        "stop after",
        "start gesture 0",
        "stop gesture 1.5",
        "start gesture cycle",
        "start gesture 2 cyc",
        "stop beat 2 cyc",
        "stop immediate cycle",
        "stop after 1 cyc",
    ):
        with pytest.raises(ValueError):
            parse_recording_sync_command(command)


def test_reset_command_parser_accepts_only_supported_modes_and_positive_beats():
    assert parse_reset_command("reset off").mode is ResetMode.OFF
    assert parse_reset_command("reset loop").mode is ResetMode.LOOP
    assert parse_reset_command("reset beat").interval_ticks == 960
    beat_reset = parse_reset_command("reset beat 8")
    assert beat_reset.mode is ResetMode.BEAT
    assert beat_reset.interval_beats == 8
    fractional_reset = parse_reset_command("reset beat 3+3/8")
    assert fractional_reset.interval_ticks == 3240
    assert fractional_reset.interval_beats == 3.375
    exactly_representable_reset = parse_reset_command("reset beat 1+2/128")
    assert exactly_representable_reset.interval_ticks == 975

    for command in (
        "reset",
        "reset beat 0",
        "reset beat 0+0",
        "reset beat -1",
        "reset beat 1.5",
        "reset beat 1+1/3",
        "reset beat 1+1/128",
        "reset beat 1+1/256",
        "reset beat nope",
        "reset unknown",
        "reset loop extra",
    ):
        with pytest.raises(ValueError):
            parse_reset_command(command)


@pytest.mark.parametrize(
    ("expression", "ticks"),
    [
        ("4", 3840),
        ("0", 0),
        ("4+0", 3840),
        ("4+1", 3900),
        ("3+3/8", 3240),
        ("0+1/64", 15),
        ("2+1/2", 2400),
    ],
)
def test_musical_length_parser_converts_supported_fractions_exactly(expression, ticks):
    assert parse_musical_length(expression) == ticks


@pytest.mark.parametrize(
    "expression",
    ("", "-1", "1.5", "4+", "4+1/3", "4+1/128", "4+16/16", "4+0/0"),
)
def test_musical_length_parser_rejects_malformed_or_unsupported_expressions(expression):
    with pytest.raises(ValueError):
        parse_musical_length(expression)


def test_length_command_requires_a_positive_length_and_exact_syntax():
    assert parse_length_command(" LENGTH 3+3/8 ") == 3240

    for command in ("length 0", "length 0+0", "length 2 3", "length -1", "3+1/2"):
        with pytest.raises(ValueError):
            parse_length_command(command)


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
        "REF channel-1  RESET OFF  MIDI TRN OFF/OFF  CLK OFF/OFF"
    )
    assert status == "Channel cleared"
    assert not any(
        hint in configuration or hint in status
        for hint in ("SPACE", "ENTER", "BACKSPACE", "DELETE")
    )
    assert _footer_lines(snapshot, 1.0, True, "start bar", "ignored")[1] == (
        "> start bar"
    )

    counted, _ = _footer_lines(
        replace(
            snapshot,
            recording_start_beat_count=2,
            reset_mode="BEAT",
            reset_interval_beats=3.375,
        ),
        1.0,
        False,
        "",
        "",
    )
    assert "START BEAT 2" in counted
    assert "STOP LOOP" in counted
    assert "RESET BEAT 3.375" in counted
    gestures, _ = _footer_lines(
        replace(
            snapshot,
            recording_start_mode="NEXT_GESTURE",
            recording_stop_mode="NEXT_GESTURE",
            recording_start_gesture_count=2,
            recording_stop_gesture_count=3,
        ),
        1.0,
        False,
        "",
        "",
    )
    assert "START GESTURE 2" in gestures
    cycle, _ = _footer_lines(
        replace(
            snapshot,
            recording_stop_mode="NEXT_GESTURE",
            recording_stop_gesture_count=3,
            recording_stop_gesture_cycle=True,
        ),
        1.0,
        False,
        "",
        "",
    )
    assert "STOP GESTURE 3 CYCLE" in cycle
    assert "STOP GESTURE 3" in gestures
    after_duration, _ = _footer_lines(
        replace(
            snapshot,
            recording_stop_mode="AFTER_DURATION",
            recording_stop_after_ticks=4 * 960 + 3 * 960 // 8,
        ),
        1.0,
        False,
        "",
        "",
    )
    assert "STOP AFTER 4.375" in after_duration
    transport, _ = _footer_lines(
        replace(
            snapshot,
            recording_start_mode="MIDI_TRANSPORT",
            recording_stop_mode="MIDI_TRANSPORT",
        ),
        1.0,
        False,
        "",
        "",
    )
    assert "START TRANSPORT" in transport
    assert "STOP TRANSPORT" in transport


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
        replace(snapshot, recording_start_mode="NEXT_GESTURE"),
        0,
        False,
        "",
        "",
    )
    assert "START GESTURE" in note_start


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


@pytest.mark.parametrize(
    "start_mode", [RecordingStartMode.IMMEDIATE, RecordingStartMode.NEXT_BEAT]
)
def test_recorded_stop_boundary_matches_reset_aligned_grid(start_mode):
    class Draw:
        def __init__(self):
            self.lines = []

        def line(self, *args):
            self.lines.append(args)

    class FakePygame:
        def __init__(self):
            self.draw = Draw()

    class Surface:
        def get_clip(self):
            return None

        def set_clip(self, clip):
            pass

    class Timeline:
        x = left = y = top = 0
        width = right = 15360
        bottom = 20

    routes = tuple(
        MidiRoute(str(index), index, CollectingMidiSink()) for index in range(4)
    )
    session = FourChannelSession(
        (0, 0, 0, 0), 120, (routes[0], routes[1], routes[2], routes[3])
    )
    session.toggle_transport()
    session.advance(200)
    session.reset(200)
    session.advance(300)
    session.configure_recording_sync(
        start_mode=start_mode, stop_mode=RecordingStopMode.NEXT_BEAT
    )
    session.set_recording_armed(True)
    session.advance(1300)
    session.request_recording_stop()
    session.advance(2120)

    snapshot = FourChannelGuiRuntime(session).get_snapshot()
    channel = snapshot.channels[0]
    assert snapshot.transport.position_ticks == 1920
    assert channel.phase_origin_tick + channel.loop_length_ticks == 1920
    assert channel.loop_length_ticks == (
        1820 if start_mode is RecordingStartMode.IMMEDIATE else 960
    )

    pygame = FakePygame()
    timeline = Timeline()
    display_beats = snapshot.transport.position_ticks / 960
    _draw_note_lane(
        pygame,
        Surface(),
        channel,
        (100, 120, 140),
        timeline,
        display_beats,
        set(),
        None,
    )
    view_start = display_beats - 16 * 0.35
    stop_x = _beat_to_x(2, view_start, timeline)
    assert any(line[1] == GRID and line[2][0] == stop_x for line in pygame.draw.lines)
    assert any(
        line[1] == LOOP_BOUNDARY and line[2][0] == stop_x
        for line in pygame.draw.lines
    )
    if start_mode is RecordingStartMode.NEXT_BEAT:
        start_x = _beat_to_x(1, view_start, timeline)
        assert any(
            line[1] == GRID and line[2][0] == start_x
            for line in pygame.draw.lines
        )
    session.close()


def test_reset_draws_free_recorded_loop_origin_on_beat_zero_grid():
    class Draw:
        def __init__(self):
            self.lines = []

        def line(self, *args):
            self.lines.append(args)

    class FakePygame:
        def __init__(self):
            self.draw = Draw()

    class Surface:
        def get_clip(self):
            return None

        def set_clip(self, clip):
            pass

    class Timeline:
        x = left = y = top = 0
        width = right = 15360
        bottom = 20

    routes = tuple(
        MidiRoute(str(index), index, CollectingMidiSink()) for index in range(4)
    )
    session = FourChannelSession(
        (0, 960, 600, 800), 120, (routes[0], routes[1], routes[2], routes[3])
    )
    session.toggle_transport()
    session.advance(480)
    session.set_recording_armed(True)
    session.process_midi_message(
        mido.Message("note_on", channel=0, note=60, velocity=90), 500
    )
    session.process_midi_message(
        mido.Message("note_off", channel=0, note=60, velocity=0), 550
    )
    session.advance(1680)
    session.set_recording_armed(False)
    session.reset(2000)

    snapshot = FourChannelGuiRuntime(session).get_snapshot()
    channel = snapshot.channels[0]
    assert snapshot.transport.position_ticks == 0
    assert channel.phase_origin_tick == 0
    assert channel.loop_length_ticks == 1200
    assert tuple(
        (interval.start, interval.duration, interval.pitch)
        for interval in channel.intervals
    ) == ((20, 50, 60),)

    pygame = FakePygame()
    timeline = Timeline()
    _draw_note_lane(
        pygame,
        Surface(),
        replace(channel, intervals=()),
        (100, 120, 140),
        timeline,
        snapshot.transport.position_ticks / 960,
        set(),
        None,
    )
    playhead_x = round(timeline.width * 0.35)
    assert any(
        line[1] == GRID and line[2][0] == playhead_x
        for line in pygame.draw.lines
    )
    assert any(
        line[1] == LOOP_BOUNDARY and line[2][0] == playhead_x
        for line in pygame.draw.lines
    )
    session.close()


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


@pytest.mark.parametrize(
    ("input_bpm", "no_clock", "expected_text"),
    [
        (None, False, "MIDI IN BPM ---"),
        (120, False, "MIDI IN BPM 120"),
        (889, True, "MIDI IN BPM 889 (no clock)"),
    ],
)
def test_zero_tempo_uses_beats_and_title_bar_displays_input_bpm(
    input_bpm, no_clock, expected_text
):
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
            self.placements = {}
            self.clip = None

        def get_size(self):
            return (800, 600)

        def fill(self, color):
            self.fill_color = color

        def blit(self, text, position):
            assert position
            self.texts.append(text)
            self.placements[text] = position

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
        midi_in_bpm=input_bpm,
        midi_in_no_clock=no_clock,
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
    assert expected_text in screen.texts
    assert "0 BPM" in screen.texts
    status_text = "MIDI IN --  OUT 0/4"
    assert status_text in screen.texts
    assert (
        screen.placements[expected_text][0] + font.size(expected_text)[0]
        < screen.placements[status_text][0]
    )
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
