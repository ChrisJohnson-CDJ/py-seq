from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import mido
import pytest

from py_seq.four_channel_session import FourChannelSession, MidiRoute
from py_seq.gui_runtime import FourChannelGuiRuntime, GuiIntent, GuiIntentKind
from py_seq.midi_config import MidiConfig, UnassignedMidiSink, config_path, load_config, save_config
from py_seq.pygame_gui import MidiDeviceCompletion, main, run_gui
from py_seq.recording_commands import parse_midi_config_command


class Port:
    def __init__(self) -> None:
        self.messages: list[mido.Message] = []
        self.closed = False

    def send(self, message: mido.Message) -> None:
        self.messages.append(message)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def midi(monkeypatch):
    inputs = ["Arturia KeyStep", "Second Input"]
    outputs = ["UM-ONE", "Other Device"]
    opened_outputs: dict[str, list[Port]] = {}
    opened_inputs: dict[str, list[tuple[Port, object]]] = {}

    def open_output(name):
        port = Port()
        opened_outputs.setdefault(name, []).append(port)
        return port

    def open_input(name, callback):
        port = Port()
        opened_inputs.setdefault(name, []).append((port, callback))
        return port

    monkeypatch.setattr(mido, "get_input_names", lambda: inputs)
    monkeypatch.setattr(mido, "get_output_names", lambda: outputs)
    monkeypatch.setattr(mido, "open_input", open_input)
    monkeypatch.setattr(mido, "open_output", open_output)
    return inputs, outputs, opened_inputs, opened_outputs


def runtime_for(config: MidiConfig, path: Path) -> FourChannelGuiRuntime:
    routes = tuple(MidiRoute("", 0, UnassignedMidiSink()) for _ in range(4))
    session = FourChannelSession((960, 960, 960, 960), 120, (routes[0], routes[1], routes[2], routes[3]))
    runtime = FourChannelGuiRuntime(session, midi_config=config, midi_config_path=path, clock_ns=lambda: 0)
    runtime.connect_configured_midi()
    return runtime


def command(runtime: FourChannelGuiRuntime, text: str, channel: str = "channel-1") -> None:
    runtime._apply_intent(GuiIntent(1, 0, GuiIntentKind.MIDI_CONFIG, channel, text), 0)


def test_default_config_and_xdg_location(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert config_path() == tmp_path / "py-seq" / "midi.json"
    config = load_config(config_path())
    assert config.input.device is None
    assert [(item.device, item.channel, item.label) for item in config.outputs] == [(None, 1, "")] * 4


def test_invalid_config_and_individual_valid_fields(tmp_path, caplog):
    path = tmp_path / "midi.json"
    path.write_text("{", encoding="utf-8")
    assert load_config(path) == MidiConfig()
    path.write_text('{"version": 5}', encoding="utf-8")
    assert load_config(path) == MidiConfig()
    path.write_text(json.dumps({
        "version": 1,
        "input": {"device": "Keyboard", "channel": 99},
        "outputs": [{"device": "Synth", "channel": 2, "label": "BASS"},
                    {"device": 123, "channel": True, "label": "too long!"}],
    }), encoding="utf-8")
    config = load_config(path)
    assert (config.input.device, config.input.channel) == ("Keyboard", 1)
    assert (config.outputs[0].device, config.outputs[0].channel, config.outputs[0].label) == ("Synth", 2, "BASS")
    assert config.outputs[1].device is None
    assert "Invalid MIDI" in caplog.text


def test_atomic_write_failure_preserves_existing_file(tmp_path, monkeypatch, caplog):
    path = tmp_path / "midi.json"
    assert save_config(path, MidiConfig())
    original = path.read_text(encoding="utf-8")
    def fail(source, destination):
        raise OSError("disk failure")
    monkeypatch.setattr(os, "replace", fail)
    assert not save_config(path, MidiConfig())
    assert path.read_text(encoding="utf-8") == original
    assert list(tmp_path.iterdir()) == [path]
    assert "disk failure" in caplog.text


def test_startup_merge_and_port_names_are_stable(tmp_path, monkeypatch, midi):
    inputs, outputs, opened_inputs, opened_outputs = midi
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    config = MidiConfig()
    config.input.device = "Arturia KeyStep"
    config.input.channel = 2
    for i, setting in enumerate(config.outputs):
        setting.device = "UM-ONE"
        setting.channel = i + 1
        setting.label = f"L{i}"
    assert save_config(config_path(), config)
    inputs.reverse()
    outputs.reverse()
    snapshots = []
    def run(runtime, *, fps):
        snapshots.append(runtime.get_snapshot())
        runtime.close()
    monkeypatch.setattr("py_seq.pygame_gui.run_gui", run)
    assert main(["--midi-in", "0", "--midi-out", "2", "0", "7", "NEW"]) == 0
    restored = load_config(config_path())
    assert (restored.input.device, restored.input.channel) == ("Second Input", 2)
    assert [(x.device, x.channel, x.label) for x in restored.outputs] == [
        ("UM-ONE", 1, "L0"), ("Other Device", 8, "NEW"),
        ("UM-ONE", 3, "L2"), ("UM-ONE", 4, "L3"),
    ]
    assert len(opened_outputs["UM-ONE"]) == 1
    assert len(opened_outputs["Other Device"]) == 1
    assert "Second Input" in opened_inputs
    assert all(channel.output_connected for channel in snapshots[0].channels)


def test_unavailable_devices_retained_and_partial_routes_connect(tmp_path, midi):
    _, _, _, opened = midi
    config = MidiConfig()
    config.input.device = "Missing Keyboard"
    config.outputs[0].device = "Missing Synth"
    config.outputs[1].device = "UM-ONE"
    runtime = runtime_for(config, tmp_path / "midi.json")
    try:
        snapshot = runtime.get_snapshot()
        assert not snapshot.input_connected
        assert not snapshot.channels[0].output_connected
        assert snapshot.channels[1].output_connected
        assert config.input.device == "Missing Keyboard"
        assert config.outputs[0].device == "Missing Synth"
        assert opened["UM-ONE"]
        assert len(runtime.drain_notifications()) == 2
    finally:
        runtime.close()


def test_startup_persists_unopenable_override_without_losing_other_routes(tmp_path, monkeypatch, midi):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    config = MidiConfig()
    config.outputs[1].device = "Other Device"
    assert save_config(config_path(), config)

    def fail(name):
        raise OSError("port busy")

    original_open = getattr(mido, "open_output")
    monkeypatch.setattr(mido, "open_output", lambda name: fail(name) if name == "UM-ONE" else original_open(name))
    snapshots = []
    def run(runtime, *, fps):
        snapshots.append(runtime.get_snapshot())
        runtime.close()
    monkeypatch.setattr("py_seq.pygame_gui.run_gui", run)
    assert main(["--midi-out", "1", "0", "2", "BASS"]) == 0
    saved = load_config(config_path())
    assert (saved.outputs[0].device, saved.outputs[0].channel) == ("UM-ONE", 3)
    assert saved.outputs[1].device == "Other Device"
    assert not snapshots[0].channels[0].output_connected
    assert snapshots[0].channels[1].output_connected


def test_runtime_shared_output_release_labels_and_failed_connection(tmp_path, midi):
    _, _, _, opened = midi
    config = MidiConfig()
    config.outputs[0].device = config.outputs[1].device = "UM-ONE"
    runtime = runtime_for(config, tmp_path / "midi.json")
    session = runtime.session
    session.toggle_transport()
    session.process_midi_message(mido.Message("note_on", note=60, velocity=100), 0)
    old = opened["UM-ONE"][0]
    state = session.recording_state
    try:
        command(runtime, "m out 3 Other Device")
        assert not old.closed
        assert any(getattr(msg, "type") == "note_off" and getattr(msg, "note") == 60 for msg in old.messages)
        assert getattr(session.routes[1].sink, "port") is old
        assert session.transport_running and session.recording_state == state
        command(runtime, "m label BASS")
        assert runtime._make_snapshot(0).channels[0].name == "BASS"
        command(runtime, "m label clear")
        command(runtime, "m out 4 Missing Synth")
        assert not runtime._make_snapshot(0).channels[0].output_connected
        assert config.outputs[0].device == "Missing Synth"
        assert config.outputs[0].channel == 4
        assert config.outputs[0].label == ""
    finally:
        runtime.close()
    assert old.closed
    assert load_config(tmp_path / "midi.json").outputs[0] == config.outputs[0]


def test_runtime_input_replacement_filters_channel_and_ignores_old_callback(tmp_path, midi):
    _, _, inputs, _ = midi
    config = MidiConfig()
    config.input.device = "Arturia KeyStep"
    runtime = runtime_for(config, tmp_path / "midi.json")
    try:
        old, callback = inputs["Arturia KeyStep"][0]
        command(runtime, "midi in 3 Second Input")
        assert old.closed
        callback(mido.Message("note_on", channel=2, note=60))
        new_callback = inputs["Second Input"][0][1]
        new_callback(mido.Message("note_on", channel=0, note=61))
        new_callback(mido.Message("note_on", channel=2, note=62))
        runtime._process_pending()
        assert runtime.session.articulators[0].active_pitch == 62
        command(runtime, "midi in 4 Missing Device")
        assert not runtime._make_snapshot(0).input_connected
        assert config.input.device == "Missing Device"
    finally:
        runtime.close()


@pytest.mark.parametrize("text", ["midi in 0 X", "midi out 17 X", "midi in x X", "midi label too-long!"])
def test_invalid_commands_do_not_change_config(tmp_path, midi, text):
    config = MidiConfig()
    runtime = runtime_for(config, tmp_path / "midi.json")
    try:
        with pytest.raises(ValueError):
            command(runtime, text)
        assert config == MidiConfig()
    finally:
        runtime.close()


def test_completion_and_literal_arguments(midi):
    complete = MidiDeviceCompletion()
    text, error = complete.complete("m i 1")
    assert (text, error) == ("m i 1 Arturia KeyStep", None)
    text, _ = complete.complete(text)
    assert text == "m i 1 Second Input"
    text, _ = complete.complete(text)
    assert text == "m i 1 Arturia KeyStep"
    assert parse_midi_config_command(text) == ("in", 1, "Arturia KeyStep")
    assert parse_midi_config_command("m lab Soft Pad") == ("label", None, "Soft Pad")
    midi[1].clear()
    assert complete.complete("midi out 3") == ("midi out 3", "No MIDI out devices available")
    assert complete.complete("midi label 3") == ("midi label 3", None)


def test_gui_tab_selection_and_escape_cancel(tmp_path, monkeypatch, midi):
    runtime = runtime_for(MidiConfig(), tmp_path / "midi.json")
    key = lambda code, text="": SimpleNamespace(type=1, key=code, unicode=text)
    events = iter((
        (key(27), *(key(0, c) for c in "midi out 3"), key(9), key(9), key(27)),
        (key(27), *(key(0, c) for c in "midi out 3"), key(9), key(13)),
        (SimpleNamespace(type=2),),
    ))
    displayed = []
    pygame = SimpleNamespace(
        QUIT=2, VIDEORESIZE=3, KEYDOWN=1, K_ESCAPE=27, K_q=113,
        K_r=114, K_SPACE=32, K_TAB=9, K_RETURN=13, K_KP_ENTER=271,
        K_BACKSPACE=8, K_DELETE=127, K_d=100, K_UP=273, K_DOWN=274,
        RESIZABLE=1, init=lambda: None, quit=lambda: None,
        display=SimpleNamespace(
            set_mode=lambda *args: object(), set_caption=lambda text: None,
            flip=lambda: None,
        ),
        font=SimpleNamespace(SysFont=lambda *args, **kwargs: object()),
        time=SimpleNamespace(Clock=lambda: SimpleNamespace(tick=lambda fps: None)),
        event=SimpleNamespace(get=lambda: next(events, ())),
    )
    monkeypatch.setitem(sys.modules, "pygame", pygame)
    monkeypatch.setattr("py_seq.pygame_gui._draw_ui", lambda *args: displayed.append(args[7]))
    run_gui(runtime)
    config = load_config(tmp_path / "midi.json")
    assert config.outputs[0].device == "UM-ONE"
    assert config.outputs[0].channel == 3
    assert "Other Device" not in [item.device for item in config.outputs]


def test_persistence_failure_notifies_without_stopping_runtime(tmp_path, monkeypatch, midi):
    runtime = runtime_for(MidiConfig(), tmp_path / "midi.json")
    monkeypatch.setattr("py_seq.gui_runtime.save_config", lambda *args: False)
    try:
        command(runtime, "midi label BASS")
        assert runtime._config_writer is not None
        runtime._config_writer.shutdown(wait=True)
        assert runtime.session.routes[0].label == "BASS"
        assert any("Could not save" in note.message for note in runtime.drain_notifications())
    finally:
        runtime.close()
