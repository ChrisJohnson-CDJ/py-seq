"""Desired MIDI settings and their user-specific persistence."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MidiSetting:
    device: str | None = None
    channel: int = 1
    label: str = ""


@dataclass
class MidiConfig:
    input: MidiSetting = field(default_factory=MidiSetting)
    outputs: list[MidiSetting] = field(
        default_factory=lambda: [MidiSetting() for _ in range(4)]
    )


class UnassignedMidiSink:
    def send(self, message: object) -> None:
        pass


def config_path() -> Path:
    root = os.environ.get("XDG_CONFIG_HOME")
    base = Path(root) if root and Path(root).is_absolute() else Path.home() / ".config"
    return base / "py-seq" / "midi.json"


def load_config(path: Path) -> MidiConfig:
    config = MidiConfig()
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        return config
    except (OSError, ValueError, UnicodeError) as error:
        logger.warning("Cannot read MIDI configuration %s: %s", path, error)
        return config
    if (
        not isinstance(data, dict)
        or type(data.get("version")) is not int
        or data["version"] != 1
    ):
        logger.warning("Unsupported or invalid MIDI configuration version in %s", path)
        return config

    def read_setting(value: object, target: MidiSetting, *, output: bool) -> None:
        if not isinstance(value, dict):
            logger.warning("Invalid MIDI configuration setting: %r", value)
            return
        device = value.get("device")
        channel = value.get("channel")
        if device is None or isinstance(device, str) and bool(device.strip()):
            target.device = device
        else:
            logger.warning("Invalid MIDI device: %r", device)
        if type(channel) is int and 1 <= channel <= 16:
            target.channel = channel
        else:
            logger.warning("Invalid MIDI channel: %r", channel)
        if output:
            label = value.get("label", "")
            if isinstance(label, str) and len(label) <= 8:
                target.label = label
            else:
                logger.warning("Invalid MIDI label: %r", label)

    read_setting(data.get("input"), config.input, output=False)
    outputs = data.get("outputs")
    if isinstance(outputs, list):
        if len(outputs) != 4:
            logger.warning("MIDI configuration must contain four outputs")
        for value, target in zip(outputs, config.outputs):
            read_setting(value, target, output=True)
    else:
        logger.warning("Invalid MIDI outputs in %s", path)
    return config


def save_config(path: Path, config: MidiConfig) -> bool:
    temporary: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".midi-", delete=False
        ) as file:
            temporary = file.name
            data = {
                "version": 1,
                "input": {
                    "device": config.input.device,
                    "channel": config.input.channel,
                },
                "outputs": [asdict(output) for output in config.outputs],
            }
            json.dump(data, file, indent=2)
            file.write("\n")
        os.replace(temporary, path)
        return True
    except (OSError, ValueError) as error:
        logger.warning("Cannot save MIDI configuration %s: %s", path, error)
        return False
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            except OSError as error:
                logger.warning(
                    "Cannot remove temporary MIDI configuration %s: %s",
                    temporary,
                    error,
                )
