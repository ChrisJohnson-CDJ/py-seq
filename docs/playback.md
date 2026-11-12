# Playback and transport

The playback path is:

```text
Tape -> PlaybackScheduler -> Articulator -> MidiOutputAdapter -> MIDI sink
```

`FourChannelSession` builds four independent paths and coordinates them on one
transport. Each channel has its own tape length and phase. The internal clock
uses 960 ticks per beat. Playback scheduling is tick-driven; it does not depend
on the Pygame redraw rate.

## Scheduling

[`playback_scheduler/core.py`](../playback_scheduler/core.py) implements a
pure scheduler for one tape. It reports `ONSET`, `END`, `ENTER_ACTIVE`, and
`EXIT_ACTIVE` events; it does not send MIDI.

- `start(tick)` and `seek(tick)` report occupancy at the selected position.
  Occupancy at an interval's designated start is an `ONSET`; starting inside
  an interval is `ENTER_ACTIVE`.
- `advance(tick)` reports every interval boundary in `(previous_tick, tick]`,
  including boundaries crossed by a large advance. At coincident boundaries,
  an interval end precedes the next onset.
- Equal-tick advances emit no events. Backward movement uses `seek()`.
- A scheduler advance is limited to 100,000 events. Larger advances raise
  `PlaybackEventLimitError` without advancing scheduler state.
- `reconcile(tick)` refreshes the scheduler after tape edits at its current
  tick. A changed or removed active interval exits; newly occupied space is
  reported as `ENTER_ACTIVE`, not as a fabricated onset.

An interval crossing the tape wrap remains active across the wrap. A
full-length interval retains its designated onset and does not retrigger just
because the loop wrapped.

## Articulation and MIDI output

[`articulator/core.py`](../articulator/core.py) turns scheduler events and
optional live input into semantic note actions. Playback sounds at genuine
onsets; entering an already-active interval does not start a note. A new
playback onset releases any previous note before retriggering. Live input takes
priority on the focused channel. When live input ends, playback does not
resume in the middle of an interval; it waits for a later onset.

[`midi_output/core.py`](../midi_output/core.py) maps note actions to Mido
messages on a configured MIDI channel. `CollectingMidiSink` supports tests
without hardware; `MidoOutputPortSink` wraps an output port. A send failure
marks the adapter desynchronized and raises `MidiOutputError`. Call `panic()`
to send All Notes Off and clear that state after successful recovery.

## Session and host runtime

[`py_seq/four_channel_session.py`](../py_seq/four_channel_session.py) owns
four tapes, MIDI routes, recording coordination, loop phases, and transport
controls. It also handles live MIDI input, program changes, manual and
automatic phase resets, and orderly shutdown.

[`playback_coordinator/core.py`](../playback_coordinator/core.py) coordinates
each channel's recorder, scheduler, articulator, and output. It accepts a
non-empty tuple of channels whose recorders all start at the same tick.
The optional
[`py_seq/gui_runtime.py`](../py_seq/gui_runtime.py) advances the session on a
host worker and hands immutable snapshots to the UI. Each snapshot identifies
the latest processed command, allowing the UI to retire optimistic state
after the corresponding authoritative state is published. Its clock and
playback work are independent of GUI rendering.

See [`pygame_gui.md`](./pygame_gui.md) for user controls and command syntax.
Relevant tests are in
[`tests/test_playback_scheduler.py`](../tests/test_playback_scheduler.py),
[`tests/test_playback_coordinator.py`](../tests/test_playback_coordinator.py),
and [`tests/test_four_channel_session.py`](../tests/test_four_channel_session.py).
