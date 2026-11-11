# Recording pipeline

The recording path is:

```text
Mido message
  -> adapt_mido_message()
  -> MidiVoicePreprocessor
  -> Recorder (transport ticks)
  -> IntervalResolver (edit plan)
  -> Tape (atomic storage)
```

The modules are independently testable. The preprocessor handles note
selection, the Recorder converts voice activity into timed segments, the
resolver plans circular edits, and Tape stores intervals. This page documents
the implemented behavior.

## MIDI voice preprocessing

[`pre_processor/core.py`](../pre_processor/core.py) defines immutable
`MidiInputEvent` and `VoiceEvent` values. `adapt_mido_message()` in
[`pre_processor/mido_adapter.py`](../pre_processor/mido_adapter.py) maps
positive-velocity `note_on` to `NOTE_ON`, `note_off` and zero-velocity
`note_on` to `NOTE_OFF`, and other message types to `OTHER`.

`MidiVoicePreprocessor(input_channel=0)` processes one configured MIDI channel
in arrival order. It selects the most recently pressed distinct held pitch:

- The first held pitch emits `VOICE_START`.
- Pressing another pitch, or releasing the selected pitch and restoring a
  held pitch, emits `VOICE_CHANGE`.
- Releasing the last held pitch emits `VOICE_END`.
- Duplicate presses for an already-held pitch, unmatched releases, non-selected
  pitch releases, other MIDI messages, and messages from other channels emit
  no voice event.

Events within one continuous held-note gesture share a monotonically
increasing `gesture_id`, starting at 1. `reset()` clears held-note state
without emitting an event and does not reuse IDs. An optional `timestamp_ns`
is caller-supplied metadata; it is not used to order events or derive transport
ticks.

## Tape and intervals

[`tape/core.py`](../tape/core.py) stores immutable `VoiceInterval` values on a
circular, monophonic timeline. An interval occupies the half-open range
`[start, start + duration)` modulo the tape length. Wrapped and full-length
intervals remain single stored values. Stored intervals cannot overlap;
adjacent intervals are allowed.

`Tape.intervals()` and `Tape.snapshot()` return deterministic immutable views.
`at()`, `overlaps()`, and `by_gesture()` query occupancy and provenance.
`insert()`, `remove()`, `replace()`, and `clear()` mutate storage. A
`TapeEdit` is applied atomically: Tape validates the resulting contents before
committing, assigns IDs to new intervals, and can preserve removed IDs for
surviving fragments. Failed edits do not partially change the tape.

A tape may start at length zero while its first loop is being defined by
`FourChannelSession`. During that pass the session grows the provisional
length; it finalizes the length when the recording window ends. Clearing an
empty loop can return it to zero length. A zero-length tape has no queryable
range and cannot accept intervals.

## Recording and circular edits

[`recorder/core.py`](../recorder/core.py) accepts `VoiceEvent` values with
nondecreasing, unwrapped transport ticks. It tracks selected voice separately
from recording state. A segment is captured only when recording is armed, the
transport is running, and a voice is selected. Segment duration is capped to
one tape loop; natural loop wrap is not a segment boundary. `seek()` handles
transport discontinuities explicitly.

`OVERDUB` and `OVERWRITE` are the implemented `RecordMode` values. In
overwrite mode, elapsed silence is erased and a completed voice segment
replaces its covered region. The Recorder sends edits through
[`interval_resolver/core.py`](../interval_resolver/core.py); the resolver
returns a `TapeEdit` and never mutates Tape itself. It preserves surviving
fragments of intersected intervals, omits old fragments at or below the
configured threshold, and does not suppress incoming intervals. An
`EraseRegion` represents overwrite silence.

`FourChannelSession` supplies the application-level recording window: immediate,
beat, reference-loop, gesture, or incoming MIDI transport start; and immediate,
beat, duration, reference-loop, gesture, or incoming MIDI transport stop.
Waiting for a synchronized start does not capture notes. MIDI transport
messages only trigger recording; they do not control playback or timing.
The Pygame guide describes those controls and the reference-channel rules.

See [`tests/test_pre_processor.py`](../tests/test_pre_processor.py),
[`tests/test_recorder.py`](../tests/test_recorder.py),
[`tests/test_interval_resolver.py`](../tests/test_interval_resolver.py),
[`tests/test_tape.py`](../tests/test_tape.py), and
[`tests/test_recording_pipeline_integration.py`](../tests/test_recording_pipeline_integration.py).
