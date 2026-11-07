# Pygame interface

The Pygame interface shows four MIDI loop lanes on a shared transport. The
session and host worker own timing and recording state; the GUI submits
commands and renders snapshots. The display is not the playback clock.

## Keys

| Key | Action |
| --- | --- |
| Space | Start or stop the transport |
| Tab | Select the next channel |
| Enter | Arm recording, cancel a waiting start, stop recording, or cancel a pending stop |
| Backspace | Toggle overwrite for the selected channel; enabling overwrite also arms recording |
| Delete | Clear the selected channel's tape |
| D | Toggle loop-length display between beats and seconds |
| R | Reset global beat position and all loop phases to zero |
| Esc | Open command entry; Esc in the prompt cancels |
| Q / close window | Shut down the host and release MIDI resources |

Changing channels exits overwrite mode. A recording that is armed remains
armed on the newly selected channel. Transport and recording state are not
controlled by GUI redraws.

## Commands

Press **Esc**, type a command, and press **Enter**.

| Command | Action |
| --- | --- |
| `pNNN` | Send a program change, from `000` to `127` |
| `pBBB:PPP` | Send bank MSB and program, each from `000` to `127` |
| `start immediate` | Begin recording as soon as armed |
| `start beat` | Begin at the next global beat boundary |
| `start loop` | Begin at the next wrap of the reference channel |
| `start note` | Begin on the next incoming note onset |
| `stop immediate` | Stop recording when Enter is pressed |
| `stop beat` | Stop at the next global beat boundary |
| `stop loop` | Stop at the next wrap of the reference channel |
| `reference N` | Select reference channel `1`-`4` for loop-synchronized start/stop |
| `reset off` | Disable automatic resets without changing position |
| `reset beat N` | Reset every `N` whole musical beats |
| `reset loop` | Reset at each future wrap of the reference channel |

Start and stop modes default to immediate. Configure a reference channel before
using loop-synchronized recording or loop reset; the recording channel cannot
reference itself for synchronized recording. Loop reset requires a defined
positive-length reference loop, which may contain no notes. Changing its
reference channel or phase selects the next future wrap. Clearing the
reference loop so its length becomes undefined disables loop reset. Beat
resets are measured from the musical position when configured.
Pausing the transport pauses automatic reset timing.

The **R** key performs one immediate reset without changing the configured
automatic reset mode. A reset commits active recording at the reset tick,
cancels a waiting start, releases playback notes, preserves live MIDI input,
tapes, tempo, and transport state, and rephases all lanes to zero. An automatic
beat reset repeats after each configured interval; a loop reset waits for a
complete reference-loop traversal after each reset.

## Recording and display

Recording status is shown as `IDLE`, `WAITING_START`, `RECORDING`, or
`WAITING_STOP`. Waiting starts do not capture notes; recording begins at the
selected start trigger. Waiting for a synchronized stop continues to record
until that boundary. A note-start captures the triggering note from the
recording window's beginning. MIDI input remains available for live audition.

An empty channel's first recording window defines its loop length. Waiting
time before a synchronized start is excluded. Overdub adds recorded notes to
the existing loop; overwrite replaces tape content over the recorded region,
including silence. Stopping the transport pauses pending synchronized
boundaries; they resume when transport starts again.

The configuration line shows zero-based musical beat position rounded to a
1/64-beat subdivision (for example, `BEAT 127:32`), recording status, start and
stop modes, reference channel, and reset mode. Rounding is display-only; the
internal tick position remains precise. The second line shows status messages
or the active command prompt.

For MIDI device selection and command-line options, run
`python -m py_seq.pygame_gui --help` or see the launch example in
[`readme.md`](../readme.md).
