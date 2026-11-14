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
| R | Reset the global beat origin, grid, and all loop phases to zero |
| Esc | Open command entry; Esc in the prompt cancels |
| Q / close window | Shut down the host and release MIDI resources |

Changing channels exits overwrite mode. A recording that is armed remains
armed on the newly selected channel. Transport and recording state are not
controlled by GUI redraws.

## Commands

Press **Esc**, type a command, and press **Enter**.

While entering a command, **Up** and **Down** browse commands successfully
applied during the current application session. The history retains the most
recent 50 original command strings, suppresses consecutive duplicates, and
restores the current draft when moving past the newest entry. Recalled commands
can be edited; edits do not change saved entries. Rejected commands are not
added, and history is cleared when the application restarts.

Command keywords accept unique prefixes in their current grammar context.
Exact keywords take precedence; ambiguous prefixes report their matches, and
unknown prefixes list the valid choices for that position. For example,
`sta ges 6` is equivalent to `start gesture 6`, and `sto ges 5 cyc` is
equivalent to `stop gesture 5 cyc`; `st` is ambiguous between `start` and
`stop`. Prefix matching applies only to command keywords, not counts, musical
lengths, channel numbers, or MIDI port names.

| Command | Action |
| --- | --- |
| `pNNN` | Send a program change, from `000` to `127` |
| `pBBB:PPP` | Send bank MSB and program, each from `000` to `127` |
| `length {beats}[+{n}[/{m}]]` | Set the selected empty channel's loop length |
| `start immediate` | Begin recording as soon as armed |
| `start beat` | Begin at the next eligible beat boundary from the current origin |
| `start beat N` | Align to the next eligible beat, then wait `N` complete beats |
| `start gesture [N]` | Begin on the next or `N`th incoming gesture start |
| `start loop` | Begin at the next wrap of the reference channel |
| `start transport` | Begin on an incoming MIDI Start message |
| `stop immediate` | Stop recording when Enter is pressed |
| `stop beat` | Stop at the next eligible beat boundary from the current origin |
| `stop after {beats}[+{n}[/{m}]]` | Stop after this duration from recording start |
| `stop gesture [N]` | Automatically stop after the next or `N`th incoming gesture ends |
| `stop gesture [N] cycle` (also `cyc` or `cy`) | Record through the `N`th gesture end, then stop at the next incoming gesture start |
| `stop loop` | Stop at the next wrap of the reference channel |
| `stop transport` | Automatically stop on an incoming MIDI Stop message |
| `reference N` | Select reference channel `1`-`4` for loop-synchronized start/stop |
| `reset off` | Disable automatic resets without changing position |
| `reset beat [length]` | Reset at interval multiples from the current beat origin (default: 1 beat) |
| `reset loop` | Reset at each future wrap of the reference channel |
| `transport off\|on\|auto` | Set outgoing MIDI transport behavior |
| `clock off\|on\|auto` | Set outgoing MIDI Clock behavior |

Start and stop modes default to immediate. Configure a reference channel before
using loop-synchronized recording or loop reset; the recording channel cannot
reference itself for synchronized recording. Loop reset requires a defined
positive-length reference loop, which may contain no notes. Changing its
reference channel or phase selects the next future wrap. Clearing the
reference loop so its length becomes undefined disables loop reset. Recording
beat counts align to the next eligible beat boundary from the current global
beat origin; any partial beat before alignment is excluded. A count of `N`
waits `N` full beats after alignment. While stopped exactly on a not-yet-
processed beat boundary, the boundary is eligible when transport starts. Beat
counts pause with transport.

`stop beat` stops at the next eligible beat boundary from the current global
beat origin; numeric counts are not accepted with this command. `stop after`
accepts the same whole and fractional beat syntax as `length` (for example
`stop after 4+3/8` or `stop after 0+1/4`). Its positive duration is converted
to exact 960-PPQN ticks and measured from the actual recording start tick,
without beat alignment. Its timer starts only when recording begins, including
when a synchronized start is used.

Gesture starts count incoming `VOICE_START` events; a gesture already in
progress when recording is armed does not count toward a gesture-synchronized
start. Gesture stops count incoming `VOICE_END` events after recording starts,
including ends from gestures that were already in progress. The start and stop
counters are independent, so the gesture that triggers a start may also
produce the first counted end. `VOICE_CHANGE` events never affect either
counter. A normal gesture stop automatically disarms recording on its Nth end.
Selecting gesture stop or changing its count during recording starts a fresh
count; the end of a gesture already in progress counts. Changing unrelated
settings or only the cycle option preserves the current count.
With `cycle` (or `cyc`), the Nth end completes the recorded gesture but
recording continues through the following silence; the next incoming gesture
start commits the tape at its onset without recording that gesture. The start
still passes through live MIDI articulation. Enter immediately commits and
disarms recording, including while waiting for the cycle-closing start.

Transport start and stop triggers are independently configured. MIDI Start
begins an armed, waiting recording; MIDI Stop commits an active recording.
Enter cancels a waiting start or immediately commits a recording waiting for
MIDI Stop. These messages do not start or stop playback, and MIDI Continue and
Clock do not control recording or timing. Recording notes still requires the
internal playback transport to be running.

Beat reset intervals accept a whole beat count and an optional binary fraction,
for example `reset beat 3+3/8`; `reset beat` and `reset beat 1` both mean every
beat. Denominators must be powers of two, and fractional intervals must convert
to an exact number of ticks at 960 PPQN. Reset events are aligned to positive
multiples of the interval from the current beat origin. Changing the interval
selects the next future multiple and does not immediately reset. A manual reset
establishes a new beat origin and restarts this schedule from that origin.

The `length` command accepts a whole beat count with an optional fractional
part, such as `length 4+1`, `length 3+3/8`, or `length 0+1/64`. Omitted
denominators default to 16; supported denominators are 2, 4, 8, 16, 32, and 64.
Lengths are converted exactly to 960-PPQN ticks. The command requires stopped
transport, an empty selected channel, and no active or pending recording on
that channel. It preserves that channel's phase origin and does not reset
musical position or change other channels. Resizing loops containing notes is
not supported.

The **R** key performs one immediate reset without changing the configured
automatic reset mode. A reset commits active recording at the reset tick,
cancels a waiting start, releases playback notes, preserves live MIDI input,
tapes, tempo, and transport state, and rephases all lanes to zero. It also
establishes the new global beat origin: the beat grid and beat-synchronized
recording align to the reset, regardless of earlier loop or recording offsets.
An automatic beat reset repeats after its configured interval from each reset;
a loop reset waits for a complete reference-loop traversal after each reset.

Outgoing MIDI transport and clock modes are independent, session-local
settings that both default to `off`. `transport on` immediately sends MIDI
Start and ignores later application transport and reset events;
`transport off` immediately sends MIDI Stop and likewise ignores those events.
`transport auto` follows application start and stop, sends Start at each reset
while running, and sends Stop when selected while stopped. If selected during
playback, its initial Start waits for the next 24-PPQN boundary. Transport
messages use MIDI Start, not Continue.

`clock on` continuously sends MIDI Timing Clock at 24 PPQN while the
application is running or stopped. `clock auto` sends clock only while the
application transport is running; selecting it during playback starts at the
next clock boundary, while selecting it when stopped leaves clock output
inactive. `clock off` disables clock output. The clock grid is independent of
whether clock messages are enabled. In `clock on` mode, its phase continues
through transport changes and resets; tempo changes update its rate without
restarting the phase. In `clock auto` mode, clock follows the application's
musical timing. The footer shows each configured mode and whether it is
currently active. These output settings do not change playback, recording, or
the application's transport.

## Recording and display

The title bar shows the internal looper BPM and `MIDI IN BPM`. The input
estimate averages 24-clock MIDI beats over four beats, rounds to the nearest
whole BPM, and requires the two halves of the window to agree within 5%; it
shows `---` before a lock. After two seconds without MIDI clocks, the last
locked value remains visible with `(no clock)`. New clocks clear that indicator
and establish a fresh estimate after four consistent beats. Incoming clocks
never change the looper's tempo or timing.

Recording status is shown as `IDLE`, `WAITING_START`, `RECORDING`, or
`WAITING_STOP`. Waiting starts do not capture notes; recording begins at the
selected start trigger. Beat- and loop-synchronized stops continue recording
until their boundary. `stop after` counts from the actual recording start tick
without beat alignment. Gesture-synchronized stops automatically disarm after
the configured number of incoming gesture ends, or at the next incoming start
when cycle mode is enabled. The gesture that triggers a start is included from
its onset. MIDI input remains available for live audition.

An empty channel's first recording window defines its loop length. Waiting
time before a synchronized start is excluded. Overdub adds recorded notes to
the existing loop; overwrite replaces tape content over the recorded region,
including silence. Stopping the transport pauses pending synchronized
boundaries; they resume when transport starts again.

The configuration line shows zero-based musical beat position rounded to a
1/64-beat subdivision (for example, `BEAT 127:32`), recording status, start and
stop modes and beat counts, reference channel, and reset mode. Rounding is
display-only; the internal tick position remains precise. The second line
shows status messages or the active command prompt.

Timeline beat grid lines follow the global beat origin used by `start beat` and
`stop beat`. A reset places beat zero and every channel's loop phase zero under
the playhead without changing recorded note offsets or loop lengths. Subsequent
loop boundaries diverge according to each loop's length. An immediate start can
still create a fractional-length loop even when its stop is beat-synchronized.

For MIDI device selection and command-line options, run
`python -m py_seq.pygame_gui --help` or see the launch example in
[`readme.md`](../readme.md).
