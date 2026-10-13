# Device selection and recovery validation

Date: October 12, 2026 (local Mac time). This implements the device selection and
reconnect portion of Phase 2 item 6. Private settings and persona are preserved.
The user requested automated checks only; no physical USB or sleep test was run.

## Software contracts

- Select by exact, case-insensitive name or `HOST::NAME`, filtered by input/output
  capability. Empty/omitted means the OS default at acquisition. Missing or
  ambiguous explicit selections never fall back. Indices are resolved again
  after reconnect; healthy sessions keep their acquired routes.
- Capture inactivity, missing callbacks, large foreground polling gaps or wall
  versus monotonic clock divergence cause a session reset. Forward clock jumps
  can also trigger this conservative behavior. Detection polls during idle
  capture, inference and playback, including clause gaps and output draining.
- Returned device errors unwind playback and capture before backend refresh.
  The runtime cancels the turn, suspends memory work, clears history, VAD and
  buffered input, then reacquires devices with capped exponential backoff.
  Failed turns/external actions are never replayed. Existing completed memory
  work is not rolled back. The normal wake policy applies to the new session.
- Recovery requires a fresh capture callback. Heartbeats continue while waiting;
  authenticated status reports connection failures and recoveries. Recordings
  have one owner across reconnects and retain earlier samples and duration limits.

## Automated evidence

- Full locked unit/process and synthetic evaluation suite: **486 passed**.

- Synthetic tests cover default/explicit selection, missing and duplicate names,
  direction filtering, index reordering, input-only operation, config validation,
  private overrides, failed backend initialization, capture stall/inactivity,
  simulated sleep with both clock behaviors, teardown-before-refresh ordering,
  fresh sessions with/without wake detection, bounded backoff, no fallback/replay,
  recording continuity, output/chime routing and abort-before-producer-cleanup.
- Native Mac enumeration identified the Jabra SPEAK 410 USB as default input and
  output. Its shared name correctly resolves separate input/output entries.
- An isolated native process validated Jabra 16 kHz mono input and 24 kHz mono
  output formats, refreshed PortAudio, resolved the same selectors again, and
  verified a nonexistent explicit input fails without fallback.
- Muted isolated native model/output smoke: three warm turns plus the initial
  turn passed, with zero output underflows, failed turns or worker restarts.
  Synthetic input, real Jabra output writes, no private context and no saved
  recordings. All smoke gates passed, including private-file preservation.
  Local reports are ignored under `tmp/latency-baseline/20261013T012206.697268Z`;
  they are not release artifacts. This exercises playback, not physical recovery.
- Offline wheel/source builds and release checks passed, including artifact
  privacy and version/lockfile consistency. Admin JavaScript syntax checked.
- Before/after hashes of config and knowledge confirmed only the two intended
  public config templates changed. Private config, persona and local settings
  were not edited.

## Implementation constraints and pending hardware gates

The installed sounddevice/PortAudio stack caches its device inventory. There is
no public refresh API in the pinned sounddevice implementation. The voice owner
closes all its streams, then balances private `_terminate()` / `_initialize()`
calls; a failed initialization can be retried without a second termination.
This is isolated to the voice process, with a guard against a shared initialize
count. Revalidate this adapter when upgrading sounddevice. See the
[sounddevice source](https://github.com/spatialaudio/python-sounddevice/blob/master/src/sounddevice.py)
and [PortAudio hot-plug discussion](https://github.com/PortAudio/portaudio/wiki/HotPlug).
PortAudio termination affects open streams, which is why teardown must precede
refresh ([API lifecycle reference](https://portaudio.com/docs/v19-doxydocs-dev/portaudio_8h.html)).

Synthetic faults do not establish CoreAudio's physical disconnect behavior.
When available, run an isolated live session and unplug/replug the selected
device during listening, model generation and speech; verify waiting status,
fresh wake activation, no stale audio or repeated external action, and a new
successful turn. Repeat for sleep/wake and changed OS defaults. Retain an
explicit-device-missing case with another microphone available to verify no
fallback. Native calls that hang instead of returning remain outside in-process
recovery and require restarting the voice process. No interruption timing or
long-soak performance claim is made here.
