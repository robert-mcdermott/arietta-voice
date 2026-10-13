# AEC alignment, pre-roll and interruption validation

October 12, 2026. Production remains on the locked model stack, including
Moonshine 0.0.54. No private config, persona, credentials, knowledge, memory or
backups are rewritten by these changes or the isolated validation.

## Changes and boundaries

- Microphone queue entries carry ADC capture times translated from PortAudio's
  device clock into the monotonic clock. Ordinary consumers retain the existing
  NumPy interface. Callback timing, rather than delayed dequeue timing, determines
  which playback samples are matched. Callback-free fixtures estimate capture
  start from callback arrival minus chunk duration.
- The bounded reference contains only submitted output blocks, with their
  estimated presentation times. Long TTS chunks cannot evict their own beginning
  before it is played. Clause gaps remain silence. Blocking writes do not expose
  DAC timestamps: reported device latency and queued sample duration provide an
  estimate. `audio.aec_delay_ms` optionally adds a measured residual delay, with
  a zero default and a 500 ms upper bound; this is not automatic calibration.
- AEC consumes uninterrupted 10 ms input/reference frames. Previously, every
  512-sample/32 ms microphone block padded its last frame, advancing the native
  processor by 40 ms. Residual samples now carry between calls. A fixed 10 ms
  output delay keeps VAD input at 512 samples; the remaining cleaned samples
  are flushed into the interruption handoff.
- Listening keeps exactly up to `audio.pre_roll_ms` (default 200, range 0–500)
  before speech. Barge-in retains that pre-roll rounded up to capture chunks,
  plus the consecutive positive-VAD chunks and the AEC flush. Handoff uses cleaned
  samples rather than replaying the raw speaker echo. Pre-roll remains in RAM,
  counts against the utterance limit and is cleared at discontinuities and
  session/turn boundaries. This does not enable recording or transcript logging.
- AEC/VAD and pending prefix reset on capture loss, timestamp discontinuities
  over 20 ms, malformed capture frames and output underflow. Stale capture older
  than the reference window is discarded. If capture overflows during inference
  cleanup after interruption, the preserved prefix is discarded too, rather than
  joined to noncontiguous audio. Playback events expose reset/underflow counts.
- Cancellation is polled between 1024-sample output writes, on producer wait
  events (every 50 ms), and while estimated device output drains (10 ms polls).
  Device abort/close occurs before generator cleanup. This fixes keyboard gaps
  before the first audio and between clauses as well as the final playback tail.
  Voice interruption still needs AEC, VAD and an existing output stream; it is
  not a microphone-based cancel during STT/model startup.

These choices follow the documented [PortAudio timestamp and blocking-stream
interfaces](https://python-sounddevice.readthedocs.io/en/latest/api/streams.html)
and [LiveKit's 10 ms APM frame/delay contract](https://docs.livekit.io/reference/python/livekit/rtc/apm.html).
The pre-aligned reference is passed with zero additional native stream delay.
Device drift, room impulse response, double-talk quality, chime echo and
hardware-buffer cancellation tails still need acoustic measurement. A stalled
native device write cannot be interrupted by these Python polling checks.

## Automated evidence

`uv run --locked pytest` passed **460 tests**. New/updated synthetic tests cover:
timestamp retention under queue delays/overflow; bounded resampled references
with real gaps; exact 10 ms framing and flush sample continuity; cleaned bounded
pre-roll; capture-gap and underflow reset; keyboard cancellation before output,
between clauses and in the final drain; and capture overflow during producer
cleanup. Existing tests retain abort-before-generator-close, cancellation-token,
queue bounds, partial-turn exclusion and recovery coverage.
Additional player/producer integration tests hold native cleanup pending and
prove device abort occurs first on cancellation, provider failure and keyboard
interruption during completion cleanup. Playback-only failure/heartbeat events
return control to the player before joining its producer; other stream consumers
retain exception semantics.

A separate locked-native check processed three seconds of synthetic 16 kHz
audio through LiveKit AEC without opening an audio device. All output was finite,
and 48,000 input samples yielded 48,160 output samples including the explicit
160-sample framing delay and final flush. This establishes native compatibility
and continuity, not an echo-reduction or speech-recognition quality score.

## Live check

The isolated hardware harness is used for a five-minute microphone/speaker check:

```bash
uv run --locked scripts/hardware_validation.py --mode live --minutes 5
```

It uses the synthetic voice-aware persona, default wake phrases, no private
knowledge/memory/tools, no recordings and no content logging. Audio settings
are copied into the temporary profile, including the new default pre-roll/delay
when omitted by the source config. Reports and logs remain in ignored `tmp/`.

The five-minute check completed on the Apple M4/Jabra SPEAK 410 USB with AEC
enabled, 16 kHz/512-sample capture, 200 ms pre-roll and zero additional AEC delay.
The operator confirmed:

- Interrupting a count to twenty stopped counting without overlapping speech
  and produced six for “What is three plus three?”
- An uninterrupted count to five finished while the operator stayed quiet.
- Interrupting near the start of another count produced ten for “Please tell
  me what five plus five is.”
- A final goodbye produced a farewell followed by silence.

The harness recorded 304.1 seconds, clean shutdown and passing existing live
gates: no turn failures/timeouts/worker restarts or input-device overflows;
heartbeat remained within five seconds; no content appeared in status/events;
config and knowledge fingerprints were unchanged. There were three logged voice
interruptions/cancellations and one explicit user-goodbye session end.

**Limits observed:** 309 application capture-queue chunks were dropped, 12
alignment resets were reported and one output-underflow flag was recorded.
These are not input-device overflows. Existing live gates do not require zero
application drops or output underflows, so passing them must not be described as
glitch-free audio. Resets include startup-boundary capture and discontinuities;
the aggregate counts do not establish the cause of every reset or drop. The
operator checks passed, but load testing must investigate these observations.
Peak sampled process-tree RSS was 2718.0 MiB, with 318.0 MiB warm-start-to-end
growth; this short run does not establish long-term memory stability.

The live run exercised the AEC/pre-roll path before the final producer-cleanup
ordering fix. That fix has dedicated integration tests; a separate muted
three-turn runtime check passed on the final code through the actual output
device, with zero output underflows, failed turns or worker restarts and unchanged
private fingerprints. This is a functional smoke, not a replacement 100-turn
latency baseline. Its first attempt exposed a missing `latency` property on the
benchmark's output wrapper; forwarding the real device property fixed it, and
the repeat passed all gates. The real live stream was unaffected by that wrapper
issue. The passing muted report is in `tmp/latency-baseline/20261013T010720.795772Z/`.
Live evidence is under `tmp/hardware-validation/20261013T005953.767739Z/`,
including separate operator confirmations. No private content was recorded.

## Remaining gates

1. Use loopback/acoustic recording in an explicitly authorized calibration run
   to measure speaker onset, echo delay and interruption-to-silence distribution.
   Count false interruptions and clipped first words across varied voices and
   background noise, with/without AEC and under simultaneous model/TTS load.
2. Re-run the 100-turn baseline and measure false cutoffs. Software poll intervals
   and consecutive-VAD thresholds do not establish the 300 ms acoustic p95 target.
3. Add device selection and sleep/disconnect/reconnect handling. A callback-based
   output path with actual DAC timestamps remains a possible follow-up if the
   current latency estimate is insufficient on the supported hardware.

To disable pre-roll, set `audio.pre_roll_ms = 0` in a private override. Leave
`audio.aec_delay_ms = 0` unless a residual delay has been measured.
`audio.aec_enabled = false` disables voice barge-in, as before; keyboard cancellation remains.
No settings are automatically tuned or migrated by this implementation.
