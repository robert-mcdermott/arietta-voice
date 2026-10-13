# Speaker-output underflow investigation

October 13, 2026. Follow-up to the four output-underflow flags during one
interrupted answer in the [capture/alignment repeat](CAPTURE_ALIGNMENT_INVESTIGATION.md).
The prior log identifies output underflows as the reset cause, but lacks the
write, producer-wait and capture-poll timings needed to identify their trigger.

## Finding and fix

The blocking output stream remained active during speech-generation gaps without
receiving samples. Once its queued audio drained, the device supplied its own
missing output; the next write reported an underflow. This is consistent with
the documented [PortAudio write contract](https://files.portaudio.com/docs/v19-doxydocs/portaudio_8h.html).
A write flag describes an underflow since the preceding write, not necessarily
a problem caused by the samples being submitted in that call.

The player now supplies explicit zero-valued PCM during producer waits **after
the first speech output**. Each poll may submit at most 20 ms (capped at 1,024
samples), and only when `write_available` reports enough free capacity. No new
queue or startup prebuffer is added. Padding joins the same bounded AEC reference
timeline as speech. Interruption, cancellation, device health and deadlines are
checked before padding; stream completion stops padding, and abort still occurs
before waiting for producer cleanup.

Playback-event consumers poll an empty speech channel every 10 ms. The channel
now respects short consumer timeouts instead of always sleeping for 50 ms.
Other consumers retain their previous 50 ms poll. This is not a hard real-time
scheduling guarantee: host stalls, long foreground work or device faults can
still exhaust the output buffer. Actual underflow flags, reference invalidation
and AEC resets remain active, including on silence writes. Padding can add a
small amount of queued silence before the next clause; acoustic onset and
interruption latency still require measurement. No private setting, buffer
capacity, audio threshold, model, dependency or default backend was changed.

Completed playback events, including passive announcements, report
`output_timing`: write count; explicit gap-silence write/sample counts; maximum
post-start producer wait, capture-poll duration, write duration and gap between
writes. Up to eight underflow contexts per playback include the write index,
speech/silence kind, time since the previous write, accumulated producer-wait
and capture-poll time, and write duration. Further contexts are counted rather
than retained. These are bounded timing metadata, with no speech/audio content.
They establish correlations; a long interval alone does not prove which thread
or host activity caused it.

## Matched native probes

`scripts/output_underflow_probe.py` runs in the existing MLX trial environment
with all four models initialized and the Jabra output selected from copied
settings. It opens no microphone. The disposable profile excludes knowledge,
memory, tools, private persona and credentials. Content logging and recordings
are disabled. Playback is muted unless `--audible` is explicitly supplied.

Before and after the fix, two repetitions each played:

- A fixed 0.6-second, low-amplitude tone twice, with continuous delivery.
- The identical PCM with a 600 ms producer gap between the two copies.
- The continuous PCM with a deliberately injected 350 ms foreground stall.
- A streamed MLX chat/TTS count from one to twenty.
- A streamed MLX chat/TTS three-sentence answer about the sun, moon and stars.

| Case (two repetitions) | Before: underflow flags | After: underflow flags |
| --- | ---: | ---: |
| Continuous fixed PCM | 0 | 0 |
| 600 ms producer gap | 2 (one each) | **0** |
| Deliberate foreground stall | 2 (one each) | **2 (one each)** |
| MLX count | 0 | 0 |
| MLX three-sentence answer | 0 | 0 |

The fixed PCM hashes matched. Both model-answer hashes also matched across
repetitions and before/after runs. With the fix, the producer-gap controls
submitted 20 ms silence blocks and reduced the largest gaps between write calls
from **600.5 / 601.0 ms to 22.4 / 32.7 ms**. The injected foreground stalls still
reported underflows after approximately 353–355 ms without a write, with zero
producer-wait time in those intervals. Thus the fix prevents the reproduced
producer starvation while retaining detection of real scheduling stalls.

The ordinary model answers did not reproduce the historical failure. The probe
does not run AEC/VAD and does not establish the trigger of those four historical
flags. Before/after blocks were sequential, not randomized host-state trials;
these results validate a specific mechanism, not a general underflow rate or a
latency improvement claim. Raw reports and baseline source fingerprints remain
under ignored `tmp/output-underflow-investigation/{baseline,fixed}/` and its
parent directory.

## Live check and regression coverage

The separate three-minute isolated MLX/Jabra session exercises real microphone
capture, AEC, interruption and farewell with a synthetic voice-aware persona.
It completed **184.7 seconds** after readiness and stopped cleanly. The operator
confirmed a clear acknowledgement/uninterrupted count with no self-triggering,
prompt mid-answer interruption without overlap, recovery with six, and farewell
followed by 25 seconds of silence. Logs corroborate **one voice interruption,
one cancellation and one explicit `user_goodbye`**.

Across **eight playbacks and 476 writes**, there were **zero output underflows,
alignment resets, capture-queue drops or native input overflows**. No turn
failures, timeouts, worker restarts or reconnect attempts occurred. All automatic
live gates passed; maximum sampled heartbeat age was 2.0 seconds. Eight expected
startup capture packets were discarded without resets. The largest capture
poll was 13.35 ms, gap between writes 13.50 ms and write duration 50.52 ms.
The maximum post-start producer wait was 0.31 ms, with **no gap-silence writes**.
Thus this live sample validates ordinary playback/AEC/interruption after the
change, but did not exercise padding during a natural generation gap. Forced-gap
device probes and synthetic tests provide that coverage; microphone/AEC behavior
during a forced gap remains a useful additional check.

Raw live reports and separate operator observations remain under ignored
`tmp/hardware-validation/20261013T161935.936118Z/`. The run loaded no private
context and saved no recordings. All 31 protected config/knowledge fingerprints
matched the pre-task snapshot. All 108 production and 134 trial package versions
matched the previously verified capture-investigation inventories. The initial
package snapshot in this task followed interpreter symlinks into base Python
and was invalid for environment comparison; it is retained with that limitation
in the preservation report. The corrected check uses the virtual-environment
paths without resolving those executable symlinks. The production ONNX default
and local settings remain unchanged.

The full suite passes **573 tests**. Added coverage verifies that gap padding
keeps a simulated device fed without altering speech samples or adding a startup
prebuffer; obeys available capacity, cancellation and deadlines; preserves genuine
underflow resets on padding; bounds timing contexts; honors short channel
timeouts; aggregates timing summaries; and preserves the probe's fixed PCM.
Existing pre-roll, inter-clause interruption, device failure and cleanup tests
remain passing.

## Reproduction and remaining limits

Stop the normal runtime and use a fresh ignored output directory:

```sh
uv run --no-project --python tmp/kokoro-integration/venv/bin/python \
  scripts/output_underflow_probe.py \
  --output tmp/output-underflow-probe-new \
  --model tmp/kokoro-experiment/Kokoro-82M-bf16 --audible

uv run --no-project --python tmp/kokoro-integration/venv/bin/python \
  scripts/hardware_validation.py --mode live --minutes 3 \
  --tts-backend mlx \
  --mlx-kokoro-model tmp/kokoro-experiment/Kokoro-82M-bf16 --host-metrics
```

The probe intentionally creates underflows in its foreground-stall control;
`status: completed` means it completed the protocol, not that every control was
underflow-free. Compare individual rows and timing contexts. Likewise, hardware
live gates do not enforce zero underflows; inspect those counts separately.

Longer live/load soaks, physical device disconnect/sleep checks, room/noise
coverage and measured acoustic onset/interruption distributions remain open.
Keep MLX opt-in. For any new spontaneous underflow, inspect the bounded timing
contexts to decide whether producer waits, capture polling, or other foreground
delays warrant further work; do not suppress the flag or relax AEC gap handling.
