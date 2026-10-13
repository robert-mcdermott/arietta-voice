# Capture-queue and AEC alignment investigation

October 13, 2026. Follow-up to the [MLX live test](MLX_LIVE_SOAK.md), which
recorded 73 application capture-queue drops and 10 alignment resets across nine
playbacks, despite passing operator checks and reporting zero device overflows
or output underflows. ONNX remains the default; the MLX environment and private
settings are preserved.

The subsequent [speaker-output investigation](OUTPUT_UNDERFLOW_INVESTIGATION.md)
reproduces producer-gap starvation with matched device controls, adds bounded
gap silence and write/wait timing telemetry, and records a separate live repeat.
Historical results below remain unchanged.

## Findings and changes

Two behaviors can be reproduced independently of models or hardware:

1. Microphone ADC packets arriving just after output starts can contain samples
   captured before that boundary. The player discarded these packets but also
   rebuilt its fresh AEC and counted an alignment fault for each one. It now
   counts the expected startup discard separately. If capture has already begun
   and a timestamp moves behind the boundary, it still invalidates the prefix.
2. While awaiting first generated audio, the player polled cancellation but
   allowed unused microphone packets to accumulate. At 512 samples / 16 kHz,
   the unchanged 64-chunk queue holds 2.048 seconds. A long generation wait could
   therefore overrun it, even though startup then drained that capture. The
   player now retires it on each producer-wait poll. Foreground transcription
   waits also retire capture that the existing turn boundary would discard.

This does not extend voice interruption to initial generation. Capture is kept
during endpoint decisions, since an incomplete utterance may continue; during
playback and gaps between clauses, AEC and pre-roll remain active. No capture
threshold, queue limit, audio setting or model dependency was changed.

Actual application queue overruns retain the original `dropped_chunks` counter
and add `drops_by_phase`: listening, endpoint, transcribe, awaiting_output,
playback or playback_cleanup. Intentional queue removals have separate
`discarded_chunks` / `discarded_by_reason` counters. Recording still receives
callbacks independently, when explicitly enabled. Bounded draining processes a
snapshot of the queue rather than chasing newly arriving callbacks indefinitely.

Playback events now report `alignment_reset_reasons` (timestamp gap, capture
overflow, malformed/stale capture or output underflow) and
`capture_discard_reasons` (startup, malformed or stale packets). Those per-player
packet counters are separate from microphone queue discards. The admin status
labels queue overruns and intentional discards separately. Logs contain counts
and fixed reason labels, not speech content.

The historical log lacked reasons. Its nearly one-reset-per-playback pattern
is consistent with the startup behavior, and unused capture waits can explain
queue pressure, but neither proves the cause of every historical event.
Historical results remain unchanged.

## Automated validation

The complete suite passes **565 tests** (556 previously). New regressions cover
expected startup packets without rebuilding AEC, a 3.2-second generation wait
without queue overrun, real overflow invalidating an interruption prefix,
bounded intentional draining without losing explicit recording callbacks,
foreground transcription versus endpoint/producer-thread capture ownership,
and reason aggregation with backward-compatible older logs. Existing timestamp
gap, output underflow, cleaned pre-roll, inter-clause interruption and producer
cleanup loss tests remain passing.

## Live repeat

The repeat uses the existing five-minute isolated MLX hardware protocol, Apple
M4 / 32 GiB and Jabra SPEAK 410 USB, copied audio settings, a synthetic voice-aware
persona and no tools, knowledge, memory, recordings or content logging. Operator
checks cover uninterrupted speech, self-triggering, mid-answer and early
interruptions, arithmetic recovery and farewell/silence.

The session completed **302.7 seconds** after readiness and shut down cleanly.
All automatic live gates passed. The operator confirmed all requested checks,
including a repeat of the early interruption/farewell sequence. The original
early check was not corroborated by a second interruption or explicit goodbye
in the event log at that point; the repeat brought the final counts to **two
voice interruptions, two cancellations and one explicit `user_goodbye`**. There
was also one idle timeout. Operator observations and event counts remain
separate evidence; content logging was disabled.

| Measure | Previous live run | This repeat |
| --- | ---: | ---: |
| Playback completion events | 9 | 14 |
| Application queue drops | 73 | **0** |
| Native input overflows | 0 | **0** |
| Alignment resets | 10, reason unknown | 4, all output underflow |
| Output underflow flags | 0 | **4** |
| Expected startup packets discarded | Not distinguished | 15 |
| Timestamp-gap / capture-overflow resets | Not distinguished | **0 / 0** |
| Turn failures / timeouts / worker restarts | 0 / 0 / 0 | **0 / 0 / 0** |

These are descriptive counts, not matched rates: the conversation, number of
playbacks and host scheduling differed. The 15 startup packets no longer
rebuild fresh AEC instances. **924 queue packets** were intentionally retired:
785 while awaiting first output, 90 during transcription and 49 at turn
boundaries. These are capture windows the prior implementation already drained;
they are not 924 newly lost utterance packets. The separate startup-packet
counter is not included in that queue-discard total.

The four output underflows occurred within one interrupted answer. Each caused
the intended reference/AEC reset. The operator reported no audible problem,
but device underflow flags are still genuine faults and prevent a claim of
glitch-free playback. Existing evidence does not distinguish producer starvation,
foreground write/poll scheduling or host contention. A follow-up should time
producer waits and device writes, then repeat a matched spoken workload before
changing buffering or thresholds. No underflow was suppressed or reclassified.

Maximum sampled heartbeat age was 2.2 seconds. There were no reconnect attempts
or recording drops. All 31 protected configuration/knowledge files matched the
pre-task fingerprints; production and trial package-version inventories also
matched. Raw profiles, redacted events/status, host/resource samples and operator
observations remain in ignored
`tmp/hardware-validation/20261013T160114.614543Z/`; preservation snapshots and
build evidence are under ignored `tmp/capture-investigation/20261013/`.

## Reproduction and limits

Use the existing pinned trial environment and cached model assets. Stop the
normal runtime, then run from the repository root:

```sh
uv run --no-project --python tmp/kokoro-integration/venv/bin/python \
  scripts/hardware_validation.py --mode live --minutes 5 \
  --tts-backend mlx \
  --mlx-kokoro-model tmp/kokoro-experiment/Kokoro-82M-bf16 --host-metrics
```

Inspect `report.json` for actual queue drops and phase counts under
`reliability.audio`, and reset/discard reasons under `events`. Intentional
discard counts are not recovered speech or evidence of an expanded listening
window. The standard live gates still permit queue drops and alignment resets;
inspect these separately rather than treating a passed gate as glitch-free audio.

A brief live repeat cannot establish rates across workloads or prove every old
drop was fixed. Acoustic echo attenuation, measured interruption latency,
physical disconnect/sleep validation and longer representative live/load soaks
remain open. Do not increase queue limits or relax gap detection merely to hide
counters.
