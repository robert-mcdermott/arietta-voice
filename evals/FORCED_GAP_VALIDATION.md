# Forced-gap playback, AEC and interruption validation

October 13, 2026. Follow-up to the
[speaker-output investigation](OUTPUT_UNDERFLOW_INVESTIGATION.md), whose live
sample did not encounter a producer gap. This test deliberately exercises gap
silence with microphone capture and AEC active. The production audio path,
ONNX default, private settings and model dependencies are unchanged.

## Isolated protocol

The hardware runner accepts `--forced-gap-seconds` for live profiles only,
with a finite range of 0–10 seconds and AEC/TTS required when nonzero. Its
test-only runtime recognizes exactly **“Start the gap test”** after wake-word
handling (case and punctuation ignored). Other requests use the normal
conversation path. The production application has no such command or setting.

On that test request, MLX Kokoro synthesizes **“The test pause begins now.”**
The harness then withholds continuation PCM for four seconds while yielding
10 ms player polls. It subsequently synthesizes **“Playback has resumed”** and
a count from one to twenty. The existing player keeps its output stream fed
with bounded silence; capture, AEC, pre-roll and interruption remain active.
This emulates unavailable next-clause audio without freezing the foreground
thread. It is not a model-inference failure or an operating-system scheduling
stall. Fixed test phrases bypass chat generation; recovery questions exercise
the normal chat/STT/TTS path.

The four-second interval starts after the lead-in PCM has been submitted, not
at a measured acoustic boundary. Queued lead-in audio and subsequent synthesis
can change the audible pause slightly. No recordings or content logging are
enabled, so acoustic onset/echo attenuation cannot be calculated from this run.

The three operator scenarios are:

For each scenario, say “Bridget,” wait for the acknowledgement, then say
“Start the gap test.” Keeping wake and command separate avoids mistaking a
wake-only acknowledgement for a started test.

1. Request the test and remain quiet through the pause and completed count.
2. Request it again; wait about one second after the lead-in ends, then interrupt **during the pause** with
   “What is three plus three?” Check that the count never starts and recovery
   answers six.
3. Request it again; remain quiet until “Playback has resumed” is followed by
   actual counting. Wait until “five,” then interrupt with “What is five plus five?”
   Check prompt stopping without overlap
   and recovery with ten. Say only “Goodbye” and remain quiet for 25 seconds.

The existing isolated profile copies required model/audio settings, uses a
synthetic voice-aware persona and disables tools, knowledge, memory and content
logging. It refuses an already-running normal runtime, owns only its test
process group, has turn/outer deadlines and a bounded session duration, and
compares protected-file fingerprints after cleanup.

## Evidence and gates

`forced_gap_started`, `forced_gap_ended` and `forced_gap_case_complete` events
record case IDs, pause duration/polls, completion/interruption phase, whether
continuation PCM was reached, cancellation, padding counts, resets and
underflows. An interrupted pause closes before constructing the continuation
stream. Test failures are reported separately. No command transcript or
response text is copied into this metadata.

The report retains at most 32 case summaries and counts omissions. Dedicated
`forced_gap_gates` require an uninterrupted completed case, an interruption
during the gap without continuation, an interruption after continuation starts,
padding in every case, no case failures/omissions, zero output underflows,
alignment resets and capture-queue drops, and an explicit goodbye. These gates
supplement the existing heartbeat, device, worker, turn and content-privacy
gates. Operator evidence remains separate: logs alone cannot prove audible
clarity, no overlapping sound, or correct recovery answers.

## Results

The first 302.6-second session passed the ordinary runtime gates but failed the
dedicated coverage gates. The operator confirmed uninterrupted playback and a
successful interruption/recovery answer. However, the latter interruption arrived
in the lead-in phase, before the controlled gap began; neither a during-gap nor
an after-gap interruption was covered, and no explicit goodbye was observed.
An earlier combined wake/command attempt produced only the wake acknowledgement.
These attempts remain in the local evidence and are not counted as passing the
missing scenarios.

The completed first case injected 4.0005 seconds and exercised 251 silence writes
(120,480 samples at 24 kHz). All seven playbacks had zero output underflows,
alignment resets or capture-queue drops. The audible phrase boundary and generator
boundary differ, so the repeat instructions add about one second before speaking
over the pause. The strict gates are unchanged.

The fresh repeat stopped cleanly after 302.5 seconds, with no recognized test
commands or playbacks. Its ordinary runtime gates passed, but its dedicated
coverage gates failed as expected. It provides no additional playback evidence.
The operator subsequently confirmed they had been away and missed that session.
Both raw reports are retained under ignored `tmp/hardware-validation/` in
`20261013T163610.345754Z` and `20261013T164138.796083Z`; operator observations for
the first session are stored separately.

At that point, live validation was partial. Uninterrupted forced-gap playback passed the
operator check and exercised actual padding with AEC active. Lead-in interruption
and recovery also passed the operator check. During-gap interruption, after-gap
interruption, and explicit goodbye were deferred to the resumed session below.
The ordinary checks above were not substituted for those missing scenarios.

### Resumed guided session

The operator returned for a fresh session with an eight-minute automatic stop.
Raw evidence is retained in `tmp/hardware-validation/20261013T164840.016895Z/`;
`operator.json` records listening observations separately from the automated report.

| Case | Observed phase | Controlled gap elapsed | Silence writes | Outcome |
| --- | --- | ---: | ---: | --- |
| 1 | During gap | 3.3545 s | 161 | Cancelled with no continuation; operator confirmed answer six |
| 2 | After gap, before continuation PCM | 4.0055 s | 214 | Cancelled early; does not cover audible-count interruption |
| 3 | After gap | 4.0007 s | 250 | Full count to twenty; operator confirmed uninterrupted playback |
| 4 | After continuation PCM began | 4.0005 s | 259 | Operator spoke before hearing numbers; retained as an early attempt |
| 5 | After continuation PCM began | 4.0007 s | 256 | Operator heard numbers first, interrupted, received ten, then confirmed farewell and silence |

One intervening spoken attempt did not trigger the exact test command and is not
counted as a forced-gap case. The operator clarified that the missing count in
case 4 followed their question before any numbers were heard; it is not evidence
of spontaneous self-interruption. Case 5 supplies the required listening evidence.
The clarified instructions explicitly wait for the spoken number “five.” Logs
alone cannot distinguish an audible count from queued continuation PCM, which is
why the operator check remains necessary even when automated coverage gates pass.

**Final status: passed for this short live protocol.** The resumed session ran
480.9 seconds and stopped cleanly. All ten ordinary runtime gates and all nine
forced-gap gates passed, with operator confirmation for the three required
scenarios and farewell/silence. Across 21 playbacks and 2,455 output writes,
1,140 writes supplied 547,200 silence samples (22.8 seconds at 24 kHz). Padding
includes next-audio synthesis waits as well as the controlled gaps.

There were zero output underflows, alignment resets, capture-queue drops, input
overflows, failed turns, turn timeouts or worker restarts. Two explicit goodbye
transitions were recorded. Across the two post-goodbye periods, four wake misses
and one 30-second utterance-limit discard were recorded;
no subsequent assistant playback occurred. With content logging disabled, their
audio source is unknown. This preserves the bounded-input observation rather
than treating all discarded audio as a queue fault.

Sampled peak process-tree RSS was 2,763 MiB; this mixed interactive/idle session
is not a memory-growth benchmark. Production code and dependencies were not
changed to obtain the result. The earlier failed-coverage reports remain intact.

All 31 protected config/knowledge fingerprints and the production/trial package
inventories (108/134 packages) matched their starting snapshots. Production
audio/runtime code and the tested harness hashes were unchanged after the runs.
No private context, recordings or local test artifacts are added to Git.

## Automated validation

The full unit/process/synthetic suite passes **590 tests**. New tests cover
invalid mode/range/AEC/TTS combinations, exact command matching, cancellation
and source-generator cleanup before/during/after the gap, no continuation after
a gap interruption, normal routing of recovery questions, real-player capture
handoff during/after an injected gap using fake devices/AEC, strict scenario
gates, and bounded summaries that exclude arbitrary content. Existing device,
underflow, streaming, pre-roll and cancellation regressions remain passing.

## Reproduction and limits

Use the prepared MLX trial environment and cached assets. Stop the normal
runtime, then run from the repository root:

```sh
uv run --no-project --python tmp/kokoro-integration/venv/bin/python \
  scripts/hardware_validation.py --mode live --minutes 8 \
  --forced-gap-seconds 4 --tts-backend mlx \
  --mlx-kokoro-model tmp/kokoro-experiment/Kokoro-82M-bf16 --host-metrics
```

Wait for readiness and perform all three scenarios above. An omitted scenario
fails the dedicated gates; normal conversation alone cannot pass this test.
Raw profiles, reports and operator observations remain under ignored `tmp/`.

This is a short functional check with one speaker, voice and room. Longer
representative live/load soaks, physical unplug/reconnect and sleep/wake checks,
varied noise/room conditions and measured acoustic onset/interruption latency
remain open. It does not prove the trigger of the four historical underflows or
establish a general glitch-free playback rate. Keep MLX opt-in.
