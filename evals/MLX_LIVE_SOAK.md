# MLX Kokoro live playback and continuous shared-model soak

October 13, 2026. This follows the
[100-turn counterbalanced comparison](KOKORO_100_TURN_COMPARISON.md). ONNX remains
the production default. Tests use the existing separate MLX trial environment,
cached public assets and disposable synthetic profiles. Private configuration,
persona, knowledge, memory, tools and credentials are not modified.

**Outcome:** the operator's live playback, echo/self-triggering and interruption
checks passed, followed by a 900.1-second soak with 390 completed turns in one
worker. Active MLX allocation remained at 5.17 GiB. The initial live and
continuous-load gates are met for this setup; retain MLX as opt-in given the
modest production latency gain and remaining acoustic, capture-drop and longer
load validation.

## Protocol

Run the normal microphone → conversation → speaker path for five minutes with
the integrated MLX adapter. The operator checks an acknowledgement, a clear
uninterrupted count, no self-triggering from speaker output, mid-answer and
early voice interruptions, correct recovery answers, and farewell followed by
silence. Event counts corroborate interruption/cancellation and explicit session
end; they cannot establish audibility or acoustic interruption latency alone.

The test uses the Apple M4 / 32 GiB Mac, Jabra SPEAK 410 USB input/output, Gemma
4 E4B 4-bit, Moonshine 0.0.54, Smart Turn and MLX Kokoro with `af_heart` at speed
1.0. Copied audio settings enable AEC at 16 kHz / 512-sample capture, 200 ms
pre-roll and zero additional AEC delay. The synthetic persona is voice-aware;
tools, knowledge, memory and content logging are disabled. Microphone audio is
processed locally and never recorded.

After the live session stops, start one uninterrupted **15-minute component
load soak** in a fresh inference worker. Cycle all ten shared synthetic question
waveforms through STT, endpointing, chat and complete TTS synthesis, with a
one-second pause between turns. Keep all four models resident and check their
presence, backend and worker PID after every turn. A changed/restarted worker,
missing model or invalid allocation sample fails the run. Memory is sampled
after each completed turn, outside turn timing.

This soak has no microphone or audible playback and uses empty conversation
history. It exercises repeated shared-model inference and generated PCM, rather
than the streamed conversation/player path. The preceding live checks exercise
that path. Neither test replaces an eight-hour/24-hour soak, varied real-room
speech evaluation, device reconnection/sleep tests, or measured acoustic p95
interruption-to-silence latency.

Host telemetry is sampled approximately every five seconds in both runs. Whole
Mac CPU/pressure/swap readings include other applications. Process-tree RSS,
worker high-water RSS and MLX allocator memory overlap and must not be added.
Compare early and late warm windows without treating allocator peaks, startup
caches or whole-host swap as an application leak by themselves.

## Live results

The live session completed **302.4 seconds** after readiness and shut down
cleanly. The operator confirmed:

- An audible acknowledgement and clear count from one to ten, with no cutoff
  or response to the assistant's own speech while remaining quiet.
- Mid-count interruption stopped speech promptly without overlap or stale
  continuation, followed by six for “What is three plus three?”
- An early interruption preserved the request's beginning and produced ten for
  “Please tell me what five plus five is.”
- Goodbye produced a farewell followed by at least 25 seconds of silence.

The event log recorded exactly **two voice interruptions and two turn
cancellations**, one explicit `user_goodbye`, and one idle timeout. All automatic
live gates passed: no turn failures/timeouts, worker failure restarts or native
input overflows; redacted status/events; valid logs; clean shutdown; and unchanged
private-file fingerprints. Maximum sampled heartbeat age was 1.7 seconds.
There were **zero output-underflow flags**, zero reconnect attempts and zero
recording drops; recording was disabled.

There were still **73 application capture-queue drops and 10 alignment resets**.
The queue capacity was 64 chunks. These are distinct from native device
overflows, and the live gates do not require their absence. No lost words were
reported in these operator checks, but the aggregate counters do not explain
each drop/reset or establish glitch-free behavior under other workloads. The
run supports functional echo/interruption behavior for this room and speaker
setup; it does not measure echo attenuation or the acoustic 300 ms p95 target.

Peak sampled process-tree RSS was 2,884 MiB. The first and last warm-window
medians were 2,344 and 979 MiB, respectively; the idle-heavy tail and changes
in resident pages make this unsuitable for a sustained-load memory conclusion.
All 62 host samples were nominal thermal state on AC power, with no telemetry
errors. Memory-pressure levels were 1 in 45 samples and 2 in 17 samples.

## Continuous-soak results

The soak completed **900.1 seconds and 390 turns**, exactly 39 repetitions of
each of the ten questions. One worker PID persisted throughout; every turn's
snapshot confirmed all four models loaded and the MLX backend selected. There
were no worker restarts, empty STT/chat outputs, invalid endpoint probabilities
or non-finite generated PCM. The child exited cleanly, every automatic component
gate passed, and protected-file fingerprints matched. Maximum sampled heartbeat
age was 1.8 seconds. The reported component job time was 1.180 s median and
1.953 s p95; it excludes real-time input, output playback and the one-second
inter-turn pause, and must not be compared with conversation-onset latency.

| Resource measure | Observation |
| --- | --- |
| Active MLX allocation | 5,291.0 MiB (5.17 GiB), effectively constant across all 390 turns |
| Shared MLX allocator peak | 6,579.5 MiB (6.43 GiB), unchanged after the first ten-question cycle |
| MLX cache allocation | 0.25–1.19 MiB; first/last ten-turn median both 0.44 MiB |
| Worker high-water RSS | 3,822.1 MiB (3.73 GiB), unchanged after warmup |
| Five-second sampled process-tree RSS peak | 2,148.5 MiB |
| First/last six-sample warm RSS medians | 2,085.5 / 1,223.4 MiB |

The first and last ten-turn active-allocation medians were identical. After the
first question cycle, minute-level process-tree RSS medians mostly ranged from
1,475 to 1,539 MiB before falling to 1,218 MiB in the final minute. There was no
continuing upward trend in these model-allocation or sampled RSS measures during
the 15-minute window. High-water RSS can exceed the five-second process-tree
sampled peak because the latter misses brief peaks and observes different
resident-page states. Neither metric measures all unified memory in isolation.

All **179 soak host samples** showed nominal thermal state, AC power and
low-power mode off, with zero telemetry errors. Host CPU utilization was 29.8%
median, 39.0% p95 and 58.9% maximum. Memory-pressure level was 1 in 113 samples
and 2 in 66 samples. Whole-host swap was already 13,062 MiB at the first sample
and ended at 13,065 MiB; compressed memory changed from 7,748 to 9,905 MiB.
The fall in process RSS therefore does not establish lower total memory demand.
Host-wide pressure/compression includes unrelated applications and cannot be
assigned to Arietta from these measurements.

No microphone/player was active during the component soak. Its absence of
playback errors is not additional evidence about sustained live AEC or capture.
The loaded-model flags confirm persistent initialized models, not that every
page remained physically resident or that the OS could not compress/swap it.

## Decision and next checks

The subsequent [capture/alignment investigation](CAPTURE_ALIGNMENT_INVESTIGATION.md)
adds reason-specific telemetry, fixes expected startup resets and prevents unused
capture from overrunning during transcription/initial generation waits. See that
report for the separate live repeat and remaining speaker-output underflows;
the historical counts above are unchanged.

Keep the ONNX default and the isolated MLX opt-in. This run closes the initial
integrated listening checks and requested 15-minute shared-model soak. It adds
evidence of correct interruption recovery and stable allocations within this
workload, without changing the earlier modest 3.4% median first-signal benefit
against production ONNX or the additional shared allocator peak.

Before broader adoption, measure acoustic onset/interruption distributions and
echo behavior under varied room/noise conditions; investigate the observed
capture drops/alignment resets; and extend representative continuous live/load
soaks beyond 15 minutes. Device disconnect/sleep validation and eight-hour/24-hour
release soaks remain separate. Leading-silence optimization remains a candidate
experiment with initial-phoneme quality checks. No audio thresholds or private
settings were tuned during this validation.

## Reproduction

Use the [existing pinned trial environment](KOKORO_INTEGRATION.md) and local
Kokoro assets. Stop the normal voice runtime. From the repository root:

```bash
uv run --no-project --python tmp/kokoro-integration/venv/bin/python \
  scripts/hardware_validation.py --mode live --minutes 5 \
  --tts-backend mlx \
  --mlx-kokoro-model tmp/kokoro-experiment/Kokoro-82M-bf16 \
  --host-metrics
```

Wait for `READY: isolated voice runtime is listening.` before the operator
checks. Once that process exits, run:

```bash
uv run --no-project --python tmp/kokoro-integration/venv/bin/python \
  scripts/hardware_validation.py --minutes 15 \
  --tts-backend mlx \
  --mlx-kokoro-model tmp/kokoro-experiment/Kokoro-82M-bf16 \
  --fixtures tmp/kokoro-integration/fixtures.npz \
  --resource-metrics --host-metrics --soak-interval 1
```

Every run prints its fresh directory under ignored `tmp/hardware-validation/`.
Reports include backend/asset evidence, aggregate events, owned-process resource
samples, host readings and protected-file checks. Component resource samples
include every turn's case ID, elapsed time, worker PID, model residency and MLX
active/cache/peak allocations. Human observations are stored separately from
automatic gates. All raw logs, profiles and local results remain ignored.

## Validation and evidence

The locked unit/synthetic evaluation suite passed **556 tests**. Added tests
verify that temporary backend/model overrides preserve the source config and
that the resource sampler rejects worker replacement/restart, missing models,
the wrong backend and invalid memory values. A one-turn native preflight passed
before the live run. Offline package builds and release privacy/version checks
passed for 0.1.0.

All **31 config/knowledge files** and both production/trial dependency inventories
matched their initial values after testing. The application implementation,
private config/persona, credentials, backups, dependency manifest and lockfile
are unchanged. Only the opt-in harness, its tests and documentation changed.

Local evidence manifest and operator confirmations:
`tmp/mlx-live-soak/20261013/runs.json`, `operator.json` and `preservation.json`.
The complete raw run directories are:

- Live: `tmp/hardware-validation/20261013T143203.464339Z/`.
- Soak: `tmp/hardware-validation/20261013T143737.910444Z/`.

The live report retains its standard `human_checks: pending` field because the
harness cannot infer human observations. The separate operator record contains
all three confirmed check groups; automatic gates and operator evidence are
reported separately here.

Measured SHA-256 identifiers:

- Hardware harness: `1b04de6d57aaa776cee9c18a18fbe1f1c8bc7f960b29f9821edbe3a632ec845b`.
- Shared input NPZ: `5d51f83f8d4808c9bc93c0403b02f6cbf5c0e24a04b8e07017e7cebffc154cf0`.
- MLX weights: `4e9ecdf03b8b6cf906070390237feda473dc13327cb8d56a43deaa374c02acd8`.
- MLX voice: `2c1c733b0e6576c810e268d3e440c21dea4e0f0131a3ba4cfc98d7fe6136d094`.
