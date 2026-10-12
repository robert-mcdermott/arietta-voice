# Phase 2 latency and resource baseline

This benchmark establishes a repeatable no-tool latency baseline on the intended
Mac and output device before streaming changes. It uses real models, paced
synthetic speech input, and the existing runtime listening/playback code. The
measurement boundary is **synthetic waveform end to first PCM write to the
output device**. This is a software measurement, not a calibrated acoustic
speech-end-to-first-sound measurement.

## Repeating the benchmark

Stop the normal voice runtime, then run from the repository:

```bash
# 100 measured warm turns, plus one separately reported first turn.
# Answers and normal thinking chimes play through the default output device.
uv run --locked scripts/latency_baseline.py --turns 100 --audible

# The same real output-device path with zero-valued samples instead of sound.
uv run --locked scripts/latency_baseline.py --turns 100
```

The script accepts `--config /path/to/base.toml` and `--idle-seconds` (default
20). A quick harness check can use `--turns 2 --idle-seconds 0`; that is not a
100-turn baseline. Limits are 200 warm turns and a 30-minute outer deadline.
Ctrl-C stops only the test-owned process group. The harness shares the hardware
validation lock and rejects an active normal-runtime heartbeat.

## Protocol and privacy

The harness builds an isolated config using the existing hardware test profile.
It reads selected local model, audio and timeout settings but does not copy the
private persona, credentials, knowledge, memory, custom tools or conversation
history into the test. Temperature is zero, the persona is synthetic and voice
aware, and the existing model/token limit is preserved. Fingerprints compare
config and knowledge before/after without copying their content into reports.
Local files are never migrated or rewritten. No microphone is opened and no
ambient speech is collected. It uses the existing locked model stack and the
same cache/download policy as [hardware validation](HARDWARE_VALIDATION.md).

Ten fixed questions in `scripts/latency_baseline.py` cover short arithmetic,
facts and two one-sentence explanations. Kokoro generates input fixtures in
memory. After resampling to the capture rate, trailing samples below -60 dB of
peak are trimmed; the end of that waveform defines input speech end. Fixture
hashes and durations are recorded. This definition differs from a human-labeled
speech endpoint.

A fresh runtime/model worker is started after fixture preparation. Its load time
is reported separately as **cached process startup**: OS/model file caches are
already populated, so it is not a cold boot or download benchmark. After the
idle measurement window, one first turn is reported separately. The 100 warm
turns contain ten repetitions of each question, shuffled with seed 1729. Every
turn starts a fresh, explicitly active session with no history; fixture audio
contains only the question. Wake-prefix recognition is excluded from this warm
conversation benchmark.

Input chunks are paced in real time through the existing `_listen` method. Its
VAD, silence threshold, Smart Turn decisions, STT, chat, speech
sanitization, TTS, chime handoff and output writes execute normally. There is no
competing speech during playback. The synthetic input adapter has no capture
queue/backlog; its drain does not discard the next fixture. Microphone transport,
real-room noise, device capture latency and barge-in performance are outside
this protocol. Those still require physical audio tests.

Audio output uses the actual default `sounddevice.OutputStream`, with its normal
buffering and blocking writes. Audible mode preserves sample amplitudes and does
not change system volume. Muted mode writes zeros to the same device, including
chimes. Changing audio devices, model/config, power/load conditions, input
fixtures or measurement boundaries makes runs less directly comparable.

## Metrics

- **Speech end to first write:** paced input waveform end to the first reply PCM
  write call; includes silence/VAD endpoint wait, endpoint inference, STT, chat,
  first TTS generation, chime handoff and output stream setup.
- **Speech end to first signal write:** first reply block containing samples
  above an absolute 0.0001 amplitude. In muted mode this refers to the unmuted
  data before replacement. It is still a write timestamp, not speaker onset.
- **Speech end to processing:** endpoint decision delay before STT begins,
  including configured silence and any repeated endpoint rejection.
- **Speech end to TTS ready:** time until the first generated TTS chunk arrives.
- **Stage times:** synchronous worker RPC durations for endpoint, STT, full chat
  generation, TTS start and TTS-next calls. All calls per stage are summed;
  TTS-next totals include later chunks, so stage totals do not sum to first-audio
  latency. First-chunk TTS-next duration is also reported separately. IPC and
  scheduling overhead are included.
- **Output setup:** stream construction/start duration and the device-reported
  output latency. The latter is metadata, not a calibrated correction for
  buffering. No acoustic latency is inferred by adding it to write timestamps.
- **Resources:** five-second samples of process-tree RSS and `ps` CPU percentage,
  split by fixture preparation, startup, idle and measured turns. RSS is not
  total Metal/unified memory. CPU values are operating-system samples, may exceed
  100% across cores, and are not instantaneous stage utilization or energy use.

Reports use NumPy's default linearly interpolated p50/p95, plus maximum and
per-question counts/quantiles. The first turn is excluded from warm aggregates.
Missing STT/chat/output boundaries or recovered failures invalidate a turn;
missing turns, underflows, worker restarts, private-file changes and child errors
fail the run rather than silently shrinking the denominator. The fixed questions
are timing fixtures, not a representative answer-quality or transcription test.
Each turn records whether STT matches its fixture after lowercasing and replacing
punctuation with spaces. Number formatting and wording changes count as a
mismatch; this diagnostic is not semantic accuracy and does not grade answers.

## Local artifacts and recorded results

Ignored `tmp/latency-baseline/<UTC timestamp>/` contains `environment.json`,
`report.json` and `resources.json`, plus redacted runtime events/status. Reports
contain case IDs, lengths, timings, hashes and counters, not transcripts or
answers. No input or output audio is saved. Only reviewed aggregate results
belong in this document; raw local artifacts should not be committed.

## October 12, 2026 run environment

The measured run used an Apple M4 with 32 GiB RAM, macOS 15.7.9 and Python
3.13.3, on AC power. Jabra SPEAK 410 USB was the default output device and
answers were audible. The Jabra was also the configured default input, but
input was synthetic replay; no physical microphone was opened. Other desktop
workloads were not controlled, and CPU/thermal conditions were not normalized.

App version was `0.1.0`, application commit `22e4760`, with the new benchmark
script added locally. The lockfile was unchanged:
`66e417fd303b79a832585a55a08b4fe1c0b97fef14c661e764404dd868543f0c`.
Model/package revisions match the [hardware baseline](HARDWARE_BASELINE.md):
Gemma 4 E4B 4-bit, Moonshine 0.0.54 / `medium-streaming-en`, and Kokoro
`af_heart`. The response cap was 220 tokens; temperature was zero. Capture
format was 16 kHz / 512 samples, silence setting 700 ms, Smart Turn and thinking
chimes enabled, TTS speed 1.0. AEC setup was enabled, but this protocol supplied
no competing microphone audio during playback.

The corpus SHA-256 was
`9917afd265d5af280a7ac0e7dff7a0f93bb97bb9c0f84985fbbac0477a576c7e`;
benchmark script SHA-256 was
`324e8203e301e6034a15e6017bef7692ba0b5036328d6ad254261ddf5a4892d9`.
Raw results reside locally in `tmp/latency-baseline/20261012T232239.970669Z/`.

### Pilot finding

A preliminary wake-prefixed run stopped after 16 valid warm turns because the
ice question produced a `wake_miss` after transcription, with no reply. That
attempt is retained locally as a failed run and excluded from the final baseline.
The exact recognition failure was not diagnosed; the runtime's VAD onset capture
and wake recognition warrant a separate labeled speech test before claims about
real-world wake accuracy. The final protocol explicitly opens each fresh session
to measure active-conversation latency. All ten questions passed a pilot with
that protocol before the 100-turn run. No application behavior or private setting
was changed to make the benchmark pass.

### Final measured baseline

The final run **passed**, with 100 valid warm turns (ten per question), plus one
separately reported first turn. Total elapsed time was 723.5 seconds, including
fixture preparation, fresh-worker warmup and a 20-second idle window. The
fresh inference-worker warmup took **6.112 seconds** with cached model assets.
The first turn took **3.247 seconds** from waveform end to first reply write.

| Warm metric | p50 | p95 | Maximum |
| --- | ---: | ---: | ---: |
| Speech end → first reply write | **2362.5 ms** | **3496.6 ms** | 3543.9 ms |
| Speech end → first signal-bearing write | 2362.5 ms | 3496.6 ms | 3543.9 ms |
| Speech end → first TTS chunk ready | 2248.6 ms | 3382.6 ms | 3435.2 ms |
| Speech end → STT processing entry | 713.7 ms | 731.8 ms | 743.8 ms |
| STT processing entry → first write | 1652.0 ms | 2790.4 ms | 2826.5 ms |
| Smart Turn RPC, included in pre-STT wait | 33.5 ms | 36.1 ms | 46.2 ms |
| STT RPC | 181.7 ms | 350.0 ms | 373.3 ms |
| Full chat-generation RPC | 687.8 ms | 933.2 ms | 950.4 ms |
| First TTS chunk RPC | 540.2 ms | 1324.8 ms | 1353.8 ms |
| All TTS-next RPCs per turn | 545.8 ms | 1343.6 ms | 1368.7 ms |
| Output stream construction/start | 4.7 ms | 5.5 ms | 6.1 ms |
| STT entry → turn completion, including playback | 3515.5 ms | 8512.4 ms | 8557.4 ms |

Percentiles across different stages do not add together. The device reported
56.3 ms output latency throughout; it has not been added to any result above.
The maximum observed replay scheduling lag was 10.15 ms.

| Question (ten warm repetitions each) | First-write p50 | First-write p95 |
| --- | ---: | ---: |
| Two plus two | 1951.5 ms | 1977.5 ms |
| Five times three | 2038.8 ms | 2066.6 ms |
| Days in a week | 2347.8 ms | 2386.8 ms |
| Opposite of hot | 2355.3 ms | 2391.0 ms |
| Primary colors of light | 3012.5 ms | 3037.7 ms |
| What bees make | 2289.0 ms | 2304.8 ms |
| Capital of France | 2368.7 ms | 2402.8 ms |
| Sides of a triangle | 2374.3 ms | 2446.9 ms |
| Why ice floats | 2951.4 ms | 3071.5 ms |
| Photosynthesis | 3500.4 ms | 3537.0 ms |

There were **zero failed turns, timeouts, worker restarts, output underflows or
Smart Turn rejections**. All 101 turns produced a response/audio-completion
event. The maximum sampled heartbeat age after readiness was 2.6 seconds.
Exact normalized STT agreement was 90/100 warm turns; all ten mismatches were
the two-plus-two fixture. This is a string comparison, not word-error rate or
semantic answer accuracy, and the mismatch's exact wording was not persisted.

| Resource observation | Result |
| --- | ---: |
| Peak sampled process-tree RSS across all phases | 3199.4 MiB |
| Peak sampled RSS during turns | 2755.4 MiB |
| Median sampled RSS during turns | 2451.1 MiB |
| First / last six-sample turn-window RSS medians | 2689.3 / 1790.3 MiB |
| Median idle RSS, four samples over 20 seconds | 2208.9 MiB |
| Sampled CPU during turns, p50 / p95 / max | 4.45% / 326.65% / 429.5% |
| Sampled CPU during idle, p50 / max | 0% / 10.6% |
| Resource samples during turns / overall | 136 / 144 |

The CPU figures include both busy inference and paced input/playback intervals;
they are not per-stage utilization. The short idle window is not a long-term
idle-power measurement. Decreasing resident memory does not establish that all
GPU allocations were released or rule out a longer-term leak.

Config/knowledge fingerprints were unchanged, event/status files contained no
conversation content, and the child exited cleanly. Model versions, lockfile,
private settings and the production runtime code were unchanged.

### Implications for Phase 2

This supplies a repeatable **software** baseline for the current batch-response
path. A 30% median reduction on this exact protocol would require at most
**1653.8 ms**, while keeping p95 at or below **3496.6 ms**. Re-run the same
question set, fixture hashes, model/config and output device to compare changes;
these thresholds do not replace the roadmap's acoustic first-audio gate.

Endpoint waiting, full chat generation and first-chunk TTS are the main measured
contributors. Start by measuring how streaming chat into speakable TTS chunks
changes these timings. Endpoint threshold changes need separate false-cutoff
and recognition checks. The initial wake miss also makes onset/pre-roll and
labeled wake testing useful follow-ups. Physical microphone/acoustic latency,
interruption timing, representative recognition/answer quality, cold-boot/model
downloads and long soaks remain separate measurements.
