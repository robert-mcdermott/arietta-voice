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
uv run --locked bench/latency_baseline.py --turns 100 --audible

# The same real output-device path with zero-valued samples instead of sound.
uv run --locked bench/latency_baseline.py --turns 100
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

Ten fixed questions in `bench/latency_baseline.py` cover short arithmetic,
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

## November 13, 2026: first Milestone 1 changes

Same protocol, questions and shared fixtures as above, on the M4/32 GB Mac with
the locked ONNX Kokoro stack. Each run is 30 warm turns (three per question)
with muted output to the real device, run back to back on an otherwise idle
machine. "Before" is `main` at `94fdd95`; "after" adds streamed tool planning,
prompt-prefix reuse, the shorter first clause and per-turn timing. These are
shorter runs than the 100-turn October baseline, and today's "before" median
(2,483 ms) is the comparison point, not October's 2,362 ms.

`--model-tools` offers the model one tool (`system_info`) on every turn. None of
the questions needs it, so this measures what the default configuration pays for
having tools available.

Speech end to first reply write, milliseconds:

| Configuration | Before p50 | After p50 | Change | Before p95 | After p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default, no tools | 2,483 | 2,167 | −13% | 3,820 | 3,395 |
| Default, `--model-tools` | 3,087 | 2,188 | −29% | 5,058 | 4,036 |

Median stage times after the change, no tools: endpoint wait 709, transcription
214, chat generation 491 (was 746), first TTS chunk 562. With a tool on offer,
chat generation fell from 1,320 to 525 because a direct answer no longer waits
for a separate planning generation. All four default runs passed every gate, with
no output underflows, and the transcript matched its fixture in 24 of 30 turns
both before and after.

### Early end-of-turn check

With `audio.endpoint_early_ms = 250`, the endpoint wait fell from about 708 ms
to about 290 ms. On the eight single-sentence questions (24 turns):

| Configuration | Before p50 | After p50 | Before p95 | After p95 |
| --- | ---: | ---: | ---: | ---: |
| No tools | 2,478 | 1,814 | 3,250 | 2,590 |
| `--model-tools` | 3,045 | 1,781 | 3,778 | 2,487 |

It also ended both two-sentence questions after their first sentence in all six
turns of each run. Those fixtures pause for 346 and 375 ms between sentences,
the first sentence is a complete question, and Smart Turn judged it finished.
Whole-run medians from those runs
(1,639 and 1,766 ms) are not valid, because a cut-off turn starts answering
before the recorded end of speech. That result is why the check was first shipped
off by default; the next section describes the fix.

### Early check with resume (now the default)

A later change keeps the turn provisional after an early decision: the
assistant goes on listening until `silence_ms` has passed since the end of
speech, and if speech resumes it discards the attempt and continues the same
utterance. With that in place and `audio.endpoint_early_ms = 250`, later the
same day on the same machine:

| Configuration | Start of day p50 | Early check + resume p50 | Start of day p95 | Early check + resume p95 |
| --- | ---: | ---: | ---: | ---: |
| No tools | 2,483 | 1,710 | 3,820 | 2,931 |
| `--model-tools` | 3,087 | 1,719 | 5,058 | 3,594 |

Both runs passed every gate with no worker restarts or failed turns. All six
two-sentence turns in each run resumed and were then transcribed in full; their
median first write was 2,660 ms without tools and 2,960 ms with a tool on offer
(3,744 and 4,420 ms at the start of the day). Single-sentence questions had a
median of 1,685 and 1,705 ms. Median stages: endpoint wait 298, transcription
199, chat generation 483 to 518, first TTS chunk 566.

In those runs the speaker always resumed while the first attempt was being
transcribed. Two short diagnostic runs with the pause lengthened by 200 and
320 ms made speech resume during answer generation instead: all three such
turns in each run were discarded and completed correctly, again with no worker
restarts. A resumed turn pays for prompt processing again, because closing the
discarded generation clears the prefix cache.

One transcript in each 30-turn run did not match its fixture where it had
before: the triangle question came back as "Yeah How many sides does a triangle
have?" on the turn after a resumed one. The answer was unchanged. The cause is
the recogniser, not the audio: calling Moonshine 0.0.54 directly with the
fixture audio in a similar order reproduces it, with the same array transcribed
correctly earlier in the same process. Its batch API therefore carries state
between calls, and the extra transcription of a discarded attempt changes that
history.

### Prompt-prefix reuse

Checked directly against the model over six-turn conversations at temperature 0,
comparing each prompt with and without reuse. On turns after the first, median
time to the first text fell from 1,064 ms to 254 ms. Prompt processing runs at
only about 190 tokens per second with this model build, which is why reuse
matters. After a cancelled generation the cache is discarded and the next turn
pays the full cost once.

Answers were identical in 11 of 12 comparisons. The twelfth differed, and
repeating the check without the repetition penalty still showed differences on
the same sixth turn (2 of 12). In each case the cached answer was a correct
reply and the uncached one was not better, but the cause was not established;
set `models.reuse_prompt_prefix = false` to compare.

### Limits

Thirty turns per run, one run per condition, synthetic speech and a software
timing boundary. Nothing here measures the acoustic path, interruption, a live
microphone, or answer quality. The raw per-turn reports from these runs were not
kept; only the figures above were recorded. The 1.0 s target is not met.

