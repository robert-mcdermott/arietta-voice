# Continuous streamed playback and live-capture soak

October 13–19, 2026. Follow-up to [forced-gap validation](FORCED_GAP_VALIDATION.md)
and the earlier [15-minute model-only soak](../docs/archive/experiments/MLX_LIVE_SOAK.md). ONNX remains the
default; this experiment uses the prepared MLX trial environment and cached
assets without changing production code, dependencies or private settings.

## Protocol

The requested workload is 30 minutes of audible synthetic answers on the Jabra,
with spoken checks before and after. Normally one runtime and microphone session
span all three phases. A stable inference worker is required throughout load;
live interruption can intentionally replace it without a failure restart. All
four models are warmed and a new PID baseline is recorded immediately before
the load clock starts. STT, chat, endpointing and TTS remain initialized during load;
each load turn checks model presence, backend and worker identity, and records
MLX active/cache/peak allocation plus worker high-water RSS. The parent samples
process-tree RSS, CPU and read-only host telemetry approximately every five seconds.

The load feeds the ten existing, hash-validated synthetic question waveforms at
real-time cadence into the normal VAD/endpoint/listening loop. Each ten-question
cycle uses a deterministic shuffled order. History resets every three turns;
inter-answer rests vary through 1, 1, 2, 5, 1, 1, 2, 10, 1 and 1 seconds. Replies
use the normal streamed chat/TTS/player path and play aloud at the existing device
settings. There is no synthetic speaker-loopback request and no artificial CPU
stress process.

During output, the unmodified player consumes timestamped **physical microphone
capture**, runs AEC and allows voice interruption. During synthetic question replay
and rests, physical capture is deliberately retired with a `synthetic_input`
counter in the report's `harness_discards`; it is not mixed into the fixture.
These test-attributed counts are a subset of the microphone's existing
`turn_boundary` discards, not additional drops to add to them.
This is a hybrid synthetic-input/live-
output load, not 30 minutes of natural microphone conversations. The real spoken
checks before/after exercise wake, room capture, interruption handoff, recovery
and goodbye separately. A detected interruption during unattended load fails the
turn; the report cannot determine whether its source was echo or a person/noise.

The fixture endpoint deadline is 15 seconds beyond its waveform end. Production
turn/stage limits still apply. The load finishes its current turn at the duration
boundary and then returns to live listening; worker replacement due to a closing
interruption is recorded separately. Each operator
window has an eight-minute deadline; an outer parent deadline covers native hangs.
Rows are capped at 5,000; cleanup signals only the owned test process group.

The isolated profile has a synthetic voice-aware persona, no tools, private
knowledge, memory, recordings or content logging. Config/knowledge fingerprints,
production-source hashes and both package inventories are compared before/after.
No system volume, private config, audio thresholds or dependency versions are tuned.

## Checks and interpretation

Before and after the load, wake with “Bridget,” wait for acknowledgement, and
request a count to twenty. Wait until numbers are audible, then interrupt with
“What is three plus three?” Verify prompt stopping without overlap and answer
six. Say “Goodbye,” then remain quiet for 25 seconds; require farewell and silence
plus an explicit goodbye transition in the log. Listening observations are stored
separately from automated gates.

Automatic gates require the full load duration, successful turns, all ten cases,
per-turn resource samples, no worker replacement, no failed/timed-out turns,
fresh heartbeats, clean shutdown and redacted status/events. Audio gates require
zero output underflows, AEC alignment resets, capture-queue drops, native input
overflows or device reconnects. Counters retain intentional discards separately.
Exact STT fixture matches are reported; they do not establish natural-speech WER
or semantic answer correctness.

Compare early/late full question cycles for resource trends; report CPU, pressure,
compression and swap context. MLX allocator, high-water RSS and sampled process
RSS overlap and must not be added. Processing time includes real-time answer
playback but excludes fixture input/rests; it is not acoustic first-audio latency.
This run cannot establish the Phase 2 acoustic p95 target, varied-room reliability,
physical disconnect/sleep recovery or eight-hour/24-hour release soak acceptance.

## Results

### October 19 uninterrupted load

A fresh isolated run completed **1,801.1 seconds (30 minutes 1 second), 169
valid load turns**, with all ten fixtures represented 16–17 times. All four
models remained loaded in the same inference worker throughout load. There were
no load turn failures, unintended interruptions, capture-queue drops, native
input overflows or failure-driven worker restarts. This completes the duration
and coverage experiment, but **fails the zero-fault audio acceptance criteria**:
39 output-underflow flags and 52 AEC alignment resets occurred across 19 turns.
Reset reasons were 39 output underflows and 13 timestamp gaps.

The first flags appeared on turn 57, roughly ten minutes into load. The first
flag followed a 215.01 ms write gap, including a 214.64 ms physical-capture poll.
Across the 37 retained flag contexts, maximum write gap was 683.42 ms, capture
poll 581.50 ms, and write duration 124.89 ms; producer waits were at most 0.144 ms.
Two further contexts were omitted by the existing bounded logger. These timings
make capture work and foreground scheduling useful investigation targets; they
do not prove an AEC, scheduler or MLX root cause. No production tuning or code
change was made during the experiment, and there is no matched ONNX control
under this host state.

Normalized fixture transcripts matched exactly on **135/169 turns**. This is
neither natural-speech WER nor semantic answer grading. Processing time was
4.685 seconds median and 10.680 seconds p95. It includes
answer playback and excludes synthetic input/rests; it must not be reported as
speech-end-to-first-audio latency.

MLX active allocation was essentially constant at **5,291.0 MiB**, including
identical first and last complete ten-turn-cycle medians. Allocator high-water
allocation reached 6,805.4 MiB; worker high-water RSS was 6,226.3 MiB. Sampled
process-tree RSS had a 740.2 MiB median and 5,853.7 MiB maximum, with first/last
load-minute medians 1,273.1/722.3 MiB. Compression and paging make sampled RSS
incomparable to model allocation; these overlapping metrics must not be added.
Stable allocation provides no evidence of growth in this window, but is not a
long-duration leak guarantee.

All 350 load host samples were on AC power with nominal thermal state and no
telemetry errors. Host CPU busy median/p95/max was 34.9%/73.0%/81.6% (empirical
p95 using sorted index `round(0.95 × (n − 1))`); memory pressure was normal in
315 samples and warning in 35. First-to-last load samples show swap use rising from 10,884.1 to 12,238.8 MiB,
while compressed memory fell from 11,986.8 to 8,636.9 MiB. These are whole-host
measurements, including unrelated applications, and do not attribute pressure
or swap growth to this runtime. Process-tree CPU median/p95/max was
5.7%/149.3%/343.7%, where 100% represents one CPU core.

The local evidence is in ignored `tmp/live-load-soak/20261020T031538.324802Z/`
(the directory uses UTC; the test ran October 19 Pacific time). This fresh run
used `--skip-before`: its opening observations are retained from October 13,
not presented as a new opening check in the same continuous session. Closing
operator checks passed: interruption after audible counting, direct answer six,
farewell and 25 seconds of silence. Events confirm one voice interruption and an
explicit `user_goodbye`. That window added one underflow and two alignment resets,
for **40 underflows and 54 resets overall**. The raw report remains `failed`
because of those two audio gates; every other automated runtime and soak gate
passed. Operator observations are retained separately in `operator.json`.

The session stopped cleanly. All 31 protected config/knowledge files, production
sources, the tested harness and both package inventories matched the October 19
preflight fingerprints. Private settings and the default ONNX backend remain
unchanged. Follow-up should isolate physical-capture processing versus scheduling
and host pressure under comparable ONNX/MLX workloads, then rerun this protocol.
The earlier planning-like recovery reply below remains a separate answer-quality
investigation; this successful closing answer does not resolve its cause.

### Earlier paused run and pilot evidence

**Paused at the operator's request for a meeting.** The isolated runtime, speaker
and inference worker stopped cleanly. The final completed turn was at 639.4 seconds
(10 minutes 39 seconds), with **66 completed load turns** and zero output
underflows, alignment resets or capture-queue drops. The requested 30 minutes and
closing live check were not completed. The raw report correctly retains
`status: failed` / `failure_code: operator_abort`; it is not a full soak pass.

Partial evidence is saved under ignored
`tmp/live-load-soak/20261013T174748.771704Z/`, including an explicit `pause.json`.
Opening operator evidence is from `20261013T174005.427506Z`. Protected config and
knowledge fingerprints, both package inventories and production-source hashes
matched after shutdown. The fresh October 19 run above is separate; disconnected
segments are not combined into a continuous-soak result.

The opening spoken check successfully interrupted audio, but its first recovery
reply contained planning-like wording instead of the requested arithmetic answer.
The operator suspected misrecognition; with content logging off, the transcript
and exact generation are unavailable and the cause is unresolved. A repeated
opening check passed interruption, a direct answer six, farewell and silence.
The first failure is retained in operator evidence and is not erased by the retry.
The local adapter forwards generated text into speech without an explicit local
reasoning-channel filter; this is a follow-up candidate, not an established cause.
Audio/resource gates must not be presented as a semantic-answer quality pass.

The first full-run attempt then stopped because the harness incorrectly required
the pre-interruption worker PID. Opening cancellation had legitimately changed
PID without a failure restart. Its resource-validation exception was recovered
by the normal listening loop, eventually producing an endpoint-deadline failure.
The harness now warms/rebaselines immediately before load and propagates resource
validation failures as fatal test failures with the partial row retained. The
opening checks from that attempt are retained separately; `--skip-before` resumes
with a fresh load and keeps the closing live window. No 30-minute continuity is
claimed for the aborted attempt.

The first short pilot stopped before any turn because the harness used an
unsupported capture-discard reason. The harness now uses the existing bounded
microphone drain and keeps test attribution separately. Production code was not
changed. A regression test exercises the real microphone API rather than a mock
that accepts arbitrary discard labels; the failed pilot is retained locally.
The next pilot completed one streamed turn, then hit `SleepOrSchedulingGap`
because replay bypassed the physical `get()` method's health poll. Replay now
polls that same health check at input cadence; a regression covers an eight-second
synthetic utterance with continuing callbacks. The failed pilot and reconnect
remain recorded and are not counted as load evidence.
The corrected 37.6-second pilot completed four valid turns with exact fixture
transcripts, continuous models and zero capture drops/reconnects. It did record
two genuine output-underflow flags on the first answer and two corresponding
AEC resets, so its audio gates failed. One flag followed a 57.25 ms capture poll;
both writes were about 57–59 ms after their preceding write, with negligible
producer wait and no silence padding. This identifies observed timing, not a
proven scheduler/AEC root cause. The three subsequent turns had no flags. The
pilot also intentionally lacked all-ten-case coverage. No production fix or
threshold adjustment was made before the longer run.

## Automated validation

The last full-suite run passed **605 tests**; the October 19 experiment used
the same fingerprint-verified harness and unchanged production sources. New regressions cover balanced fixture
cycles, paced input with physical timestamped playback capture, live-input
routing, bounded turns and endpoint waits, rests/duration boundaries, real
microphone drain attribution and capture-health polling, phase-scoped control
markers, operator deadlines, worker continuity scoped to load, fatal resource
validation and strict coverage/audio/resource gates. Offline
wheel/sdist build and release checks also pass, including an October 19 rebuild.
The fresh isolated build cache lacked setuptools; the successful offline rebuild
used the existing environment with `--no-build-isolation`, without installing or
changing dependencies. Production sources are unchanged.

## Reproduction

With the normal runtime stopped and the existing trial environment prepared:

```sh
uv run --no-project --python tmp/kokoro-integration/venv/bin/python \
  bench/live_load_soak.py --minutes 30 \
  --fixtures tmp/kokoro-integration/fixtures.npz \
  --tts-backend mlx \
  --mlx-kokoro-model tmp/kokoro-experiment/Kokoro-82M-bf16
```

The runner prints a fresh ignored `tmp/live-load-soak/<timestamp>/` directory and
phase updates. Wait for `live_before`, perform the opening check, then create an
empty `start-load` file in that exact run directory. At `live_after`, perform the
closing check and create `finish` there. An `abort` file or Ctrl-C stops the run
without calling incomplete coverage a pass. Marker contents are never executed.
`--automated-only` omits both live windows; it cannot supply operator evidence.
`--skip-before` reuses separately recorded opening checks and retains the closing
window; record the source run rather than implying both occurred in one worker.
Reports, synthetic profiles, markers and host/resource traces remain local and
ignored by Git. There is no production-app soak mode or spoken control command.
