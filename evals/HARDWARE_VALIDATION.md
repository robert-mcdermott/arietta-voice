# Local hardware smoke and soak testing

This opt-in check exercises the installed, locked model stack on the actual Mac.
It is separate from the synthetic CI suite and does not run in GitHub Actions.
Keep the normal voice runtime stopped while testing. The harness refuses an
active runtime heartbeat and locks out another invocation of the harness.

## Repeatable commands

From the repository, with the existing `uv` environment:

```bash
# Real VAD, STT, endpoint, MLX chat and TTS using a synthetic audio round trip.
# Does not open the microphone or speaker.
uv run --locked scripts/hardware_validation.py

# Also check five seconds of microphone callbacks and play one synthetic phrase.
uv run --locked scripts/hardware_validation.py --microphone-seconds 5 --playback

# Run the complete microphone -> runtime -> speaker path, then stop automatically.
uv run --locked scripts/hardware_validation.py --mode live --minutes 15

# Optional separate repeated-model load test; no microphone or audible playback.
uv run --locked scripts/hardware_validation.py --minutes 15
```

For the prepared, separate MLX integration environment, the harness accepts
temporary `--tts-backend mlx --mlx-kokoro-model PATH` overrides. They apply only
to the disposable test profile. `--host-metrics` records read-only macOS host
CPU, power, thermal and memory-pressure samples. See
[the MLX live and continuous-soak protocol](MLX_LIVE_SOAK.md) for commands.
Playback summaries also include AEC reset reasons and startup/stale packet
discards; runtime audio status separates real queue overruns by phase from
intentional queue discards. Inspect these alongside the automatic gates, which
do not require zero overruns/resets. See the
[capture/alignment investigation](CAPTURE_ALIGNMENT_INVESTIGATION.md).

In components mode, `--resource-metrics` checks that STT, chat, endpointing and
TTS remain resident in the same inference worker and records allocator/RSS
samples after every turn. It requires TTS and Smart Turn enabled. `--fixtures`
cycles an existing synthetic NPZ and validates its corpus/hash manifest; it
does not capture microphone audio. `--soak-interval` controls the pause between
turns (0–30 seconds, default 5). Fixture-driven component turns have empty
conversation history and do not run the live playback/AEC path. Worker residency
and repeated inference are distinct from a full live conversation soak.

Use `--config /path/to/base.toml` if the normal app uses an explicit base config.
The same private override precedence applies on read. The harness requires a
local chat provider and never calls Bedrock. macOS microphone access may need to
be granted to the terminal/app hosting the command. Run from a normal Mac
terminal: a restricted sandbox can show zero devices even when audio works.

Wait for `READY: isolated voice runtime is listening.` before speaking. `Ctrl-C`
stops the test and its inference worker. An outer deadline and escalation from
interrupt to terminate/kill also cover native calls that do not return. Only
the test's owned process group is signalled. The harness caps each run at 60
minutes; eight-hour and 24-hour release soaks need a separate longer-run plan.

For the Phase 2 AEC/pre-roll changes and completed operator interruption checks,
see [AEC validation](AEC_VALIDATION.md). Hardware reports include the configured
pre-roll/delay and playback alignment-reset/output-underflow totals. Existing
live gates do not require zero application queue drops or output underflows;
review these counts separately before claiming glitch-free playback.

## Isolation and artifacts

The harness reads the selected config, then builds a separate configuration under
ignored `tmp/hardware-validation/<timestamp>/`. It copies audio settings, runtime
limits, the local chat model, STT language and output-token limit. It uses a
voice-aware synthetic persona, default `Bridget` wake phrases, temperature zero,
and empty history. Custom tools, knowledge, memory, source editing, conversation
logging and recordings are disabled. It never migrates or rewrites private
settings. File fingerprints detect changes under `config/` and `knowledge/`
without copying those files into the report; concurrent user edits are reported
as changes and are never undone.

Hugging Face model loading is offline and uses cached snapshots. Normal app
loaders may obtain missing Moonshine, Kokoro or Smart Turn assets from their
configured public download locations. This does not upgrade the lockfile. Model
caches can gain assets/lock files; private config/persona/credentials remain
untouched. The test does not exercise private tools, private knowledge, or cloud
behavior.

Outputs are local JSON metadata, event counts, heartbeats and resource samples.
They include `environment.json`, `report.json`, `resources.json`, and the isolated
runtime's redacted event/status files. They are ignored by Git and are not CI
upload artifacts. Microphone audio is processed locally in live mode but is not
recorded; passive transcripts are not saved. A separate optional microphone
probe reports RMS/peak and callback counts without transcription. Review any
report before sharing it: hardware names, model IDs and local errors can identify
a machine even when conversation content is absent.

## Hands-on smoke checklist

Record observations separately from automatic results. Hardware enumeration
and a non-empty generated answer do not establish that speech was audible or
that the model answered correctly.

1. Say **“Bridget.”** Confirm an audible acknowledgement.
2. Say **“What is two plus two?”** Confirm a clear spoken answer of four.
3. Say **“What does goodbye mean?”** Confirm this question does not close the
   session. Then say **“Goodbye.”** Confirm a farewell and a `session_ended`
   event whose reason is `user_goodbye`, not merely a later idle timeout.
4. Say **“Bridget, count from one to twenty.”** During speech, interrupt with
   **“What is three plus three?”** Confirm playback stops and it answers six.
   Check `voice_interruptions` in the event summary. This does not measure the
   Phase 2 target of 300 ms interruption latency.
5. Allow the session to time out. Confirm `idle_timeout` and return to
   `waiting_for_wake`. Leave ordinary background speech without the wake prefix
   unanswered. Repeat one wake/answer/farewell near the end if practical.

The synthetic persona explicitly says that replies are spoken by TTS. A generic
“test assistant” persona can cause the model to claim it cannot speak; this is a
test-fixture problem and should not be “fixed” by overwriting a user's persona.

## Soak acceptance and interpretation

For a 15-minute initial live baseline, require the full requested duration after
readiness, clean shutdown, no failed/timed-out turns, no worker restart or native
input overflow, fresh heartbeats (at most five seconds old), and no conversation
content in event/status files. Confirm private file fingerprints remain equal.
The report keeps human confirmation pending; `completed` means the timed run and
automatic checks completed, not that every manual or longer-term release gate
passed. Missing duration, crash, deadline, privacy or reliability checks fail.

Inspect resource samples and the recorded queue capacity. Dropped capture chunks
are distinct from native input-device overflows: the bounded queue may discard
old audio while the runtime is busy. Report those drops and investigate lost
speech; do not silently classify all drops as a device failure or omit them.

Memory measurements sum **resident memory (RSS) of the test process tree**.
They are not a complete measurement of Metal allocations or total unified
memory. The report separates startup from warm samples and reports first/last
warm-window medians. Review the trace for a continuing upward trend; one short
stable interval cannot prove absence of a long-term leak. CPU is sampled using
`ps`, not a calibrated throughput benchmark. Component turn latency is the whole
synthetic STT/endpoint/chat/TTS job, not speech-end-to-first-audio latency.

An idle-heavy microphone soak and a repeated-model load soak are different tests;
label the one actually run. Keep the Phase 2 latency baseline, representative
retrieval/model evaluations, device reconnect/sleep tests, and Phase 4/5 eight-hour
and 24-hour soaks as separate gates. Do not mark them passed from this baseline.

## Recorded baseline

See [HARDWARE_BASELINE.md](HARDWARE_BASELINE.md) for the current measured run and
remaining limitations. Raw reports and recordings must not be committed; only
reviewed aggregate observations belong in the repository.

For the separate 100-turn Phase 2 timing protocol and measured stage/resource
results, see [LATENCY_BASELINE.md](LATENCY_BASELINE.md).
