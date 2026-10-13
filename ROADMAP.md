# Arietta Voice Improvement Roadmap

Plan dated October 11, 2026. Preserve the local Apple Silicon assistant and
improve privacy, reliability, and response speed in that order. The evidence
and technology comparisons are in [PROJECT_REVIEW.md](PROJECT_REVIEW.md).
The existing app baseline is `0.1.0`; versions below are proposed milestones,
not releases that have already been published.

Estimates are focused engineering effort for one developer, including tests,
with model/hardware experiments adding uncertainty. Release when the exit
criteria pass rather than on a fixed calendar date.

| Phase | Proposed version | Outcome | Effort | Depends on |
| --- | --- | --- | --- | --- |
| 0 | `0.1.1` | Personal knowledge protected by default; consistent version reporting | 1–3 days | Current checkout |
| 1 | `0.2.0` | Private, bounded, testable operation with known correctness fixes | 1–2 weeks | Phase 0 |
| 2 | `0.3.0` | Streaming answers and dependable interruption | 2–4 weeks | Phase 1 and a recorded baseline |
| 3 | `0.4.0` | Better grounded answers, structured tools, manageable memory | 1–2 weeks | Phase 1; integrate against Phase 2 interfaces |
| 4 | `0.5.0` | Repeatable installation and useful optional integrations | 2–3 weeks | Phases 2 and 3 |
| 5 | `1.0.0` | Stable supported interface and an operationally proven release | Based on validation | Prior release gates |

## Phase 0 Establish repository and release hygiene

Implemented in this working tree:

- Ignore everything below `knowledge/` except its README and getting-started
  article; relocate the article template to `examples/knowledge-template.md`.
- Explain tracked-file exceptions, custom-directory limits, and private backups.
- Use `pyproject.toml` as the sole version declaration and installed package
  metadata for runtime reporting. Add `arietta-voice --version` and reuse the
  same version in the existing API fields.
- Add a changelog, release instructions, this roadmap, and the project review.

To close the phase:

1. Review and commit these changes with an explicit file list, preserving the
   user's separate config edits.
2. Hosted CI verified October 11, 2026: the [first run](https://github.com/robert-mcdermott/arietta-voice/actions/runs/38199350677)
   passed the tracked-knowledge allowlist, forced-addition regression,
   config/private-path, package-content and version checks.
3. Choose whether to tag the historical `0.1.0` baseline; its version already
   existed, but no local release tag was found. Prepare `0.1.1` with the release
   procedure and publish only after review.

**Exit criteria:** 60 existing/new tests pass; version reporting agrees across
CLI/package/API; private and nested knowledge is ignored; CI rejects a private
tracked knowledge fixture; wheel/sdist have the expected version and no local
personalization files. No release tag is created by this review.

## Phase 1 Protect privacy and fix the foundation

**Status: complete as of October 11, 2026.** Implementation, synthetic tests,
hosted CI, hardware smoke and the initial 15-minute live soak have passed.
Release/version tagging remains a separate step under [RELEASING.md](RELEASING.md).

The first batch implements items 1, 4, and 5 below:
minimal public health; opt-in active conversation logging/status; no passive
transcript logging; configurable, process-safe JSONL rotation; disabled-tool
enforcement; pair-based history; preserved inline command text; direct-farewell
matching; startup config validation; normalized BGE CLS embeddings with index
format 2; and retrieval root containment. Regression tests use synthetic data
and mocked hardware/models. Rebuild old semantic indexes with
`uv run arietta-voice knowledge-index` and restart runtime/API processes.

The second batch implements items 2 and 3: explicit admin credential setup with
private password hashes (or explicit environment credentials); source writes
disabled by default; static tool inventories/source reads; Host/origin checks;
ignored local config/persona overrides; a preserving migration command; and
private persistence of admin tool selections. Bedrock's text-data boundary is
documented in README. Tests cover authentication, blocked mutations, non-executing
inventory views, config precedence, migration, and preserved settings.

The third batch implements item 6: bounded capture, utterance, recording, and
playback-reference buffers; killable serialized inference workers; shared turn
budgets; endpoint/startup deadlines; failure recovery; exception-safe playback
cleanup; atomic memory writes; and an independent heartbeat. Tests exercise real
spawned workers with synthetic handlers, including timeouts, crashes, shutdown,
and busy requests, plus simulated audio and state transitions. A focused admin
dashboard/chat/editor usability update has been brought forward from Phase 4.

The fourth batch implements item 7: Apple Silicon CI with locked installs,
isolated tests, tracked-file privacy and distribution/version checks, and a
43-case synthetic corpus covering runtime sessions, wake behavior, prompts,
memory, tool policy and keyword retrieval. Corrupt-index regressions now cover
safe fallback/rebuild, and spawned-worker tests cover config snapshot invalidation
alongside concurrency/crash/deadline coverage. The corpus caught and fixed memory
facts being discarded when they contained the substring `NONE`. See
[evals/README.md](evals/README.md) for repeatable commands and limitations.

The fifth batch adds an isolated opt-in hardware harness and records a passing
real-model smoke test and 15-minute live microphone soak on the locked stack,
with operator-confirmed speech, interruption and farewell behavior. Private
settings remained unchanged. See [the measured baseline](evals/HARDWARE_BASELINE.md)
for resource measurements, queue drops, the synthetic persona correction, and
the limited load exercised; [the test guide](evals/HARDWARE_VALIDATION.md) covers
repeatable live and repeated-model checks.

The [first hosted CI run](https://github.com/robert-mcdermott/arietta-voice/actions/runs/38199350677)
passed on `6023154` (308 unit/process tests and 43 synthetic evaluations).
The [hardware-validation commit's run](https://github.com/robert-mcdermott/arietta-voice/actions/runs/38201095806)
also passed on `fe04096` (326 unit/process tests and 43 synthetic evaluations).
Both Apple Silicon/Python 3.13 runs passed locked installation, tracked-file
privacy/version checks, wheel/sdist inspection, and installed-wheel smoke tests.
There were no hosted CI failures requiring a code or workflow fix.

Longer/load/reconnect tests and the Phase 2 latency baseline remain separate
work. Native audio-device stalls and admin
maintenance actions are not covered by inference deadlines. Voice and API workers
are serialized individually; shared GPU scheduling remains future work. No release
bump or tag accompanies these implementation batches.

1. Remove transcripts/paths from public health; make content logging opt-in,
   suppress passive-speech persistence, and add retention/rotation settings.
2. Replace known default admin credentials with a setup requirement. Gate source
   editing; stop importing tool code just to render inventory. Preserve detailed
   diagnostics behind authentication.
3. Introduce ignored local config/persona overrides with a migration that
   preserves current files. Explain the data sent to Bedrock when enabled.
4. Fix `use_tools=False`, history-pair counting, inline wake-command preservation,
   goodbye intent, and invalid config handling.
5. Correct BGE pooling against the reference, version the embedding pipeline,
   invalidate old indexes, and enforce retrieval root containment.
6. Bound microphone queues, utterance duration, and recording memory. Add
   per-turn timeouts/recovery, exception-safe audio cleanup, inference ownership,
   and a heartbeat that continues during long work.
7. Add CI and a synthetic evaluation corpus before introducing new models.
   Include runtime state-transition, prompt, memory, tool-policy, corrupt-index,
   and process-concurrency regression coverage.

**Exit criteria:** unauthenticated health contains no transcript or local paths;
passive speech leaves no text in logs/status by default; tools disabled means
zero executions; configured history counts pairs correctly; malformed config
fails with actionable errors. A failed model/tool turn returns to listening;
continuous audio and an unresponsive endpoint model cannot grow buffers without
limit. A hardware smoke test passes with the existing locked model stack.

**Rollback:** preserve private config backups and the old lockfile; knowledge
indexes are rebuildable. Do not automatically re-enable privacy-leaking logging
when rolling back another component.

## Phase 2 Reduce the wait before speech

**Initial measurement status (October 12, 2026):** the isolated Mac benchmark
completed 100 warm no-tool turns through paced synthetic input and actual Jabra
speaker output. Waveform-end-to-first-write latency was 2362.5 ms p50 and
3496.6 ms p95, with no failed turns or output underflows. Fresh-worker warmup
with cached assets was 6.112 seconds; startup, first-turn and warm measurements
are separate. See [the latency/resource baseline](evals/LATENCY_BASELINE.md) for
stage timings, RSS/CPU, repeatable commands and the failed wake-prefix pilot.
This is a software timing boundary; physical capture/acoustic onset and
interruption latency remain unmeasured.

**Cancellation and queue status (October 12, 2026):** item 2 is implemented.
Per-turn cancellation reaches inference, bounded text/audio handoffs and speaker
playback. The console has a Stop action, and console/voice share conversation
policy. Regression coverage checks cancellation, stale-result rejection,
backpressure, cleanup and subsequent-turn recovery. An isolated muted real-model
smoke passed its first turn plus two warm turns on the Jabra without failures or
underflows; an isolated browser check passed Stop and next-chat recovery. Private
settings were preserved. This is a correctness check, not a new latency baseline
or evidence that the interruption target is met.

**Provider streaming status (October 12, 2026):** item 3 is implemented for MLX
and Bedrock, with structured text/completion events and incremental clauses sent
to Kokoro. One inference owner alternates generation and synthesis while the
speaker plays queued frames. Tool planning remains buffered; final post-tool
answers stream, while direct planner answers are already complete. Console HTTP
responses remain aggregated. Regression tests cover provider completion/errors,
early audio, clause limits, cancellation during generation gaps, tool-payload
withholding and partial-turn exclusion. Cloud behavior is checked with synthetic
Bedrock events; live AWS validation remains a release check. The original
100-turn baseline remains unchanged; no performance target is claimed yet.
An isolated muted Mac smoke passed its first turn plus two warm turns through
STT and the real output device. A separate three-sentence MLX/Kokoro check wrote
its first audio 1802.5 ms before model generation completed, delivered all three
clauses and reported zero output underflows. Both used synthetic context; local
reports stay in ignored `tmp/`. This single response establishes ordering only,
not a latency percentile, perceived speech quality or the acoustic interrupt target.

**Background memory status (October 12, 2026):** item 4 is implemented. Delivered
voice turns enqueue bounded, RAM-only work and return to listening. Quiet-time
jobs reuse the inference worker, yield between tokens, and give new speech
priority; stuck calls are terminated before foreground reuse. Extraction and
consolidation commit together after checking for intervening file changes.
Separate memory status counters cover dropped, cancelled, failed and conflicting
jobs. No private settings or persona changes are required. Regression tests cover
queue pressure, preemption, native-call termination, file preservation and
shutdown. Memory remains best-effort; preemption may discard an update and native
worker termination may add reload latency to the next turn.
An isolated cached local-model check completed extraction and consolidation in
the background, then answered the next synthetic chat on the same worker with
zero restarts. Enqueueing took 0.04 ms in that check. Only a temporary synthetic
memory file was written; this is a functional check, not a latency percentile.

**Moonshine experiment status (October 12, 2026):** the isolated portion of item
5 is complete. A separate pinned 0.1.5 environment exercises batch and paced
streaming against the same ten synthetic fixtures as 0.0.54, plus explicit
keyterms and context-derived hints. Finalization, silence, partial revisions,
language defaults and asset changes are documented in
[the experiment report](evals/MOONSHINE_EXPERIMENT.md). The candidate remains
experimental: no normalized accuracy gain was observed, and chunk-feed delays
require live capture/backpressure validation. Production dependencies and the
batch adapter remain unchanged; hardware/language coverage and migration gates
must pass before promotion. Upstream labels this non-prerelease package alpha.

**AEC/pre-roll/cancellation status (October 12, 2026):** the first part of item 6
is implemented: ADC-timestamped capture, a bounded playback-reference timeline,
continuous 10 ms AEC framing, configurable pre-roll, cleaned interruption
handoff, alignment resets on gaps/underflow, and keyboard cancellation during
generation gaps and final device draining. Private settings remain unchanged.
[Validation and timing limitations](evals/AEC_VALIDATION.md) distinguish software
contracts from physical interruption latency. Measured acoustic calibration and
the 300 ms p95 target remain open.
The five-minute isolated Jabra check passed operator-confirmed early/later
interruption, quiet playback and farewell checks, with no turn failures. It
recorded application queue drops and one output underflow, retained as load-test
follow-ups. A final-code muted three-turn smoke passed with zero underflows or
failures. The synthetic suite now passes 460 tests.

**Device selection/recovery status (October 12, 2026):** the remaining software
portion of item 6 is implemented: exact input/output selectors, private/CLI
overrides, direction-aware diagnostics, stale capture and sleep-gap detection,
stream teardown, inventory refresh and bounded reconnect backoff. Recovery
discards the interrupted turn/session; it does not replay external actions.
Admin status distinguishes a live process waiting for audio from a ready one.
[Validation](evals/DEVICE_RECOVERY_VALIDATION.md) records automated coverage and
native checks. Physical unplug/replug and sleep/wake checks are deferred at the
user's request, so those hardware acceptance gates remain open.

1. Record stage-level latency and resource baselines on the intended Mac and
   microphone/speaker. Separate cold start from warm conversational turns.
2. Extend the Phase 1 serialized inference workers with cancellation tokens and bounded text/audio
   queues. Share the conversation service between console and voice paths.
3. Extend the MLX and Bedrock adapters with structured streaming events. Send
   speakable clauses to Kokoro while generation continues; withhold tool payloads.
4. Move memory work after first response delivery and off the interactive path.
5. Evaluate Moonshine in an isolated experiment from `0.0.54` to a pinned
   streaming candidate, covering asset formats, language selection, partial/final
   transcript semantics, and terminology hints. Retain the old path until validated.
6. Improve AEC alignment, pre-roll capture, and interruption cancellation. Add
   input/output device selection and sleep/reconnect handling.
7. Compare current ONNX Kokoro with MLX Kokoro; then optionally test small
   Qwen3-TTS. Compare Gemma with one compatible small alternate model. Change
   only one component per comparison. Acoustic wake detection is an optional
   experiment and does not block the streaming release.

**Kokoro comparison status (October 12, 2026):** an isolated four-block comparison
of locked ONNX 0.5.0 and MLX Audio 0.5.8 completed 48 warm synthetic utterances per
backend, with identical `af_heart` voice tensors. MLX reduced median TTS
first-audio time from 1,175 to 422 ms; initial startup was slower. Lower CPU/RSS
must be considered alongside its 1.90 GiB MLX allocator peak. In the three-pair
blinded review, the user found MLX's currency pronunciation more natural and
otherwise found the engines similar. An opt-in adapter and separate full-app
trial environment now support conversation latency and shared-model memory
measurement; production remains on ONNX. [Protocol, results and migration gates](evals/KOKORO_EXPERIMENT.md) keep this
component result separate from end-to-end Phase 2 acceptance criteria.

**MLX integration trial:** the shared worker now supports explicitly selected
local MLX Kokoro assets, bounded materialized PCM and the existing cancellation
and recovery mechanism. Matched five-turn conversation blocks measured a
21.8% median first-write improvement (2,952 → 2,308 ms), or 14.6% to the first
signal-containing write (2,952 → 2,520 ms). Shared MLX allocator peak increased
from 5.23 to 6.43 GiB. These are short synthetic runs with muted real-device
output. Against the bracketed production ONNX reference, the median first-write
gain was 11.5%, or only 3.3% to the first signal-containing write. Keep MLX opt-in
pending longer workload and live playback/interruption
validation. See [full integration results and reproduction](evals/KOKORO_INTEGRATION.md),
including the production dependency control and remaining migration gates.

**ONNX environment follow-up:** a reversed-order repeat completed 44 synthetic
conversation turns. Trial ONNX was 9.1% faster at the median, reversing the earlier
13.2% slowdown. Across 120 warm component probes, the ONNX implementation and
corresponding audio hashes matched; with all four models loaded, median TTS
latency differed by only 7.3 ms (0.8%). No dependency rollback is justified by
these results. Keep environments isolated and use longer, counterbalanced
measurements with host-state context before migration. See
[the investigation and reproducible probes](evals/ONNX_ENVIRONMENT_INVESTIGATION.md).

**100-turn counterbalanced follow-up:** all 318 synthetic turns passed across
six execution orders: 100 warm turns per condition plus 18 initial turns.
MLX improved pooled median first-write latency by 12.5% against production ONNX,
but first signal-containing write by only 3.4% (2,447 → 2,364 ms); first-signal
p95 improved 12.4%. Shared MLX allocator peak increased from 5.23 to 6.43 GiB.
Trial ONNX's pooled 5.8% slowdown varied across rounds and included a busy-host
block. All thermal/power samples were nominal/AC, but CPU load and memory
pressure varied; no fixed dependency regression is established. Keep ONNX as
default and MLX opt-in pending integrated audible interruption/AEC checks and a
continuous shared-model soak. Investigating MLX's leading silence is a separate
candidate optimization, with initial-phoneme quality checks required. See
[the full comparison, host context and remaining gates](evals/KOKORO_100_TURN_COMPARISON.md).

**MLX live/soak follow-up (October 13, 2026):** a five-minute Jabra check passed
operator playback, no-self-triggering, early/mid-answer interruption, recovery
and farewell checks. Logs confirm two intended voice interruptions and no
output underflows or turn failures; 73 capture-queue drops and 10 alignment
resets remain documented limits. The subsequent 15-minute component soak
completed 390 varied synthetic turns in one worker, with all four models loaded,
no restarts and stable 5.17 GiB active MLX allocation; peak was 6.43 GiB.
Private files and both dependency environments remained unchanged. Keep MLX
opt-in: acoustic timing/echo measurement, capture-drop investigation and longer
representative live/load soaks remain open. See
[operator evidence, memory results and reproduction](evals/MLX_LIVE_SOAK.md).

**Capture/alignment follow-up (October 13, 2026):** expected ADC packets crossing
playback startup no longer reset fresh AEC instances. Transcription and
first-output waits now retire capture that their existing boundaries discarded,
preventing unused backlogs from filling the queue. Phase/reason counters separate
actual queue overruns, intentional discards and alignment faults. The 565-test
suite passes. A 302.7-second isolated MLX repeat recorded zero capture drops,
zero timestamp-gap resets and passed operator interruption/farewell checks.
It recorded four genuine output underflows during one answer; their resets
remain active and visible. Private files and dependency inventories matched.
Next, investigate output starvation/write scheduling with matched playback
timings; longer live/load and acoustic acceptance gates remain open. See
[findings, regression coverage and limits](evals/CAPTURE_ALIGNMENT_INVESTIGATION.md).

**Exit criteria:** using the same warm, no-tool question set, reduce median
speech-end-to-first-audio time by at least 30% without worsening p95 relative
to the Phase 1 baseline. Target p95 interruption-to-silence at or below 300 ms
on the chosen audio hardware, with no stale playback after cancellation.
Measure false cutoffs, false interruptions, and transcription accuracy; accept
no unexplained regression. These are proposed targets, not current measurements.

**Rollback:** select the previous adapter/model profile through configuration;
retain the batch STT and existing Kokoro paths. Schema/tool planning still needs
to complete before speaking an action result.

## Phase 3 Improve knowledge and tool reliability

1. Add bounded retrieval chunks, stable source IDs, atomic index generations,
   visible stale/corrupt-index state, and incremental indexing where worthwhile.
2. Compare keyword and hybrid retrieval on at least 50 representative synthetic
   queries, including paraphrases, proper names, ambiguous questions, and absent
   facts. Add source references to API responses and explicit uncertainty when
   evidence is missing. Consider reranking only if it improves measured quality.
3. Add tool argument schemas, native provider tool calls where available,
   deadlines, bounded results, and structured errors. Retain a one-call budget
   until evidence shows a need for longer tool sequences.
4. Separate untrusted retrieved text from instructions. Add prompt-injection
   fixtures and enforce allowed actions outside the model.
5. Add memory inspect/delete controls, provenance, size limits, atomic writes,
   and background consolidation.

**Exit criteria:** target relevant-source recall@3 of at least 90% on the agreed
answerable corpus, with separate false-positive and unsupported-answer reporting
on unanswerable questions. Disabled or invalid tools cannot execute; timeouts
produce a useful response; interrupted/retried actions cannot execute twice.
Corrupt indexes recover without losing source files. Memory can be inspected,
cleared, and disabled without residual prompt injection.

**Rollback:** keyword retrieval and the prior index generation remain available;
reverting an encoder or chunker requires the matching index or a rebuild.

## Phase 4 Make installation and extensions durable

1. Package starter assets and add `arietta-voice init` plus config migration.
   Keep private data outside the installed package and custom tools outside
   application source. Consolidate the overlapping config templates.
2. Extract API routes and frontend assets; add focused browser smoke checks.
   Optimize event polling and logs without requiring a frontend rewrite.
   Phase 1 already adds a responsive reliability overview, expandable technical
   details, persistent error notices, guarded chat submission with failed-draft
   recovery, and unsaved-editor warnings. Follow up with automated browser CI,
   shared model-ownership status, and deadline coverage for admin maintenance.
3. Provide generated LaunchAgent setup and a single chosen supervisor model,
   startup backoff, clean shutdown, and update/rollback instructions.
4. Add one useful integration chosen by actual usage, such as read-only home
   status followed by explicitly authorized device control. Start with schemas,
   permission boundaries, action confirmation, and idempotency.
5. Optionally prototype a cloud voice session using Nova 2 Sonic or OpenAI
   Realtime. Keep this feature opt-in, define audio/text egress and cost limits,
   and test connection renewal, interruption, tool handoff, and offline fallback.
   This experiment is not required to ship the local release.

**Exit criteria:** a clean Apple Silicon installation works without editing
package source; an upgrade preserves knowledge/persona/settings; a rollback is
documented and exercised. An eight-hour soak has no crash, lost device, duplicate
runtime, or unbounded memory growth. Network loss does not prevent local mode.

## Phase 5 Declare the supported stable interface

Ship `1.0.0` when the CLI, config schema, documented HTTP endpoints, and extension
interfaces have an explicit compatibility policy; the supported Mac/Python
matrix is documented; migrations and rollback work; and the privacy, speech,
knowledge, and operations gates above are repeatable. Complete a 24-hour soak
with reconnect/restart scenarios and publish the measured resource envelope.
The version number should communicate a stable promise, not just accumulated features.

## Benchmark and release discipline

Use synthetic or explicitly approved local fixtures. Keep private recordings
and real knowledge out of the repository. Record app version, dependency lock,
model revision/quantization, Mac chip/RAM, macOS version, audio device, and config.

| Measure | Evaluation approach |
| --- | --- |
| Response latency | At least 100 warm no-tool turns; p50/p95 end-to-first-audio and stage timings; report cold-start cases separately |
| Speech recognition | Quiet and noisy clips, accents, names, numbers; word errors and task success |
| Wake behavior | Positive custom-phrase trials and long negative background audio; misses and false activations per hour |
| Endpointing and interruption | Hesitations, long pauses, speaker echo, and deliberate barge-in; premature cutoffs and stop latency |
| Grounding and tools | Answerable/unanswerable questions, source recall, malformed requests, disabled tools, retries, and injected instructions |
| Resources and operation | Idle CPU, peak memory, startup/download time, queue bounds, sleep/reconnect, eight-hour then 24-hour soak |

Model replacements must show a useful latency, quality, or resource advantage
within the same test conditions. Keep raw private evaluations local; publish
only synthetic fixtures and aggregate results. Track shipped changes in
[CHANGELOG.md](CHANGELOG.md), and use [RELEASING.md](RELEASING.md) for version bumps.
