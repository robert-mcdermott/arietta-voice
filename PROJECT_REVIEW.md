# Arietta Voice Project Review

Reviewed October 11, 2026 against checkout `7f90488` and its existing local
configuration. Arietta is a useful Apple Silicon assistant foundation. The
highest-value work is to protect personal data, correct runtime behavior,
and reduce the delay before the first spoken answer. Keep the local default
and the existing component boundaries while introducing measured upgrades.

The implementation sequence and release gates are in [ROADMAP.md](ROADMAP.md).
App versioning is described in [RELEASING.md](RELEASING.md).

## Current architecture

The microphone feeds Silero VAD, then Smart Turn endpoint detection. Moonshine
transcribes the completed utterance; transcript matching activates the assistant.
Deterministic tools run first. A matching tool either answers directly or adds
knowledge context. Otherwise, local Gemma on MLX or AWS Bedrock generates an
answer, optionally after one model-requested tool call. Kokoro streams synthesized
audio once the full response text is available.

The FastAPI admin service has its own chat-model instance. It can observe a
separate voice process through a status file, or own a child process through
`RuntimeSupervisor`. Knowledge is local Markdown with keyword, vector, and
hybrid retrieval. Optional memory is generated and consolidated by the chat
model. User tools are Python modules, including modules editable in the console.

Useful foundations to preserve:

- Typed configuration and small STT, TTS, wake, and chat adapters.
- Exact deterministic answers without model inference.
- Local knowledge retrieval with index freshness checks and keyword fallback
  when a hybrid index is missing or stale.
- Authenticated management routes, path containment checks on knowledge writes,
  AWS credential redaction in configuration responses, and loopback binding by default.
- A committed `uv.lock`, 57 passing baseline tests, and detailed operational docs.

The baseline tests cover config, API behavior, knowledge, tools, wake matching,
model calls with fakes, status, and supervision. There are no dedicated tests for
the microphone loop, playback cancellation, real STT/TTS, memory, or prompt
history. No repository CI workflow or local release tags were present.

### Locked runtime baseline

These versions were read from `uv.lock` and matched the installed environment.
They are a reproducibility baseline, not a recommendation to upgrade everything.

| Component | Current implementation |
| --- | --- |
| Application | `0.1.0`; Python `>=3.13`; Apple Silicon is the required uv environment |
| STT | `moonshine-voice 0.0.54`; project requirement explicitly caps versions below `0.0.55`; non-streaming transcription call |
| TTS | `kokoro-onnx 0.5.0`, Kokoro v1.0 model and voice assets |
| Local chat | `mlx-vlm 0.4.4`, `mlx 0.31.1`; default `mlx-community/gemma-4-E4B-it-4bit` |
| Endpointing | `silero-vad 6.2.1`; Smart Turn asset named `smart-turn-v3.2-cpu.onnx`; default silence window 700 ms |
| Audio | `sounddevice`, `livekit 1.1.6` for echo cancellation; 16 kHz, 512-sample input chunks |
| Knowledge | BGE small English v1.5; custom lexical ranking and NumPy vector search |
| Supporting inference | `torch 2.11.0`, `transformers 5.5.4`, `onnxruntime 1.24.4` |
| Web service | `fastapi 0.136.0`; service, routes, and embedded frontend share a roughly 3,000-line module |

## Recommendations in priority order

### 1 Protect personal data throughout the application

**Priority: immediate.** The new ignore rule protects additional files and
subdirectories under the default `knowledge/` directory. The two starter
documents remain tracked, and the article template now lives in `examples/`.
No custom knowledge files were tracked in the current Git index. Ignore rules
cannot prevent forced adds, protect edits to tracked starter files, or erase
earlier commits. Add a CI check that rejects tracked knowledge paths outside the
two-file allowlist; this gives protection beyond a local ignore convention.

There are separate privacy issues to fix in the application:

- [`runtime.py`](src/arietta_voice/runtime.py), `process_utterance`, logs every
  transcription before checking the wake word and also logs wake misses.
  Passive speech can therefore persist in JSONL and process logs. Make text
  logging opt-in, omit passive transcripts from status, and add retention limits.
- [`api.py`](src/arietta_voice/api.py), `public_health`, returns the same detailed
  payload as authenticated health. That payload includes runtime transcripts,
  assistant responses, paths, and diagnostics. A synthetic check confirmed that
  `/health` returns transcript text without credentials. Return only minimal
  liveness/version information publicly; retain details on authenticated routes.
- `config/arietta_voice.toml` and `config/SOUL.md` are tracked personalization
  files. The config accepts AWS credentials. Introduce ignored local overrides
  and first-run creation from a public template, preserve existing settings,
  and prefer the AWS credential chain. The user's current config was not edited.
- Local STT and retrieval do not mean every interaction stays local: Bedrock
  receives assembled prompts containing transcript text, retrieved snippets,
  and enabled memory. Make that boundary explicit when choosing a provider.

### 2 Make console privileges explicit

**Priority: immediate.** [`api.py`](src/arietta_voice/api.py) falls back to
`admin` / `password`, while [`tool_admin.py`](src/arietta_voice/tool_admin.py)
allows source editing and executes modules during validation and inventory.
Console access therefore grants code execution as the assistant's OS user.

Require credentials to be configured at setup, keep loopback as the default,
and reject non-loopback startup with default credentials. Put source editing
behind an explicit development setting. Use static inspection for inventory;
perform executable validation only as an explicit operation in a subprocess
with a deadline. Add atomic saves, a last-known-good copy, and rollback on
validation failure. AST checks alone do not make Python plugins safe.

### 3 Correct behavioral inconsistencies before changing models

**Priority: next patch or foundation release.** These are small, concrete
correctness issues with clear regression tests:

| Finding | Evidence | Recommended behavior |
| --- | --- | --- |
| Disabling tools only disables deterministic routing | `AriettaApiService.chat` always builds a model-tool registry; a synthetic request with `use_tools=False` executed a fake model tool | Disable both tool paths for that request |
| History counts messages in one place and pairs in another | `runtime._store_turn` retains `max_history_turns * 2` entries; `PromptManager.build_messages` slices only `max_history_turns`; eight pairs became four | Define a turn as one user/assistant pair and enforce that consistently, including zero history |
| Goodbye matching can end unrelated requests | `GOODBYE_PATTERN.search` matches phrases anywhere in the utterance | Recognize a session-ending intent, including tests such as “what does goodbye mean?” |
| Wake matching changes inline command content | `TranscriptWakeWordDetector.match` returns a lowercased, punctuation-stripped command | Match a normalized wake phrase but preserve the original command text |
| Configuration conversion lacks semantic validation | `config.py` casts values without validating ranges and combinations | Reject zero memory intervals, invalid history limits, unsupported sample rates/chunk sizes, and invalid retrieval weights at startup |

### 4 Add bounded work and dependable recovery

**Priority: foundation release.** [`audio.py`](src/arietta_voice/audio.py) has
an unbounded microphone queue; recordings accumulate in RAM until exit.
[`runtime.py`](src/arietta_voice/runtime.py) has no maximum utterance duration,
and inference blocks the listening loop. Repeated negative endpoint scores can
keep growing the speech buffer. Warmup also occurs before the runtime's
main error/final-status handling.

Bound audio queues and utterance duration, stream recordings to disk, and define
how overflow is reported. Add timeouts, cancellation, and per-turn recovery for
model/tool failures. Keep heartbeat publication independent of blocking
inference so long downloads or generations do not appear as dead processes.
Close output streams in `finally` blocks. Serialize access to each model and
runtime lifecycle operation; concurrent synchronous API routes currently share
mutable service state without explicit locking. Eventually use one inference
worker so console chat and voice do not load competing copies of the local model.

### 5 Stream the whole response pipeline

**Priority: largest conversational improvement after the foundation work.**
Current audio streaming begins only after complete model generation. With
memory enabled, additional memory-model calls happen before playback as well.

Introduce cancellation-aware interfaces for STT partial/final events, chat text
deltas, tool events, and TTS audio chunks. Feed finalized clauses or sentences
to Kokoro while the model continues generating. Keep incomplete tool JSON and
internal reasoning out of speech. Move memory updates off the response path.
On interruption, cancel generation, clear queued playback, and record only the
portion of the response actually delivered in conversation history.

Measure speech end to first audible response, rather than only tokens per
second. Add per-stage timings for endpoint delay, STT finalization, retrieval,
first model token, first TTS chunk, and playback start. Existing telemetry mainly
records STT latency and state transitions, so there is no measured end-to-end
baseline yet.

### 6 Improve interruption and wake behavior

**Priority: streaming release.** Voice barge-in currently requires AEC to be
enabled and waits through a 500 ms playback grace period. The AEC reference uses
a fixed 160-sample frame size and an estimated stream offset; arbitrary sample
rate configuration is not a supported DSP contract. Validate the 16 kHz path,
measure echo alignment on the target speakerphone, and test speech onset capture,
false interruptions, and stopping latency. Add device selection and recovery
after sleep or USB reconnection.

Keep transcript wake detection available for arbitrary names. Prototype an
acoustic detector behind a separate audio-frame interface and a short pre-roll
buffer. The existing `match(transcript)` protocol is not sufficient by itself
for acoustic detection. Select a backend only after testing custom-word training,
macOS installation, false activations, and model licensing.

### 7 Fix retrieval correctness before buying more model capacity

**Priority: foundation fix, then retrieval release.**
[`BgeTextEmbedder`](src/arietta_voice/knowledge.py) uses mean pooling, whereas
the BGE model's reference Transformers integration uses the first token, CLS,
then normalization. Correct the implementation against a small reference
fixture; bump `INDEX_VERSION`, record pooling and model revision, and rebuild
existing vectors. Do not silently query vectors created by the old encoder.
[BGE model card](https://huggingface.co/BAAI/bge-small-en-v1.5#using-huggingface-transformers)

Heading-based sections can exceed the embedder's 512-token truncation limit.
Add bounded chunks with modest overlap and preserve titles, headings, and source
IDs. Guard against symlinks escaping the configured root during retrieval as
well as during edits. Publish index generations atomically; handle corrupt
manifests/vectors with an explicit degraded status. Build a synthetic question
set measuring relevant-source recall, irrelevant matches, and answerable versus
unanswerable questions before tuning hybrid weights or adding a reranker.

Return source references in API answers. Treat retrieved passages and model-tool
output as untrusted evidence; currently both can become system messages.
Instruction boundaries and tool authorization must be enforced outside the model.
Keep flat-file/NumPy storage until measured scale warrants a database.

### 8 Strengthen tools and memory before adding actions

**Priority: retrieval and tools release.** Preserve deterministic routing, but
replace examples-only model arguments with explicit schemas. Use provider-native
tool calls where supported, with a bounded fallback for local models. Validate
arguments, enforce timeouts and result size limits, and return structured errors
instead of speaking raw failed tool requests. Initially retain the one-tool-call
budget. Require confirmation and idempotency for consequential device actions.

Memory should stay opt-in. Add inspect/delete controls, provenance, atomic
writes, length limits, and background consolidation. Keep model-generated memory
from becoming unrestricted system instructions or overwriting a good file after
an incomplete generation. Add tests for extraction failures and consolidation.

### 9 Make operation and customization survive upgrades

**Priority: incremental foundation and operations work.** Extract API routes,
service code, and frontend assets from `api.py` in behavior-preserving steps;
a new frontend framework is not necessary. Avoid repeated tool imports and
full-log scans on dashboard polling. Add log rotation and incremental event reads.

Move custom tools to a user data directory or installed extension packages
instead of editing the application's installed source tree. Add an `init`
command, packaged default assets, config schema/migrations, and a single documented
process-ownership mode for each installation. A wheel currently builds, but
does not include the default config/SOUL/knowledge assets; normal setup still
depends on a repository checkout. Keep that distinction clear before publishing
an installable end-user package.

Add CI with locked uv installs, unit/contract tests, build verification, a
knowledge-path allowlist check, and dependency/secret scanning. Use a separate
Apple Silicon integration job or manual release checklist for actual model and
audio validation. Upgrade related MLX/Transformers/ONNX dependencies in tested
groups; the existing lockfile is valuable and should remain committed.

## Technology options worth evaluating

Sources were checked October 11, 2026. Upstream moving branches can describe
features beyond a stable package release; select an explicit release and model
revision for each experiment. These are candidate experiments, not measured
speed or quality improvements on the user's Mac.

| Option | Relevant advancement | Decision for Arietta |
| --- | --- | --- |
| Moonshine streaming | The upstream 0.1.x changelog documents speculative STT decoding, terminology hints, and newer streaming APIs, alongside breaking model-format/API changes. [Changelog](https://github.com/moonshine-ai/moonshine/blob/main/CHANGELOGS.md) | Highest-priority speech upgrade experiment. Keep `0.0.54` as the comparison and fallback; explicitly select architecture/assets and test before lifting the `<0.0.55` cap. |
| MLX-VLM streaming and caching | Current upstream supports `stream_generate`, prefix caching, and Gemma 4 draft-model experiments. [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) | Stream with the existing Gemma model first. Test in-memory cache reuse with clear invalidation. Defer speculative decoding until first-audio latency is measured; it consumes additional memory. |
| MLX-LM and alternate small chat models | MLX-LM supports streaming and prompt caching; Qwen3.5-4B is another published small model candidate. [MLX-LM](https://github.com/ml-explore/mlx-lm), [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-4B) | Compare a verified compatible MLX conversion against Gemma on short grounded answers, tool selection, RAM, and first-token delay. Keep MLX-VLM where model packaging requires it; removing vision UI does not guarantee MLX-LM compatibility. |
| MLX Audio | Supports Kokoro and Qwen3-TTS, with streamed audio generation on Apple Silicon. [MLX Audio](https://github.com/Blaizzy/mlx-audio) | Benchmark MLX Kokoro against current ONNX Kokoro first; evaluate small Qwen3-TTS for voice quality separately. Account for GPU contention with the LLM. Keep ONNX as the fallback until a candidate wins. |
| Smart Turn and Silero | Smart Turn remains designed for 16 kHz mono audio, using up to eight seconds and a lightweight VAD gate. [Smart Turn](https://github.com/pipecat-ai/smart-turn#notes-on-input-format) | Keep the two-stage design. Tune silence/endpoint behavior and pin the existing asset by revision/hash before pursuing replacement. |
| Acoustic wake detection | openWakeWord provides an audio-based detection/training path; its included pretrained models carry a noncommercial license separate from its code. [openWakeWord](https://github.com/dscripka/openWakeWord) | Prototype only. Verify the exact custom model's terms and native macOS dependencies before adoption; retain transcript wake mode. |
| Bedrock streaming and tools | `ConverseStream` supports streamed output and tool configuration for supported models. [AWS API](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html) | Extend the existing Bedrock adapter with streaming and structured tool events. Check model/region capability at setup rather than hardcoding a replacement model ID. |
| Cloud speech-to-speech | OpenAI Realtime supports audio sessions, tools, and interruptions over WebRTC/WebSocket; Nova 2 Sonic offers bidirectional speech sessions and tools. [OpenAI Realtime](https://developers.openai.com/api/docs/guides/realtime), [Nova 2 Sonic](https://docs.aws.amazon.com/nova/latest/nova2-userguide/using-conversational-speech.html) | Optional later experiment behind a session-level voice interface. These require cloud audio, connection lifecycle, cost controls, and explicit user selection; they cannot be implemented merely by replacing `ChatModel.generate`. |

Do not make a framework rewrite, larger default model, voice cloning, vector
database, or cloud speech a prerequisite for better conversation. The current
architecture has usable seams for incremental improvements. Reconsider a voice
orchestration framework if browser/mobile transports and distributed audio become
core requirements.

## Verification and limits

- Baseline: `uv run --no-sync pytest` passed **57 tests**. After the version
  changes, **60 tests** pass, including CLI early exit and API version agreement.
- Synthetic, temporary-directory checks reproduced public-health transcript
  exposure, model-tool execution with `use_tools=False`, and history truncation.
  No real model invocation was needed for those checks.
- `git check-ignore` verified private Markdown, text, hidden files, nested
  directories, and a nested README are ignored; only the two root starter paths
  are allowed. The user's custom articles were not read or changed.
- `uv build --offline` produced a wheel and source archive labeled `0.1.0`.
  Both exclude the local config and knowledge directories.
- Audio quality, microphone reliability, AEC, latency, long-running stability,
  cloud cost, and new dependency compatibility still require real integration
  measurements. The code findings above are not claims of measured hardware performance.

Only knowledge-file hygiene, documentation, and version reporting were
implemented in this review. The other findings are queued in the roadmap.
