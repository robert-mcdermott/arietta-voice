# Changelog

User-visible changes to Arietta Voice. See [docs/releasing.md](docs/releasing.md) for
version policy and the release procedure.

## Unreleased

### Added

- Static checks of the console's JavaScript in the test suite: every imported
  name is exported, every element looked up by id exists in the page, and each
  module parses (with Node, which the CI runners have).

### Changed

- The README's console screenshot shows the current design.

### Fixed

- New built-in tools now switch on after an upgrade even when the private
  config has its own tool list. Before, `config-init` and the console's Tools
  page saved the whole list, so tools added by later releases stayed off until
  switched on by hand. The private config now also records
  `[tools].disabled`, the standard tools you switched off; a standard tool in
  neither list is added just before the next standard tool in yours. Lists
  saved by earlier versions are read against the release they came from, so
  tools you switched off stay off.

## 0.2.0rc2 — 2026-11-14

Second release candidate for 0.2.0: adds the calculator, date questions,
optional weather, a spoken filler for slow tools, streaming console chat and
editable settings, and fixes the problems an independent review found in the
first candidate's new code (see Fixed). The final 0.2.0 still waits on the
30-minute audible soak.

### Added

- A `calculate` tool for arithmetic, percentages, powers, roots and unit
  conversions (length, weight, volume, speed, time and temperature), said in
  words or digits: "what's seventeen times twenty three?", "15% of 80",
  "convert 5 miles to kilometers", "how many cups are in a liter". Common
  phrasings are answered instantly; the model can call it for word problems.
  Expressions are evaluated from a parsed syntax tree of numbers and operators
  only, with limits on size. Enabled by default; add it to a private
  `[tools].enabled` list to use it there.
- A `date_math` tool for days until, since and between dates, the day of the
  week a date falls on, and the date after some days or weeks: "how many days
  until Christmas?", "what day of the week is July 4th?", "what's the date in
  three weeks?". It reads holidays (including Easter, Thanksgiving, Mother's
  Day and Labor Day), month-and-day dates in digits or words, `12/25`, ISO dates
  and weekdays. Enabled by default.

- An optional `weather` tool: current conditions and a seven-day forecast,
  rain and umbrella questions, the weekend, and other places ("what's the
  weather in Paris on Friday?"), from Open-Meteo (no account or key). It is off
  until `[weather].location` is set and `weather` is added to `[tools].enabled`;
  it sends the place name and its coordinates to open-meteo.com, and caches
  answers for ten minutes. New `[weather]` settings: `location` and `units`.

- When a tool is still working after 0.8 seconds (the weather service, say), the
  assistant says "One moment." while it waits, once per turn. The filler is
  spoken only; it is not part of the answer shown in the console.

- The console's Settings page can change common settings: the assistant's name,
  wake word and conversation timeout, voice and speaking speed, the thinking
  chime, model temperature and answer length, native tool calls, timer alerts,
  weather, memory, conversation logging and the knowledge search backend. Each
  change is checked with the runtime's own validation and saved to the private
  config (`GET`/`PUT /api/settings`); the page offers to restart the voice
  runtime. Everything else still lives in the private config file.
- The console's Chat page shows answers as they are generated, clause by clause,
  with the time to the first words, through a new `POST /api/chat/stream`
  endpoint (server-sent events). `POST /api/chat` is unchanged.

### Changed

- A reminder set a day or more ahead with a delay ("in three weeks") that would
  fall at night now goes off at 9 AM that day instead.
- A tool file in `tools/` now replaces a built-in tool of the same name, so a
  built-in added in an upgrade (such as `weather`) never hides a tool you
  already have.
- Timer and reminder apologies read naturally ("Sorry, timers can run from 1
  second to 24 hours; for longer, ask for a reminder.").

### Fixed

- A console chat stream closed before its first event (a tab closed while the
  worker started) no longer leaves every later chat refused as busy until the
  console restarts.
- Alarms and reminders set across a daylight-saving change fire at the hour
  asked for, not an hour off; delays across the change count real time.
  Impossible hours ("24 am", "13 pm") are refused.
- An instant-answer matcher that trips on odd input (a date three million days
  away, a 300-digit number) no longer fails the turn and restarts the worker.
- "How long until Christmas?" is answered by `date_math` instead of the timer
  tool.
- While an alert rings, "cancel my dentist reminder" is no longer taken as
  "stop" for the ringing timer, and without the wake word only a clear
  dismissal ("stop", "turn it off", "snooze") silences it; "okay" or "thanks"
  said to someone else, or by a television, does not.
- The calculator answers every failure plainly ("too big to say", "doesn't have
  a real answer") instead of handing the question to the model.
- Weather for a named place uses that place's own date for "today" and
  "tomorrow"; odd service replies are reported as unreachable.
- An alert whose ring cannot be recorded (an unwritable timers file) is said
  once instead of every 30 seconds forever, and a hand-edited entry with an
  impossible time is dropped instead of stopping the runtime.
- A garbled native tool call gets a spoken apology instead of failing the turn
  and reloading the models; JSON-quoted tool arguments may contain commas.

## 0.2.0rc1 — 2026-11-14

Release candidate for 0.2.0, the first tagged release: everything since the
0.1.0 baseline. The final 0.2.0 waits on one 30-minute audible soak on the
default ONNX speech stack (Milestone 0 in the roadmap).

### Changed

- One tool interface replaces the separate "deterministic" and
  "model-requested" tools. Each tool has a name, a description of when to use
  it, a JSON schema for its arguments, optional trigger patterns that answer
  instantly, a time limit, and a result with `content` for the model and an
  optional `speak` answer. Arguments are validated before a tool runs; results
  are bounded. Built-in tools are now named `assistant_identity`, `local_time`,
  `search_knowledge` and `system_info`, and `local_time` can also be called by
  the model (for example for the date).
- Your own tools now live in `tools/` (set by `[tools].directory`), outside
  the installed package. Only `tools/README.md` and `tools/example_dice.py` are
  tracked; release checks reject other tracked files there. Existing configs
  keep working: `[model_tools].enabled` is merged into `[tools].enabled` and
  `knowledge_retrieval` maps to `search_knowledge`. Tools written for the old
  interfaces must be ported (see `tools/README.md`).
- When the model asks for a tool that is unknown or disabled, sends invalid
  arguments, or the tool fails or times out, the model is told and answers
  without it. Previously the turn failed.
- The console's Tools page shows one ordered list with instant-answer and
  model-call badges, a per-tool Test (sample request or arguments), and a
  preview of what the model is told. The `/api/model-tools/*` routes are
  replaced by `/api/tools/*`, including `/api/tools/{name}/test` and
  `/api/tools/instructions`.
- Speech recognition upgraded from Moonshine 0.0.54 to 0.1.6, lifting the
  last dependency cap. Each utterance is now transcribed in its own stream, so
  earlier utterances can no longer add stray words (the old recogniser once
  heard "Yeah How many sides does a triangle have?"). Transcription takes about
  180 ms instead of 210 ms; median time to first audio in the latency benchmark
  went from 1.62 s to 1.56 s. The new model (about 270 MB) downloads on first
  run; the old one (about 430 MB) can be deleted from
  `~/Library/Caches/moonshine_voice/download.moonshine.ai/model/medium-streaming-en/quantized`.
- All dependencies upgraded to their latest releases (81 packages), including
  transformers 5.5.4 to 5.23.0, torch 2.11 to 2.15, mlx 0.31 to 0.32, mlx-vlm
  0.4.4 to 0.7.12, huggingface-hub 1.x to 2.4, onnxruntime 1.24 to 1.32,
  kokoro-onnx 0.5 to 0.6, livekit 1.1 to 1.2 and fastapi 0.136 to 0.162.
  Transcription, end-of-turn detection, voice activity, echo cancellation,
  knowledge embeddings and chat answers were unchanged with the real models;
  synthesised speech differs very slightly with the new Kokoro release. Median
  time to first audio in the latency benchmark went from 1.75 s to 1.68 s.
  Run `uv sync` after updating. The environment is about 0.4 GB larger.
- `moonshine-voice` stays below 0.0.55 because 0.1.x changed its API and model
  format; the reason is now recorded in `pyproject.toml`.
- Silence around each synthesised clip is trimmed to 40 ms before and 100 ms
  after (`audio.tts_trim_silence`, on by default). ONNX Kokoro is already close
  to these margins. MLX Kokoro pads every clip with about 0.3 s before and 0.4 s
  after, which delayed the first word and left pauses of about 0.7 s between
  clauses.
- Faster turn-taking. The end of a turn is now also checked after 250 ms of
  silence (`audio.endpoint_early_ms`), and work on the answer starts at once
  when Smart Turn is at least 70% confident. The assistant keeps listening
  until `audio.silence_ms` has passed; if you carry on speaking it discards
  that attempt and continues with the whole utterance, logging `turn_resumed`.
  The thinking chime now starts only once that window has passed. Set
  `audio.endpoint_early_ms = 0` for the previous behaviour.
- Faster first response. With model tools enabled, a direct answer is now
  spoken as it is generated instead of after a complete, separate planning
  generation; only a response that starts as a JSON tool request is held back.
- The first spoken clause may end at a comma or dash once it is 25 characters
  long, so speech can begin before a long opening sentence finishes.
- The local chat model reuses the part of each conversation prompt it has
  already processed (persona, tool instructions, earlier turns).
- The web console is redesigned. It now has Home, Chat, Knowledge, Tools,
  Activity, System and Settings sections, light and dark themes that follow the
  system preference, and a layout that works at phone width. Home shows what the
  assistant is doing; audio counters and process detail moved to System. The
  event log is described in plain language, and Settings groups the effective
  configuration by section.
- Console pages use `#/section` addresses, so a section can be bookmarked.
- Model tools can be invoked from the Tools list again. The button had been
  permanently disabled since the tool inventory stopped importing tool code.
- The console frontend is served as packaged static files under
  `/static/` (authenticated like the rest of the console) with a content
  security policy that forbids inline script and style. The API implementation
  moved to `arietta_voice.web`; `arietta_voice.api` remains as an import alias.
- Repository layout: guides now live in `docs/` (`getting-started.md`,
  `architecture.md`, `releasing.md`), images in `assets/`, and the latency,
  hardware and soak harnesses in `bench/` (previously `scripts/`). Concluded
  Kokoro, Moonshine, ONNX-environment and underflow experiments are archived
  under `bench/archive/` and `docs/archive/experiments/` and no longer run with
  the default test suite; run them with `uv run pytest bench/archive/tests`.
- `config/arietta_voice.toml` is now the single shared template and lists every
  supported key at its built-in default. The duplicate `.dist` and `.example`
  templates are removed. Private `*.local.toml` overrides are unaffected.
- The roadmap is replaced by an outcome-based plan; the October plan and review
  are preserved under `docs/archive/`.
- Source is formatted and linted with `ruff`, enforced in CI.

### Removed

- The unused image-input method on the local chat model.

### Added

- Native tool calling for the local model. Tools are declared through Gemma 4's
  own chat template instead of being described in the prompt with a JSON reply
  format, and the model may call up to three tools in a turn, with each result
  sent back as a tool message. In a real-model check it chose the right tool
  for every request, including phrasings the prompted format missed ("Is
  anything counting down right now?", "Scrap the tea countdown"). Words before a
  call are spoken and the answer continues after it; Gemma's hidden reasoning
  channel is ruled out, since it cost 10 to 20 seconds per answer. New setting
  `models.native_tool_calls` (on); Bedrock keeps the prompted format.
- Tools can set `speaks_for_itself`: when the model calls one and it returns
  `speak`, that is the answer, without a second generation. The timer,
  reminder and alarm tools do, so a model-called timer is confirmed exactly.
- Timers, reminders and alarms. "Set a 10 minute pasta timer", "remind me to
  call Mom at 5:30", "wake me up at 7", "how much time is left?" and "cancel the
  pasta timer" are answered instantly by new built-in tools (`set_timer`,
  `set_reminder`, `set_alarm`, `check_timers`, `cancel_timer`); the model can
  call all but `set_alarm`, and asks for a reminder instead. Durations and
  times are understood in digits or words. When one is due the assistant plays a
  tone and says it between turns, repeating every 30 seconds until answered (up
  to four more times); "stop", "okay" or "snooze for 10 minutes" work without the
  wake word while it rings. They are kept in `state/timers.json` (private), so
  they survive restarts, and anything missed while the assistant was off is
  announced when it starts. New `[timers]` settings: `path`, `repeat_s`,
  `max_repeats`, `snooze_minutes` and `alert_tone`. A private config that lists
  `[tools].enabled` needs the five tools added to use them by voice.
- The console's Home page lists timers, reminders and alarms with live
  countdowns, and can add, stop and cancel them (`/api/timers`).
- Tools can declare `aliases` for argument names a model might use instead
  (`{"note": "text"}`), and a number sent for a text argument is accepted as
  text. The tool instructions now tell the model to answer general knowledge
  itself and never to claim it set or cancelled something no tool did.
- Prompt-prefix reuse now keeps working past Gemma's 512-token sliding window.
  Its sliding-window layers used a ring buffer that cannot be rewound once a
  prompt and answer pass 512 tokens, so every later turn reprocessed the whole
  prompt; they now use a plain cache with the same window applied by the
  attention mask. Greedy answers matched the stock cache in 11 of 12 checks
  (the twelfth differed late in a long answer).
- Optional MLX Kokoro speech from the main install: `uv sync --group
  mlx-speech`, download the model, and set `audio.tts_backend = "mlx"` (see
  README). It no longer needs a separate trial environment or exact package
  versions. In the latency benchmark it answered in 1.28 s median against
  1.59 s with ONNX Kokoro. ONNX remains the default until a live soak passes.
- A `turn_timing` event for each spoken microphone turn, with milliseconds
  from the end of speech to the end-of-turn decision, first audio and turn end,
  plus transcription time. It carries no conversation text. The latest values
  are also in the runtime status under `reliability.last_turn_timing`.
- `bench/latency_baseline.py --model-tools` measures the tool-planning path.

- Isolated continuous live/load soak with paced synthetic input, real microphone
  capture/AEC and streamed speaker output, live checks before/after continuous load,
  bounded control phases, and per-turn model/memory plus host telemetry.

- Isolated forced-gap live validation with a test-only spoken command, bounded
  pause injection, AEC/capture handoff coverage, and separate gates for quiet
  playback and interruption during/after the pause. Metadata verifies actual
  padding and rejects missing scenarios or audio faults; production settings
  and the normal conversation path remain unchanged.

- Speaker-output timing diagnostics and an isolated MLX/Jabra underflow probe
  with matched continuous, producer-gap and deliberate scheduling-stall controls.
  Playback now supplies bounded silence during post-start producer waits, honoring
  device capacity, cancellation and AEC reference alignment. Real underflow flags
  remain visible; short speech-channel timeouts now respect their requested limit.

- Capture-overrun counts by runtime phase, separate intentional-discard counters
  in audio status/admin, and reason-specific AEC reset/playback discard events.
  Expected packets crossing playback startup no longer reset a fresh AEC;
  transcription and first-output waits retire capture that would otherwise be
  discarded at their boundary, preventing unused backlogs from filling the queue.
  Real gaps/overflows still invalidate interruption prefixes and reset alignment.

- Isolated MLX hardware-test overrides, host telemetry and continuous shared-model
  soak checks, including varied synthetic fixtures, configurable turn spacing,
  per-turn allocator samples and failure on worker replacement or missing models.
  Live playback/AEC/interruption observations are recorded separately from
  automated gates; private settings and the ONNX default remain unchanged.

- Counterbalanced production ONNX, trial ONNX and MLX conversation comparison
  with 100 warm turns per condition, separate initial turns, matched synthetic
  inputs/history, per-block results and read-only macOS host-state monitoring.
  The isolated runner checks private-file and dependency preservation and keeps
  raw evidence under ignored `tmp/`.

- Isolated ONNX environment diagnostics with dependency/native-code fingerprints,
  phonemization/inference/trimming timings and guarded conversation comparisons.
  Reversed-order measurements did not reproduce the earlier trial-environment
  slowdown; production dependencies and settings remain unchanged.

- Opt-in MLX Kokoro adapter in the shared inference worker, explicit local model
  selection, dependency/asset diagnostics, and bounded materialized audio frames.
  A separate pinned trial environment supports grouped conversation benchmarks,
  identical input fixtures, shared-model memory telemetry and cancellation/recovery
  checks. ONNX remains the default; production dependencies are unchanged.

- Isolated ONNX/MLX Kokoro comparison with pinned candidate dependencies,
  bounded synthetic benchmarks, latency/CPU/memory reports and blinded listening
  pairs. Production speech synthesis and private settings remain unchanged.

- Input/output selection by exact or host-qualified name in private config and
  CLI, direction/default metadata in `devices`, and selected-format diagnostics.
- Automatic audio reacquisition after returned device errors, callback stalls
  and sleep/scheduling gaps, with bounded backoff, fresh sessions and no turn or
  tool replay. Explicit recordings survive stream reopens without truncation.
- Admin device/recovery status and synthetic device failure regression coverage.
  Physical USB and sleep/wake validation is still pending.

- Timestamped microphone capture and bounded pre-roll for listening and voice
  interruption, with documented `audio.pre_roll_ms` and `audio.aec_delay_ms`
  settings. Playback events report alignment resets and output underflows.

- Isolated Moonshine 0.1.5 experiment comparing batch/streaming transcription,
  language catalogs, model assets and terminology hints against locked 0.0.54.
  Synthetic fixtures, bounded subprocess runs and partial/final/error regression
  tests accompany the measured upgrade assessment; production STT is unchanged.

- Bounded background memory jobs after delivered voice responses, scheduled during
  quiet listening on the existing model worker. New speech takes priority; memory
  queue, completion, cancellation, failure, drop and conflict counters are exposed.

- MLX and Bedrock token streaming with early clause-based voice playback,
  bounded sentence segmentation, and interruption polling between clauses.
- Provider stream, early-audio, tool-planning and partial-turn regression tests.
- Per-turn cancellation across inference and speech, a web Chat Test Stop action,
  and bounded text/audio handoffs with backpressure and turn deadlines.
- Cancellation, queue-pressure, stale-result and recovery regression coverage;
  a short isolated real-model smoke and browser Stop/recovery check.
- Opt-in Phase 2 latency benchmark with paced synthetic questions, real output
  device writes, separate startup/first-turn/warm timings, stage and resource
  reports, and private-settings preservation.
- Opt-in local hardware/model smoke and bounded live/repeated-model soak
  harnesses, with isolated settings, private-file checks, process cleanup,
  resource/heartbeat reports and a hands-on audio checklist.
- Recorded Apple M4/Jabra hardware smoke and 15-minute live soak results,
  including manual interruption/farewell checks and baseline limitations.

- Apple Silicon CI for locked installs, isolated unit/process tests, synthetic
  evaluations, distribution builds and version/private-file checks. Actions are
  pinned; only synthetic JUnit reports are uploaded.
- A 43-case synthetic corpus for wake/session behavior, prompts, memory, tool
  policy and keyword retrieval; corrupt-index and config snapshot regressions.
- Source distributions include tests/evaluations and exclude local config and
  knowledge. Tracked-file checks reject force-added private paths.

- Configurable microphone, utterance, recording, and echo-reference limits;
  streamed WAV recording with pressure/error reporting.
- Serialized inference processes with turn, endpoint, and startup deadlines;
  model/tool failure recovery and an independent runtime heartbeat.
- Admin reliability cards, recording/recovery warnings, expandable runtime
  details, clearer errors, and unsaved source/knowledge editor warnings.

- `config-init` creates ignored, owner-only private config/persona snapshots and
  backups while retaining original files. Sibling `.local.toml` settings override
  the selected base config; CLI runtime overrides take precedence.
- `admin-setup` creates a private salted PBKDF2-SHA256 password hash. Credentials
  are now required; explicit environment credentials remain supported.
- Admin Host/origin checks and non-cacheable responses.

- `arietta-voice --version` and `arietta-voice run --version` without loading
  configuration, models, or audio hardware.
- Project review, phased roadmap, and release/versioning instructions.

### Changed

- AEC uses a timeline of submitted playback blocks and continuous 10 ms frames,
  preserving clause gaps and cleaned interruption audio. Keyboard cancellation
  now covers generation gaps and final playback draining; capture loss resets
  alignment and prevents splicing discontinuous follow-up audio.

- Memory extraction/consolidation use an independent background deadline and no
  longer delay return to listening or extend the interactive turn duration. Jobs
  stay in RAM, and interrupted/overflowed jobs are discarded without retries.

- Console and voice share conversation, prompt, history and tool policy through
  one service implementation, with separate worker processes. Voice streams
  speakable clauses; web chat continues to return one complete response.
- Removed the built-in admin password and authentication bypass. Credential
  setup must complete before starting the API.
- Tool creation/source writes require `admin.allow_source_editing = true`.
  Inventory and source reads inspect syntax/literal metadata without loading
  plugins. Runtime validation remains an explicit authenticated action.
- Tool enablement changes persist atomically in the private config instead of
  changing tracked defaults. Instruction preview now uses POST rather than GET.

- Package and API version reporting now read installed metadata derived from
  `pyproject.toml`, removing duplicate hardcoded version declarations.
- New knowledge files and subdirectories are ignored by default. Only
  `knowledge/README.md` and `knowledge/getting-started.md` are allowed by the
  repository's ignore rules.
- The knowledge article template moved from `knowledge/_template.md` to
  `examples/knowledge-template.md`; customization documentation follows it.
- macOS `.DS_Store` files are ignored.
- Public `/health` now returns only liveness and version. Authenticated
  `/api/health` retains detailed diagnostics.
- Active transcript/response logging and runtime status text now require
  `logging.content_enabled = true`. Passive wake misses never log text.
  Interactive text-only answers remain visible; redirected responses follow
  the content setting.
- JSONL event logs rotate with configurable `max_bytes` (default 5,000,000)
  and `backup_count` (default 3), with process-safe appends/rotation and
  owner-only file permissions. Oversized events are replaced with markers.
- Config loading rejects unknown keys, invalid types, non-finite values, and
  unsupported audio/retrieval/generation settings before model startup.

### Fixed

- Background memory generation commits one atomic proposal after checking for
  intervening file edits. Failed extraction/consolidation and cancelled proposals
  leave existing memory intact; shutdown clears queued private conversation text.

- Tool-planning payloads are withheld from speech; malformed or unavailable model
  tool requests fail without exposing their JSON. Failed/incomplete provider
  streams discard pending audio and do not store a partial conversation turn.
- Interrupted speech aborts output before generation cleanup and drops queued
  audio. Cancelled partial replies are not stored or memorized; cancellation of
  post-delivery memory work preserves the answer without reporting a turn failure.
- Malformed, incomplete or incompatible semantic index files fall back to keyword
  search in hybrid mode (no result in vector mode); explicit builds repair corrupt
  caches instead of treating them as current. Source documents remain untouched.
- Memory extraction treats `NONE` as an exact sentinel, preserving facts that
  contain that substring in a name or word.

- Failed/timed-out inference is stopped before the next job; failed turns are
  not stored or automatically retried. Playback cleanup closes resources on
  generation/device failures. Memory updates use private atomic replacements.
- Web chat prevents duplicate submissions and duplicate current-message history,
  keeps failed drafts, and returns safe busy/timeout/failure responses.

- API `use_tools=false` bypasses both deterministic and model-requested tools,
  including model-tool registry construction.
- History limits consistently count user/assistant pairs; zero clears history.
- Inline wake commands retain original case, punctuation, numbers, and Unicode.
- Mentions of goodbye inside a question no longer end the session.
- Runtime status omits conversation text by default, including when reading
  older status files, and session reset clears previous conversation text.
- Voice-runtime failures suppress potentially private exception details unless
  content logging is enabled, while preserving a nonzero exit status.
- BGE v1.5 uses normalized CLS embeddings instead of mean pooling. Index
  format 2 invalidates incompatible cached embeddings.
- Knowledge retrieval excludes links outside its root and hidden directories.

### Migration notes

- Bedrock streaming needs `bedrock:InvokeModelWithResponseStream` permission and
  a compatible model. Streaming failures are not retried through the batch API.
  Existing private settings remain valid; restart the runtime/API after updating.
- New audio/runtime limits apply through defaults without changing private
  settings. Restart the runtime and API. Slow first-run model downloads may need
  a larger `runtime.startup_timeout_s`; see README for limits and recovery scope.

- Run `uv run arietta-voice config-init` once to preserve existing settings and
  persona in private files, then customize those files. The command never resets
  tracked originals; review the copies before restoring tracked defaults.
- Run `uv run arietta-voice admin-setup` once with the same base `--config` used
  by `serve`. Restart the API after credential changes. Old default credentials
  no longer work; remove obsolete password environment entries to use the hash
  file. Source editing stays disabled unless explicitly enabled in config.
- Local settings, persona copies, credential hashes, migration backups, and
  `.working` config copies under `config/` are now ignored. Custom directories
  require their own ignore rules; already tracked private files remain tracked.

- Existing custom articles remain in place. If an article was previously
  tracked in your checkout, remove that specific path from the Git index with
  `git rm --cached -- path`; this retains its local file, but notifies Git to
  remove it in the next commit. Earlier commits still retain their content.
- Keep private knowledge in new files rather than the two tracked starter files.
  Knowledge directories configured elsewhere need their own protection.
- Run `uv sync --locked` after a version change to refresh installed metadata.
- Rebuild an existing semantic index with `uv run arietta-voice knowledge-index`,
  then restart the runtime/API. Until rebuilt, hybrid retrieval falls back to
  keywords and vector-only retrieval returns no matches.
- New logging defaults apply when settings are absent; no local config edits
  are required. Restart both runtime and API to apply them. Old logs, recordings,
  and memory files are not erased or scrubbed by this change. JSONL rotation
  does not manage launchd stdout/stderr logs. See README's privacy section.

## 0.1.0 baseline

The repository already declared `0.1.0` before this changelog was introduced.
This entry records that baseline, not a newly published or dated release.
It includes local wake-word voice interaction, Moonshine STT, Kokoro TTS,
Silero/Smart Turn endpointing, MLX/Bedrock chat, knowledge retrieval, two tool
paths, optional memory, and the web admin/API with runtime supervision.
