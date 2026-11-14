# Arietta Voice Roadmap

Revised November 13, 2026. This replaces the October plan, which is preserved
with its full status log in
[docs/archive/ROADMAP-2026-10.md](docs/archive/ROADMAP-2026-10.md).

**Goal:** a state-of-the-art personal voice assistant that runs on an Apple
Silicon Mac — one that answers about as fast as a person would, can actually do
things, remembers its owner, and is pleasant to run and configure.

## Where the project stands

The October work built a strong foundation and then stalled inside it.

**Solid and worth keeping**

- Privacy by default: no passive transcript logging, minimal public health
  endpoint, private config/persona overrides, admin credentials required.
- Reliability: bounded audio buffers, a killable serialized inference worker,
  per-turn deadlines and cancellation, device reconnect, background memory jobs.
- Streaming plumbing: provider token streaming into clause-level Kokoro
  synthesis, AEC-based barge-in, pre-roll capture.
- 605 passing tests, Apple Silicon CI, a synthetic evaluation corpus and a
  repeatable latency benchmark.

**Where it went off course**

- **Latency did not move.** The baseline was 2,362 ms median from end of speech
  to first audio. After streaming, cancellation and the Kokoro comparisons, the
  closest later measurement for the default stack is 2,447 ms. The old Phase 2
  target was a 30% reduction. (Protocols differ slightly; the conclusion does
  not.)
- **Streaming is bypassed in the default configuration.** When model tools are
  enabled — the default — every turn first runs a complete, unstreamed planning
  generation, and a direct answer from that call is spoken only after it
  finishes.
- **Effort went to measurement instead of outcomes.** Nine commits and nine
  reports went to comparing ONNX and MLX Kokoro and chasing the audio faults
  that surfaced, for a 3% end-to-end gain. The last 30-minute soak on the
  opt-in MLX path failed its audio gate.
- **The assistant cannot do much.** Its tools are local time, its own identity,
  Markdown knowledge search and system info. The first real integration was
  scheduled for the fourth of five phases.
- **Nothing shipped.** The version is still an untagged `0.1.0` with a large
  unreleased changelog.

## How this roadmap works

1. Every milestone ends in something the owner can hear or see, with one
   headline number or demo.
2. Experiments are time-boxed to one working day with a decision rule written
   down first. No result by then means keep the current component and move on.
3. This file stays short. Status goes in `CHANGELOG.md`; measurements go in
   `evals/`.
4. One standing benchmark (`latency_baseline`) is rerun after each pipeline
   change. New harnesses need a reason the standing one cannot serve.
5. Tag a release at the end of every milestone.

| Milestone | Version | Outcome |
| --- | --- | --- |
| 0 Close out | `0.2.0` | October work released; repository tidy; experiments parked |
| 1 Feels instant | `0.3.0` | About one second from end of speech to first audio |
| 2 Does things | `0.4.0` | Real tools: timers, weather, web, calendar, home control |
| 3 Knows you | `0.5.0` | Durable, inspectable memory and grounded answers |
| 4 Always there | `0.6.0` | Acoustic wake word, one-command install, talk from any device |
| Stable | `1.0.0` | Compatibility promise and a 24-hour soak |

The console track below runs alongside milestones 0–3.

## Milestone 0 — Close out and tidy (`0.2.0`)

1. Park the Kokoro question: ONNX stays the default, MLX stays opt-in, and no
   further comparison runs until Milestone 1 names TTS as the bottleneck.
2. Run one 30-minute soak on the default ONNX stack. If it passes the existing
   audio gate, release. If not, fix that failure only.
3. Tag `0.2.0` from the current unreleased changelog.
4. Reorganize the repository (see [Repository layout](#repository-layout)).
5. Add `ruff` formatting and linting to CI and reformat the dense October code.

**Done when:** `0.2.0` is tagged, the root holds only the standard project
files, and CI enforces formatting.

## Milestone 1 — Feels instant (`0.3.0`)

**Target on the M4/32 GB reference Mac, default local stack, warm:** end of
speech to first audio at or below **1.0 s median and 1.6 s p95**; barge-in to
silence at or below 300 ms p95.

Measured so far (30 warm turns, muted output; see
[the latency report](evals/LATENCY_BASELINE.md#november-13-2026-first-milestone-1-changes)):

| Configuration | At the start | Now |
| --- | ---: | ---: |
| No tools on offer | 2.48 s | 1.71 s |
| A model tool on offer (the default setup) | 3.09 s | 1.72 s |
| A model tool on offer, MLX Kokoro trial with trimmed silence | — | 1.34 s |

Done:

- Direct answers stream past the tool-planning step instead of waiting for a
  separate generation.
- The local model reuses the already-processed start of each prompt. This used
  mlx-vlm's own prefix cache, so no move to `mlx-lm` was needed.
- The first spoken clause can end at a comma or dash.
- Each spoken turn logs a stage timeline, shown on the console's Home page.
- The end of a turn is checked after 250 ms of silence, and work starts at once
  on a confident answer. The turn stays provisional until the normal silence
  window has passed and is discarded if the speaker carries on.

- Silence around each synthesised clip is trimmed. It matters only for MLX
  Kokoro, which pads every clip with about 0.3 s before and 0.4 s after.

- All dependencies are on their latest releases, including Moonshine 0.1.6,
  which now transcribes each utterance in its own stream. That removed the
  stray leading word the old recogniser sometimes added and cut transcription
  to about 180 ms.

Remaining, in order. With ONNX Kokoro a turn spends about 300 ms deciding the
speaker has finished, 180 ms transcribing, 480 ms generating and 490 ms on the
first speech chunk. ONNX needs about 450 ms even for three words, so a shorter
first chunk cannot fix that; MLX produces the first chunk in about 190 ms.

1. **Decide on MLX Kokoro as the default speech engine.** It now installs
   from the main project (`uv sync --group mlx-speech`) and answered in
   1.28 s median against 1.59 s on ONNX, with a p95 of 2.18 s against 3.38 s.
   It adds misaki (which declares support only below Python 3.13, although uv
   installs it and it works), spaCy and an English spaCy model. One earlier
   100-turn muted run had an output underflow. Gate it on a live soak.
2. **Stream transcription during speech.** Moonshine's streaming mode leaves
   about 70 ms of work after the turn ends instead of about 170 ms. Getting
   that needs a way to feed audio to the inference worker while the user is
   speaking without blocking the listening loop, since each streaming update
   takes up to about 160 ms. The gates in
   [the Moonshine experiment](docs/archive/experiments/MOONSHINE_EXPERIMENT.md#promotion-gates)
   still apply.
3. **Keep the prompt cache across a discarded attempt.** Only matters when the
   speaker resumes after answer generation has begun; resuming during
   transcription, the common case in the benchmark, keeps the cache.
4. **Native tool calling** where the provider supports it (shared with
   Milestone 2's tool interface).
5. **A live-microphone session** covering the early check, resuming and the
   deferred chime. Everything above was measured with synthetic speech.

**Done when:** the standing benchmark meets the target over 100 warm turns with
no increase in false cutoffs, and interruption latency has been measured
acoustically at least once.

## Milestone 2 — Does things (`0.4.0`)

1. **One tool interface.** Merge the deterministic and model-requested tool
   systems: each tool has a name, description, JSON-schema arguments, an
   optional fast-path trigger, a deadline and a structured result. Move user
   tools out of the installed package into a user directory.
2. **Agent loop.** Native tool calls, up to three steps per turn, a spoken
   filler when a tool is slow, and confirmation before consequential actions.
3. **MCP client.** Connect to Model Context Protocol servers so calendar,
   notes, home automation and similar integrations come from the existing
   ecosystem rather than hand-written adapters.
4. **Built-in skills:** timers, alarms and reminders (requires a scheduler and
   the ability to speak unprompted), weather, web search with spoken source
   attribution, and unit/date arithmetic.
5. **Home control** through Home Assistant: read-only status first, then
   explicitly authorized actions.
6. **Model routing.** Keep the small local model for fast turns and tool
   selection; optionally escalate hard questions to a cloud model. Off by
   default, with a clear statement of what leaves the machine. Refresh the
   stale Bedrock model examples.
7. Treat retrieved text and tool output as untrusted data, with
   prompt-injection fixtures in the evaluation corpus.

**Done when:** "set a timer for ten minutes", "what's the weather tomorrow",
"what's on my calendar" and "turn off the office lights" all work by voice;
disabled tools never execute; an interrupted action never runs twice.

## Milestone 3 — Knows you (`0.5.0`)

1. Replace the single generated `MEMORY.md` with structured memory: a profile
   of durable facts plus dated episodic notes, each with provenance, all
   viewable, editable and deletable in the console.
2. Persist conversation history across sessions and make it searchable.
3. Knowledge retrieval: bounded chunks with stable source IDs, atomic index
   generations, and a 50-question synthetic recall set. Target recall@3 of at
   least 90% on answerable questions, with an explicit "I don't know" on
   unanswerable ones.
4. Return source references with answers.

**Done when:** the assistant recalls a fact stated in a previous session, the
owner can see and remove it, and the retrieval target is met.

## Milestone 4 — Always there (`0.6.0`)

1. Acoustic wake word behind the existing detector seam, with a custom-trained
   phrase and a measured false-activation rate. Transcript matching stays
   available as the fallback. This also stops transcribing all room speech.
2. `arietta-voice init`, packaged starter assets, a generated LaunchAgent, and
   a tested upgrade and rollback path.
3. A browser voice client (WebRTC) so the assistant can be used from the
   console on a phone or another computer.
4. Eight-hour soak with no crash, lost device or unbounded memory growth.

Full-duplex speech-to-speech models and cloud realtime voice sessions stay
optional experiments. They do not yet combine tool use, grounding and local
operation well enough to replace the pipeline.

## Console track

**C1 — with Milestones 0–1**

Done:

- The frontend is static files served from the package (`web/static/`, no
  build step), and `api.py` is split into app, service and schema modules
  under `web/`. A strict content security policy forbids inline script.
- One design system in the brand's violet: shared tokens for type, spacing
  and colour, a single primary action per view, and light and dark themes.
- **Home** shows the assistant's state, runtime controls, turn progress, the
  current exchange and recent activity. Debug counters moved to **System**.
- **Chat** is a conversation view with per-answer tool and timing detail.
- **Activity** describes events in plain language; **Settings** shows the
  effective configuration by section instead of a JSON dump.

Still to do:

- Push updates over server-sent events instead of polling.
- Stream chat responses token by token.
- A per-turn latency timeline on Home (needs the Milestone 1 stage timings).
- Browser smoke tests in CI.

**C2 — with Milestones 2–3**

- **Settings** as forms with validation, replacing the JSON dump.
- **Persona and memory** editors.
- **Tools and integrations**: enable, configure and test tools and MCP servers.
- **Conversations**: searchable history.
- **Diagnostics**: logs, health checks and the audio counters, moved off Home.

## Repository layout

Do this in Milestone 0, in behaviour-preserving commits.

- `docs/`: architecture, getting started, releasing, the October review, and
  the archived roadmap. `assets/`: logo and screenshots. The root keeps
  `README.md`, `ROADMAP.md`, `CHANGELOG.md`, `LICENSE` and build files.
- `config/`: one tracked template instead of three overlapping ones, kept in
  step with the code's defaults.
- `bench/`: the standing kit — latency baseline, hardware validation and
  live soak — with their tests. Concluded experiment harnesses (Kokoro,
  Moonshine, ONNX environment, underflow probes) and their reports move to
  `docs/archive/experiments/` and leave the main test run.
- `src/arietta_voice/`: group the flat modules into `audio/`, `speech/`,
  `brain/`, `runtime/`, `tools/` and `web/` subpackages as each area is
  touched, starting with `web/` for the console work.
- Remove dead code such as the unused image-input path in `llm.py`.
- Delete the 3 GB of ignored experiment environments under `tmp/` once the
  experiments are archived.

## Stable (`1.0.0`)

Ship when the CLI, configuration schema, HTTP API and tool interface have a
written compatibility policy, migrations and rollback work, and a 24-hour soak
with reconnect and restart scenarios passes.
