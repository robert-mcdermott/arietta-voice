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

The baseline breaks down as roughly 715 ms waiting to decide the user finished,
180 ms batch transcription, 690 ms full chat generation and 540 ms to the first
TTS chunk. In order of expected gain:

1. **Remove the buffered planning call.** Stream every generation; hold back
   only output that begins as a tool call. Use native tool calling where the
   provider supports it.
2. **Endpoint adaptively.** Run Smart Turn after about 200 ms of silence and
   respond immediately on a confident end of turn; keep the long silence window
   only as a fallback. Measure false cutoffs on the existing fixtures.
3. **Cache the prompt prefix.** Persona, memory and tool instructions are
   re-processed on every turn. Move text chat to `mlx-lm` with a cached system
   prefix (one-day experiment: confirm the Gemma build loads and first-token
   time drops).
4. **Stream transcription.** Promote the Moonshine streaming path so the
   transcript is ready when the turn ends, and begin generation speculatively
   on a stable partial.
5. **Shorten the first spoken clause** so synthesis starts on a few words
   rather than a full sentence.
6. Revisit MLX Kokoro only if TTS is the largest remaining stage after 1–5.

Add a per-turn stage timeline to the event log so the console can show where
each turn's time went.

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

The console today is a working admin tool with a generic look: equal-weight
coloured buttons, raw JSON panes, a read-only configuration dump, a dashboard
of debug counters, and 2,100 lines of HTML, CSS and JavaScript inside a Python
string in `api.py`.

**C1 — with Milestones 0–1**

- Extract the frontend into static files served from the package; split
  `api.py` into app, routes and service modules. No build step required.
- One design system: typography, spacing, a restrained palette, a single
  primary action per view, and proper light and dark themes.
- **Home** becomes a live view of the assistant: current state, the
  conversation as it happens, and a per-turn latency timeline, pushed over
  server-sent events instead of polling.
- **Chat** streams responses and shows tool activity inline.
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
