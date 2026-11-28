# Arietta Voice Roadmap

Revised November 27, 2026. This replaces the October plan, which is preserved
with its full status log in
[docs/archive/ROADMAP-2026-10.md](docs/archive/ROADMAP-2026-10.md).

**Goal:** a state-of-the-art personal voice assistant that runs on an Apple
Silicon Mac — one that answers about as fast as a person would, can actually do
things, remembers its owner, and is pleasant to run and configure.

## Release priorities · November 27, 2026

The stable release is `0.3.0`. It
supports timers and reminders, arithmetic, answers from notes, optional weather
and web search, and tools supplied by MCP servers. The console exposes nine
views, including streamed text chat and private persona and memory editors.

1. **Get a new owner to a useful answer.** The README now takes a fresh clone
   through installation, sign-in and the first Chat answer. In a clean setup,
   Chrome at 1440 px and 375 px returned 391 for 17 × 23 with voice stopped
   and model downloads disabled. A first-time user also successfully installed
   the app.
2. **Console Chat cites sources.** In Chrome,
   the local model cited the right passage for a synthetic spare key, then
   cited the corrected location in a fresh chat. After deletion, Chat reported
   that it couldn't find the fact. Old links reported the missing passage.
   The browser suite repeats this workflow with a scripted model.
3. **Learned preferences are editable in the console.** In fresh Chat sessions,
   the local model recalled jasmine tea, used oolong tea after a correction
   on the Persona page, and reported no saved preference after forgetting.
4. **Reminders survive runtime restarts.** A Chrome test sets a synthetic
   reminder in Chat and finds it on Home. With a controlled clock and silent
   output, a fresh voice runtime announces it once when due. After the user
   selects Stop, another fresh runtime stays quiet and Home shows “Stopped.”
5. **Keep the voice qualification results as a baseline.** On November 27
   the 100-turn audible benchmark measured 3.36 s median and 11.35 s p95
   to the first non-silent speaker write, above the Milestone 1 targets.
   The quiet-room Jabra soak completed 206 turns in 30 minutes with three
   speaker underflows and three AEC resets. The first-chunk investigation
   found no benefit from limiting ONNX CPU threads. Capture polling
   around missed writes, acoustic interruption-to-silence timing, and live
   qualification of the opt-in MLX path remain open. Return to the 100-turn
   run for a concrete latency change. Reserve guided Jabra checks for changes
   to audible behavior, a reported regression, or a release decision.
   [Latency measurements](evals/LATENCY_BASELINE.md) ·
   [Soak results](evals/LIVE_LOAD_SOAK.md).

The ONNX Model Zoo ended Git LFS downloads in July 2025 and directs users
to Hugging Face. Arietta gets Kokoro and Smart Turn from their publishers.
On November 27 all three configured model and voice URLs returned HTTP 206
to one-byte requests. The installed Kokoro ONNX engine also synthesized “Your
timer is ready” from the cache (31,744 samples at 24 kHz). Keep the speech-engine
decision based on live performance.
[Model Zoo notice](https://github.com/onnx/models/blob/main/README.md) ·
[Kokoro assets](https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0) ·
[Smart Turn model](https://huggingface.co/pipecat-ai/smart-turn-v3/blob/main/smart-turn-v3.2-cpu.onnx).

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
| 1 Voice and console | `0.3.0` | Source-backed Chat, editable memory and persistent reminders. Voice latency target deferred. |
| 2 Does things | `0.4.0` | Real tools: timers, weather, web, calendar, home control |
| 3 Knows you | `0.5.0` | Durable, inspectable memory and grounded answers |
| 4 Always there | `0.6.0` | Acoustic wake word, one-command install, talk from any device |
| Stable | `1.0.0` | Compatibility promise and a 24-hour soak |

The console track below runs alongside milestones 0–3.

## Milestone 0 — Close out and tidy (`0.2.0`)

1. Park the Kokoro question: ONNX stays the default, MLX stays opt-in, and no
   further comparison runs until Milestone 1 names TTS as the bottleneck.
2. Run one 30-minute soak on the default ONNX stack. If it passes the existing
   audio gate, release. If not, fix that failure only. Deferred by the owner
   on November 14: a TV in the room was heard as requests, which would
   invalidate the run. It moves to Milestone 1, item 5.
3. Tag `0.2.0` from the current unreleased changelog. Done: `0.2.0rc1` to
   `0.2.0rc7` and then `0.2.0` were tagged on November 14.
4. Reorganize the repository (see [Repository layout](#repository-layout)).
5. Add `ruff` formatting and linting to CI and reformat the dense October code.

**Done when:** `0.2.0` is tagged, the root holds only the standard project
files, and CI enforces formatting.

## Milestone 1 — Voice and console (`0.3.0`)

**Deferred performance target on the M4/32 GB reference Mac, default local stack, warm:** end of
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
  mlx-vlm's own prefix cache, so no move to `mlx-lm` was needed. Since the
  timers work it also keeps working past Gemma's 512-token sliding window,
  which longer conversations and the full tool list reach on every turn.
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

- Notes go after the conversation so far and are attached only when close in
  meaning. Unrelated notes had been attached to 17 of 40 general questions on
  the owner's setup; the median there fell from 1.21 s to 0.66 s and the p95
  from 2.89 s to 1.09 s (text only). Questions that do need notes still spend
  about 1.3 s reading three of them (see item 6 below).

In the earlier muted baseline, a turn with ONNX Kokoro spent about 300 ms deciding the
speaker has finished, 180 ms transcribing, 480 ms generating and 490 ms on the
first speech chunk. ONNX needs about 450 ms even for three words, so a shorter
first chunk cannot fix that; MLX produces the first chunk in about 190 ms.

1. **Decide on MLX Kokoro as the default speech engine.** It now installs
   from the main project (`uv sync --group mlx-speech`) and answered in
   1.28 s median against 1.59 s on ONNX, with a p95 of 2.18 s against 3.38 s.
   It adds misaki (which declares support only below Python 3.13, although uv
   installs it and it works), spaCy and an English spaCy model. One earlier
   100-turn muted run had an output underflow; a new one, with streaming
   transcription too, passed every gate at 1,269 ms median and 2,361 ms p95
   (November 14). Gate it on a live soak.
2. **Stream transcription during speech.** Built as an experimental setting,
   off by default (`[audio].streaming_stt`): in the muted benchmark the
   transcription after speech fell from 160 ms to 85 ms at the median and first
   audio from 1.63 s to 1.54 s, with the same transcripts. Still to do before it
   becomes the default: the live checks below. Moonshine's streaming mode leaves
   about 70 ms of work after the turn ends instead of about 170 ms. Getting
   that needs a way to feed audio to the inference worker while the user is
   speaking without blocking the listening loop, since each streaming update
   takes up to about 160 ms. The gates in
   [the Moonshine experiment](docs/archive/experiments/MOONSHINE_EXPERIMENT.md#promotion-gates)
   still apply.
3. **Keep the prompt cache across a discarded attempt** — done. A stopped
   answer (an interruption, a resumed turn, a hand-over to the smarter model)
   had dropped the cache, so the next turn read the whole prompt again: 2.5 s
   to the first clause instead of 0.8 s.
4. **Native tool calling** — done for the local model (Milestone 2, item 2).
   The prompted JSON format was fragile with the 4-bit Gemma; native calls
   chose the right tool in every check.
5. **A live-microphone session** covering the early check, resuming and the
   deferred chime. Everything above was measured with synthetic speech. Run
   the 30-minute audible soak deferred from Milestone 0 in the same quiet room.
6. **Read the question and its notes while the user is still speaking** —
   tried and rejected
   ([result](evals/LATENCY_BASELINE.md#experiment-reading-the-question-while-it-is-spoken)).
   A question missing its last two words found the same notes as the full one
   for 3 of 8 made-up questions and none of the owner's, so the early reading
   rarely carried over. Fewer tokens per note: the labels are done (a quarter
   fewer tokens, about 125 ms), and long sections now come in chunks, so the
   part that matches is sent rather than a section's first 420 characters
   (Milestone 3, item 3).

`0.3.0rc1` (November 14) carries the work above for the owner's live checks.

Voice timing work is deferred beyond `0.3.0`. When it resumes, use the
standing 100-turn benchmark, check for false cutoffs, and measure acoustic
interruption latency. The November 27 measurements above remain the baseline.

## Milestone 2 — Does things (`0.4.0`)

1. **One tool interface.** Done: each tool has a name, description,
   JSON-schema arguments, optional triggers for instant answers, a time limit
   and a structured result, and user tools live in `tools/`. Unknown, invalid,
   failing and slow tools are reported to the model instead of failing the turn.
2. **Agent loop.** Native tool calls and up to three steps per turn are done
   (the local model uses Gemma 4's own tool-call format; Bedrock keeps the
   prompted JSON). In a real-model check every tool request was routed to the
   right tool, including phrasings the prompted format missed. A slow tool
   gets a spoken "One moment." after 0.8 seconds. Confirmation before
   consequential actions is done: the call is held and the assistant asks
   "Should I …?"; only a clear spoken yes runs it, once, and each server asks
   every time, once per action or never. Still to do: native calls for
   Bedrock's Converse API.
3. **MCP client.** Done: servers in `config/mcp.local.json` (the common
   `mcpServers` format) over stdio, Streamable HTTP or SSE, kept connected and
   reloaded on change, set up and tested from the console's Connections page.
   In a real-model check with a test lights server next to the built-in tools,
   20 of 20 requests went the right way: actions asked first, status questions
   ran directly, and instant answers were unaffected. Still to do: OAuth sign-in
   for remote servers, and checking against a real Home Assistant and calendar
   server.
4. **Built-in skills.** Timers, reminders and alarms are done: instant answers
   for the usual phrasings, model calls for the rest, a shared store that
   survives restarts, spoken alerts between turns that repeat until answered,
   "stop" and "snooze" without the wake word, and a console card. Arithmetic
   and unit conversion are done (`calculate`; the 4-bit Gemma had said
   17 × 23 is 749), and so are date questions (`date_math`: "how many days
   until Christmas?"), and weather is available as an opt-in tool (Open-Meteo,
   checked against the live service on November 14). Web search with spoken
   source attribution is done and opt-in: Wikipedia (no account), Brave Search
   or SearXNG; in a real-model check, 16 of 16 requests were handled as
   expected, with answers such as "According to Wikipedia, ...".
5. **Home control** through Home Assistant, now through its own MCP server
   (`/api/mcp`) rather than a custom adapter: read-only status first, then
   explicitly authorized actions.
6. **Model routing.** Keep the small local model for fast turns and tool
   selection; optionally escalate hard questions to a cloud model. Off by
   default, with a clear statement of what leaves the machine. Refresh the
   stale Bedrock model examples. The 4-bit Gemma declining ordinary questions
   ("I don't have that information") turned out not to be the model: knowledge
   notes and the persona told it to, and prompt reuse corrupted its input.
   With those fixed it declined 2 of 80 and got 48 of 48 facts right
   ([answer quality](evals/ANSWER_QUALITY.md)). Optional escalation is done:
   `[smarter_model]`, off by default, hands a question to a bigger Ollama or
   Bedrock model when the local one says it doesn't know, or every answer
   that needs no tool; Ollama also works as the main provider on bigger Macs.
7. Treat retrieved text and tool output as untrusted data, with
   prompt-injection fixtures in the evaluation corpus. Done: MCP results and
   errors and web results reach the model marked as data, the model is told
   never to follow instructions in tool results, and an action needs the
   owner's spoken yes however the model was led to it. Five injected payloads
   are checked in CI (`evals/test_injection.py`), and with injected notes the
   real model followed none of them (20 of 20 answers right, no tool called).

**Done when:** "set a timer for ten minutes", "what's the weather tomorrow",
"what's on my calendar" and "turn off the office lights" all work by voice;
disabled tools never execute; an interrupted action never runs twice.

## Milestone 3 — Knows you (`0.5.0`)

1. Replace the single generated `MEMORY.md` with structured memory: a profile
   of durable facts plus dated episodic notes, each with provenance, all
   viewable, editable and deletable in the console. The existing Markdown store
   now exposes individual facts, their learning channel, and owner corrections.
   Dated episodic notes remain open.
2. Persist conversation history across sessions and make it searchable.
3. Knowledge retrieval: bounded chunks with stable source IDs, atomic index
   generations, and a 50-question synthetic recall set. Target recall@3 of at
   least 90% on answerable questions, with an explicit "I don't know" on
   unanswerable ones. Bounded chunks with stable IDs are done: long sections
   are split at sentence and list boundaries, so facts past the first 420
   characters can be found (on the owner's notes, questions about them went
   from 10 to 16 of 20 right). The recall set has 14 answerable questions so
   far, checked in CI with keyword search; atomic index generations are still
   to do.
4. Console Chat now links cited answers to the exact passage in the current note.

**Done when:** the assistant recalls a fact stated in a previous session, the
owner can see and remove it, and the retrieval target is met.

## Milestone 4 — Always there (`0.6.0`)

1. Acoustic wake word behind the existing detector seam, with a custom-trained
   phrase and a measured false-activation rate. Transcript matching stays
   available as the fallback. This also stops transcribing all room speech.
   Until then, after a wake any speech counts as a request until 20 s pass
   without any, so a TV in the room can be answered and keep the session open
   (a known issue in `0.2.0`). Interim: `[wake_word].follow_ups = false`
   requires the wake word for every request except an answer to the
   assistant's own question.
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
- Chat answers stream clause by clause (server-sent events), and Home has a
  per-turn latency bar from the Milestone 1 stage timings.

- Push status updates over server-sent events instead of polling — done for
  the runtime status on Home.
- Browser smoke tests in CI — done: every page is opened in Chrome through
  Playwright against a scratch console and must load with no script error,
  failed request or error banner; navigation and the theme switch are
  exercised too.

**C2 — with Milestones 2–3**

- **Settings** as forms with validation — done for the common settings
  (devices, models and paths are still file-only).
- **Persona and memory** editors — done (the Persona page).
- **Tools and integrations**: enable, configure and test tools and MCP servers —
  done (the Tools and Connections pages).
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
