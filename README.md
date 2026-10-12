# Arietta Voice

![Arietta Logo](arietta_logo.png)

Arietta Voice is a local-first framework for building a wake-word-driven voice
assistant on Apple Silicon Macs. It combines local speech-to-text,
text-to-speech, turn detection, tool routing, markdown knowledge retrieval, and
a web admin console into a customizable assistant runtime. It's like your own private,
custom, locally host Alexa/Echo, but better.

The project is meant to become *your* assistant, not a finished general-purpose
assistant out of the box. As shipped, it is mostly a working runtime and admin
surface; it becomes useful when you provide the assistant persona in `SOUL.md`,
add knowledge articles for your environment, configure the wake word, and add
deterministic or model-called tools for the things you want it to do.

The default assistant persona is named `Bridget`, and the default wake word is
also `Bridget`. Both are easy to change in your private
`config/arietta_voice.local.toml` after running `uv run arietta-voice config-init`.

For a detailed customization walkthrough, see [GETTING_STARTED.md](./GETTING_STARTED.md).

For planned improvements, see [ROADMAP.md](./ROADMAP.md). The
[release guide](./RELEASING.md) explains app versioning, and
[CHANGELOG.md](./CHANGELOG.md) records user-visible changes.

## What This Project Includes

- local speech-to-text with Moonshine
- local text-to-speech with Kokoro
- local VAD and turn-end detection with Silero VAD and Smart Turn
- default local chat with Gemma on MLX
- optional AWS Bedrock chat integration through the same chat-model seam
- deterministic tool routing
- model-requested follow-up tools
- local markdown knowledge retrieval
- editable `SOUL.md` and optional memory file
- a wake-word-first runtime designed for an always-on Mac mini
- authenticated HTTP API and admin console for inspection, chat testing, logs, diagnostics, tools, and knowledge

## Design Scope

The project is intentionally focused on a voice-first assistant runtime with a small, editable surface area:

- the typed config system
- the knowledge indexing and retrieval pipeline
- deterministic tools and model-requested tools
- the local/Bedrock LLM adapter seam
- STT, TTS, VAD, Smart Turn, and audio barge-in handling
- prompt assembly and optional memory consolidation
- a local admin/API surface for operational visibility and configuration workflows

It intentionally does not include:

- webcam presence detection
- camera and vision tools

## Setup

Install the project environment:

```bash
uv sync
```

Recommended on macOS for Kokoro:

```bash
brew install espeak-ng
```

First run downloads model assets as needed. Depending on your config, that can include Moonshine, Kokoro, Smart Turn, and the local MLX chat model.

## Commands

Run the assistant:

```bash
uv run arietta-voice
uv run arietta-voice run
```

Useful variations:

```bash
uv run arietta-voice --version
uv run arietta-voice --no-wake
uv run arietta-voice --wake-word Bridget
uv run arietta-voice --wake-word Bridget --wake-word Arietta
uv run arietta-voice --session-timeout 30
uv run arietta-voice --voice bf_emma
uv run arietta-voice --chat-provider bedrock --model us.anthropic.claude-sonnet-4-20250514-v1:0
uv run arietta-voice --record
uv run arietta-voice --record tmp/session.wav
```

Inspect configuration:

```bash
uv run arietta-voice config
```

Run diagnostics:

```bash
uv run arietta-voice doctor
```

Set up admin credentials once, then run the HTTP API and admin console:

```bash
uv run arietta-voice admin-setup
uv run arietta-voice serve
```

The admin console is available at `http://127.0.0.1:8765/admin`.
There is no default password. `admin-setup` prompts without echoing your password
and stores a salted PBKDF2-SHA256 hash in ignored, owner-only
`config/arietta_voice.admin.json`. The username defaults to `admin`; use
`--username` to change it and `--replace` to rotate credentials. Passwords must
contain at least 12 printable ASCII characters. Restart the API after rotation.

Alternatively, supply `ARIETTA_ADMIN_PASSWORD` and optionally
`ARIETTA_ADMIN_USERNAME` through your environment. An explicitly set environment
password takes precedence over the credential file and must meet the same
requirements. The server refuses to start without valid credentials.

Keep the default loopback binding. For remote use, put the service behind HTTPS
and list exact hostnames in comma-separated `ARIETTA_ADMIN_ALLOWED_HOSTS`;
HTTP Basic credentials are not encrypted by HTTP itself. The API rejects
untrusted Host headers and cross-origin browser mutations, and marks responses
non-cacheable. These controls do not provide TLS or a multi-user permission model.

**Screenshot of the Arietta admin console's tool managment panel:**
![arietta console](arietta_console.png)


The voice runtime and web console can run as separate processes. For example,
you can start the voice runtime in one terminal and the admin console in
another:

```bash
uv run arietta-voice
uv run arietta-voice serve
```

In this mode, the admin console still provides observability and editing:
health, status heartbeat, logs, diagnostics, chat testing, knowledge editing,
tool validation, opt-in source editing, and private-config-backed tool enablement. The
voice runtime publishes a heartbeat to `logs/runtime_status.json`, which lets
the admin console show whether the separate voice process is running, stopped,
or stale.

The main limitation is process ownership. If the voice runtime was started
manually with `uv run arietta-voice`, the admin console can observe it but
cannot stop or restart that terminal process. After changing tools, knowledge
settings, or config used by the voice runtime, restart the manually started
runtime yourself so it picks up those changes.

The admin console can also start, stop, and restart an API-managed voice
runtime. Use the admin console's Start button when you want the web API to own
the child voice process and make the Start, Stop, and Restart buttons fully
effective. Managed runtime stdout and stderr are written to
`logs/managed_runtime.stdout.log` and `logs/managed_runtime.stderr.log`.

If you use a custom config path, pass the same config to both processes:

```bash
uv run arietta-voice --config config/arietta_voice.toml
uv run arietta-voice serve --config config/arietta_voice.toml
```

For a headless Mac mini or other always-on Mac, use macOS `launchd` to start
the voice runtime and admin console automatically after reboot/login. Because
the voice runtime uses microphone/audio access, run it as a user LaunchAgent
rather than a root LaunchDaemon. See [Run Automatically on macOS With
launchd](./GETTING_STARTED.md#9-run-automatically-on-macos-with-launchd) for a
step-by-step setup.

The Knowledge page can list, create, edit, delete, search, and re-index local
knowledge files inside the configured knowledge directory.

List audio devices:

```bash
uv run arietta-voice devices
```

Work with the knowledge base:

```bash
uv run arietta-voice knowledge-search "what can you help with"
uv run arietta-voice knowledge-index
```

Run the isolated unit tests and synthetic evaluations:

```bash
uv sync --locked --group dev
uv run --locked pytest
```

Tests use temporary config/knowledge and fake models/audio; they preserve local
settings and require no credentials. CI adds tracked-private-file, locked-install,
version, and wheel/sdist checks on Apple Silicon. See
[synthetic evaluations and CI](evals/README.md) for coverage, individual checks,
and the remaining hardware/model validation gates.

For actual microphone/speaker checks and a bounded local soak, follow
[hardware validation](evals/HARDWARE_VALIDATION.md). The opt-in harness uses an
isolated synthetic persona and preserves private settings and knowledge.

## Wake Word Behavior

The initial wake-word implementation is local and transcript-based:

- Arietta Voice listens continuously with VAD.
- When speech ends, it transcribes the utterance locally.
- If the transcript starts with a configured wake phrase like `Bridget`, the assistant opens a short active session.
- If the wake utterance also contains a command, like `Bridget what time is it`, the assistant handles that in the same turn.
- If you only say `Bridget`, the assistant answers with the configured acknowledgement and keeps listening for the follow-up.

This approach keeps the system simple and fully local while reusing the existing STT/VAD pipeline. It is also deliberately extensible: the wake-word logic lives in [`src/arietta_voice/wake.py`](./src/arietta_voice/wake.py), so a future acoustic hotword backend can be added without rewriting the runtime.

## Configuration

The tracked [`config/arietta_voice.toml`](./config/arietta_voice.toml) provides
shared defaults. Create private settings and a private persona before customizing:

```bash
uv run arietta-voice config-init
```

This snapshots your effective settings into `config/arietta_voice.local.toml`,
copies the current persona to `config/SOUL.local.md`, and points the private
config at that copy. It keeps the originals and stores backups in
`config/.private-backups/`. Existing private files are never overwritten.
All generated private files are owner-only and ignored by Git. If you already
edited the tracked files, review the private copies before restoring the tracked
files to their committed defaults. Ignore rules do not protect tracked files or
remove content from earlier commits.

Precedence is built-in defaults, selected base TOML, its sibling `.local.toml`,
then CLI runtime overrides. Sections merge by setting; arrays replace the entire
base array, including `[]`. With `--config /path/custom.toml`, the sibling is
`/path/custom.local.toml`; unrelated project overrides are not loaded. Relative
paths resolve beside the selected config. Custom locations outside `config/`
need their own ignore rules. A full snapshot keeps existing values pinned;
delete a private setting to inherit its shared default again.

The most important editable files are:

- `config/arietta_voice.local.toml`: private configuration overrides
- `config/SOUL.local.md`: private system prompt/personality
- `config/MEMORY.md`: optional long-lived memory file, created automatically when memory is enabled
- [`knowledge`](./knowledge): your editable local knowledge articles

The main config sections are:

- `[assistant]`: persona name, identity text, short greetings, goodbye phrases, history length
- `[wake_word]`: wake phrases, acknowledgement, backend, and session timeout
- `[audio]`: TTS, Smart Turn, AEC, chime, and Kokoro settings
- `[models]`: local vs Bedrock chat provider, model ids, AWS settings, STT language, generation settings
- `[prompts]`: `SOUL.md` and memory file locations
- `[memory]`: optional memory behavior
- `[logging]`: runtime log locations, content opt-in, and event-log rotation
- `[knowledge]`: knowledge directory, index directory, backend, and retrieval scoring knobs
- `[tools]`: deterministic tools checked before the model
- `[model_tools]`: model-requested follow-up tools
- `[admin]`: source-editing permission (disabled by default)
- `[runtime]`: turn, endpoint detection, and model startup time limits

Relative paths in the TOML are resolved relative to the config file, not the current shell directory.

`max_history_turns` counts complete user/assistant pairs; `0` disables conversation
history. Invalid TOML types, unknown settings, and unsupported values fail at
startup with the setting name. The current audio pipeline requires 16,000 Hz
input and 512-sample chunks.

### Audio limits and failure recovery

These defaults are included in all three tracked TOML templates. Override only
the settings you want in your private `.local.toml`; restart the voice runtime
and API to apply them.

| Setting | Default | Behavior |
| --- | --- | --- |
| `audio.queue_max_chunks` | `64` | About 2 seconds of queued microphone input; drops oldest chunks under load. |
| `audio.max_utterance_s` | `30.0` | Maximum captured utterance, including trailing silence. |
| `audio.recording_queue_max_chunks` | `256` | Bounded queue for the background WAV writer. |
| `audio.max_recording_s` | `3600.0` | Maximum saved audio per `--record` run. Listening continues after the limit. |
| `audio.playback_reference_s` | `4.0` | Rolling echo-cancellation reference, rather than the entire response. |
| `runtime.turn_timeout_s` | `60.0` | Shared budget for STT, chat/tools, speech generation/playback, and optional memory. Also limits web chat. |
| `runtime.endpoint_timeout_s` | `5.0` | Separate Smart Turn deadline for each endpoint check. |
| `runtime.startup_timeout_s` | `300.0` | Model-worker warmup deadline; increase for slow first-run downloads. |

Oversized or interrupted input is discarded until a quiet interval; it is not
submitted as a truncated command. Recordings stream to an owner-only WAV file
instead of accumulating in RAM. A slow disk can drop recording chunks; the
dashboard reports dropped chunks, recording limits, and write failures. Such
recordings may be incomplete. A forced process kill can leave an unfinished WAV.

Heavy STT, endpoint, chat, tool, and TTS work runs in a persistent spawned process
with one operation at a time. On timeout, crash, or model/tool exception, that
worker is terminated before another is created. The voice runtime clears failed
turn state and resumes listening; successful history remains. A failed memory
update does not undo an already delivered answer. Memory writes replace the old
file atomically, and memory extraction is skipped when speech interrupts an answer.

Tools are **never automatically retried**. A timed-out external action may already
have happened, so check its result before repeating it. Web chat returns `503`
when busy, `504` on timeout, or `502` on inference failure, with no failed history
entry. Voice and web chat each have a separate worker; they do not yet share a
single GPU scheduler. The timeout bounds inference work, not a stuck native audio
driver or arbitrary subprocesses started by a custom plugin. Admin tool-validation,
probe/invoke, and knowledge-index maintenance actions are outside the chat deadline.

The admin dashboard shows the processing stage, turn budget, microphone queue,
recovery counters, and recording warnings while a background heartbeat continues
during model work. Technical process/file details are expandable. Chat prevents
duplicate submissions, retains failed drafts, and shows actionable errors. Source
and knowledge editors warn before leaving with unsaved changes. Larger frontend
restructuring and polling improvements remain in Phase 4 of the roadmap.

### Privacy and logging

Public `/health` returns only liveness and the app version. Detailed diagnostics
and runtime state remain at authenticated `/api/health`.

Conversation text is omitted from event logs, process logs, and the runtime
status file by default. Passive speech that does not activate the assistant is
never logged as text, even with content logging enabled. Active conversation
text can be enabled for debugging in the existing `[logging]` section:

```toml
content_enabled = false
max_bytes = 5000000
backup_count = 3
```

Set `content_enabled = true` only when you want active speech, responses, and
model-tool details in logs, plus active conversation text in runtime status.
The runtime monitor's last-user/last-assistant fields are otherwise blank.
Interactive text-only responses remain visible in a terminal. Redirected
runtime responses require content opt-in.
Uncaught voice-runtime failures report the exception type and exit with a
nonzero status; full exception details require the same content opt-in.

The JSONL event log rotates at 5 MB by default and retains three backups
(`events.jsonl.1` through `.3`). `backup_count = 0` keeps no archives. Event logs
and newly written status files use owner-only permissions. Rotation does not
cover launchd's stdout/stderr files or remove content already written by earlier
versions. Review old logs and recordings separately if you need to remove them.

These settings control diagnostic persistence. Optional memory still saves
facts when enabled, and `--record` explicitly saves microphone audio, including
passive audio. Bedrock sends assembled text prompts—including relevant history,
persona, memory, knowledge snippets, and tool context—to AWS; speech recognition
and synthesis remain local. Custom plugins may have their own logging or network
behavior. Prefer the AWS credential chain over putting AWS credentials in TOML;
if used, keep them only in private config. Admin/CLI config displays redact AWS
credentials, and admin password hashes are kept separately from app config.

### Admin tool permissions

Inventory and source views inspect syntax and literal metadata without importing
plugin code. They report **not validated**, not runtime readiness. Explicit
Validate, Probe/Invoke, Save Enabled, and Preview Instructions actions may import
and execute existing plugins. Instruction preview uses
`POST /api/model-tools/instructions`.

Creating tools and saving Python source are disabled by default, including at
the HTTP endpoints. To enable them deliberately, add this to private config and
restart the API:

```toml
[admin]
allow_source_editing = true
```

Saving source can execute code with the API process's privileges. Tool enablement
changes are written atomically to private config; they preserve unrelated settings
and never modify the tracked base. Restart an already-running voice process to
apply changes there.

## Customizing The Assistant

### Change the assistant name and wake word

Edit `config/arietta_voice.local.toml`:

```toml
[assistant]
name = "Arietta"

[wake_word]
phrases = ["arietta"]
```

### Change the system prompt

Edit `config/SOUL.local.md`. Keep it focused and durable. Project-specific facts should usually live in `knowledge/*.md`, not in the soul file.

### Add knowledge

Add markdown files to [`knowledge`](./knowledge). Copy the template at
[`examples/knowledge-template.md`](./examples/knowledge-template.md) into a new
file there. Only `knowledge/README.md` and `knowledge/getting-started.md` are
tracked by default; other knowledge files and subdirectories are ignored.
Keep personal content in new files, since edits to the two starter documents
remain tracked. See the [knowledge README](./knowledge/README.md) for existing
tracked files and custom knowledge directory locations.

### Add a deterministic tool

Copy [`src/arietta_voice/tools/tool_template.py`](./src/arietta_voice/tools/tool_template.py) to a new module in the same directory, implement `maybe_handle(...)`, then add the module name to `[tools].enabled`.

Use deterministic tools when:

- the answer should be exact
- routing should be explicit
- the tool itself should produce the final answer

The built-in `local_time` tool is the canonical example.

### Add a model-requested tool

Copy [`src/arietta_voice/model_tools/tool_template.py`](./src/arietta_voice/model_tools/tool_template.py) to a new module in the same directory, implement `invoke(...)`, then add the module name to `[model_tools].enabled`.

Use model-requested tools when:

- the model should decide if the lookup is necessary
- the tool gathers supporting facts instead of speaking directly
- you want one grounded follow-up answer after tool execution

This is the natural path for future home-automation actions and richer environment lookups.

## Bedrock Integration

Local chat is the default:

```toml
[models]
chat_provider = "local"
chat_model = "mlx-community/gemma-4-E4B-it-4bit"
```

To switch chat turns to Bedrock:

```toml
[models]
chat_provider = "bedrock"
chat_model = "us.anthropic.claude-sonnet-4-20250514-v1:0"
bedrock_region = "us-west-2"
bedrock_profile = "default"
```

The rest of the runtime stays the same. STT, TTS, tools, wake word handling, and knowledge retrieval remain local.

## Recommended Dedicated Mac Mini Setup

For an always-on voice assistant box, the intended target is a dedicated Apple Silicon Mac mini with:

- a reliable USB microphone or speakerphone
- a good near-field speaker
- a quiet location with stable power
- local knowledge kept in ignored files, with private backups managed separately

The current repo is a strong foundation for a home assistant or office assistant. Home automation should be added through tools, not by hardcoding behaviors into the runtime.

## Project Layout

```text
.
├── config/                     # Configuration files, SOUL definitions, and memory examples
├── knowledge/                  # User-editable Markdown knowledge base
├── src/
│   └── arietta_voice/
│       ├── tools/              # Deterministic tools (directly invoked)
│       ├── model_tools/        # Tools invoked by model reasoning
│       └── ...                 # Core runtime, models, audio, and knowledge handling
├── tests/                      # Unit tests for config, tools, knowledge, wake word, and model tooling
```

## Notes

- Licensed under the Apache License 2.0.
- The default wake-word backend is transcript-based, not an acoustic hotword model.
- The local chat path is optimized for Apple Silicon and MLX.
- `knowledge-index` is only required when you use the semantic or hybrid backend.
- `backend = "keyword"` is the simplest starting point for a user-customized deployment.
