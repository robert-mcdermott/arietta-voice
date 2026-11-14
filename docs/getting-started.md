# Getting Started With Arietta Voice Customization

This guide walks through the main ways to customize Arietta Voice for your own use:

- add knowledge
- choose a knowledge backend
- build the semantic index when needed
- add tools
- personalize the assistant's name, wake word, and `SOUL.md`
- verify your changes

If you want the short overview, start with [README.md](../README.md). If you want the practical customization walkthrough, this is the file to use.

## 1. Know Which Files Matter

The main files and directories you will touch are:

- [`config/arietta_voice.toml`](../config/arietta_voice.toml): tracked shared defaults
- `config/arietta_voice.local.toml`: ignored private configuration
- `config/SOUL.local.md`: ignored private personality, tone, and behavior rules
- `config/MEMORY.md`: optional long-term memory file, created automatically if memory is enabled
- [`knowledge/`](../knowledge): your markdown knowledge base
- [`tools/`](../tools): your own tools, one Python file each
- [`src/arietta_voice/tools/builtin/`](../src/arietta_voice/tools/builtin): the built-in tools

Create private copies once before customizing:

```bash
uv run arietta-voice config-init
```

This preserves existing effective settings and persona content, leaves the
original files untouched, and saves private backups under `config/.private-backups/`.
It refuses to overwrite existing private files. Once you have verified the
copies, restore any personal edits in tracked templates to their committed
defaults; keep future customization in the private files. See
[configuration precedence](../README.md#configuration) for custom paths and merging.

Good first commands:

```bash
uv run arietta-voice config
uv run arietta-voice doctor
uv run arietta-voice knowledge-search "what can you help with"
```

For operational work, the admin console can be started with:

```bash
uv run arietta-voice admin-setup
uv run arietta-voice serve
```

Run `admin-setup` once, using the same `--config` as `serve` if customized.
There is no built-in password. It writes an ignored credential file containing
a salted password hash, usable by both interactive and LaunchAgent starts.
Source editing is disabled until explicitly enabled; see
[Admin tool permissions](../README.md#admin-tool-permissions).

You can run `uv run arietta-voice` and `uv run arietta-voice serve` in separate
terminals. In that setup the admin console can inspect health, logs, runtime
heartbeat, knowledge, tools, and config, but it cannot stop or restart the
manually started voice runtime process. If you want the Start, Stop, and
Restart buttons to fully control the runtime, start the runtime from inside the
admin console instead. After changing tools or runtime config while using a
manually started runtime, restart that runtime terminal yourself.

Conversation text is hidden from runtime logs and status by default; the
runtime monitor's last-user/last-assistant fields will be blank. See
[Privacy and logging](../README.md#privacy-and-logging) for debugging opt-in,
rotation settings, and how memory, recordings, and Bedrock differ from logs.
Restart the runtime and API after changing logging settings so both processes
use the same policy. Existing log files are not scrubbed during an upgrade.

## 2. Personalize The Assistant

The default assistant name and wake word are `Bridget`. The examples below use `Maya` to show customization, because it is a common, easy-to-pronounce name. In general, avoid using the product name `Arietta` as the wake word unless you are confident your users will say it consistently.

### Change the spoken name

Edit `config/arietta_voice.local.toml`:

```toml
[assistant]
name = "Maya"
identity = "I'm Maya, a local voice-first assistant for our home."
```

`assistant.name` is the user-facing assistant name. `assistant.identity` is the short factual self-description used by the built-in identity tool.

### Change the wake word

Edit the same config file:

```toml
[wake_word]
phrases = ["maya"]
acknowledgement = "I'm here."
session_timeout_s = 30.0
```

Notes:

- `phrases` can contain more than one wake phrase.
- The current wake-word backend is transcript-based, so the phrase needs to be spoken near the start of the utterance.
- `session_timeout_s` controls how long Arietta keeps listening for follow-up questions after activation.

Example with multiple wake phrases:

```toml
[wake_word]
phrases = ["maya", "computer"]
acknowledgement = "Yes?"
session_timeout_s = 20.0
```

### Give it a new soul

Edit `config/SOUL.local.md`. Keep this file focused on durable behavior, not changing facts.

Good things to put in `SOUL.md`:

- personality and tone
- response style
- safety rules
- how honest or concise it should be
- how it should handle uncertainty

Avoid putting long project or household facts in `SOUL.md`. Those belong in `knowledge/*.md`.

Example soul:

```md
# Soul

Your name is Maya.
You are a warm, concise local voice assistant for our household.
Prefer short spoken answers.
If you are not sure, say so clearly.
Use local knowledge when relevant.
Use tools when they are the source of truth.
Do not invent device states or automation results.
```

## 3. Add Knowledge

Knowledge is the easiest and safest way to teach Arietta new facts.

Create markdown files in [`knowledge/`](../knowledge) by copying
[`examples/knowledge-template.md`](../examples/knowledge-template.md) to a new
file there. Personal articles and subdirectories are ignored by Git. Only
`knowledge/README.md` and `knowledge/getting-started.md` remain tracked, so
keep private content out of those two starter files. See the
[knowledge README](../knowledge/README.md) for previously tracked articles and
knowledge directories outside this default location.

### A minimal article

```md
---
title: House Rules
tags: house, quiet hours, guests
aliases: what are the house rules
---

## Quiet Hours

Quiet hours are from 10 PM to 7 AM on weekdays.

## Guest Wi-Fi

Guests should use the network named "Guest" and the password stored in the entry drawer card.
```

### Recommended structure

Use:

- one topic per file
- short YAML frontmatter
- strong headings
- short factual paragraphs
- durable information

Frontmatter fields you should know:

- `title`: the main title used in retrieval results
- `tags`: comma-separated keywords
- `aliases`: alternate phrasings people might say aloud

### What belongs in knowledge vs tools

Put something in knowledge if:

- it is mostly factual
- it changes infrequently
- the assistant can answer from a text snippet

Use a tool instead if:

- the answer must be live or computed
- the action changes the outside world
- the source of truth is a file, API, sensor, or device

## 4. Choose A Knowledge Backend

Knowledge retrieval is configured in `[knowledge]` inside `config/arietta_voice.local.toml`.

Retrieval skips hidden/underscore-prefixed files and directories, and symbolic
links that resolve outside the configured knowledge directory. Keep searchable
articles inside that directory.

After upgrading from the original mean-pooled BGE implementation, rebuild any
semantic index with:

```bash
uv run arietta-voice knowledge-index
```

The corrected embedding pipeline uses normalized CLS embeddings and index
version 2. Old indexes are rejected even if the source articles are unchanged.
Until rebuilt, `hybrid` uses keyword matches and `vector` returns no matches.
Restart a running runtime/API after rebuilding to reload its index.

### Option 1: `keyword`

```toml
[knowledge]
backend = "keyword"
```

Use `keyword` when:

- you want the simplest setup
- your article titles, tags, and wording are fairly direct
- you do not want to manage a semantic index yet

How it works:

- live lexical matching over titles, tags, headings, filenames, and text
- no index build required

This is the best default for a new deployment.

### Option 2: `hybrid`

```toml
[knowledge]
backend = "hybrid"
index_directory = "../.knowledge_index"
embedding_model = "BAAI/bge-small-en-v1.5"
hybrid_alpha = 0.55
min_vector_score = 0.3
```

Use `hybrid` when:

- you want semantic matching plus exact-match strength
- users may ask the same thing in different words
- you still want keyword precision for titles, tags, and known names

How it works:

- combines keyword scoring with semantic vector similarity
- needs a built index for the vector portion
- still benefits from clear titles and tags

`hybrid` is usually the best long-term mode once your content grows.

### Option 3: `vector`

```toml
[knowledge]
backend = "vector"
index_directory = "../.knowledge_index"
embedding_model = "BAAI/bge-small-en-v1.5"
min_vector_score = 0.3
```

Use `vector` when:

- you want purely semantic retrieval
- you are testing embeddings specifically
- your users phrase things many different ways

How it works:

- retrieves only from the semantic index
- requires `knowledge-index`
- is less dependent on exact keyword overlap

`vector` is powerful, but it is usually less transparent than `keyword` or `hybrid`.

## 5. Build Or Refresh The Knowledge Index

You only need to build the index for `hybrid` or `vector`.

Run:

```bash
uv run arietta-voice knowledge-index
```

This builds derived files under `.knowledge_index/`.

Recommended workflow:

1. Add or update markdown files in `knowledge/`
2. Run `uv run arietta-voice knowledge-index`
3. Verify retrieval with `uv run arietta-voice knowledge-search "your question here"`

Example:

```bash
uv run arietta-voice knowledge-search "what are the house rules"
uv run arietta-voice knowledge-index
uv run arietta-voice knowledge-search "what are the house rules"
```

Important behavior:

- `keyword` does not need indexing
- `vector` depends entirely on the index
- `hybrid` should be reindexed after meaningful knowledge changes if you want the semantic side to stay fresh

If you edit a file and forget to reindex, `hybrid` may still find it through keyword matching, but you should not treat that as your normal workflow.

## 6. Add A Tool

A tool is a Python file in [`tools/`](../tools). Start from the working example:

```bash
cp tools/example_dice.py tools/office_lights.py
```

Then edit it. The `name` must match the file name:

```python
from arietta_voice.tools import BaseTool, ToolResult


class OfficeLightsTool(BaseTool):
    name = "office_lights"
    description = "the user asks whether the office lights are on"
    triggers = (r"\bare the (?:office )?lights on\b",)
    parameters = {"type": "object", "properties": {}}

    def run(self, arguments):
        state = "on"  # Replace with a real lookup.
        return ToolResult(
            content=f"The office lights are {state}.",
            speak=f"The office lights are {state}.",
        )


def create_tool(context):
    return OfficeLightsTool(context)
```

Enable it by adding its name to `[tools].enabled` in your private config, before
`search_knowledge`:

```toml
[tools]
enabled = ["assistant_identity", "local_time", "office_lights", "search_knowledge", "system_info"]
```

Or open the console's Tools page, switch it on, move it above
`search_knowledge` and choose Save. Restart the voice runtime afterwards.

Test it without speaking: the Tools page's Test button runs one tool with a
sample request or with arguments, and "Which tool answers?" shows which enabled
tool would handle a request. [`tools/README.md`](../tools/README.md) lists every
attribute a tool can set.

### How a tool is used

- **Triggers** are regular expressions tried before the model, in the order of
  `[tools].enabled`. The first tool whose trigger matches runs. If it returns
  `speak`, that is the whole answer. If it returns only `content`, the content
  is added to the prompt and the model answers. Named groups in a trigger, such
  as `(?P<room>\w+)`, become arguments.
- **Model calls.** Unless `model_callable = False`, the model is shown the
  tool's name, description and arguments and may call it once before answering.
  It then answers using the tool's `content`.
- **Arguments** are checked against `parameters` before `run` is called. Use
  types string, integer, number or boolean, with `enum`, `minimum`, `maximum`
  and `maxLength`, and list required ones in `required`.
- **Failures** never end the conversation. A tool that raises, takes longer than
  `timeout_s` (10 seconds by default) or receives invalid arguments is reported
  to the model, which answers without it.

Only `tools/README.md` and `tools/example_dice.py` are tracked by Git; your own
tool files stay local and survive upgrades.

## 7. Choose Instant Answers Or Model Calls

Give a tool triggers and return `speak` when:

- the answer must be exact, such as the time or a sensor reading
- the request is easy to recognise
- waiting for the model adds nothing

Leave triggers out and write a good `description` when:

- the model should decide whether the tool is needed
- the model has to pick arguments from free-form speech
- the tool gathers facts the model should put into words

A tool can do both: `local_time` answers "what time is it" instantly and also
lets the model look up the date. Write descriptions as "use when ...": the
model sees `use when <description>`.

Mark tools that change something in the world, such as turning off lights,
with `consequential = True`. The planned agent loop will ask before running
them.

## 8. Verify Your Customizations

After making changes, the most useful commands are:

```bash
uv run arietta-voice config
uv run arietta-voice doctor
uv run arietta-voice knowledge-search "your test query"
uv run pytest
```

Practical checks:

- if you changed knowledge, test retrieval explicitly
- if you added a tool, make sure it is listed in the correct config section
- if you changed the wake word, say it at the start of the utterance
- if you switched to `hybrid` or `vector`, rebuild the index

## 9. Run Automatically on macOS With launchd

For an always-on Mac mini, use macOS `launchd` so Arietta Voice starts again
after a reboot. The recommended setup is two user LaunchAgents:

- one LaunchAgent for the voice runtime
- one LaunchAgent for the HTTP API and admin console

This matches the normal two-terminal development workflow:

```bash
uv run arietta-voice
uv run arietta-voice serve
```

### Important Mac mini considerations

Use a user LaunchAgent under `~/Library/LaunchAgents`, not a root LaunchDaemon
under `/Library/LaunchDaemons`, for the voice runtime. The voice runtime needs
microphone/audio access, and macOS privacy permissions are tied to the logged-in
user session. A root LaunchDaemon is more likely to run into microphone,
speaker, keychain, or environment issues.

For a headless Mac mini:

- create or choose the macOS user that should own Arietta
- log into that user once
- run `uv run arietta-voice` manually once and grant microphone/audio prompts
- make sure the user session exists after reboot, for example with automatic
  login or your normal remote-login workflow
- keep the Mac awake enough for an always-on assistant
- access the machine through Screen Sharing, SSH, Tailscale, or another private
  remote-access method

Keep the admin console bound to `127.0.0.1` unless it is behind SSH tunneling,
Tailscale, a VPN, or a properly secured reverse proxy. The built-in admin auth
is intentionally simple and should not be exposed directly to the public
internet.

### Before creating LaunchAgents

From the project directory, verify the app works manually:

```bash
uv run arietta-voice doctor
uv run arietta-voice
```

In another terminal, verify the admin console:

```bash
uv run arietta-voice serve
```

Open `http://127.0.0.1:8765/admin` and confirm the console loads.

Find the full path to `uv`:

```bash
command -v uv
```

On Apple Silicon Homebrew installs this is often `/opt/homebrew/bin/uv`. Use
the path from your machine in the plist files below.

Create the logs directory if it does not already exist:

```bash
mkdir -p logs
mkdir -p ~/Library/LaunchAgents
```

### Voice runtime LaunchAgent

Create `~/Library/LaunchAgents/com.arietta.voice.plist`.

Replace:

- `/Users/your-user/mycode/arietta-voice` with your project path
- `/opt/homebrew/bin/uv` with the output from `command -v uv`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>com.arietta.voice</string>

    <key>WorkingDirectory</key>
    <string>/Users/your-user/mycode/arietta-voice</string>

    <key>ProgramArguments</key>
    <array>
      <string>/opt/homebrew/bin/uv</string>
      <string>run</string>
      <string>--frozen</string>
      <string>arietta-voice</string>
    </array>

    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>/Users/your-user/mycode/arietta-voice/logs/launchd.voice.out.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/your-user/mycode/arietta-voice/logs/launchd.voice.err.log</string>
  </dict>
</plist>
```

If you use a custom config path, include it in `ProgramArguments`:

```xml
<string>--config</string>
<string>config/arietta_voice.toml</string>
```

### Admin console LaunchAgent

Create `~/Library/LaunchAgents/com.arietta.admin.plist`.

Replace the same paths. First run `uv run arietta-voice admin-setup` from the
project directory (with the same `--config`, if customized). The LaunchAgent
reads that private credential file; no password needs to be stored in its plist.
If an older plist sets `ARIETTA_ADMIN_PASSWORD`, remove that environment entry
to use the credential file, since explicit environment credentials take precedence.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>com.arietta.admin</string>

    <key>WorkingDirectory</key>
    <string>/Users/your-user/mycode/arietta-voice</string>

    <key>ProgramArguments</key>
    <array>
      <string>/opt/homebrew/bin/uv</string>
      <string>run</string>
      <string>--frozen</string>
      <string>arietta-voice</string>
      <string>serve</string>
      <string>--host</string>
      <string>127.0.0.1</string>
      <string>--port</string>
      <string>8765</string>
    </array>

    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>/Users/your-user/mycode/arietta-voice/logs/launchd.admin.out.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/your-user/mycode/arietta-voice/logs/launchd.admin.err.log</string>
  </dict>
</plist>
```

If the voice runtime uses `--config`, give the admin console the same config
path:

```xml
<string>--config</string>
<string>config/arietta_voice.toml</string>
```

### Load and manage the jobs

Check the plist syntax:

```bash
plutil -lint ~/Library/LaunchAgents/com.arietta.voice.plist
plutil -lint ~/Library/LaunchAgents/com.arietta.admin.plist
```

Load both LaunchAgents:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.arietta.voice.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.arietta.admin.plist
```

Restart a job after changing a plist:

```bash
launchctl kickstart -k gui/$(id -u)/com.arietta.voice
launchctl kickstart -k gui/$(id -u)/com.arietta.admin
```

Unload a job:

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.arietta.voice.plist
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.arietta.admin.plist
```

Inspect launchd state:

```bash
launchctl print gui/$(id -u)/com.arietta.voice
launchctl print gui/$(id -u)/com.arietta.admin
```

Watch the launchd logs:

```bash
tail -f logs/launchd.voice.out.log logs/launchd.voice.err.log
tail -f logs/launchd.admin.out.log logs/launchd.admin.err.log
```

### What the admin console can and cannot control

With the two LaunchAgent setup, the admin console can still show health, logs,
runtime heartbeat, tools, knowledge, diagnostics, and config state. It can also
edit knowledge files and tool source files.

The admin console does not own the separately launched voice process. That
means the Start, Stop, and Restart runtime buttons are not the primary way to
control the launchd-started voice job. Use `launchctl kickstart -k` to restart
the voice runtime after changing runtime config or tool source.

If you want the admin console to fully own Start, Stop, and Restart, start only
the admin console LaunchAgent and use the admin console's Start button to launch
the voice runtime as an API-managed child process. That mode gives better
admin-console lifecycle control, but launchd is only supervising the admin
console directly.

## 10. Common Customization Patterns

### Household assistant

- rename the assistant
- change the wake word
- add house rules, Wi-Fi info, appliance notes, and routines as knowledge
- add tools with triggers for exact home-control commands
- add model-called tools for optional sensor lookups

### Office assistant

- put stable policy and support info in knowledge
- use tools with triggers for exact internal utilities
- use model-called tools for live status lookups
- keep `SOUL.md` concise and professional

### Bedrock-backed setup

Keep STT, TTS, knowledge, and tools local, but switch chat to Bedrock:

```toml
[models]
chat_provider = "bedrock"
chat_model = "us.anthropic.claude-sonnet-4-20250514-v1:0"
bedrock_region = "us-west-2"
bedrock_profile = "default"
```

## 11. Troubleshooting

### "I changed knowledge but answers did not improve"

Check:

- are you editing files under `knowledge/`?
- are the article titles and headings clear?
- if using `hybrid` or `vector`, did you run `uv run arietta-voice knowledge-index`?

### "My new tool never runs"

Check:

- the file is in `tools/` and its `name` matches the file name
- the name is listed in `[tools].enabled`, before `search_knowledge` if it has triggers
- `create_tool(context)` exists
- the console's Tools page shows no error for it, and Validate passes
- its trigger matches your request (use Test with a sample request)
- the voice runtime was restarted after the change

### "The assistant does not wake reliably"

Remember:

- the current wake-word backend is transcript-based
- the wake phrase should be near the start of the utterance
- shorter, clearer wake phrases usually work better

### "I put too much in SOUL.md"

If the file starts becoming long and factual, move those facts into knowledge articles instead.

## 12. Suggested First Customization Pass

If you are setting up Arietta Voice for the first time, this is a good sequence:

1. Change the assistant name and wake word.
2. Customize `config/SOUL.local.md` after running `config-init` once.
3. Add 3-5 knowledge articles for the most common questions.
4. Start with `backend = "keyword"`.
5. Verify retrieval with `knowledge-search`.
6. Add one tool for an exact action or lookup.
7. Let the model call it too, if requests for it vary in wording.
8. Once your knowledge base grows, try `hybrid` and build the index.

That path usually gives the fastest progress with the least complexity.
