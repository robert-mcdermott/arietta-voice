# Getting Started With Arietta Voice Customization

This guide walks through the main ways to customize Arietta Voice for your own use:

- add knowledge
- choose a knowledge backend
- build the semantic index when needed
- add deterministic tools
- add model-requested tools
- personalize the assistant's name, wake word, and `SOUL.md`
- verify your changes

If you want the short overview, start with [README.md](./README.md). If you want the practical customization walkthrough, this is the file to use.

## 1. Know Which Files Matter

The main files and directories you will touch are:

- [`config/arietta_voice.toml`](./config/arietta_voice.toml): main configuration
- [`config/SOUL.md`](./config/SOUL.md): the assistant's personality, tone, and behavior rules
- `config/MEMORY.md`: optional long-term memory file, created automatically if memory is enabled
- [`knowledge/`](./knowledge): your markdown knowledge base
- [`src/arietta_voice/tools/`](./src/arietta_voice/tools): deterministic tools
- [`src/arietta_voice/model_tools/`](./src/arietta_voice/model_tools): model-requested tools

Good first commands:

```bash
uv run arietta-voice config
uv run arietta-voice doctor
uv run arietta-voice knowledge-search "what can you help with"
```

For operational work, the admin console can be started with:

```bash
uv run arietta-voice serve
```

You can run `uv run arietta-voice` and `uv run arietta-voice serve` in separate
terminals. In that setup the admin console can inspect health, logs, runtime
heartbeat, knowledge, tools, and config, but it cannot stop or restart the
manually started voice runtime process. If you want the Start, Stop, and
Restart buttons to fully control the runtime, start the runtime from inside the
admin console instead. After changing tools or runtime config while using a
manually started runtime, restart that runtime terminal yourself.

## 2. Personalize The Assistant

The default assistant name and wake word are `Bridget`. The examples below use `Maya` to show customization, because it is a common, easy-to-pronounce name. In general, avoid using the product name `Arietta` as the wake word unless you are confident your users will say it consistently.

### Change the spoken name

Edit [`config/arietta_voice.toml`](./config/arietta_voice.toml):

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

Edit [`config/SOUL.md`](./config/SOUL.md). Keep this file focused on durable behavior, not changing facts.

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

Create markdown files in [`knowledge/`](./knowledge). A template is already included at [`knowledge/_template.md`](./knowledge/_template.md).

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

Knowledge retrieval is configured in `[knowledge]` inside [`config/arietta_voice.toml`](./config/arietta_voice.toml).

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

## 6. Add A Deterministic Tool

Deterministic tools are the tools Arietta checks before calling the model.

Use them when:

- you want exact routing
- the answer should come directly from the tool
- the tool performs a precise action or lookup

The built-in example is [`src/arietta_voice/tools/local_time.py`](./src/arietta_voice/tools/local_time.py).

### How deterministic tools work

Each deterministic tool:

- lives in [`src/arietta_voice/tools/`](./src/arietta_voice/tools)
- exposes `create_tool(context)`
- returns an object with `name` and `maybe_handle(user_text)`
- returns `ToolResponse` when it wants to handle the turn

The base context lives in [`src/arietta_voice/tools/base.py`](./src/arietta_voice/tools/base.py) and gives the tool access to:

- `context.config`
- `context.knowledge_retriever`

Start from the template at [`src/arietta_voice/tools/tool_template.py`](./src/arietta_voice/tools/tool_template.py).

### Deterministic tool order matters

Deterministic tools are checked in the order they appear in `[tools].enabled`, and Arietta stops at the first tool that returns a match.

That means order is important:

- put narrow, exact tools first
- put broader fallback tools later
- keep `knowledge_retrieval` last in most configurations

Why `knowledge_retrieval` should usually be last:

- it is much broader than most exact tools
- if it finds a match first, later tools never get a chance
- it works best as the fallback grounding step after exact tools have already had their turn

Good ordering example:

```toml
[tools]
enabled = [
  "local_time",
  "assistant_identity",
  "home_mode",
  "knowledge_retrieval",
]
```

In that order:

- `local_time` gets first chance to answer exact time questions
- `assistant_identity` handles direct identity questions
- `home_mode` handles its own exact routing
- `knowledge_retrieval` acts as the broad fallback context tool

If you put `knowledge_retrieval` before `home_mode`, then some home-mode questions may get routed into knowledge retrieval instead of your exact tool.

### Example deterministic tool

Create `src/arietta_voice/tools/home_mode.py`:

```python
from __future__ import annotations

from pathlib import Path
import re

from arietta_voice.models import ToolResponse
from arietta_voice.tools.base import ToolContext


MATCH_PATTERNS = (
    re.compile(r"\bwhat(?:'s| is) the home mode\b"),
    re.compile(r"\bwhat mode is the house in\b"),
)


class HomeModeTool:
    name = "get_home_mode"

    def __init__(self, context: ToolContext) -> None:
        self.project_root = context.config.project_root

    def maybe_handle(self, user_text: str) -> ToolResponse | None:
        lowered = user_text.lower()
        if not any(pattern.search(lowered) for pattern in MATCH_PATTERNS):
            return None

        mode_path = self.project_root / "data" / "home_mode.txt"
        mode = mode_path.read_text(encoding="utf-8").strip() if mode_path.exists() else "unknown"
        return ToolResponse(
            name=self.name,
            content=f"The home is currently in {mode} mode.",
        )


def create_tool(context: ToolContext) -> HomeModeTool:
    return HomeModeTool(context)
```

Then enable it:

```toml
[tools]
enabled = [
  "local_time",
  "assistant_identity",
  "home_mode",
  "knowledge_retrieval",
]
```

### Deterministic tool patterns to follow

- Return `None` if the tool should not handle the utterance.
- Return a final answer if the tool is the source of truth.
- Keep matching logic clear and predictable.
- Put live lookups or device control here when you want exact intent routing.

### Context-only deterministic tools

Deterministic tools do not always have to answer directly. The built-in knowledge retrieval tool in [`src/arietta_voice/tools/knowledge_retrieval.py`](./src/arietta_voice/tools/knowledge_retrieval.py) returns `ToolResponse(..., bypass_model=False, context=...)`, which means:

- the tool matched
- the tool gathered grounding context
- the model still writes the final answer

That is a good pattern when you want explicit routing but still want the model to phrase the reply naturally.

## 7. Add A Model-Requested Tool

Model-requested tools are the non-deterministic tool path. Arietta exposes these to the model after the deterministic pass.

Use them when:

- the model should decide whether a lookup is needed
- the tool gathers supporting information rather than speaking directly
- the tool is useful, but only for some questions

The built-in example is [`src/arietta_voice/model_tools/search_knowledge.py`](./src/arietta_voice/model_tools/search_knowledge.py).

### How model-requested tools work

Each model-requested tool:

- lives in [`src/arietta_voice/model_tools/`](./src/arietta_voice/model_tools)
- exposes `create_tool(context)`
- defines `name`, `description`, and `arguments_example`
- implements `invoke(arguments)`
- returns `ModelToolExecution(followup_context=...)`

The runtime flow is in [`src/arietta_voice/model_tooling.py`](./src/arietta_voice/model_tooling.py):

1. the model is shown the available tool metadata
2. it may emit a single JSON tool call
3. Arietta executes the tool
4. the tool returns `followup_context`
5. the model gets one more chance to answer using that grounded context

Start from the template at [`src/arietta_voice/model_tools/tool_template.py`](./src/arietta_voice/model_tools/tool_template.py).

### Example model-requested tool

Create `src/arietta_voice/model_tools/room_sensor.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from arietta_voice.model_tools.base import ModelToolContext, ModelToolExecution


class RoomSensorTool:
    name = "get_room_sensor"
    description = "you need the latest room temperature or humidity reading from the local sensor file"
    arguments_example = {"metric": "temperature"}

    def __init__(self, context: ModelToolContext) -> None:
        self.project_root = context.config.project_root

    def invoke(self, arguments: dict[str, object]) -> ModelToolExecution | None:
        metric = str(arguments.get("metric", "")).strip().lower()
        if metric not in {"temperature", "humidity"}:
            return None

        sensor_path = self.project_root / "data" / "room_sensor.json"
        if not sensor_path.exists():
            return ModelToolExecution(
                followup_context=(
                    "Room sensor tool was requested, but the sensor data file is missing. "
                    "Say that the reading is unavailable instead of inventing one."
                ),
                details={"metric": metric, "status": "missing"},
            )

        payload = json.loads(sensor_path.read_text(encoding="utf-8"))
        value = payload.get(metric)
        unit = "degrees Fahrenheit" if metric == "temperature" else "percent"
        return ModelToolExecution(
            followup_context=(
                f"Room sensor result:\n"
                f"{metric.title()}: {value} {unit}\n"
                "Use this tool result directly when answering."
            ),
            details={"metric": metric, "status": "ok"},
        )


def create_tool(context: ModelToolContext) -> RoomSensorTool:
    return RoomSensorTool(context)
```

Then enable it:

```toml
[model_tools]
enabled = [
  "search_knowledge",
  "system_info",
  "room_sensor",
]
```

### Deterministic vs model-requested: how to choose

Choose a deterministic tool when:

- the user intent is obvious from the utterance
- the tool should either answer or act immediately
- you do not want the model deciding whether to call it

Choose a model-requested tool when:

- the lookup is optional
- the tool result is supporting context
- you want the model to decide if the extra step is necessary

Good rule of thumb:

- `What time is it?` -> deterministic
- `Can you check the room sensor if that would help answer this?` -> model-requested

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

## 9. Common Customization Patterns

### Household assistant

- rename the assistant
- change the wake word
- add house rules, Wi-Fi info, appliance notes, and routines as knowledge
- add deterministic tools for exact home-control commands
- add model-requested tools for optional sensor lookups

### Office assistant

- put stable policy and support info in knowledge
- use deterministic tools for exact internal utilities
- use model-requested tools for live status lookups
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

## 10. Troubleshooting

### "I changed knowledge but answers did not improve"

Check:

- are you editing files under `knowledge/`?
- are the article titles and headings clear?
- if using `hybrid` or `vector`, did you run `uv run arietta-voice knowledge-index`?

### "My new tool never runs"

Check:

- the module is in the right directory
- the module name is listed in `[tools].enabled` or `[model_tools].enabled`
- `create_tool(context)` exists
- the matching logic actually triggers

### "The assistant does not wake reliably"

Remember:

- the current wake-word backend is transcript-based
- the wake phrase should be near the start of the utterance
- shorter, clearer wake phrases usually work better

### "I put too much in SOUL.md"

If the file starts becoming long and factual, move those facts into knowledge articles instead.

## 11. Suggested First Customization Pass

If you are setting up Arietta Voice for the first time, this is a good sequence:

1. Change the assistant name and wake word.
2. Rewrite `config/SOUL.md`.
3. Add 3-5 knowledge articles for the most common questions.
4. Start with `backend = "keyword"`.
5. Verify retrieval with `knowledge-search`.
6. Add one deterministic tool for an exact action or lookup.
7. Add one model-requested tool only if you truly need optional follow-up lookups.
8. Once your knowledge base grows, try `hybrid` and build the index.

That path usually gives the fastest progress with the least complexity.
