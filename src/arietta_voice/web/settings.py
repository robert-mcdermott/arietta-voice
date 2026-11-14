"""The settings the console can change, how to check them, and how to save them.

Only a curated set is editable: things an owner tunes, none of them secrets or
file paths. A change is checked with the same validation the runtime uses
before it is written to the private config, and the voice runtime picks it up
on its next start.
"""

from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arietta_voice.config import AppConfig, validate_config
from arietta_voice.private_config import local_config_path, update_private_section


@dataclass(slots=True, frozen=True)
class Field:
    section: str
    key: str
    label: str
    kind: str  # text, list, number, integer, boolean, choice
    help: str = ""
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    max_length: int = 120


GROUPS: list[tuple[str, str, list[Field]]] = [
    (
        "assistant",
        "Assistant",
        [
            Field("assistant", "name", "Name", "text", "What the assistant calls itself.", max_length=40),
            Field("wake_word", "enabled", "Wake word", "boolean", "Listen for the wake word before answering."),
            Field("wake_word", "phrases", "Wake phrases", "list", "Comma-separated, such as bridget, hey bridget."),
            Field("wake_word", "session_timeout_s", "Conversation timeout (seconds)", "number",
                  "How long a conversation stays open after the last exchange.", minimum=5, maximum=300),
            Field("assistant", "max_history_turns", "Turns remembered in a conversation", "integer",
                  minimum=0, maximum=20),
        ],
    ),
    (
        "voice",
        "Voice",
        [
            Field("audio", "voice", "Kokoro voice", "text", "Such as af_heart, af_bella or bm_george.", max_length=40),
            Field("audio", "tts_speed", "Speaking speed", "number", "1 is normal.", minimum=0.5, maximum=2.0),
            Field("audio", "chime_enabled", "Thinking chime", "boolean", "A soft tick while an answer is prepared."),
        ],
    ),
    (
        "model",
        "Model",
        [
            Field("models", "temperature", "Temperature", "number",
                  "Lower is more predictable; higher is more varied.", minimum=0, maximum=2),
            Field("models", "max_tokens", "Longest answer (tokens)", "integer", minimum=16, maximum=1024),
            Field("models", "native_tool_calls", "Native tool calls", "boolean",
                  "Let the local model call tools in its own format."),
        ],
    ),
    (
        "timers",
        "Timers and reminders",
        [
            Field("timers", "repeat_s", "Repeat an unanswered alert every (seconds)", "number",
                  minimum=5, maximum=600),
            Field("timers", "max_repeats", "Repeats before giving up", "integer", minimum=0, maximum=20),
            Field("timers", "snooze_minutes", "Snooze length (minutes)", "number", minimum=1, maximum=120),
            Field("timers", "alert_tone", "Alert tone", "boolean", "A short tone before an alert is spoken."),
        ],
    ),
    (
        "weather",
        "Weather",
        [
            Field("weather", "location", "Town or city", "text",
                  "Sent to open-meteo.com with its coordinates. Empty turns weather off.", max_length=120),
            Field("weather", "units", "Units", "choice", choices=("imperial", "metric")),
        ],
    ),
    (
        "privacy",
        "Memory and privacy",
        [
            Field("memory", "enabled", "Remember facts between conversations", "boolean",
                  "Keeps a short memory file of things you tell it."),
            Field("logging", "content_enabled", "Log conversation text", "boolean",
                  "Off keeps what you say out of logs and the console's status."),
            Field("knowledge", "backend", "Knowledge search", "choice",
                  "Hybrid and vector need the index built.", choices=("keyword", "hybrid", "vector")),
        ],
    ),
]  # fmt: skip
FIELDS = {(field.section, field.key): field for _, _, fields in GROUPS for field in fields}


def describe(config: AppConfig, config_path: Path) -> dict[str, Any]:
    """Editable settings with their current values and which ones are set privately."""
    private = _private(config_path)
    groups = []
    for name, title, fields in GROUPS:
        items = []
        for field in fields:
            value = getattr(getattr(config, field.section), field.key)
            items.append(
                {
                    "section": field.section,
                    "key": field.key,
                    "label": field.label,
                    "kind": field.kind,
                    "help": field.help,
                    "choices": list(field.choices),
                    "minimum": field.minimum,
                    "maximum": field.maximum,
                    "max_length": field.max_length,
                    "value": list(value) if isinstance(value, tuple) else value,
                    "private": field.key in private.get(field.section, {}),
                }
            )
        groups.append({"name": name, "title": title, "fields": items})
    return {"groups": groups, "private_file": str(local_config_path(config_path))}


def apply(config: AppConfig, config_path: Path, changes: dict[str, dict[str, Any]]) -> list[str]:
    """Check and save changes; returns the "section.key" names that changed.

    Raises ValueError with a readable message when a value is not allowed. On
    success the private config is updated and ``config`` is changed in place.
    """
    candidate = copy.deepcopy(config)
    accepted: dict[str, dict[str, Any]] = {}
    for section, values in changes.items():
        if not isinstance(values, dict):
            raise ValueError(f"{section} must be an object")
        for key, raw in values.items():
            field = FIELDS.get((section, key))
            if field is None:
                raise ValueError(f"{section}.{key} cannot be changed here")
            value = _coerce(field, raw)
            if getattr(getattr(candidate, section), key) != value:
                setattr(getattr(candidate, section), key, value)
                accepted.setdefault(section, {})[key] = value
    validate_config(candidate)
    for section, values in accepted.items():
        update_private_section(
            config_path,
            section,
            {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in values.items()
            },
        )
        for key, value in values.items():
            setattr(getattr(config, section), key, value)
    return [f"{section}.{key}" for section, values in accepted.items() for key in values]


def _coerce(field: Field, raw: Any) -> Any:
    name = f"{field.label}"
    if field.kind == "boolean":
        if not isinstance(raw, bool):
            raise ValueError(f"{name} must be on or off")
        return raw
    if field.kind in ("number", "integer"):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"{name} must be a number")
        if field.kind == "integer" and int(raw) != raw:
            raise ValueError(f"{name} must be a whole number")
        value = int(raw) if field.kind == "integer" else float(raw)
        if field.minimum is not None and value < field.minimum:
            raise ValueError(f"{name} must be at least {field.minimum:g}")
        if field.maximum is not None and value > field.maximum:
            raise ValueError(f"{name} must be at most {field.maximum:g}")
        return value
    if field.kind == "list":
        items = raw if isinstance(raw, list) else str(raw).split(",")
        values = tuple(" ".join(str(item).split()) for item in items if str(item).strip())
        if any(len(item) > field.max_length for item in values):
            raise ValueError(f"{name}: each entry must be at most {field.max_length} characters")
        return values
    if not isinstance(raw, str):
        raise ValueError(f"{name} must be text")
    value = " ".join(raw.split())
    if len(value) > field.max_length:
        raise ValueError(f"{name} must be at most {field.max_length} characters")
    if field.kind == "choice" and value not in field.choices:
        raise ValueError(f"{name} must be one of {', '.join(field.choices)}")
    return value


def _private(config_path: Path) -> dict[str, dict[str, Any]]:
    path = local_config_path(config_path)
    try:
        return tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, tomllib.TOMLDecodeError):
        return {}
