"""Parse the tool requests a model writes: prompted JSON, or a native tool call."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


@dataclass(slots=True, frozen=True)
class ModelToolRequest:
    name: str
    arguments: dict[str, Any]


def parse_model_tool_request(response_text: str) -> ModelToolRequest | None:
    payload = _strip_code_fence(response_text).strip()
    if not payload.startswith("{"):
        return None

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None

    tool_call = parsed.get("tool_call")
    if not isinstance(tool_call, dict):
        return None
    name = tool_call.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    name = name.strip()

    arguments = tool_call.get("arguments")
    if not isinstance(arguments, dict):
        return None
    return ModelToolRequest(name=name, arguments=arguments)


def _strip_code_fence(text: str) -> str:
    match = CODE_FENCE_RE.match(text.strip())
    return match.group(1) if match else text


# Native tool calls ------------------------------------------------------------
#
# Gemma 4's chat template declares tools in the system turn and the model
# answers with ``<|tool_call>call:name{key:value,...}<tool_call|>``, where text
# values are wrapped in ``<|"|>``. The model then stops; the template renders a
# tool result as ``<|tool_response>`` and the model continues the same turn.

NATIVE_CALL_START = "<|tool_call>"
NATIVE_CALL_END = "<tool_call|>"
_QUOTE = '<|"|>'
_NATIVE_CALL = re.compile(r"\s*(?:call:)?([A-Za-z_][\w.-]*)\s*\{", re.DOTALL)


def tool_specs(tools) -> list[dict[str, Any]]:
    """Tools in the OpenAI function format chat templates expect."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": f"Use when {tool.description}.",
                "parameters": tool.parameters,
            },
        }
        for tool in tools
    ]


def parse_native_tool_calls(text: str) -> list[ModelToolRequest] | None:
    """Calls in a native tool-call response, or None when it is not one.

    Only text made of calls (and whitespace) counts, so an answer that merely
    mentions a tool is never mistaken for a request.
    """
    calls: list[ModelToolRequest] = []
    rest = text.strip()
    if not rest.startswith(NATIVE_CALL_START):
        return None
    while rest:
        if not rest.startswith(NATIVE_CALL_START):
            return None
        end = rest.find(NATIVE_CALL_END)
        if end < 0:
            return None
        body = rest[len(NATIVE_CALL_START) : end]
        rest = rest[end + len(NATIVE_CALL_END) :].strip()
        found = _NATIVE_CALL.match(body)
        if found is None or not body.rstrip().endswith("}"):
            return None
        try:
            arguments = _parse_object(body[found.end() : body.rstrip().rfind("}")])
        except ValueError:
            return None
        calls.append(ModelToolRequest(name=found.group(1), arguments=arguments))
    return calls or None


def _split_top_level(text: str) -> list[str]:
    parts, depth, start, i = [], 0, 0, 0
    while i < len(text):
        if text.startswith(_QUOTE, i):
            close = text.find(_QUOTE, i + len(_QUOTE))
            if close < 0:
                raise ValueError("Unterminated text value")
            i = close + len(_QUOTE)
            continue
        char = text[i]
        if char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced brackets")
        elif char == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
        i += 1
    if depth:
        raise ValueError("Unbalanced brackets")
    if text[start:].strip():
        parts.append(text[start:])
    return parts


def _parse_object(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for entry in _split_top_level(text):
        entry = entry.strip()
        if not entry:
            continue
        key, colon, value = entry.partition(":")
        if not colon or not key.strip():
            raise ValueError("Expected key:value")
        result[key.strip().strip('"')] = _parse_value(value.strip())
    return result


def _parse_value(text: str) -> Any:
    if text.startswith(_QUOTE):
        if not text.endswith(_QUOTE) or len(text) < 2 * len(_QUOTE):
            raise ValueError("Unterminated text value")
        return text[len(_QUOTE) : -len(_QUOTE)]
    if text.startswith("{") and text.endswith("}"):
        return _parse_object(text[1:-1])
    if text.startswith("[") and text.endswith("]"):
        return [_parse_value(item.strip()) for item in _split_top_level(text[1:-1])]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text  # A bare word the model did not quote.
