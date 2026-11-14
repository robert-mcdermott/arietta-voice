"""Parse the JSON tool request a model writes when it wants to call a tool."""

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
