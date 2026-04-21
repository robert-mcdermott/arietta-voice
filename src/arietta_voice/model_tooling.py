from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Callable

from arietta_voice.model_tools import ModelToolRegistry


CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


@dataclass(slots=True, frozen=True)
class ModelToolUse:
    name: str
    details: dict[str, Any]


@dataclass(slots=True, frozen=True)
class ModelToolRequest:
    name: str
    arguments: dict[str, Any]


def maybe_generate_with_model_tools(
    messages: list[dict[str, str]],
    *,
    model_tool_registry: ModelToolRegistry,
    llm_generate: Callable[[list[dict[str, str]]], str],
) -> tuple[str, ModelToolUse | None]:
    instruction = model_tool_registry.instruction_text()
    if not instruction:
        return llm_generate(messages), None

    planning_messages = [
        {"role": "system", "content": instruction},
        *messages,
    ]
    first_response = llm_generate(planning_messages)
    request = parse_model_tool_request(first_response)
    if request is None:
        return first_response, None

    execution = model_tool_registry.execute(request.name, request.arguments)
    if execution is None:
        return first_response, None

    followup_messages = [
        *messages,
        {"role": "system", "content": execution.followup_context},
    ]
    final_response = llm_generate(followup_messages)
    return final_response, ModelToolUse(name=request.name, details=execution.details)


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
    name = str(tool_call.get("name", "")).strip()
    if not name:
        return None

    arguments = tool_call.get("arguments")
    if not isinstance(arguments, dict):
        return None
    return ModelToolRequest(name=name, arguments=arguments)


def _strip_code_fence(text: str) -> str:
    match = CODE_FENCE_RE.match(text.strip())
    return match.group(1) if match else text
