"""Load enabled tools and run them with validated arguments and a time limit.

Built-in tools live in ``arietta_voice.tools.builtin``. User tools are single
Python files in the configured tools directory (``tools/`` by default), so they
survive upgrades and never require editing the installed package.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from arietta_voice.config import LEGACY_TOOL_NAMES, canonical_tool_names
from arietta_voice.tools.base import ToolContext, ToolError, ToolResult
from arietta_voice.tools.schema import check_schema, coerce_arguments, validate_arguments

logger = logging.getLogger("arietta_voice.tools")
BUILTIN_PACKAGE = "arietta_voice.tools.builtin"
TOOL_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
MAX_CONTENT_CHARS = 4000
MAX_SPEAK_CHARS = 1000
MAX_TIMEOUT_S = 120.0
LEGACY_NAMES = LEGACY_TOOL_NAMES


def enabled_tool_names(config) -> list[str]:
    """Enabled tools in order, folding in the old [model_tools] list and names."""
    return canonical_tool_names((*config.tools.enabled, *config.model_tools.enabled))


def builtin_names() -> list[str]:
    package = importlib.import_module(BUILTIN_PACKAGE)
    folder = Path(package.__file__).parent
    return sorted(p.stem for p in folder.glob("*.py") if not p.stem.startswith("_"))


def user_tool_path(config, name: str) -> Path:
    if not TOOL_NAME.fullmatch(name):
        raise ValueError(f"Tool names are lowercase letters, digits and underscores: {name!r}")
    directory = Path(config.tools.directory).resolve()
    path = (directory / f"{name}.py").resolve()
    if path.parent != directory:
        raise ValueError("Tool files must be directly inside the tools directory")
    return path


def load_module(config, name: str) -> tuple[ModuleType, str, Path | None]:
    """Import a tool module by name: your tools directory first, then the built-ins.

    A file of your own replaces a built-in of the same name, so a new built-in
    in an upgrade never silently shadows a tool you already had.
    """
    if not TOOL_NAME.fullmatch(name):
        raise ValueError(f"Tool names are lowercase letters, digits and underscores: {name!r}")
    path = user_tool_path(config, name)
    if not path.is_file():
        if name in builtin_names():
            module = importlib.import_module(f"{BUILTIN_PACKAGE}.{name}")
            return module, "builtin", Path(module.__file__)
        raise ValueError(f"No tool named {name!r}: expected {path}")
    spec = importlib.util.spec_from_file_location(f"arietta_user_tools.{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, "user", path


def create_tool(module: ModuleType, name: str, context: ToolContext):
    """Instantiate and check a tool; None means it is unavailable in this config."""
    factory = getattr(module, "create_tool", None)
    if not callable(factory):
        raise ValueError(f"Tool {name!r} must define create_tool(context)")
    tool = factory(context)
    if tool is None:
        return None
    problems = describe_problems(tool, name)
    if problems:
        raise ValueError(f"Tool {name!r}: " + "; ".join(problems))
    return tool


def describe_problems(tool, name: str) -> list[str]:
    problems = []
    if getattr(tool, "name", None) != name:
        problems.append(f"name must be {name!r} to match its file")
    if not isinstance(getattr(tool, "description", None), str) or not tool.description.strip():
        problems.append("description must say when to use the tool")
    problems += check_schema(getattr(tool, "parameters", None))
    for pattern in getattr(tool, "triggers", ()):
        try:
            re.compile(pattern)
        except (re.error, TypeError):
            problems.append(f"trigger {pattern!r} is not a valid regular expression")
    timeout = getattr(tool, "timeout_s", None)
    if not isinstance(timeout, (int, float)) or not 0 < timeout <= MAX_TIMEOUT_S:
        problems.append(f"timeout_s must be between 0 and {MAX_TIMEOUT_S:g} seconds")
    aliases = getattr(tool, "aliases", {})
    properties = (getattr(tool, "parameters", None) or {}).get("properties", {})
    if not isinstance(aliases, dict) or any(
        not isinstance(alias, str) or target not in properties for alias, target in aliases.items()
    ):
        problems.append("aliases must map other names to declared arguments")
    if not callable(getattr(tool, "run", None)):
        problems.append("run(arguments) is missing")
    return problems


def run_tool(tool, arguments: dict[str, Any]) -> ToolResult:
    """Validate arguments, run within the tool's time limit, and bound the result."""
    name = tool.name
    if not isinstance(arguments, dict):
        raise ToolError(name, "invalid_arguments", "arguments must be an object")
    arguments = coerce_arguments(tool.parameters, _apply_aliases(tool, arguments))
    problems = validate_arguments(tool.parameters, arguments)
    if problems:
        raise ToolError(name, "invalid_arguments", "; ".join(problems))
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tool-{name}")
    try:
        future = executor.submit(tool.run, arguments)
        result = future.result(timeout=tool.timeout_s)
    except FutureTimeout as exc:
        # The thread cannot be stopped; it is abandoned and its result ignored.
        raise ToolError(name, "timeout", f"no result within {tool.timeout_s:g} s") from exc
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(name, "failed", type(exc).__name__) from exc
    finally:
        executor.shutdown(wait=False)
    if result is None:
        return ToolResult()
    if not isinstance(result, ToolResult):
        raise ToolError(name, "failed", "run() must return a ToolResult")
    return ToolResult(
        content=result.content[:MAX_CONTENT_CHARS],
        speak=None if result.speak is None else result.speak[:MAX_SPEAK_CHARS],
        data=result.data,
    )


def _apply_aliases(tool, arguments: dict[str, Any]) -> dict[str, Any]:
    aliases = getattr(tool, "aliases", None) or {}
    renamed = dict(arguments)
    for alias, target in aliases.items():
        if alias in renamed and target not in renamed:
            renamed[target] = renamed.pop(alias)
    return renamed


@dataclass(slots=True, frozen=True)
class FastPathHit:
    name: str
    arguments: dict[str, Any]
    result: ToolResult


class ToolRegistry:
    def __init__(self, config, *, knowledge_retriever=None) -> None:
        self.config = config
        self.context = ToolContext(config=config, knowledge_retriever=knowledge_retriever)
        self.errors: dict[str, str] = {}
        self.tools = {}
        for name in enabled_tool_names(config):
            try:
                module, _, _ = load_module(config, name)
                tool = create_tool(module, name, self.context)
            except Exception as exc:  # One broken tool must not disable the rest.
                self.errors[name] = f"{type(exc).__name__}: {exc}"
                continue
            if tool is not None:
                self.tools[name] = tool

    @property
    def model_tools(self) -> list:
        return [tool for tool in self.tools.values() if tool.model_callable]

    def fast_path(self, text: str) -> FastPathHit | None:
        """Run the first tool, in configured order, whose trigger matches."""
        for name, tool in self.tools.items():
            match = getattr(tool, "match", None)
            try:
                arguments = match(text) if callable(match) else None
            except Exception as exc:
                # A matcher that trips on odd input must not fail the turn.
                logger.warning("Tool %s could not read the request (%s)", name, type(exc).__name__)
                continue
            if arguments is None:
                continue
            try:
                result = run_tool(tool, arguments)
            except ToolError:
                continue  # A failed instant answer falls through to the model.
            if not result.empty:
                return FastPathHit(name, arguments, result)
        return None

    def run(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None or not tool.model_callable:
            raise ToolError(name, "unavailable", "no such tool is enabled")
        return run_tool(tool, arguments)

    def instruction_text(self) -> str | None:
        """What the model is told about the tools it can call.

        Kept short on purpose: this text starts every prompt, and Gemma's
        512-token sliding attention window only lets the cached prompt be reused
        while the prompt and answer fit inside it.
        """
        tools = self.model_tools
        if not tools:
            return None
        lines = [
            "You may make at most one tool call before answering the user.",
            "Use a tool only when it is needed to answer accurately.",
            "Answer general knowledge questions from what you know, without a tool.",
            "Never say you set, changed or cancelled something unless a tool did it.",
            "",
            "Available tools:",
        ]
        for tool in tools:
            example = argument_example(tool.parameters)
            optional = [name for name in example if name not in tool.parameters.get("required", [])]
            note = f" ({', '.join(optional)} optional)" if optional else ""
            lines.append(f"- name: {tool.name}")
            lines.append(f"- arguments: {json.dumps(example)}{note}")
            lines.append(f"- use when {tool.description}")
        lines += [
            "",
            "If you need a tool, reply with JSON only in exactly this shape:",
            '{"tool_call":{"name":"<tool_name>","arguments":{...}}}',
            "",
            "Otherwise answer the user normally in plain text.",
            "Do not explain the JSON format.",
            "Do not call a tool if the provided snippets already answer the question.",
        ]
        return "\n".join(lines)


def argument_example(schema: dict[str, Any]) -> dict[str, str]:
    """Example arguments for the model: {"duration": "10 minutes"}.

    Taken from "such as ..." in each description, else the description itself in
    angle brackets; choices are listed as "a|b".
    """
    example = {}
    for name, spec in schema.get("properties", {}).items():
        if "enum" in spec:
            example[name] = "|".join(map(str, spec["enum"]))
            continue
        description = spec.get("description") or spec["type"]
        _, found, sample = description.partition("such as ")
        example[name] = sample.strip() if found else f"<{description}>"
    return example
