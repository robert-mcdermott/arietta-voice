from __future__ import annotations

import importlib
import importlib.util
import json
import keyword
import re
from pathlib import Path
from types import ModuleType
from typing import Any

from arietta_voice.config import AppConfig
from arietta_voice.knowledge import KnowledgeRetriever
from arietta_voice.tools.base import AssistantTool, ToolContext
from arietta_voice.tools.registry import _tool_module_name


EXCLUDED_TOOL_MODULES = {"__init__", "base", "registry", "tool_template"}
MODULE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
NEW_TOOL_NAME_PATTERN = re.compile(r"[^a-z0-9_]+")


class ToolAdmin:
    def __init__(
        self,
        config: AppConfig,
        *,
        config_path: Path | None = None,
        knowledge_retriever: KnowledgeRetriever | None = None,
        tools_directory: Path | None = None,
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.knowledge_retriever = knowledge_retriever
        self.tools_directory = tools_directory or Path(__file__).resolve().parent / "tools"

    def inventory(self) -> dict[str, Any]:
        enabled = _unique_tool_names(self.config.tools.enabled)
        enabled_set = set(enabled)
        available = self.available_tool_names()
        configured_or_available = [
            *enabled,
            *(name for name in available if name not in enabled_set),
        ]

        records = [self.validate_tool(name) for name in configured_or_available]
        for record in records:
            record["enabled"] = record["configured_name"] in enabled_set

        enabled_errors = [
            {
                "configured_name": record["configured_name"],
                "error": str(record["error"]),
            }
            for record in records
            if record["enabled"] and record["error"]
        ]
        load_errors = [
            {
                "configured_name": record["configured_name"],
                "error": str(record["error"]),
            }
            for record in records
            if record["error"]
        ]
        loaded = [
            {
                "configured_name": record["configured_name"],
                "name": record["tool_name"],
                "module": record["module"],
                "class": record["class"],
                "file": record["file"],
            }
            for record in records
            if record["enabled"] and record["valid"]
        ]
        return {
            "enabled": enabled,
            "available": available,
            "tools_directory": str(self.tools_directory),
            "tools": records,
            "loaded": loaded,
            "errors": enabled_errors,
            "load_errors": load_errors,
        }

    def available_tool_names(self) -> list[str]:
        if not self.tools_directory.exists():
            return []
        names = []
        for path in self.tools_directory.glob("*.py"):
            name = path.stem
            if name in EXCLUDED_TOOL_MODULES or name.startswith("_"):
                continue
            names.append(name)
        return sorted(names)

    def validate_tool(self, tool_name: str) -> dict[str, Any]:
        record = {
            "configured_name": str(tool_name).strip(),
            "enabled": False,
            "available": False,
            "valid": False,
            "module": None,
            "file": None,
            "class": None,
            "tool_name": None,
            "source": "external" if "." in str(tool_name) else "local",
            "error": None,
        }
        try:
            normalized = _normalize_configured_tool_name(tool_name)
            record["enabled"] = normalized in _unique_tool_names(self.config.tools.enabled)
            module, module_name, file_path = self._load_module(normalized)
            record["configured_name"] = normalized
            record["module"] = module_name
            record["file"] = str(file_path) if file_path is not None else None
            record["available"] = file_path is not None or "." in normalized
            tool = self._create_tool(module, normalized)
            record["class"] = type(tool).__name__
            record["tool_name"] = tool.name
            record["valid"] = True
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    def set_enabled(self, enabled_tools: list[str]) -> dict[str, Any]:
        enabled = _unique_tool_names(enabled_tools)
        available = set(self.available_tool_names())
        for tool_name in enabled:
            if "." not in tool_name and tool_name not in available:
                raise ValueError(f"Tool '{tool_name}' is not available in {self.tools_directory}.")
            record = self.validate_tool(tool_name)
            if not record["valid"]:
                raise ValueError(f"Tool '{tool_name}' is not valid: {record['error']}")

        self.config.tools.enabled = tuple(enabled)
        self._persist_enabled_tools(enabled)
        return self.inventory()

    def probe_tool(self, tool_name: str, text: str) -> dict[str, Any]:
        normalized = _normalize_configured_tool_name(tool_name)
        module, module_name, file_path = self._load_module(normalized)
        tool = self._create_tool(module, normalized)
        enabled = normalized in _unique_tool_names(self.config.tools.enabled)
        response = tool.maybe_handle(text.strip())
        if response is None:
            return {
                "configured_name": normalized,
                "enabled": enabled,
                "module": module_name,
                "file": str(file_path) if file_path is not None else None,
                "matched": False,
                "tool": None,
                "content": None,
                "bypass_model": None,
                "context": None,
            }
        return {
            "configured_name": normalized,
            "enabled": enabled,
            "module": module_name,
            "file": str(file_path) if file_path is not None else None,
            "matched": True,
            "tool": response.name,
            "content": response.content,
            "bypass_model": response.bypass_model,
            "context": response.context,
        }

    def create_tool(
        self,
        *,
        module_name: str,
        trigger_phrase: str,
        response: str,
        enable: bool = True,
    ) -> dict[str, Any]:
        normalized = normalize_new_tool_module_name(module_name)
        self.tools_directory.mkdir(parents=True, exist_ok=True)
        path = self.tools_directory / f"{normalized}.py"
        if path.exists():
            raise FileExistsError(f"Tool module '{normalized}' already exists.")

        path.write_text(
            _render_tool_template(
                module_name=normalized,
                trigger_phrase=trigger_phrase.strip() or normalized.replace("_", " "),
                response=response.strip() or "The custom tool is wired up.",
            ),
            encoding="utf-8",
        )
        importlib.invalidate_caches()

        enabled = list(self.config.tools.enabled)
        if enable and normalized not in enabled:
            enabled.append(normalized)
            self.config.tools.enabled = tuple(enabled)
            self._persist_enabled_tools(enabled)

        record = self.validate_tool(normalized)
        return {
            "created": True,
            "module_name": normalized,
            "file": str(path),
            "enabled": normalized in self.config.tools.enabled,
            "tool": record,
            "inventory": self.inventory(),
        }

    def reload(self) -> dict[str, Any]:
        importlib.invalidate_caches()
        return self.inventory()

    def _load_module(self, tool_name: str) -> tuple[ModuleType, str, Path | None]:
        module_name = _tool_module_name(tool_name)
        file_path = None
        if "." not in tool_name:
            candidate = self.tools_directory / f"{tool_name}.py"
            if candidate.exists():
                file_path = candidate
                module = _load_module_from_file(module_name, candidate)
                return module, module_name, file_path

        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                raise ValueError(f"Tool '{tool_name}' could not be imported from '{module_name}'.") from exc
            raise
        module_file = getattr(module, "__file__", None)
        return module, module_name, Path(module_file) if module_file else None

    def _create_tool(self, module: ModuleType, tool_name: str) -> AssistantTool:
        factory = getattr(module, "create_tool", None)
        if not callable(factory):
            raise ValueError(f"Tool '{tool_name}' must expose a callable create_tool(context).")

        tool = factory(
            ToolContext(
                config=self.config,
                knowledge_retriever=self.knowledge_retriever
                if self.config.knowledge.enabled
                else None,
            )
        )
        if not callable(getattr(tool, "maybe_handle", None)):
            raise ValueError(f"Tool '{tool_name}' must define maybe_handle(user_text).")
        tool_id = getattr(tool, "name", None)
        if not isinstance(tool_id, str) or not tool_id.strip():
            raise ValueError(f"Tool '{tool_name}' must define a non-empty name.")
        return tool

    def _persist_enabled_tools(self, enabled: list[str]) -> None:
        path = self.config_path
        if path is None:
            path = self.config.project_root / "config" / "arietta_voice.toml"
        _persist_tools_enabled(path, enabled)


def normalize_new_tool_module_name(value: str) -> str:
    normalized = NEW_TOOL_NAME_PATTERN.sub("_", value.strip().lower()).strip("_")
    normalized = re.sub(r"_+", "_", normalized)
    if normalized and normalized[0].isdigit():
        normalized = f"tool_{normalized}"
    if not normalized:
        raise ValueError("Tool module name cannot be blank.")
    if keyword.iskeyword(normalized):
        normalized = f"{normalized}_tool"
    if not re.fullmatch(r"[a-z_][a-z0-9_]{0,63}", normalized):
        raise ValueError("Tool module name must be 64 characters or fewer.")
    return normalized


def _normalize_configured_tool_name(value: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("Tool name cannot be blank.")
    if "/" in normalized or "\\" in normalized:
        raise ValueError("Tool name must be a module name, not a path.")
    if not MODULE_NAME_PATTERN.fullmatch(normalized):
        raise ValueError("Tool name must be a valid Python module name.")
    return normalized


def _unique_tool_names(values: list[str] | tuple[str, ...]) -> list[str]:
    names = []
    seen: set[str] = set()
    for value in values:
        name = _normalize_configured_tool_name(value)
        if name in seen:
            continue
        names.append(name)
        seen.add(name)
    return names


def _load_module_from_file(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load tool module from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _persist_tools_enabled(path: Path, enabled: list[str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    replacement = _format_tools_enabled(enabled)
    section_start = _find_section(lines, "tools")
    if section_start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append("[tools]")
        lines.extend(replacement)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    section_end = _find_next_section(lines, section_start)
    enabled_start = _find_key_in_section(lines, section_start, section_end, "enabled")
    if enabled_start is None:
        lines[section_start + 1 : section_start + 1] = replacement
    else:
        enabled_end = _find_array_end(lines, enabled_start)
        lines[enabled_start:enabled_end] = replacement
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_tools_enabled(enabled: list[str]) -> list[str]:
    if not enabled:
        return ["enabled = []"]
    return [
        "enabled = [",
        *(f"    {json.dumps(name, ensure_ascii=True)}," for name in enabled),
        "]",
    ]


def _find_section(lines: list[str], name: str) -> int | None:
    header = f"[{name}]"
    for index, line in enumerate(lines):
        if line.strip() == header:
            return index
    return None


def _find_next_section(lines: list[str], section_start: int) -> int:
    for index in range(section_start + 1, len(lines)):
        stripped = lines[index].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            return index
    return len(lines)


def _find_key_in_section(
    lines: list[str],
    section_start: int,
    section_end: int,
    key: str,
) -> int | None:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    for index in range(section_start + 1, section_end):
        if pattern.match(lines[index]):
            return index
    return None


def _find_array_end(lines: list[str], key_start: int) -> int:
    first_line = lines[key_start]
    if "[" not in first_line or "]" in first_line:
        return key_start + 1
    for index in range(key_start + 1, len(lines)):
        if "]" in lines[index]:
            return index + 1
    return key_start + 1


def _render_tool_template(*, module_name: str, trigger_phrase: str, response: str) -> str:
    class_name = "".join(part.capitalize() for part in module_name.split("_") if part) + "Tool"
    trigger_literal = json.dumps(trigger_phrase, ensure_ascii=True)
    response_literal = json.dumps(response, ensure_ascii=True)
    name_literal = json.dumps(module_name, ensure_ascii=True)
    return f'''from __future__ import annotations

from arietta_voice.models import ToolResponse
from arietta_voice.tools.base import ToolContext


TRIGGER_PHRASE = {trigger_literal}
RESPONSE_TEXT = {response_literal}


class {class_name}:
    name = {name_literal}

    def maybe_handle(self, user_text: str) -> ToolResponse | None:
        if TRIGGER_PHRASE.lower() not in user_text.lower():
            return None
        return ToolResponse(name=self.name, content=RESPONSE_TEXT)


def create_tool(context: ToolContext) -> {class_name}:
    del context
    return {class_name}()
'''
