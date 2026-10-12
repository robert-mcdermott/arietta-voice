from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import keyword
import re
from pathlib import Path
from types import ModuleType
from typing import Any

from arietta_voice.config import AppConfig
from arietta_voice.private_config import update_private_section
from arietta_voice.knowledge import KnowledgeRetriever
from arietta_voice.model_tools.base import ModelInvokedTool, ModelToolContext
from arietta_voice.model_tools.registry import ModelToolRegistry
from arietta_voice.model_tools.registry import _tool_module_name as _model_tool_module_name
from arietta_voice.tools.base import AssistantTool, ToolContext
from arietta_voice.tools.registry import _tool_module_name


EXCLUDED_TOOL_MODULES = {"__init__", "base", "registry", "tool_template"}
EXCLUDED_MODEL_TOOL_MODULES = {"__init__", "base", "registry", "tool_template"}
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

        records = [self.inspect_tool(name) for name in configured_or_available]
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
        return {
            "enabled": enabled,
            "available": available,
            "tools_directory": str(self.tools_directory),
            "tools": records,
            "loaded": [],
            "inventory_mode": "static",
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

    def inspect_tool(self, tool_name: str) -> dict[str, Any]:
        return _inspect_tool_source(self.tools_directory, tool_name, self.config.tools.enabled, _tool_module_name)

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
            record["configured_name"] = normalized
            record["enabled"] = normalized in _unique_tool_names(self.config.tools.enabled)
            if "." not in normalized:
                candidate = self.tools_directory / f"{normalized}.py"
                if candidate.exists():
                    record["available"] = True
                    record["file"] = str(candidate.resolve())
            module, module_name, file_path = self._load_module(normalized)
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

        self._persist_enabled_tools(enabled)
        self.config.tools.enabled = tuple(enabled)
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
            self._persist_enabled_tools(enabled)
            self.config.tools.enabled = tuple(enabled)

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

    def read_source(self, tool_name: str) -> dict[str, Any]:
        normalized, path = _resolve_local_tool_source(self.tools_directory, tool_name)
        return {
            "configured_name": normalized,
            "enabled": normalized in _unique_tool_names(self.config.tools.enabled),
            "file": str(path),
            "content": path.read_text(encoding="utf-8"),
            "tool": self.inspect_tool(normalized),
        }

    def write_source(self, tool_name: str, content: str) -> dict[str, Any]:
        normalized, path = _resolve_local_tool_source(self.tools_directory, tool_name)
        path.write_text(content, encoding="utf-8")
        importlib.invalidate_caches()
        return {
            "saved": True,
            "configured_name": normalized,
            "enabled": normalized in _unique_tool_names(self.config.tools.enabled),
            "file": str(path),
            "tool": self.validate_tool(normalized),
        }

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
        update_private_section(path, "tools", {"enabled": enabled})


class ModelToolAdmin:
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
        self.tools_directory = tools_directory or Path(__file__).resolve().parent / "model_tools"

    def inspect_tool(self, tool_name: str) -> dict[str, Any]:
        return _inspect_tool_source(self.tools_directory, tool_name, self.config.model_tools.enabled, _model_tool_module_name)

    def inventory(self) -> dict[str, Any]:
        enabled = _unique_tool_names(self.config.model_tools.enabled)
        enabled_set = set(enabled)
        available = self.available_tool_names()
        configured_or_available = [
            *enabled,
            *(name for name in available if name not in enabled_set),
        ]

        records = [self.inspect_tool(name) for name in configured_or_available]
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
        return {
            "enabled": enabled,
            "available": available,
            "tools_directory": str(self.tools_directory),
            "tools": records,
            "loaded": [],
            "inventory_mode": "static",
            "errors": enabled_errors,
            "load_errors": load_errors,
            "instruction_text": None,
            "instruction_error": None,
        }

    def available_tool_names(self) -> list[str]:
        if not self.tools_directory.exists():
            return []
        names = []
        for path in self.tools_directory.glob("*.py"):
            name = path.stem
            if name in EXCLUDED_MODEL_TOOL_MODULES or name.startswith("_"):
                continue
            names.append(name)
        return sorted(names)

    def validate_tool(self, tool_name: str) -> dict[str, Any]:
        record = {
            "configured_name": str(tool_name).strip(),
            "enabled": False,
            "available": False,
            "valid": False,
            "active": False,
            "module": None,
            "file": None,
            "class": None,
            "tool_name": None,
            "description": None,
            "arguments_example": None,
            "source": "external" if "." in str(tool_name) else "local",
            "error": None,
        }
        try:
            normalized = _normalize_configured_tool_name(tool_name)
            record["configured_name"] = normalized
            record["enabled"] = normalized in _unique_tool_names(self.config.model_tools.enabled)
            if "." not in normalized:
                candidate = self.tools_directory / f"{normalized}.py"
                if candidate.exists():
                    record["available"] = True
                    record["file"] = str(candidate.resolve())
            module, module_name, file_path = self._load_module(normalized)
            record["module"] = module_name
            record["file"] = str(file_path) if file_path is not None else None
            record["available"] = file_path is not None or "." in normalized
            tool = self._create_tool(module, normalized)
            record["valid"] = True
            if tool is None:
                return record
            record["active"] = True
            record["class"] = type(tool).__name__
            record["tool_name"] = tool.name
            record["description"] = tool.description
            record["arguments_example"] = tool.arguments_example
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    def set_enabled(self, enabled_tools: list[str]) -> dict[str, Any]:
        enabled = _unique_tool_names(enabled_tools)
        available = set(self.available_tool_names())
        for tool_name in enabled:
            if "." not in tool_name and tool_name not in available:
                raise ValueError(
                    f"Model tool '{tool_name}' is not available in {self.tools_directory}."
                )
            record = self.validate_tool(tool_name)
            if not record["valid"]:
                raise ValueError(f"Model tool '{tool_name}' is not valid: {record['error']}")

        self._persist_enabled_tools(enabled)
        self.config.model_tools.enabled = tuple(enabled)
        return self.inventory()

    def invoke_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        normalized = _normalize_configured_tool_name(tool_name)
        module, module_name, file_path = self._load_module(normalized)
        tool = self._create_tool(module, normalized)
        enabled = normalized in _unique_tool_names(self.config.model_tools.enabled)
        if tool is None:
            return {
                "configured_name": normalized,
                "enabled": enabled,
                "active": False,
                "module": module_name,
                "file": str(file_path) if file_path is not None else None,
                "invoked": False,
                "tool": None,
                "arguments": arguments,
                "arguments_example": {},
                "followup_context": None,
                "details": {},
                "error": "Tool factory returned None for the current configuration.",
            }
        result = tool.invoke(arguments)
        if result is None:
            return {
                "configured_name": normalized,
                "enabled": enabled,
                "active": True,
                "module": module_name,
                "file": str(file_path) if file_path is not None else None,
                "invoked": False,
                "tool": tool.name,
                "arguments": arguments,
                "arguments_example": tool.arguments_example,
                "followup_context": None,
                "details": {},
                "error": None,
            }
        return {
            "configured_name": normalized,
            "enabled": enabled,
            "active": True,
            "module": module_name,
            "file": str(file_path) if file_path is not None else None,
            "invoked": True,
            "tool": tool.name,
            "arguments": arguments,
            "arguments_example": tool.arguments_example,
            "followup_context": result.followup_context,
            "details": result.details,
            "error": None,
        }

    def create_tool(
        self,
        *,
        module_name: str,
        description: str,
        arguments_example: dict[str, Any],
        followup_context: str,
        enable: bool = True,
    ) -> dict[str, Any]:
        normalized = normalize_new_tool_module_name(module_name)
        self.tools_directory.mkdir(parents=True, exist_ok=True)
        path = self.tools_directory / f"{normalized}.py"
        if path.exists():
            raise FileExistsError(f"Model tool module '{normalized}' already exists.")

        path.write_text(
            _render_model_tool_template(
                module_name=normalized,
                description=description.strip()
                or "you need a custom model-called tool result before answering",
                arguments_example=arguments_example,
                followup_context=followup_context.strip()
                or "Custom model tool result. Use this result when composing the final answer.",
            ),
            encoding="utf-8",
        )
        importlib.invalidate_caches()

        enabled = list(self.config.model_tools.enabled)
        if enable and normalized not in enabled:
            enabled.append(normalized)
            self._persist_enabled_tools(enabled)
            self.config.model_tools.enabled = tuple(enabled)

        record = self.validate_tool(normalized)
        return {
            "created": True,
            "module_name": normalized,
            "file": str(path),
            "enabled": normalized in self.config.model_tools.enabled,
            "tool": record,
            "inventory": self.inventory(),
        }

    def reload(self) -> dict[str, Any]:
        importlib.invalidate_caches()
        return self.inventory()

    def read_source(self, tool_name: str) -> dict[str, Any]:
        normalized, path = _resolve_local_tool_source(self.tools_directory, tool_name)
        return {
            "configured_name": normalized,
            "enabled": normalized in _unique_tool_names(self.config.model_tools.enabled),
            "file": str(path),
            "content": path.read_text(encoding="utf-8"),
            "tool": self.inspect_tool(normalized),
        }

    def write_source(self, tool_name: str, content: str) -> dict[str, Any]:
        normalized, path = _resolve_local_tool_source(self.tools_directory, tool_name)
        path.write_text(content, encoding="utf-8")
        importlib.invalidate_caches()
        return {
            "saved": True,
            "configured_name": normalized,
            "enabled": normalized in _unique_tool_names(self.config.model_tools.enabled),
            "file": str(path),
            "tool": self.validate_tool(normalized),
        }

    def instruction_text(self) -> str | None:
        registry = ModelToolRegistry(
            self.config,
            knowledge_retriever=self.knowledge_retriever if self.config.knowledge.enabled else None,
        )
        return registry.instruction_text()

    def _load_module(self, tool_name: str) -> tuple[ModuleType, str, Path | None]:
        module_name = _model_tool_module_name(tool_name)
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
                raise ValueError(
                    f"Model tool '{tool_name}' could not be imported from '{module_name}'."
                ) from exc
            raise
        module_file = getattr(module, "__file__", None)
        return module, module_name, Path(module_file) if module_file else None

    def _create_tool(self, module: ModuleType, tool_name: str) -> ModelInvokedTool | None:
        factory = getattr(module, "create_tool", None)
        if not callable(factory):
            raise ValueError(f"Model tool '{tool_name}' must expose a callable create_tool(context).")

        tool = factory(
            ModelToolContext(
                config=self.config,
                knowledge_retriever=self.knowledge_retriever
                if self.config.knowledge.enabled
                else None,
            )
        )
        if tool is None:
            return None
        if not callable(getattr(tool, "invoke", None)):
            raise ValueError(f"Model tool '{tool_name}' must define invoke(arguments).")
        if not isinstance(getattr(tool, "name", None), str) or not tool.name.strip():
            raise ValueError(f"Model tool '{tool_name}' must define a non-empty name.")
        if not isinstance(getattr(tool, "description", None), str) or not tool.description.strip():
            raise ValueError(f"Model tool '{tool_name}' must define a description.")
        arguments_example = getattr(tool, "arguments_example", None)
        if not isinstance(arguments_example, dict):
            raise ValueError(f"Model tool '{tool_name}' must define arguments_example as a dict.")
        return tool

    def _persist_enabled_tools(self, enabled: list[str]) -> None:
        path = self.config_path
        if path is None:
            path = self.config.project_root / "config" / "arietta_voice.toml"
        update_private_section(path, "model_tools", {"enabled": enabled})


def _inspect_tool_source(directory: Path, tool_name: str, enabled, module_name_for) -> dict[str, Any]:
    """Read syntax/literal metadata only. Never import plugins (including parents)."""
    normalized = _normalize_configured_tool_name(tool_name)
    record = {
        "configured_name": normalized, "enabled": normalized in enabled,
        "available": False, "valid": None, "active": None,
        "validation": "not_run", "syntax_valid": None,
        "module": module_name_for(normalized), "file": None, "class": None,
        "tool_name": None, "description": None, "arguments_example": None,
        "source": "external" if "." in normalized else "local", "error": None,
    }
    if "." in normalized:
        # find_spec can import an external package's __init__; defer even
        # discovery of external plugins to an explicit validation action.
        return record
    try:
        _, path = _resolve_local_tool_source(directory, normalized)
        record.update(available=True, file=str(path))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        record["syntax_valid"] = True
        if not any(isinstance(node, ast.FunctionDef) and node.name == "create_tool" for node in tree.body):
            raise ValueError("No top-level create_tool factory found; explicit validation is required")
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                targets = item.targets if isinstance(item, ast.Assign) else [item.target] if isinstance(item, ast.AnnAssign) else []
                for target in targets:
                    if not isinstance(target, ast.Name) or target.id not in {"name", "description", "arguments_example"}:
                        continue
                    try:
                        value = ast.literal_eval(item.value)
                    except (ValueError, TypeError):
                        continue
                    if (target.id == "arguments_example" and isinstance(value, dict)) or (target.id != "arguments_example" and isinstance(value, str)):
                        record["tool_name" if target.id == "name" else target.id] = value
                        record["class"] = node.name
    except (OSError, ValueError, SyntaxError) as exc:
        record["error"] = f"Static inspection failed ({type(exc).__name__})"
        record["syntax_valid"] = False
    return record


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


def _resolve_local_tool_source(tools_directory: Path, tool_name: str) -> tuple[str, Path]:
    normalized = _normalize_configured_tool_name(tool_name)
    if "." in normalized:
        raise ValueError("Only local tool modules can be edited from the admin console.")
    root = tools_directory.resolve()
    path = (root / f"{normalized}.py").resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Tool source path must stay inside the tool directory.") from exc
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Tool source for '{normalized}' was not found.")
    return normalized, path


def _load_module_from_file(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load tool module from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _render_model_tool_template(
    *,
    module_name: str,
    description: str,
    arguments_example: dict[str, Any],
    followup_context: str,
) -> str:
    class_name = "".join(part.capitalize() for part in module_name.split("_") if part) + "ModelTool"
    name_literal = json.dumps(module_name, ensure_ascii=True)
    description_literal = json.dumps(description, ensure_ascii=True)
    arguments_literal = json.dumps(arguments_example, ensure_ascii=True, indent=4)
    context_literal = json.dumps(followup_context, ensure_ascii=True)
    return f'''from __future__ import annotations

import json

from arietta_voice.model_tools.base import ModelToolContext, ModelToolExecution


FOLLOWUP_CONTEXT = {context_literal}


class {class_name}:
    name = {name_literal}
    description = {description_literal}
    arguments_example = {arguments_literal}

    def invoke(self, arguments: dict[str, object]) -> ModelToolExecution | None:
        return ModelToolExecution(
            followup_context=(
                f"{{FOLLOWUP_CONTEXT}}\\n"
                f"Tool arguments: {{json.dumps(arguments, sort_keys=True, ensure_ascii=True)}}"
            ),
            details={{"arguments": arguments}},
        )


def create_tool(context: ModelToolContext) -> {class_name}:
    del context
    return {class_name}()
'''
