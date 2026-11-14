"""Console tool management: list, inspect, test, validate, create and edit tools.

Listing and source views read files as text and never import them, so opening
the console cannot run a tool. Validate, Test and the instruction preview import
and run tool code on purpose. Only user tools (files in the tools directory) can
be created or edited, and only when ``admin.allow_source_editing`` is on;
built-in tools are read-only.
"""

from __future__ import annotations

import ast
import importlib
import json
import keyword
import re
from pathlib import Path
from typing import Any

from arietta_voice.private_config import update_private_section
from arietta_voice.tools.base import ToolContext, ToolError
from arietta_voice.tools.registry import (
    TOOL_NAME,
    ToolRegistry,
    builtin_names,
    create_tool,
    describe_problems,
    enabled_tool_names,
    load_module,
    run_tool,
    user_tool_path,
)

INSPECTED_ATTRIBUTES = (
    "name",
    "description",
    "parameters",
    "triggers",
    "model_callable",
    "consequential",
    "timeout_s",
)


class ToolAdmin:
    def __init__(self, config, *, config_path: Path | None = None, knowledge_retriever=None):
        self.config = config
        self.config_path = config_path
        self.knowledge_retriever = knowledge_retriever

    @property
    def tools_directory(self) -> Path:
        return Path(self.config.tools.directory)

    def user_names(self) -> list[str]:
        if not self.tools_directory.is_dir():
            return []
        builtin = set(builtin_names())
        return sorted(
            p.stem
            for p in self.tools_directory.glob("*.py")
            if TOOL_NAME.fullmatch(p.stem) and p.stem not in builtin
        )

    def inventory(self) -> dict[str, Any]:
        enabled = enabled_tool_names(self.config)
        known = [*builtin_names(), *self.user_names()]
        names = [*enabled, *(name for name in known if name not in enabled)]
        records = [self.inspect(name) for name in names]
        return {
            "enabled": enabled,
            "tools_directory": str(self.tools_directory),
            "tools": records,
            "errors": [
                {"name": r["name"], "error": r["error"]}
                for r in records
                if r["enabled"] and r["error"]
            ],
        }

    def inspect(self, name: str) -> dict[str, Any]:
        """Describe a tool from its source text, without importing it."""
        record = {
            "name": name,
            "enabled": name in enabled_tool_names(self.config),
            "origin": "missing",
            "file": None,
            "editable": False,
            "validation": "not_run",
            "error": None,
            "description": None,
            "parameters": None,
            "triggers": [],
            "model_callable": True,
            "consequential": False,
            "timeout_s": None,
        }
        try:
            path, origin = self._source_path(name)
        except (ValueError, FileNotFoundError) as exc:
            record["error"] = str(exc)
            return record
        record.update(origin=origin, file=str(path), editable=origin == "user")
        try:
            record.update(_static_metadata(path.read_text(encoding="utf-8")))
        except (OSError, SyntaxError, ValueError) as exc:
            record["error"] = f"Could not read the tool definition ({type(exc).__name__})"
        return record

    def validate(self, name: str) -> dict[str, Any]:
        """Import and build the tool, then check its definition."""
        record = self.inspect(name)
        try:
            tool = self._load(name)
        except Exception as exc:
            record.update(validation="failed", error=f"{type(exc).__name__}: {exc}")
            return record
        if tool is None:
            record.update(validation="unavailable", error="Not available with the current settings")
            return record
        record.update(
            validation="valid",
            error=None,
            description=tool.description,
            parameters=tool.parameters,
            triggers=list(tool.triggers),
            model_callable=bool(tool.model_callable),
            consequential=bool(getattr(tool, "consequential", False)),
            timeout_s=tool.timeout_s,
        )
        return record

    def test(self, name: str, *, text: str | None = None, arguments: dict | None = None):
        """Run one tool: on a sample request through its triggers, or with arguments."""
        tool = self._load(name)
        if tool is None:
            raise ValueError(f"Tool {name!r} is not available with the current settings")
        if text is not None:
            arguments = tool.match(text.strip()) if hasattr(tool, "match") else None
            if arguments is None:
                return {"name": name, "matched": False, "arguments": None, "result": None}
        try:
            result = run_tool(tool, arguments or {})
        except ToolError as error:
            return {
                "name": name,
                "matched": text is not None,
                "arguments": arguments,
                "result": None,
                "error": {"kind": error.kind, "message": str(error)},
            }
        return {
            "name": name,
            "matched": text is not None,
            "arguments": arguments,
            "result": {"content": result.content, "speak": result.speak, "data": result.data},
        }

    def probe(self, text: str) -> dict[str, Any]:
        """Which enabled tool would answer this request instantly, if any."""
        hit = self._registry().fast_path(text.strip())
        if hit is None:
            return {"matched": False, "name": None, "arguments": None, "result": None}
        result = hit.result
        return {
            "matched": True,
            "name": hit.name,
            "arguments": hit.arguments,
            "answers_directly": bool(result.speak),
            "result": {"content": result.content, "speak": result.speak, "data": result.data},
        }

    def instruction_text(self) -> str | None:
        return self._registry().instruction_text()

    def set_enabled(self, names: list[str]) -> dict[str, Any]:
        enabled = []
        for name in names:
            name = str(name).strip()
            if not name or name in enabled:
                continue
            record = self.validate(name)
            if record["validation"] == "failed" or record["origin"] == "missing":
                raise ValueError(f"Tool {name!r} cannot be enabled: {record['error']}")
            enabled.append(name)
        path = self._config_path()
        update_private_section(path, "tools", {"enabled": enabled})
        # The old separate list is folded into [tools]; clear it so it cannot re-add tools.
        update_private_section(path, "model_tools", {"enabled": []})
        self.config.tools.enabled = tuple(enabled)
        self.config.model_tools.enabled = ()
        return self.inventory()

    def create(
        self, *, name: str, description: str, trigger: str = "", response: str = "", enable=True
    ) -> dict[str, Any]:
        name = normalize_tool_name(name)
        if name in builtin_names():
            raise FileExistsError(f"{name!r} is a built-in tool")
        path = user_tool_path(self.config, name)
        if path.exists():
            raise FileExistsError(f"Tool {name!r} already exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            render_tool_template(
                name=name,
                description=description.strip() or f"the user asks about {name.replace('_', ' ')}",
                trigger=trigger.strip(),
                response=response.strip() or "The new tool is wired up.",
            ),
            encoding="utf-8",
        )
        importlib.invalidate_caches()
        if enable and name not in enabled_tool_names(self.config):
            self.set_enabled([*enabled_tool_names(self.config), name])
        return {"created": True, "name": name, "file": str(path), "tool": self.validate(name)}

    def read_source(self, name: str) -> dict[str, Any]:
        path, origin = self._source_path(name)
        return {
            "name": name,
            "origin": origin,
            "editable": origin == "user",
            "file": str(path),
            "content": path.read_text(encoding="utf-8"),
            "tool": self.inspect(name),
        }

    def write_source(self, name: str, content: str) -> dict[str, Any]:
        path, origin = self._source_path(name)
        if origin != "user":
            raise ValueError("Built-in tools are read-only; copy one into the tools directory")
        path.write_text(content, encoding="utf-8")
        importlib.invalidate_caches()
        return {"saved": True, "name": name, "file": str(path), "tool": self.validate(name)}

    def reload(self) -> dict[str, Any]:
        importlib.invalidate_caches()
        return self.inventory()

    def _registry(self) -> ToolRegistry:
        knowledge = self.knowledge_retriever if self.config.knowledge.enabled else None
        return ToolRegistry(self.config, knowledge_retriever=knowledge)

    def _load(self, name: str):
        module, _, _ = load_module(self.config, name)
        knowledge = self.knowledge_retriever if self.config.knowledge.enabled else None
        tool = create_tool(module, name, ToolContext(self.config, knowledge))
        if tool is not None and describe_problems(tool, name):
            raise ValueError("; ".join(describe_problems(tool, name)))
        return tool

    def _source_path(self, name: str) -> tuple[Path, str]:
        if not TOOL_NAME.fullmatch(name):
            raise ValueError(f"Tool names are lowercase letters, digits and underscores: {name!r}")
        if name in builtin_names():
            package = importlib.import_module("arietta_voice.tools.builtin")
            return Path(package.__file__).parent / f"{name}.py", "builtin"
        path = user_tool_path(self.config, name)
        if not path.is_file():
            raise FileNotFoundError(f"No tool file for {name!r} in {self.tools_directory}")
        return path, "user"

    def _config_path(self) -> Path:
        return self.config_path or self.config.project_root / "config" / "arietta_voice.toml"


def _static_metadata(source: str) -> dict[str, Any]:
    """Literal class attributes of the tool class (the one that defines ``name``)."""
    tree = ast.parse(source)
    found: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        values = {}
        for item in node.body:
            if isinstance(item, ast.Assign):
                targets, value = item.targets, item.value
            elif isinstance(item, ast.AnnAssign) and item.value is not None:
                targets, value = [item.target], item.value
            else:
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id in INSPECTED_ATTRIBUTES:
                    try:
                        values[target.id] = ast.literal_eval(value)
                    except (ValueError, TypeError, SyntaxError):
                        pass
        if "name" in values:
            found = values
    if not any(isinstance(n, ast.FunctionDef) and n.name == "create_tool" for n in tree.body):
        raise ValueError("No top-level create_tool(context) function")
    if "triggers" in found:
        found["triggers"] = list(found["triggers"])
    found.pop("name", None)
    return found


def normalize_tool_name(value: str) -> str:
    name = re.sub(r"[^a-z0-9_]+", "_", value.strip().lower()).strip("_")
    name = re.sub(r"_+", "_", name)
    if name and name[0].isdigit():
        name = f"tool_{name}"
    if keyword.iskeyword(name):
        name = f"{name}_tool"
    if not TOOL_NAME.fullmatch(name):
        raise ValueError("Tool names need 1 to 64 letters, digits or underscores")
    return name


def render_tool_template(*, name: str, description: str, trigger: str, response: str) -> str:
    class_name = "".join(part.capitalize() for part in name.split("_")) + "Tool"
    triggers = f"({json.dumps(trigger)},)" if trigger else "()"
    return f'''"""{name}: created from the Arietta console. Edit freely."""

from arietta_voice.tools import BaseTool, ToolResult


class {class_name}(BaseTool):
    name = "{name}"
    # When the model should call this tool.
    description = {json.dumps(description)}
    # Regular expressions that answer instantly, without the model.
    triggers = {triggers}
    # Arguments the model passes, as a JSON schema object.
    parameters = {{"type": "object", "properties": {{}}}}

    def run(self, arguments):
        # content is for the model to use; speak answers the user directly.
        return ToolResult(content={json.dumps(response)}, speak={json.dumps(response)})


def create_tool(context):
    return {class_name}(context)
'''
