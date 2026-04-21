from __future__ import annotations

import importlib
import json

from arietta_voice.config import AppConfig
from arietta_voice.knowledge import KnowledgeRetriever
from arietta_voice.model_tools.base import ModelInvokedTool, ModelToolContext


MODEL_TOOLS_PACKAGE = "arietta_voice.model_tools"


class ModelToolRegistry:
    def __init__(
        self,
        config: AppConfig,
        *,
        knowledge_retriever: KnowledgeRetriever | None = None,
    ) -> None:
        self.config = config
        self.context = ModelToolContext(config=config, knowledge_retriever=knowledge_retriever)
        self.tools = tuple(self._load_tools(config.model_tools.enabled))
        self._by_name = {tool.name: tool for tool in self.tools}

    def instruction_text(self) -> str | None:
        if not self.tools:
            return None

        lines = [
            "You may make at most one tool call before answering the user.",
            "Use a tool only when it is needed to answer accurately.",
            "",
            "Available tools:",
        ]
        for tool in self.tools:
            lines.append(f"- name: {tool.name}")
            lines.append(f"- arguments: {json.dumps(tool.arguments_example, ensure_ascii=True)}")
            lines.append(f"- use when {tool.description}")
        lines.extend(
            [
                "",
                'If you need a tool, reply with JSON only in exactly this shape:',
                '{"tool_call":{"name":"<tool_name>","arguments":{...}}}',
                "",
                "Otherwise answer the user normally in plain text.",
                "Do not explain the JSON format.",
                "Do not call a tool if the provided snippets already answer the question.",
            ]
        )
        return "\n".join(lines)

    def execute(self, name: str, arguments: dict[str, object]) -> ModelToolExecution | None:
        tool = self._by_name.get(name)
        if tool is None:
            return None
        return tool.invoke(arguments)

    def _load_tools(self, configured_tools: tuple[str, ...]) -> list[ModelInvokedTool]:
        loaded: list[ModelInvokedTool] = []
        seen: set[str] = set()
        for tool_name in configured_tools:
            normalized = tool_name.strip()
            if not normalized or normalized in seen:
                continue
            tool = self._load_tool(normalized)
            if tool is not None:
                loaded.append(tool)
            seen.add(normalized)
        return loaded

    def _load_tool(self, tool_name: str) -> ModelInvokedTool | None:
        module_name = _tool_module_name(tool_name)
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                raise ValueError(
                    f"Configured model tool '{tool_name}' could not be imported from '{module_name}'."
                ) from exc
            raise

        factory = getattr(module, "create_tool", None)
        if not callable(factory):
            raise ValueError(
                f"Configured model tool '{tool_name}' must expose a callable create_tool(context)."
            )

        tool = factory(self.context)
        if tool is None:
            return None
        if not callable(getattr(tool, "invoke", None)):
            raise ValueError(f"Configured model tool '{tool_name}' must define invoke(arguments).")
        if not isinstance(getattr(tool, "name", None), str) or not tool.name.strip():
            raise ValueError(f"Configured model tool '{tool_name}' must define a non-empty name.")
        if not isinstance(getattr(tool, "description", None), str) or not tool.description.strip():
            raise ValueError(f"Configured model tool '{tool_name}' must define a description.")
        arguments_example = getattr(tool, "arguments_example", None)
        if not isinstance(arguments_example, dict):
            raise ValueError(
                f"Configured model tool '{tool_name}' must define arguments_example as a dict."
            )
        return tool


def _tool_module_name(tool_name: str) -> str:
    if "." in tool_name:
        return tool_name
    return f"{MODEL_TOOLS_PACKAGE}.{tool_name}"
