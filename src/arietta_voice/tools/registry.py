from __future__ import annotations

import importlib

from arietta_voice.config import AppConfig
from arietta_voice.knowledge import KnowledgeRetriever
from arietta_voice.models import ToolResponse
from arietta_voice.tools.base import AssistantTool, ToolContext


TOOLS_PACKAGE = "arietta_voice.tools"


class ToolRegistry:
    def __init__(
        self,
        config: AppConfig,
        *,
        knowledge_retriever: KnowledgeRetriever | None = None,
    ) -> None:
        self.config = config
        self.context = ToolContext(
            config=config,
            knowledge_retriever=knowledge_retriever,
        )
        self.tools = tuple(self._load_tools(config.tools.enabled))

    def maybe_handle(self, user_text: str) -> ToolResponse | None:
        for tool in self.tools:
            response = tool.maybe_handle(user_text)
            if response is not None:
                return response
        return None

    def _load_tools(self, configured_tools: tuple[str, ...]) -> list[AssistantTool]:
        loaded: list[AssistantTool] = []
        seen: set[str] = set()
        for tool_name in configured_tools:
            normalized = tool_name.strip()
            if not normalized or normalized in seen:
                continue
            loaded.append(self._load_tool(normalized))
            seen.add(normalized)
        return loaded

    def _load_tool(self, tool_name: str) -> AssistantTool:
        module_name = _tool_module_name(tool_name)
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                raise ValueError(
                    f"Configured tool '{tool_name}' could not be imported from '{module_name}'."
                ) from exc
            raise

        factory = getattr(module, "create_tool", None)
        if not callable(factory):
            raise ValueError(
                f"Configured tool '{tool_name}' must expose a callable create_tool(context)."
            )

        tool = factory(self.context)
        if not callable(getattr(tool, "maybe_handle", None)):
            raise ValueError(f"Configured tool '{tool_name}' must define maybe_handle(user_text).")

        tool_id = getattr(tool, "name", None)
        if not isinstance(tool_id, str) or not tool_id.strip():
            raise ValueError(f"Configured tool '{tool_name}' must define a non-empty name.")
        return tool


def _tool_module_name(tool_name: str) -> str:
    if "." in tool_name:
        return tool_name
    return f"{TOOLS_PACKAGE}.{tool_name}"
