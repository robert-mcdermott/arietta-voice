from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from arietta_voice.config import AppConfig
from arietta_voice.knowledge import KnowledgeRetriever
from arietta_voice.models import ToolResponse


@dataclass(slots=True, frozen=True)
class ToolContext:
    config: AppConfig
    knowledge_retriever: KnowledgeRetriever | None = None


class AssistantTool(Protocol):
    name: str

    def maybe_handle(self, user_text: str) -> ToolResponse | None: ...
