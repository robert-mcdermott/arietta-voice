from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from arietta_voice.config import AppConfig
from arietta_voice.knowledge import KnowledgeRetriever


@dataclass(slots=True, frozen=True)
class ModelToolContext:
    config: AppConfig
    knowledge_retriever: KnowledgeRetriever | None = None


@dataclass(slots=True, frozen=True)
class ModelToolExecution:
    followup_context: str
    details: dict[str, Any] = field(default_factory=dict)


class ModelInvokedTool(Protocol):
    name: str
    description: str
    arguments_example: dict[str, Any]

    def invoke(self, arguments: dict[str, Any]) -> ModelToolExecution | None: ...
