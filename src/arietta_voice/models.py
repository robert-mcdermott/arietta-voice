from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class AssistantState(str, Enum):
    BOOTING = "booting"
    WAITING_FOR_WAKE = "waiting_for_wake"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    STOPPING = "stopping"
    WAITING_FOR_AUDIO = "waiting_for_audio"


@dataclass(slots=True)
class ConversationTurn:
    role: Literal["user", "assistant"]
    content: str


@dataclass(slots=True)
class ToolResponse:
    name: str
    content: str
    bypass_model: bool = True
    context: str | None = None


@dataclass(slots=True)
class SessionContext:
    active: bool = False
    history: list[ConversationTurn] = field(default_factory=list)
    turn_count: int = 0
    activated_at: float = 0.0
    last_activity_at: float = 0.0
    last_wake_phrase: str | None = None


@dataclass(slots=True)
class UtteranceResult:
    buffered_chunks: list
    end_session_requested: bool = False
