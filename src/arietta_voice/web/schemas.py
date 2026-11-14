"""Request bodies accepted by the console API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12_000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12_000)
    history: list[ChatHistoryItem] = Field(default_factory=list, max_length=128)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    use_tools: bool = True
    request_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class KnowledgeFileWriteRequest(BaseModel):
    content: str = Field(max_length=1_000_000)


class ToolProbeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class ToolEnabledRequest(BaseModel):
    enabled: list[str] = Field(default_factory=list, max_length=200)


class ToolCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=600)
    trigger: str = Field(default="", max_length=240)
    response: str = Field(default="", max_length=4000)
    enable: bool = True


class ToolTestRequest(BaseModel):
    """Either a sample request, run through the tool's triggers, or arguments."""

    text: str | None = Field(default=None, min_length=1, max_length=4000)
    arguments: dict[str, Any] | None = None


class ToolSourceWriteRequest(BaseModel):
    content: str = Field(min_length=1, max_length=1_000_000)


class TimerCreateRequest(BaseModel):
    """A timer needs ``duration``; a reminder ``when`` and ``label``; an alarm ``when``."""

    kind: Literal["timer", "reminder", "alarm"] = "timer"
    duration: str = Field(default="", max_length=120)
    when: str = Field(default="", max_length=120)
    label: str = Field(default="", max_length=200)
