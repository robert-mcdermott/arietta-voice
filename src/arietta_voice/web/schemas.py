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
    module_name: str = Field(min_length=1, max_length=80)
    trigger_phrase: str = Field(default="example custom tool", min_length=1, max_length=240)
    response: str = Field(
        default="The custom tool is wired up. Replace this response with your real behavior.",
        min_length=1,
        max_length=4000,
    )
    enable: bool = True


class ToolSourceWriteRequest(BaseModel):
    content: str = Field(min_length=1, max_length=1_000_000)


class ModelToolInvokeRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)


class ModelToolCreateRequest(BaseModel):
    module_name: str = Field(min_length=1, max_length=80)
    description: str = Field(
        default="you need a custom model-called tool result before answering",
        min_length=1,
        max_length=600,
    )
    arguments_example: dict[str, Any] = Field(default_factory=dict)
    followup_context: str = Field(
        default="Custom model tool result. Use this result when composing the final answer.",
        min_length=1,
        max_length=4000,
    )
    enable: bool = True
