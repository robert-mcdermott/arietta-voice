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


class SettingsUpdateRequest(BaseModel):
    """Changes by section and key, such as {"timers": {"repeat_s": 45}}."""

    changes: dict[str, dict[str, Any]] = Field(default_factory=dict, max_length=20)


class McpServerRequest(BaseModel):
    """One MCP server, as the console's form sends it."""

    name: str = Field(min_length=1, max_length=40)
    # The name it was saved under, when renaming.
    original_name: str | None = Field(default=None, max_length=40)
    transport: Literal["stdio", "http", "sse"]
    command: str = Field(default="", max_length=1000)
    args: list[str] = Field(default_factory=list, max_length=100)
    env: dict[str, str] = Field(default_factory=dict, max_length=100)
    cwd: str = Field(default="", max_length=1000)
    url: str = Field(default="", max_length=2000)
    headers: dict[str, str] = Field(default_factory=dict, max_length=50)
    enabled: bool = True
    confirm: Literal["ask", "first_use", "never"] = "ask"
    disabled_tools: list[str] = Field(default_factory=list, max_length=200)

    def entry(self) -> dict[str, Any]:
        """The server as an entry in the servers file."""
        if self.transport == "stdio":
            entry: dict[str, Any] = {"command": self.command, "args": self.args, "env": self.env}
            if self.cwd:
                entry["cwd"] = self.cwd
        else:
            entry = {"type": self.transport, "url": self.url, "headers": self.headers}
        entry.update(enabled=self.enabled, confirm=self.confirm, disabledTools=self.disabled_tools)
        return entry


class McpImportRequest(BaseModel):
    # A snippet in the common {"mcpServers": {...}} format, as in a server's README.
    text: str = Field(min_length=2, max_length=100_000)
